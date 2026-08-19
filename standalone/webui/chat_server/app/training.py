"""
'Train now' background job.

Workflow:
  1. Take a snapshot backup of model.pt + tokenizer.json into backups/<ts>/
  2. Load a FRESH copy of the model for training (so live inference is
     unaffected until we swap).
  3. For each pending feedback (rating in {+++, ++, +} and corrected_text
     not empty, applied_at IS NULL), run `steps_per_sample` train_step()
     iterations on `prompt + corrected_text`.
  4. Save updated weights over MODEL_PATH.
  5. Atomically swap the live model via inference.replace_state().
  6. Mark feedback rows applied_at=now.

Only one training job at a time (module-level lock).
"""
from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import inference
from app.config import settings
from app.models import Feedback, Message
from app.schemas import TrainStatus


BACKUPS_DIR = (Path(__file__).resolve().parent.parent / "backups")
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


_status_lock = threading.Lock()
_status = TrainStatus(state="idle")

# Guarantees only one training job at a time.
_job_lock = threading.Lock()


def current_status() -> TrainStatus:
    with _status_lock:
        return _status.model_copy()


def _set_status(**kwargs) -> None:
    global _status
    with _status_lock:
        _status = _status.model_copy(update=kwargs)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")


def _unique_backup_dir() -> Path:
    """Return a fresh backups/<ts>/ path, adding a numeric suffix on collision."""
    base = BACKUPS_DIR / _timestamp()
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    for i in range(1, 1000):
        alt = base.with_name(base.name + f"_{i}")
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt
    raise RuntimeError("Cannot allocate unique backup directory")


def make_backup() -> Path:
    """Copy live model.pt + tokenizer.json into backups/<ts>/."""
    src_model = settings.resolve(settings.MODEL_PATH)
    src_tok   = settings.resolve(settings.TOKENIZER_PATH)

    dst = _unique_backup_dir()
    shutil.copy2(src_model, dst / src_model.name)
    shutil.copy2(src_tok,   dst / src_tok.name)
    return dst


def _collect_pending_sync(db_url: str) -> list[tuple[int, str, str]]:
    """Synchronous feedback collection — called from worker thread."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    sync_url = db_url.replace("+aiosqlite", "")
    eng = create_engine(sync_url, future=True)

    out: list[tuple[int, str, str]] = []
    with Session(eng) as s:
        rows = s.execute(
            select(Feedback.id, Message.prompt, Feedback.corrected_text)
            .join(Message, Feedback.message_id == Message.id)
            .where(Feedback.applied_at.is_(None))
            .where(Feedback.rating.in_(("+++", "++", "+")))
            .where(Feedback.corrected_text.isnot(None))
            .where(Feedback.corrected_text != "")
            .order_by(Feedback.created_at.asc())
        ).all()
        for fid, prompt, corrected in rows:
            out.append((fid, prompt, corrected))
    eng.dispose()
    return out


def _mark_applied_sync(db_url: str, feedback_ids: list[int]) -> None:
    from sqlalchemy import create_engine, update
    from sqlalchemy.orm import Session

    sync_url = db_url.replace("+aiosqlite", "")
    eng = create_engine(sync_url, future=True)

    now = datetime.utcnow()
    with Session(eng) as s:
        s.execute(
            update(Feedback)
            .where(Feedback.id.in_(feedback_ids))
            .values(applied_at=now)
        )
        s.commit()
    eng.dispose()


def _run_training(lr: float, steps_per_sample: int) -> None:
    """Worker body. Runs inside a background thread."""
    from splx.torch_model import TorchGPT, TorchAdamOptimizer
    from splx.tokenizer   import BPETokenizer

    started = datetime.utcnow()
    _set_status(
        state="running", progress=0.0, total=0, processed=0,
        last_loss=None, backup_path=None,
        started_at=started, finished_at=None, error=None,
    )

    try:
        backup = make_backup()
        _set_status(backup_path=str(backup))

        pending = _collect_pending_sync(settings.DATABASE_URL)
        total   = len(pending)
        _set_status(total=total)

        if total == 0:
            _set_status(state="done", progress=1.0,
                        finished_at=datetime.utcnow())
            return

        # Fresh training copy
        model_path = settings.resolve(settings.MODEL_PATH)
        tok_path   = settings.resolve(settings.TOKENIZER_PATH)

        train_model = TorchGPT.load(str(model_path))
        train_model.train()
        optimizer = TorchAdamOptimizer(train_model.parameters(), lr=lr)

        tok = BPETokenizer()
        tok.load(str(tok_path))

        block = train_model.max_seq_len - 1  # train_step uses ids[:-1] / ids[1:]

        applied_ids: list[int] = []
        last_loss: Optional[float] = None

        for i, (fid, prompt, corrected) in enumerate(pending, start=1):
            full_text = (prompt + "\n" + corrected) if not prompt.endswith(("\n", " ")) \
                        else prompt + corrected
            ids = tok.encode(full_text)
            if len(ids) < 2:
                applied_ids.append(fid)
                _set_status(processed=i, progress=i / total)
                continue

            # Clip to block size; drop leading tokens if too long
            if len(ids) > block:
                ids = ids[-block:]

            ids_t = torch.tensor(ids, dtype=torch.long)
            for _ in range(steps_per_sample):
                last_loss = float(train_model.train_step(ids_t, optimizer))

            applied_ids.append(fid)
            _set_status(processed=i, progress=i / total, last_loss=last_loss)

        # Save updated weights over the live checkpoint
        train_model.eval()
        train_model.save(str(model_path))

        # Hot-swap the live model
        inference.replace_state(train_model, tok)

        if applied_ids:
            _mark_applied_sync(settings.DATABASE_URL, applied_ids)

        _set_status(state="done", progress=1.0,
                    finished_at=datetime.utcnow())

    except Exception as e:
        _set_status(state="error", error=str(e),
                    finished_at=datetime.utcnow())
        raise


def start_training(lr: Optional[float] = None,
                   steps_per_sample: Optional[int] = None) -> TrainStatus:
    """
    Non-blocking: spawns a worker thread if none is running.
    Returns immediate status.
    """
    use_lr    = lr if lr is not None else settings.TRAIN_LR
    use_steps = steps_per_sample if steps_per_sample is not None \
                else settings.TRAIN_STEPS_PER_SAMPLE

    if not _job_lock.acquire(blocking=False):
        return current_status()

    def _thread_target():
        try:
            _run_training(use_lr, use_steps)
        finally:
            _job_lock.release()

    t = threading.Thread(target=_thread_target, daemon=True, name="train-now")
    t.start()
    return current_status()
