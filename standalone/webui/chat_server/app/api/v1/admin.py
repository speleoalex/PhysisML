"""
Admin-only endpoints: backup, feedback export, Train now.
"""
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app import inference, training
from app.auth import current_admin
from app.db import get_db
from app.models import Feedback, Message, User
from app.schemas import (ApiResponse, BackupResponse, TrainStartRequest,
                         TrainStatus)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/backup", response_model=ApiResponse[BackupResponse])
async def backup(admin: User = Depends(current_admin)):
    dst = training.make_backup()
    total = sum(f.stat().st_size for f in dst.iterdir() if f.is_file())
    return ApiResponse(data=BackupResponse(
        path=str(dst),
        size_bytes=total,
        created_at=datetime.utcnow(),
    ))


@router.get("/feedback/export", response_class=PlainTextResponse)
async def feedback_export(
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Return all feedback as JSONL (one object per line).
    Schema: {message_id, session_id, prompt, reply,
             rating, corrected_text, created_at, applied_at}.
    """
    res = await db.execute(
        select(Feedback, Message)
        .join(Message, Feedback.message_id == Message.id)
        .order_by(Feedback.created_at.asc())
    )
    lines = []
    for fb, msg in res.all():
        entry = {
            "message_id":     msg.id,
            "session_id":     msg.session_id,
            "prompt":         msg.prompt,
            "reply":          msg.reply,
            "rating":         fb.rating,
            "corrected_text": fb.corrected_text,
            "created_at":     fb.created_at.isoformat() + "Z",
            "applied_at":     fb.applied_at.isoformat() + "Z" if fb.applied_at else None,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""),
                             media_type="application/jsonl")


@router.get("/model/info", response_model=ApiResponse[dict])
async def model_info(admin: User = Depends(current_admin)):
    return ApiResponse(data=inference.model_info())


@router.post("/train", response_model=ApiResponse[TrainStatus])
async def train(body: TrainStartRequest,
                admin: User = Depends(current_admin)):
    s = training.start_training(lr=body.lr, steps_per_sample=body.steps_per_sample)
    return ApiResponse(data=s)


@router.get("/train/status", response_model=ApiResponse[TrainStatus])
async def train_status(admin: User = Depends(current_admin)):
    return ApiResponse(data=training.current_status())


@router.delete("/sessions/{session_id}", response_model=ApiResponse[dict])
async def delete_session(
    session_id: str,
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete all messages (and cascaded feedback) in a session."""
    res = await db.execute(
        select(Message)
        .options(selectinload(Message.feedback))
        .where(Message.session_id == session_id)
    )
    msgs = list(res.scalars().unique().all())
    if not msgs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    for m in msgs:
        await db.delete(m)
    await db.commit()
    return ApiResponse(data={"session_id": session_id, "deleted": len(msgs)})


@router.delete("/messages/{message_id}", response_model=ApiResponse[dict])
async def delete_message(
    message_id: int,
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a single message (and its feedback)."""
    res = await db.execute(
        select(Message)
        .options(selectinload(Message.feedback))
        .where(Message.id == message_id)
    )
    msg = res.scalar_one_or_none()
    if msg is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")
    await db.delete(msg)
    await db.commit()
    return ApiResponse(data={"message_id": message_id, "deleted": True})
