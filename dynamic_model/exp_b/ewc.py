"""
Online EWC (Elastic Weight Consolidation) — the parametric anti-forgetting
baseline for exp_i, benchmarked against the dream's cross-level replay.

The dream fights forgetting by replaying every earlier level's material (N1
corpus + N3 memory bank). EWC fights it without storing any text: at the end
of each level a diagonal Fisher information estimate marks which parameters
carried that level's knowledge, and during the next level a quadratic penalty
    0.5 * lambda * sum_i F_i * (theta_i - theta*_i)^2
pulls exactly those parameters back toward their anchored values theta*.

This is the ONLINE variant (Schwarz et al. 2018): one running Fisher
    F <- gamma * F_prev + F_new
and one anchor theta*, refreshed at every level boundary. Per-level Fisher
accumulation was rejected on two grounds: 13 anchors of a 23M-parameter model
are ~2.4GB, which does not fit the Arc next to the model, the Adam moments and
the N1 activations; and the curriculum's levels are cumulative refinements of
one competence, not disjoint tasks — decaying stale constraints with gamma is
the behaviour that matches, while 13 anchors pulling in 13 directions is the
failure mode long curricula create for per-task EWC.

What is deliberately NOT constrained:
  - LayerNorm parameters and pos_emb.weight. The TorchAdamOptimizer docstring
    records the precedent: constant coupled weight-decay pressure drove the
    ln_f gain from 0.87 at L0 to 0.008 at L10 over the build's hundreds of
    thousands of single-sample steps. A quadratic pull toward an anchor is the
    same shape of constant pressure, so those tensors stay free.
  - Vocabulary rows activated AFTER the Fisher snapshot: they carry F=0 by
    construction (dormant rows take zero gradient), so new vocabulary learns
    unconstrained — which is what the experiment wants.

The Fisher/anchor pair lives in a SIDECAR file (level_N/fisher.pt), never in
the model's state_dict: TorchGPT.load calls load_state_dict strict, and the
exporters (scripts/export_gguf.py, huggingface/generate.py) read raw state
dicts — registering buffers would break every existing checkpoint.
"""
import hashlib
import os
import time

import torch

# The one home of the EWC tuning constants. build.sh's EWC_LAMBDA/EWC_GAMMA
# fallbacks must carry the same values — tests/test_ewc.py parses that file
# and fails when they drift (same contract as dream_until_plateau.DEFAULTS).
DEFAULTS = {"lambda": 1000.0, "gamma": 0.95}

# Sidecar filename, next to final_dreamed.pt in the level's checkpoint dir.
SIDECAR = "fisher.pt"


def file_sha256(path: str) -> str:
    """Hash of a checkpoint file, for the anchor-staleness warning."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EWC:
    """Online EWC state: running Fisher diagonal + anchor weights.

    Attach an instance to TrainerB (trainer.ewc = ...) and every backward
    site adds `penalty(model)` to its loss. `estimate_fisher` +
    `consolidate` run once per level boundary (scripts/compute_fisher.py).
    """

    def __init__(self, lam: float, gamma: float = DEFAULTS["gamma"],
                 device: str = "cpu"):
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.device = device
        self.fisher = {}       # param name -> Tensor (same shape as the param)
        self.theta_star = {}   # param name -> Tensor
        self.meta = {}

    # ------------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------------

    @staticmethod
    def protected_parameters(model) -> list:
        """(name, param) pairs the penalty applies to.

        Built from named_parameters(), which yields the weight-tied
        lm_head/tok_emb Parameter ONCE (as 'tok_emb.weight'). Never build a
        Fisher from state_dict(): it yields the shared tensor twice and the
        largest tensor in the model would be double-counted.

        LayerNorm parameters and pos_emb are excluded — see the module
        docstring for the measured ln_f precedent.
        """
        ln_owners = {name for name, mod in model.named_modules()
                     if isinstance(mod, torch.nn.LayerNorm)}
        out = []
        for name, p in model.named_parameters():
            owner = name.rsplit(".", 1)[0] if "." in name else ""
            if owner in ln_owners:
                continue
            if name == "pos_emb.weight":
                continue
            out.append((name, p))
        return out

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self, model) -> torch.Tensor:
        """0.5 * lam * sum F * (theta - theta*)^2, as an autograd scalar.

        Accumulated tensor-by-tensor (36 tensors at most) — no full-model
        temporaries, which matters on the Arc. With no Fisher loaded or
        lam <= 0 it returns a detached zero so callers never branch.
        """
        if self.lam <= 0.0 or not self.fisher:
            return torch.zeros((), device=next(model.parameters()).device)
        total = None
        for name, p in model.named_parameters():
            f = self.fisher.get(name)
            if f is None:
                continue
            term = (f * (p - self.theta_star[name]) ** 2).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), device=next(model.parameters()).device)
        return 0.5 * self.lam * total

    @torch.no_grad()
    def penalty_grad_norm(self, model) -> float:
        """L2 norm of the penalty's gradient, computed analytically:
        grad = lam * F * (theta - theta*). No second backward pass — this is
        what makes the penalty/task gradient-ratio log affordable."""
        sq = None
        for name, p in model.named_parameters():
            f = self.fisher.get(name)
            if f is None:
                continue
            g = self.lam * f * (p - self.theta_star[name])
            s = (g * g).sum()
            sq = s if sq is None else sq + s
        return float(sq.sqrt()) if sq is not None else 0.0

    @torch.no_grad()
    def per_tensor_report(self, model, top_k: int = 5) -> list:
        """Top-k (name, penalty contribution) — the slow-death detector.
        LayerNorm/pos_emb are already excluded from the penalty, but an
        embedding can develop the same pathology; this makes it visible."""
        contribs = []
        for name, p in model.named_parameters():
            f = self.fisher.get(name)
            if f is None:
                continue
            val = float(0.5 * self.lam
                        * (f * (p - self.theta_star[name]) ** 2).sum())
            contribs.append((name, val))
        contribs.sort(key=lambda kv: kv[1], reverse=True)
        return contribs[:top_k]

    # ------------------------------------------------------------------
    # Fisher estimation + online consolidation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_fisher(model, tokenizer, pairs, device=None) -> tuple:
        """Empirical diagonal Fisher over gold (prompt, response) pairs.

        One pair at a time, using encode_prompt_response and the same
        prompt-masked loss the training steps optimise — the Fisher must
        measure the loss geometry training actually used, or the anchor
        protects the wrong directions.

        Returns (fisher_dict, n_used, n_skipped). Pairs whose loss or
        gradients are non-finite are skipped and counted (same guard as the
        dormant-slot desync guard in TrainerB.step).
        """
        # Imported here, not at module top: trainer.py is the heavyweight
        # module (it drags in the affect system) and ewc.py must stay
        # importable by tests and scripts without it.
        from dynamic_model.exp_b.trainer import encode_prompt_response

        device = device or next(model.parameters()).device
        was_training = model.training
        model.eval()
        protected = EWC.protected_parameters(model)
        fisher = {name: torch.zeros_like(p) for name, p in protected}
        max_len = model.max_seq_len - 1
        n_used = n_skipped = 0

        for pair in pairs:
            p_txt = (pair.get("prompt") or "").strip()
            r_txt = (pair.get("response") or "").strip()
            if not p_txt or not r_txt:
                continue
            ids_np, n_prompt = encode_prompt_response(
                tokenizer, p_txt, r_txt, max_len)
            if len(ids_np) < 2 or len(ids_np) <= n_prompt:
                continue
            ids = torch.from_numpy(ids_np).long().to(device)
            model.zero_grad(set_to_none=True)
            logits = model.forward(ids)
            loss = model.loss(logits, ids, prompt_len=n_prompt)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            finite = torch.ones((), dtype=torch.bool, device=device)
            for _, p in protected:
                if p.grad is not None:
                    finite &= torch.isfinite(p.grad).all()
            if not bool(finite):
                n_skipped += 1
                continue
            for name, p in protected:
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            n_used += 1

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        if n_used:
            for name in fisher:
                fisher[name] /= n_used
        return fisher, n_used, n_skipped

    def consolidate(self, model, fisher_new: dict) -> None:
        """Online update: F = gamma * F_prev + F_new; anchor refreshed to the
        model's current weights for every constrained tensor."""
        for name, f_new in fisher_new.items():
            f_prev = self.fisher.get(name)
            self.fisher[name] = (f_new if f_prev is None
                                 else self.gamma * f_prev + f_new)
        for name, p in EWC.protected_parameters(model):
            if name in self.fisher:
                self.theta_star[name] = p.detach().clone()

    # ------------------------------------------------------------------
    # Sidecar persistence
    # ------------------------------------------------------------------

    def save(self, path: str, **extra_meta) -> None:
        """Write the sidecar atomically (tmp + rename): dream_until_plateau's
        snapshot/restore copies it whole, and a torn file must never pair
        with a restored checkpoint."""
        meta = {"gamma": self.gamma, "lambda": self.lam,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **extra_meta}
        payload = {
            "fisher": {k: v.detach().to("cpu", torch.float32)
                       for k, v in self.fisher.items()},
            "theta_star": {k: v.detach().to("cpu", torch.float32)
                           for k, v in self.theta_star.items()},
            "meta": meta,
        }
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.meta = meta

    @classmethod
    def load(cls, path: str, lam: float, gamma: float = None,
             device: str = "cpu") -> "EWC":
        """Load a sidecar. lam is a run-time choice, never taken from the
        file; gamma defaults to what the sidecar was built with so the next
        consolidation continues the same decay."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        meta = data.get("meta", {})
        obj = cls(lam=lam,
                  gamma=gamma if gamma is not None
                  else meta.get("gamma", DEFAULTS["gamma"]),
                  device=device)
        obj.fisher = {k: v.to(device) for k, v in data["fisher"].items()}
        obj.theta_star = {k: v.to(device)
                          for k, v in data["theta_star"].items()}
        obj.meta = meta
        return obj
