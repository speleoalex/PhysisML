"""
AxiomRegistry — registry of immutable or highly protected patterns.

An axiom is a token sequence with a protection level [0, 1].
Protection scales the gradient of involved tokens before the optimizer step:
  effective_grad[token] *= (1 - protection_level)
  protection=1.0 → gradient zeroed (absolute axiom)
  protection=0.5 → gradient halved

Objective/subjective distinction:
  is_objective=True  → protection can rise up to 1.0
  is_objective=False → protection does not exceed 0.6
                       (opinions are partially modifiable)

Usage example:
  registry = AxiomRegistry()
  # 1+1=2 is an inviolable objective truth
  registry.register([tok("1"), tok("+"), tok("1"), tok("="), tok("2")],
                    description="1+1=2", is_objective=True)

  # Apply protection to gradients before the optimizer step
  for name, param in model.named_parameters():
      if param.grad is not None and name == "tok_emb.weight":
          registry.apply_to_grad(param.grad)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set
import torch


@dataclass
class Axiom:
    token_sequence:   List[int]
    description:      str
    protection_level: float    # 0..1
    is_objective:     bool     # True = objective fact, False = preference
    frequency_seen:   int = 0  # times observed with positive feedback

    MAX_OBJECTIVE  = 1.0
    MAX_SUBJECTIVE = 0.6

    def __post_init__(self):
        max_p = self.MAX_OBJECTIVE if self.is_objective else self.MAX_SUBJECTIVE
        self.protection_level = min(self.protection_level, max_p)


class AxiomRegistry:

    AUTO_AXIOM_THRESHOLD    = 100   # positive observations before becoming an axiom
    CONSISTENCY_STEPS_BOOST = 50    # consecutive consistent steps → increase protection
    PROTECTION_INCREMENT    = 0.1   # protection increment per consistency step

    def __init__(self):
        self._axioms: Dict[str, Axiom] = {}
        self._protected_ids: Set[int] = set()   # token IDs under active protection

    # ------------------------------------------------------------------
    # Manual registration
    # ------------------------------------------------------------------

    def register(self, token_sequence: List[int], description: str = "",
                 is_objective: bool = True,
                 protection_level: float = 1.0) -> None:
        """Register an axiom by hand."""
        key = self._key(token_sequence)
        axiom = Axiom(token_sequence, description, protection_level, is_objective)
        self._axioms[key] = axiom
        for tid in token_sequence:
            self._protected_ids.add(tid)

    # ------------------------------------------------------------------
    # Automatic update from observations
    # ------------------------------------------------------------------

    def observe(self, token_sequence: List[int], feedback: float,
                is_objective: bool = True) -> None:
        """
        Update the count for a sequence. If it exceeds the threshold and
        feedback is consistently positive, increases protection.
        Does not create new axioms automatically (requires explicit register()),
        but increments protection of existing axioms.
        """
        if feedback <= 0:
            return
        key = self._key(token_sequence)
        if key in self._axioms:
            self._axioms[key].frequency_seen += 1
            ax = self._axioms[key]
            if ax.frequency_seen % self.CONSISTENCY_STEPS_BOOST == 0:
                max_p = Axiom.MAX_OBJECTIVE if ax.is_objective else Axiom.MAX_SUBJECTIVE
                ax.protection_level = min(
                    ax.protection_level + self.PROTECTION_INCREMENT, max_p)

    # ------------------------------------------------------------------
    # Application to the PyTorch gradients
    # ------------------------------------------------------------------

    def apply_to_grad(self, embedding_grad: torch.Tensor) -> None:
        """
        Scale the embedding gradient for protected tokens.
        embedding_grad: (V, d_model) — grad of tok_emb.weight
        Modifies in-place.
        """
        if not self._protected_ids or embedding_grad is None:
            return
        for tid in self._protected_ids:
            if tid >= embedding_grad.shape[0]:
                continue
            ax = self._get_axiom_for_token(tid)
            scale = 1.0 - (ax.protection_level if ax else 0.0)
            if scale < 1.0:
                embedding_grad[tid] *= scale

    def register_hook(self, model) -> None:
        """
        Register a backward hook on tok_emb.weight to automatically apply
        axiom protection on every backward pass.
        """
        def _hook(grad):
            g = grad.clone()
            self.apply_to_grad(g)
            return g
        model.tok_emb.weight.register_hook(_hook)

    # ------------------------------------------------------------------

    def get_protection(self, token_id: int) -> float:
        """Return the protection level for a token (0 if not protected)."""
        ax = self._get_axiom_for_token(token_id)
        return ax.protection_level if ax else 0.0

    def list_axioms(self) -> List[Axiom]:
        return list(self._axioms.values())

    def _key(self, token_sequence: List[int]) -> str:
        return ",".join(str(t) for t in token_sequence)

    def _get_axiom_for_token(self, token_id: int) -> object:
        for ax in self._axioms.values():
            if token_id in ax.token_sequence:
                return ax
        return None
