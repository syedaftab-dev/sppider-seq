"""
src/ppis/pu_loss.py
Task T3.2: PU-learning loss variants (Phase 3).

Implements V0 (standard BCE), V1 (masked BCE – unresolved residues dropped),
and V2 (weighted BCE – unresolved residues down-weighted).

The nnPU variant (V3) is left as future work; V1 and V2 are the main variants.

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    variant: str = "V0",
    neg_weight: float = 0.3,
) -> torch.Tensor:
    """Compute loss according to the selected PU variant.

    Parameters
    ----------
    logits: torch.Tensor
        Predicted logits (pre-sigmoid) for all residues of a focal protein.
    targets: torch.Tensor
        Ground-truth binary labels (0/1) matching ``logits`` shape.
    observed_mask: torch.Tensor (bool)
        ``True`` for residues that are observed (i.e., not "-") in the PDB alignment.
        Unobserved residues are considered *unlabeled*.
    variant: str
        One of ``"V0"``, ``"V1"`` or ``"V2"``.
    neg_weight: float
        Weight applied to negative residues for ``V2``. Must be in (0,1].

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if variant == "V0":
        # Standard BCE over all residues (unresolved treated as negatives).
        return F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="mean"
        )

    elif variant == "V1":
        # Masked BCE: drop unresolved residues from the loss.
        if observed_mask.sum() == 0:
            return logits.new_tensor(0.0)
        return F.binary_cross_entropy_with_logits(
            logits[observed_mask].float(),
            targets[observed_mask].float(),
            reduction="mean",
        )

    elif variant == "V2":
        # Weighted BCE: unresolved residues treated as negatives with weight neg_weight.
        weights = torch.where(observed_mask, torch.ones_like(logits), logits.new_full((), neg_weight))
        return F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), weight=weights, reduction="mean"
        )

    else:
        raise ValueError(f"Unsupported loss variant: {variant!r}. Choose V0, V1, or V2.")
