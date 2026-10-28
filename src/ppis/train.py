"""
src/ppis/train.py
Task T1.4 & Section 8: Fast trainer using cached embeddings for ablations M0, M1a, M1b, M2.

Variants:
  M0: Architecture reproduction (cross-attention query vs partner)
  M1a: Self-attention (context = query chunk)
  M1b: No cross-attention (MLP on query chunk only)
  M2: Random partner at train time (resampled per epoch from split, length-matched)

Training spec (AGENTS.md §8):
  - Input: cached ESM-2 embeddings. One example = one (focal, partner) pair.
  - Optimizer: Adam, LR grid {1e-4, 3e-4, 1e-3}, gradient accumulation 8-16 pairs.
  - Early stopping: patience 10 on val loss, max 50 epochs.
  - Loss: BCEWithLogitsLoss (V0) or PU variants from pu_loss.py.
  - Logging: per-epoch train/val loss, val AUROC & PRAUC to results/<run_id>.json.

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ppis.embed import load_embedding
from src.ppis.model import ChunkwiseInteractionModel
from src.ppis.pu_loss import compute_loss


class M1aSelfAttentionModel(nn.Module):
    """M1a: Context is query chunk itself (Self-attention)."""

    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q: torch.Tensor, _partner: torch.Tensor) -> torch.Tensor:  # noqa: D401
        attn_out, _ = self.attn(q, q, q)
        x = self.norm(q + attn_out)
        return self.mlp(x)


class M1bNoCrossAttentionModel(nn.Module):
    """M1b: No cross-attention – LayerNorm + MLP on query only."""

    def __init__(self, embed_dim: int = 1280, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q: torch.Tensor, _partner: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.mlp(self.norm(q))


def build_model(variant: str, initial_bias: float = 0.0) -> nn.Module:
    """Return the appropriate model for a given variant string."""
    if variant in ("M0", "M2"):
        model = ChunkwiseInteractionModel(initial_bias=initial_bias)
    elif variant == "M1a":
        model = M1aSelfAttentionModel()
        if initial_bias != 0.0:
            model.mlp[-1].bias.data.fill_(initial_bias)
    elif variant == "M1b":
        model = M1bNoCrossAttentionModel()
        if initial_bias != 0.0:
            model.mlp[-1].bias.data.fill_(initial_bias)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
    return model


def sample_length_matched_partner(target_pair, all_pairs: list, rng: random.Random):
    """Return a random length-matched partner sequence (within +-20%) from all_pairs."""
    true_len = len(target_pair.partner_full)
    low, high = int(true_len * 0.8), int(true_len * 1.2)
    candidates = [
        p for p in all_pairs
        if p.pair_id != target_pair.pair_id and low <= len(p.partner_full) <= high
    ]
    if not candidates:
        return target_pair  # fall back to true partner
    return rng.choice(candidates)


def train_one_epoch(
    model: nn.Module,
    pairs: list,
    optimizer: torch.optim.Optimizer,
    criterion,  # unused when PU variants handle loss
    emb_dir: str,
    device: torch.device,
    variant: str = "M0",
    loss_variant: str = "V0",
    neg_weight: float = 0.3,
    all_train_pairs: Optional[list] = None,
    accum_steps: int = 8,
    seed: int = 0,
) -> float:
    """Train for one epoch; return mean loss."""
    model.train()
    rng = random.Random(seed)
    total_loss = 0.0
    optimizer.zero_grad()

    for step, pair in enumerate(pairs):
        try:
            focal_emb = load_embedding(pair.focal_full, emb_dir)
            if variant == "M2" and all_train_pairs is not None:
                rand_pair = sample_length_matched_partner(pair, all_train_pairs, rng)
                partner_emb = load_embedding(rand_pair.partner_full, emb_dir)
            else:
                partner_emb = load_embedding(pair.partner_full, emb_dir)
        except FileNotFoundError:
            continue

        q_t = torch.tensor(focal_emb, dtype=torch.float32).unsqueeze(0).to(device)
        c_t = torch.tensor(partner_emb, dtype=torch.float32).unsqueeze(0).to(device)

        logits = model(q_t, c_t).squeeze(0).squeeze(-1)  # (L,)
        y = torch.tensor(pair.focal_y, dtype=torch.float32).to(device)
        obs = torch.tensor(pair.focal_observed, dtype=torch.bool).to(device)

        loss = compute_loss(logits, y, obs, variant=loss_variant, neg_weight=neg_weight)
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # flush remaining gradients
    optimizer.step()
    optimizer.zero_grad()
    return total_loss / max(len(pairs), 1)


def evaluate_loss_and_metrics(
    model: nn.Module,
    pairs: list,
    emb_dir: str,
    device: torch.device,
    loss_variant: str = "V0",
    neg_weight: float = 0.3,
) -> dict:
    """Evaluate on a split; return dict with loss, auroc, prauc."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for pair in pairs:
            try:
                focal_emb = load_embedding(pair.focal_full, emb_dir)
                partner_emb = load_embedding(pair.partner_full, emb_dir)
            except FileNotFoundError:
                continue

            q_t = torch.tensor(focal_emb, dtype=torch.float32).unsqueeze(0).to(device)
            c_t = torch.tensor(partner_emb, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(q_t, c_t).squeeze(0).squeeze(-1)
            y = torch.tensor(pair.focal_y, dtype=torch.float32).to(device)
            obs = torch.tensor(pair.focal_observed, dtype=torch.bool).to(device)

            loss = compute_loss(logits, y, obs, variant=loss_variant, neg_weight=neg_weight)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_logits.append(probs)
            all_labels.append(np.array(pair.focal_y))

    if not all_logits:
        return {"loss": float("nan"), "auroc": float("nan"), "prauc": float("nan")}

    y_score = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)

    try:
        auroc = float(roc_auc_score(y_true, y_score))
        prauc = float(average_precision_score(y_true, y_score))
    except Exception:
        auroc, prauc = float("nan"), float("nan")

    return {
        "loss": total_loss / max(len(pairs), 1),
        "auroc": auroc,
        "prauc": prauc,
    }
