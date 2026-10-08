"""
src/ppis/model.py
T0.3 – Model classes copied from aporollo-lab/SPPIDER-seq cli/sppider_seq.py.

Attribution: GPL-3.0, Porollo et al. 2026.
Source: https://github.com/aporollo-lab/SPPIDER-seq/blob/main/cli/sppider_seq.py

Changes from source:
  - Removed global state; all functions take explicit arguments.
  - forward() accepts pre-computed chunk tensors (fp32 on device) instead of
    calling ESM-2 at runtime. This enables cached-embedding workflows.
  - merge_chunk_logits_to_full() takes an explicit device argument.
"""

from __future__ import annotations

import torch
import torch.nn as nn


EMBED_DIM = 1280
NUM_HEADS = 16


# ---------------------------------------------------------------------------
# Exact copies of the published architecture (VERIFY against source)
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """
    Unilateral cross-attention: query attends to context (partner).
    Residual + LayerNorm applied after attention.
    Source: cli/sppider_seq.py CrossAttentionLayer (lines ~227-235).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.cross_attn(query, context, context, key_padding_mask=context_mask)
        return self.norm(query + out)


class ChunkwiseInteractionModel(nn.Module):
    """
    Cross-attention head over chunked ESM-2 embeddings.
    Source: cli/sppider_seq.py ChunkwiseInteractionModel (lines ~238-275).

    forward() takes lists of [L, D] tensors (one per chunk, already on device).
    Returns a list of [L_chunk] logit vectors (one per query chunk).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        initial_bias: float | None = None,
    ) -> None:
        super().__init__()
        self.cross = CrossAttentionLayer(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim, 1),
        )
        if initial_bias is not None:
            with torch.no_grad():
                self.mlp[-1].bias.fill_(initial_bias)

    def forward(
        self,
        q_chunks: list[torch.Tensor],
        c_chunks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        q_chunks: list of [L_q, D] tensors (query chunks, on device)
        c_chunks: list of [L_c, D] tensors (context/partner chunks, on device)
        Returns list of [L_q_chunk] logit tensors.
        """
        pooled_per_q: list[torch.Tensor] = []
        for q in q_chunks:
            if q.ndim == 2:
                q = q.unsqueeze(0)             # [1, L_q, D]
            ctx_logits: list[torch.Tensor] = []
            for c in c_chunks:
                if c.ndim == 2:
                    c = c.unsqueeze(0)         # [1, L_c, D]
                ctx_mask = c.abs().sum(dim=-1) == 0   # padding mask
                x = self.cross(q, c, context_mask=ctx_mask)
                logits = self.mlp(x).squeeze(0).squeeze(-1)  # [L_q]
                ctx_logits.append(logits)
            # max over partner chunks (H2: inflates scores for many-chunk partners)
            pooled = torch.max(torch.stack(ctx_logits, dim=0), dim=0).values
            pooled_per_q.append(pooled)
        return pooled_per_q


def merge_chunk_logits_to_full(
    pooled_per_q: list[torch.Tensor],
    q_starts: list[int],
    full_len: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Max-pool overlapping query-chunk logits into a full-length logit vector.
    Source: cli/sppider_seq.py merge_chunk_logits_to_full (lines ~278-293).
    """
    out = torch.full((full_len,), -1e9, dtype=torch.float32, device=device)
    for vec, st in zip(pooled_per_q, q_starts):
        vec = vec.to(device)
        length = vec.shape[0]
        placed = torch.full((full_len,), -1e9, dtype=torch.float32, device=device)
        placed[st : st + length] = vec
        out = torch.maximum(out, placed)
    out[out <= -1e8] = -30.0
    return out


# ---------------------------------------------------------------------------
# Inference wrapper that consumes pre-cached chunk tensors
# ---------------------------------------------------------------------------

def predict_from_embeddings(
    model: ChunkwiseInteractionModel,
    q_chunks: list[torch.Tensor],
    q_starts: list[int],
    full_len: int,
    c_chunks: list[torch.Tensor],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Run inference and return per-residue probabilities (shape [L]).
    Embeddings must already be float32 tensors on `device`.
    """
    model.eval()
    with torch.no_grad():
        pooled = model(q_chunks, c_chunks)
        logits = merge_chunk_logits_to_full(pooled, q_starts, full_len, device)
        probs = torch.sigmoid(logits)
    return probs


# ---------------------------------------------------------------------------
# Checkpoint loader (weights_only=True first, as per rule 6)
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device | str = "cpu") -> ChunkwiseInteractionModel:
    """
    Load a pretrained ChunkwiseInteractionModel checkpoint.
    Tries weights_only=True first per AGENTS rule 6.
    """
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        raise RuntimeError(
            f"torch.load with weights_only=True failed for {path}. "
            "Do not use pickle fallback without asking the user (rule 6)."
        )
    model = ChunkwiseInteractionModel()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
