"""
src/ppis/controls.py
Task T1.2: Partner-usage audit controls (Phase 1).

Implements partner substitution controls C0–C5 as described in AGENTS.md:
  C0 – true partner (baseline)
  C1 – scrambled partner (residue permutation, composition preserved)
  C2 – non-cognate real partner (K draws from same split, length-matched)
  C3 – homolog partner (different protein, same MMseqs2 cluster)
  C4 – partner removed (zero-embedding and mean-embedding variants)
  C5 – aggregator sensitivity (max vs mean vs logsumexp over partner-chunk logits)

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.ppis.embed import load_embedding
from src.ppis.model import ChunkwiseInteractionModel, merge_chunk_logits_to_full


def get_c0_embedding(
    focal_seq: str,
    partner_seq: str,
    emb_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """C0: Return true partner embedding (baseline)."""
    focal_emb = load_embedding(focal_seq, emb_dir)
    partner_emb = load_embedding(partner_seq, emb_dir)
    return focal_emb, partner_emb


def get_c1_scrambled_sequence(partner_seq: str, seed: int = 0) -> str:
    """Scramble partner sequence preserving amino-acid composition."""
    rng = random.Random(seed)
    chars = list(partner_seq)
    rng.shuffle(chars)
    return "".join(chars)


def get_c1_embedding(
    partner_seq: str,
    tokenizer,
    esm,
    device: torch.device,
    emb_dir: str,
    seed: int = 0,
) -> np.ndarray:
    """C1: Scramble partner then embed (needs ESM pass; uses cache when available)."""
    from src.ppis.embed import embed_sequence
    scrambled = get_c1_scrambled_sequence(partner_seq, seed=seed)
    return embed_sequence(scrambled, tokenizer, esm, device, emb_dir)


def get_c2_non_cognate_partners(
    target_pair,
    all_split_pairs: list,
    seq_to_cluster: dict,
    K: int = 10,
    seed: int = 0,
) -> list:
    """
    C2: Return K non-cognate real partners from same split:
    - Length within +-20% of true partner
    - Different MMseqs2 cluster from true partner
    """
    rng = random.Random(seed)
    true_partner_seq = target_pair.partner_full
    true_cluster = seq_to_cluster.get(true_partner_seq, None)
    true_len = len(true_partner_seq)
    low, high = int(true_len * 0.8), int(true_len * 1.2)

    candidates = []
    for pair in all_split_pairs:
        pseq = pair.partner_full
        if pseq == true_partner_seq:
            continue
        if not (low <= len(pseq) <= high):
            continue
        clu = seq_to_cluster.get(pseq, None)
        if true_cluster is not None and clu == true_cluster:
            continue
        candidates.append(pseq)

    rng.shuffle(candidates)
    return candidates[:K]


def get_c3_homolog_partner(
    target_pair,
    all_split_pairs: list,
    seq_to_cluster: dict,
) -> Optional[str]:
    """
    C3: Return a homolog partner (different protein, same cluster as true partner) if available.
    """
    true_partner_seq = target_pair.partner_full
    true_cluster = seq_to_cluster.get(true_partner_seq, None)
    if true_cluster is None:
        return None
    for pair in all_split_pairs:
        pseq = pair.partner_full
        if pseq == true_partner_seq:
            continue
        if seq_to_cluster.get(pseq) == true_cluster:
            return pseq
    return None


def get_c4_zero_partner_embedding(partner_len: int, embed_dim: int = 1280) -> np.ndarray:
    """C4a: Replace partner with an all-zero embedding of the same length."""
    return np.zeros((partner_len, embed_dim), dtype=np.float16)


def get_c4_mean_partner_embedding(partner_len: int, mean_emb: np.ndarray) -> np.ndarray:
    """C4b: Replace partner with the mean embedding (broadcast to partner length)."""
    return np.tile(mean_emb, (partner_len, 1)).astype(np.float16)


def predict_c5_aggregator(
    model: ChunkwiseInteractionModel,
    q_tensors: List[torch.Tensor],
    c_tensors: List[torch.Tensor],
    q_starts: List[int],
    full_len: int,
    aggregator: str = "max",
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    C5: Inference wrapper allowing custom partner-chunk aggregation ('max', 'mean', 'logsumexp').
    Uses same query chunking as the released model; only the partner aggregation changes.
    """
    model.eval()
    all_query_logits: List[np.ndarray] = []
    all_query_starts: List[int] = []

    with torch.no_grad():
        for q_emb, q_start in zip(q_tensors, q_starts):
            # For each query chunk, aggregate logits over partner chunks
            chunk_logits = []
            for c_emb in c_tensors:
                q_t = q_emb.unsqueeze(0).to(device)
                c_t = c_emb.unsqueeze(0).to(device)
                logit = model.head(q_t, c_t).squeeze(0).squeeze(-1).cpu().numpy()
                chunk_logits.append(logit)

            chunk_logits_arr = np.stack(chunk_logits, axis=0)  # (n_partner_chunks, q_len)
            if aggregator == "max":
                agg = chunk_logits_arr.max(axis=0)
            elif aggregator == "mean":
                agg = chunk_logits_arr.mean(axis=0)
            elif aggregator == "logsumexp":
                agg = np.log(np.exp(chunk_logits_arr).sum(axis=0))
            else:
                raise ValueError(f"Unknown aggregator: {aggregator!r}")

            all_query_logits.append(agg)
            all_query_starts.append(q_start)

    return merge_chunk_logits_to_full(all_query_logits, all_query_starts, full_len)
