"""
src/ppis/embed.py
T0.2 – ESM-2 embedding cache.

Dedupes sequences by SHA-1. Outputs:
  data/emb/<sha1>.npy   fp16, shape [L, 1280]
  data/emb/index.json   {sha1: sequence}

Chunking matches the released repo exactly:
  window = 1022 residues, stride = 512.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
WINDOW = 1022
STRIDE = 512
EMBED_DIM = 1280


# ---------------------------------------------------------------------------
# SHA-1 key
# ---------------------------------------------------------------------------

def seq_sha1(seq: str) -> str:
    return hashlib.sha1(seq.upper().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Chunk a sequence the same way as the released repo
# ---------------------------------------------------------------------------

def chunk_sequence(seq: str, window: int = WINDOW, stride: int = STRIDE) -> list[dict[str, Any]]:
    """
    Return list of {start_idx, end_idx, subseq}.
    Matches embed_sequence_chunks() in cli/sppider_seq.py.
    """
    length = len(seq)
    chunks = []
    for start in range(0, length, stride):
        end = min(start + window, length)
        if end <= start:
            break
        chunks.append({"start_idx": start, "end_idx": end, "subseq": seq[start:end]})
        if end == length:
            break
    return chunks


# ---------------------------------------------------------------------------
# Embed one chunk
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_chunk(
    subseq: str,
    tokenizer: EsmTokenizer,
    esm: EsmModel,
    device: torch.device,
) -> np.ndarray:
    """Return fp32 numpy array [L_chunk, 1280]."""
    inputs = tokenizer(subseq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = esm(**inputs)
    # remove BOS and EOS tokens
    hidden = out.last_hidden_state[0, 1:-1, :].cpu().float().numpy()
    return hidden


# ---------------------------------------------------------------------------
# Full sequence embedding (chunked, reassembled as fp16)
# ---------------------------------------------------------------------------

def embed_sequence(
    seq: str,
    tokenizer: EsmTokenizer,
    esm: EsmModel,
    device: torch.device,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Embed a full sequence.
    Returns:
      emb:    fp16 array [L, 1280]  (max-pooled over overlapping chunks)
      chunks: list of chunk dicts with 'start_idx', 'end_idx', 'embedding' (fp32 tensor)
    """
    chunks = chunk_sequence(seq, window, stride)
    L = len(seq)
    full_fp32 = np.full((L, EMBED_DIM), -1e9, dtype=np.float32)
    chunk_dicts = []

    for ch in chunks:
        s, e = ch["start_idx"], ch["end_idx"]
        emb_chunk = _embed_chunk(ch["subseq"], tokenizer, esm, device)  # [e-s, 1280]
        # max-pool overlapping positions
        full_fp32[s:e] = np.maximum(full_fp32[s:e], emb_chunk)
        chunk_dicts.append({
            "start_idx": s,
            "end_idx": e,
            "embedding": torch.from_numpy(emb_chunk),  # fp32 tensor
        })

    # clip sentinel
    full_fp32[full_fp32 <= -1e8] = 0.0
    return full_fp32.astype(np.float16), chunk_dicts


# ---------------------------------------------------------------------------
# Cache read/write
# ---------------------------------------------------------------------------

def cache_path(sha1: str, emb_dir: str | Path) -> Path:
    return Path(emb_dir) / f"{sha1}.npy"


def save_embedding(sha1: str, emb: np.ndarray, emb_dir: str | Path) -> None:
    Path(emb_dir).mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path(sha1, emb_dir)), emb.astype(np.float16))


def load_embedding(sha1: str, emb_dir: str | Path) -> np.ndarray | None:
    p = cache_path(sha1, emb_dir)
    if p.exists():
        return np.load(str(p))  # fp16
    return None


def update_index(sha1: str, seq: str, emb_dir: str | Path) -> None:
    idx_path = Path(emb_dir) / "index.json"
    idx: dict[str, str] = {}
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    idx[sha1] = seq
    idx_path.write_text(json.dumps(idx, indent=2))


def get_or_embed(
    seq: str,
    tokenizer: EsmTokenizer,
    esm: EsmModel,
    device: torch.device,
    emb_dir: str | Path,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Return (fp16 full embedding, chunk dicts) from cache or by computing.
    chunk dicts contain fp32 tensors for direct model use.
    """
    sha1 = seq_sha1(seq)
    cached = load_embedding(sha1, emb_dir)

    if cached is not None:
        # Rebuild chunk dicts from cached full embedding (for model inference)
        chunks = chunk_sequence(seq, window, stride)
        chunk_dicts = []
        for ch in chunks:
            s, e = ch["start_idx"], ch["end_idx"]
            chunk_dicts.append({
                "start_idx": s,
                "end_idx": e,
                "embedding": torch.from_numpy(cached[s:e].astype(np.float32)),
            })
        return cached, chunk_dicts

    # Compute and cache
    emb_fp16, chunk_dicts = embed_sequence(seq, tokenizer, esm, device, window, stride)
    save_embedding(sha1, emb_fp16, emb_dir)
    update_index(sha1, seq, emb_dir)
    return emb_fp16, chunk_dicts
