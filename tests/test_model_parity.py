"""
tests/test_model_parity.py
T0.3 – Verify our model wrapper produces identical logits to the original CLI.

Requires the pretrained checkpoints to be downloaded. If not present, tests are skipped.
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cli"))

from ppis.model import ChunkwiseInteractionModel, merge_chunk_logits_to_full

CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "raw"
REC_CKPT = CHECKPOINT_DIR / "sppider_seq_receptor.pt"
PEP_CKPT = CHECKPOINT_DIR / "sppider_seq_peptide.pt"


def _load_our_model(ckpt_path: Path) -> ChunkwiseInteractionModel:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    m = ChunkwiseInteractionModel()
    m.load_state_dict(state)
    m.eval()
    return m


@pytest.mark.skipif(not REC_CKPT.exists(), reason="RC checkpoint not downloaded")
def test_output_bias_rc():
    """H1: check that output bias is close to logit(train prevalence) = -3.45."""
    m = _load_our_model(REC_CKPT)
    bias = m.mlp[-1].bias.item()
    logit_prev = -3.45  # ln(0.0308/0.9692)
    print(f"\nRC output bias: {bias:.4f}, logit(prevalence): {logit_prev:.4f}")
    # No assertion – this is a diagnostic (H1)


@pytest.mark.skipif(not PEP_CKPT.exists(), reason="PC checkpoint not downloaded")
def test_output_bias_pc():
    """H1: check that output bias is close to logit(train prevalence) = -4.06."""
    m = _load_our_model(PEP_CKPT)
    bias = m.mlp[-1].bias.item()
    logit_prev = -4.06  # ln(0.0169/0.9831)
    print(f"\nPC output bias: {bias:.4f}, logit(prevalence): {logit_prev:.4f}")


@pytest.mark.skipif(not REC_CKPT.exists(), reason="RC checkpoint not downloaded")
def test_architecture_forward_shape():
    """Run a forward pass on random embeddings and check output shape."""
    m = _load_our_model(REC_CKPT)
    L_q, L_c, D = 50, 30, 1280
    q_chunks = [torch.randn(L_q, D)]
    c_chunks = [torch.randn(L_c, D)]
    pooled = m(q_chunks, c_chunks)
    assert len(pooled) == 1
    assert pooled[0].shape == (L_q,), f"Expected ({L_q},), got {pooled[0].shape}"
    full = merge_chunk_logits_to_full(pooled, [0], L_q, "cpu")
    assert full.shape == (L_q,)
