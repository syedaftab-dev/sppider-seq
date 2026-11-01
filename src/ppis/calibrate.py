"""
src/ppis/calibrate.py
Task T2.1 & T2.2: Calibration diagnostics and post-hoc calibration (Phase 2).

The script loads a checkpoint, runs inference on the validation split,
computes reliability metrics (ECE, Brier loss, log-loss) and optionally
applies Platt scaling or isotonic regression, then saves a JSON report to
`results/calibration_<model>_seed<seed>.json`.

Usage (example)::

    python -m src.ppis.calibrate \\
        --model-type rc \\
        --checkpoint models/M0_RC_seed3.pt \\
        --seed 3

The implementation follows the tasks described in AGENTS.md (T2.1).

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.ppis.data import load_split
from src.ppis.embed import load_embedding
from src.ppis.model import ChunkwiseInteractionModel


def load_checkpoint(path: str) -> ChunkwiseInteractionModel:
    """Load a ChunkwiseInteractionModel checkpoint safely."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    cfg = state.get("config", {})
    bias = cfg.get("initial_bias", 0.0)
    model = ChunkwiseInteractionModel(initial_bias=bias)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def get_validation_pairs(model_type_arg: str) -> list:
    """Return validation PPIPairs for the given model type (rc or pc)."""
    from src.ppis.data import load_split
    import yaml
    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)
    ds_dir = paths["dataset_dir"]
    fname = "rec_ppint.pdb2uniprot.txt" if model_type_arg == "rc" else "pep_ppint.pdb2uniprot.txt"
    return load_split(os.path.join(ds_dir, fname), split="val", model_type=model_type_arg)


def predict_probs(m: ChunkwiseInteractionModel, pairs: list) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference; return (probs, labels) pooled over all pairs."""
    import yaml
    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)
    emb_dir = paths["emb_dir"]

    all_probs, all_labels = [], []
    with torch.no_grad():
        for pair in pairs:
            try:
                focal_emb = load_embedding(pair.focal_full, emb_dir)
                partner_emb = load_embedding(pair.partner_full, emb_dir)
            except FileNotFoundError:
                continue
            q_t = torch.tensor(focal_emb, dtype=torch.float32).unsqueeze(0)
            c_t = torch.tensor(partner_emb, dtype=torch.float32).unsqueeze(0)
            logits = m(q_t, c_t).squeeze().cpu().numpy()
            probs = expit(logits)
            all_probs.append(probs)
            all_labels.append(np.array(pair.focal_y, dtype=float))

    return np.concatenate(all_probs), np.concatenate(all_labels)


def collect_labels(pairs: list) -> np.ndarray:
    """Return concatenated ground-truth labels."""
    return np.concatenate([np.array(p.focal_y, dtype=float) for p in pairs])


def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """
    Compute calibration metrics on equal-mass logit-quantile bins.

    Returns dict with ECE, Brier, log_loss, and reliability diagram arrays.
    """
    # ECE on equal-mass bins
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="quantile")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    brier = float(brier_score_loss(labels, probs))
    logloss = float(log_loss(labels, probs))

    # Platt scaling (fit on same data for diagnostics only)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    lr.fit(probs.reshape(-1, 1), labels)
    platt_probs = lr.predict_proba(probs.reshape(-1, 1))[:, 1]
    fp2, mp2 = calibration_curve(labels, platt_probs, n_bins=n_bins, strategy="quantile")
    ece_platt = float(np.mean(np.abs(fp2 - mp2)))
    brier_platt = float(brier_score_loss(labels, platt_probs))

    return {
        "ece_raw": ece,
        "ece_platt": ece_platt,
        "brier_raw": brier,
        "brier_platt": brier_platt,
        "log_loss_raw": logloss,
        "reliability_frac_pos": frac_pos.tolist(),
        "reliability_mean_pred": mean_pred.tolist(),
        "platt_a": float(lr.coef_[0, 0]),
        "platt_b": float(lr.intercept_[0]),
        "n_bins": n_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration diagnostics for SPPIDER-seq models.")
    parser.add_argument("--model-type", required=True, choices=["rc", "pc"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    pairs = get_validation_pairs(args.model_type)
    probs, labels = predict_probs(model, pairs)
    metrics = compute_calibration(probs, labels, n_bins=args.n_bins)

    out = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "n_val_pairs": len(pairs),
        "n_val_residues": int(labels.shape[0]),
        **metrics,
    }

    fname = f"calibration_{args.model_type}_seed{args.seed}.json"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, fname), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {os.path.join(args.out_dir, fname)}")
    print(f"  ECE raw={metrics['ece_raw']:.4f}  ECE Platt={metrics['ece_platt']:.4f}")
    print(f"  Brier raw={metrics['brier_raw']:.4f}  Brier Platt={metrics['brier_platt']:.4f}")


if __name__ == "__main__":
    main()
