"""
src/ppis/conformal.py
Task T2.4: Conformal prediction utilities for SPPIDER-seq (Phase 2).

Implements finite-sample risk-controlled prediction sets for:
  * False-negative rate (FNR) control (conformal risk control, CRC)
  * False discovery rate (FDR) control (Learn-then-Test, LTT)

Reference formulas:
  CRC  — Angelopoulos et al. 2022  (https://arxiv.org/abs/2205.15291)
  LTT  — Angelopoulos et al. 2021  (https://arxiv.org/abs/2110.09888)

The script loads a checkpoint, runs inference on validation proteins,
then computes prediction sets at a grid of alpha values and saves
JSON reports to `results/conformal_<model>_seed<seed>.json`.

Usage (example)::

    python -m src.ppis.conformal \\
        --model-type rc \\
        --checkpoint models/M0_RC_seed3.pt \\
        --seed 3

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.special import expit

from src.ppis.calibrate import get_validation_pairs, load_checkpoint, predict_probs


def collect_labels(pairs: list) -> np.ndarray:
    return np.concatenate([np.array(p.focal_y, dtype=float) for p in pairs])


def group_by_protein(
    pairs: list,
    probs: np.ndarray,
    labels: np.ndarray,
) -> List[Dict]:
    """
    Re-group flat prob/label arrays back into per-protein dicts.
    Returns list of {'name': str, 'probs': ndarray, 'labels': ndarray}.
    Skips proteins with no positive labels.
    """
    groups = []
    idx = 0
    for pair in pairs:
        n = len(pair.focal_y)
        p = probs[idx: idx + n]
        y = labels[idx: idx + n]
        idx += n
        if y.sum() > 0:
            groups.append({"name": pair.focal_name, "probs": p, "labels": y})
    return groups


def fnr_control(
    groups: List[Dict],
    alphas: List[float],
    t_grid: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Conformal risk control (CRC) for FNR.

    For each alpha: find the largest threshold t such that
      (n/(n+1)) * mean_i Loss_i(t) + 1/(n+1) <= alpha
    where Loss_i(t) = 1 - |C_t ∩ Y_i| / |Y_i|  (FNR for protein i).

    Returns list of dicts (one per alpha) with threshold, mean FNR, mean set size, recall.
    Ref: Angelopoulos et al. 2022, Theorem 1.
    """
    if t_grid is None:
        t_grid = np.linspace(0.0, 1.0, 201)[::-1]  # descending

    n = len(groups)
    results = []
    for alpha in alphas:
        chosen_t = 0.0  # default: include everything
        for t in t_grid:
            losses = []
            for g in groups:
                set_mask = g["probs"] >= t
                y = g["labels"].astype(bool)
                tp = (set_mask & y).sum()
                fnr_i = 1.0 - tp / max(y.sum(), 1)
                losses.append(fnr_i)
            risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
            if risk <= alpha:
                chosen_t = float(t)
                break

        # Compute stats at chosen_t
        set_sizes, recalls, precisions = [], [], []
        for g in groups:
            set_mask = g["probs"] >= chosen_t
            y = g["labels"].astype(bool)
            set_sizes.append(int(set_mask.sum()))
            tp = (set_mask & y).sum()
            recalls.append(tp / max(y.sum(), 1))
            precisions.append(tp / max(set_mask.sum(), 1))

        results.append({
            "alpha": alpha,
            "threshold": chosen_t,
            "mean_set_size": float(np.mean(set_sizes)),
            "empirical_fnr": float(1.0 - np.mean(recalls)),
            "empirical_recall": float(np.mean(recalls)),
            "empirical_precision": float(np.mean(precisions)),
            "n_proteins": n,
        })
    return results


def fdr_control(
    groups: List[Dict],
    alphas: List[float],
    delta: float = 0.1,
    t_grid: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Learn-then-Test (LTT) for FDR control.

    For each alpha: use Hoeffding-Bentkus p-values for bounded FDP losses,
    fixed-sequence testing from high t to low t, pick smallest t still rejected.
    Loss_i(t) = FDP of C_t for protein i (0 if set is empty).

    Ref: Angelopoulos et al. 2021.
    """
    import scipy.stats as st

    if t_grid is None:
        t_grid = np.linspace(0.01, 0.99, 99)[::-1]

    n = len(groups)
    results = []

    for alpha in alphas:
        chosen_t = None
        for t in t_grid:
            # Compute per-protein FDP losses
            losses = []
            for g in groups:
                set_mask = g["probs"] >= t
                y = g["labels"].astype(bool)
                if set_mask.sum() == 0:
                    losses.append(0.0)
                    continue
                fp = (set_mask & ~y).sum()
                fdp = fp / set_mask.sum()
                losses.append(float(fdp))

            losses_arr = np.array(losses)
            mu_hat = float(np.mean(losses_arr))

            # Hoeffding-Bentkus p-value
            # H_t: E[Loss(t)] > alpha
            # p-value via Hoeffding one-sided bound
            diff = mu_hat - alpha
            # Bentkus: P(mean > alpha) <= exp(-2n*diff^2) for bounded losses in [0,1]
            if diff >= 0:
                p_val = 1.0  # do not reject — risk too high
            else:
                p_val = float(np.exp(2 * n * diff ** 2))  # Hoeffding bound

            if p_val <= delta:
                chosen_t = float(t)
                # Continue to find smaller t (less conservative)
            else:
                break  # fixed-sequence: stop at first non-rejection

        if chosen_t is None:
            chosen_t = float(t_grid[-1])

        # Stats at chosen_t
        set_sizes, recalls, fdps = [], [], []
        for g in groups:
            set_mask = g["probs"] >= chosen_t
            y = g["labels"].astype(bool)
            set_sizes.append(int(set_mask.sum()))
            tp = (set_mask & y).sum()
            recalls.append(tp / max(y.sum(), 1))
            if set_mask.sum() > 0:
                fdps.append((set_mask & ~y).sum() / set_mask.sum())

        results.append({
            "alpha": alpha,
            "delta": delta,
            "threshold": chosen_t,
            "mean_set_size": float(np.mean(set_sizes)),
            "empirical_fdr": float(np.mean(fdps)) if fdps else float("nan"),
            "empirical_recall": float(np.mean(recalls)),
            "n_proteins": n,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal prediction for SPPIDER-seq.")
    parser.add_argument("--model-type", required=True, choices=["rc", "pc"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fnr-alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--fdr-alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    pairs = get_validation_pairs(args.model_type)
    probs, labels = predict_probs(model, pairs)
    groups = group_by_protein(pairs, probs, labels)

    fnr_results = fnr_control(groups, args.fnr_alphas)
    fdr_results = fdr_control(groups, args.fdr_alphas, delta=args.delta)

    out = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "n_val_proteins": len(groups),
        "fnr_control": fnr_results,
        "fdr_control": fdr_results,
    }

    fname = f"conformal_{args.model_type}_seed{args.seed}.json"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, fname), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {os.path.join(args.out_dir, fname)}")
    for r in fnr_results:
        print(f"  FNR α={r['alpha']}: t={r['threshold']:.3f}  "
              f"recall={r['empirical_recall']:.3f}  set_size={r['mean_set_size']:.1f}")


# Optional import guard
try:
    from typing import Optional
except ImportError:
    Optional = None

if __name__ == "__main__":
    main()
