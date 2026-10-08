"""
src/ppis/metrics.py
Evaluation metrics used throughout all phases.

All pooled metrics operate over flat arrays (all residues concatenated).
Per-protein metrics operate over a list of per-protein arrays.
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from dataclasses import dataclass, field


@dataclass
class Metrics:
    auroc: float
    prauc: float
    acc: float
    recall: float
    precision: float
    f1: float
    mcc: float
    lift: float   # prauc / prevalence
    n_pos: int
    n_neg: int
    threshold: float = 0.5


def pooled_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute pooled residue-level metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    prevalence = n_pos / len(y_true) if len(y_true) else 0.0

    auroc = float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else float("nan")
    prauc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else float("nan")
    lift = prauc / prevalence if prevalence > 0 else float("nan")

    return Metrics(
        auroc=auroc,
        prauc=prauc,
        acc=float(accuracy_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        lift=lift,
        n_pos=n_pos,
        n_neg=n_neg,
        threshold=threshold,
    )


def per_protein_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """AUROC for a single protein. Returns None if undefined."""
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return float(roc_auc_score(y_true, y_prob))


def per_protein_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """PRAUC for a single protein. Returns None if undefined."""
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_prob))


def top_k_hit(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> bool:
    """True if at least one of the top-k predicted residues is a true site."""
    if int(y_true.sum()) == 0:
        return False
    top_k_idx = np.argsort(y_prob)[-k:]
    return bool(y_true[top_k_idx].sum() > 0)


# ---------------------------------------------------------------------------
# Cluster bootstrap CI
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    values: list[float],
    cluster_ids: list[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Percentile CI via cluster bootstrap.
    Resample clusters with replacement, compute mean each time.
    Returns (mean, lower, upper).
    """
    rng = np.random.default_rng(seed)
    clusters = np.unique(cluster_ids)
    values_arr = np.array(values, dtype=float)
    cluster_ids_arr = np.array(cluster_ids)

    boot_means = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idxs = np.concatenate([
            np.where(cluster_ids_arr == c)[0] for c in sampled_clusters
        ])
        boot_means.append(float(np.nanmean(values_arr[idxs])))

    lo = float(np.nanpercentile(boot_means, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(np.nanmean(values_arr))
    return mean, lo, hi


def metrics_to_dict(m: Metrics) -> dict:
    return {
        "auroc": round(m.auroc, 4) if not math.isnan(m.auroc) else None,
        "prauc": round(m.prauc, 4) if not math.isnan(m.prauc) else None,
        "acc": round(m.acc, 4),
        "recall": round(m.recall, 4),
        "precision": round(m.precision, 4),
        "f1": round(m.f1, 4),
        "mcc": round(m.mcc, 4),
        "lift": round(m.lift, 4) if not math.isnan(m.lift) else None,
        "n_pos": m.n_pos,
        "n_neg": m.n_neg,
        "threshold": m.threshold,
    }
