"""
src/ppis/eval.py
Task T1.3: Audit Metrics & Evaluation Module for Phase 1.

Metrics:
  - Per-protein / per-pair AUROC, PRAUC, MCC at 0.5 under controls C0-C5
  - Partner sensitivity: 1 - Spearman(p(A|B), p(A|B')), averaged over K non-cognate draws
  - Cognate discrimination: Top-1 Accuracy and MRR ranking true vs K=10 non-cognate partners
  - Statistical testing: Cluster bootstrap CIs (1,000 reps) & Paired Wilcoxon with Holm correction

GPL-3.0 – Reliable partner-aware PPI site prediction on SPPIDER-seq
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import List, Optional, Tuple


def compute_spearman_distance(
    p_true: np.ndarray,
    p_ctrl: np.ndarray,
) -> float:
    """
    Partner sensitivity for a single pair.
    Returns 1 - Spearman(p(A|B), p(A|B')) where B' is the control partner.
    Higher means predictions changed more when partner was swapped.
    """
    if len(p_true) < 4:
        return float("nan")
    rho, _ = stats.spearmanr(p_true, p_ctrl)
    if np.isnan(rho):
        return float("nan")
    return float(1.0 - rho)


def compute_cognate_discrimination(
    p_true_partner: np.ndarray,
    p_non_cognate_partners: List[np.ndarray],
    y_true: np.ndarray,
) -> Tuple[float, float]:
    """
    For a single focal protein with true labels y_true:
    Compute PRAUC of p(A|B) for the true partner and for K non-cognate partners against y_true.
    Returns: (is_top_1: float 1.0 or 0.0, reciprocal_rank: float)
    Ties receive average rank. Skip proteins with no positives (returns nan).
    """
    from sklearn.metrics import average_precision_score

    if y_true.sum() == 0:
        return float("nan"), float("nan")

    scores = []
    try:
        scores.append(average_precision_score(y_true, p_true_partner))
    except Exception:
        return float("nan"), float("nan")

    for p_nc in p_non_cognate_partners:
        try:
            scores.append(average_precision_score(y_true, p_nc))
        except Exception:
            scores.append(0.0)

    true_score = scores[0]
    all_scores = np.array(scores)

    # Rank (1 = best). Ties: assign average rank.
    n_better = (all_scores > true_score).sum()
    n_equal = (all_scores == true_score).sum()
    rank = n_better + (n_equal + 1) / 2.0

    is_top1 = float(rank <= 1.0)
    mrr = float(1.0 / rank)
    return is_top1, mrr


def cluster_bootstrap_ci(
    per_cluster_values: dict,
    n_reps: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap: resample focal-protein clusters with replacement.
    Returns: (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    clusters = list(per_cluster_values.keys())
    all_vals = [v for vs in per_cluster_values.values() for v in (vs if isinstance(vs, list) else [vs])]
    if len(all_vals) == 0:
        return float("nan"), float("nan"), float("nan")

    boot_means = []
    for _ in range(n_reps):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_vals = []
        for c in sampled_clusters:
            v = per_cluster_values[c]
            if isinstance(v, list):
                boot_vals.extend(v)
            else:
                boot_vals.append(v)
        boot_vals = [x for x in boot_vals if not np.isnan(x)]
        if boot_vals:
            boot_means.append(np.mean(boot_vals))

    if not boot_means:
        return float("nan"), float("nan"), float("nan")

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(all_vals)), lo, hi
