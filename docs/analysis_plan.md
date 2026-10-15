# docs/analysis_plan.md
# Analysis plan for "Is partner-aware PPIS actually reliable?"
# Written before any Phase 2/3 blind runs. AGENTS.md rule: commit before any blind run.

## Status: active — Phase 0/1 in progress (as of 2026-10-15)

---

## 1. Datasets

| Model | Split | File prefix | Pairs | Focal (unique) | Residues | Interface |
|---|---|---|---|---|---|---|
| RC | train | rec_ | 717 | 717 | 409,148 | 12,612 (3.08%) |
| RC | val | rec_ | 119 | 119 | 57,307 | 2,027 (3.54%) |
| RC | blind | rec_ | 359 | 357 | 211,219 | 6,300 (2.98%) |
| PC | train | pep_ | 1,105 | 1,105 | 694,286 | 11,705 (1.69%) |
| PC | val | pep_ | 184 | 184 | 114,388 | 1,985 (1.74%) |
| PC | blind | pep_ | 536 | 506 | 343,673 | 5,666 (1.65%) |

Counts are from the current HuggingFace dataset (downloaded 2026-10-08).
The paper reports slightly different residue totals; pair counts match exactly.
The HF dataset is a revised post-publication version.

VERIFY: residue totals counted per pair (focal counted once per pair).

---

## 2. Metrics (all phases)

### Primary
- AUROC (sklearn.metrics.roc_auc_score)
- PRAUC (sklearn.metrics.average_precision_score)
- MCC at threshold (default 0.5)
- Acc, Recall, Precision, F1 at threshold
- Lift = PRAUC / prevalence

### Secondary
- Per-protein AUROC, PRAUC, top-k hit (k=5, 10, L/10)
- Cluster bootstrap 95% CI (1,000 reps, resample focal-protein clusters)
- Paired Wilcoxon on per-protein deltas, Holm correction

### Evaluation scope
- All-residue (as in the paper): include unresolved positions (labeled 0)
- Observed-only (fair view): exclude where pdb_aln == '-'
- Both reported everywhere.

### Threshold selection
- Default 0.5 for reproduction
- Platt-calibrated threshold (max MCC on val) for Phase 2 onwards

---

## 3. G0 discrepancy — dataset version

Released HF checkpoints achieve AUROC=0.959 / MCC=0.858 on RC blind
and AUROC=0.959 / MCC=0.612 on PC blind (all-residue). Logged in blind_log.jsonl.

The paper reports AUROC=0.790 / MCC=0.252 (RC) and AUROC=0.797 / MCC=0.269 (PC).

Working hypothesis: HF dataset has been updated since the preprint.
Pair/residue/positive counts match the paper; label prevalence is unchanged.
The performance improvement likely reflects improved labels or re-training.

VERIFY by checking HF commit history and contacting authors if needed.

Impact: our reproduction numbers will not match paper Table 3.
We report both, note the discrepancy, and use the current HF dataset throughout.

G0 gate status: formally fails the within-0.005 AUROC check against the paper.
Interpreted as a dataset-version discrepancy. Proceeding pending user sign-off.

---

## 4. Phase 1 audit controls

| Control | Description |
|---|---|
| C0 | True partner (cached embeddings) |
| C1 | Row-permuted partner embedding (seed 0) — composition preserved, position destroyed |
| C2 | Non-cognate real partner (K=10, length +-20%, different cluster) |
| C3 | Homolog partner (same 25% identity cluster) — only where available |
| C4a | Zero partner embedding |
| C4b | Global mean partner embedding |
| C5a | Mean over partner-chunk logits (vs max in C0) |
| C5b | LogSumExp over partner-chunk logits |

Partner sensitivity: mean over K of (1 - Spearman(p(A|B_true), p(A|B'))).
Floor = value under row-permuted C1.

Cognate discrimination: PRAUC of true vs K non-cognate partners scored against y_i.
Top-1 accuracy and MRR. Chance level = 1/(K+1) for K=10.

---

## 5. Phase 2 calibration plan

### T2.1 Diagnostics
- 15 equal-mass bins on logit scale; ECE, Brier, log-loss
- Stratify by: protein length, partner chunk count, overlap position

### T2.2 Post-hoc calibration
- Platt scaling: sigma(a*z + b), L-BFGS on val residues (observed only)
- Isotonic regression (sklearn)
- Calibration fit on val only; evaluated on blind

### T2.4 Conformal prediction
- FNR control (Angelopoulos et al. 2022):
  (n/(n+1)) * mean_i L_i(t) + 1/(n+1) <= alpha
  where L_i(t) = 1 - |C_t ∩ Y_i| / |Y_i|
- FDR control (LTT, Angelopoulos et al. 2021):
  fixed-sequence from high t to low t, Hoeffding-Bentkus p-values, delta=0.1
- Alpha grids: FNR {0.1, 0.2, 0.3}; FDR {0.1, 0.2, 0.3, 0.5}
- Stress test: coverage on full blind vs strict blind

---

## 6. Phase 3 PU plan

Residue states: P (observed y=1), N_obs (observed y=0), U (pdb_aln == '-').

Loss variants (V0 baseline through V4 pairwise ranking).
Primary comparison: V1 and V2 vs V0 on observed-only metrics and hidden positives.

---

## 7. Decision rules

G0 (caveat): proceed with updated dataset; user sign-off on discrepancy required.
G1: strict blind built; full audit table for released models and M0/M1a/M1b/M2.
G2: calibration tables and coverage plots for RC and PC.
G3: V1 or V2 wins on observed-only or hidden positives; else negative result.

If a gate fails twice: stop and ask the user.

---

## 8. Pre-blind checklist

1. This analysis_plan.md exists and is committed
2. Config YAML written to configs/
3. Model checkpoint frozen
4. Run ID is unique
5. blind_log.jsonl does not contain the run_id
6. User has approved the model+config

---

*Last updated: 2026-10-15*
