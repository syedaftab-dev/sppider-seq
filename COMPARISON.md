# Baseline vs. Audited Models: Experimental Comparison

This document presents a focused empirical comparison between the published **SPPIDER-seq baseline** (Porollo et al., *Bioinformatics* 2026) and our **audited implementations**, centered on three core research questions:

1. **Partner-Usage Audit:** Evaluating whether the model genuinely leverages partner sequence information versus functioning partner-blind.
2. **Probability Calibration:** Assessing and correcting uncalibrated raw model logits via post-hoc scaling.
3. **Strict-Blind Benchmark:** Quantifying performance on a homology-free benchmark with zero train/test sequence overlap.

All metrics are directly computed from evaluation runs logged in `results/`.

---

## 1. Partner-Usage Audit (Partner-Aware vs. Partner-Blind)

### Methodology
The original SPPIDER-seq architecture incorporates a unilateral cross-attention layer over partner sequence chunks. To evaluate whether the model genuinely depends on partner-specific features or merely infers query interface propensities, systematic partner perturbation controls were executed on the validation split (`results/audit_rc_val.json`, `results/audit_pc_val.json`):

* **C0 (True Cognate Partner):** Baseline prediction with the true biological partner.
* **C1 (Scrambled Partner):** Partner sequence randomly permuted while preserving exact amino-acid composition.
* **C4 Zero:** Partner embedding zeroed out completely.
* **C4 Mean:** Partner replaced with the global mean background embedding.
* **Cognate Discrimination:** Top-1 accuracy and Mean Reciprocal Rank (MRR) when ranking the true partner against $K=10$ non-cognate, length-matched decoys (random baseline = $9.09\%$).

### Results

| Model | Control Condition | AUROC | PRAUC | Precision (%) | Recall (%) | MCC | Lift |
|---|---|---|---|---|---|---|---|
| **Receptor-Centric (RC)** | **C0 (True Partner)** | **0.9439** | **0.8205** | **94.95** | **74.15** | **0.8341** | **23.20×** |
| Receptor-Centric (RC) | C1 (Scrambled Partner) | 0.9434 | 0.8200 | 94.94 | 74.10 | 0.8338 | 23.18× |
| Receptor-Centric (RC) | C4 (Zero Embedding) | 0.5000 | 0.0354 | 0.00 | 0.00 | 0.0000 | 1.00× |
| Receptor-Centric (RC) | C4 (Mean Embedding) | 0.9386 | 0.7600 | 95.38 | 48.89 | 0.6756 | 21.49× |
| **Peptide-Centric (PC)** | **C0 (True Partner)** | **0.9527** | **0.6156** | **66.82** | **52.75** | **0.5874** | **35.47×** |
| Peptide-Centric (PC) | C1 (Scrambled Partner) | 0.9526 | 0.6131 | 66.92 | 52.59 | 0.5870 | 35.33× |
| Peptide-Centric (PC) | C4 (Zero Embedding) | 0.5000 | 0.0174 | 0.00 | 0.00 | 0.0000 | 1.00× |
| Peptide-Centric (PC) | C4 (Mean Embedding) | 0.8870 | 0.3550 | 69.20 | 22.52 | 0.3893 | 20.46× |

### Partner Discriminability

| Metric | Receptor-Centric (RC) | Peptide-Centric (PC) | Baseline / Chance |
|---|---|---|---|
| **Partner Sensitivity** ($1 - \text{Spearman}(p_A, p_B)$) | **0.1719** | **0.2829** | 0.00 (invariant) |
| **Cognate Discrimination Top-1 Accuracy** | **28.57%** | **63.59%** | 9.09% ($1 / 11$) |
| **Cognate Discrimination MRR** | **0.4979** | **0.7628** | 0.2745 (random rank) |

---

## 2. Probability Calibration (Reliability & Risk Control)

### Methodology
In native PPI data, interface residues exhibit extreme class imbalance ($1.6\%$ to $3.1\%$ prevalence). Raw sigmoid outputs from uncalibrated networks typically distort posterior probabilities. 

Calibration was evaluated on validation residues across 15 equal-mass quantile bins, and post-hoc **Platt scaling** (logistic calibration) was fitted to align predicted scores with true empirical frequencies (`results/calibration_rc_seed3.json`, `detailed_report.md` Section 3.4).

### Calibration Results

| Model Split | Calibration Method | ECE (Expected Calibration Error) ↓ | Brier Score ↓ | Log-Loss ↓ | AUROC |
|---|---|---|---|---|---|
| **Receptor-Centric (RC)** | **Raw Baseline** | 0.042 | 0.087 | 0.1273 | 0.9439 |
| **Receptor-Centric (RC)** | **Platt Calibrated** | **0.019** *(-54.8%)* | **0.074** *(-14.9%)* | **0.1082** | **0.9439** *(preserved)* |
| **Peptide-Centric (PC)** | **Raw Baseline** | 0.058 | 0.092 | 0.1415 | 0.9527 |
| **Peptide-Centric (PC)** | **Platt Calibrated** | **0.023** *(-60.3%)* | **0.080** *(-13.0%)* | **0.1190** | **0.9527** *(preserved)* |

* Platt scaling reduces Expected Calibration Error (ECE) by $>50\%$ while preserving rank ordering and AUROC.

---

## 3. Strict-Blind Benchmark (Data Leakage Control)

### Methodology
Standard dataset splitting can retain homologous proteins across train and test partitions. To evaluate true out-of-distribution generalization, **MMseqs2** (`--min-seq-id 0.25 -c 0.8 -s 7.5`) was used to cluster all focal and partner sequences at **25% sequence identity** (alongside a 30% relaxed cutoff). 

Pairs where either focal or partner shares a sequence cluster with any training or validation protein were filtered out, producing a strict homology-free test set (`results/strict_blind_summary.json`).

### Benchmark Results

| Model | Evaluation Set | Pairs | Total Residues | Interface Positives | AUROC | PRAUC | Precision (%) | Recall (%) | MCC |
|---|---|---|---|---|---|---|---|---|---|
| **RC Released** | **Original Full Blind** | 359 | 211,219 | 6,300 (2.98%) | **0.9592** | **0.8614** | 96.76 | 76.76 | **0.8582** |
| **RC Released** | **Strict Blind (25% ID)** | 42 | 27,782 | 827 (2.98%) | 0.9586 | 0.8398 | 98.87 | 73.88 | 0.8511 |
| **RC Released** | **Strict Blind (30% ID)** | 44 | 28,914 | 861 (2.98%) | 0.9598 | 0.8408 | 98.88 | 74.00 | 0.8504 |
| *RC Paper Published* | *Published Table 3* | *359* | *211,219* | *6,300 (2.98%)* | *0.7900* | *0.2120* | *49.47* | *13.97* | *0.2520* |
| **PC Released** | **Original Full Blind** | 536 | 343,673 | 5,666 (1.65%) | **0.9590** | **0.6391** | 74.64 | 50.97 | **0.6117** |
| **PC Released** | **Strict Blind (25% ID)** | 49 | 31,450 | 516 (1.64%) | 0.9709 | 0.7157 | 80.20 | 59.88 | 0.6792 |
| **PC Released** | **Strict Blind (30% ID)** | 50 | 32,108 | 529 (1.65%) | 0.9673 | 0.7068 | 80.00 | 59.18 | 0.6744 |
| *PC Paper Published* | *Published Table 3* | *536* | *343,673* | *5,666 (1.65%)* | *0.7970* | *0.2050* | *46.95* | *16.15* | *0.2690* |
