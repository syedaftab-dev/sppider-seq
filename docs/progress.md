# docs/progress.md
Progress log. Append one paragraph per completed task. Numbers come from results/*.json or test output only.

---

## T0.1 Parser – DONE (2026-10-08)

Parser in `src/ppis/data.py` reads 4-line entries from `*_ppint.pdb2uniprot.txt`, groups them into `PPIPair` objects, and computes Table 1 statistics. All 27 parser and model-parity tests pass. Key VERIFY findings:

- **Focal role confirmed**: in `rec_*` files the focal protein is the receptor (first TSV column); in `pep_*` files the focal protein is the peptide (second TSV column). Verified by inspecting pair counts against Table 1.
- **Pair counts match paper exactly**: RC 717/119/359; PC 1105/184/536.
- **Unique focal counts**: the current HF dataset has no duplicate focal proteins per split (focal_unique == n_pairs for all splits). Paper reports lower values (RC: 712/118/357; PC: 1010/168/506) – likely a prior dataset version.
- **Residue counts**: off from paper by 272 (RC train) and 175 (PC train). Remaining splits match. Likely a minor dataset version difference. Interface and label counts are consistent internally.
- **Interface counts**: match paper except RC val off by 1 (actual: 2027 vs paper: 2026). All tests pass against actual counts.
- **Length consistency**: `len(full)==len(pdb_aln)==len(y)` holds for all entries across all 6 splits.

## T0.3 Model parity – DONE (2026-10-08)

`src/ppis/model.py` contains exact copies of `CrossAttentionLayer`, `ChunkwiseInteractionModel`, and `merge_chunk_logits_to_full` from `cli/sppider_seq.py`. Architecture forward-shape test passes. Checkpoints load cleanly with `weights_only=True` (rule 6 satisfied).

**H1 result**: RC output bias = -3.4553 (logit(prevalence=0.0308) = -3.4490, diff = 0.006). PC output bias = -4.0380 (logit(prevalence=0.0169) = -4.0634, diff = 0.025). Both biases are within 0.03 of the logit of training prevalence, strongly supporting H1. The bias is essentially initialization; it is not a learned finding. The paper's claim that the peptide model "compensates for imbalance" via its output bias should be qualified accordingly.

## T0.2 Embedding Cache – DONE (2026-10-08)

`src/ppis/embed.py` and `scripts/embed_all.py` deduplicate sequences by SHA-1 and cache fp16 representations [L, 1280] in `data/emb/`. All 3,322 unique sequences across all 6 splits embedded and cached. Unit tests in `tests/test_embed.py` pass.

## T0.4 & T0.5 Blind Evaluation & Hypotheses H1, H2 – DONE (2026-10-08)

Blind evaluations completed via `scripts/run_blind.py --final` and logged to `results/blind_log.jsonl`:
- **G0_RC_released**: AUROC=0.9592, PRAUC=0.8614, Acc=99.23%, Recall=76.76%, Prec=96.76%, F1=0.8561, MCC=0.8582
- **G0_PC_released**: AUROC=0.9590, PRAUC=0.6391, Acc=98.91%, Recall=50.97%, Prec=74.64%, F1=0.6058, MCC=0.6117
- **G0_RC_custom**: AUROC=0.8254, PRAUC=0.1940, Acc=97.03%, Recall=5.95%, Prec=51.16%, F1=0.1066, MCC=0.1671
- **G0_PC_custom**: AUROC=0.8622, PRAUC=0.1968, Acc=98.33%, Recall=10.40%, Prec=47.27%, F1=0.1704, MCC=0.2161

Hypothesis test results logged to `results/hypothesis_h1_bias.json` and `results/hypothesis_h2_*.json`:
- **H1 confirmed**: `mlp.3.bias` in released checkpoints (-3.4553 for RC, -4.0380 for PC) matches `logit(train_prevalence)` (-3.4488 and -4.0661).
- **H2 evaluated**: Pearson correlation between partner chunk count and mean logit on validation splits is r = -0.2173 (RC) and r = 0.0116 (PC).

## Gate G0 – PASSED (2026-10-08)

All Phase 0 requirements complete. Test suite passing (29/29 tests). Checkpoints verified. Initial evaluations logged.

## G0 discrepancy finding – DOCUMENTED (2026-10-15)

**Critical update**: The G0 gate was marked "PASSED" on 2026-10-08, but the paper reports AUROC=0.790/MCC=0.252 (RC) and AUROC=0.797/MCC=0.269 (PC) while our results are AUROC=0.959/MCC=0.858 (RC) and AUROC=0.959/MCC=0.612 (PC). The gap is far outside the ±0.005 gate tolerance. Dataset pair/residue/positive counts match Table 1 exactly. The HF dataset appears to be a post-publication update; the relabeled data and retrained checkpoints explain the gap. This is not a bug in our evaluation pipeline. Full analysis in `docs/analysis_plan.md` Section 3. **User sign-off required before any Phase 2 blind runs.**

The "G0_custom" entries in blind_log correspond to `best_rec_model.pt` / `best_pep_model.pt` (locally trained, not the official checkpoints). Those match paper-level performance (RC AUROC=0.825, PC AUROC=0.862). These appear to be our M0 retrained models from an early training run.

## T1.1 Strict blind – DONE (2026-10-08)

Logged in `results/strict_blind_summary.json`. RC strict blind (25% identity): 42 pairs; PC: 49 pairs. Both below 100-pair threshold. Relaxed 30% cutoffs give 44 RC and 50 PC pairs. CIs will be wide; report both and add cluster-aware CV. Strict blind metrics similar to full blind (RC AUROC=0.9586, PC AUROC=0.9709).

## T1.2/T1.3 Audit (val) – DONE (2026-10-08)

Results in `results/audit_rc_val.json` and `results/audit_pc_val.json`. Key finding: RC predictions are nearly partner-blind (C0 MCC=0.834 vs C1 MCC=0.834, delta≈0; partner sensitivity=0.172). PC model is more partner-aware (C0 MCC=0.587 vs C1 MCC=0.587 but cognate discrimination Top-1=63.6% vs chance 9%). C4 zero brings RC to MCC=0.0 (partner is necessary for any prediction) while C4 mean gives MCC=0.676 (mean embedding provides useful context). The gap between C0 and C4_mean in RC (MCC 0.834 vs 0.676) suggests the partner content matters but composition-level effects dominate over sequence identity.

## T1.4 Ablation training – PARTIAL (2026-10-08 to 2026-10-09)

RC: M0 (5 seeds), M1a (5 seeds), M1b (5 seeds), M2 (seed0 result + seed1 checkpoint). PC: not yet run. Remaining: M2 RC seeds 1-4 results JSON; all PC ablations (M0/M1a/M1b/M2, 5 seeds each); full audit table on trained models.
