# AGENTS.md: Reliable partner-aware PPI site prediction on SPPIDER-seq

Project root file. Antigravity loads it before every agent task. Keep it at the repo root.
Owner: Aftab and team (3 people). Deadline: submit before March 2027.
Read sections 0 to 3 before doing anything. Then work phase by phase (section 6 onward).

---

## 0. Agent rules (hard rules, always apply)

1. **Blind set is sealed.** No tuning, plotting, or debugging on blind data. Blind runs go only through `scripts/run_blind.py --final`. That script appends every call to `results/blind_log.jsonl` (timestamp, git hash, config, model). Max one blind evaluation per final model.
2. **No invented numbers.** Every number in a doc, table, or message must come from a file in `results/`. If a run fails or is skipped, say so. Unverified claims get the tag `VERIFY`.
3. **Gates are blocking.** Do not start a phase until the previous gate (G0 to G3) passes. If a gate fails twice, stop and ask the user.
4. **Reproducibility.** Seed everything. Every run reads a YAML config from `configs/` and writes `results/<run_id>.json` with config, git hash, seed, metrics. Default seeds: 0,1,2,3,4.
5. **License.** SPPIDER-seq code and data are GPL-3.0. Any copied code keeps its header and gets an attribution comment. This repo is GPL-3.0.
6. **Checkpoint safety.** Load checkpoints only from the official HF repo `aporollo-lab/SPPIDER-seq`. Try `torch.load(..., weights_only=True)` first. Do not use pickle fallbacks without asking.
7. **Ask before** long jobs (more than 1 GPU-hour), large downloads (more than 5 GB), or deleting anything.
8. **Small commits.** One branch per phase (`phase0-repro`, `phase1-audit`, `phase2-calib`, `phase3-pu`). Tests must pass before merge.
9. **Style.** Short comments, type hints, no clever tricks. Docs and reports: plain, direct sentences. No overclaiming. Write "to our knowledge", never "first".
10. **Statistics.** Pooled residue metrics alone are not enough. Always also report per-protein metrics with cluster-bootstrap CIs (section 7).

---

## 1. Research questions

Source paper: Porollo et al., SPPIDER-seq (bioRxiv 2026.04.28.721359; published in Bioinformatics 2026). Frozen ESM-2 650M plus unilateral cross-attention head. Two models: receptor-centric (RC) and peptide-centric (PC). Trained on BioLiP receptor-peptide pairs.

Our project: **Is partner-aware PPIS actually reliable?**

- **RQ1 Partner usage audit.** Do the models use the partner? How much does it depend on partner-side leakage? Leakage-controlled benchmark.
- **RQ2 Calibration and uncertainty.** Are probabilities calibrated? Can we give residue sets with finite-sample error guarantees (conformal)? Do the repo's scramble-null q-values control real false discoveries?
- **RQ3 Label noise.** Unresolved residues are labeled negative. What happens if we mask or model them (PU / PNU)?

Honest outcomes are all publishable: partner-blind result, failed coverage under shift, no PU gain.

Optional stretch (only after G3): hard-negative partner pairs for binary interaction discrimination. The C2 controls in Phase 1 already produce hard non-cognate partners.

---

## 2. What the SPPIDER-seq repo contains (read from code)

Repo: https://github.com/aporollo-lab/SPPIDER-seq (GPL-3.0). Models and datasets are on Hugging Face, not in git.

| Fact | Consequence |
|---|---|
| Inference only. No training script, no hyperparameters (lr, batch, optimizer, seed) | We write our own trainer |
| Files: `cli/sppider_seq.py` (main), `cli/parse_sppider_seq.py` (aggregation over partners), `cli/alanine_scanning*.py`, `cli/scramble_fasta.py`, `cli/amplify_sequence_signal.py`, notebook, examples | Reuse classes, not the CLI, for bulk work |
| Model `ChunkwiseInteractionModel`: `nn.MultiheadAttention(1280, 16)` then `LayerNorm(query + attn_out)` then MLP 1280 to 1280 to 1 with dropout 0.1. Sigmoid applied at inference | Paper omits the residual and norm. Copy the class exactly. `VERIFY` against source |
| ESM-2 `esm2_t33_650M_UR50D` frozen, window 1022, stride 512 | Precompute and cache embeddings |
| Partner chunks merged by **max over logits**. Query windows merged by **max over logits** | H2, H5 below |
| Output bias is a constructor argument `initial_bias` | H1 below |
| Null: scramble the **partner**, default 5 scrambles, z-score on probabilities with normal approximation, FDR across residues | Tests partner dependence, not truth. Audit in Phase 2 |
| `parse_sppider_seq.py` aggregates per residue across partners: mean, q25, q75, max, list of partners at cutoffs 0.7, 0.5, 0.3, 0.1 | This is paper Fig. 3. No per-partner divergence statistic |
| Dataset entries: 4 lines each: header `PDB_chain|UniProt|ChainPair`; full UniProt sequence; PDB-derived sequence with `-` for unresolved residues; binary labels. Unresolved positions are labeled 0 | Mask for Phase 3 is already in the data |
| Sizes: peptide-centric about 1,830 rows (train 1.11k, val 184, blind 536). Receptor-centric about 1.2k rows | Small data. Use many seeds |

### Hypotheses to test first (cheap, run in Phase 0)

- **H1.** Checkpoint output bias is mostly initialization. Reported biases are -3.40 (RC) and -3.96 (PC). logit(train prevalence) is ln(0.0308/0.9692) = -3.45 and ln(0.0169/0.9831) = -4.06. Check by loading the checkpoint and by comparing to init. If true, the paper's "peptide model compensates for imbalance" is not a learned finding.
- **H2.** Max over partner-chunk logits inflates scores for partners longer than 1022 residues (more chunks, more chances). Plot mean logit vs number of partner chunks.
- **H3.** 5 scrambles give unstable null p-values. Test with 5 vs 20+ scrambles.
- **H4.** RC model is nearly partner-blind (paper: MCC 0.252 true vs 0.223 scrambled).
- **H5.** Query overlap windows (stride 512 on window 1022) give residues in overlaps two shots under max pooling. Plot mean logit vs position within overlap.

---

## 3. Paper numbers to reproduce (Gate G0)

Pooled residue-level, threshold 0.5 unless noted. Source: SPPIDER-seq v1 Tables 1 and 3.

| Set | Model | AUROC | PRAUC | Acc % | Recall % | Prec % | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| RC-BS359 | RC | 0.790 | 0.212 | 97.01 | 13.97 | 49.47 | 0.218 | 0.252 |
| RC-BS359 scrambled partners | RC | n/a | n/a | 96.95 | 12.08 | 45.57 | 0.191 | 0.223 |
| PC-BS536 | PC | 0.797 | 0.205 | 98.32 | 16.15 | 46.95 | 0.240 | 0.269 |
| PC-BS536 scrambled partners | PC | n/a | n/a | 98.23 | 1.84 | 16.22 | 0.033 | 0.049 |

Dataset counts (Table 1), used as parser unit tests:

| Model | Split | Unique focal / pairs | Residues | Interface (n, %) |
|---|---|---|---|---|
| RC | train | 712 / 717 | 409,420 | 12,612 (3.08%) |
| RC | val | 118 / 119 | 57,307 | 2,026 (3.54%) |
| RC | blind | 357 / 359 | 211,219 | 6,300 (2.98%) |
| PC | train | 1010 / 1105 | 694,461 | 11,705 (1.69%) |
| PC | val | 168 / 184 | 114,388 | 1,985 (1.74%) |
| PC | blind | 506 / 536 | 343,673 | 5,666 (1.65%) |

Unclear whether residue totals count unique focal proteins or pairs. Test both, record which matches.

---

## 4. Environment

- Python 3.12 venv. No DGL needed.
- `pip install "torch>=2.1" "transformers>=4.40" numpy scipy pandas matplotlib biopython statsmodels scikit-learn huggingface_hub pyyaml tqdm pytest`
- MMseqs2 for clustering (conda bioconda or apt). `VERIFY` install route.
- Compute: only ESM-2 embedding needs a GPU. Options: (a) run `scripts/embed_all.py` once on a free GPU runtime (Colab or Kaggle), download the shards; (b) CPU overnight. The head and all controls run fine on CPU or small GPU on cached embeddings.
- Cache: fp16, about 2.5 KB per residue. Expect 5 to 10 GB total including scrambles. Use a data drive path from `configs/paths.yaml`.

Downloads:

```python
from huggingface_hub import snapshot_download, hf_hub_download
ds  = snapshot_download("aporollo-lab/SPPIDER-seq-datasets", repo_type="dataset")
rec = hf_hub_download("aporollo-lab/SPPIDER-seq", "models/sppider_seq_receptor.pt")  # VERIFY filename
pep = hf_hub_download("aporollo-lab/SPPIDER-seq", "models/sppider_seq_peptide.pt")
```

Clone the original repo to `third_party/SPPIDER-seq` (read only). Copy needed classes into `src/ppis/model.py` with attribution.

---

## 5. Repo layout

```
reliable-ppis/
  AGENTS.md  LICENSE(GPL-3.0)  README.md  pyproject.toml
  third_party/SPPIDER-seq/            # read only clone
  configs/    paths.yaml  train_*.yaml  eval_*.yaml
  data/       (not committed) raw/ emb/ clusters/ splits/
  src/ppis/   data.py embed.py model.py train.py eval.py metrics.py
              controls.py calibrate.py conformal.py pu.py stats.py
  scripts/    embed_all.py run_blind.py make_strict_blind.py ...
  tests/      test_parser.py test_model_parity.py test_metrics.py ...
  notebooks/  00_repro.ipynb 10_audit.ipynb 20_calib.ipynb 30_pu.ipynb
  results/    *.json  blind_log.jsonl
  figures/
  docs/       analysis_plan.md  paper_outline.md
```

Write `docs/analysis_plan.md` (metrics, thresholds, controls, decision rules) **before** any blind run. Commit it.

---

## 6. Phases

### Phase 0: Setup and reproduction (W1-2, Oct 8 to Oct 21)

**T0.1 Parser** (`data.py`)
- Read 4-line entries. Output per entry: `name, uniprot, pair_id, full, pdb_aln, observed (bool list: pdb char != '-'), y (int list)`.
- Group entries into pairs (focal, partner). Work out which entry is focal in `rec_*` vs `pep_*` files. `VERIFY` by inspecting data and by the unit tests below.
- Tests: Table 1 counts exact for all 6 splits. Assert `len(full)==len(pdb_aln)==len(y)`.

**T0.2 Embedding cache** (`embed.py`, `scripts/embed_all.py`)
- Dedupe by sha1 of sequence. Same chunking as the repo's embedding function (window 1022, stride 512).
- Output `data/emb/<sha1>.npy` (fp16, shape L x 1280) plus index JSON.
- Test: embedding of 3 sequences matches the repo CLI output within fp16 tolerance.

**T0.3 Model parity** (`model.py`)
- Copy `CrossAttentionLayer`, `ChunkwiseInteractionModel`, `merge_chunk_logits_to_full`. Wrap in a function that takes cached embeddings, not the ESM model.
- Test: logits from our wrapper equal logits from the original CLI function on 5 pairs, max abs diff < 1e-3.

**T0.4 Evaluate released checkpoints on blind** (this is the one allowed blind use before G0; log it)
- Compute pooled AUROC, PRAUC, Acc, Recall, Precision, F1, MCC at 0.5 for RC on RC-BS359 and PC on PC-BS536.

**T0.5 Hypotheses H1, H2, H5** on validation data (not blind).

**Gate G0.** AUROC and PRAUC within 0.005 of Section 3, MCC within 0.01. If not, debug in this order: label mapping, focal/partner role, chunk merge, handling of `X` and non-standard residues, sequences over 1022, off-by-one in chunk ends. Ask the user after two failed rounds.

### Phase 1: Partner-usage audit (W3-6, Oct 22 to Nov 18)

**T1.1 Clustering and strict blind** (`scripts/make_strict_blind.py`)
- Pool all proteins from all splits, both roles. MMseqs2 easy-cluster at 25% and 30% identity: `mmseqs easy-cluster all.fa clu tmp --min-seq-id 0.25 -c 0.8 -s 7.5` (`VERIFY` flags and coverage mode).
- **Strict blind** = blind pairs where neither focal nor partner shares a cluster with any train or val protein (any role). Save pair ids to `data/splits/strict_blind_{rc,pc}.json`.
- Report counts. If under 100 pairs, report that and also give a relaxed version (30% identity).
- Report released-checkpoint metrics on full vs strict blind (blind use, log it).

**T1.2 Controls** (`controls.py`). Each control returns a partner embedding provider for a pair.
- C0 true partner.
- C1 scrambled partner: residue permutation, composition preserved, 3 seeds (needs ESM passes, cache them).
- C2 non-cognate real partner: K=10 draws from the same split. Length within +-20% of true partner. Different cluster from the true partner. Cached embeddings only.
- C3 homolog partner: different protein, same cluster as true partner. Only where available.
- C4 partner removed: zero embedding and mean-embedding variants.
- C5 aggregator sensitivity (tests H2 without retraining): replace max over partner-chunk logits with mean and with logsumexp at inference. Report effect by partner chunk count.

**T1.3 Audit metrics** (`eval.py`)
- Per pair and per protein: AUROC, PRAUC, MCC at 0.5 under each control. Deltas vs C0.
- **Partner sensitivity:** `1 - Spearman(p(A|B), p(A|B'))`, averaged over K non-cognate draws. Floor is the scrambled value.
- **Cognate discrimination:** for pair i with true label y_i, compute PRAUC of p(A_i|B) for the true partner and for K=10 non-cognate partners against y_i. Report top-1 accuracy and MRR (ties get average rank; skip pairs with no positives). Chance is 1/(K+1).
- **Interaction signal (exploratory):** score s(A,B)=mean of top-k probabilities. AUROC for true vs non-cognate partners. The model was not trained for this. Report as exploratory.
- Stats: cluster bootstrap (1,000 reps, resample focal-protein clusters), paired Wilcoxon on per-protein deltas, Holm correction.

**T1.4 Retrained ablations** (`train.py`, section 8 for training spec)
- M0: our reproduction of the architecture and training.
- M1a: self-attention (context = query chunk), matched parameters.
- M1b: no cross-attention (LayerNorm plus MLP on the query only).
- M2: random partner at train time (resample each epoch within split, length-matched).
- 5 seeds each, RC and PC. Gate: M0 within 0.01 AUROC of the released model.
- Audit all of M0, M1a, M1b, M2, and the released checkpoints with the same controls.

**T1.5 Multi-partner set (optional, cap 1 week).** Proteins with at least 2 structures with distinct partners from BioLiP2 / PDB, filtered against SPPIDER-seq train at 30% identity. Need at least 100 focal proteins, else treat as case studies. Question: do predicted maps change with partner as true maps do?

**Gate G1.** Strict blind built. Audit table complete for released RC and PC and for M0/M1a/M1b/M2. Interpret both ways: model uses partner (sensitivity and discrimination above floor with CIs), or does not.

### Phase 2: Calibration and conformal (W7-10, Nov 19 to Dec 16)

**T2.1 Diagnostics** (`calibrate.py`)
- Reliability on **logit-quantile bins** (probabilities mostly below 0.3), equal-mass bins (15), ECE, Brier, log-loss.
- Stratify by protein length, partner chunk count, overlap position.

**T2.2 Post-hoc calibration, fit on validation only**
- Platt: `p = sigmoid(a*z + b)`, fit by L-BFGS on validation residues (observed residues only, see Phase 3).
- Isotonic (sklearn).
- Threshold chosen by max MCC on validation. Report test MCC at that threshold and at 0.5.
- AUROC does not change under monotone maps. Say so.

**T2.3 Operating-point reporting** (supports a fairer AF3 comparison)
- Precision at recall 0.65, recall at precision 0.30, full PR curves, per-protein top-k (k = 5, 10, L/10), hit rate (at least one true site in top-k).
- AF3 runs are optional and need the user's server access. If done, rank residues by an AF3-derived confidence and compare at matched recall.

**T2.4 Conformal** (`conformal.py`). Exchangeable unit = protein. Prediction set `C_t = {r : p_r >= t}`.
- **FNR control (conformal risk control).** Loss_i(t) = 1 - |C_t ∩ Y_i| / |Y_i|, defined only for proteins with at least one true site. Non-increasing as t decreases. Choose the largest t with `(n/(n+1)) * mean_i Loss_i(t) + 1/(n+1) <= alpha` (loss bound B=1). `VERIFY` against Angelopoulos et al. 2022.
- **FDR control (Learn-then-Test).** Loss_i(t) = FDP of C_t (0 if the set is empty). Not monotone, so use LTT. For t on a descending grid, test H_t: risk(t) > alpha using Hoeffding-Bentkus p-values for bounded losses. Fixed-sequence testing from high t to low t. Pick the smallest t still rejected. delta = 0.1. `VERIFY` against Angelopoulos et al. 2021.
- Always report with the chosen t: mean set size, recall, precision. An empty set trivially satisfies FDP, so recall must be shown.
- Mondrian variant: groups by protein length bin.
- Alpha grid: 0.1, 0.2, 0.3 for FNR; 0.1, 0.2, 0.3, 0.5 for FDR (FDR below 0.5 may be unreachable at this model quality, report that).
- Calibration sets are small (val: 118 RC, 168 PC proteins). Add cluster-aware 5-fold retrain on pooled data using our trainer to get larger calibration folds. Report empirical coverage per fold with CIs.
- Stress test: empirical coverage on full blind vs strict blind. Expect shift. Report it as a finding.

**T2.5 Audit the repo's null.** Use `cli` scramble-based q-values (default 5 scrambles, and 20+ scrambles on a subset of about 100 pairs, `VERIFY` cost). Question: do residues with q < 0.05 have FDP <= 0.05 against true labels on blind? Compare to calibrated-threshold FDP.

**Gate G2.** Calibration tables and coverage plots done for RC and PC. Document where guarantees hold or fail.

Preprint v1 target: right after G2 (mid-December to January), with Phases 1 and 2.

### Phase 3: Label noise and PU (W11-14, Dec 17 to Jan 13)

**T3.1 Residue states.** For each focal protein: P (observed and y=1), N_obs (observed and y=0), U (unresolved, `-`). Report the fraction of U per split and by protein length. This decides how much Phase 3 matters.

**T3.2 Variants** (`pu.py`), retrained with the same trainer, 5 seeds:

| Variant | Loss |
|---|---|
| V0 | BCE over all residues, U treated as negative (baseline, equals M0) |
| V1 | Masked BCE: U residues dropped from the loss |
| V2 | U residues weighted w in {0.1, 0.3, 0.5} as negatives |
| V3 | Supervised BCE on observed residues plus lambda times an nnPU term on U, class prior pi_U in {0.005, 0.01, 0.02, 0.03} and an estimated value; lambda in {0.1, 0.3, 1}. Exploratory |
| V4 | Pairwise ranking (AUC surrogate) loss on observed residues only |

nnPU term for U vs labeled positives P (Kiryo et al. 2017): `pi*E_P[l(g,+1)] + max(0, E_U[l(g,-1)] - pi*E_P[l(g,-1)])`. `VERIFY` the formula.
Caution: PU methods assume labeled positives are a random sample of all positives. Unresolved regions are systematically different (often disordered). V3 is therefore likely misspecified. Present V1 and V2 as the main variants, V3 as exploratory.

**T3.3 Evaluation**
- Standard blind metrics (all residues, as in the paper).
- Observed-only metrics (U excluded). This is the fair view.
- Per-protein top-k on observed residues.
- **Hidden positives:** residues unresolved in one entry but labeled 1 in another PDB entry of the same UniProt, with both entries in the same split. Measure recall of those residues at fixed precision. Small n, report counts and CIs. Under the pretrained models this set may be contaminated by train, so use it only for retrained models with cluster-separated splits.
- Same seeds and splits as Phase 1 retrains.

**Gate G3.** If V1 or V2 matches V0 on all-residue metrics and wins on observed-only or hidden positives: main result. Otherwise report as negative.

### Phase 4: Integration, writing, submission (W15-21, Jan 14 to Mar 3)

- Final pipeline: best loss variant, post-hoc calibration, conformal sets, evaluated on strict blind. One `--final` blind run per final model.
- Figures: (1) reproduction table, (2) full vs strict blind, (3) partner sensitivity distributions, (4) cognate discrimination, (5) reliability and coverage plots, (6) PU table, (7) per-protein top-k curves.
- Paper outline in `docs/paper_outline.md`: Abstract, Intro, Methods (data, controls, calibration, PU), Results (audit, calibration, PU, integrated), Discussion and limitations, Availability (GitHub, Zenodo DOI, GPL-3.0).
- Venue plan: preprint on bioRxiv, then Bioinformatics Advances, BMC-family, or Briefings in Bioinformatics depending on strength. Check current APCs and waiver rules at submission. Agree author order and corresponding author before writing.
- Run a literature rescan one week before writing (Handshake, X-PAIR, ReCLIP, Disobind, PepNN-Seq, ConGA-PepPI and anything new).

---

## 7. Evaluation protocol (applies everywhere)

- **Pooled metrics:** `roc_auc_score`, `average_precision_score`, MCC at threshold. Prevalence is 1.65 to 2.98%, so also report lift = PRAUC / prevalence.
- **Per-protein metrics:** AUROC, PRAUC, top-k hit. Skip proteins with no positives or no negatives.
- **Cluster bootstrap:** resample focal-protein clusters with replacement, 1,000 reps, percentile 95% CI. Report for pooled and per-protein means.
- **Tests:** paired Wilcoxon on per-protein metrics between conditions or models, Holm correction. Report effect sizes, not only p-values.
- **Seeds:** 5 per trained model. Report mean +- sd across seeds, then CI across proteins.
- **Do not** claim improvements smaller than the seed sd.

---

## 8. Training spec for our trainer

Unknown in the original (`VERIFY` by matching released performance). Use:
- Input: cached ESM-2 embeddings. One training example = one (focal, partner) pair. Loss over focal residues, token-level mean.
- Model: copied architecture. Output bias init = logit(train prevalence) only if H1 shows the checkpoint did the same. Otherwise zero init.
- Loss: `BCEWithLogitsLoss` (V0). Variants in Phase 3.
- Optimizer: Adam. LR grid {1e-4, 3e-4, 1e-3} on validation. Gradient accumulation over 8 to 16 pairs. Dropout 0.1.
- Early stopping: patience 10 on validation loss, max 50 epochs, keep best (as in the paper).
- Chunking: same window 1022 and stride 512 as inference. Same max-over-logits merging as the released code. Keep an option for mean merging for H2.
- Log: per-epoch train and val loss, val AUROC and PRAUC. Save best checkpoint with config.

---

## 9. Task cards for parallel agents (Antigravity Manager view)

After G0, three agents can run in parallel on separate branches. Merge weekly.

- **Agent A (data):** T0.1, T0.2, T1.1, T1.5, T3.1. Output: parser, embedding cache, clusters, splits, hidden-positive set.
- **Agent B (model and audit):** T0.3, T0.4, T1.2, T1.3, T1.4. Output: model parity, controls, audit, ablation models.
- **Agent C (calibration and PU):** T2.1 to T2.5, T3.2, T3.3. Starts on released checkpoints and updates when B delivers retrained models.

Each task is done when: code merged, tests green, `results/*.json` written, one-paragraph summary appended to `docs/progress.md` with exact numbers from `results/`.

---

## 10. Timeline

| Weeks | Dates | Work | Gate |
|---|---|---|---|
| 1-2 | Oct 8 to Oct 21 | Phase 0 | G0 |
| 3-6 | Oct 22 to Nov 18 | Phase 1 | G1 |
| 7-10 | Nov 19 to Dec 16 | Phase 2 | G2, preprint v1 |
| 11-14 | Dec 17 to Jan 13 | Phase 3 (holiday slack) | G3 |
| 15-18 | Jan 14 to Feb 10 | Integration, ablations, figures, draft | |
| 19-21 | Feb 11 to Mar 3 | Revise, preprint v2, submit | |

---

## 11. Risks and decision rules

- Released model not reproducible: debug per G0 list. If still off after two rounds, ask the user and consider contacting the authors.
- Model is partner-blind: this is the headline for RQ1. Continue with Phases 2 and 3 on the same story.
- Strict blind too small: relax identity cutoff, report both, add cluster-aware CV.
- Conformal coverage fails under shift: report as limitation with numbers.
- PU shows no gain: publish as negative result.
- Compute shortfall: reduce scramble seeds to 1, subsample the null audit, cap K to 5. Document every reduction.
- Scoop risk (Handshake, X-PAIR, ReCLIP, Disobind are active): preprint after G2, rescan literature monthly.
- Label definition: the dataset uses RSA change above 4% on chain separation. Do not change labels in the main analysis. A contact-distance relabel is an optional robustness check.

---

## 12. Unverified items (resolve while working)

- HF model filenames and dataset file names (`VERIFY` after download).
- Which dataset entries are focal vs partner in `rec_*` and `pep_*` files.
- Whether Table 1 residue totals count proteins or pairs.
- Exact model class details (read source, do not trust section 2 summary).
- H1 to H5 outcomes (not yet run).
- MMseqs2 flags for 25% identity clustering.
- Conformal formulas (CRC, LTT) and nnPU formula against the original papers.
- Current APCs and waiver rules at submission time.
- The v1 PDF differs from the published Bioinformatics version (published adds 5-fold CV and an IUPred2A/ANCHOR2 baseline). Cite the published version when it is final.
