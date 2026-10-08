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

## Tasks remaining in Phase 0

- T0.2: Embedding cache (`embed.py`) – code written, not yet tested end-to-end.
- T0.4: Blind evaluation of released checkpoints – **pending** (requires full GPU embedding pass; ask before starting, per rule 7 – estimated ~1 GPU-hour).
- T0.5: Hypotheses H2 and H5 on validation data – pending T0.4.
- Gate G0: pending T0.4.
