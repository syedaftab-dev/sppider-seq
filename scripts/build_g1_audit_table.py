"""
scripts/build_g1_audit_table.py
Gate G1 audit table builder.

Reads all results/{variant}_{mt}_seed{seed}.json files,
computes mean ± sd across seeds, and prints a markdown table
suitable for docs/progress.md and the paper.

Also runs partner-usage audit (C0 vs C4_mean) on the best seed
of each trained RC and PC model.

Usage:
  python3 scripts/build_g1_audit_table.py

Output:
  results/ablation_summary.json
  results/g1_audit_table.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ppis.data import load_split
from ppis.model import load_checkpoint, predict_from_embeddings
from ppis.metrics import pooled_metrics, metrics_to_dict
from ppis.embed import seq_sha1, chunk_sequence
from ppis.controls import get_c4_zero_partner_embedding, get_c4_mean_partner_embedding


def load_model_results(variant: str, model_type: str, seeds: list[int]) -> list[dict]:
    """Load result JSONs for a variant-modeltype combo across seeds."""
    short_mt = "RC" if model_type == "receptor-centric" else "PC"
    results = []
    for seed in seeds:
        p = Path(f"results/{variant}_{short_mt}_seed{seed}.json")
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
    return results


def summarize_seeds(results: list[dict]) -> dict:
    """Compute mean ± sd of best_val_metrics across seeds."""
    if not results:
        return {}
    metrics_keys = ["auroc", "prauc", "mcc", "acc", "recall", "precision", "f1"]
    out = {"n_seeds": len(results)}
    for k in metrics_keys:
        vals = [r["best_val_metrics"][k] for r in results if "best_val_metrics" in r]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_sd"] = float(np.std(vals))
    return out


def run_quick_audit(
    checkpoint: str,
    model_type: str,
    split: str = "val",
    emb_dir: str = "data/emb",
    data_dir: str = "SPPIDER-seq-datasets",
    device: torch.device | None = None,
) -> dict:
    """Quick audit: C0 true partner, C4 zero, C4 mean on a split."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(checkpoint, device)
    pairs = load_split(data_dir, model_type, split)

    y_c0, p_c0 = [], []
    y_c4z, p_c4z = [], []
    y_c4m, p_c4m = [], []

    # Compute global mean partner embedding
    all_p_embs = []
    for pair in pairs:
        p_sha = seq_sha1(pair.partner.full)
        emb = np.load(Path(emb_dir) / f"{p_sha}.npy")
        all_p_embs.append(np.mean(emb, axis=0))
    global_mean = np.mean(all_p_embs, axis=0).astype(np.float16)

    for pair in pairs:
        focal = pair.focal
        partner = pair.partner
        f_sha = seq_sha1(focal.full)
        p_sha = seq_sha1(partner.full)

        f_emb = np.load(Path(emb_dir) / f"{f_sha}.npy")
        p_emb = np.load(Path(emb_dir) / f"{p_sha}.npy")

        q_chunks_meta = chunk_sequence(focal.full)
        c_chunks_meta = chunk_sequence(partner.full)

        q_tensors = [
            torch.from_numpy(f_emb[c["start_idx"]:c["end_idx"]].astype(np.float32)).to(device)
            for c in q_chunks_meta
        ]
        c_tensors = [
            torch.from_numpy(p_emb[c["start_idx"]:c["end_idx"]].astype(np.float32)).to(device)
            for c in c_chunks_meta
        ]
        q_starts = [c["start_idx"] for c in q_chunks_meta]
        L = len(focal.full)

        # C0 true
        probs_c0 = predict_from_embeddings(model, q_tensors, q_starts, L, c_tensors, device).cpu().numpy()
        y_c0.extend(focal.y)
        p_c0.extend(probs_c0.tolist())

        # C4 zero
        c_zero = [torch.zeros_like(t) for t in c_tensors]
        probs_c4z = predict_from_embeddings(model, q_tensors, q_starts, L, c_zero, device).cpu().numpy()
        y_c4z.extend(focal.y)
        p_c4z.extend(probs_c4z.tolist())

        # C4 mean
        c_mean = [
            torch.from_numpy(
                get_c4_mean_partner_embedding(t.shape[0], global_mean).astype(np.float32)
            ).to(device)
            for t in c_tensors
        ]
        probs_c4m = predict_from_embeddings(model, q_tensors, q_starts, L, c_mean, device).cpu().numpy()
        y_c4m.extend(focal.y)
        p_c4m.extend(probs_c4m.tolist())

    def m(y, p):
        return metrics_to_dict(pooled_metrics(np.array(y, dtype=int), np.array(p, dtype=float)))

    return {
        "C0_true": m(y_c0, p_c0),
        "C4_zero": m(y_c4z, p_c4z),
        "C4_mean": m(y_c4m, p_c4m),
    }


def main():
    variants = ["M0", "M1a", "M1b", "M2"]
    model_types = [
        ("receptor-centric", "RC"),
        ("peptide-centric", "PC"),
    ]
    seeds = list(range(5))

    summary: dict = {}

    print("\n=== Ablation Val Metrics Summary (mean ± sd across seeds) ===\n")
    header = f"{'Variant':<12} {'MT':<4} {'n':<3} {'AUROC':>10} {'PRAUC':>10} {'MCC':>10}"
    print(header)
    print("-" * 55)

    for model_type, short_mt in model_types:
        for var in variants:
            results = load_model_results(var, model_type, seeds)
            stats = summarize_seeds(results)
            key = f"{var}_{short_mt}"
            summary[key] = {"variant": var, "model_type": model_type, **stats}

            if stats:
                auroc_str = f"{stats['auroc_mean']:.4f}±{stats['auroc_sd']:.4f}"
                prauc_str = f"{stats['prauc_mean']:.4f}±{stats['prauc_sd']:.4f}"
                mcc_str   = f"{stats['mcc_mean']:.4f}±{stats['mcc_sd']:.4f}"
                print(f"{var:<12} {short_mt:<4} {stats['n_seeds']:<3} {auroc_str:>10} {prauc_str:>10} {mcc_str:>10}")
            else:
                print(f"{var:<12} {short_mt:<4} {'0':<3} {'no results':>30}")

    # Released checkpoint comparison
    print("\n--- Released checkpoint (val, from audit results) ---")
    for fname, label in [
        ("results/audit_rc_val.json", "Released RC val"),
        ("results/audit_pc_val.json", "Released PC val"),
    ]:
        if Path(fname).exists():
            with open(fname) as f:
                audit = json.load(f)
            c0 = audit["controls"]["C0_true"]
            print(f"{label}: AUROC={c0['auroc']:.4f}, PRAUC={c0['prauc']:.4f}, MCC={c0['mcc']:.4f}")

    # Quick audit on best seed of M0 RC and best seed of M0 PC (if available)
    print("\n=== Quick Partner-Usage Audit on best M0 seeds ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audit_summary: dict = {}

    for model_type, short_mt in model_types:
        results = load_model_results("M0", model_type, seeds)
        if not results:
            print(f"M0_{short_mt}: No results found, skipping audit.")
            continue
        # Find best seed by AUROC
        best_r = max(results, key=lambda r: r["best_val_metrics"]["auroc"])
        best_ckpt = best_r["checkpoint"]
        if not Path(best_ckpt).exists():
            print(f"M0_{short_mt}: Checkpoint {best_ckpt} not found, skipping audit.")
            continue
        print(f"Auditing M0_{short_mt} (best seed {best_r['seed']}, ckpt={best_ckpt})...")
        try:
            audit_res = run_quick_audit(best_ckpt, model_type, "val", device=device)
            for ctrl, m in audit_res.items():
                print(f"  {ctrl}: AUROC={m['auroc']:.4f}, PRAUC={m['prauc']:.4f}, MCC={m['mcc']:.4f}")
            audit_summary[f"M0_{short_mt}_val"] = {
                "seed": best_r["seed"],
                "checkpoint": best_ckpt,
                **audit_res,
            }
        except Exception as e:
            print(f"  Audit failed: {e}")

    # Save outputs
    Path("results/ablation_summary.json").write_text(json.dumps(summary, indent=2))
    Path("results/g1_audit_table.json").write_text(json.dumps(audit_summary, indent=2))
    print("\nSaved results/ablation_summary.json and results/g1_audit_table.json")


if __name__ == "__main__":
    main()
