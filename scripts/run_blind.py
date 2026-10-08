"""
scripts/run_blind.py
Sealed blind evaluator – AGENTS rule 1.

Usage:
  python scripts/run_blind.py --final \
      --model-type receptor-centric \
      --checkpoint data/raw/sppider_seq_receptor.pt \
      --config configs/eval_rc.yaml \
      --run-id G0_RC_released

Rules:
  - Only runs with --final flag.
  - Appends timestamp, git hash, config, metrics to results/blind_log.jsonl.
  - One call per final model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ppis.data import load_split
from ppis.model import load_checkpoint, predict_from_embeddings
from ppis.metrics import pooled_metrics, metrics_to_dict
from ppis.embed import get_or_embed, seq_sha1

import yaml


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", required=True,
                   help="Required flag – confirms this is a sealed blind run.")
    p.add_argument("--model-type", choices=["receptor-centric", "peptide-centric"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--config", required=True, help="Path to eval YAML config.")
    p.add_argument("--run-id", required=True, help="Unique run identifier.")
    p.add_argument("--data-dir", default="SPPIDER-seq-datasets")
    p.add_argument("--emb-dir", default="data/emb")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.final:
        print("ERROR: --final flag is required for blind evaluation.", file=sys.stderr)
        sys.exit(1)

    log_path = Path("results/blind_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check run_id not already used
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("run_id") == args.run_id:
                print(
                    f"ERROR: run_id '{args.run_id}' already exists in blind_log.jsonl. "
                    "Use a new run_id for each final model.",
                    file=sys.stderr,
                )
                sys.exit(1)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device)

    print(f"Loading blind split: {args.model_type}")
    pairs = load_split(args.data_dir, args.model_type, "blind")

    all_y: list[int] = []
    all_p: list[float] = []

    esm = tokenizer = None  # lazy load only if embeddings not cached

    print(f"Running inference on {len(pairs)} pairs...")
    for pair in pairs:
        focal = pair.focal
        partner = pair.partner

        # Try to load embeddings from cache
        focal_sha = seq_sha1(focal.full)
        partner_sha = seq_sha1(partner.full)
        focal_cached = Path(args.emb_dir) / f"{focal_sha}.npy"
        partner_cached = Path(args.emb_dir) / f"{partner_sha}.npy"

        if not focal_cached.exists() or not partner_cached.exists():
            # Lazy-load ESM
            if esm is None:
                from transformers import EsmModel, EsmTokenizer
                from ppis.embed import ESM_MODEL_NAME
                print("Loading ESM-2 (first uncached sequence)...")
                tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
                esm = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
                esm.eval()

        from ppis.embed import get_or_embed, chunk_sequence
        import numpy as np as npa

        _, q_chunks = get_or_embed(focal.full, tokenizer, esm, device, args.emb_dir)
        _, c_chunks = get_or_embed(partner.full, tokenizer, esm, device, args.emb_dir)

        q_tensors = [c["embedding"].float().to(device) for c in q_chunks]
        c_tensors = [c["embedding"].float().to(device) for c in c_chunks]
        q_starts = [c["start_idx"] for c in q_chunks]

        probs = predict_from_embeddings(
            model, q_tensors, q_starts, len(focal.full), c_tensors, device
        ).cpu().numpy()

        all_y.extend(focal.y)
        all_p.extend(probs.tolist())

    y_arr = np.array(all_y, dtype=int)
    p_arr = np.array(all_p, dtype=float)

    m = pooled_metrics(y_arr, p_arr, threshold=args.threshold)
    m_dict = metrics_to_dict(m)

    config = load_config(args.config)

    log_entry = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "config": config,
        "threshold": args.threshold,
        "metrics": m_dict,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    result_path = Path(f"results/{args.run_id}.json")
    result_path.write_text(json.dumps(log_entry, indent=2))

    print(f"\n=== Blind Evaluation: {args.run_id} ===")
    for k, v in m_dict.items():
        print(f"  {k}: {v}")
    print(f"\nLogged to {log_path} and {result_path}")


if __name__ == "__main__":
    main()
