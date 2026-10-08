"""
src/ppis/data.py
T0.1 – Parser for SPPIDER-seq 4-line annotation files.

Each entry is 4 consecutive lines:
  1. >header   (PDB_chain|UniProtID|ChainPair)
  2. full UniProt sequence
  3. PDB-derived sequence (- for unresolved)
  4. binary labels (0/1 per position)

Focal roles:
  rec_* files  -> focal = receptor (first in pair)
  pep_* files  -> focal = peptide  (second in pair)

GPL-3.0 – attribution: dataset format from aporollo-lab/SPPIDER-seq
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProteinEntry:
    """One annotated protein from a 4-line block."""
    name: str           # full header string
    uniprot: str        # UniProt accession
    pair_id: str        # chain-pair field (e.g. "AG=EG")
    pdb_code: str       # 4-char PDB id + chain (e.g. "1a1r_A")
    full: str           # full UniProt sequence
    pdb_aln: str        # PDB-derived sequence (may contain '-')
    observed: list[bool]  # True where pdb_aln[i] != '-'
    y: list[int]        # 0/1 interface labels

    def __post_init__(self) -> None:
        if not (len(self.full) == len(self.pdb_aln) == len(self.y)):
            raise ValueError(
                f"Length mismatch for {self.name}: "
                f"full={len(self.full)}, pdb_aln={len(self.pdb_aln)}, y={len(self.y)}"
            )


@dataclass
class PPIPair:
    """One receptor–peptide pair with focal and partner roles."""
    focal: ProteinEntry
    partner: ProteinEntry
    model_type: str     # "receptor-centric" or "peptide-centric"


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_header(header: str) -> tuple[str, str, str]:
    """
    Parse '>1a1r_A|Q12933|AG=EG' into (pdb_code, uniprot, pair_id).
    Handles multi-pair entries like 'AG=EG=BH' or 'AC=BC'.
    """
    # strip leading >
    h = header.lstrip(">").strip()
    parts = h.split("|")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse header: {header!r}")
    pdb_code = parts[0]
    uniprot = parts[1]
    pair_id = "|".join(parts[2:])  # keep remainder as-is
    return pdb_code, uniprot, pair_id


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------

def iter_entries(txt_path: str | Path) -> Iterator[ProteinEntry]:
    """
    Yield ProteinEntry objects from a *_ppint.pdb2uniprot.txt file.
    Skips blank lines between blocks.
    """
    lines = Path(txt_path).read_text().splitlines()
    # collect non-empty lines
    nonempty = [l for l in lines if l.strip()]

    i = 0
    while i < len(nonempty):
        header_line = nonempty[i]
        if not header_line.startswith(">"):
            i += 1
            continue
        # next 3 lines: full seq, pdb_aln, labels
        if i + 3 >= len(nonempty):
            break
        full = nonempty[i + 1].strip()
        pdb_aln = nonempty[i + 2].strip()
        label_str = nonempty[i + 3].strip()
        i += 4

        pdb_code, uniprot, pair_id = _parse_header(header_line)

        # Build observed and y lists
        if len(full) != len(pdb_aln) or len(full) != len(label_str):
            raise ValueError(
                f"Length mismatch at {header_line!r}: "
                f"full={len(full)}, pdb_aln={len(pdb_aln)}, labels={len(label_str)}"
            )

        observed = [c != "-" for c in pdb_aln]
        y = [int(c) for c in label_str]

        yield ProteinEntry(
            name=header_line.lstrip(">").strip(),
            uniprot=uniprot,
            pair_id=pair_id,
            pdb_code=pdb_code,
            full=full,
            pdb_aln=pdb_aln,
            observed=observed,
            y=y,
        )


def load_entry_dict(txt_path: str | Path) -> dict[str, ProteinEntry]:
    """Return {entry.name: entry} for all entries in a txt file."""
    return {e.name: e for e in iter_entries(txt_path)}


# ---------------------------------------------------------------------------
# TSV pair loader
# ---------------------------------------------------------------------------

def load_pairs(
    tsv_path: str | Path,
    entry_dict: dict[str, ProteinEntry],
    model_type: str,
) -> list[PPIPair]:
    """
    Load pairs from a *_ids.tsv file.

    TSV columns: receptor_id  peptide_id
    model_type: "receptor-centric" -> focal = receptor
                "peptide-centric"  -> focal = peptide
    """
    pairs: list[PPIPair] = []
    missing: list[str] = []

    for line in Path(tsv_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("receptor_id"):
            continue
        cols = line.split("\t")
        if len(cols) != 2:
            continue
        rec_id, pep_id = cols[0].strip(), cols[1].strip()

        rec_entry = entry_dict.get(rec_id)
        pep_entry = entry_dict.get(pep_id)

        if rec_entry is None:
            missing.append(rec_id)
        if pep_entry is None:
            missing.append(pep_id)
        if rec_entry is None or pep_entry is None:
            continue

        if model_type == "receptor-centric":
            focal, partner = rec_entry, pep_entry
        else:
            focal, partner = pep_entry, rec_entry

        pairs.append(PPIPair(focal=focal, partner=partner, model_type=model_type))

    if missing:
        import warnings
        warnings.warn(
            f"{len(set(missing))} unique IDs in TSV not found in txt file. "
            f"First 5: {list(set(missing))[:5]}"
        )

    return pairs


# ---------------------------------------------------------------------------
# Dataset statistics (for Table 1 verification)
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    split: str
    model_type: str
    n_pairs: int
    n_focal_unique: int
    n_residues: int               # over unique focal proteins
    n_residues_by_pairs: int      # over all pairs (focal counted per pair)
    n_interface: int
    interface_pct: float


def compute_stats(pairs: list[PPIPair], split: str, model_type: str) -> SplitStats:
    """Compute Table 1 statistics for a set of pairs."""
    seen_focal: dict[str, ProteinEntry] = {}
    n_res_pairs = 0
    n_iface_pairs = 0

    for p in pairs:
        name = p.focal.name
        if name not in seen_focal:
            seen_focal[name] = p.focal
        n_res_pairs += len(p.focal.full)
        n_iface_pairs += sum(p.focal.y)

    n_res_unique = sum(len(e.full) for e in seen_focal.values())
    n_iface_unique = sum(sum(e.y) for e in seen_focal.values())

    return SplitStats(
        split=split,
        model_type=model_type,
        n_pairs=len(pairs),
        n_focal_unique=len(seen_focal),
        n_residues=n_res_unique,
        n_residues_by_pairs=n_res_pairs,
        n_interface=n_iface_unique,
        interface_pct=100.0 * n_iface_unique / n_res_unique if n_res_unique else 0.0,
    )


# ---------------------------------------------------------------------------
# Convenience loader for a full dataset
# ---------------------------------------------------------------------------

def load_split(
    data_dir: str | Path,
    model_type: str,
    split: str,
) -> list[PPIPair]:
    """
    Load one split (train/val/blind) for a given model type.

    data_dir: directory containing *_ppint.pdb2uniprot.txt and *_*_ids.tsv
    model_type: "receptor-centric" or "peptide-centric"
    split: "train" | "val" | "blind"
    """
    prefix = "rec" if model_type == "receptor-centric" else "pep"
    data_dir = Path(data_dir)

    txt_file = data_dir / f"{prefix}_ppint.pdb2uniprot.txt"
    tsv_file = data_dir / f"{prefix}_{split}_ids.tsv"

    entry_dict = load_entry_dict(txt_file)
    return load_pairs(tsv_file, entry_dict, model_type)
