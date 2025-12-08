"""
Download reaction SMILES from a HuggingFace dataset (e.g., pingzhili/uspto-50k)
and export a retro-style template list usable by run_real_data_gp (--templates).

Usage:
    python scripts/fetch_uspto_templates.py \
        --dataset pingzhili/uspto-50k \
        --split train \
        --limit 50000 \
        --output data/reaction_template/uspto50k_retro.txt

Notes:
- This exports per-reaction SMILES as retro SMIRKS: <product_smiles>>><reactants_smiles>.
  It is NOT a generalized template extraction; for true templates you need rdchiral
  template extraction on atom-mapped reactions.
- Requires the `datasets` package and network access to HuggingFace Hub.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def parse_rxn_smiles(rxn: str) -> str:
    """Convert USPTO-style rxn_smiles (reactants>reagents>products) to retro SMIRKS."""
    parts = rxn.split(">")
    if len(parts) == 3:
        reactants, _, products = parts
    elif len(parts) == 2:
        reactants, products = parts
    else:
        # fallback: cannot parse, return raw
        return rxn.strip()
    reactants = reactants.strip()
    products = products.strip()
    if not reactants or not products:
        return rxn.strip()
    return f"{products}>>{reactants}"


def iter_rxn_smiles(dataset, rxn_column: str) -> Iterable[str]:
    for row in dataset:
        rxn = row.get(rxn_column, "")
        if not rxn:
            continue
        yield parse_rxn_smiles(str(rxn))


def main():
    parser = argparse.ArgumentParser(description="Fetch USPTO reaction SMILES from HuggingFace datasets.")
    parser.add_argument("--dataset", type=str, default="pingzhili/uspto-50k", help="HF dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load")
    parser.add_argument(
        "--rxn-column",
        type=str,
        default=None,
        help="Column containing reaction SMILES (auto-detect rxn_smiles/reaction_smiles if None)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reaction_template/uspto50k_retro.txt"),
        help="Output file for retro SMIRKS list",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(f"[ERROR] Please install `datasets`: pip install datasets  ({e})")

    print(f"Loading dataset {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)

    rxn_col = args.rxn_column
    if rxn_col is None:
        for cand in ("rxn_smiles", "reaction_smiles", "rxn"):
            if cand in ds.column_names:
                rxn_col = cand
                break
    if rxn_col is None:
        raise SystemExit(f"Could not find a reaction SMILES column in dataset columns: {ds.column_names}")

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()
    with out_path.open("w", encoding="utf-8") as f:
        for smirks in iter_rxn_smiles(ds, rxn_col):
            if not smirks:
                continue
            if smirks in seen:
                continue
            seen.add(smirks)
            f.write(smirks + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} retro SMIRKS to {out_path}")


if __name__ == "__main__":
    main()
