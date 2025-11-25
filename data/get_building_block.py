"""
Utility: extract purchasable building blocks matched to HB templates from a
ChemProjector matrix cache and write a SMILES list usable by retrogp.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import shutil


def export_smiles(
    matrix_path: pathlib.Path,
    output_path: pathlib.Path,
    copy_to_building_block: bool = False,
) -> None:
    with open(matrix_path, "rb") as f:
        m = pickle.load(f)

    # Keep molecules that matched at least one reaction template.
    keep_mask = m.matrix.any(axis=1)
    smis = [mol.smiles for mol, keep in zip(m.reactants, keep_mask) if keep]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(smis))
    print(f"Wrote {len(smis)} SMILES to {output_path}")

    if copy_to_building_block:
        dest = pathlib.Path(__file__).resolve().parent / "building_block" / output_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, dest)
        print(f"Copied to {dest} for retrogp data loading")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data1/ytg/retrogp/ChemProjector-main/data/processed/all/matrix_20251007.pkl"
        ),
        help="ChemProjector matrix cache produced by create_reaction_matrix.py",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "/data1/ytg/retrogp/ChemProjector-main/data/processed/all/enamine_hb_matched_20251007.smi"
        ),
        help="Output SMILES file (matched building blocks)",
    )
    parser.add_argument(
        "--copy-to-retrogp",
        action="store_true",
        help="Also copy the output into data/building_block/ for retrogp",
    )
    args = parser.parse_args()

    export_smiles(args.matrix, args.out, args.copy_to_retrogp)


if __name__ == "__main__":
    main()
