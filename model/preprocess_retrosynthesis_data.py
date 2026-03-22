import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem

from decoder.tokenizer import BasicSmilesTokenizer


REACTION_FIELD = "reactants>reagents>production"
DEFAULT_SPLITS = ("train", "eval", "test")


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def tokenize_smiles(tokenizer, smiles):
    return tokenizer.tokenize(smiles)


def build_output_row(tokenizer, row_idx, row):
    raw_reaction = row[REACTION_FIELD]
    raw_precursor, raw_product = raw_reaction.split(">>", 1)

    precursor_raw = raw_precursor.replace(" ", "").strip()
    product_raw = raw_product.replace(" ", "").strip()

    product_canonical = canonicalize_smiles(product_raw)
    if product_canonical is None:
        return None, "invalid_product"

    precursor_components = [part for part in precursor_raw.split(".") if part]
    precursor_components_canonical = []
    for component in precursor_components:
        component_canonical = canonicalize_smiles(component)
        if component_canonical is None:
            return None, "invalid_precursor"
        precursor_components_canonical.append(component_canonical)

    precursor_components_canonical.sort()
    precursor_canonical_sorted = ".".join(precursor_components_canonical)

    product_tokens = tokenize_smiles(tokenizer, product_canonical)
    precursor_tokens = tokenize_smiles(tokenizer, precursor_canonical_sorted)

    output_row = {
        "sample_id": row_idx,
        "class": row.get("class", ""),
        "product_raw": product_raw,
        "product_canonical": product_canonical,
        "product_tokens": " ".join(product_tokens),
        "precursor_raw": precursor_raw,
        "precursor_canonical_sorted": precursor_canonical_sorted,
        "precursor_components_canonical": json.dumps(precursor_components_canonical, ensure_ascii=False),
        "precursor_tokens": " ".join(precursor_tokens),
        REACTION_FIELD: f"{' '.join(precursor_tokens)}>>{' '.join(product_tokens)}",
    }
    return output_row, None


def process_split(tokenizer, input_path, output_path, limit=None):
    fieldnames = [
        "sample_id",
        "class",
        "product_raw",
        "product_canonical",
        "product_tokens",
        "precursor_raw",
        "precursor_canonical_sorted",
        "precursor_components_canonical",
        "precursor_tokens",
        REACTION_FIELD,
    ]
    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_rows": 0,
        "invalid_product_rows": 0,
        "invalid_precursor_rows": 0,
        "max_product_tokens": 0,
        "max_precursor_tokens": 0,
    }

    with open(input_path, encoding="utf-8") as src, open(output_path, "w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(reader):
            if limit is not None and summary["total_rows"] >= limit:
                break
            summary["total_rows"] += 1
            output_row, error = build_output_row(tokenizer, row_idx, row)
            if error is not None:
                summary["dropped_rows"] += 1
                summary[f"{error}_rows"] += 1
                continue

            summary["kept_rows"] += 1
            summary["max_product_tokens"] = max(
                summary["max_product_tokens"],
                len(output_row["product_tokens"].split()),
            )
            summary["max_precursor_tokens"] = max(
                summary["max_precursor_tokens"],
                len(output_row["precursor_tokens"].split()),
            )
            writer.writerow(output_row)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("model/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = BasicSmilesTokenizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for split in args.splits:
        input_path = args.input_dir / f"{split}.csv"
        output_path = args.output_dir / f"{split}.csv"
        summary_path = args.output_dir / f"{split}_summary.json"
        summary = process_split(tokenizer, input_path, output_path, limit=args.limit)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        all_summaries[split] = summary

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
