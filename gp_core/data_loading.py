"""Data loading: inventory, templates, targets."""
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Union, Optional

from gp_retro_repr import Inventory
from .templates import load_retro_templates
from . import config


def _nonempty_lines(path: Union[str, Path]) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            yield ln


def load_smiles_list(path: Union[str, Path]) -> List[str]:
    return [ln.split()[0] for ln in _nonempty_lines(path)]


def load_smiles_csv(path: Union[str, Path], column: str = "smiles") -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get(column) or row.get(column.upper()) or "").strip()
            if smi:
                out.append(smi)
    return out


def load_inventory_from_files(files: Sequence[Union[str, Path]], smiles_column: str = "smiles") -> Inventory:
    inv = Inventory()
    for f in files:
        p = Path(f)
        if p.suffix.lower() == ".csv":
            inv.add_many(load_smiles_csv(p, column=smiles_column))
        else:
            inv.add_many(load_smiles_list(p))
    return inv


def load_targets(path: Union[str, Path], limit: Optional[int] = None) -> List[str]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        t = load_smiles_csv(p)
    else:
        t = load_smiles_list(p)
    return t if limit is None else t[:limit]


def load_inventory_and_templates(
    files: Optional[Sequence[Union[str, Path]]] = None,
    templates_path: Optional[Union[str, Path]] = None,
) -> Inventory:
    """Load purchasable inventory and reaction templates.

    If `files` is provided, only those building block files (relative to
    data_root / \"building_block\") are used. Otherwise, all .smi/.txt/.csv
    under the building_block folder are included.
    """
    root = config.data_root
    bb_root = root / "building_block"

    if files:
        inv_files = []
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = bb_root / p
            if not p.exists():
                # Special-case fallback for EvoRRP naming mismatch
                if p.name == "building_block_dataset.txt":
                    alt = p.with_name("building_blocks_dataset.txt")
                    if alt.exists():
                        p = alt
                    else:
                        raise FileNotFoundError(f"Inventory file not found: {p}")
                else:
                    raise FileNotFoundError(f"Inventory file not found: {p}")
            inv_files.append(p)
    else:
        inv_files = sorted(
            p for p in bb_root.glob("*")
            if p.suffix.lower() in {".smi", ".txt", ".csv"}
        )

    if not inv_files:
        raise FileNotFoundError(f"No inventory files found in {bb_root}")

    templates_path = Path(templates_path) if templates_path else (root / "reaction_template" / "hb.txt")
    if not templates_path.is_absolute():
        templates_path = root / templates_path
    if not templates_path.exists():
        raise FileNotFoundError(f"Template file not found: {templates_path}")

    inventory = load_inventory_from_files(inv_files)
    reg = load_retro_templates(
        templates_path,
        prefix="HB",
        reverse_sides=True,
        keep_multi_product=False,
    )
    return inventory, reg
