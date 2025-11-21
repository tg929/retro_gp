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
    t = load_smiles_list(path)
    return t if limit is None else t[:limit]


def load_world_from_data(limit_targets: Optional[int] = None):
    root = config.DATA_ROOT
    inv_files = [
        root / "building_block" / "test_building_blocks.smi",
        root / "building_block" / "test_chembl.csv",
        root / "building_block" / "enamine_smiles_1k.csv",
    ]
    templates_path = root / "reaction_template" / "hb.txt"
    targets_path = root / "target molecular" / "chembl_small.txt"

    inventory = load_inventory_from_files(inv_files)
    reg = load_retro_templates(
        templates_path,
        prefix="HB",
        reverse_sides=True,
        keep_multi_product=False,
    )
    targets = load_targets(targets_path, limit=limit_targets)
    return inventory, reg, targets
