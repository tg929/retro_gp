# demo_utils.py
import csv
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from gp_retro_repr import (
    Inventory, ReactionTemplateRegistry, ReactionTemplate,
)

from gp_retro_obj import ObjectiveSpec


def _nonempty_lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            yield ln


def load_smiles_list(path: Union[os.PathLike, str]) -> List[str]:
    """Load SMILES from a .smi/.txt file (first token per line)."""
    p = Path(path)
    out: List[str] = []
    for ln in _nonempty_lines(p):
        out.append(ln.split()[0])
    return out


def load_smiles_csv(path: Union[os.PathLike, str], column: str = "smiles") -> List[str]:
    """Load SMILES from a CSV file (default column 'smiles')."""
    p = Path(path)
    out: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(column) or row.get(column.upper())
            if smi:
                smi = smi.strip()
                if smi:
                    out.append(smi)
    return out


def load_inventory_from_files(files: Sequence[str], smiles_column: str = "smiles") -> Inventory:
    """Build an Inventory from a mix of .smi/.txt/.csv files."""
    inv = Inventory()
    for f in files:
        p = Path(f)
        if p.suffix.lower() == ".csv":
            inv.add_many(load_smiles_csv(p, column=smiles_column))
        else:
            inv.add_many(load_smiles_list(p))
    return inv


def load_templates_from_smirks_file(
    path: Union[os.PathLike, str],
    prefix: str = "HB",
    keep_multi_product: bool = False,
    reverse_sides: bool = False,
) -> ReactionTemplateRegistry:
    """Load SMIRKS retro-templates (one per line) and auto-assign IDs.

    Many public template dumps包含多产物左侧（对于 rdchiral 逆合成会报 reactant/template 数不匹配）。
    默认过滤掉左侧含多个片段的模板；若需全部保留，将 keep_multi_product 设为 True。
    """
    reg = ReactionTemplateRegistry()
    kept = 0
    skipped = 0
    for idx, smirks in enumerate(_nonempty_lines(Path(path)), start=1):
        if reverse_sides and ">>" in smirks:
            lhs_raw, rhs_raw = smirks.split(">>", 1)
            smirks_use = f"{rhs_raw}>>{lhs_raw}"
        else:
            smirks_use = smirks
        lhs = smirks_use.split(">>", 1)[0]
        if (lhs.count(".") + 1 > 1) and not keep_multi_product:
            skipped += 1
            continue
        kept += 1
        reg.add(ReactionTemplate(f"{prefix}_{kept:03d}", smirks_use))
    if skipped:
        extra = " (after reversal)" if reverse_sides else ""
        print(f"[load_templates] skipped multi-product templates{extra}: {skipped}, kept: {kept}")
    return reg


def load_targets(path: Union[os.PathLike, str], limit: Union[int, None] = None) -> List[str]:
    targets = load_smiles_list(path)
    return targets if limit is None else targets[:limit]


def build_world_t1():
    stock = Inventory(["CC=O", "O"])
    reg = ReactionTemplateRegistry()
    reg.add(ReactionTemplate("T1", "[C:1]-[O:2]>>[C:1]=O.[O:2]", metadata={"family":"oxidation"}))
    target = "CCO"
    return stock, reg, target


def build_objectives_default():
    return {
        "solved":            ObjectiveSpec("solved", "max", weight=100.0),
        "route_len":         ObjectiveSpec("route_len", "min", weight=1.0),
        "valid_prefix":      ObjectiveSpec("valid_prefix", "max", weight=1.0),
        "sc_partial_reward": ObjectiveSpec("sc_partial_reward", "max", weight=5.0),
        "purch_frac":        ObjectiveSpec("purch_frac", "max", weight=2.0),
        "qed":               ObjectiveSpec("qed", "max", weight=1.0),
    }
