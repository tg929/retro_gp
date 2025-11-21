"""Template loading helpers."""
from pathlib import Path
from typing import Iterable, List, Union

from gp_retro_repr import ReactionTemplate, ReactionTemplateRegistry


def _iter_nonempty_lines(path: Union[str, Path]) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield ln


def load_retro_templates(
    path: Union[str, Path],
    prefix: str = "HB",
    reverse_sides: bool = True,
    keep_multi_product: bool = False,
) -> ReactionTemplateRegistry:
    """
    Load templates from SMIRKS lines.
    - reverse_sides: swap product/reactant (for forward dumps -> retro).
    - keep_multi_product: allow multiple products on LHS; default False.
    """
    reg = ReactionTemplateRegistry()
    kept = skipped = 0
    for smirks in _iter_nonempty_lines(path):
        if reverse_sides and ">>" in smirks:
            lhs_raw, rhs_raw = smirks.split(">>", 1)
            smirks_use = f"{rhs_raw}>>{lhs_raw}"
        else:
            smirks_use = smirks
        lhs = smirks_use.split(">>", 1)[0]
        products = lhs.count(".") + 1
        if products > 1 and not keep_multi_product:
            skipped += 1
            continue
        kept += 1
        reg.add(ReactionTemplate(f"{prefix}_{kept:03d}", smirks_use))
    if skipped:
        print(f"[templates] skipped multi-product: {skipped}, kept: {kept}")
    return reg


def template_ids(reg: ReactionTemplateRegistry) -> List[str]:
    return list(reg.templates.keys())
