
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable

from gp_retro_repr import ReactionTemplateRegistry, Inventory
from .preselect import TemplatePreselector
from .engine import FeasibilityEngine, FeasibilityResult

@dataclass
class ActionMask:
    product_smiles: str
    candidate_templates: List[str]
    feasible_templates: List[str]
    reasons: Dict[str, str]  # e.g., {"tid": "ok|no_reactant_set|inventory_violation|..."}

class ActionMaskBuilder:
    """
    Build a mask of feasible ApplyTemplate actions for a given product.
    Steps:
      1) FG/type preselect to reduce candidates
      2) rdchiral applicability check (and optional inventory gating)
    """
    def __init__(self, reg: ReactionTemplateRegistry, inventory: Optional[Inventory] = None, preselector: Optional[TemplatePreselector] = None):
        self.reg = reg
        self.preselector = preselector or TemplatePreselector(reg)
        self.engine = FeasibilityEngine(reg, inventory=inventory)

    def build(self, product_smiles: str, require_all_purchasable: bool = False, max_reactants: Optional[int] = None) -> ActionMask:
        pre = self.preselector.preselect(product_smiles)
        feasible: List[str] = []
        reasons: Dict[str, str] = {}
        for tid in pre.candidate_template_ids:
            res = self.engine.check_and_choose(tid, product_smiles, require_all_purchasable=require_all_purchasable, max_reactants=max_reactants)
            reasons[tid] = res.reason
            if res.ok:
                feasible.append(tid)
        return ActionMask(product_smiles=product_smiles,
                          candidate_templates=pre.candidate_template_ids,
                          feasible_templates=feasible,
                          reasons=reasons)
