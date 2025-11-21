
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional, Set, Dict, Tuple
from functools import lru_cache

from gp_retro_repr import ReactionTemplateRegistry
from .fg_patterns import find_functional_groups
from .typing import TemplateTyping

@dataclass
class PreselectResult:
    product_smiles: str
    product_fgs: Set[str]
    candidate_template_ids: List[str]
    reasoning: Dict[str, str]

class TemplatePreselector:
    """
    Lightweight "by-construction" pruning before calling rdchiral:
     - FG gating (product must have required FG for the template's family)
     - (optional) length/reactant-count heuristic could be added here later
    """
    def __init__(self, reg: ReactionTemplateRegistry, typing: Optional[TemplateTyping] = None):
        self.reg = reg
        self.typing = typing or TemplateTyping()

    @lru_cache(maxsize=10000)
    def preselect(self, product_smiles: str) -> PreselectResult:
        all_tids = list(self.reg._by_id.keys())  # internal access by design; registry is small
        fgs = find_functional_groups(product_smiles)
        cand = self.typing.compatible_templates(all_tids, fgs)
        reason = {
            "total_templates": str(len(all_tids)),
            "fg_detected": ",".join(sorted(fgs)),
            "after_fg_gating": str(len(cand))
        }
        return PreselectResult(
            product_smiles=product_smiles,
            product_fgs=fgs,
            candidate_template_ids=cand,
            reasoning=reason
        )
