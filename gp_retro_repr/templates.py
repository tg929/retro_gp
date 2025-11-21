
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Iterable, Tuple

# Optional dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    rdChemReactions = None  # type: ignore

try:
    # rdchiral is the typical tool to apply retro-templates on products
    from rdchiral.initialization import rdchiralReaction, rdchiralReactants
    from rdchiral.main import rdchiralRun
except Exception:  # pragma: no cover
    rdchiralReaction = None  # type: ignore
    rdchiralReactants = None  # type: ignore
    rdchiralRun = None  # type: ignore

@dataclass(frozen=True)
class ReactionTemplate:
    """
    A (retro) reaction template expressed as SMARTS/SMIRKS string.
    Expected format (retro):  [Product SMARTS]>>[Reactant1].[Reactant2]
    """
    template_id: str
    smirks: str
    metadata: Optional[dict] = None

    def apply_to_product(self, product_smiles: str) -> List[List[str]]:
        """
        Apply this retro-template to a product SMILES to generate lists of reactant SMILES.
        Returns a list of reactant-sets (each set = list[str]).
        Requires rdchiral at runtime for robust behavior.
        """
        if rdchiralReaction is None:
            raise RuntimeError("rdchiral not available. Install `rdchiral` to apply retro-templates.")
        rxn = rdchiralReaction(self.smirks)
        prod = rdchiralReactants(product_smiles)
        outputs = rdchiralRun(rxn, prod)
        # rdchiral returns list of strings "r1.r2" possibly repeated
        out_sets: List[List[str]] = []
        for out in outputs:
            parts = [p for p in out.split('.') if p]
            out_sets.append(parts)
        return out_sets

class ReactionTemplateRegistry:
    "Registry of known templates (by id)."
    def __init__(self):
        self._by_id: Dict[str, ReactionTemplate] = {}

    def add(self, tmpl: ReactionTemplate):
        self._by_id[tmpl.template_id] = tmpl

    def get(self, template_id: str) -> ReactionTemplate:
        if template_id not in self._by_id:
            raise KeyError(f"Template id not found: {template_id}")
        return self._by_id[template_id]

    @property
    def templates(self) -> Dict[str, ReactionTemplate]:
        """Expose template mapping for callers that expect a dict-like attribute."""
        return self._by_id

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    @classmethod
    def from_dicts(cls, items: Iterable[dict]) -> "ReactionTemplateRegistry":
        reg = cls()
        for d in items:
            reg.add(ReactionTemplate(template_id=d["id"], smirks=d["smirks"], metadata=d.get("meta")))
        return reg
