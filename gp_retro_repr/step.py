
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class RetrosynthesisStep:
    "One backward reaction step in sequential format (see ICML'25 LLM-Syn-Planner)."
    molecule_set: List[str]
    rational: str
    product: str
    template_id: str
    reactants: List[str]
    updated_molecule_set: List[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Molecule set": self.molecule_set,
            "Rational": self.rational,
            "Product": [self.product],
            "Reaction": [self.template_id],
            "Reactants": self.reactants,
            "Updated molecule set": self.updated_molecule_set,
            "Diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrosynthesisStep":
        return cls(
            molecule_set=list(d["Molecule set"]),
            rational=str(d.get("Rational", "")),
            product=d["Product"][0],
            template_id=d["Reaction"][0],
            reactants=list(d["Reactants"]),
            updated_molecule_set=list(d["Updated molecule set"]),
            diagnostics=dict(d.get("Diagnostics", {})),
        )
