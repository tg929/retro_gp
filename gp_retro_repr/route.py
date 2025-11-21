
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterable
import json

from .step import RetrosynthesisStep

@dataclass
class Route:
    "A multi-step route represented as a linear list of RetrosynthesisStep."
    steps: List[RetrosynthesisStep] = field(default_factory=list)

    def append(self, step: RetrosynthesisStep):
        # Connectivity invariant: step.molecule_set == previous.updated_molecule_set
        if self.steps:
            prev = self.steps[-1].updated_molecule_set
            if list(step.molecule_set) != list(prev):
                raise ValueError(f"Connectivity mismatch: {step.molecule_set} != {prev}")
        self.steps.append(step)

    @property
    def molecule_set(self) -> List[str]:
        return self.steps[-1].updated_molecule_set if self.steps else []

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_list_of_dicts(), indent=indent)

    @classmethod
    def from_list_of_dicts(cls, items: Iterable[Dict[str, Any]]) -> "Route":
        steps = [RetrosynthesisStep.from_dict(d) for d in items]
        r = cls()
        for s in steps:
            r.append(s)
        return r

    def is_solved(self, inventory) -> bool:
        "Solved if every molecule in final set is purchasable."
        final = self.molecule_set
        return all(inventory.is_purchasable(m) for m in final)
