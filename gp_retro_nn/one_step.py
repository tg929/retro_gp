from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class OneStepPrediction:
    reactants: List[str]
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class OneStepRetrosynthesisModel(Protocol):
    name: str

    def predict(self, product_smiles: str, topk: int) -> List[OneStepPrediction]:
        """Return top-k reactant sets for a product SMILES."""
        raise NotImplementedError

