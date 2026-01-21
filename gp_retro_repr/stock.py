from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set

from .molecules import canonical_smiles

from rdkit import Chem


@dataclass(frozen=True)
class ChemHistory:
    """Minimal chemical history record (ASKCOS-compatible fields)."""

    as_reactant: int = 0
    as_product: int = 0


class ChemHistorian:
    """
    Minimal historian for chemical popularity stopping criteria, inspired by ASKCOS.

    It stores counts for canonical SMILES:
      - as_reactant
      - as_product
    """

    def __init__(self):
        self._hist: Dict[str, ChemHistory] = {}

    def get(self, smiles: str) -> ChemHistory:
        try:
            key = canonical_smiles(smiles)
        except Exception:
            return ChemHistory()
        return self._hist.get(key, ChemHistory())

    def load_from_file(self, path: str | Path) -> None:
        """
        Load chemical history from JSON/JSON.GZ.

        Accepted formats:
          - list[{"smiles": "...", "as_reactant": int, "as_product": int}, ...]
          - dict[smiles] -> {"as_reactant": int, "as_product": int}
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ChemHistorian file not found: {p}")

        if p.suffix == ".gz":
            with gzip.open(p, "rb") as f:
                raw = f.read().decode("utf-8")
        else:
            raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)

        if isinstance(data, dict):
            records = list(data.items())
        elif isinstance(data, list):
            records = [(d.get("smiles"), d) for d in data if isinstance(d, dict)]
        else:
            raise ValueError(f"Unsupported historian format in {p}: {type(data).__name__}")

        hist: Dict[str, ChemHistory] = {}
        for smi, entry in records:
            if not smi:
                continue
            try:
                key = canonical_smiles(str(smi))
            except Exception:
                continue
            if isinstance(entry, dict):
                ar = int(entry.get("as_reactant", 0) or 0)
                ap = int(entry.get("as_product", 0) or 0)
            else:
                ar = ap = 0
            hist[key] = ChemHistory(as_reactant=ar, as_product=ap)
        self._hist = hist


@dataclass(frozen=True)
class LeafCriteriaConfig:
    """
    Leaf/stop criteria config aligned with ASKCOS tree_builder.py.

    - max_natom_dict: {"logic": "none|or|and", "C": int, "N": int, "O": int, "H": int}
    - min_chemical_history_dict: {"logic": "none|or", "as_reactant": int, "as_product": int}
    """

    max_natom_dict: Dict[str, Any]
    min_chemical_history_dict: Dict[str, Any]

    @staticmethod
    def make_max_natom_dict(*, logic: str = "none", C: int = 0, N: int = 0, O: int = 0, H: int = 0) -> Dict[str, Any]:
        default_val = int(1e9) if str(logic).lower() == "and" else 0
        d: Dict[str, Any] = defaultdict(lambda: default_val)  # mirrors ASKCOS
        d.update({"logic": str(logic), "C": int(C), "N": int(N), "O": int(O), "H": int(H)})
        return d

    @staticmethod
    def make_min_history_dict(
        *, logic: str = "none", as_reactant: int = 5, as_product: int = 5
    ) -> Dict[str, Any]:
        return {"logic": str(logic), "as_reactant": int(as_reactant), "as_product": int(as_product)}


class LeafCriteria:
    """
    Leaf evaluator mimicking ASKCOS stop criteria (buyable / small_enough / popular_enough).

    Usage: call `is_leaf(smiles, buyable=inventory.is_purchasable(smiles))`.
    """

    def __init__(self, *, cfg: LeafCriteriaConfig, chemhistorian: Optional[ChemHistorian] = None):
        self.cfg = cfg
        self.chemhistorian = chemhistorian
        self._leaf_fn = self._build_leaf_fn()

    def _is_small_enough(self, smiles: str) -> bool:
        natom_dict = defaultdict(lambda: 0)
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        for a in mol.GetAtoms():
            natom_dict[a.GetSymbol()] += 1
        natom_dict["H"] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
        max_natom_dict = self.cfg.max_natom_dict
        return all(natom_dict[k] <= v for (k, v) in max_natom_dict.items() if k != "logic")

    def _is_popular_enough(self, hist: ChemHistory) -> bool:
        cfg = self.cfg.min_chemical_history_dict
        return hist.as_reactant >= int(cfg["as_reactant"]) or hist.as_product >= int(cfg["as_product"])

    def _build_leaf_fn(self):
        # Ported from ASKCOS/makeit/retrosynthetic/tree_builder.py
        max_natom_dict = self.cfg.max_natom_dict
        min_hist_dict = self.cfg.min_chemical_history_dict

        min_hist_logic = str(min_hist_dict.get("logic", "none")).lower()
        max_natom_logic = str(max_natom_dict.get("logic", "none")).lower()

        if min_hist_logic in [None, "none"]:
            if max_natom_logic in [None, "none"]:

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return bool(buyable)

            elif max_natom_logic == "or":

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return bool(buyable) or self._is_small_enough(smiles)

            else:

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return bool(buyable) and self._is_small_enough(smiles)

        else:
            if max_natom_logic in [None, "none"]:

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return bool(buyable) or self._is_popular_enough(hist)

            elif max_natom_logic == "or":

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return bool(buyable) or self._is_popular_enough(hist) or self._is_small_enough(smiles)

            else:

                def is_a_leaf_node(smiles: str, buyable: bool, hist: ChemHistory) -> bool:
                    return self._is_popular_enough(hist) or (bool(buyable) and self._is_small_enough(smiles))

        return is_a_leaf_node

    def is_leaf(self, smiles: str, *, buyable: bool) -> bool:
        hist = self.chemhistorian.get(smiles) if self.chemhistorian is not None else ChemHistory()
        return bool(self._leaf_fn(smiles, bool(buyable), hist))


class Inventory:
    "Purchasable building block inventory."

    def __init__(self, purchasable: Iterable[str] | None = None, *, leaf_criteria: Optional[LeafCriteria] = None):
        self._set: Set[str] = set()
        self._leaf_criteria = leaf_criteria
        if purchasable:
            self.add_many(purchasable)

    def add(self, smi: str) -> None:
        try:
            self._set.add(canonical_smiles(smi))
        except Exception:
            # Skip invalid SMILES in inventory files; they cannot be purchasable anyway.
            return

    def add_many(self, smis: Iterable[str]) -> None:
        for s in smis:
            self.add(s)

    def is_purchasable(self, smi: str) -> bool:
        return canonical_smiles(smi) in self._set

    def is_leaf(self, smi: str) -> bool:
        """
        Leaf/stop criterion aligned with ASKCOS:
          - default: same as is_purchasable
          - if leaf_criteria is configured: buyable OR small_enough OR popular_enough (depending on logic)
        """
        buyable = self.is_purchasable(smi)
        if self._leaf_criteria is None:
            return bool(buyable)
        return bool(self._leaf_criteria.is_leaf(smi, buyable=bool(buyable)))

    def set_leaf_criteria(self, leaf_criteria: Optional[LeafCriteria]) -> None:
        self._leaf_criteria = leaf_criteria

    def __contains__(self, smi: str) -> bool:
        return self.is_purchasable(smi)

    def __iter__(self) -> Iterator[str]:
        return iter(self._set)

    @classmethod
    def from_file(cls, path: str) -> "Inventory":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return cls(lines)
