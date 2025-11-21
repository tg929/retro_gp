
from __future__ import annotations
from typing import Iterable, Iterator, Set
from .molecules import canonical_smiles

class Inventory:
    "Purchasable building block inventory."
    def __init__(self, purchasable: Iterable[str] | None = None):
        self._set: Set[str] = set()
        if purchasable:
            self.add_many(purchasable)

    def add(self, smi: str) -> None:
        self._set.add(canonical_smiles(smi))

    def add_many(self, smis: Iterable[str]) -> None:
        for s in smis:
            self.add(s)

    def is_purchasable(self, smi: str) -> bool:
        return canonical_smiles(smi) in self._set

    def __contains__(self, smi: str) -> bool:
        return self.is_purchasable(smi)

    def __iter__(self) -> Iterator[str]:
        return iter(self._set)

    @classmethod
    def from_file(cls, path: str) -> "Inventory":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
        return cls(lines)
