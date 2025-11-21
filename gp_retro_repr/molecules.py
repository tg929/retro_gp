
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

def canonical_smiles(smiles: str) -> str:
    "Return RDKit-canonical SMILES if RDKit is available; otherwise trimmed input."
    if Chem is None:
        return smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)

def molecule_from_smiles(smiles: str):
    if Chem is None:
        raise RuntimeError("RDKit not available. Please install RDKit to use molecule features.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol

@dataclass(frozen=True)
class Molecule:
    smiles: str
    def __post_init__(self):
        object.__setattr__(self, "smiles", canonical_smiles(self.smiles))

    @property
    def rdkit(self):
        if Chem is None:
            raise RuntimeError("RDKit not available.")
        return Chem.MolFromSmiles(self.smiles)

    def __str__(self):
        return self.smiles
