
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem

def canonical_smiles(smiles: str) -> str:
    "Return RDKit-canonical SMILES."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)

def molecule_from_smiles(smiles: str):
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
        return Chem.MolFromSmiles(self.smiles)

    def __str__(self):
        return self.smiles
