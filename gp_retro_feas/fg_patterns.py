
from __future__ import annotations
from typing import Dict, List, Set
try:
    from rdkit import Chem
except Exception:
    Chem = None  # type: ignore

# Minimal functional-group SMARTS (extend as needed)
FG_LIBRARY: Dict[str, str] = {
    "alcohol": "[CX4;H1,H2,H3][OX2H]",           # R-CH2/CH-OH
    "phenol": "c[OX2H]",                          # Ar-OH
    "aldehyde": "[CX3H1](=O)[#6]",                # R-CHO
    "ketone": "[#6][CX3](=O)[#6]",                # R-CO-R
    "carboxylic_acid": "C(=O)[OX2H1]",           # -COOH
    "ester": "C(=O)O[#6]",                        # -COOR
    "ether": "[#6]-O-[#6]",                       # -O-
    "amine_primary": "[NX3H2][#6]",
    "amine_secondary": "[NX3H1]([#6])[#6]",
    "amide": "C(=O)N",
    "acid_halide": "C(=O)[Cl,Br,I,F]",
    "alkyl_halide": "[CX4][Cl,Br,I,F]",
    "aryl_halide": "c[Cl,Br,I]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "nitrile": "C#N",
    "boronic_acid": "B(O)O",
    "nitro": "[NX3](=O)=O",
}

def find_functional_groups(smiles: str) -> Set[str]:
    "Return set of FG labels detected in the molecule."
    if Chem is None:
        # Fallback: cannot parse; return empty set
        return set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    found: Set[str] = set()
    for name, smarts in FG_LIBRARY.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            found.add(name)
    return found
