from __future__ import annotations
from typing import Callable, Dict, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import QED
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

def qed_oracle(smiles: str) -> Optional[float]:
    if not _HAS_RDKIT:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return float(QED.qed(m))
    except Exception:
        return None

class PropertyOracleRegistry(dict):
    """Simple registry mapping name -> callable(smiles)->float|None"""
    def register(self, name: str, fn: Callable[[str], Optional[float]]):
        self[name] = fn

def qed_oracle_available() -> bool:
    return _HAS_RDKIT
