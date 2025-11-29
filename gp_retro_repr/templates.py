
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

    # def apply_to_product(self, product_smiles: str) -> List[List[str]]:
    #     """
    #     Apply this retro-template to a product SMILES to generate lists of reactant SMILES.
    #     Returns a list of reactant-sets (each set = list[str]).
    #     Requires rdchiral at runtime for robust behavior.
    #     """
    #     if rdchiralReaction is None:
    #         raise RuntimeError("rdchiral not available. Install `rdchiral` to apply retro-templates.")
    #     rxn = rdchiralReaction(self.smirks)
    #     prod = rdchiralReactants(product_smiles)
    #     outputs = rdchiralRun(rxn, prod)
    #     # rdchiral returns list of strings "r1.r2" possibly repeated
    #     out_sets: List[List[str]] = []
    #     for out in outputs:
    #         parts = [p for p in out.split('.') if p]
    #         out_sets.append(parts)
    #     return out_sets
    def apply_to_product(self, product_smiles: str) -> List[List[str]]:
        """
        Apply this retro-template to a product SMILES to generate lists of reactant SMILES.
        Returns a list of reactant-sets (each set = list[str]).

        首选使用 rdchiral；如果 rdchiral 不可用，则在 RDKit 可用的情况下退化为
        RDKit 的 ReactionFromSmarts 以便至少在调试/简单模板上能运行。
        """
        # --- 首选：rdchiral 路径 ---
        if rdchiralReaction is not None:
            rxn = rdchiralReaction(self.smirks)
            prod = rdchiralReactants(product_smiles)
            outputs = rdchiralRun(rxn, prod)
            # rdchiral 返回的是类似 "r1.r2" 的字符串列表
            out_sets: List[List[str]] = []
            for out in outputs:
                parts = [p for p in out.split(".") if p]
                if parts:
                    out_sets.append(parts)
            return out_sets

        # --- 兜底：使用 RDKit 反应引擎（尽量不影响原有行为，只在 rdchiral 缺失时启用） ---
        if rdChemReactions is not None and Chem is not None:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(self.smirks, useSmarts=True)
                if rxn is None:
                    raise RuntimeError(f"RDKit failed to parse SMIRKS: {self.smirks}")
                mol = Chem.MolFromSmiles(product_smiles)
                if mol is None:
                    raise RuntimeError(f"Invalid product SMILES for template {self.template_id}: {product_smiles}")
                outputs = rxn.RunReactants((mol,))
                out_sets: List[List[str]] = []
                for prods in outputs:
                    parts: List[str] = []
                    for m in prods:
                        if m is None:
                            continue
                        smi = Chem.MolToSmiles(m, isomericSmiles=True)
                        if smi:
                            parts.append(smi)
                    if parts:
                        out_sets.append(parts)
                return out_sets
            except Exception as e:
                # 统一走到下面的 RuntimeError
                raise RuntimeError(
                    f"Neither rdchiral nor RDKit could apply template {self.template_id} "
                    f"on product {product_smiles}: {e}"
                )

        # 如果既没有 rdchiral 也没有 RDKit，则保持之前的行为：立即报错
        raise RuntimeError(
            "rdchiral not available and RDKit reaction engine unavailable. "
            "Install `rdchiral` (recommended) 或者 RDKit 以应用逆合成模板。"
        )


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
