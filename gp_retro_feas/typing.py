
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Iterable

@dataclass(frozen=True)
class TemplateType:
    "Template family/type label (e.g., 'oxidation', 'esterification', 'reduction', 'SN2',...)."
    name: str

@dataclass(frozen=True)
class FamilyRequirement:
    "Requirements for a template family to be considered applicable (FG prerequisites)."
    family: TemplateType
    requires_fg: Set[str]

class TemplateTyping:
    """
    Holds mappings:
      - template_id -> family/type
      - family -> required functional groups on product
    If a template has no entry, we consider no FG precondition.
    """
    def __init__(self):
        self._tmpl2fam: Dict[str, TemplateType] = {}
        self._fam2req: Dict[str, FamilyRequirement] = {}

    def set_template_family(self, template_id: str, family_name: str):
        self._tmpl2fam[template_id] = TemplateType(family_name)

    def set_family_requirement(self, family_name: str, requires_fg: Iterable[str]):
        self._fam2req[family_name] = FamilyRequirement(
            family=TemplateType(family_name),
            requires_fg=set(requires_fg)
        )

    def get_family(self, template_id: str) -> Optional[TemplateType]:
        return self._tmpl2fam.get(template_id)

    def family_requires(self, family_name: str) -> Optional[FamilyRequirement]:
        return self._fam2req.get(family_name)

    def compatible_templates(self, template_ids: Iterable[str], required_fgs: Set[str]) -> List[str]:
        """
        Filter template ids whose family requirement is subset of required_fgs.
        If template has no family or family has no requirement, keep it.
        """
        out = []
        for tid in template_ids:
            fam = self.get_family(tid)
            if fam is None:
                out.append(tid)
                continue
            req = self.family_requires(fam.name)
            if req is None or req.requires_fg.issubset(required_fgs):
                out.append(tid)
        return out
