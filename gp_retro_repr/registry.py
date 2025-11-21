
from __future__ import annotations
from typing import Iterable, Dict
from .templates import ReactionTemplate, ReactionTemplateRegistry

def load_templates_from_jsonl(path: str) -> ReactionTemplateRegistry:
    import json
    reg = ReactionTemplateRegistry()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            reg.add(ReactionTemplate(template_id=d["id"], smirks=d["smirks"], metadata=d.get("meta")))
    return reg
