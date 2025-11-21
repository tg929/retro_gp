
"""
gp_retro_feas: Feasibility & Constraints layer for GP-based retrosynthesis.

Depends on Module-1 `gp_retro_repr`:
  - ReactionTemplateRegistry, Inventory
  - Program / Route structures

What this module adds:
  - Functional-group library & template typing (families/requirements)
  - TemplatePreselector: fast, by-construction pruning
  - FeasibilityEngine: rdchiral applicability + inventory gating
  - ActionMaskBuilder: feasible ApplyTemplate actions for a selected product
  - FeasibleExecutor: execute a Program with "choose-valid-reactants" and optional repair
"""
from .fg_patterns import FG_LIBRARY, find_functional_groups
from .typing import TemplateTyping, TemplateType, FamilyRequirement
from .preselect import TemplatePreselector
from .engine import FeasibilityEngine, FeasibilityResult
from .mask import ActionMaskBuilder, ActionMask
from .executor import FeasibleExecutor, ExecutePolicy
