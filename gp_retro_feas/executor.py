
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from gp_retro_repr import Program, Route, Select, ApplyTemplate, Stop, RetrosynthesisStep
from gp_retro_repr import ReactionTemplateRegistry, Inventory
from .mask import ActionMaskBuilder
from .engine import FeasibilityEngine

@dataclass(frozen=True)
class ExecutePolicy:
    """
    How to execute ApplyTemplate when multiple reactant sets exist / when a template fails.
    """
    require_all_purchasable: bool = False
    max_reactants: Optional[int] = None
    repair_on_failure: bool = True  # if selected template fails, try other feasible templates

class FeasibleExecutor:
    """
    Execute a Program under feasibility constraints:
      - Only accept ApplyTemplate that passes applicability (and optionally inventory gating)
      - If the chosen template fails, optionally repair by choosing another feasible template
    """
    def __init__(self, reg: ReactionTemplateRegistry, inventory: Optional[Inventory] = None, policy: Optional[ExecutePolicy] = None):
        self.reg = reg
        self.inventory = inventory
        # Default policy must exist even if caller passes None (bug fix).
        self.policy = policy or ExecutePolicy()
        self.mask_builder = ActionMaskBuilder(reg, inventory=inventory)
        self.engine = FeasibilityEngine(reg, inventory=inventory)

    def execute(self, program: Program, target_smiles: str) -> Route:
        route = Route()
        molecule_set: List[str] = [target_smiles]
        last_selected: Optional[int] = None

        for instr in program.instructions:
            if isinstance(instr, Stop):
                break
            if isinstance(instr, Select):
                last_selected = instr.index
                continue
            if isinstance(instr, ApplyTemplate):
                if last_selected is None or last_selected < 0 or last_selected >= len(molecule_set):
                    raise IndexError("Select index invalid before ApplyTemplate")
                product = molecule_set[last_selected]

                # If the intended template fails, optionally repair by picking a feasible alternative
                res = self.engine.check_and_choose(
                    instr.template_id, product,
                    require_all_purchasable=self.policy.require_all_purchasable,
                    max_reactants=self.policy.max_reactants
                )
                chosen_tid = instr.template_id
                if not res.ok and self.policy.repair_on_failure:
                    mask = self.mask_builder.build(product,
                                                   require_all_purchasable=self.policy.require_all_purchasable,
                                                   max_reactants=self.policy.max_reactants)
                    # choose first feasible alternative
                    for alt_tid in mask.feasible_templates:
                        if alt_tid == instr.template_id:
                            continue
                        res_alt = self.engine.check_and_choose(
                            alt_tid, product,
                            require_all_purchasable=self.policy.require_all_purchasable,
                            max_reactants=self.policy.max_reactants
                        )
                        if res_alt.ok:
                            res = res_alt
                            chosen_tid = alt_tid
                            break

                if not res.ok:
                    raise RuntimeError(f"ApplyTemplate failed for product={product}, template={instr.template_id}, reason={res.reason}")

                reactants = res.chosen_reactants
                updated = [m for i, m in enumerate(molecule_set) if i != last_selected] + reactants

                step = RetrosynthesisStep(
                    molecule_set=molecule_set.copy(),
                    rational=instr.rational,
                    product=product,
                    template_id=chosen_tid,
                    reactants=reactants,
                    updated_molecule_set=updated,
                    diagnostics={"executor_reason": res.reason}
                )
                route.append(step)
                molecule_set = updated
                last_selected = None
                continue

            raise TypeError(f"Unknown instruction: {instr}")

        return route
