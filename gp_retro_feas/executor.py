
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

from gp_retro_repr import Program, Route, Select, ApplyTemplate, ApplyOneStepModel, Stop, RetrosynthesisStep
from gp_retro_repr import ReactionTemplateRegistry, Inventory
from .mask import ActionMaskBuilder
from .engine import FeasibilityEngine

try:  # optional dependency
    from gp_retro_nn import OneStepRetrosynthesisModel
except Exception:  # pragma: no cover
    OneStepRetrosynthesisModel = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from gp_retro_nn import OneStepPrediction

@dataclass(frozen=True)
class ExecutePolicy:
    """
    How to execute ApplyTemplate when multiple reactant sets exist / when a template fails.
    """
    require_all_purchasable: bool = False
    max_reactants: Optional[int] = None
    repair_on_failure: bool = True  # if selected template fails, try other feasible templates
    stop_when_all_purchasable: bool = True  # early-stop route once all molecules are purchasable
    one_step_topk: int = 10  # how many candidates to request from one-step model
    one_step_repair: bool = True  # if selected rank fails, try other ranks

class FeasibleExecutor:
    """
    Execute a Program under feasibility constraints:
      - Only accept ApplyTemplate that passes applicability (and optionally inventory gating)
      - If the chosen template fails, optionally repair by choosing another feasible template
    """
    def __init__(
        self,
        reg: ReactionTemplateRegistry,
        inventory: Optional[Inventory] = None,
        policy: Optional[ExecutePolicy] = None,
        one_step_model: Optional["OneStepRetrosynthesisModel"] = None,
        one_step_ranker: Optional[Callable[[str, List["OneStepPrediction"]], List["OneStepPrediction"]]] = None,
    ):
        self.reg = reg
        self.inventory = inventory
        # Default policy must exist even if caller passes None (bug fix).
        self.policy = policy or ExecutePolicy()
        self.mask_builder = ActionMaskBuilder(reg, inventory=inventory)
        self.engine = FeasibilityEngine(reg, inventory=inventory)
        self.one_step_model = one_step_model
        self.one_step_ranker = one_step_ranker

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

                # Early stopping: if at any point all molecules are purchasable
                # we treat the route as complete and ignore remaining instructions.
                if (
                    self.policy.stop_when_all_purchasable
                    and self.inventory is not None
                    and molecule_set
                    and all(self.inventory.is_purchasable(m) for m in molecule_set)
                ):
                    break
                continue

            if isinstance(instr, ApplyOneStepModel):
                if self.one_step_model is None:
                    raise RuntimeError("ApplyOneStepModel encountered but no one_step_model was provided to FeasibleExecutor")
                if last_selected is None or last_selected < 0 or last_selected >= len(molecule_set):
                    raise IndexError("Select index invalid before ApplyOneStepModel")

                product = molecule_set[last_selected]
                preds = self.one_step_model.predict(product, topk=int(self.policy.one_step_topk))
                if not preds:
                    raise RuntimeError(f"One-step model produced no candidates for product={product}")
                if self.one_step_ranker is not None:
                    preds = list(self.one_step_ranker(product, list(preds)))
                    if not preds:
                        raise RuntimeError(f"One-step ranker rejected all candidates for product={product}")

                # Choose requested rank; optionally repair by searching other ranks
                rank0 = max(0, int(instr.rank))
                rank_order = [rank0] + [i for i in range(len(preds)) if i != rank0]
                if not self.policy.one_step_repair:
                    rank_order = [rank0]

                chosen = None
                chosen_rank = None
                chosen_reason = ""
                for r in rank_order:
                    if r < 0 or r >= len(preds):
                        continue
                    cand = preds[r]
                    reactants = list(cand.reactants or [])
                    reactants = [s for s in reactants if s]
                    if not reactants:
                        chosen_reason = "empty_reactants"
                        continue
                    if self.policy.max_reactants is not None and len(reactants) > int(self.policy.max_reactants):
                        chosen_reason = "exceed_max_reactants"
                        continue
                    if self.policy.require_all_purchasable and self.inventory is not None:
                        if not all(self.inventory.is_purchasable(s) for s in reactants):
                            chosen_reason = "inventory_violation"
                            continue
                    chosen = cand
                    chosen_rank = r
                    chosen_reason = "ok"
                    break

                if chosen is None:
                    raise RuntimeError(
                        f"One-step model candidates rejected for product={product}, "
                        f"requested_rank={rank0}, reason={chosen_reason}"
                    )

                updated = [m for i, m in enumerate(molecule_set) if i != last_selected] + list(chosen.reactants)
                model_name = getattr(self.one_step_model, "name", "one_step")
                original_rank = (getattr(chosen, "meta", {}) or {}).get("rank", chosen_rank)
                step = RetrosynthesisStep(
                    molecule_set=molecule_set.copy(),
                    rational=instr.rational,
                    product=product,
                    template_id=f"{model_name}@rank={original_rank}",
                    reactants=list(chosen.reactants),
                    updated_molecule_set=updated,
                    diagnostics={
                        "executor_reason": chosen_reason,
                        "one_step_score": chosen.score,
                        "one_step_meta": dict(getattr(chosen, "meta", {}) or {}),
                        "one_step_step_score": (getattr(chosen, "meta", {}) or {}).get("step_score"),
                        "one_step_log_score_single": (getattr(chosen, "meta", {}) or {}).get("log_score_single"),
                        "one_step_log_p_fwd": (getattr(chosen, "meta", {}) or {}).get("step_meta", {}).get("log_p_fwd"),
                        "one_step_requested_rank": rank0,
                        "one_step_chosen_rank": chosen_rank,
                        "one_step_original_rank": original_rank,
                        "one_step_n_candidates": len(preds),
                    },
                )
                route.append(step)
                molecule_set = updated
                last_selected = None

                if (
                    self.policy.stop_when_all_purchasable
                    and self.inventory is not None
                    and molecule_set
                    and all(self.inventory.is_purchasable(m) for m in molecule_set)
                ):
                    break
                continue

            raise TypeError(f"Unknown instruction: {instr}")

        return route
