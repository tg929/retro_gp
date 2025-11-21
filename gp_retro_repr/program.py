
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

from .templates import ReactionTemplateRegistry
from .route import Route
from .step import RetrosynthesisStep
from .validation import RouteChecks

@dataclass(frozen=True)
class Instruction:
    pass

@dataclass(frozen=True)
class Select(Instruction):
    "Select a product index from the current molecule set."
    index: int

@dataclass(frozen=True)
class ApplyTemplate(Instruction):
    "Apply a template by id to the last selected product."
    template_id: str
    rational: str = ""

@dataclass(frozen=True)
class Stop(Instruction):
    pass

@dataclass
class ExecutionConfig:
    template_registry: ReactionTemplateRegistry
    inventory: Any  # expects .is_purchasable(smiles)

class Program:
    "A linear Decision Program: [Select, ApplyTemplate] repeated; Stop to finish."
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions

    def execute(self, target_smiles: str, config: ExecutionConfig) -> Route:
        route = Route()
        molecule_set: List[str] = [target_smiles]
        last_selected: Optional[int] = None

        it = iter(self.instructions)
        for instr in it:
            if isinstance(instr, Stop):
                break
            if isinstance(instr, Select):
                last_selected = instr.index
                # bounds check deferred until Apply
                continue
            if isinstance(instr, ApplyTemplate):
                if last_selected is None:
                    raise ValueError("ApplyTemplate encountered before Select.")
                if last_selected < 0 or last_selected >= len(molecule_set):
                    raise IndexError(f"Select index {last_selected} out of range for molecule_set={molecule_set}")

                product = molecule_set[last_selected]
                # apply retro-template to product
                outs = config.template_registry.get(instr.template_id).apply_to_product(product)
                if not outs:
                    raise RuntimeError(f"Template {instr.template_id} produced no reactant sets for product {product}")

                # naive choice: take the first reactant set (search/sampling comes later modules)
                reactants = outs[0]
                updated = [m for i, m in enumerate(molecule_set) if i != last_selected] + reactants

                step = RetrosynthesisStep(
                    molecule_set=molecule_set.copy(),
                    rational=instr.rational,
                    product=product,
                    template_id=instr.template_id,
                    reactants=reactants,
                    updated_molecule_set=updated,
                    diagnostics={},
                )

                # Connectivity check and append
                if route.steps:
                    assert RouteChecks.connectivity_ok(route.steps[-1].updated_molecule_set, step.molecule_set)
                route.append(step)

                # advance state
                molecule_set = updated
                last_selected = None
                continue
            raise TypeError(f"Unknown instruction: {instr}")

        return route
