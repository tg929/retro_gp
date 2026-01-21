
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from rdkit import Chem

@dataclass
class MoleculeChecks:
    "Molecule-level checks (validity + availability)."
    @staticmethod
    def is_valid(smiles: str) -> bool:
        return Chem.MolFromSmiles(smiles) is not None

    @staticmethod
    def availability_flags(smiles_list: List[str], inventory) -> Tuple[bool, List[int]]:
        unavailable = [i for i, smi in enumerate(smiles_list) if not inventory.is_purchasable(smi)]
        return (len(unavailable) == 0, unavailable)

@dataclass
class ReactionChecks:
    "Reaction-level checks (existence + applicability)."
    @staticmethod
    def exists(template_id: str, template_registry) -> bool:
        try:
            template_registry.get(template_id)
            return True
        except KeyError:
            return False

    @staticmethod
    def applicable(template_id: str, product_smiles: str, template_registry) -> bool:
        # Try applying the retro-template to see if at least one reactant set is produced
        try:
            tmpl = template_registry.get(template_id)
        except KeyError:
            return False
        try:
            outs = tmpl.apply_to_product(product_smiles)
            return len(outs) > 0
        except Exception:
            return False

@dataclass
class RouteChecks:
    "Route-level checks (connectivity)."
    @staticmethod
    def connectivity_ok(prev_updated_set: List[str], current_molecule_set: List[str]) -> bool:
        return list(prev_updated_set) == list(current_molecule_set)

    @staticmethod
    def evaluate_step(step_dict: Dict[str, Any], template_registry, inventory) -> Tuple[bool, Dict[str, Any]]:
        # Returns (step_valid_without_availability, diagnostics_dict)
        mol_set = list(step_dict["Molecule set"])
        product = step_dict["Product"][0]
        template_id = step_dict["Reaction"][0]
        reactants = list(step_dict["Reactants"])
        updated_set = list(step_dict["Updated molecule set"])

        # Molecule validity (syntactic)
        mol_valid = all(MoleculeChecks.is_valid(s) for s in mol_set + [product] + reactants + updated_set)

        # Reaction existence + applicability
        rxn_exists = ReactionChecks.exists(template_id, template_registry)
        rxn_valid = ReactionChecks.applicable(template_id, product, template_registry) if rxn_exists else False

        # Route connectivity (product must be in mol_set, and updated set must equal mol_set - {product} + reactants)
        connectivity = (product in mol_set) and (sorted(updated_set) == sorted([s for s in mol_set if s != product] + reactants))

        # Availability on updated set
        avail_ok, unavailable_ids = MoleculeChecks.availability_flags(updated_set, inventory)

        ok_without_avail = mol_valid and rxn_exists and rxn_valid and connectivity
        diag = dict(
            molecule_valid=mol_valid,
            reaction_exists=rxn_exists,
            reaction_valid=rxn_valid,
            connectivity=connectivity,
            check_availability=avail_ok,
            unavailable_mol_id=unavailable_ids,
        )
        return ok_without_avail, diag
