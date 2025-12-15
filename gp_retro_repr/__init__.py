
"""
gp_retro_repr: Problem representation for GP-based multi-step retrosynthesis (Decision Program style).

Modules:
  - molecules: Molecule wrapper and helpers
  - stock: purchasable building block inventory
  - templates: reaction templates & registry
  - step: single retrosynthesis step structure
  - route: multi-step route structure + (de)serialization
  - program: typed instruction set + interpreter that executes a program to produce a Route
  - validation: three-level checks (molecule / reaction / route)
  - registry: registries for templates and stocks
"""
from .molecules import Molecule, canonical_smiles, molecule_from_smiles
from .stock import Inventory
from .templates import ReactionTemplate, ReactionTemplateRegistry
from .step import RetrosynthesisStep
from .route import Route
from .program import Instruction, Select, ApplyTemplate, ApplyOneStepModel, Stop, Program, ExecutionConfig
from .validation import MoleculeChecks, ReactionChecks, RouteChecks
