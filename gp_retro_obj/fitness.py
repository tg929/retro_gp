from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any, Iterable, Tuple
import math
import statistics

try:
    # Optional: RDKit for QED/SA helpers (not required).
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False

# ---- Light adapters to gp_retro_feas ---------------------------------------

def _try_import_gp_modules():
    feas = None
    reprm = None
    try:
        import gp_retro_feas as feas  # type: ignore
    except Exception:
        feas = None
    try:
        import gp_retro_repr as reprm  # type: ignore
    except Exception:
        reprm = None
    return feas, reprm

# ---- Types -----------------------------------------------------------------

@dataclass
class FitnessResult:
    objectives: Dict[str, float]
    scalar: float
    extra: Dict[str, Any]

class Scalarizer:
    """Common scalarization recipes for multi-objective vectors."""

    @staticmethod
    def weighted_sum(obj: Dict[str, float],
                     specs: Dict[str, 'ObjectiveSpec']) -> float:
        s = 0.0
        for k, v in obj.items():
            spec = specs.get(k)
            if spec is None:
                continue
            # Convert to "maximize" sense by flipping sign if needed
            d = spec.direction()
            s += spec.weight * (d * v)
        return s

    @staticmethod
    def tchebycheff(obj: Dict[str, float],
                    specs: Dict[str, 'ObjectiveSpec'],
                    ref_point: Optional[Dict[str, float]] = None,
                    p: float = math.inf) -> float:
        """Tchebycheff scalarization. If ref_point not provided, build from specs.target.
        For maximize objectives, we *maximize* distance improvement; implement as
        negative distance to convert into a 'higher is better' scalar.
        """
        if ref_point is None:
            ref_point = {k: (spec.target if spec and spec.target is not None else 0.0)
                         for k, spec in specs.items()}
        # distance across objectives
        distances: List[float] = []
        for k, spec in specs.items():
            if k not in obj:
                continue
            d = spec.direction()
            rp = ref_point.get(k, 0.0)
            # We want higher better after sign flip; compute |(d*v) - (d*rp)|
            distances.append(abs(d * obj[k] - d * rp))
        if not distances:
            return 0.0
        if p == math.inf:
            dist = max(distances)
        else:
            dist = (sum(x**p for x in distances) / len(distances)) ** (1.0 / p)
        return -dist  # negative distance => larger is better

# ---- Property oracle helpers -----------------------------------------------

def _fallback_scscore(smiles: str) -> float:
    """A very rough SCScore proxy if real SCScore model is unavailable.

    SCScore (Coley et al., 2018) is learned from reactions. As a fallback,
    we use a crude proxy: heavier and more hetero-rich molecules are considered
    harder to synthesize.

    This function will be replaced if user wires a real scscore() callable.
    """
    if not _HAS_RDKIT:
        # Use a dumb proxy: length- and character-based complexity
        return min(5.0, 0.5 + 0.02 * len(smiles))
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 5.0
    n_heavy = m.GetNumHeavyAtoms()
    n_hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (6, 1))
    return min(5.0, 0.5 + 0.03 * n_heavy + 0.1 * math.tanh(0.1 * n_hetero))

def _qed_or_none(smiles: str) -> Optional[float]:
    if not _HAS_RDKIT:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return float(QED.qed(m))
    except Exception:
        return None


def _heavy_atom_count(smiles: str) -> int:
    """Return heavy atom count if RDKit is available; otherwise use a crude proxy."""
    if not _HAS_RDKIT:
        # Rough proxy: 1 heavy atom per 3 characters (avoid zero)
        return max(1, len(smiles) // 3)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    try:
        return int(m.GetNumHeavyAtoms())
    except Exception:
        return 0


def _ring_count(smiles: str) -> int:
    """Return ring count using RDKit; 0 if unavailable or invalid."""
    if not _HAS_RDKIT:
        return 0
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    try:
        return int(m.GetRingInfo().NumRings())
    except Exception:
        return 0

# ---- RouteFitnessEvaluator --------------------------------------------------

class RouteFitnessEvaluator:
    """Compute objectives + scalar fitness for a single route.

    This class expects the *feasibility* module to provide two callables, which
    you already implemented earlier:
        - `audit_route(route) -> dict` or `evaluate_route(route) -> dict`
          returning keys:
              is_solved: bool
              first_invalid_molecule_set: List[str]   # SMILES at the first invalid step
              current_molecule_set: List[str]         # Molecules at first invalid (or final solved)
              n_steps: int                            # total steps in the route
              n_valid_steps: int                      # prefix of valid steps
        - `is_purchasable(smiles) -> bool`

    If your feasibility module uses different names, pass them explicitly via the
    constructor arguments.
    """

    def __init__(self,
                 objective_specs: Dict[str, 'ObjectiveSpec'],
                 purchasable_fn: Optional[Callable[[str], bool]] = None,
                 audit_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
                 scscore_fn: Optional[Callable[[str], float]] = None,
                 target_smiles: Optional[str] = None,
                 property_oracles: Optional[Dict[str, Callable[[str], Optional[float]]]] = None,
                 llm_style_scalar: bool = False,
                 partial_reward_fn: Optional[Callable[[Iterable[str], Callable[[str], bool], Callable[[str], float]], float]] = None):
        self.specs = objective_specs
        feas, reprm = _try_import_gp_modules()

        # Wire defaults from gp_retro_feas when not explicitly passed
        if purchasable_fn is None and feas is not None:
            purchasable_fn = getattr(feas, "is_purchasable", None)
        if audit_fn is None and feas is not None:
            audit_fn = getattr(feas, "audit_route", None) or getattr(feas, "evaluate_route", None)

        self.purchasable_fn = purchasable_fn or (lambda s: False)
        self.audit_fn = audit_fn or (lambda route: {
            'is_solved': False,
            'first_invalid_molecule_set': [],
            'current_molecule_set': [],
            'n_steps': len(route) if isinstance(route, (list, tuple)) else 0,
            'n_valid_steps': 0,
        })

        self.scscore_fn = scscore_fn or _fallback_scscore
        self.target_smiles = target_smiles
        self.llm_style_scalar = llm_style_scalar
        # Optional override for partial reward (e.g., multi-score implementation)
        self.partial_reward_fn = partial_reward_fn

        # default oracles: QED on the target
        self.property_oracles = property_oracles or {}
        if "qed" not in self.property_oracles:
            self.property_oracles["qed"] = _qed_or_none

    # Core: SCScore partial reward from the first invalid step
    def _sc_partial_reward(self, molset: Iterable[str]) -> float:
        penalty = 0.0
        for smi in molset:
            try:
                if not self.purchasable_fn(smi):
                    penalty += float(self.scscore_fn(smi))
            except Exception:
                penalty += float(self.scscore_fn(smi))
        return -penalty

    def _fragment_score(self, molset: Iterable[str]) -> float:
        """Penalize tiny fragments in the frontier (soft constraint).

        For each molecule, if heavy-atom count < n_min, accumulate the gap.
        Return negative penalty so that higher is better.
        """
        n_min = 6  # target minimum heavy atoms; tweak as needed
        penalty = 0.0
        for smi in molset:
            n = _heavy_atom_count(smi)
            if n <= 0:
                continue
            gap = max(0, n_min - n)
            penalty += float(gap)
        return -penalty

    def _step_smoothness(self, route: Any) -> float:
        """Step-level penalty inspired by AutoSynRoute.

        For each step, measure ring-count change and SMILES-length change between
        product and reactants. Large discontinuities get penalized.
        Returns negative penalty so that higher is better.
        """
        steps = getattr(route, "steps", None)
        if not steps:
            return 0.0
        c_rings = 1.0
        c_size = 0.05
        penalty = 0.0
        for st in steps:
            try:
                product = getattr(st, "product", None)
                reactants = getattr(st, "reactants", None)
                if not product or not reactants:
                    continue
                input_smi = str(product)
                output_smi = ".".join(reactants)
                dr = abs(_ring_count(output_smi) - _ring_count(input_smi))
                ds = abs(len(output_smi) - len(input_smi))
                penalty += c_rings * float(dr) + c_size * float(ds)
            except Exception:
                continue
        return -penalty

    def evaluate(self, route: Any) -> FitnessResult:
        report = self.audit_fn(route)  # must return expected keys

        is_solved = bool(report.get('is_solved', False))
        n_steps = int(report.get('n_steps', 0))
        n_valid = int(report.get('n_valid_steps', 0))

        if is_solved:
            molset_at_invalid = []
            current_set = report.get('current_molecule_set', [])
        else:
            molset_at_invalid = list(report.get('first_invalid_molecule_set', []))
            current_set = list(report.get('current_molecule_set', molset_at_invalid))

        # 1) Partial reward (SCScore) at the first invalid step
        if is_solved:
            sc_partial = 0.0
        else:
            try:
                if self.partial_reward_fn is not None:
                    sc_partial = float(self.partial_reward_fn(current_set, self.purchasable_fn, self.scscore_fn))
                else:
                    sc_partial = self._sc_partial_reward(current_set)
            except Exception:
                sc_partial = self._sc_partial_reward(current_set)

        # 2) Fraction purchasable in the current set (progress proxy)
        purch_flags: List[int] = []
        for smi in current_set:
            try:
                purch_flags.append(1 if self.purchasable_fn(smi) else 0)
            except Exception:
                purch_flags.append(0)
        purch_frac = sum(purch_flags) / max(1, len(purch_flags))

        # 3) Fragmentation-related scores on the frontier (current_set)
        fragment_score = self._fragment_score(current_set)
        n_components = float(len(current_set))

        # 4) Step-level smoothness: penalize drastic per-step changes
        step_smoothness = self._step_smoothness(route)

        # 5) QED or other property oracles on target molecule (if provided)
        prop_vals: Dict[str, Optional[float]] = {}
        tsmi = self.target_smiles
        for name, oracle in self.property_oracles.items():
            try:
                prop_vals[name] = oracle(tsmi) if tsmi else None
            except Exception:
                prop_vals[name] = None

        # Build objective vector (direction defined by specs)
        objs: Dict[str, float] = {}
        # Hard success indicator
        objs['solved'] = 1.0 if is_solved else 0.0
        # Fewer steps usually better
        objs['route_len'] = float(n_steps)
        # Valid prefix length (the larger the better)
        objs['valid_prefix'] = float(n_valid)
        # SCScore-based partial reward (already negative value <= 0)
        objs['sc_partial_reward'] = float(sc_partial)
        # More purchasable in current set => better
        objs['purch_frac'] = float(purch_frac)
        # Penalize tiny fragments and many frontier components
        objs['fragment_score'] = float(fragment_score)
        objs['n_components'] = float(n_components)
        # Penalize drastic per-step changes (higher is smoother)
        objs['step_smoothness'] = float(step_smoothness)

        # Add property oracles (maximize by default)
        for k, v in prop_vals.items():
            if v is not None:
                objs[k] = float(v)

        # Scalarize
        if self.llm_style_scalar:
            # LLM-Syn style: final score is negative (sum + mean) of SCScore over non-purchasable set
            # and zero if solved. Keeps objectives for diagnostics.
            if is_solved:
                scalar = 0.0
            else:
                sc_scores = []
                for smi in current_set:
                    if self.purchasable_fn(smi):
                        continue
                    try:
                        sc_scores.append(float(self.scscore_fn(smi)))
                    except Exception:
                        sc_scores.append(5.0)
                if sc_scores:
                    s_sum = sum(sc_scores)
                    s_mean = sum(sc_scores) / len(sc_scores)
                    scalar = -(s_sum + s_mean)
                else:
                    scalar = 0.0
        else:
            scalar = Scalarizer.weighted_sum(objs, self.specs)

        extra = {
            'is_solved': is_solved,
            'current_set': current_set,
            'molset_at_first_invalid': molset_at_invalid,
            'properties': prop_vals,
        }
        return FitnessResult(objectives=objs, scalar=scalar, extra=extra)
