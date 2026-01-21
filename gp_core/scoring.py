from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gp_retro_nn import OneStepPrediction
from gp_retro_obj import FitnessResult, ObjectiveSpec
from gp_retro_repr import Route, Program, ApplyTemplate, ApplyOneStepModel
from gp_retro_repr.molecules import canonical_smiles

from rdkit import Chem


SaFn = Callable[[str], float]
PriceFn = Callable[[str], float]

# forward_model: (reactants, template_id?, product?) -> (pred_product_smiles, prob)
ForwardModel = Callable[[Sequence[str], Optional[str], Optional[str]], Tuple[Optional[str], float]]

_ION_BRACKET_ONLY_RE = re.compile(r"^\[[^\]]*[\+\-][^\]]*\]$")


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(float(x), eps))


def _logsumexp(xs: Sequence[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _default_sa(smiles: str) -> float:
    return float(len(smiles or ""))


@lru_cache(maxsize=200_000)
def _canonicalize_smiles_cached(smiles: str, strip_atom_maps: bool) -> str:
    """
    RDKit-canonical SMILES with optional atom-map stripping.

    Notes:
    - NAG2G candidates can contain salts/charged fragments; we keep them.
    - Atom-mapping numbers are not useful for inventory matching / GP equality tests.
    """
    smi = (smiles or "").strip()
    if not smi:
        raise ValueError("empty smiles")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if strip_atom_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _safe_canonical(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        return _canonicalize_smiles_cached(smiles, True)
    except Exception:
        try:
            return canonical_smiles(smiles)
        except Exception:
            return None


def _is_valid_smiles(smiles: str) -> bool:
    return _safe_canonical(smiles) is not None


def _is_salt_like_ion(smiles: str) -> bool:
    """
    Heuristic: penalize only *standalone* salt/ion reactants, not charged substructures
    within an organic molecule (e.g. [N-]=[N+]=...).

    Typical examples we want to penalize:
      - [NH4+], [Cl-], [Na+], [K+], [Br-], [I-]
      - small inorganic ions with no carbon and non-zero formal charge
    """
    smi = (smiles or "").strip()
    if not smi:
        return False

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False

    formal_charge = sum(int(a.GetFormalCharge()) for a in mol.GetAtoms())
    if formal_charge == 0:
        return False

    n_heavy = int(mol.GetNumHeavyAtoms())
    n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    # Salt-like: tiny ions OR no-carbon ions.
    return n_heavy <= 2 or n_c == 0


def _count_salt_like_ions(reactants: Sequence[str]) -> int:
    return sum(1 for r in (reactants or []) if _is_salt_like_ion(r))


def _program_length_actions(program: Optional[Program]) -> int:
    if program is None:
        return 0
    n = 0
    for instr in getattr(program, "instructions", []):
        if isinstance(instr, (ApplyTemplate, ApplyOneStepModel)):
            n += 1
    return n


@dataclass(frozen=True)
class StepScoreWeights:
    w_m: float = 1.0
    w_rt: float = 1.0
    w_sa: float = 0.02
    w_bb: float = 0.5
    # Weak-rule penalties/bonuses (do not hard-reject)
    w_nreact: float = 0.25  # penalty per reactant above reactant_soft_limit
    w_progress: float = 0.01  # bonus for (sa(product) - sum(sa(reactants)))
    w_charge: float = 0.05  # penalty per salt-like ion reactant (e.g. [NH4+])


@dataclass(frozen=True)
class StepScoreConfig:
    """Scoring config for re-ranking/truncating one-step candidates."""

    # Candidate pool sizes (topN requested from model, topB kept after re-ranking).
    topN: int = 100
    topB: int = 20

    # How to interpret/normalize candidate.score.
    # - "prob": score is probability in (0,1]
    # - "logprob": score is log(prob)
    # - "rank": use candidate rank (meta["rank"] or list order), normalize by softmax(-rank)
    # - "raw": treat score as unnormalized logit, normalize by log-softmax over the list
    score_type: str = "rank"  # default matches current NAG2G adapter (no score, only rank)
    rank_temperature: float = 2.0  # higher => weaker rank dominance when score_type="rank"

    # Hard filters
    max_smiles_len: int = 300
    max_reactants: Optional[int] = 8
    reject_loops: bool = True  # reject reactant == product (after canonicalization)
    dedup_reactants: bool = True  # remove duplicates within a candidate
    dedup_candidate_sets: bool = True  # remove duplicate sets across candidates

    # Soft rules
    reactant_soft_limit: int = 2

    # If forward_model is enabled but fails/mismatches, apply a large negative log-prob.
    forward_fail_logp: float = -20.0

    weights: StepScoreWeights = field(default_factory=StepScoreWeights)


@dataclass(frozen=True)
class RouteScoreConfig:
    lambda_u: float = 3.0
    lambda_sa: float = 0.05
    lambda_step: float = 0.5
    lambda_bb: float = 1.0


@dataclass(frozen=True)
class FitnessConfig:
    # Scalar must be comparable across states:
    #   solved  >> unsolved >> invalid
    # so that selection/printing is consistent and summary stats (mean) remain interpretable.
    invalid_scalar: float = -1e6
    bloating_eta: float = 0.1

    # Solved routes get a large bonus so they always outrank unsolved routes
    # even when unsolved routes have higher shaping rewards.
    solved_bonus: float = 1e6

    # Packing coefficients for a solved-route key (smaller is better).
    # Keep these on a moderate scale so solved scalars don't dominate stats.
    pack_steps: float = 1e3
    pack_cost: float = 1.0
    pack_max_sa: float = 10.0
    pack_neg_sum_logp_single: float = 1.0
    pack_neg_sum_logp_fwd: float = 1.0
    pack_program_len: float = 0.1


@dataclass(frozen=True)
class ScoringConfig:
    step: StepScoreConfig = field(default_factory=StepScoreConfig)
    route: RouteScoreConfig = field(default_factory=RouteScoreConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)


@dataclass(frozen=True)
class StepCandidate:
    reactants: List[str]
    score_single: float
    log_score_single: float
    step_score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """Generic single-step model candidate (model-agnostic)."""

    reactants: List[str]
    score_single: float
    template_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class StepScorer:
    """Re-rank/truncate one-step candidates for a given product molecule."""

    def __init__(
        self,
        *,
        config: StepScoreConfig,
        bb_is_purchasable: Callable[[str], bool],
        sa_fn: Optional[SaFn] = None,
        forward_model: Optional[ForwardModel] = None,
    ):
        self.cfg = config
        self.bb_is_purchasable = bb_is_purchasable
        self.sa_fn = sa_fn or _default_sa
        self.forward_model = forward_model

    def _log_scores_single(self, preds: Sequence[OneStepPrediction]) -> List[float]:
        st = (self.cfg.score_type or "rank").lower()

        if st == "prob":
            out = []
            for p in preds:
                s = float(p.score) if p.score is not None else 0.0
                out.append(_safe_log(s))
            return out

        if st == "logprob":
            return [float(p.score) if p.score is not None else float("-inf") for p in preds]

        if st == "raw":
            logits = [float(p.score) if p.score is not None else float("-inf") for p in preds]
            z = _logsumexp([x for x in logits if math.isfinite(x)])
            return [x - z if math.isfinite(x) and math.isfinite(z) else float("-inf") for x in logits]

        # default: "rank"
        ranks: List[float] = []
        for i, p in enumerate(preds):
            r = p.meta.get("rank")
            try:
                ranks.append(float(r) if r is not None else float(i))
            except Exception:
                ranks.append(float(i))
        # prob ~ softmax(-rank)
        t = float(self.cfg.rank_temperature) if float(self.cfg.rank_temperature) > 0 else 1.0
        logits = [-(r / t) for r in ranks]
        z = _logsumexp(logits)
        return [x - z for x in logits]

    def _hard_filter(self, product_smiles: str, reactants: Sequence[str]) -> bool:
        if not reactants:
            return False
        if self.cfg.max_reactants is not None and len(reactants) > int(self.cfg.max_reactants):
            return False
        if any((not r) or (len(r) > int(self.cfg.max_smiles_len)) for r in reactants):
            return False
        c_prod = _safe_canonical(product_smiles)
        if c_prod is None:
            return False
        if self.cfg.reject_loops:
            for r in reactants:
                c_r = _safe_canonical(r)
                if c_r is not None and c_r == c_prod:
                    return False
        return all(_is_valid_smiles(r) for r in reactants)

    def _normalize_reactants(self, reactants: Sequence[str]) -> Tuple[List[str], Dict[str, Any]]:
        raw = [str(x).strip() for x in (reactants or []) if str(x).strip()]
        meta: Dict[str, Any] = {"n_reactants_raw": int(len(raw))}

        canon: List[str] = []
        for r in raw:
            c = _safe_canonical(r)
            if c is None:
                return [], {**meta, "reject_reason": "invalid_smiles"}
            canon.append(c)

        if self.cfg.dedup_reactants:
            seen = set()
            uniq: List[str] = []
            for c in canon:
                if c in seen:
                    continue
                seen.add(c)
                uniq.append(c)
            meta["n_reactants_dedup_removed"] = int(len(canon) - len(uniq))
            canon = uniq

        meta["n_reactants"] = int(len(canon))
        meta["n_salt_like_ions"] = int(_count_salt_like_ions(canon))
        return canon, meta

    def step_score(
        self,
        *,
        product_smiles: str,
        reactants: Sequence[str],
        log_score_single: float,
        score_single: float,
        template_id: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        w = self.cfg.weights

        sa_sum = sum(float(self.sa_fn(r)) for r in reactants)
        bb_hits = sum(1 for r in reactants if self.bb_is_purchasable(r))
        try:
            sa_prod = float(self.sa_fn(product_smiles))
        except Exception:
            sa_prod = float(_default_sa(product_smiles))
        progress = float(sa_prod) - float(sa_sum)
        n_excess = max(0, int(len(reactants)) - int(self.cfg.reactant_soft_limit))
        n_salt = int(_count_salt_like_ions(reactants))

        log_p_fwd = 0.0
        fwd_pred_product = None
        if self.forward_model is not None:
            try:
                fwd_pred_product, p_fwd = self.forward_model(reactants, template_id, product_smiles)
                log_p_fwd = _safe_log(float(p_fwd))
                # If forward model provides a product prediction and it disagrees with the target,
                # treat it as a failed round-trip (heavy penalty, but keep the candidate).
                if fwd_pred_product:
                    c_pred = _safe_canonical(str(fwd_pred_product))
                    c_tgt = _safe_canonical(str(product_smiles))
                    if c_pred is not None and c_tgt is not None and c_pred != c_tgt:
                        log_p_fwd = float(self.cfg.forward_fail_logp)
            except Exception:
                log_p_fwd = float(self.cfg.forward_fail_logp)

        score = (
            float(w.w_m) * float(log_score_single)
            + float(w.w_rt) * float(log_p_fwd)
            - float(w.w_sa) * float(sa_sum)
            + float(w.w_bb) * float(bb_hits)
            - float(w.w_nreact) * float(n_excess)
            + float(w.w_progress) * float(progress)
            - float(w.w_charge) * float(n_salt)
        )
        meta = {
            "sa_prod": float(sa_prod),
            "sa_sum": sa_sum,
            "progress": float(progress),
            "bb_hits": bb_hits,
            "n_reactants": int(len(reactants)),
            "n_excess_reactants": int(n_excess),
            "n_salt_like_ions": int(n_salt),
            "log_p_fwd": log_p_fwd,
            "fwd_pred_product": fwd_pred_product,
        }
        return float(score), meta

    def rank_and_truncate(
        self,
        product_smiles: str,
        preds: Sequence[OneStepPrediction],
        *,
        topB: Optional[int] = None,
    ) -> List[OneStepPrediction]:
        preds = list(preds)
        if not preds:
            return []
        product_canon = _safe_canonical(product_smiles)
        if product_canon is None:
            return []
        topB = int(topB if topB is not None else self.cfg.topB)
        topB = max(1, min(topB, len(preds)))

        log_scores = self._log_scores_single(preds)
        candidates: List[Tuple[float, int, OneStepPrediction]] = []

        for i, (p, logp) in enumerate(zip(preds, log_scores)):
            reactants_raw = [str(x) for x in (p.reactants or []) if str(x)]
            reactants, norm_meta = self._normalize_reactants(reactants_raw)
            if not self._hard_filter(product_smiles, reactants):
                continue

            if p.score is not None:
                score_single = float(p.score)
            else:
                # For rank-based models (like current NAG2G adapter), store a comparable score.
                # Higher is better => use -rank.
                rank_val = (p.meta or {}).get("rank", i)
                try:
                    score_single = -float(rank_val)
                except Exception:
                    score_single = -float(i)
            step_s, step_meta = self.step_score(
                product_smiles=product_canon,
                reactants=reactants,
                log_score_single=float(logp),
                score_single=float(score_single),
                template_id=p.meta.get("template_id"),
            )
            new_meta = dict(p.meta or {})
            new_meta.update(norm_meta)
            new_meta.update(
                {
                    "log_score_single": float(logp),
                    "score_single": float(score_single),
                    "step_score": float(step_s),
                    "step_meta": step_meta,
                }
            )
            scored_pred = OneStepPrediction(reactants=reactants, score=p.score, meta=new_meta)
            candidates.append((float(step_s), i, scored_pred))

        # sort by step_score desc; keep deterministic tie-break by original index
        candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        ranked = [p for _, _, p in candidates]
        if not self.cfg.dedup_candidate_sets:
            return ranked[:topB]

        out: List[OneStepPrediction] = []
        seen = set()
        for p in ranked:
            key = tuple(sorted(list(p.reactants or [])))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= topB:
                break
        return out


def get_topB_candidates(
    *,
    product_smiles: str,
    single_step_model: Callable[[str, int], Sequence[Candidate]],
    topN: int = 100,
    topB: int = 20,
    step_config: Optional[StepScoreConfig] = None,
    bb_is_purchasable: Callable[[str], bool],
    sa_fn: Optional[SaFn] = None,
    forward_model: Optional[ForwardModel] = None,
) -> List[StepCandidate]:
    """Model-agnostic helper: query SSM(topN) then re-rank and keep topB with StepScore."""
    step_config = step_config or StepScoreConfig(topN=int(topN), topB=int(topB))
    preds_raw = list(single_step_model(product_smiles, int(topN)) or [])

    preds: List[OneStepPrediction] = []
    for i, c in enumerate(preds_raw):
        reactants = [str(x) for x in (c.reactants or []) if str(x)]
        meta = dict(getattr(c, "meta", {}) or {})
        meta.setdefault("rank", i)
        if getattr(c, "template_id", None):
            meta["template_id"] = getattr(c, "template_id")
        preds.append(OneStepPrediction(reactants=reactants, score=float(c.score_single), meta=meta))

    scorer = StepScorer(config=step_config, bb_is_purchasable=bb_is_purchasable, sa_fn=sa_fn, forward_model=forward_model)
    ranked = scorer.rank_and_truncate(product_smiles, preds, topB=int(topB))

    out: List[StepCandidate] = []
    for p in ranked:
        meta = dict(p.meta or {})
        out.append(
            StepCandidate(
                reactants=list(p.reactants),
                score_single=float(meta.get("score_single", p.score if p.score is not None else 0.0)),
                log_score_single=float(meta.get("log_score_single", float("-inf"))),
                step_score=float(meta.get("step_score", float("-inf"))),
                meta=meta,
            )
        )
    return out


@dataclass(frozen=True)
class RouteState:
    target_smiles: str
    steps_so_far: int
    leaves: List[str]
    solved: bool
    solved_leaves: List[str]
    unsolved_leaves: List[str]
    total_cost: float = 0.0
    max_sa: float = 0.0
    sum_logp_single: float = 0.0
    sum_logp_fwd: float = 0.0


def route_state_from_route(
    *,
    route: Route,
    target_smiles: str,
    bb_is_purchasable: Callable[[str], bool],
    sa_fn: Optional[SaFn] = None,
    price_fn: Optional[PriceFn] = None,
) -> RouteState:
    sa_fn = sa_fn or _default_sa
    price_fn = price_fn or (lambda _s: 0.0)

    if route.steps:
        leaves = list(route.steps[-1].updated_molecule_set)
    else:
        leaves = [target_smiles]

    solved_leaves = [m for m in leaves if bb_is_purchasable(m)]
    unsolved_leaves = [m for m in leaves if not bb_is_purchasable(m)]
    solved = len(unsolved_leaves) == 0

    molecules_for_sa: List[str] = list(leaves)
    sum_logp_single = 0.0
    sum_logp_fwd = 0.0
    for st in route.steps:
        molecules_for_sa.append(getattr(st, "product", "") or "")
        molecules_for_sa.extend(list(getattr(st, "reactants", []) or []))
        diag = dict(getattr(st, "diagnostics", {}) or {})
        v = diag.get("one_step_log_score_single")
        if v is not None:
            try:
                sum_logp_single += float(v)
            except Exception:
                pass
        v2 = diag.get("one_step_log_p_fwd")
        if v2 is not None:
            try:
                sum_logp_fwd += float(v2)
            except Exception:
                pass

    max_sa = 0.0
    for smi in molecules_for_sa:
        if not smi:
            continue
        try:
            max_sa = max(max_sa, float(sa_fn(smi)))
        except Exception:
            max_sa = max(max_sa, float(_default_sa(smi)))

    total_cost = 0.0
    if solved:
        for smi in leaves:
            try:
                total_cost += float(price_fn(smi))
            except Exception:
                total_cost += 0.0

    return RouteState(
        target_smiles=target_smiles,
        steps_so_far=len(route.steps),
        leaves=leaves,
        solved=solved,
        solved_leaves=solved_leaves,
        unsolved_leaves=unsolved_leaves,
        total_cost=float(total_cost),
        max_sa=float(max_sa),
        sum_logp_single=float(sum_logp_single),
        sum_logp_fwd=float(sum_logp_fwd),
    )


def route_score_unsolved(state: RouteState, cfg: RouteScoreConfig, sa_fn: Optional[SaFn] = None) -> float:
    sa_fn = sa_fn or _default_sa
    U = list(state.unsolved_leaves)
    S = list(state.solved_leaves)

    sa_sum = 0.0
    for m in U:
        try:
            sa_sum += float(sa_fn(m))
        except Exception:
            sa_sum += float(_default_sa(m))

    return (
        -float(cfg.lambda_u) * float(len(U))
        -float(cfg.lambda_sa) * float(sa_sum)
        -float(cfg.lambda_step) * float(state.steps_so_far)
        +float(cfg.lambda_bb) * float(len(S))
    )


def route_key_solved(state: RouteState) -> Tuple[float, float, float, float, float]:
    # Minimize lexicographically
    return (
        float(state.steps_so_far),
        float(state.total_cost),
        float(state.max_sa),
        float(-state.sum_logp_single),
        float(-state.sum_logp_fwd),
    )


class GPFitnessEvaluator:
    """Fitness evaluator implementing StepScore/RouteScore/Fitness as described in the prompt."""

    def __init__(
        self,
        *,
        target_smiles: str,
        bb_is_purchasable: Callable[[str], bool],
        config: Optional[ScoringConfig] = None,
        sa_fn: Optional[SaFn] = None,
        price_fn: Optional[PriceFn] = None,
    ):
        self.target_smiles = target_smiles
        self.bb_is_purchasable = bb_is_purchasable
        self.cfg = config or ScoringConfig()
        self.sa_fn = sa_fn or _default_sa
        self.price_fn = price_fn

        # Expose objective specs for NSGA-II (directions matter).
        # Keep it small and aligned with the designed scoring.
        self.specs: Dict[str, ObjectiveSpec] = {
            "solved": ObjectiveSpec("solved", "max", weight=100.0),
            "route_score": ObjectiveSpec("route_score", "max", weight=1.0),
            "steps": ObjectiveSpec("steps", "min", weight=1.0),
            "total_cost": ObjectiveSpec("total_cost", "min", weight=1.0),
            "max_sa": ObjectiveSpec("max_sa", "min", weight=1.0),
            "sum_logp_single": ObjectiveSpec("sum_logp_single", "max", weight=1.0),
            "sum_logp_fwd": ObjectiveSpec("sum_logp_fwd", "max", weight=1.0),
            "program_len": ObjectiveSpec("program_len", "min", weight=0.1),
            "n_unsolved": ObjectiveSpec("n_unsolved", "min", weight=3.0),
        }

    def evaluate(self, route: Route, *, program: Optional[Program] = None, invalid: bool = False) -> FitnessResult:
        program_len = _program_length_actions(program)

        if invalid:
            return FitnessResult(
                objectives={
                    "solved": 0.0,
                    "route_score": float(self.cfg.fitness.invalid_scalar),
                    "steps": float("inf"),
                    "total_cost": float("inf"),
                    "max_sa": float("inf"),
                    "sum_logp_single": float("-inf"),
                    "sum_logp_fwd": float("-inf"),
                    "program_len": float(program_len),
                    "n_unsolved": float("inf"),
                },
                scalar=float(self.cfg.fitness.invalid_scalar),
                extra={"invalid": True, "program_len": program_len},
            )

        state = route_state_from_route(
            route=route,
            target_smiles=self.target_smiles,
            bb_is_purchasable=self.bb_is_purchasable,
            sa_fn=self.sa_fn,
            price_fn=self.price_fn,
        )

        if state.solved:
            key = route_key_solved(state)
            packed = (
                key[0] * float(self.cfg.fitness.pack_steps)
                + key[1] * float(self.cfg.fitness.pack_cost)
                + key[2] * float(self.cfg.fitness.pack_max_sa)
                + key[3] * float(self.cfg.fitness.pack_neg_sum_logp_single)
                + key[4] * float(self.cfg.fitness.pack_neg_sum_logp_fwd)
                + float(program_len) * float(self.cfg.fitness.pack_program_len)
            )
            scalar = float(self.cfg.fitness.solved_bonus) - float(packed)  # larger is better
            objectives = {
                "solved": 1.0,
                "route_score": 0.0,
                "steps": float(state.steps_so_far),
                "total_cost": float(state.total_cost),
                "max_sa": float(state.max_sa),
                "sum_logp_single": float(state.sum_logp_single),
                "sum_logp_fwd": float(state.sum_logp_fwd),
                "program_len": float(program_len),
                "n_unsolved": 0.0,
            }
            extra = {"route_key_solved": key, "program_len": program_len, "state": state}
            return FitnessResult(objectives=objectives, scalar=scalar, extra=extra)

        # unsolved: dense shaping reward + bloating penalty
        rs = route_score_unsolved(state, self.cfg.route, sa_fn=self.sa_fn)
        scalar = float(rs) - float(self.cfg.fitness.bloating_eta) * float(program_len)
        objectives = {
            "solved": 0.0,
            "route_score": float(rs),
            "steps": float(state.steps_so_far),
            "total_cost": 0.0,
            "max_sa": float(state.max_sa),
            "sum_logp_single": float(state.sum_logp_single),
            "sum_logp_fwd": float(state.sum_logp_fwd),
            "program_len": float(program_len),
            "n_unsolved": float(len(state.unsolved_leaves)),
        }
        extra = {"route_score_unsolved": rs, "program_len": program_len, "state": state}
        return FitnessResult(objectives=objectives, scalar=scalar, extra=extra)
