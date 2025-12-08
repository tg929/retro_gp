"""SCScore loading and fitness evaluator helpers."""
import sys
import pathlib
from typing import Callable, Dict, Optional
import warnings

import numpy as np
from gp_retro_obj import ObjectiveSpec, RouteFitnessEvaluator
from . import config

# Optional: multi-score SCScore wrapper (cached partial reward)
_HAS_MULTISCORE = False
try:
    _repo_root = pathlib.Path(__file__).resolve().parents[1]
    # Ensure vendored packages (scscore) are importable for multi-score loader
    sys.path.append(str(_repo_root))
    _ms_dir = _repo_root / "multi-score"
    if _ms_dir.exists():
        sys.path.append(str(_ms_dir))
        from scscore_reward import (  # type: ignore
            make_scscore as _ms_make_scscore,
            sc_partial_reward as _ms_sc_partial_reward,
            SCScoreCache as _MSCache,
        )
        _HAS_MULTISCORE = True
except Exception:
    _HAS_MULTISCORE = False

# SCScore (standalone numpy) loader
try:
    from scscore.scscore.standalone_model_numpy import SCScorer
    _HAS_SCSCORE = True
except ImportError:
    # Try to use the vendored scscore in the repo root if available
    _HAS_SCSCORE = False
    try:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        vendored = repo_root / "scscore"
        if (vendored / "scscore").exists():
            # Need parent of top-level package on sys.path
            sys.path.append(str(repo_root))
            from scscore.scscore.standalone_model_numpy import SCScorer  # type: ignore
            _HAS_SCSCORE = True
        else:
            _HAS_SCSCORE = False
    except Exception:
        _HAS_SCSCORE = False


def build_objectives(weights: Dict[str, float]) -> Dict[str, ObjectiveSpec]:
    return {
        "solved": ObjectiveSpec("solved", "max", weight=weights.get("solved", 100.0)),
        "route_len": ObjectiveSpec("route_len", "min", weight=weights.get("route_len", 1.0)),
        "valid_prefix": ObjectiveSpec("valid_prefix", "max", weight=weights.get("valid_prefix", 1.0)),
        "sc_partial_reward": ObjectiveSpec("sc_partial_reward", "max", weight=weights.get("sc_partial_reward", 1.0)),
        "purch_frac": ObjectiveSpec("purch_frac", "max", weight=weights.get("purch_frac", 1.0)),
        "qed": ObjectiveSpec("qed", "max", weight=weights.get("qed", 1.0)),
        # Prefer routes whose frontier has fewer/smaller fragments
        "fragment_score": ObjectiveSpec("fragment_score", "max", weight=weights.get("fragment_score", 1.0)),
        "n_components": ObjectiveSpec("n_components", "min", weight=weights.get("n_components", 1.0)),
        "step_smoothness": ObjectiveSpec("step_smoothness", "max", weight=weights.get("step_smoothness", 0.5)),
    }


_scscore_model = None
_scscore_cache = None


def build_scscore_fn(model_dir=None, fp_length=1024) -> Callable[[str], float]:
    global _scscore_model
    
    # Prefer multi-score loader if available (returns callable identical signature)
    if _HAS_MULTISCORE:
        try:
            sc_fn, cache = _ms_make_scscore(
                model_dir=str(model_dir or config.scscore_dir),
                cache_path=None,
                fp_length=fp_length,
            )
            # keep cache for optional reuse in partial reward
            global _scscore_cache
            _scscore_cache = cache
            return sc_fn
        except Exception:
            # fall back to legacy path
            warnings.warn("multi-score SCScore loader failed; falling back to standalone SCScore.")

    if not _HAS_SCSCORE:
        warnings.warn("SCScore not available; falling back to a dummy complexity scorer (constant 5.0).")
        return lambda smiles: 5.0

    model_dir = model_dir or config.scscore_dir
    if _scscore_model is None:
        try:
            model = SCScorer()
            model.restore(str(model_dir), fp_length)
            _scscore_model = model
        except Exception as e:
            raise RuntimeError(f"Failed to load SCScore model: {e}")

    def _score(smiles: str) -> float:
        if not smiles:
            return 5.0
        try:
            out = _scscore_model.get_score_from_smi(smiles)
            arr = out[1] if isinstance(out, (tuple, list)) and len(out) == 2 else out
            arr = np.asarray(arr, dtype=float)
            return float(arr.mean()) if arr.ndim else float(arr)
        except Exception:
            return 5.0

    return _score


def build_scscore_with_cache(model_dir=None, fp_length=1024, cache_path: Optional[str] = None):
    """
    Return (sc_fn, cache) tuple. Uses multi-score if available; otherwise cache is None.
    """
    if _HAS_MULTISCORE:
        try:
            sc_fn, cache = _ms_make_scscore(
                model_dir=str(model_dir or config.scscore_dir),
                cache_path=cache_path,
                fp_length=fp_length,
            )
            return sc_fn, cache
        except Exception as e:
            warnings.warn(f"multi-score SCScore loader failed ({e}); falling back to standalone SCScore.")
    return build_scscore_fn(model_dir=model_dir, fp_length=fp_length), None


def build_partial_reward(sc_fn, cache=None):
    """Optional partial reward builder (multi-score if available)."""
    if _HAS_MULTISCORE:
        def _partial(molset, purch_fn, scscore_fn):
            return _ms_sc_partial_reward(molset, purch_fn, scscore_fn, cache=cache)
        return _partial
    return None


def make_evaluator(specs, inventory, audit_fn, sc_fn, target, partial_reward_fn=None):
    return RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=inventory.is_purchasable,
        audit_fn=audit_fn,
        scscore_fn=sc_fn,
        target_smiles=target,
        llm_style_scalar=config.llm_style_scalar,
        partial_reward_fn=partial_reward_fn,
    )
