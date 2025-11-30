"""SCScore loading and fitness evaluator helpers."""
import os
from typing import Callable, Dict
import warnings

import numpy as np
try:
    from scscore.scscore.standalone_model_numpy import SCScorer
    _HAS_SCSCORE = True
except ImportError:
    # Try to use the vendored scscore in the repo root if available
    import sys
    import pathlib

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

from gp_retro_obj import ObjectiveSpec, RouteFitnessEvaluator
from . import config


def build_objectives(weights: Dict[str, float]) -> Dict[str, ObjectiveSpec]:
    return {
        "solved": ObjectiveSpec("solved", "max", weight=weights.get("solved", 100.0)),
        "route_len": ObjectiveSpec("route_len", "min", weight=weights.get("route_len", 1.0)),
        "valid_prefix": ObjectiveSpec("valid_prefix", "max", weight=weights.get("valid_prefix", 1.0)),
        "sc_partial_reward": ObjectiveSpec("sc_partial_reward", "max", weight=weights.get("sc_partial_reward", 1.0)),
        "purch_frac": ObjectiveSpec("purch_frac", "max", weight=weights.get("purch_frac", 1.0)),
        "qed": ObjectiveSpec("qed", "max", weight=weights.get("qed", 1.0)),
    }


_scscore_model = None


def build_scscore_fn(model_dir=None, fp_length=1024) -> Callable[[str], float]:
    global _scscore_model
    
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


def make_evaluator(specs, inventory, audit_fn, sc_fn, target):
    return RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=inventory.is_purchasable,
        audit_fn=audit_fn,
        scscore_fn=sc_fn,
        target_smiles=target,
        llm_style_scalar=config.llm_style_scalar,
    )
