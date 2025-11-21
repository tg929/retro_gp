from __future__ import annotations

"""
SCScore integration + redesigned partial reward for LLM‑Syn‑Planner / RetroGP.

This module gives you:
- SCScoreModel: robust loader for the official SCScore package (multiple backends)
- SCScoreCache: tiny disk cache so repeated scoring is fast
- sc_partial_reward(): redesigned partial reward used at the "first invalid step"
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, Optional, Tuple, List
import os
import json
import math
import atexit

# -------------------------------
# 1) SCScore loader (robust)
# -------------------------------

@dataclass
class SCScoreModel:
    """
    Wrapper around Connor Coley's SCScore package.

    It tries several backends in the following order:
    - scscore.standalone_model_numpy.SCScorer  (pure NumPy, recommended)
    - scscore.standalone_model.SCScorer        (TF backend, older)
    You can provide model_dir explicitly, or rely on environment variable SCSCORE_MODEL_DIR.
    """
    model_dir: Optional[str] = None
    fp_length: int = 1024

    def __post_init__(self):
        self._backend = None
        self._model = None
        self._restore()

    def _restore(self):
        # Resolve model dir
        model_dir = self.model_dir or os.environ.get("SCSCORE_MODEL_DIR", None)

        # Try NumPy standalone first
        try:
            from scscore.standalone_model_numpy import SCScorer  # type: ignore
            self._backend = "numpy"
            self._model = SCScorer()
            if model_dir:
                self._model.restore(model_dir, self.fp_length)
            else:
                # some releases have a load() that auto-locates defaults
                try:
                    self._model.load()  # type: ignore
                except Exception:
                    raise
            return
        except Exception:
            pass

        # Try TF backend
        try:
            from scscore.standalone_model import SCScorer  # type: ignore
            self._backend = "tf"
            self._model = SCScorer()
            if model_dir:
                self._model.restore(model_dir, self.fp_length)
            else:
                # some releases have a load() that auto-locates defaults
                try:
                    self._model.load()  # type: ignore
                except Exception:
                    raise
            return
        except Exception as e:
            raise RuntimeError(
                "SCScore package not available or model cannot be loaded. "
                "Install `scscore` and set SCSCORE_MODEL_DIR to the pretrained models directory."
            ) from e

    @lru_cache(maxsize=200_000)
    def score(self, smiles: str) -> float:
        """
        Return SCScore in the canonical 1..5 range (float).
        If the backend returns a (norm, score) tuple, we take the second.
        """
        if smiles is None:
            return 5.0
        s = smiles.strip()
        if not s:
            return 5.0
        val = self._model.get_score_from_smi(s)  # type: ignore
        # many impls return (norm, score)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[1])
        return float(val)

# -------------------------------
# 2) Tiny on-disk cache
# -------------------------------

class SCScoreCache:
    """
    A tiny JSONL cache: each line is {"smi": "...", "score": x}.
    Avoids recomputing SCScore repeatedly across runs.
    """
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._mem: Dict[str, float] = {}
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        self._mem[obj["smi"]] = float(obj["score"])
            except Exception:
                # ignore corrupted cache
                self._mem = {}
        atexit.register(self._flush)

    def get(self, smi: str) -> Optional[float]:
        return self._mem.get(smi)

    def put(self, smi: str, score: float):
        self._mem[smi] = float(score)

    def _flush(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for smi, s in self._mem.items():
                    f.write(json.dumps({"smi": smi, "score": float(s)}) + "\n")
        except Exception:
            pass

# -------------------------------
# 3) Redesigned partial reward
# -------------------------------

@dataclass
class PartialRewardConfig:
    """
    Hyper-parameters for sc_partial_reward().
    All terms are designed so the final reward is in [-6, 0], where 0 is best.
    """
    alpha_mean: float = 1.0   # weight for mean SCScore of non-purchasable set (1..5)
    beta_logcount: float = 0.25  # penalty per log(1 + |U|)
    gamma_coverage: float = 1.0  # penalty for fraction of non-purchasable molecules |U|/|M|
    cap_min: float = -6.0
    cap_max: float = 0.0

def _cap(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sc_partial_reward(
    molecule_set: Iterable[str],
    purchasable_fn: Callable[[str], bool],
    scscore_fn: Callable[[str], float],
    cache: Optional[SCScoreCache] = None,
    cfg: Optional[PartialRewardConfig] = None,
) -> float:
    """
    Compute partial reward on the "current molecule set" at the first invalid step.

    Let M be the current set; U = { m in M | not purchasable(m) }.
    Reward is negative; 0 is best (everything purchasable).

        r = - [ alpha * mean_SCScore(U) + beta * log(1 + |U|) + gamma * (|U|/|M|) ]

    Rationale:
    - mean_SCScore(U) reflects synthetic difficulty of the *remaining* unsourced intermediates
    - log(1+|U|) softly penalizes many remaining intermediates
    - |U|/|M| penalizes low purchasability fraction to encourage progress
    """
    cfg = cfg or PartialRewardConfig()
    M = [s for s in molecule_set]
    if not M:
        return 0.0
    U = [s for s in M if not _safe_purch(purchasable_fn, s)]

    if not U:
        return 0.0

    # SCScore mean
    scores: List[float] = []
    for smi in U:
        if cache:
            v = cache.get(smi)
            if v is None:
                v = float(scscore_fn(smi))
                cache.put(smi, v)
        else:
            v = float(scscore_fn(smi))
        # SCScore nominal range: 1..5; clamp to be safe
        v = _cap(v, 0.0, 10.0)
        scores.append(v)
    mean_sc = sum(scores) / max(1, len(scores))

    term_mean = cfg.alpha_mean * mean_sc
    term_logc = cfg.beta_logcount * math.log1p(len(U))
    term_cov  = cfg.gamma_coverage * (len(U) / max(1, len(M)))

    reward = -(term_mean + term_logc + term_cov)
    return _cap(reward, cfg.cap_min, cfg.cap_max)

def _safe_purch(purch_fn: Callable[[str], bool], smi: str) -> bool:
    try:
        return bool(purch_fn(smi))
    except Exception:
        return False

# -------------------------------
# 4) Convenience factory
# -------------------------------

def make_scscore(model_dir: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 fp_length: int = 1024) -> Tuple[Callable[[str], float], Optional[SCScoreCache]]:
    """
    Returns (scscore_fn, cache) tuple ready to be passed to sc_partial_reward().
    """
    model = SCScoreModel(model_dir=model_dir, fp_length=fp_length)
    cache = SCScoreCache(cache_path) if cache_path else None
    return model.score, cache
