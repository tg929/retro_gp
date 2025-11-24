"""Default paths and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SCSCORE_DIR = ROOT / "scscore" / "models" / "full_reaxys_model_1024bool"

# GP defaults
POP_SIZE = 12
GENERATIONS = 8
PCROSS = 0.7
PMUT = 0.4
SEED = 123
MAX_TEMPLATES_PER_PROG = 5

# Evaluation weights; tweak to discourage empty routes
OBJECTIVE_WEIGHTS = {
    "solved": 100.0,
    "route_len": 1.0,
    "valid_prefix": 2.0,
    "sc_partial_reward": 1.0,
    "purch_frac": 3.0,
    "qed": 1.0,
}

# Non-empty bonus to avoid empty programs dominating
NONEMPTY_BONUS = 2.0

# Optional: override scalarization with an LLM-Syn-Planner style reward
# (negative sum/mean of SCScore on the current non-purchasable set).
LLM_STYLE_SCALAR = False
