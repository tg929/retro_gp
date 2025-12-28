"""Default paths and hyperparameters."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_root = root / "data"
scscore_dir = root / "scscore" / "models" / "full_reaxys_model_1024bool"

# GP defaults
pop_size = 10
generations = 20
pcross = 0.7
pmut = 0.4
seed = 123
max_templates_per_prog = 8

# Evaluation weights; tweak to discourage empty routes
objective_weights = {
    "solved": 100.0,
    "route_len": 1.0,
    "valid_prefix": 2.0,
    "sc_partial_reward": 1.0,
    "purch_frac": 3.0,
    "qed": 1.0,
    # New: fragmentation / smoothness related terms
    "fragment_score": 2.0,     # reward fewer/less tiny fragments in frontier
    "n_components": 1.0,       # penalize many frontier components
    "step_smoothness": 0.5,    # prefer templates with gentle size/ring changes
}

# Non-empty bonus to avoid empty programs dominating
nonempty_bonus = 2.0

# Optional: override scalarization with an LLM-Syn-Planner style reward
# (negative sum/mean of SCScore on the current non-purchasable set).
llm_style_scalar = False
