from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gp_core.scoring import (
    GPFitnessEvaluator,
    ScoringConfig,
    StepScoreConfig,
    StepScorer,
)
from gp_retro_nn import OneStepPrediction
from gp_retro_repr import Inventory, LeafCriteria, LeafCriteriaConfig, Route, RetrosynthesisStep


def main() -> None:
    inv = Inventory(["O", "CO"])  # purchasable building blocks

    scorer = StepScorer(
        config=StepScoreConfig(score_type="rank", topB=2, topN=5),
        bb_is_purchasable=inv.is_purchasable,
        sa_fn=lambda s: float(len(s)),
        forward_model=None,
    )

    product = "CCO"
    preds = [
        OneStepPrediction(reactants=["CCCC"], score=None, meta={"rank": 0}),   # not purchasable, long
        OneStepPrediction(reactants=["O"], score=None, meta={"rank": 1}),      # purchasable, short
        OneStepPrediction(reactants=["CO", "O"], score=None, meta={"rank": 2}),  # both purchasable
    ]

    top = scorer.rank_and_truncate(product, preds)
    assert len(top) == 2
    # With rank-based scoring, we still expect weak-rule signals to matter:
    # purchasable / simpler candidates should not be ranked behind a clearly worse one.
    assert top[0].reactants != ["CCCC"]

    # Build a tiny solved route
    r = Route()
    r.append(
        RetrosynthesisStep(
            molecule_set=[product],
            rational="test",
            product=product,
            template_id="nag2g@rank=0",
            reactants=["O"],
            updated_molecule_set=["O"],
            diagnostics={"one_step_log_score_single": -0.1},
        )
    )
    assert r.is_solved(inv)

    evaluator = GPFitnessEvaluator(
        target_smiles=product,
        bb_is_purchasable=inv.is_purchasable,
        config=ScoringConfig(),
        sa_fn=lambda s: float(len(s)),
        price_fn=None,
    )
    fit = evaluator.evaluate(r, program=None, invalid=False)
    assert fit.objectives["solved"] == 1.0

    # Unsolved route should score worse than solved (scalar is packed negative key for solved)
    r2 = Route()
    fit2 = evaluator.evaluate(r2, program=None, invalid=False)
    assert fit2.objectives["solved"] == 0.0

    # ASKCOS-style leaf criteria: allow non-buyable small molecules (e.g. 'N') to be treated as leaves
    try:
        import rdkit  # noqa: F401
    except Exception:
        return

    inv2 = Inventory(["O"])
    inv2.set_leaf_criteria(
        LeafCriteria(
            cfg=LeafCriteriaConfig(
                max_natom_dict=LeafCriteriaConfig.make_max_natom_dict(logic="or", C=0, N=1, O=0, H=3),
                min_chemical_history_dict=LeafCriteriaConfig.make_min_history_dict(logic="none"),
            )
        )
    )
    assert inv2.is_purchasable("N") is False
    assert inv2.is_leaf("N") is True

    print("scoring_smoke_test: OK")


if __name__ == "__main__":
    main()
