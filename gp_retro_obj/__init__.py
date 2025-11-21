"""
gp_retro_obj: Objectives & Fitness (multi-objective) layer for GP retrosynthesis

This package plugs into your existing gp_retro_repr (route representation) and gp_retro_feas (feasibility)
modules to compute route-level objectives, partial rewards, and to perform multi-objective selection.
"""
from .objectives import Objective, ObjectiveSpec, ObjectiveVector
from .fitness import RouteFitnessEvaluator, FitnessResult, Scalarizer
from .selectors import epsilon_lexicase_select, nsga2_survivor_selection
from .oracles import PropertyOracleRegistry, qed_oracle_available
