"""
gp_retro_nn: One-step retrosynthesis model adapters for GP.

This package provides a small interface (`OneStepRetrosynthesisModel`) plus
implementations that can be plugged into `gp_retro_feas.FeasibleExecutor`.
"""

from .one_step import OneStepPrediction, OneStepRetrosynthesisModel
from .nag2g_subprocess import NAG2GSubprocessConfig, NAG2GSubprocessModel
