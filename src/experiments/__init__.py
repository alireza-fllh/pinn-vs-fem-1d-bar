"""
Experiment orchestration: dataset generation, sweeps, comparisons.
"""
from .data_gen import gen_dataset
# If you have functions in compare_bb_vs_pinn.py you want to reuse, export them here.

__all__ = ["gen_dataset"]
