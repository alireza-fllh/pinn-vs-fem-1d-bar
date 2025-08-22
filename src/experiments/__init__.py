"""
Experiment orchestration and comparative analysis tools.

This module provides functionality for:
- Dataset generation for supervised learning experiments
- Comparative analysis between PINN and black-box approaches  
- Parameter sweeps and performance evaluation

Author: Alireza Fallahnejad
"""
from .data_gen import gen_dataset

__all__ = ["gen_dataset"]
