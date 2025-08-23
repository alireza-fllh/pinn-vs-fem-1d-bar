"""
Training modules and command-line interfaces for individual model execution.

Provides CLI scripts for running FEM solvers, PINN training, and black-box
neural network training, along with evaluation utilities.

Author: Alireza Fallahnejad
"""
# Black-box model helpers
from .train_blackbox import BBConfig, BBNet, train_bb

__all__ = ["BBConfig", "BBNet", "train_bb"]
