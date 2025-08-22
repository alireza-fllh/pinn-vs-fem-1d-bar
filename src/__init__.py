"""
Top-level package for the PINN–FEM 1D elastic bar project.
Exposes subpackages: core, training, experiments, visualization.
"""
from . import core, training, experiments, visualization

__all__ = ["core", "training", "experiments", "visualization"]