"""
Core reusable components: FEM, PINN, physics, shared utilities.
Import from here in most code to avoid long relative imports.
"""
from .fem import FEMConfig, solve_1d_bar
from .pinn import PINNConfig, PINNTrainer
from .utils import BCSpec  # add more exports here if you add them to utils

__all__ = [
    "FEMConfig", "solve_1d_bar",
    "PINNConfig", "PINNTrainer",
    "BCSpec",
]
