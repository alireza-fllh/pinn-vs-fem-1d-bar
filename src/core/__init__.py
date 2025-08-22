"""
Core reusable components: FEM, PINN, physics, shared utilities.
Import from here in most code to avoid long relative imports.
"""
from .fem import FEMConfig, solve_1d_bar
from .pinn import PINNConfig, PINNTrainer
from .utils import BCSpec
from .physics import get_case_config

__all__ = [
    "FEMConfig", "solve_1d_bar",
    "PINNConfig", "PINNTrainer",
    "BCSpec",
    "get_case_config",
]
