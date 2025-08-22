"""
Physics definitions for 1D elastic bar problems.

Provides predefined material properties, loading functions, and case configurations
for common benchmark problems.

Author: Alireza Fallahnejad
"""

from __future__ import annotations
import math
import torch


# -----------------------------
# Material property functions
# -----------------------------

def constant_E(x):
    """Constant Young's modulus E = 1.0"""
    return torch.ones_like(x)

def constant_A(x):
    """Constant cross-sectional area A = 1.0"""
    return torch.ones_like(x)

def E_two_layer(x):
    """Piecewise Young's modulus: E = 1.0 for x < 0.5, E = 3.0 for x >= 0.5"""
    return torch.where(x < 0.5, torch.ones_like(x), 3.0 * torch.ones_like(x))

def A_taper(x):
    """Tapered cross-section: A(x) = 1 + 0.5*x"""
    return 1.0 + 0.5 * x


# -----------------------------
# Loading functions
# -----------------------------

def body_force_uniform(x):
    """Uniform body force f = 1.0"""
    return torch.ones_like(x)

def body_force_zero(x):
    """Zero body force f = 0.0"""
    return torch.zeros_like(x)

def body_force_sine(x):
    """Sinusoidal body force f = 2*sin(2πx)"""
    return 2.0 * torch.sin(2 * math.pi * x)


# -----------------------------
# Case configurations
# -----------------------------

def get_case_config(case: str):
    """
    Get material and loading functions for predefined cases.

    Args:
        case: Case name ("body_force", "tip_load", "hetero")

    Returns:
        tuple: (E_fn, A_fn, f_fn, P_neumann)
    """
    if case == "body_force":
        return constant_E, constant_A, body_force_uniform, 0.0
    elif case == "tip_load":
        return constant_E, constant_A, body_force_zero, 1.0
    elif case == "hetero":
        return E_two_layer, A_taper, body_force_sine, 0.0
    else:
        raise ValueError(f"Unknown case: {case}")


# Legacy support (kept for backward compatibility)
CASE_MAP = {
    "body_force": lambda: (body_force_uniform, 0.0),
    "tip_load": lambda: (body_force_zero, 1.0),
}
