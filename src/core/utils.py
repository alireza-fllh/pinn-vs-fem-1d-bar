"""
Utility functions and data structures for 1D elastic bar solvers.

Provides error metrics and boundary condition specifications used by both
FEM and PINN implementations.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_error(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))


@dataclass
class BCSpec:
    """
    Boundary condition specification for 1D elastic bar problems.

    Supports three types of boundary conditions:
    - Dirichlet: u = prescribed_value
    - Neumann: EA*du/dx = prescribed_traction
    - Robin: alpha*u + beta*EA*du/dx = g (mixed condition)

    Args:
        kind: Boundary condition type ("dirichlet", "neumann", "robin")
        u: Prescribed displacement value (for Dirichlet BC)
        P: Prescribed traction EA*du/dx (for Neumann BC)
        alpha: Coefficient for displacement term (for Robin BC)
        beta: Coefficient for traction term (for Robin BC, typically 1.0)
        g: Right-hand side value (for Robin BC)

    Examples:
        >>> # Fixed displacement at left end
        >>> bc_left = BCSpec(kind="dirichlet", u=0.0)
        >>>
        >>> # Applied traction at right end
        >>> bc_right = BCSpec(kind="neumann", P=100.0)
        >>>
        >>> # Spring support: k*u + EA*du/dx = 0
        >>> bc_spring = BCSpec(kind="robin", alpha=k, beta=1.0, g=0.0)
    """
    kind: str                       # "dirichlet" | "neumann" | "robin"
    u: float = 0.0                  # for dirichlet
    P: float = 0.0                  # for neumann: EA*u' = P
    alpha: float = 0.0              # for robin
    beta: float = 0.0               # for robin (usually 1.0)
    g: float = 0.0                  # for robin: alpha*u + beta*EA*u' = g
