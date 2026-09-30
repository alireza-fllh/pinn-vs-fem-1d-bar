"""
Finite Element Method (FEM) solver for 1D elastic bar problems.

Solves: -d/dx(E(x) * A(x) * du/dx) = f(x) on domain [0, L]
Supports variable material properties and Dirichlet/Neumann/Robin BCs.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import BCSpec


@dataclass
class FEMConfig:
    """
    FEM solver configuration.

    Args:
        L: Domain length [0, L]. Default: 1.0
        N: Number of elements. Default: 40
        E_fn: Young's modulus E(x). Default: constant 1.0
        A_fn: Cross-sectional area A(x). Default: constant 1.0
        body_force_fn: Body force f(x). Default: zero
    """
    L: float = 1.0
    N: int = 40
    E_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    A_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    body_force_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


def solve_1d_bar(cfg: FEMConfig, bc_left: BCSpec, bc_right: BCSpec):
    """
    Solve 1D elastic bar using linear finite elements.

    Args:
        cfg: FEM configuration
        bc_left: Left boundary condition (x=0)
        bc_right: Right boundary condition (x=L)

    Returns:
        x: Nodal coordinates (N+1,)
        u: Nodal displacements (N+1,)
    """
    L, N = cfg.L, cfg.N
    x = np.linspace(0.0, L, N + 1)
    h = L / N
    E_fn = cfg.E_fn or (lambda xx: np.ones_like(xx))
    A_fn = cfg.A_fn or (lambda xx: np.ones_like(xx))
    f_fn = cfg.body_force_fn or (lambda xx: np.zeros_like(xx))

    K = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    # assembly (midpoint)
    for e in range(N):
        xL, xR = x[e], x[e+1]
        xm = 0.5 * (xL + xR)
        EA_m = E_fn(np.array([xm]))[0] * A_fn(np.array([xm]))[0]
        k_local = (EA_m / h) * np.array([[1, -1], [-1, 1]])
        dofs = [e, e+1]
        K[np.ix_(dofs, dofs)] += k_local
        fA_m = f_fn(np.array([xm]))[0] * A_fn(np.array([xm]))[0]
        F[dofs] += fA_m * h * 0.5

    # ---- apply BCs ----
    # Integrating by parts leaves the boundary term s * EA u'(x_b) v(x_b), with
    # s = +1 at x = L and s = -1 at x = 0 (the outward normal points in -x there).
    def apply_dirichlet(node: int, value: float):
        F[:] -= K[:, node] * value  # move the known column to the RHS
        K[node, :] = 0.0
        K[:, node] = 0.0
        K[node, node] = 1.0
        F[node] = value

    dirichlet = []
    for node, s, bc in ((0, -1.0, bc_left), (N, 1.0, bc_right)):
        if bc.kind == "dirichlet":
            dirichlet.append((node, bc.u))
        elif bc.kind == "neumann":
            # EA u' = P
            F[node] += s * bc.P
        elif bc.kind == "robin":
            # alpha u + beta EA u' = g  ->  EA u' = (g - alpha u) / beta
            if bc.beta == 0.0:
                dirichlet.append((node, bc.g / bc.alpha))
            else:
                K[node, node] += s * bc.alpha / bc.beta
                F[node] += s * bc.g / bc.beta

    # essential BCs last, so the RHS lift sees the final stiffness
    for node, value in dirichlet:
        apply_dirichlet(node, value)

    # solve
    u = np.linalg.solve(K, F)
    return x, u
