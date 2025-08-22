from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .utils import BCSpec


@dataclass
class FEMConfig:
    L: float = 1.0
    N: int = 40
    E_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    A_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    body_force_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


def solve_1d_bar(cfg: FEMConfig, bc_left: BCSpec, bc_right: BCSpec):
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
    def apply_dirichlet(node: int, value: float):
        K[node, :] = 0.0
        K[:, node] = 0.0
        K[node, node] = 1.0
        F[node] = value

    def apply_neumann(node: int, P: float):
        F[node] += P

    def EA_at(node_x: float) -> float:
        return float(E_fn(np.array([node_x]))[0] * A_fn(np.array([node_x]))[0])

    # left
    if bc_left.kind == "dirichlet":
        apply_dirichlet(0, bc_left.u)
    elif bc_left.kind == "neumann":
        apply_neumann(0, bc_left.P)
    elif bc_left.kind == "robin":
        EA0 = EA_at(0.0)
        K[0, 0] += bc_left.alpha            # alpha * u(0)
        F[0] += bc_left.g                   # RHS g
        # beta * EA * u'(0) is naturally handled by stiffness; beta used here as scaling if needed
        if bc_left.beta != 1.0:
            K *= bc_left.beta  # very unusual; better: warn or incorporate in formulation

    # right
    if bc_right.kind == "dirichlet":
        apply_dirichlet(N, bc_right.u)
    elif bc_right.kind == "neumann":
        apply_neumann(N, bc_right.P)
    elif bc_right.kind == "robin":
        EAL = EA_at(L)
        K[N, N] += bc_right.alpha
        F[N] += bc_right.g
        if bc_right.beta != 1.0:
            K *= bc_right.beta

    # solve
    u = np.linalg.solve(K, F)
    return x, u
