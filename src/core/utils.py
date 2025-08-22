from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2_error(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))


@dataclass
class BCSpec:
    kind: str                       # "dirichlet" | "neumann" | "robin"
    u: float = 0.0                  # for dirichlet
    P: float = 0.0                  # for neumann: EA*u' = P
    alpha: float = 0.0              # for robin
    beta: float = 0.0               # for robin (usually 1.0)
    g: float = 0.0                  # for robin: alpha*u + beta*EA*u' = g
