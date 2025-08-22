from __future__ import annotations

import torch


def material_params():
    E = 1.0
    A = 1.0
    L = 1.0
    return E, A, L


# -----------------------------
# Load / material / area fields
# -----------------------------
# Simple baselines

def body_force_const():
    def f(x):
        return torch.ones_like(x)
    P = 0.0
    return f, P

def tip_load_const():
    def f(x):
        return torch.zeros_like(x)
    P = 1.0
    return f, P

def E_two_layer(x):
    # piecewise Young's modulus: [0,0.5): 1.0; [0.5,1]: 3.0
    return torch.where(x < 0.5, torch.full_like(x, 1.0), torch.full_like(x, 3.0))

def A_taper(x):
    # tapered cross-section: A(x) = 1 + 0.5 x
    return 1.0 + 0.5 * x

def f_sine(x):
    # non-uniform body force
    return 2.0 * torch.sin(2 * torch.pi * x)


CASE_MAP = {
"body_force": body_force_const,             # f(x)=1, P=0
"tip_load": tip_load_const,                 # f(x)=0, P=1
}


# Default coefficient fields (can be overridden per case)
DEFAULT_E_FN = lambda x: torch.ones_like(x)
DEFAULT_A_FN = lambda x: torch.ones_like(x)
