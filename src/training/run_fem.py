"""
Command-line interface for Finite Element Method solver.

Provides a flexible CLI for running FEM simulations with configurable
boundary conditions, material properties, and predefined case studies.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.core import BCSpec, FEMConfig, solve_1d_bar, get_case_config


def bc_from_args(side: str, args: argparse.Namespace) -> BCSpec:
    """
    Create boundary condition specification from command-line arguments.

    Args:
        side: Boundary side ("left" or "right")
        args: Parsed command-line arguments

    Returns:
        BCSpec object configured according to arguments

    Raises:
        ValueError: If unknown boundary condition type specified
    """
    if side == "left":
        kind = args.left_type
        u, P = args.u0, args.P0
        alpha, beta, g = args.alpha0, args.beta0, args.g0
    else:
        kind = args.right_type
        u, P = args.uL, args.PL
        alpha, beta, g = args.alphaL, args.betaL, args.gL

    if kind == "dirichlet":
        return BCSpec(kind="dirichlet", u=u)
    elif kind == "neumann":
        return BCSpec(kind="neumann", P=P)
    elif kind == "robin":
        return BCSpec(kind="robin", alpha=alpha, beta=beta, g=g)
    else:
        raise ValueError(f"Unknown BC kind for {side}: {kind}")


def main():
    """
    Command-line interface for FEM solver execution.

    Supports flexible boundary condition specification, predefined case studies,
    and automatic material property assignment. Results are saved as CSV files
    suitable for comparison with other solution methods.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["body_force", "tip_load", "hetero"], default="body_force")
    ap.add_argument("--elements", type=int, default=40)

    # --- BCs (same interface for both ends) ---
    ap.add_argument("--left-type",  choices=["dirichlet","neumann","robin"], default="dirichlet")
    ap.add_argument("--right-type", choices=["dirichlet","neumann","robin"], default=None)  # set default by case

    # left parameters
    ap.add_argument("--u0", type=float, default=0.0)
    ap.add_argument("--P0", type=float, default=0.0)
    ap.add_argument("--alpha0", type=float, default=0.0)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--g0", type=float, default=0.0)

    # right parameters
    ap.add_argument("--uL", type=float, default=0.0)
    ap.add_argument("--PL", type=float, default=None)  # may come from case
    ap.add_argument("--alphaL", type=float, default=0.0)
    ap.add_argument("--betaL", type=float, default=1.0)
    ap.add_argument("--gL", type=float, default=0.0)

    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    # Get case configuration and convert to numpy functions
    E_fn_torch, A_fn_torch, f_fn_torch, P_default = get_case_config(args.case)

    # Convert torch functions to numpy for FEM
    E_fn = lambda x: E_fn_torch(torch.from_numpy(x)).numpy()
    A_fn = lambda x: A_fn_torch(torch.from_numpy(x)).numpy()
    f_fn = lambda x: f_fn_torch(torch.from_numpy(x)).numpy()
    default_right = ("neumann", P_default)

    # Default right BC by case unless user overrides type/PL
    if args.right_type is None:
        args.right_type = default_right[0]
        if args.PL is None:
            args.PL = default_right[1]

    bc_left = bc_from_args("left", args)
    bc_right = bc_from_args("right", args)

    print(f"[FEM] BCs: left={bc_left} | right={bc_right}")

    cfg = FEMConfig(
        N=args.elements,
        E_fn=E_fn, A_fn=A_fn, body_force_fn=f_fn,
    )

    x, u = solve_1d_bar(cfg, bc_left=bc_left, bc_right=bc_right)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outdir / f"fem_{args.case}.csv", np.c_[x, u], delimiter=",", header="x,u", comments="")
    print(f"Saved FEM solution to {outdir}/fem_{args.case}.csv")

if __name__ == "__main__":
    main()
