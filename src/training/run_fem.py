from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core import BCSpec, FEMConfig, solve_1d_bar


def bc_from_args(side: str, args: argparse.Namespace) -> BCSpec:
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

    # Loads/coefficients by case
    if args.case == "body_force":
        E_fn = lambda x: np.ones_like(x)
        A_fn = lambda x: np.ones_like(x)
        f_fn = lambda x: np.ones_like(x)
        default_right = ("neumann", 0.0)
    elif args.case == "tip_load":
        E_fn = lambda x: np.ones_like(x)
        A_fn = lambda x: np.ones_like(x)
        f_fn = lambda x: np.zeros_like(x)
        default_right = ("neumann", 1.0)
    else:  # hetero
        E_fn = lambda x: np.where(x < 0.5, 1.0, 3.0)
        A_fn = lambda x: 1.0 + 0.5 * x
        f_fn = lambda x: 2.0 * np.sin(2*np.pi*x)
        default_right = ("neumann", 0.0)

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
