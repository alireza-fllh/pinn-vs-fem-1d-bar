"""
Basic plotting utilities for FEM vs PINN comparison.

Generates static plots comparing displacement solutions and training loss evolution
with error quantification and visualization.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Generate comparison plots between FEM and PINN solutions.

    Creates three types of plots:
    1. Displacement comparison with L2 error
    2. Pointwise error distribution
    3. Training loss evolution (if available)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["body_force", "tip_load", "hetero"], default="body_force")
    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    outdir = Path(args.out)
    fem = np.loadtxt(outdir / f"fem_{args.case}.csv", delimiter=",", skiprows=1)
    pinn = np.load(outdir / f"pinn_{args.case}.npz")

    # Interpolate FEM to PINN grid for error
    u_fem_on_pinn = np.interp(pinn["x"], fem[:, 0], fem[:, 1])
    err = pinn["u"] - u_fem_on_pinn
    l2 = float(np.sqrt(np.mean(err ** 2)))

    # Plot displacement
    plt.figure()
    plt.plot(fem[:, 0], fem[:, 1], label="FEM")
    plt.plot(pinn["x"], pinn["u"], "--", label="PINN")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"Displacement — {args.case} (L2={l2:.3e})")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"fig_{args.case}_disp.png", dpi=200)

    # Error vs x
    plt.figure()
    plt.plot(pinn["x"], np.abs(err))
    plt.xlabel("x"); plt.ylabel("|u_PINN - u_FEM|")
    plt.title(f"Pointwise error — {args.case}")
    plt.tight_layout()
    plt.savefig(outdir / f"fig_{args.case}_err.png", dpi=200)

    # Loss components if present
    if "losses" in pinn:
        losses = pinn["losses"]
        plt.figure()
        plt.plot(losses[:, 0], label="total")
        plt.plot(losses[:, 1], label="pde")
        plt.plot(losses[:, 2], label="bc_robin0")
        plt.plot(losses[:, 3], label="bc_neuL")
        plt.yscale("log")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title(f"Training losses — {args.case}")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"fig_{args.case}_loss.png", dpi=200)


    print(f"Saved plots to {outdir} (L2={l2:.3e})")


if __name__ == "__main__":
    main()
