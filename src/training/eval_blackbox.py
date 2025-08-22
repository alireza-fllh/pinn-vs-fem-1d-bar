"""
Evaluation utilities for trained black-box neural networks.

Provides tools for testing black-box model performance against FEM references
with optional visualization and error quantification.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core import BCSpec, FEMConfig, solve_1d_bar

from .train_blackbox import BBNet


def eval_one_P(model_path: str, P: float, plot: bool, outdir: str):
    """
    Evaluate black-box model performance for a single parameter value.

    Args:
        model_path: Path to trained model weights
        P: Traction parameter value for evaluation
        plot: Whether to generate comparison plots
        outdir: Output directory for plots and results

    Returns:
        L2 error compared to FEM reference solution
    """
    # load model
    model = BBNet(in_dim=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # BB prediction on dense grid
    x = torch.linspace(0,1,401).reshape(-1,1)
    X = torch.cat([x, torch.full_like(x, P)], dim=1)
    with torch.no_grad():
        u_bb = model(X).cpu().numpy().squeeze()

    # FEM reference
    cfg = FEMConfig(N=200)
    bcL = BCSpec(kind="dirichlet", u=0.0)
    bcR = BCSpec(kind="neumann", P=P)
    xf, uf = solve_1d_bar(cfg, bcL, bcR)

    # L2 error (on same grid)
    u_fem_on_x = np.interp(x.numpy().squeeze(), xf, uf)
    l2 = float(np.sqrt(np.mean((u_bb - u_fem_on_x)**2)))
    print(f"[eval_bb] P={P:.3f}  L2={l2:.3e}")

    # plot
    if plot:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(xf, uf, label="FEM")
        plt.plot(x.numpy().squeeze(), u_bb, "--", label="BB")
        plt.xlabel("x"); plt.ylabel("u(x)"); plt.title(f"BB vs FEM (P={P:.2f}, L2={l2:.2e})")
        plt.legend(); plt.tight_layout()
        path = Path(outdir) / f"bb_vs_fem_P{P:.2f}.png"
        plt.savefig(path, dpi=200)
        print(f"[eval_bb] saved {path}")

    return l2

def main():
    """
    Command-line interface for black-box model evaluation.

    Evaluates trained models across multiple parameter values with optional
    visualization and error quantification against FEM references.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--Ps", type=float, nargs="+", required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    results = []
    for P in args.Ps:
        l2 = eval_one_P(args.model, P, plot=args.plot, outdir=args.out)
        results.append((P, l2))

    # save CSV
    Path(args.out).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) / "bb_eval.csv"
    np.savetxt(csv_path, np.array(results), delimiter=",", header="P,L2", comments="")
    print(f"[eval_bb] saved CSV to {csv_path}")

if __name__ == "__main__":
    main()
