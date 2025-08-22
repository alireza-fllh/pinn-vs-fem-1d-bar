"""
Comparative analysis between black-box and PINN approaches.

Provides tools for comparing the performance of supervised black-box models
against physics-informed neural networks on the same test cases.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core import BCSpec, FEMConfig, solve_1d_bar
from src.training import BBNet


def fem_ref(P: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate FEM reference solution for given traction parameter."""
    cfg = FEMConfig(N=200)
    bcL = BCSpec(kind="dirichlet", u=0.0)
    bcR = BCSpec(kind="neumann", P=P)
    xf, uf = solve_1d_bar(cfg, bcL, bcR)
    return xf, uf

def pinn_from_log(pinn_log: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract final PINN prediction from training log.

    Args:
        pinn_log: Path to PINN training log npz file

    Returns:
        Tuple of (x_coordinates, u_predictions) from final snapshot

    Raises:
        ValueError: If no snapshots found in log file
    """
    data = np.load(pinn_log)
    x = data["x"]
    snaps = data["snaps"]
    if snaps.shape[0] == 0:
        raise ValueError("No snapshots in PINN log. Enable snapshots in train_pinn.py.")
    u = snaps[-1]  # last snapshot
    return x, u

def bb_pred(model_path: str, P: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate black-box model prediction for given parameter.

    Args:
        model_path: Path to trained black-box model
        P: Traction parameter value

    Returns:
        Tuple of (x_coordinates, u_predictions)
    """
    model = BBNet(in_dim=2); model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    x = torch.linspace(0,1,401).reshape(-1,1)
    X = torch.cat([x, torch.full_like(x, P)], dim=1)
    with torch.no_grad():
        u = model(X).cpu().numpy().squeeze()
    return x.numpy().squeeze(), u

def compare_one(P: float, bb_model: str, pinn_log: str | None, outdir: str):
    """
    Compare FEM, black-box, and PINN solutions for a single parameter value.

    Args:
        P: Traction parameter value
        bb_model: Path to trained black-box model
        pinn_log: Path to PINN training log (optional)
        outdir: Output directory for plots

    Returns:
        Tuple of (l2_error_bb, l2_error_pinn)
    """
    xf, uf = fem_ref(P)
    xb, ub = bb_pred(bb_model, P)
    l2_bb = float(np.sqrt(np.mean((np.interp(xb, xf, uf) - ub)**2)))

    if pinn_log:
        xp, up = pinn_from_log(pinn_log)
        l2_pinn = float(np.sqrt(np.mean((np.interp(xp, xf, uf) - up)**2)))
    else:
        xp = up = None; l2_pinn = None

    # plot
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(xf, uf, label="FEM")
    plt.plot(xb, ub, "--", label=f"BB (L2={l2_bb:.2e})")
    if pinn_log:
        plt.plot(xp, up, ":", label=f"PINN (L2={l2_pinn:.2e})")
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.title(f"Comparison at P={P:.2f}")
    plt.legend(); plt.tight_layout()
    fig_path = Path(outdir) / f"compare_P{P:.2f}.png"
    plt.savefig(fig_path, dpi=200)
    print(f"[compare] saved {fig_path}")

    return l2_bb, l2_pinn

def main():
    """Command-line interface for comparative analysis."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=float, nargs="+", required=True)
    ap.add_argument("--bb-model", required=True)
    ap.add_argument("--pinn-log", default=None, help="PINN trainlog npz (must match a single P)")
    ap.add_argument("--pinn-P", type=float, default=None, help="P value used to train/log the PINN")
    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    results = []
    for P in args.Ps:
        pinn_log = args.pinn_log if (args.pinn_log is not None and args.pinn_P is not None and abs(P - args.pinn_P) < 1e-12) else None
        l2_bb, l2_pinn = compare_one(P, args.bb_model, pinn_log, args.out)
        results.append([P, l2_bb, (l2_pinn if l2_pinn is not None else np.nan)])

    # CSV
    csv = Path(args.out) / "compare_bb_pinn.csv"
    np.savetxt(csv, np.array(results), delimiter=",", header="P,L2_BB,L2_PINN", comments="")
    print(f"[compare] saved CSV to {csv}")

if __name__ == "__main__":
    main()
