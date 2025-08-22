"""
Prediction evolution animation utilities.

Creates dynamic visualizations of model predictions during training, showing
how PINN and black-box network solutions converge to the FEM reference solution.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_preds(pinn_log: str, bb_log: str, fem_csv: str, outgif: str, title: str):
    """
    Create animated visualization of prediction evolution during training.

    Generates an animated plot showing how PINN and black-box predictions
    evolve over training epochs compared to the FEM reference solution.

    Args:
        pinn_log: Path to PINN training log with prediction snapshots
        bb_log: Path to black-box training log with prediction snapshots
        fem_csv: Path to FEM solution CSV file
        outgif: Output path for the generated GIF animation
        title: Title for the animation plot
    """
    P = Path(outgif).parent; P.mkdir(parents=True, exist_ok=True)

    fem = np.loadtxt(fem_csv, delimiter=",", skiprows=1)
    xf, uf = fem[:, 0], fem[:, 1]

    Pn = np.load(pinn_log)
    Bb = np.load(bb_log)

    x = Pn["x"]                               # common grid for PINN snapshots
    snaps_p = Pn["snaps"]; ep_p = Pn["snaps_epochs"]
    snaps_b = Bb["snaps"]; ep_b = Bb["snaps_epochs"]
    T = min(len(snaps_p), len(snaps_b))
    if T == 0:
        raise RuntimeError("No snapshots found. Ensure both trainers save snaps every N epochs.")

    y_min = min(uf.min(), snaps_p.min(), snaps_b.min()) * 1.1
    y_max = max(uf.max(), snaps_p.max(), snaps_b.max()) * 1.1

    fig, ax = plt.subplots()
    ax.plot(xf, uf, lw=2, label="FEM")
    lp, = ax.plot([], [], lw=2, label="PINN")
    lb, = ax.plot([], [], lw=2, ls="--", label="Black‑box")
    ax.set_xlim(0, 1); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.legend()

    def update(i):
        lp.set_data(x, snaps_p[i])
        # bb snapshots may have their own x; if so, resample:
        xb = Bb["x"]
        if np.allclose(xb, x):
            lb.set_data(x, snaps_b[i])
        else:
            from numpy import interp
            lb.set_data(x, interp(x, xb, snaps_b[i]))
        ax.set_title(f"{title} | epochs: PINN {int(ep_p[i])}, BB {int(ep_b[i])}")
        return lp, lb

    ani = FuncAnimation(fig, update, frames=T, blit=False, interval=120)
    ani.save(outgif, writer=PillowWriter(fps=6))
    print(f"[animate_predictions] saved {outgif}")

def main():
    """
    Command-line interface for prediction animation generation.

    Parses command-line arguments and generates animated prediction evolution plots.
    Compares PINN and black-box predictions against FEM reference solutions,
    showing convergence behavior over training epochs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True, help="pinn_*_trainlog.npz")
    ap.add_argument("--bb", required=True, help="bb_trainlog.npz")
    ap.add_argument("--fem", required=True, help="fem_*.csv matching the case/P")
    ap.add_argument("--out", default="data/outputs/anim_pred.gif")
    ap.add_argument("--title", default="Training evolution")
    args = ap.parse_args()
    animate_preds(args.pinn, args.bb, args.fem, args.out, args.title)

if __name__ == "__main__":
    main()
