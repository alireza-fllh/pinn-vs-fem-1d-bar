"""
Training loss evolution animation utilities.

Creates dynamic visualizations of training progress showing loss curves over time,
useful for understanding convergence behavior and comparing different training runs.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_losses(pinn_log: str, bb_log: str, outgif: str):
    """
    Create animated visualization of loss evolution during training.

    Generates an animated plot comparing PINN and black-box network training
    losses over epochs, useful for visualizing convergence characteristics.

    Args:
        pinn_log: Path to PINN training log (.npz file)
        bb_log: Path to black-box network training log (.npz file)
        outgif: Output path for the generated GIF animation
    """
    P = Path(outgif).parent; P.mkdir(parents=True, exist_ok=True)

    Lp = np.load(pinn_log)["losses"][:, 0]  # total PINN loss
    Lb = np.load(bb_log)["losses"]          # total BB loss
    T = min(len(Lp), len(Lb))

    fig, ax = plt.subplots()
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss"); ax.set_yscale("log")
    lp, = ax.plot([], [], label="PINN")
    lb, = ax.plot([], [], label="Black‑box")
    ax.legend()

    def update(i):
        n = i + 1
        xs = np.arange(n)
        lp.set_data(xs, Lp[:n])
        lb.set_data(xs, Lb[:n])
        ax.set_xlim(0, T)
        ymin = max(1e-12, min(Lp[:n].min(), Lb[:n].min()) * 0.8)
        ymax = max(Lp[:n].max(), Lb[:n].max()) * 1.2
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Loss vs Epoch (n={n})")
        return lp, lb

    ani = FuncAnimation(fig, update, frames=T, blit=False, interval=30)
    ani.save(outgif, writer=PillowWriter(fps=20))
    print(f"[animate_losses] saved {outgif}")

def main():
    """
    Command-line interface for loss animation generation.

    Parses command-line arguments and generates animated loss comparison plots.
    Expects PINN and black-box training logs, outputs animated GIF showing
    loss evolution over training epochs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True, help="pinn_*_trainlog.npz")
    ap.add_argument("--bb", required=True, help="bb_trainlog.npz")
    ap.add_argument("--out", default="data/outputs/anim_loss.gif")
    args = ap.parse_args()
    animate_losses(args.pinn, args.bb, args.out)

if __name__ == "__main__":
    main()
