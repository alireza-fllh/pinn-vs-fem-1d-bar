from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_preds(pinn_log: str, bb_log: str, fem_csv: str, outgif: str, title: str):
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
