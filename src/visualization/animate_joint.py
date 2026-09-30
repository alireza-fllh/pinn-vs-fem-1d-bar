"""
Joint training-dynamics animation for the README banner.

Top panel: PINN and black-box predictions converging to the FEM reference.
Bottom panel: total training loss of both models up to the current epoch.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _load_style():
    # Use repo style if present
    here = Path(__file__).resolve().parent
    style = here / ".." / ".." / "assets" / "figstyle.mplstyle"
    style = style.resolve()
    if style.exists():
        plt.style.use(str(style))

def _load_fem(csv_path: str):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]

def _load_log(npz_path: str, key_losses: str = "losses"):
    d = np.load(npz_path, allow_pickle=True)
    x = np.asarray(d["x"]).ravel()

    # snaps can be [T, N] (preferred) or sometimes [N] with no time dimension
    snaps = d["snaps"] if "snaps" in d else None
    if snaps is not None:
        snaps = np.asarray(snaps)
        if snaps.ndim == 1:
            snaps = snaps[None, :]          # -> [1, N]
        elif snaps.shape[-1] != x.size and snaps.shape[0] == x.size:
            snaps = snaps.T                 # -> [T, N]

    losses = d.get(key_losses, None)
    if losses is not None:
        losses = np.asarray(losses)
        if losses.ndim == 2:                # PINN: [epochs, 4]
            losses = losses[:, 0]           # total

    epochs = d.get("snaps_epochs", None)
    if epochs is not None:
        epochs = np.asarray(epochs).ravel()

    return x, snaps, losses, epochs

def _interp_to(x_src, y_src, x_tgt):
    x_src = np.asarray(x_src).ravel()
    x_tgt = np.asarray(x_tgt).ravel()
    y_src = np.asarray(y_src).ravel()

    # If grids match in length and values, return as-is
    if x_src.shape == x_tgt.shape and np.allclose(x_src, x_tgt, rtol=1e-8, atol=1e-12):
        return y_src

    # Ensure x_src is increasing for np.interp
    order = np.argsort(x_src)
    return np.interp(x_tgt, x_src[order], y_src[order])

def _pad_index(i, T):
    # clamp index to last frame if i exceeds sequence length
    return min(i, T - 1)

def _frame_epoch(i, snaps_epochs, total_epochs, T):
    """Map frame index -> epoch number.
    If snaps_epochs available, use it; otherwise interpolate linearly to total_epochs.
    """
    if snaps_epochs is not None and len(snaps_epochs) > 0:
        i = min(i, len(snaps_epochs) - 1)
        return int(snaps_epochs[i])
    # fallback: proportional mapping
    return int(round((i + 1) / T * total_epochs))

def _slice_upto_epoch(losses, epoch):
    """Return n where losses[0..n-1] corresponds to epochs 1..n (typical logging).
    Clamp to array length.
    """
    if losses is None:
        return 0
    n = min(epoch, len(losses))
    return n


def animate_joint(
    pinn_log: str,
    bb_log: str,
    fem_csv: str,
    out_path: str,
    title: str = "Training dynamics",
    fps: int = 8,
    snap_stride: int | None = None,
    ylim: tuple[float, float] | None = None,
    annotate: str | None = None,
    mp4: bool = True
):
    _load_style()

    # --- load all data
    xf, uf = _load_fem(fem_csv)

    xp, snaps_p, losses_p, ep_p = _load_log(pinn_log)     # PINN
    xb, snaps_b, losses_b, ep_b = _load_log(bb_log)       # BB

    if snaps_p is None or len(snaps_p) == 0:
        raise RuntimeError(f"No PINN snapshots in {pinn_log}")
    if snaps_b is None or len(snaps_b) == 0:
        raise RuntimeError(f"No BB snapshots in {bb_log}")

    if snap_stride:
        snaps_p, snaps_b = snaps_p[::snap_stride], snaps_b[::snap_stride]
        ep_p = ep_p[::snap_stride] if ep_p is not None else None
        ep_b = ep_b[::snap_stride] if ep_b is not None else None

    Tp, Tb = len(snaps_p), len(snaps_b)
    T = max(Tp, Tb)  # number of frames; the shorter stream holds its last frame

    E_p = len(losses_p) if losses_p is not None else 0  # total epochs for PINN
    E_b = len(losses_b) if losses_b is not None else 0  # total epochs for BB

    # common x grid for plotting (use FEM grid as canonical)
    x_plot = xf

    # pre-resample all snapshots to x_plot for speed
    snaps_p_rs = np.stack([_interp_to(xp, s, x_plot) for s in snaps_p])
    snaps_b_rs = np.stack([_interp_to(xb, s, x_plot) for s in snaps_b])

    # --- figure layout
    fig, (ax_pred, ax_loss) = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
    fig.suptitle(title)

    # predictions panel
    ax_pred.plot(xf, uf, c="gray", lw=2.8, alpha=0.85, label="FEM")
    lp_pinn, = ax_pred.plot(x_plot, snaps_p_rs[0], color='blue', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='blue',
                            markeredgewidth=0.65, linewidth=0, alpha=1.0, label="PINN")
    lp_bb,   = ax_pred.plot(x_plot, snaps_b_rs[0], color='orange', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='orange',
                            markeredgewidth=0.65, linewidth=0, alpha=1.0, label="BB")
    ax_pred.set_ylabel("u(x)")
    ax_pred.legend()
    if ylim is not None:
        ax_pred.set_ylim(*ylim)
    if annotate:
        ax_pred.text(0.99, 0.04, annotate, ha="right", va="bottom", transform=ax_pred.transAxes, alpha=0.85)

    # losses panel
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("total loss")
    ax_loss.set_yscale("log")
    ll_pinn, = ax_loss.plot([], [], c="blue", label=r"$\text{PINN} = $"+r"$\mathcal{L}_{data} + \mathit{\lambda_{1}}\mathcal{L}_{PDE} + \mathit{\lambda_{2}}\mathcal{L}_{BC}$")
    ll_bb,   = ax_loss.plot([], [], c="orange", label=r"$\text{BB} = $"+r"$\mathcal{L}_{data}$")

    tip_pinn, = ax_loss.plot([], [], "o", ms=8, mfc="blue", mec="white", mew=1.5, zorder=5, label="_nolegend_")
    tip_bb,   = ax_loss.plot([], [], "o", ms=8, mfc="orange", mec="white", mew=1.5, zorder=5, label="_nolegend_")

    ll_pinn_shadow, = ax_loss.plot([], [], c="blue", lw=2.0, alpha=0.25, label="_nolegend_")
    ll_bb_shadow,   = ax_loss.plot([], [], c="orange", lw=2.0, alpha=0.25, label="_nolegend_")

    ax_loss.legend()

    # epoch labels
    def epoch_label(i, ep_array, default=None):
        if ep_array is None: return default if default is not None else (i + 1)
        idx = _pad_index(i, len(ep_array))
        return int(ep_array[idx])

    def update(i):
        ip = _pad_index(i, Tp)
        ib = _pad_index(i, Tb)

        # predictions
        lp_pinn.set_ydata(snaps_p_rs[ip])
        lp_bb.set_ydata(snaps_b_rs[ib])
        ax_pred.set_title(f"Prediction evolution | epoch: {epoch_label(i, ep_p)}")

        # Epochs for this frame (from snaps_epochs if present; else proportional)
        ep_now_p = _frame_epoch(i, ep_p, E_p, T)
        ep_now_b = _frame_epoch(i, ep_b, E_b, T)

        ax_pred.set_title(
            f"Prediction evolution | epoch: {ep_now_p}"
        )

        # --- losses: plot up to the current epoch for each model ---
        if losses_p is not None and E_p > 0:
            n = _slice_upto_epoch(losses_p, ep_now_p)
            xs = np.arange(1, n + 1)
            ll_pinn.set_data(xs, losses_p[:n])
            ll_pinn_shadow.set_data(xs, losses_p[:n])      # if you added the shadow
            tip_pinn.set_data([xs[-1]], [max(losses_p[n-1], 1e-12)]) if n > 0 else tip_pinn.set_data([], [])

        if losses_b is not None and E_b > 0:
            n = _slice_upto_epoch(losses_b, ep_now_b)
            xs = np.arange(1, n + 1)
            ll_bb.set_data(xs, losses_b[:n])
            ll_bb_shadow.set_data(xs, losses_b[:n])        # if you added the shadow
            tip_bb.set_data([xs[-1]], [max(losses_b[n-1], 1e-12)]) if n > 0 else tip_bb.set_data([], [])

        # Fix the x‑axis to total max epochs so the scale doesn't change frame-to-frame
        xmax = max(E_p, E_b) if max(E_p, E_b) > 0 else T
        ax_loss.set_xlim(1, xmax)
        ax_loss.relim(); ax_loss.autoscale_view(scalex=False, scaley=True)

        return lp_pinn, lp_bb, ll_pinn, ll_bb, tip_pinn, tip_bb  # + shadows if present

    ani = FuncAnimation(fig, update, frames=T, interval=max(30, int(1000/fps)), blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GIF
    ani.save(str(out_path.with_suffix(".gif")), writer=PillowWriter(fps=fps))
    # MP4 (optional)
    if mp4:
        try:
            ani.save(str(out_path.with_suffix(".mp4")), writer="ffmpeg", dpi=180)
        except Exception as e:
            print("[animate_joint] MP4 export skipped (no ffmpeg?):", e)
    print(f"[animate_joint] saved {out_path.with_suffix('.gif')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True, help="PINN train log npz (with snaps, losses)")
    ap.add_argument("--bb",   required=True, help="BB train log npz (with snaps, losses)")
    ap.add_argument("--fem",  required=True, help="FEM CSV (x,u)")
    ap.add_argument("--out",  default="data/outputs/anim_joint")
    ap.add_argument("--title", default="Training dynamics")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th snapshot frame")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="y-limits for prediction panel, e.g. --ylim 0 1")
    ap.add_argument("--annotate", type=str, default=None, help="small note on prediction panel (e.g., 'P outside BB range')")
    ap.add_argument("--no-mp4", action="store_true")
    args = ap.parse_args()

    animate_joint(
        pinn_log=args.pinn,
        bb_log=args.bb,
        fem_csv=args.fem,
        out_path=args.out,
        title=args.title,
        fps=args.fps,
        snap_stride=None if args.stride <= 1 else args.stride,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        annotate=args.annotate,
        mp4=not args.no_mp4
    )

if __name__ == "__main__":
    main()
