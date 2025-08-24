from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# --- style (robust relative load) ---
STYLE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "figstyle.mplstyle"))
if os.path.exists(STYLE): plt.style.use(STYLE)

def _load_fem(csv_path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return arr[:,0], arr[:,1]

def _load_pinn(log_npz):
    d = np.load(log_npz)
    # accept either 'snaps' (training log) or 'u' (final)
    if "snaps" in d and len(d["snaps"])>0:
        x = d["x"]; u = d["snaps"][-1]
        losses = d["losses"][:,0] if "losses" in d else None
    else:
        x = d["x"]; u = d["u"]; losses = d["losses"][:,0] if "losses" in d else None
    return x, u, losses

def _load_bb(log_npz):
    d = np.load(log_npz)
    x = d["x"]
    if "snaps" in d and len(d["snaps"])>0:
        u = d["snaps"][-1]
    else:
        u = d["Yhat"] if "Yhat" in d else None  # optional
    losses = d["losses"] if "losses" in d else None
    return x, u, losses

def _l2(x_ref, u_ref, x_pred, u_pred):
    up = np.interp(x_pred, x_ref, u_ref)
    return float(np.sqrt(np.mean((u_pred - up)**2)))

def _savefig_all(stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(stem.with_suffix(".png"), dpi=240)
    plt.savefig(stem.with_suffix(".svg"))

def _reliability_from_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path): return None
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    # Choose the median settings present in CSV
    Ms = sorted(set(arr["M"].astype(float)))
    sigmas = sorted(set(arr["sigma"].astype(float)))
    Mstar = Ms[len(Ms)//2]; sstar = sigmas[len(sigmas)//2]
    m = (arr["M"].astype(float)==Mstar) & (arr["sigma"].astype(float)==sstar)
    subset = arr[m]
    stats = {}
    for model in ["BB","PINN"]:
        vals = [float(r["L2"]) for r in subset if r["model"]==model]
        if vals:
            stats[model] = (np.mean(vals), np.std(vals))
    return (Mstar, sstar, stats)

def make_hero(in_fem, in_pinn, in_bb, ex_fem=None, ex_pinn=None, ex_bb=None,
              metrics_csv=None, title="PINN vs BlackBox",
              out="data/outputs/hero_figure"):
    # --- Panel A: in‑range overlay ---
    xf_in, uf_in = _load_fem(in_fem)
    xp_in, up_in, lp_in = _load_pinn(in_pinn)
    xb_in, ub_in, lb_in = _load_bb(in_bb)
    l2p_in = _l2(xf_in, uf_in, xp_in, up_in)
    l2b_in = _l2(xf_in, uf_in, xb_in, ub_in)

    # --- Panel B: extrapolation (optional) ---
    have_ex = ex_fem and ex_pinn and ex_bb and all(os.path.exists(p) for p in [ex_fem, ex_pinn, ex_bb])
    if have_ex:
        xf_ex, uf_ex = _load_fem(ex_fem)
        xp_ex, up_ex, lp_ex = _load_pinn(ex_pinn)
        xb_ex, ub_ex, lb_ex = _load_bb(ex_bb)
        l2p_ex = _l2(xf_ex, uf_ex, xp_ex, up_ex)
        l2b_ex = _l2(xf_ex, uf_ex, xb_ex, ub_ex)

    # --- Panel C: losses (if available) ---
    # prefer in‑range logs
    losses_p = lp_in
    losses_b = lb_in

    # --- Panel D: reliability from CSV ---
    rel = _reliability_from_csv(metrics_csv)

    # --- figure ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    fig.suptitle(title)

    # A) In‑range
    ax = axs[0,0]
    ax.plot(xf_in, uf_in,  c="gray", lw=2.8, alpha=0.85, label="FEM")
    ax.plot(xp_in, up_in, color='blue', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='blue', markeredgewidth=0.65, linewidth=0, alpha=1.0, markevery=10, label=f"PINN (L2={l2p_in:.2e})")
    ax.plot(xb_in, ub_in, color='orange', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=0.65, linewidth=0, alpha=1.0, markevery=10, label=f"BB (L2={l2b_in:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Interpolation"); ax.legend()

    # B) Extrapolation (if provided)
    ax = axs[0,1]
    if have_ex:
        ax.plot(xf_ex, uf_ex, c="gray", lw=2.8, alpha=0.7, label="FEM")
        ax.plot(xp_ex, up_ex, color='blue', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='blue', markeredgewidth=0.65, linewidth=0, alpha=1.0, markevery=10, label=f"PINN (L2={l2p_ex:.2e})")
        ax.plot(xb_ex, ub_ex, color='blue', marker='o', markersize=2.6, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=0.65, linewidth=0, alpha=1.0, markevery=10, label=f"BB (L2={l2b_ex:.2e})")
        ax.set_facecolor((1.0, 0.98, 0.95))
        ax.text(0.98, 0.06, "P outside BB training range", ha="right", va="bottom",
        transform=ax.transAxes, fontsize=10, alpha=0.8)
        ax.set_title("Extrapolation"); ax.legend()
    else:
        ax.axis("off"); ax.text(0.5,0.5,"Extrapolation panel\n(attach logs to enable)", ha="center", va="center")

    # C) Loss curves (log‑y)
    ax = axs[1,0]
    has_any_loss = False
    if losses_p is not None and len(losses_p)>0:
        ax.plot(np.arange(1,len(losses_p)+1), losses_p, label=r"$\text{PINN} = $"+r"$\mathcal{L}_{data} + \mathit{\lambda_{1}}\mathcal{L}_{PDE} + \mathit{\lambda_{2}}\mathcal{L}_{BC}$")
        has_any_loss = True
    if losses_b is not None and len(losses_b)>0:
        ax.plot(np.arange(1,len(losses_b)+1), losses_b, label=r"$\text{BB} = $"+r"$\mathcal{L}_{data}$")
        has_any_loss = True
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss"); ax.set_yscale("log")
    ax.set_title("Training losses")
    if has_any_loss: ax.legend()
    else:
        ax.text(0.5,0.5,"(no loss logs found)", ha="center", va="center"); ax.set(xticks=[], yticks=[])

    # D) Reliability bar (mean±std)
    ax = axs[1,1]
    if rel is not None and "BB" in rel[2] and "PINN" in rel[2]:
        labels = ["BB", "PINN"]
        means = [rel[2]["BB"][0], rel[2]["PINN"][0]]
        stds  = [rel[2]["BB"][1], rel[2]["PINN"][1]]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("L2 error")
        ax.set_title(f"Reliability @ M={int(rel[0])},  " +  r"$\sigma$" + f"={rel[1]:.3f}")
    else:
        ax.text(0.5,0.5,"(metrics.csv not provided)", ha="center", va="center"); ax.set(xticks=[], yticks=[])

    # save
    out = Path(out)
    _savefig_all(out)
    print(f"[hero_figure] saved {out.with_suffix('.png')} and {out.with_suffix('.svg')}")

def main():
    ap = argparse.ArgumentParser()
    # In‑range (required)
    ap.add_argument("--in-fem", required=True, help="CSV from run_fem (in‑range P)")
    ap.add_argument("--in-pinn", required=True, help="pinn_*_trainlog.npz for same P")
    ap.add_argument("--in-bb", required=True, help="bb_trainlog.npz with p‑eval matching P")
    # Extrapolation (optional)
    ap.add_argument("--ex-fem", default=None)
    ap.add_argument("--ex-pinn", default=None)
    ap.add_argument("--ex-bb", default=None)
    # Reliability CSV (optional)
    ap.add_argument("--metrics-csv", default=None, help="data/outputs/metrics.csv")
    ap.add_argument("--title", default="PINN vs BlackBox")
    ap.add_argument("--out", default="data/outputs/hero_figure")
    args = ap.parse_args()

    make_hero(args.in_fem, args.in_pinn, args.in_bb,
              args.ex_fem, args.ex_pinn, args.ex_bb,
              args.metrics_csv, args.title, args.out)

if __name__ == "__main__":
    main()
