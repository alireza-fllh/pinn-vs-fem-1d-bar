"""
Advanced metric visualization for parameter sweep results.

Processes and visualizes comparative performance metrics from systematic studies,
including data efficiency analysis, noise robustness, and reliability assessment.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("assets/figstyle.mplstyle")

import numpy as np


def load_csv(path:str):
    """Load CSV data with proper handling of mixed data types."""
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return arr

def lineplot(ax, xs, ys, label):
    """Create line plot with markers, properly sorted by x-values."""
    xs = np.array(xs, dtype=float); ys = np.array(ys, dtype=float)
    idx = np.argsort(xs); xs, ys = xs[idx], ys[idx]
    ax.plot(xs, ys, marker="o", label=label)

def main():
    """
    Generate comprehensive metric visualization plots.

    Creates three types of analysis plots:
    1. Data efficiency: Error vs dataset size
    2. Noise robustness: Error vs label noise level
    3. Reliability: Mean±std performance comparison
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/outputs/metrics.csv")
    ap.add_argument("--out", default="data/outputs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load_csv(args.csv)

    # Split
    data_eff = data[data["exp"]=="data_eff"]
    noise    = data[data["exp"]=="noise"]

    # ----- Error vs M (fix sigma = median) -----
    fig, ax = plt.subplots()
    for model in ["BB","PINN"]:
        d = data_eff[data_eff["model"]==model]
        # mean over seeds for each M
        byM = defaultdict(list)
        for row in d:
            byM[row["M"]].append(float(row["L2"]))
        Ms, means = [], []
        for M, vals in byM.items():
            Ms.append(float(M)); means.append(np.mean(vals))
        lineplot(ax, Ms, means, model)
    ax.set_xlabel("samples per config (M)"); ax.set_ylabel("L2 error"); ax.set_yscale("log")
    ax.set_title("Data efficiency"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "err_vs_M.png", dpi=200)

    # ----- Error vs sigma (fix M = median) -----
    fig, ax = plt.subplots()
    for model in ["BB","PINN"]:
        d = noise[noise["model"]==model]
        byS = defaultdict(list)
        for row in d:
            byS[row["sigma"]].append(float(row["L2"]))
        Ss, means = [], []
        for s, vals in byS.items():
            Ss.append(float(s)); means.append(np.mean(vals))
        lineplot(ax, Ss, means, model)
    ax.set_xlabel("label noise σ"); ax.set_ylabel("L2 error"); ax.set_yscale("log")
    ax.set_title("Noise robustness"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "err_vs_sigma.png", dpi=200)

    # ----- Reliability bar (mean±std across seeds) -----
    # pick one operating point: M=median(Ms in data_eff), sigma=median(sigmas in noise)
    Mstar = np.median([float(m) for m in np.unique(data_eff["M"])])
    sstar = np.median([float(s) for s in np.unique(noise["sigma"])])
    # collect runs at those settings
    mask = (data["M"].astype(float)==Mstar) & (data["sigma"].astype(float)==sstar)
    d = data[mask]
    labels, means, stds = [], [], []
    for model in ["BB","PINN"]:
        vals = [float(r["L2"]) for r in d if r["model"]==model]
        labels.append(model); means.append(np.mean(vals)); stds.append(np.std(vals))
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("L2 error"); ax.set_title(f"Reliability @ M={int(Mstar)}, σ={sstar:.3f}")
    fig.tight_layout(); fig.savefig(out / "reliability_bar.png", dpi=200)

    print(f"[plot] wrote {out}/err_vs_M.png, err_vs_sigma.png, reliability_bar.png")

if __name__ == "__main__":
    main()
