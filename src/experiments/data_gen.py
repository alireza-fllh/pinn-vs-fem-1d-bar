"""
Dataset generation for supervised learning experiments.

Generates training datasets by sampling FEM solutions across parameter ranges
with optional label noise for robustness testing.

Author: Alireza Fallahnejad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core import BCSpec, FEMConfig, solve_1d_bar


def gen_dataset(n_configs: int, n_points: int, P_low: float, P_high: float,
                sigma: float, seed: int, outdir: str):
    """
    Generate supervised learning dataset from FEM solutions.

    Creates a dataset by sampling FEM solutions across a parameter range,
    suitable for training black-box neural networks.

    Args:
        n_configs: Number of distinct parameter configurations
        n_points: Number of spatial points sampled per configuration
        P_low: Lower bound for traction parameter P
        P_high: Upper bound for traction parameter P
        sigma: Standard deviation of Gaussian noise added to labels
        seed: Random seed for reproducibility
        outdir: Output directory for dataset files

    Returns:
        Path to saved dataset file

    Dataset format:
        - X: Input features [x_coordinate, P_parameter] shape (N, 2)
        - Y: Target displacements with optional noise, shape (N, 1)
        - Ps: Parameter values used, shape (n_configs,)
        - meta: Dictionary with generation parameters
    """
    rng = np.random.default_rng(seed)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    X_list, Y_list, Ps = [], [], []
    for i in range(n_configs):
        P = float(rng.uniform(P_low, P_high))
        # FEM (fine mesh for ground truth)
        cfg = FEMConfig(N=200)  # dense reference
        bcL = BCSpec(kind="dirichlet", u=0.0)
        bcR = BCSpec(kind="neumann", P=P)
        x_full, u_full = solve_1d_bar(cfg, bc_left=bcL, bc_right=bcR)

        # sample n_points from the reference
        idx = rng.choice(len(x_full), size=n_points, replace=False)
        xs = x_full[idx][:, None]
        us = u_full[idx][:, None]

        # add Gaussian label noise (on displacement)
        if sigma > 0.0:
            us = us + rng.normal(0.0, sigma, size=us.shape)

        # inputs for the black box (BB) model: [x, P]
        X = np.hstack([xs, np.full_like(xs, P)])
        X_list.append(X); Y_list.append(us); Ps.append(P)

    X_all = np.vstack(X_list).astype(np.float32)  # (n_configs*n_points, 2)
    Y_all = np.vstack(Y_list).astype(np.float32)  # (n_configs*n_points, 1)
    Ps = np.array(Ps, dtype=np.float32)

    tag = f"P_{P_low:.2f}_{P_high:.2f}_N{n_configs}_M{n_points}_sigma{sigma:.3f}_seed{seed}"
    npz_path = out / f"dataset_{tag}.npz"
    np.savez(npz_path, X=X_all, Y=Y_all, Ps=Ps,
             meta=dict(P_low=P_low, P_high=P_high, n_configs=n_configs,
                       n_points=n_points, sigma=sigma, seed=seed))
    print(f"[data_gen] Saved {npz_path}")
    return str(npz_path)

def main():
    """Command-line interface for dataset generation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=40)
    ap.add_argument("--n-points", type=int, default=40)
    ap.add_argument("--P-low", type=float, default=0.2)
    ap.add_argument("--P-high", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.02, help="noise std added to u labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/supervised")
    args = ap.parse_args()
    gen_dataset(args.n_configs, args.n_points, args.P_low, args.P_high,
                args.sigma, args.seed, args.out)

if __name__ == "__main__":
    main()
