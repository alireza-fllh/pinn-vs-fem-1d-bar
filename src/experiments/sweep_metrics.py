from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.core import BCSpec, FEMConfig, PINNConfig, PINNTrainer, solve_1d_bar
from src.training import BBConfig, BBNet, train_bb

from .data_gen import gen_dataset


def fem_ref(P: float, N:int=400) -> Tuple[np.ndarray, np.ndarray]:
    cfg = FEMConfig(N=N)
    x, u = solve_1d_bar(cfg,
                        BCSpec(kind="dirichlet", u=0.0),
                        BCSpec(kind="neumann", P=P))
    return x, u


def l2_on_grid(x_ref,u_ref, x_pred,u_pred)->float:
    u_ref_on_pred = np.interp(x_pred, x_ref, u_ref)
    return float(np.sqrt(np.mean((u_pred - u_ref_on_pred)**2)))


def train_eval_bb(dataset_npz: str, P_eval: float, seed:int,
                  epochs:int, batch:int, lr:float, width:int, depth:int,
                  outdir: Path)->float:
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"bb_seed{seed}.pt"
    log_path   = outdir / f"bb_seed{seed}_trainlog.npz"

    cfg = BBConfig(lr=lr, epochs=epochs, batch=batch, width=width, depth=depth,
                   seed=seed, p_eval=P_eval, snap_every=max(epochs//20,1))
    train_bb(dataset_npz, cfg, str(model_path), str(log_path))

    # eval on dense grid
    model = BBNet(in_dim=2); model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    x = torch.linspace(0,1,401).reshape(-1,1)
    X = torch.cat([x, torch.full_like(x, P_eval)], dim=1)

    with torch.no_grad(): u_bb = model(X).cpu().numpy().squeeze()
    xf, uf = fem_ref(P_eval)
    return l2_on_grid(xf, uf, x.numpy().squeeze(), u_bb)


def train_eval_pinn(P_eval: float, seed:int, epochs:int, w_bc:float, lr:float, n_col:int,
                    outdir: Path) -> float:
    outdir.mkdir(parents=True, exist_ok=True)

    # PINN setup for tip_load with Dirichlet(0) at x=0 and Neumann(P_eval) at x=L
    f_fn  = lambda x: torch.zeros_like(x)
    E_fn  = lambda x: torch.ones_like(x)
    A_fn  = lambda x: torch.ones_like(x)
    cfg = PINNConfig(case="tip_load", epochs=epochs, w_bc=w_bc, lr=lr,
                     n_collocations=n_col, normalize_x=True, normalize_u=True, seed=seed)
    trainer = PINNTrainer(cfg, E_fn=E_fn, A_fn=A_fn, f_fn=f_fn,
                          bc_left=BCSpec(kind="dirichlet", u=0.0),
                          bc_right=BCSpec(kind="neumann", P=P_eval))

    # --- training loop with loss logging ---
    losses = []
    for _ in range(cfg.epochs):
        loss, *_ = trainer.step()
        losses.append(loss)

    # --- predict on dense grid ---
    x = np.linspace(0,1,401).astype(np.float32)[:,None]
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        x_in = (x_t - 0.5)/0.5 if cfg.normalize_x else x_t
        u_hat = trainer.model(x_in).cpu().numpy().squeeze()
        u_pinn = trainer.u_scale * u_hat

    # --- reference FEM solution ---
    xf, uf = fem_ref(P_eval)
    L2 = l2_on_grid(xf, uf, x.squeeze(), u_pinn)

    # --- SAVE everything ---
    np.savez(outdir / "results.npz",
             losses=np.array(losses),
             x=x.squeeze(),
             u_pinn=u_pinn,
             u_fem=uf,
             xf=xf,
             L2=L2,
             P=P_eval,
             seed=seed)

    return L2


def main():
    ap = argparse.ArgumentParser()
    # sweeps
    ap.add_argument("--P", type=float, default=0.6, help="evaluation P (in-range)")
    ap.add_argument("--M-list", type=int, nargs="+", default=[5,10,20,40,80])
    ap.add_argument("--sigma-list", type=float, nargs="+", default=[0.0,0.01,0.03,0.05,0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    # dataset configs
    ap.add_argument("--n-configs", type=int, default=40, help="distinct P samples for dataset")
    # BB hyperparams
    ap.add_argument("--bb-epochs", type=int, default=4000)
    ap.add_argument("--bb-batch", type=int, default=256)
    ap.add_argument("--bb-lr", type=float, default=1e-3)
    ap.add_argument("--bb-width", type=int, default=64)
    ap.add_argument("--bb-depth", type=int, default=4)
    # PINN hyperparams
    ap.add_argument("--pinn-epochs", type=int, default=4000)
    ap.add_argument("--pinn-wbc", type=float, default=100.0)
    ap.add_argument("--pinn-lr", type=float, default=1e-3)
    ap.add_argument("--pinn-ncol", type=int, default=200)
    # io
    ap.add_argument("--supervised-out", type=str, default="data/supervised")
    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ----- Data-efficiency sweep (vary M, fix sigma at median) -----
    sigma_mid = args.sigma_list[len(args.sigma_list)//2]
    for M in args.M_list:
        for seed in args.seeds:
            npz = gen_dataset(n_configs=args.n_configs, n_points=M,
                              P_low=0.2, P_high=1.0, sigma=sigma_mid,
                              seed=seed, outdir=args.supervised_out)
            l2_bb = train_eval_bb(npz, args.P, seed,
                                  args.bb_epochs, args.bb_batch, args.bb_lr,
                                  args.bb_width, args.bb_depth,
                                  outdir / f"bb_M{M}_s{sigma_mid:.3f}")
            l2_pinn = train_eval_pinn(args.P, seed,
                                      args.pinn_epochs, args.pinn_wbc, args.pinn_lr, args.pinn_ncol,
                                      outdir / f"pinn_M{M}_s{sigma_mid:.3f}")
            rows.append(["data_eff", args.P, M, sigma_mid, seed, "BB",   l2_bb])
            rows.append(["data_eff", args.P, M, sigma_mid, seed, "PINN", l2_pinn])

    # ----- Noise sweep (vary sigma, fix M at median) -----
    M_mid = args.M_list[len(args.M_list)//2]
    for sigma in args.sigma_list:
        for seed in args.seeds:
            npz = gen_dataset(n_configs=args.n_configs, n_points=M_mid,
                              P_low=0.2, P_high=1.0, sigma=sigma,
                              seed=seed, outdir=args.supervised_out)
            l2_bb = train_eval_bb(npz, args.P, seed,
                                  args.bb_epochs, args.bb_batch, args.bb_lr,
                                  args.bb_width, args.bb_depth,
                                  outdir / f"bb_M{M_mid}_s{sigma:.3f}")
            l2_pinn = train_eval_pinn(args.P, seed,
                                      args.pinn_epochs, args.pinn_wbc, args.pinn_lr, args.pinn_ncol,
                                      outdir / f"pinn_M{M_mid}_s{sigma:.3f}")
            rows.append(["noise", args.P, M_mid, sigma, seed, "BB",   l2_bb])
            rows.append(["noise", args.P, M_mid, sigma, seed, "PINN", l2_pinn])

    # Save CSV (tidy)
    csv = outdir / "metrics.csv"
    header = "exp,P,M,sigma,seed,model,L2"
    np.savetxt(csv, np.array(rows, dtype=object), fmt="%s", delimiter=",", header=header, comments="")
    print(f"[sweep] wrote {csv}")

if __name__ == "__main__":
    main()
