from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.core import BCSpec, PINNConfig, PINNTrainer


def bc_from_args(side: str, args: argparse.Namespace) -> BCSpec:
    if side == "left":
        kind = args.left_type
        u, P = args.u0, args.P0
        alpha, beta, g = args.alpha0, args.beta0, args.g0
    else:
        kind = args.right_type
        u, P = args.uL, args.PL
        alpha, beta, g = args.alphaL, args.betaL, args.gL

    if kind == "dirichlet":
        return BCSpec(kind="dirichlet", u=u)
    elif kind == "neumann":
        return BCSpec(kind="neumann", P=P)
    elif kind == "robin":
        return BCSpec(kind="robin", alpha=alpha, beta=beta, g=g)
    else:
        raise ValueError(f"Unknown BC kind for {side}: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["body_force", "tip_load", "hetero"], default="body_force")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--w_bc", type=float, default=50.0)

    # BCs
    ap.add_argument("--left-type",  choices=["dirichlet","neumann","robin"], default="dirichlet")
    ap.add_argument("--right-type", choices=["dirichlet","neumann","robin"], default=None)

    ap.add_argument("--u0", type=float, default=0.0)
    ap.add_argument("--P0", type=float, default=0.0)
    ap.add_argument("--alpha0", type=float, default=0.0)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--g0", type=float, default=0.0)

    ap.add_argument("--uL", type=float, default=0.0)
    ap.add_argument("--PL", type=float, default=None)
    ap.add_argument("--alphaL", type=float, default=0.0)
    ap.add_argument("--betaL", type=float, default=1.0)
    ap.add_argument("--gL", type=float, default=0.0)

    # normalization flags
    ap.add_argument("--normalize_x", action="store_true"); ap.set_defaults(normalize_x=True)
    ap.add_argument("--no-normalize_x", dest="normalize_x", action="store_false")
    ap.add_argument("--normalize_u", action="store_true"); ap.set_defaults(normalize_u=True)
    ap.add_argument("--no-normalize_u", dest="normalize_u", action="store_false")

    ap.add_argument("--out", type=str, default="data/outputs")
    args = ap.parse_args()

    # Defaults by case
    if args.case == "body_force":
        f_fn = lambda x: torch.ones_like(x)
        E_fn = lambda x: torch.ones_like(x)
        A_fn = lambda x: torch.ones_like(x)
        default_right = ("neumann", 0.0)
    elif args.case == "tip_load":
        f_fn = lambda x: torch.zeros_like(x)
        E_fn = lambda x: torch.ones_like(x)
        A_fn = lambda x: torch.ones_like(x)
        default_right = ("neumann", 1.0)
    else:  # hetero
        import math
        f_fn = lambda x: 2.0 * torch.sin(2 * math.pi * x)
        E_fn = lambda x: torch.where(x < 0.5, torch.ones_like(x), 3.0*torch.ones_like(x))
        A_fn = lambda x: 1.0 + 0.5 * x
        default_right = ("neumann", 0.0)

    if args.right_type is None:
        args.right_type = default_right[0]
        if args.PL is None:
            args.PL = default_right[1]

    bc_left = bc_from_args("left", args)
    bc_right = bc_from_args("right", args)
    print(f"[PINN] BCs: left={bc_left} | right={bc_right}")

    cfg = PINNConfig(
        case=args.case, epochs=args.epochs, w_bc=args.w_bc,
        normalize_x=args.normalize_x, normalize_u=args.normalize_u
    )
    trainer = PINNTrainer(cfg, E_fn=E_fn, A_fn=A_fn, f_fn=f_fn, bc_left=bc_left, bc_right=bc_right)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    SNAP_EVERY = 200  # save a prediction every 200 epochs
    losses = []
    snapshots = []     # list of arrays u(x) over time
    snap_epochs = []   # matching epoch numbers

    # fixed grid for visualization
    x_plot = np.linspace(0.0, 1.0, 401).astype(np.float32)
    x_plot_t = torch.from_numpy(x_plot[:, None])


    for ep in range(cfg.epochs):
        tot, lpde, lL, lR = trainer.step()
        losses.append([tot, lpde, lL, lR])

        if (ep + 1) % SNAP_EVERY == 0:
            with torch.no_grad():
                x_in = (x_plot_t - 0.5)/0.5 if cfg.normalize_x else x_plot_t
                u_hat = trainer.model(x_in).cpu().numpy().squeeze()
                u = trainer.u_scale * u_hat
            snapshots.append(u)
            snap_epochs.append(ep + 1)
            print(f"[PINN] snapshot @ epoch {ep+1}: total={tot:.3e}")

    # a) regular final predictions
    np.savez(outdir / f"pinn_{args.case}.npz",
            x=x_plot, u=snapshots[-1] if snapshots else np.array([]),
            losses=np.array(losses, dtype=np.float32))

    # b) training log with snapshots for animation/comparison
    np.savez(outdir / f"pinn_{args.case}_trainlog.npz",
            losses=np.array(losses, dtype=np.float32),
            x=x_plot,
            snaps_epochs=np.array(snap_epochs, dtype=np.int32),
            snaps=np.stack(snapshots) if snapshots else np.zeros((0, x_plot.size), dtype=np.float32))
    print(f"[PINN] saved train log → {outdir}/pinn_{args.case}_trainlog.npz")


if __name__ == "__main__":
    main()
