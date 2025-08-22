from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BBNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: x = x.unsqueeze(1)
        return self.net(x)


@dataclass
class BBConfig:
    lr: float = 1e-3
    epochs: int = 6000
    batch: int = 256
    width: int = 64
    depth: int = 4
    seed: int = 0
    snap_every: int = 200
    p_eval: float = 0.6  # P used for visualization snapshots

def train_bb(npz_path: str, cfg: BBConfig, out_model: str, out_log: str):
    torch.manual_seed(cfg.seed)
    data = np.load(npz_path)
    X = torch.tensor(data["X"], dtype=torch.float32)  # [x, P]
    Y = torch.tensor(data["Y"], dtype=torch.float32)  # u noisy

    model = BBNet(in_dim=2, width=cfg.width, depth=cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    # Snapshot setup (prediction on a fixed x grid for P=cfg.p_eval)
    x_plot = torch.linspace(0, 1, 401).reshape(-1, 1)
    P_plot = torch.full_like(x_plot, fill_value=cfg.p_eval)
    X_plot = torch.cat([x_plot, P_plot], dim=1)

    losses, snaps, snap_epochs = [], [], []
    for ep in range(cfg.epochs):
        # mini-batch
        idx = torch.randint(0, len(X), (min(cfg.batch, len(X)),))
        yhat = model(X[idx])
        loss = loss_fn(yhat, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

        if (ep + 1) % cfg.snap_every == 0:
            with torch.no_grad():
                u_pred = model(X_plot).cpu().numpy().squeeze()
            snaps.append(u_pred); snap_epochs.append(ep + 1)
            if (ep + 1) % 1000 == 0:
                print(f"[BB] epoch {ep+1}: loss={losses[-1]:.3e}")

    # Save model + log
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)
    print(f"[BB] Saved model to {out_model}")

    log = dict(losses=np.array(losses, dtype=np.float32),
               x=x_plot.numpy().squeeze(),
               P=float(cfg.p_eval),
               snaps=np.stack(snaps) if snaps else np.zeros((0, len(x_plot))),
               snaps_epochs=np.array(snap_epochs, dtype=np.int32))
    np.savez(out_log, **log)
    print(f"[BB] Saved train log to {out_log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to dataset npz")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--p-eval", type=float, default=0.6)
    ap.add_argument("--snap-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-out", type=str, default="data/outputs/bb_model.pt")
    ap.add_argument("--log-out", type=str, default="data/outputs/bb_trainlog.npz")
    args = ap.parse_args()

    cfg = BBConfig(lr=args.lr, epochs=args.epochs, batch=args.batch,
                   width=args.width, depth=args.depth, seed=args.seed,
                   p_eval=args.p_eval, snap_every=args.snap_every)
    train_bb(args.data, cfg, args.model_out, args.log_out)

if __name__ == "__main__":
    main()
