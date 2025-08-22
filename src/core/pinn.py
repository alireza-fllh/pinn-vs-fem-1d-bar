"""
Physics-Informed Neural Network (PINN) implementation for 1D elastic bar problems.

Implements automatic differentiation-based solution of the PDE:
    -d/dx(E(x) * A(x) * du/dx) = f(x)
with Dirichlet, Neumann, or Robin boundary conditions.

Features:
    - Input/output normalization for stable training
    - Flexible boundary condition support
    - Automatic scaling for physical displacement units
    - Multi-component loss function (PDE + BC terms)

Author: Alireza Fallahnejad
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import BCSpec


class PinnNet(nn.Module):
    """
    Multi-layer perceptron for PINN displacement approximation.

    Uses tanh activation functions throughout for smooth derivatives
    required by the PDE residual computation.

    Args:
        in_dim: Input dimension (typically 1 for spatial coordinate)
        out_dim: Output dimension (typically 1 for displacement)
        width: Hidden layer width
        depth: Number of hidden layers
    """
    def __init__(self, in_dim=1, out_dim=1, width=64, depth=4):
        super().__init__()
        layers = []

        # input layer: 1 -> width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        # hidden layers: width -> width
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        # output layer: width -> 1
        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # ensure shape (N, 1)
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return self.net(x)


@dataclass
class PINNConfig:
    """
    PINN training configuration.

    Args:
        case: Problem case identifier
        n_collocations: Number of collocation points for PDE residual
        w_bc: Weight factor for boundary condition loss terms
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        seed: Random seed for reproducibility
        normalize_x: Whether to normalize input coordinates to [-1,1]
        normalize_u: Whether to auto-scale displacement outputs
    """
    case: str = "body_force"
    n_collocations: int = 200
    w_bc: float = 50.0
    epochs: int = 10000
    lr: float = 1e-3
    seed: int = 0
    normalize_x: bool = True
    normalize_u: bool = True


class PINNTrainer:
    """
    PINN training manager for 1D elastic bar problems.

    Handles the complete training workflow including:
    - Loss function computation (PDE + boundary conditions)
    - Automatic differentiation for physics residuals
    - Input/output normalization
    - Gradient-based optimization

    Args:
        cfg: Training configuration
        E_fn: Young's modulus function E(x)
        A_fn: Cross-sectional area function A(x)
        f_fn: Body force function f(x)
        bc_left: Left boundary condition specification
        bc_right: Right boundary condition specification
    """
    def __init__(self, cfg: PINNConfig, E_fn, A_fn, f_fn, bc_left: BCSpec, bc_right: BCSpec):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.model = PinnNet(in_dim=1, out_dim=1, width=64, depth=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # Collocation & BC points
        self.E_fn, self.A_fn, self.f_fn = E_fn, A_fn, f_fn
        self.L = 1.0
        self.x_col = torch.linspace(0.0, self.L, cfg.n_collocations, dtype=torch.float32).reshape(-1, 1)
        self.x0 = torch.tensor([[0.0]], dtype=torch.float32)
        self.xL = torch.tensor([[self.L]], dtype=torch.float32)
        self.bc_left, self.bc_right = bc_left, bc_right

        # Automatic displacement scaling for numerical stability
        if cfg.normalize_u:
            with torch.no_grad():
                xs = torch.linspace(0.0, self.L, 256).reshape(-1,1)
                EA = E_fn(xs) * A_fn(xs)
                EA_mean = EA.abs().mean().clamp_min(1e-8)
                fA_mean = (f_fn(xs) * A_fn(xs)).abs().mean()
                # rough bound (Neumann part uses PL if provided)
                P_R = (bc_right.P if bc_right.kind=="neumann" else 0.0)
                self.u_scale = float((self.L*abs(P_R))/EA_mean + (self.L**2 * fA_mean)/(2*EA_mean))
                if self.u_scale == 0.0: self.u_scale = 1.0
        else:
            self.u_scale = 1.0

    def _u_and_du(self, x):
        """Compute displacement and its derivative with proper scaling."""
        x = x.clone().requires_grad_(True)
        x_in = (x - 0.5)/0.5 if self.cfg.normalize_x else x
        u_hat = self.model(x_in)
        u = self.u_scale * u_hat
        du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        return u, du

    def _bc_loss_one_side(self, side: str) -> torch.Tensor:
        """Compute boundary condition loss for one side."""
        if side == "left":
            x = self.x0; bc = self.bc_left
        else:
            x = self.xL; bc = self.bc_right
        u, du = self._u_and_du(x)
        EA = self.E_fn(x) * self.A_fn(x)

        if bc.kind == "dirichlet":
            return (u - bc.u)**2
        elif bc.kind == "neumann":
            return (EA * du - bc.P)**2
        elif bc.kind == "robin":
            return (bc.alpha * u + bc.beta * EA * du - bc.g)**2
        else:
            raise ValueError(f"Unknown BC kind: {bc.kind}")

    def compute_loss_terms(self):
        """
        Compute all loss components.

        Returns:
            loss_pde: PDE residual loss
            l_left: Left boundary condition loss
            l_right: Right boundary condition loss
            loss_bc: Combined boundary condition loss
        """
        Res = self._compute_residual(self.model, self.x_col, self.f_fn, self.E_fn, self.A_fn,
                                     u_scale=self.u_scale, normalize_x=self.cfg.normalize_x)
        loss_pde = (Res ** 2).mean()

        # BCs
        l_left = self._bc_loss_one_side("left").mean()
        l_right = self._bc_loss_one_side("right").mean()
        loss_bc = l_left + l_right

        return loss_pde, l_left, l_right, loss_bc

    def step(self):
        self.optimizer.zero_grad()
        lpde, lL, lR, lbc = self.compute_loss_terms()
        loss = lpde + self.cfg.w_bc * lbc
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(lpde.item()), float(lL.item()), float(lR.item())

    def _compute_residual(self, network, x_col: torch.Tensor, f_fn, E_fn, A_fn, u_scale: float = 1.0, normalize_x: bool = True):
        """
        Compute PDE residual: R = -d/dx(EA du/dx) - f(x)

        Uses automatic differentiation to compute second derivatives
        required by the strong form PDE.
        """
        x = x_col.clone().requires_grad_(True)

        # optional input normalization: map x in [0,1] to [-1,1]
        x_in = (x - 0.5) / 0.5 if normalize_x else x
        u_hat = network(x_in) # network outputs normalized displacement
        u = u_scale * u_hat # physical scale
        du_dx = self.__grad(u, x)
        # Res = -d/dx(EA du/dx) - f(x)
        EA = E_fn(x) * A_fn(x)
        flux = EA * du_dx
        dflux_dx = self.__grad(flux, x)
        return -(dflux_dx) - f_fn(x)

    @staticmethod
    def __grad(outputs: torch.Tensor, inputs: torch.Tensor):
        return torch.autograd.grad(outputs, inputs, torch.ones_like(outputs), create_graph=True)[0]
