import numpy as np

from src.core import FEMConfig, solve_1d_bar, BCSpec


def test_fem_shapes():
    """Test that FEM solver returns correct array shapes."""
    cfg = FEMConfig(N=10, E_fn=lambda x: np.ones_like(x), A_fn=lambda x: np.ones_like(x),
                    body_force_fn=lambda x: np.ones_like(x))
    bc_left = BCSpec(kind="dirichlet", u=0.0)
    bc_right = BCSpec(kind="neumann", P=0.0)
    x, u = solve_1d_bar(cfg, bc_left, bc_right)
    assert x.shape == u.shape
    assert x.ndim == 1
    assert len(x) == 11 # N+1 nodes


if __name__ == "__main__":
    test_fem_shapes()
    print("✅ FEM shapes test passed.")
