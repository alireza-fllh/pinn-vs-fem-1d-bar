import numpy as np

from src.core import FEMConfig, solve_1d_bar, BCSpec


def test_zero_load_is_zero_displacement():
    """Test that zero loading produces zero displacement."""
    cfg = FEMConfig(N=10, E_fn=lambda x: np.ones_like(x), A_fn=lambda x: np.ones_like(x),
                    body_force_fn=lambda x: np.zeros_like(x))
    bc_left = BCSpec(kind="dirichlet", u=0.0)
    bc_right = BCSpec(kind="neumann", P=0.0)
    x, u = solve_1d_bar(cfg, bc_left, bc_right)
    assert np.allclose(u, 0.0, atol=1e-10)


if __name__ == "__main__":
    test_zero_load_is_zero_displacement()
    print("✅ Zero load test passed.")
