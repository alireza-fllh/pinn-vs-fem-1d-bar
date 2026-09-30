"""FEM against closed-form solutions of the 1D bar (E = A = 1, L = 1)."""
import numpy as np

from src.core import FEMConfig, solve_1d_bar, BCSpec


def ONE(x):
    return np.ones_like(x)


def ZERO(x):
    return np.zeros_like(x)


def _solve(f, bc_left, bc_right, N=40):
    cfg = FEMConfig(N=N, E_fn=ONE, A_fn=ONE, body_force_fn=f)
    return solve_1d_bar(cfg, bc_left, bc_right)


def test_tip_load_is_linear():
    """u(0) = 0, EA u'(1) = P  ->  u = P x"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=0.0), BCSpec(kind="neumann", P=0.6))
    assert np.allclose(u, 0.6 * x, atol=1e-12)


def test_uniform_body_force():
    """-u'' = 1, u(0) = 0, u'(1) = 0  ->  u = x - x^2 / 2"""
    x, u = _solve(ONE, BCSpec(kind="dirichlet", u=0.0), BCSpec(kind="neumann", P=0.0))
    assert np.allclose(u, x - x**2 / 2, atol=1e-12)


def test_neumann_on_the_left_end():
    """EA u'(0) = P, u(1) = 0  ->  u = P (x - 1)"""
    x, u = _solve(ZERO, BCSpec(kind="neumann", P=0.6), BCSpec(kind="dirichlet", u=0.0))
    assert np.allclose(u, 0.6 * (x - 1.0), atol=1e-12)


def test_robin_on_the_right_end():
    """u(0) = 0, 2 u(1) + EA u'(1) = 1  ->  u = x / 3"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=0.0),
                  BCSpec(kind="robin", alpha=2.0, beta=1.0, g=1.0))
    assert np.allclose(u, x / 3.0, atol=1e-12)


def test_robin_on_the_left_end():
    """2 u(0) + EA u'(0) = 1, u(1) = 0  ->  u = 1 - x"""
    x, u = _solve(ZERO, BCSpec(kind="robin", alpha=2.0, beta=1.0, g=1.0),
                  BCSpec(kind="dirichlet", u=0.0))
    assert np.allclose(u, 1.0 - x, atol=1e-12)


def test_robin_with_beta_not_one():
    """u(0) = 0, 2 u(1) + 2 EA u'(1) = 1  ->  u = x / 4"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=0.0),
                  BCSpec(kind="robin", alpha=2.0, beta=2.0, g=1.0))
    assert np.allclose(u, x / 4.0, atol=1e-12)


def test_robin_with_beta_zero_is_dirichlet():
    """u(0) = 0, 2 u(1) = 1 (BCSpec default beta = 0)  ->  u = x / 2"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=0.0), BCSpec(kind="robin", alpha=2.0, g=1.0))
    assert np.allclose(u, x / 2.0, atol=1e-12)


def test_nonzero_dirichlet_on_both_ends():
    """u(0) = 1, u(1) = 0  ->  u = 1 - x"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=1.0), BCSpec(kind="dirichlet", u=0.0))
    assert np.allclose(u, 1.0 - x, atol=1e-12)


def test_nonzero_dirichlet_with_tip_load():
    """u(0) = 1, EA u'(1) = P  ->  u = 1 + P x"""
    x, u = _solve(ZERO, BCSpec(kind="dirichlet", u=1.0), BCSpec(kind="neumann", P=0.6))
    assert np.allclose(u, 1.0 + 0.6 * x, atol=1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("✅ FEM analytic tests passed.")
