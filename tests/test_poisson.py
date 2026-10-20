"""Poisson solver tests. Run: pytest tests/test_poisson.py -v"""
import numpy as np
import pytest

from cylflow.backend import DTYPE, to_numpy, xp
from cylflow.grid import make_grid
from cylflow.operators import divergence, gradient, laplacian
from cylflow.poisson import PoissonSolver


@pytest.mark.parametrize("dims", [(32, 48, 1), (24, 32, 16)])
def test_recovers_exact_discrete_solution(dims):
    Nr, Nth, Nz = dims
    g = make_grid(0.5, 25.0, Nr, Nth, Nz, np.pi, dr_min=0.01)
    rng = np.random.default_rng(1)
    phi = xp.asarray(rng.standard_normal(dims), dtype=DTYPE)
    phi = phi - phi.mean()
    f = laplacian(phi, g)
    sol = PoissonSolver(g).solve(f)
    err = float(xp.abs(sol - phi).max() / xp.abs(phi).max())
    print("relative error", err)
    assert err < 1e-10


def test_projection_removes_divergence():
    Nr, Nth, Nz = 32, 48, 8
    g = make_grid(0.5, 25.0, Nr, Nth, Nz, np.pi, dr_min=0.01)
    rng = np.random.default_rng(2)
    ur = rng.standard_normal((Nr + 1, Nth, Nz))
    ur[0] = 0.0
    ur[-1] = 0.0
    ut = rng.standard_normal((Nr, Nth, Nz))
    w = rng.standard_normal((Nr, Nth, Nz))
    ur, ut, w = (xp.asarray(a, dtype=DTYPE) for a in (ur, ut, w))
    phi = PoissonSolver(g).solve(divergence(ur, ut, w, g))
    gr, gt, gz = gradient(phi, g)
    ur, ut, w = ur - gr, ut - gt, w - gz
    d = float(xp.abs(divergence(ur, ut, w, g)).max())
    u = float(max(xp.abs(ur).max(), xp.abs(ut).max(), xp.abs(w).max()))
    print("relative divergence", d / u)
    assert d / u < 1e-10