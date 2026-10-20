"""Discrete operator tests (Phase 3). Run: pytest tests/test_operators.py -v"""
import numpy as np

from cylflow.backend import DTYPE, to_numpy, xp
from cylflow.grid import make_grid
from cylflow.operators import divergence, gradient

R, R_EXT, LZ, BETA = 0.5, 3.0, 2 * np.pi, 1.5


def mesh(n):
    # Fixed beta: the mesh is refined "smoothly", so the order shows up cleanly
    return make_grid(R, R_EXT, n, n, n, LZ, beta=BETA)


def coords(a_r, a_t, a_z):
    """3D coordinate arrays built from 1D vectors (on the CPU)."""
    return np.meshgrid(to_numpy(a_r), to_numpy(a_t), to_numpy(a_z), indexing="ij")


def order(errors):
    """log2 slope between errors of meshes that double in size."""
    e = np.array(errors)
    return np.log2(e[:-1] / e[1:])


def divergence_error(n):
    g = mesh(n)
    r, t, z = coords(g.rf, g.thc, g.zc)
    ur = r**2 * np.cos(t)
    r, t, z = coords(g.rc, g.thf, g.zc)
    ut = r * np.sin(t)
    r, t, z = coords(g.rc, g.thc, g.zf)
    w = r * np.sin(z)
    d = divergence(*(xp.asarray(a, dtype=DTYPE) for a in (ur, ut, w)), g)
    r, t, z = coords(g.rc, g.thc, g.zc)
    exact = 3 * r * np.cos(t) + np.cos(t) + r * np.cos(z)
    return np.abs(to_numpy(d) - exact).max()


def test_divergence_second_order():
    errors = [divergence_error(n) for n in (16, 32, 64)]
    p = order(errors)
    print("errors:", errors, "order:", p)
    assert np.all(p > 1.8), f"observed order {p}, expected ~2"


def gradient_errors(n):
    g = mesh(n)
    r, t, z = coords(g.rc, g.thc, g.zc)
    p = xp.asarray(r**2 * np.cos(t) * np.cos(z), dtype=DTYPE)
    gr, gt, gz = (to_numpy(a) for a in gradient(p, g))
    r, t, z = coords(g.rf, g.thc, g.zc)
    e_r = np.abs(gr[1:-1] - (2 * r * np.cos(t) * np.cos(z))[1:-1]).max()
    r, t, z = coords(g.rc, g.thf, g.zc)
    e_t = np.abs(gt - (-r * np.sin(t) * np.cos(z))).max()
    r, t, z = coords(g.rc, g.thc, g.zf)
    e_z = np.abs(gz - (-r**2 * np.cos(t) * np.sin(z))).max()
    return e_r, e_t, e_z


def test_gradient_second_order():
    errs = np.array([gradient_errors(n) for n in (16, 32, 64)])
    for k, name in enumerate(("radial", "azimuthal", "axial")):
        p = order(errs[:, k])
        print(name, errs[:, k], p)
        assert np.all(p > 1.8), f"{name} gradient: order {p}"


def test_radial_gradient_zero_at_boundaries():
    g = mesh(16)
    p = xp.asarray(np.random.default_rng(0).standard_normal((16, 16, 16)), dtype=DTYPE)
    gr, _, _ = gradient(p, g)
    assert np.all(to_numpy(gr[0]) == 0) and np.all(to_numpy(gr[-1]) == 0)