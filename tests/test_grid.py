"""Mesh tests. Run: pytest tests/test_grid.py -v"""
import numpy as np
import pytest

from cylflow.backend import to_numpy
from cylflow.grid import beta_from_dr_min, make_grid, radial_faces


def test_radial_faces_endpoints_and_order():
    rf = radial_faces(0.5, 25.0, 64, beta=2.5)
    assert len(rf) == 65
    assert rf[0] == pytest.approx(0.5)
    assert rf[-1] == pytest.approx(25.0)
    assert np.all(np.diff(rf) > 0), "faces must be sorted in increasing order"


def test_beta_gives_requested_first_cell():
    beta = beta_from_dr_min(0.5, 25.0, 128, dr_min=0.003)
    rf = radial_faces(0.5, 25.0, 128, beta)
    assert rf[1] - rf[0] == pytest.approx(0.003, rel=1e-6)


def test_smooth_stretching():
    g = make_grid(0.5, 25.0, 160, 256, 1, 1.0, dr_min=0.003)
    dr = to_numpy(g.dr)
    ratio = dr[1:] / dr[:-1]
    assert ratio.max() < 1.10, f"max ratio between cells {ratio.max():.3f} >= 1.10"


def test_centers_between_faces():
    g = make_grid(0.5, 25.0, 64, 32, 4, np.pi, beta=2.0)
    rf, rc = to_numpy(g.rf), to_numpy(g.rc)
    assert np.all(rc > rf[:-1]) and np.all(rc < rf[1:])
    assert np.allclose(to_numpy(g.dr), rf[1:] - rf[:-1])
    assert np.allclose(to_numpy(g.drc), rc[1:] - rc[:-1])


def test_total_volume_is_annulus_volume():
    R, Re, Lz = 0.5, 25.0, np.pi
    g = make_grid(R, Re, 64, 32, 8, Lz, beta=2.0)
    exact = np.pi * (Re**2 - R**2) * Lz
    assert float(to_numpy(g.vol).sum()) == pytest.approx(exact, rel=1e-12)


def test_angles_and_z():
    g = make_grid(0.5, 25.0, 16, 8, 4, 2.0, beta=2.0)
    assert g.dth == pytest.approx(2 * np.pi / 8)
    assert g.dz == pytest.approx(0.5)
    assert np.allclose(to_numpy(g.thf) - to_numpy(g.thc), 0.5 * g.dth)
    assert np.allclose(to_numpy(g.zf) - to_numpy(g.zc), 0.5 * g.dz)