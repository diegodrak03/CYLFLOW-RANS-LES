"""Discrete operators on the staggered mesh."""
from cylflow.backend import xp

def col(a):
    """Turns a 1D radial array (N,) into (N, 1, 1) so it combines with 3D arrays."""
    return a[:, None, None]


def divergence(ur, ut, w, g):
    """Divergence at the cell centers. Inputs on faces, output (Nr, Nth, Nz)."""

    div_z = w - xp.roll(w, 1, axis=2) /g.dz                          # w[k] - w[k-1]
    
    flux = col(g.rf) * ur
    div_r = (flux[1:] - flux[:-1]) / col(g.rc * g.dr)

    div_th = (ut - xp.roll(ut, 1, axis=1)) / col(g.rc * g.dth)

    return div_r + div_th + div_z


def gradient(p, g):
    """Gradient of a cell-centered field. Returns (gr, gt, gz) on the faces.

    gr has shape (Nr+1, Nth, Nz) with gr[0] = gr[-1] = 0.
    """

    gr_r = (p[:-1] - p[1:]) / col(g.dr)      

    raise NotImplementedError


def laplacian(p, g):
    """Laplacian = divergence(gradient(p))."""
    return divergence(*gradient(p, g), g)