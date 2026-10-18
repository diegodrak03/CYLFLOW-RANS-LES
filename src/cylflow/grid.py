"""Staggered polar mesh around the cylinder."""
from dataclasses import dataclass

import numpy as np

from cylflow.backend import DTYPE, xp


def radial_faces(R, R_ext, Nr, beta):
    """Radii of the Nr+1 radial faces with tanh stretching.

    Returns a NumPy (CPU) array of size Nr+1, with rf[0] = R and rf[-1] = R_ext
    exactly, sorted in increasing order.
    """
    xi = np.arange(Nr + 1) / Nr   

    return R + (R_ext - R) * (1 - np.tanh(beta * (1 - xi)) / np.tanh(beta))



def beta_from_dr_min(R, R_ext, Nr, dr_min):
    """Finds beta so that the first cell has size dr_min.

    Raises ValueError if dr_min is greater than or equal to that of a uniform mesh
    (no stretching is needed then). Accuracy: |dr_obtained - dr_min| / dr_min < 1e-6.
    """
    if dr_min >= (R_ext - R) / Nr:
        raise ValueError("dr_min is too large for the given R, R_ext, and Nr.")
    
    beta_low = 1e-6         # almost uniform mesh: first cell too big
    beta_high = 20.0    # very stretched mesh: first cell too small

    for _ in range(60):
        beta_mid = 0.5 * (beta_low + beta_high)
        rf = radial_faces(R, R_ext, Nr, beta_mid)
        dr = rf[1] - rf[0]

        if dr > dr_min:
            beta_low = beta_mid
        else:
            beta_high = beta_mid

    return 0.5 * (beta_low + beta_high)
        
        

@dataclass
class Grid:
    R: float          # cylinder radius
    R_ext: float      # outer radius
    Nr: int
    Nth: int
    Nz: int
    Lz: float
    beta: float
    rf: object        # (Nr+1,)  radial faces
    rc: object        # (Nr,)    radial centers
    dr: object        # (Nr,)    cell widths
    drc: object       # (Nr-1,)  distances between centers
    dth: float        # 2*pi/Nth
    thc: object       # (Nth,)   center angles   j*dth
    thf: object       # (Nth,)   face angles     (j+1/2)*dth
    dz: float         # Lz/Nz
    zc: object        # (Nz,)    (k+1/2)*dz
    zf: object        # (Nz,)    (k+1)*dz
    vol: object       # (Nr, Nth, Nz) volume of each cell


def make_grid(R, R_ext, Nr, Nth, Nz, Lz, dr_min=None, beta=None):
    """Builds the mesh. Give either dr_min or beta (exactly one of them).

    The Grid arrays must live in the backend (xp) with type DTYPE.
    """
    raise NotImplementedError