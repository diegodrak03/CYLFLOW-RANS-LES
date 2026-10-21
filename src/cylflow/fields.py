"""Container for the simulation state."""
from dataclasses import dataclass

from cylflow.backend import DTYPE, xp

import numpy as np


@dataclass
class State:
    ur: object       # (Nr+1, Nth, Nz) radial velocity on radial faces
    ut: object       # (Nr,   Nth, Nz) azimuthal velocity on azimuthal faces
    w: object        # (Nr,   Nth, Nz) axial velocity on axial faces
    p: object        # (Nr,   Nth, Nz) pressure at the centers
    ut_in: object    # (Nth, Nz) u_theta at the wall r = R
    ut_out: object   # (Nth, Nz) u_theta at the outer boundary
    w_in: object     # (Nth, Nz) w at the wall
    w_out: object    # (Nth, Nz) w at the outer boundary
    nut: object      # (Nr,   Nth, Nz) turbulent viscosity (0 in laminar)
    t: float = 0.0


def zeros_state(g):
    """State with every field set to zero, with the correct shapes."""
    centers = (g.Nr, g.Nth, g.Nz)
    boundary = (g.Nth, g.Nz)

    def z(shape):
        return xp.zeros(shape, dtype=DTYPE)

    return State(ur=z((g.Nr + 1, g.Nth, g.Nz)), ut=z(centers), w=z(centers), p=z(centers),
                 ut_in=z(boundary), ut_out=z(boundary), w_in=z(boundary), w_out=z(boundary),
                 nut=z(centers))



def potential_flow(st, g, noise=0.0, seed=0):
    """Fills st with the potential flow around the cylinder (+ optional noise in ut and w).

    It does not impose no-slip: the first projection and the solver itself will do that.
    Set ur[0] = 0 (no penetration at the wall) and ut_out to the free stream.
    """
    R = g.R

    # 1. u_r on the radial faces: radius rf, angle thc
    r = g.rf[:, None, None]                          # (Nr+1, 1, 1)
    th = g.thc[None, :, None]                        # (1, Nth, 1)
    st.ur[...] = (1 - R**2 / r**2) * xp.cos(th)      # -> (Nr+1, Nth, 1), copied along z
    st.ur[0] = 0.0                                   # no penetration at the wall

    # 2. u_theta on the azimuthal faces: radius rc, angle thf
    r = g.rc[:, None, None]                          # (Nr, 1, 1)
    th = g.thf[None, :, None]                        # (1, Nth, 1)
    st.ut[...] = -(1 + R**2 / r**2) * xp.sin(th)

    # 3. no axial velocity, pressure zero
    st.w[...] = 0.0
    st.p[...] = 0.0

    # 4. boundary values
    st.ut_in[...] = 0.0                              # wall: the fluid sticks to it
    st.w_in[...] = 0.0
    st.ut_out[...] = -xp.sin(g.thf)[:, None]         # outer boundary: free stream
    st.w_out[...] = 0.0

    # 5. random noise (on the CPU, so the same seed gives the same field with NumPy and CuPy)
    if noise > 0:
        rng = np.random.default_rng(seed)
        st.ut += xp.asarray(noise * rng.standard_normal(st.ut.shape), dtype=DTYPE)
        st.w += xp.asarray(noise * rng.standard_normal(st.w.shape), dtype=DTYPE)

    st.t = 0.0
    return st