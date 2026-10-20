"""Direct Poisson solver: FFT in theta and z, tridiagonal (Thomas) in r.

Solves laplacian(phi) = f exactly (up to round-off), where laplacian is the discrete
operator of operators.py: laplacian(p) = divergence(gradient(p)).
"""
import numpy as np

from cylflow.backend import DTYPE, to_numpy, xp


class PoissonSolver:
    def __init__(self, g):
        """Precomputes everything that does not change during the simulation.

        Built ONCE at the start: eigenvalues, tridiagonal coefficients and the forward
        sweep of the Thomas algorithm for every Fourier mode.
        """
        Nr, Nth, Nz = g.Nr, g.Nth, g.Nz

        # The geometry is small: work on the CPU in float64 and convert at the end.
        rf = to_numpy(g.rf).astype(np.float64)    # (Nr+1,) radial faces
        rc = to_numpy(g.rc).astype(np.float64)    # (Nr,)   radial centers
        dr = to_numpy(g.dr).astype(np.float64)    # (Nr,)   cell widths
        drc = to_numpy(g.drc).astype(np.float64)  # (Nr-1,) distances between centers

        # ---- 1. Eigenvalues of the discrete second differences in theta and z ----
        # fftfreq(N) * N gives the wave number of each FFT position: 0, 1, 2, ..., -2, -1
        m = np.fft.fftfreq(Nth) * Nth
        n = np.fft.fftfreq(Nz) * Nz
        lam_t = -4.0 / g.dth**2 * np.sin(m * g.dth / 2) ** 2   # (Nth,)
        lam_z = -4.0 / g.dz**2 * np.sin(n * np.pi / Nz) ** 2    # (Nz,)

        # ---- 2. Coefficients of the radial part (identical to divergence(gradient)) ----
        # Row i of the system:  a[i]*phi[i-1] + b[i]*phi[i] + c[i]*phi[i+1] = f[i]
        a = np.zeros(Nr)                                  # neighbour inside  (i-1)
        c = np.zeros(Nr)                                  # neighbour outside (i+1)
        a[1:] = rf[1:-1] / (rc[1:] * dr[1:] * drc)        # a[0] = 0: the wall
        c[:-1] = rf[1:-1] / (rc[:-1] * dr[:-1] * drc)     # c[-1] = 0: the outer boundary
        b_r = -(a + c)                                    # radial part of the diagonal

        # ---- 3. Full diagonal for every mode (m, n): radial part + theta + z ----
        shape = (Nr, Nth, Nz)
        b = (b_r[:, None, None]
             + lam_t[None, :, None] / rc[:, None, None] ** 2
             + lam_z[None, None, :])                      # (Nr, Nth, Nz)
        c3 = np.broadcast_to(c[:, None, None], shape).copy()

        # ---- 4. The singular mode (m, n) = (0, 0): replace its first row by phi[0] = 0 ----
        b[0, 0, 0] = 1.0
        c3[0, 0, 0] = 0.0

        # ---- 5. Forward sweep of Thomas on the matrix (does not depend on f) ----
        # cp[i]  : modified upper coefficient,  cp[i] = c[i] / den[i]
        # inv[i] : 1 / den[i],  with den[0] = b[0] and den[i] = b[i] - a[i] * cp[i-1]
        cp = np.zeros(shape)
        inv = np.zeros(shape)
        inv[0] = 1.0 / b[0]
        cp[0] = c3[0] * inv[0]
        for i in range(1, Nr):
            inv[i] = 1.0 / (b[i] - a[i] * cp[i - 1])
            cp[i] = c3[i] * inv[i]

        # Keep on the backend only what solve() needs
        self.a = xp.asarray(a, dtype=DTYPE)[:, None, None]   # (Nr, 1, 1)
        self.cp = xp.asarray(cp, dtype=DTYPE)
        self.inv = xp.asarray(inv, dtype=DTYPE)
        self.Nr = Nr

    def solve(self, f):
        """Solves laplacian(phi) = f. Input and output have shape (Nr, Nth, Nz).

        Returns phi with zero mean.
        """
        # 1. From cells to waves: FFT in theta (axis 1) and z (axis 2)
        fh = xp.fft.fftn(f, axes=(1, 2))
        fh[0, 0, 0] = 0.0                       # right-hand side of the row phi[0] = 0

        # 2. Thomas, the part that depends on f: one loop over r, all modes at once
        a, cp, inv = self.a, self.cp, self.inv
        d = xp.empty_like(fh)
        d[0] = fh[0] * inv[0]
        for i in range(1, self.Nr):             # forward: from the wall outwards
            d[i] = (fh[i] - a[i] * d[i - 1]) * inv[i]
        x = xp.empty_like(fh)
        x[-1] = d[-1]
        for i in range(self.Nr - 2, -1, -1):    # backward: from outside to the wall
            x[i] = d[i] - cp[i] * x[i + 1]

        # 3. From waves back to cells, keep the real part
        phi = xp.fft.ifftn(x, axes=(1, 2)).real.astype(DTYPE)

        # 4. The constant is arbitrary: choose the one with zero mean
        return phi - phi.mean()