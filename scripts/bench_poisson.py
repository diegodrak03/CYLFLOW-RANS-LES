"""Times PoissonSolver.solve.
  python scripts/bench_poisson.py                      # uses CYLFLOW_BACKEND / CYLFLOW_PRECISION
  CYLFLOW_BACKEND=cupy python scripts/bench_poisson.py
  CYLFLOW_BACKEND=cupy CYLFLOW_PRECISION=float32 python scripts/bench_poisson.py
"""
import time

import numpy as np

from cylflow.backend import BACKEND, DTYPE, synchronize, xp
from cylflow.grid import make_grid
from cylflow.poisson import PoissonSolver

MESHES = {"LES base": (160, 256, 48), "LES fine": (192, 384, 64)}
REPEATS = 20


def bench(Nr, Nth, Nz):
    g = make_grid(0.5, 25.0, Nr, Nth, Nz, np.pi, dr_min=0.003)

    t0 = time.perf_counter()
    solver = PoissonSolver(g)                 # built once per simulation
    synchronize()
    t_init = time.perf_counter() - t0

    f = xp.asarray(np.random.default_rng(0).standard_normal((Nr, Nth, Nz)), dtype=DTYPE)
    f = f - f.mean()                          # compatible right-hand side
    solver.solve(f)                           # warm-up: the first call is always slower
    synchronize()

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        solver.solve(f)
        synchronize()                         # wait for the GPU before stopping the clock
        times.append(time.perf_counter() - t0)
    return t_init, float(np.median(times))


def main():
    print(f"backend = {BACKEND}, precision = {np.dtype(DTYPE).name}")
    for name, (Nr, Nth, Nz) in MESHES.items():
        t_init, t_solve = bench(Nr, Nth, Nz)
        print(f"{name:9s} {Nr}x{Nth}x{Nz} ({Nr * Nth * Nz / 1e6:.1f} M cells): "
              f"init {t_init:.2f} s, solve {t_solve * 1e3:.1f} ms (median of {REPEATS}), "
              f"3 solves per time step = {3 * t_solve * 1e3:.1f} ms")


if __name__ == "__main__":
    main()