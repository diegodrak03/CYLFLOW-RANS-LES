# cylflow: RANS vs LES around a cylinder

An incompressible Navier-Stokes solver written from scratch in Python, running on the CPU
(NumPy) or the GPU (CuPy) with the same code. The goal is to compare **URANS**
(Spalart-Allmaras) and **LES** (Smagorinsky and WALE) for the flow around a circular
cylinder at **Re = 3900**, a classic benchmark with extensive experimental and numerical
reference data, and to validate both against the literature and against OpenFOAM.

> **Status: work in progress.** The mesh, the discrete operators and the Poisson solver are
> finished and verified. The time integration and the laminar validation (Re = 40 and
> Re = 100) are under development. See the [roadmap](#roadmap).

## Numerical method

| Component | Choice |
| --- | --- |
| Equations | Incompressible Navier-Stokes in cylindrical coordinates (r, θ, z) |
| Mesh | Polar O-grid around the cylinder, staggered (pressure at cell centres, each velocity component on its own faces), tanh stretching towards the wall, uniform and periodic in θ and z |
| Spatial discretisation | Second-order finite differences / finite volumes |
| Pressure | Projection method with a **direct Poisson solver**: FFT in θ and z, tridiagonal (Thomas) systems in r |
| Time integration | Low-storage RK3 (in progress) |
| Turbulence | URANS Spalart-Allmaras; LES Smagorinsky (with van Driest damping) and WALE (planned) |

### Why these choices

- **Staggered mesh:** couples pressure and velocity at neighbouring points and avoids the
  checkerboard pressure modes of collocated grids.
- **Laplacian = divergence(gradient):** the Poisson operator is built from the same discrete
  divergence and gradient used in the momentum equations, so the projection removes the
  divergence exactly, down to round-off.
- **Discrete eigenvalues in the FFT:** the Poisson solver uses the eigenvalues of the
  discrete second differences, not the continuous ones, so it inverts the discrete Laplacian
  exactly.

## Verification so far

| Test | Result |
| --- | --- |
| Mesh: stretching, volumes, staggering | Total volume equals the exact annulus volume to 1e-12 |
| Discrete divergence and gradient | Observed order of accuracy 1.83 → 1.94 → 1.97 → 1.99 on meshes of 8 to 128 cells (second order) |
| Poisson solver, random right-hand side | Recovers the exact discrete solution with relative error ~1e-14 |
| Projection of a random velocity field | Residual divergence ~1e-13 relative to the velocity |

Poisson solve time (median of 20 runs, laptop GPU):

| Mesh | Cells | NumPy, CPU, float64 | CuPy, GPU, float64 | CuPy, GPU, float32 |
| --- | --- | --- | --- | --- |
| 160 × 256 × 48 | 2.0 M | 64.5 ms | 18.0 ms | 16.7 ms |
| 192 × 384 × 64 | 4.7 M | 197.4 ms | 23.4 ms | 20.9 ms |

## Installation

```bash
git clone https://github.com/diegodrak03/RANS_LES.git
cd RANS_LES
conda create -n cylflow python=3.12 -y
conda activate cylflow
pip install -e ".[dev]"                 # CPU only
pip install -e ".[dev,gpu-cuda13]"      # with an NVIDIA GPU and CUDA 13 (use gpu-cuda12 for CUDA 12)
```

The backend and the precision are chosen with environment variables:

```bash
export CYLFLOW_BACKEND=cupy        # numpy (default) or cupy
export CYLFLOW_PRECISION=float32   # float64 (default) or float32
```

## Usage

```bash
pytest -v                                        # all tests
CYLFLOW_BACKEND=cupy pytest -v                   # the same tests on the GPU
python scripts/plot_grid.py                      # draw the polar mesh
python -m verification.operator_convergence      # convergence plot of the operators
python scripts/bench_poisson.py                  # time the Poisson solver
```

## Repository structure

```
config/                  YAML case files (base.yaml holds the defaults)
src/cylflow/
    backend.py           NumPy / CuPy selection and precision
    config.py            YAML loading with defaults
    grid.py              stretched staggered polar mesh
    operators.py         discrete divergence, gradient and Laplacian
    poisson.py           direct FFT + tridiagonal Poisson solver
    fields.py            simulation state and potential-flow initial condition
scripts/                 mesh plot, Poisson benchmark
verification/            convergence studies
tests/                   pytest suite
```

## Roadmap

- [x] Stretched staggered polar mesh
- [x] Discrete operators, verified second order
- [x] Direct Poisson solver (FFT + tridiagonal), CPU and GPU
- [ ] 2D laminar solver: convection, viscous terms, boundary conditions, RK3, forces
- [ ] Laminar validation at Re = 40 (steady) and Re = 100 (vortex shedding)
- [ ] 3D, Taylor-Couette and manufactured-solution verification
- [ ] URANS with Spalart-Allmaras at Re = 3900
- [ ] LES (Smagorinsky, WALE), Taylor-Green vortex verification and cylinder at Re = 3900
- [ ] Comparison with OpenFOAM and with experimental and LES reference data
- [ ] Post-processing to ParaView (velocity, pressure, vorticity, Q-criterion) and videos

## Author

Diego Antúnez · Mechanical engineer, MSc student in Mathematical Engineering and Computing