"""Draws the polar mesh. Run: python scripts/plot_grid.py"""
import matplotlib.pyplot as plt
import numpy as np

from cylflow.backend import to_numpy
from cylflow.grid import make_grid


# 1. Build a small mesh (Nz = 1: we only look at the r-theta plane)
g = make_grid(R=0.5, R_ext=5.0, Nr=200, Nth=200, Nz=1, Lz=1.0, dr_min=0.003)

# 2. Bring the arrays to the CPU (matplotlib only understands NumPy)
rf = to_numpy(g.rf)
thf = to_numpy(g.thf)

# 3. Draw everything on one set of axes
fig, ax = plt.subplots(figsize=(7, 7))

theta = np.linspace(0, 2 * np.pi, 400)       # many points: smooth circles
for r in rf:                                 # one circle per ring
    ax.plot(r * np.cos(theta), r * np.sin(theta), color="black", linewidth=0.5)

for th in thf:                               # one ray per face angle
    ax.plot([g.R * np.cos(th), g.R_ext * np.cos(th)],
            [g.R * np.sin(th), g.R_ext * np.sin(th)],
            color="black", linewidth=0.5)

ax.fill(g.R * np.cos(theta), g.R * np.sin(theta), color="gray")   # the cylinder
ax.set_aspect("equal")                       # circles look like circles
ax.set_title(f"Polar mesh {g.Nr} x {g.Nth}")
fig.savefig("grid_full.png", dpi=150)

# 4. Second figure: the same axes, zoomed near the wall
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-0.35, 0.35)
ax.set_title("Zoom near the cylinder wall")
fig.savefig("grid_zoom.png", dpi=150)

plt.show()

