"""Convergence plot of the divergence operator.
Run from the project root: python -m verification.operator_convergence
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tests.test_operators import divergence_error

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ns = np.array([8, 16, 32, 64, 128])
h = 1.0 / ns
errors = np.array([divergence_error(n) for n in ns])     # all errors in one array

orders = np.log2(errors[:-1] / errors[1:])
print("errors:", errors)
print("observed orders:", orders)

C = errors[0] / h[0] ** 2                                # line through the first point

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(h, errors, "o-", label="divergence error")
ax.loglog(h, C * h**2, "k--", label="slope 2 reference")
ax.set_xlabel("cell size h = 1/n")
ax.set_ylabel("max error")
ax.set_title("Convergence of the divergence operator")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "operator_convergence.png", dpi=150)
plt.show()
