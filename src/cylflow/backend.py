"""Choose backend (NumPy or CuPy) and precision (float32 or float64) for cylflow."""
import os
import numpy as np

BACKEND = os.environ.get("CYLFLOW_BACKEND", "numpy").lower()
PRECISION = os.environ.get("CYLFLOW_PRECISION", "float64").lower()

if BACKEND == "cupy":
    import cupy as xp
elif BACKEND == "numpy":
    xp = np
else:
    raise ValueError(f"CYLFLOW_BACKEND unknown: {BACKEND!r} (use 'numpy' or 'cupy')")

if PRECISION not in ("float32", "float64"):
    raise ValueError(f"CYLFLOW_PRECISION unknown: {PRECISION!r}")

DTYPE = xp.float64 if PRECISION == "float64" else xp.float32
CDTYPE = xp.complex128 if PRECISION == "float64" else xp.complex64


def to_numpy(a):
    """Returns a copy in CPU (NumPy) of an array, regardless of whether it comes from NumPy or CuPy."""
    if BACKEND == "cupy":
        return xp.asnumpy(a)
    return np.asarray(a)


def synchronize():
    """Waits for the GPU to finish. Essential before measuring times."""
    if BACKEND == "cupy":
        xp.cuda.Stream.null.synchronize()