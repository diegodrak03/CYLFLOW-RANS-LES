from cylflow.backend import DTYPE, to_numpy, xp


def test_xp_creates_arrays():
    a = xp.ones(3, dtype=DTYPE)
    assert float(a.sum()) == 3.0


def test_to_numpy_returns_numpy():
    import numpy as np
    a = to_numpy(xp.arange(4))
    assert isinstance(a, np.ndarray)