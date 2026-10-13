"""R's ``NA_real_`` as a float.

R distinguishes NA (a NaN with payload 1954) from NaN: ``escalc()`` recodes
``yi`` and ``vi`` to NA together only when one of them is NaN or infinite, so
an NA in one of them alone (e.g. ``vi`` undefined for ``ni <= 3``) is kept. NaN
payloads propagate through IEEE arithmetic the same way in NumPy as in R, so
using the same bit pattern reproduces R's behaviour.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ['NA', 'is_na']

NA = float(np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0])


def is_na(x: NDArray[np.float64]) -> NDArray[np.bool_]:
    """R's ``is.na(x) & !is.nan(x)``."""
    arr = np.ascontiguousarray(x, dtype=np.float64)
    bits = arr.view(np.uint64)
    out: NDArray[np.bool_] = np.isnan(arr) & ((bits & np.uint64(0xFFFFFFFF)) == np.uint64(1954))
    return out
