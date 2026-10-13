"""R's ``NA_real_`` as a float.

R distinguishes NA (a NaN with payload 1954) from NaN: ``escalc()`` recodes
``yi`` and ``vi`` to NA together only when one of them is NaN or infinite, so
an NA in one of them alone (e.g. ``vi`` undefined for ``ni <= 3``) is kept. NaN
payloads propagate through IEEE arithmetic the same way in NumPy as in R, so
using the same bit pattern reproduces R's behaviour.
"""

import numpy
import numpy.typing

__all__ = ['NA', 'is_na']

NA = float(numpy.array([0x7FF00000000007A2], dtype=numpy.uint64).view(numpy.float64)[0])


def is_na(x: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.bool_]:
    """R's ``is.na(x) & !is.nan(x)``."""
    arr = numpy.ascontiguousarray(x, dtype=numpy.float64)
    bits = arr.view(numpy.uint64)
    out: numpy.typing.NDArray[numpy.bool_] = numpy.isnan(arr) & ((bits & numpy.uint64(0xFFFFFFFF)) == numpy.uint64(1954))
    return out
