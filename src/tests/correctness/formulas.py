"""Independent re-derivations of the quantities metafopy computes, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as the value of an ``expect`` key. The function is called with the arguments of the case that it declares
(see ``cases.formula_value``).
"""

import numpy
import numpy.typing

type FloatArray = numpy.typing.NDArray[numpy.float64]


def risk_difference_yi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Risk difference ``p1 - p2`` of two independent proportions, ``p1 = ai / (ai + bi)`` and ``p2 = ci / (ci + di)`` (the definition)."""
    return ai / (ai + bi) - ci / (ci + di)


def risk_difference_vi(ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> FloatArray:
    """Variance of the difference of two independent binomial proportions, ``p1 (1 - p1) / n1 + p2 (1 - p2) / n2``.

    Derivation: the variance of a difference of independent estimates is the sum of their variances, and that of a binomial proportion is ``p (1 - p) / n``.
    """
    n1, n2 = ai + bi, ci + di
    p1, p2 = ai / n1, ci / n2
    return p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
