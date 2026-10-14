"""R's ``p.adjust()``: adjusted p-values for multiple testing.

A port of ``stats::p.adjust`` (R 4.3), used by ``anova(..., adjust=)`` as metafor does.
R source: https://github.com/wch/r-source/blob/trunk/src/library/stats/R/p.adjust.R;
Holm (1979) https://www.jstor.org/stable/4615733, Hochberg (1988) https://doi.org/10.1093/biomet/75.4.800,
Hommel (1988) https://doi.org/10.1093/biomet/75.2.383, Benjamini & Hochberg (1995) https://doi.org/10.1111/j.2517-6161.1995.tb02031.x,
Benjamini & Yekutieli (2001) https://doi.org/10.1214/aos/1013699998.
"""

import collections.abc

import numpy
import numpy.typing

import metafopy.errors

__all__ = ['P_ADJUST_METHODS', 'match_adjust', 'p_adjust']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] R's p.adjust.methods order
P_ADJUST_METHODS = ('holm', 'hochberg', 'hommel', 'bonferroni', 'BH', 'BY', 'fdr', 'none')


def match_adjust(adjust: bool | str) -> str:
    """metafor's handling of ``adjust``: ``True`` is ``'bonferroni'``, ``False`` is ``'none'``, a string is partially matched.

    Raises:
        metafopy.errors.InputError: the method is unknown.
    """
    if isinstance(adjust, bool):
        return 'bonferroni' if adjust else 'none'
    if adjust in P_ADJUST_METHODS:
        return adjust
    hits = [m for m in P_ADJUST_METHODS if m.startswith(adjust)] if adjust else []
    if len(hits) != 1:
        raise metafopy.errors.InputError("Unknown 'adjust' method specified (see 'help(p.adjust)' for options).")
    return hits[0]


def order(p: FloatArray, decreasing: bool = False) -> numpy.typing.NDArray[numpy.intp]:
    """R's (stable) ``order()``."""
    return numpy.argsort(-p if decreasing else p, kind='stable')


def p_adjust(p: collections.abc.Sequence[float] | FloatArray, method: str = 'holm', n: int | None = None) -> FloatArray:
    """Adjusted p-values, as R's ``p.adjust(p, method, n = length(p))``; NaN entries stay NaN and do not count towards ``n``.

    Raises:
        metafopy.errors.InputError: the method is unknown or ``n`` is less than the number of p-values.
    """
    method = match_adjust(method)
    if method == 'fdr':
        method = 'BH'
    p0 = numpy.array(p, dtype=numpy.float64).reshape(-1)
    nna = ~numpy.isnan(p0)
    pv = p0[nna]
    lp = pv.size
    if n is None:
        # R's default n = length(p) is evaluated lazily, after the NAs were dropped from p
        n = lp
    if n < lp:
        raise metafopy.errors.InputError('n >= lp is not TRUE')
    if n <= 1:
        return p0
    if n == 2 and method == 'hommel':
        method = 'hochberg'
    out: FloatArray
    if method == 'bonferroni':
        out = numpy.minimum(1.0, n * pv)
    elif method == 'holm':
        i = numpy.arange(1, lp + 1)
        o = order(pv)
        ro = numpy.argsort(o, kind='stable')
        out = numpy.minimum(1.0, numpy.maximum.accumulate((n + 1 - i) * pv[o]))[ro]
    elif method == 'hommel':
        q_p = numpy.concatenate([pv, numpy.ones(n - lp)]) if n > lp else pv.copy()
        i = numpy.arange(1, n + 1)
        o = order(q_p)
        q_p = q_p[o]
        ro = numpy.argsort(o, kind='stable')
        q = numpy.full(n, float(numpy.min(n * q_p / i)))
        pa = q.copy()
        for j in range(n - 1, 1, -1):
            ij = numpy.arange(0, n - j + 1)
            i2 = numpy.arange(n - j + 1, n)
            q1 = float(numpy.min(j * q_p[i2] / numpy.arange(2, j + 1)))
            q[ij] = numpy.minimum(j * q_p[ij], q1)
            q[i2] = q[n - j]
            pa = numpy.maximum(pa, q)
        out = numpy.maximum(pa, q_p)[ro[:lp] if lp < n else ro]
    elif method in ('hochberg', 'BH', 'BY'):
        i = numpy.arange(lp, 0, -1)
        o = order(pv, decreasing=True)
        ro = numpy.argsort(o, kind='stable')
        if method == 'hochberg':
            scaled = (n + 1 - i) * pv[o]
        elif method == 'BH':
            scaled = n / i * pv[o]
        else:
            scaled = float(numpy.sum(1.0 / numpy.arange(1, n + 1))) * n / i * pv[o]
        out = numpy.minimum(1.0, numpy.minimum.accumulate(scaled))[ro]
    else:
        out = pv
    p0[nna] = out
    return p0
