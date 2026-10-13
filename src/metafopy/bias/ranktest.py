"""Rank correlation test for funnel plot asymmetry (metafor's ``ranktest``).

Begg & Mazumdar (1994), doi:10.2307/2533446. Kendall's tau between the
standardised estimates and their sampling variances; the p-value follows R's
``cor.test(method='kendall')``: exact when there are fewer than 50 studies and no
ties, a normal approximation otherwise.
See https://wviechtb.github.io/metafor/reference/ranktest.html.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.bias._inputs import data_of, fp, fx, vector_inputs
from metafopy.models.rma_uni import RMAResult

__all__ = ['RanktestResult', 'kendall_test', 'ranktest']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RanktestResult:
    tau: float
    pval: float
    digits: int = 4

    def summary(self, digits: int | None = None) -> str:
        d = self.digits if digits is None else digits
        return f"\nRank Correlation Test for Funnel Plot Asymmetry\n\nKendall's tau = {fx(self.tau, d)}, {fp(self.pval, d, equal=True, sep=True, pname='p ')}\n"

    def __str__(self) -> str:
        return self.summary()


def _pkendall(q: int, n: int) -> float:
    """P(T <= q) for Kendall's statistic (number of concordant pairs) of ``n`` untied observations."""
    counts = [1]
    for i in range(2, n + 1):
        new = [0] * (len(counts) + i - 1)
        for j, c in enumerate(counts):
            for m in range(i):
                new[j + m] += c
        counts = new
    total = math.factorial(n)
    return sum(counts[: max(q, -1) + 1]) / total


def kendall_test(x: FloatArray, y: FloatArray, exact: bool | None = None) -> tuple[float, float]:
    """``(tau, p-value)`` of R's two-sided ``cor.test(x, y, method='kendall', exact=exact)``; NaN p-value when undefined."""
    n = x.shape[0]
    tau = float(stats.kendalltau(x, y).statistic) if n > 1 else math.nan
    if not math.isfinite(tau):
        return math.nan, math.nan
    ties = min(np.unique(x).shape[0], np.unique(y).shape[0]) < n
    if exact is None:
        exact = n < 50
    if exact and not ties:
        q = round((tau + 1) * n * (n - 1) / 4)
        p = 1 - _pkendall(q - 1, n) if q > n * (n - 1) / 4 else _pkendall(q, n)
        return tau, min(2 * p, 1.0)
    if exact and ties:
        warnings.warn('Cannot compute exact p-value with ties', stacklevel=2)

    def tie_counts(v: FloatArray) -> FloatArray:
        _, c = np.unique(v, return_counts=True)
        return c[c > 1].astype(np.float64)

    xt, yt = tie_counts(x), tie_counts(y)
    t0 = n * (n - 1) / 2
    t1 = float(np.sum(xt * (xt - 1))) / 2
    t2 = float(np.sum(yt * (yt - 1))) / 2
    s = tau * math.sqrt((t0 - t1) * (t0 - t2))
    v0 = n * (n - 1) * (2 * n + 5)
    vt = float(np.sum(xt * (xt - 1) * (2 * xt + 5)))
    vu = float(np.sum(yt * (yt - 1) * (2 * yt + 5)))
    v1 = float(np.sum(xt * (xt - 1))) * float(np.sum(yt * (yt - 1)))
    v2 = float(np.sum(xt * (xt - 1) * (xt - 2))) * float(np.sum(yt * (yt - 1) * (yt - 2)))
    var_s = (v0 - vt - vu) / 18
    if n > 1:
        var_s += v1 / (2 * n * (n - 1))
    if n > 2:
        var_s += v2 / (9 * n * (n - 1) * (n - 2))
    z = s / math.sqrt(var_s)
    return tau, float(2 * min(stats.norm.cdf(z), stats.norm.sf(z)))


def ranktest(
    x: RMAResult | ArrayLike | str,
    vi: ArrayLike | str | None = None,
    *,
    sei: ArrayLike | str | None = None,
    subset: ArrayLike | None = None,
    data: pd.DataFrame | None = None,
    digits: int | None = None,
    exact: bool = True,
) -> RanktestResult:
    """Rank correlation test for funnel plot asymmetry; ``x`` is a fitted model without moderators or a vector of estimates."""
    if isinstance(x, RMAResult):
        if vi is not None or sei is not None or subset is not None:
            warnings.warn("Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object.", stacklevel=2)
        if not x.int_only:
            raise ValueError('Test only applicable to models without moderators.')
        yi, v, _ = data_of(x)
    else:
        yi, v_in, _ = vector_inputs(x, vi, sei, subset=subset, data=data, what='the test')
        if v_in is None:
            raise ValueError("Must specify the 'vi' or 'sei' argument.")
        v = v_in
    wi = 1 / v
    theta = float(np.sum(wi * yi) / np.sum(wi))
    vb = 1 / float(np.sum(wi))
    yi_star = (yi - theta) / np.sqrt(v - vb)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tau, pval = kendall_test(yi_star, v, exact)
        if math.isnan(pval):
            tau, pval = kendall_test(yi_star, v, False)
    return RanktestResult(tau=tau, pval=pval, digits=4 if digits is None else digits)
