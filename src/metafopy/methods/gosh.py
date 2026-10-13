"""Graphical display of study heterogeneity (metafor's ``gosh.rma``).

The model is fitted to a large number of subsets of the ``k`` studies (all
``2^k - p - ...`` of them if feasible, otherwise a random sample) and the
pooled estimate(s) and heterogeneity statistics of each fit are collected, as
proposed by Olkin, Dahabreh & Trikalinos (2012), doi:10.1002/jrsm.1033. See
https://wviechtb.github.io/metafor/reference/gosh.rma.html.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from metafopy._print import fmt_num
from metafopy.core.rng import RRandom, get_rng, set_seed
from metafopy.methods._common import refit
from metafopy.methods.blup import format_r_frame
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult

__all__ = ['GoshResult', 'gosh']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GoshResult:
    """Statistics of the model fitted to each subset of studies.

    ``res`` has one row per subset (sorted by the number of studies ``k``) with the
    columns ``k``, ``QE``, ``I2``, ``H2``, ``tau2``, ``tau`` and the coefficient estimates;
    ``incl`` is the corresponding matrix of inclusion indicators, ``fit`` marks successful fits.
    """

    res: pd.DataFrame
    incl: NDArray[np.bool_]
    fit: NDArray[np.bool_]
    k: int
    int_only: bool
    method: str
    measure: str
    digits: int = 4

    def summary(self) -> str:
        return _format(self)

    def __str__(self) -> str:
        return _format(self)


def _format(x: GoshResult) -> str:
    d = x.digits
    res = x.res
    stats_: list[tuple[str, Callable[[FloatArray], float]]] = [
        ('mean', lambda c: float(np.nanmean(c))),
        ('min', lambda c: float(np.nanmin(c))),
        ('q1', lambda c: float(np.nanquantile(c, 0.25))),
        ('median', lambda c: float(np.nanquantile(c, 0.5))),
        ('q3', lambda c: float(np.nanquantile(c, 0.75))),
        ('max', lambda c: float(np.nanmax(c))),
    ]
    names = list(res.columns)
    rows: list[list[str]] = [[fmt_num(f(res[c].to_numpy()), d) for _, f in stats_] for c in names]
    labels = list(names)
    if len(names) == 6:
        labels[1] = 'Q'
    if len(names) > 6:
        rows = [*rows[:5], [''] * 6, *rows[5:]]
        labels = [*labels[:5], '', *labels[5:]]
    if x.method in FIXED_EFFECTS:
        del rows[4]
        del labels[4]
    cols = [(h, [r[j] for r in rows]) for j, (h, _) in enumerate(stats_)]
    head = ['', f'Model fits attempted: {len(x.fit)}', f'Model fits succeeded: {int(x.fit.sum())}', '']
    return '\n'.join([*head, *format_r_frame(cols, labels), ''])


@singledispatch
def gosh(x: Any, *args: Any, **kwargs: Any) -> GoshResult:
    """GOSH analysis of a fitted model; see :func:`_gosh_rma_uni`."""
    raise TypeError(f'gosh() not implemented for {type(x).__name__}')


@gosh.register
def _gosh_rma_uni(x: RMAResult, subsets: int | None = None, *, seed: int | None = None, rng: RRandom | None = None) -> GoshResult:
    """Fit the model to subsets of the studies.

    By default all subsets (with at least ``p`` studies) are used if there are
    at most ``10**6`` of them, otherwise ``10**6`` random subsets; ``subsets``
    sets that number. ``seed`` seeds the R-compatible generator (so that
    ``gosh(res, seed=s)`` draws the same subset sizes and study indices as
    R's ``set.seed(s)`` does, apart from the sampling probabilities being computed by SciPy
    rather than R's ``dbinom``, which can reorder tied sizes); without it the module's global
    generator is used (:func:`metafopy.set_seed`).
    """
    k, p = x.k, x.p
    if k == 1:
        raise ValueError('Stopped because k = 1.')
    n_tot = sum(math.comb(k, m) for m in range(p, k + 1))
    if subsets is None:
        exact = n_tot <= 10**6
        if not exact:
            n_tot = 10**6
    else:
        subsets = round(subsets)
        if subsets <= 1:
            raise ValueError("Argument 'subsets' must be >= 2.")
        exact = n_tot <= subsets
        if not exact:
            n_tot = subsets

    if exact:
        r = np.arange(2**k)
        incl_all = ((r[:, None] >> np.arange(k)[None, :]) & 1).astype(np.bool_)
        incl = incl_all[incl_all.sum(axis=1) >= p]
    else:
        gen = rng if rng is not None else (set_seed(seed) if seed is not None else get_rng())
        sizes = np.arange(p, k + 1)
        j = sizes[gen.sample_int(len(sizes), n_tot, replace=True, prob=np.asarray(stats.binom.pmf(sizes, k, 0.5), dtype=np.float64)) - 1]
        incl = np.zeros((n_tot, k), dtype=np.bool_)
        for i, m in enumerate(j):
            incl[i, gen.sample_int(k, int(m)) - 1] = True

    model = 0
    if x.method in FIXED_EFFECTS and x.weighted and x.weights is None and x.int_only:
        model = 1
    if x.method == 'DL' and x.weighted and x.weights is None and x.int_only:
        model = 2

    n = incl.shape[0]
    beta = np.full((n, p), np.nan)
    het = np.full((n, 5), np.nan)
    for i in range(n):
        mask = incl[i]
        if model:
            b, h = _closed_form(x, mask, model)
            beta[i], het[i] = b, h
            continue
        fit = refit(x, mask)
        if fit is None:
            continue
        beta[i] = fit.beta
        het[i] = (fit.k, fit.QE, fit.I2, fit.H2, fit.tau2)
    ok = ~np.isnan(het).any(axis=1)
    if not ok.any():
        raise ValueError('All model fits failed.')
    het[:, 0] = incl.sum(axis=1)
    cols = ['k', 'QE', 'I2', 'H2', 'tau2']
    het6 = np.column_stack([het, np.sqrt(het[:, 4])])
    names = ['estimate'] if x.int_only else list(x.coef_names)
    frame = pd.DataFrame(np.column_stack([het6, beta]), columns=[*cols, 'tau', *names])
    order = np.argsort(frame['k'].to_numpy(), kind='stable')
    frame = frame.iloc[order].reset_index(drop=True)
    incl = incl[order]
    fit_ok = ~frame.isna().any(axis=1).to_numpy()
    return GoshResult(res=frame, incl=incl, fit=fit_ok, k=k, int_only=x.int_only, method=x.method, measure=x.measure)


def _closed_form(x: RMAResult, mask: NDArray[np.bool_], model: int) -> tuple[FloatArray, FloatArray]:
    """The EE/FE and DL fits to a subset of an intercept-only model in closed form (metafor's ``model`` shortcut)."""
    yi, vi = x.yi[mask], x.vi[mask]
    k = yi.shape[0]
    wi = 1.0 / vi
    sumwi = wi.sum()
    est = float(np.sum(wi * yi) / sumwi)
    Q, I2, H2, tau2 = 0.0, 0.0, 1.0, 0.0
    if k > 1:
        Q = float(np.sum(wi * (yi - est) ** 2))
        I2 = max(0.0, 100 * (Q - (k - 1)) / Q)
        H2 = Q / (k - 1)
        if model == 2:
            tau2 = max(0.0, (Q - (k - 1)) / (sumwi - float(np.sum(wi**2)) / sumwi))
            wi = 1.0 / (vi + tau2)
            est = float(np.sum(wi * yi) / wi.sum())
    return np.array([est]), np.array([k, Q, I2, H2, tau2], dtype=np.float64)
