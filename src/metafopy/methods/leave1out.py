"""Leave-one-out and cumulative meta-analyses (metafor's ``leave1out.rma.uni`` and ``cumul.rma.uni``).

``leave1out`` refits an intercept-only model with each study (or cluster of
studies) left out in turn; ``cumul`` adds the studies one at a time in a given
order (Lau et al. 1992, doi:10.1056/NEJM199207023270206 popularised the
cumulative approach). See
https://wviechtb.github.io/metafor/reference/leave1out.rma.uni.html and
https://wviechtb.github.io/metafor/reference/cumul.rma.uni.html.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from metafopy._print import fmt_num
from metafopy.methods._common import NAAction, check_na_action, pad_to_supplied, psort, refit
from metafopy.methods.blup import format_r_frame
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult
from metafopy.transf import apply_transf

__all__ = ['CumulResult', 'Leave1outResult', 'cumul', 'leave1out']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Leave1outResult:
    """One row per left-out study with the re-estimated pooled effect and heterogeneity statistics.

    ``se`` is NaN when a transformation was applied, as in metafor. ``tau2`` is
    ``None`` for equal/fixed/common-effects models.
    """

    estimate: FloatArray
    se: FloatArray
    stat: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    Q: FloatArray
    Qp: FloatArray
    tau2: FloatArray | None
    I2: FloatArray
    H2: FloatArray
    slab: tuple[str, ...]
    stat_name: str = 'zval'
    digits: int = 4
    transf: bool = False
    k: FloatArray | None = None
    order: NDArray[Any] | None = None

    def to_frame(self) -> pd.DataFrame:
        """Table with metafor's column names (``estimate``, ``se``, ``zval``/``tval``, ``pval``, ``ci.lb``, ...)."""
        cols: dict[str, Any] = {}
        if self.k is not None:
            cols['k'] = self.k
        cols.update(
            {
                'estimate': self.estimate,
                'se': self.se,
                self.stat_name: self.stat,
                'pval': self.pval,
                'ci.lb': self.ci_lb,
                'ci.ub': self.ci_ub,
                'Q': self.Q,
                'Qp': self.Qp,
            }
        )
        if self.tau2 is not None:
            cols['tau2'] = self.tau2
        cols['I2'] = self.I2
        cols['H2'] = self.H2
        return pd.DataFrame(cols, index=list(self.slab))

    def __str__(self) -> str:
        d = self.digits
        spec: list[tuple[str, FloatArray, int]] = []
        if self.k is not None:
            spec.append(('k', self.k, 0))
        spec.append(('estimate', self.estimate, d))
        if not self.transf:
            spec.append(('se', self.se, d))
        spec += [
            (self.stat_name, self.stat, d),
            ('pval', self.pval, d),
            ('ci.lb', self.ci_lb, d),
            ('ci.ub', self.ci_ub, d),
            ('Q', self.Q, d),
            ('Qp', self.Qp, d),
        ]
        if self.tau2 is not None:
            spec.append(('tau2', self.tau2, d))
        spec += [('I2', self.I2, d), ('H2', self.H2, d)]
        cols = [(name, [fmt_num(float(v), dd) for v in vals]) for name, vals, dd in spec]
        return '\n'.join(['', *format_r_frame(cols, self.slab), ''])


@dataclass(frozen=True, slots=True)
class CumulResult(Leave1outResult):
    """Cumulative meta-analysis results (:class:`Leave1outResult` plus ``k`` and the ordering variable)."""


@singledispatch
def leave1out(x: Any, *args: Any, **kwargs: Any) -> Leave1outResult:
    """Leave-one-out analysis of a fitted model; see :func:`_leave1out_rma_uni`."""
    raise TypeError(f'leave1out() not implemented for {type(x).__name__}')


def _collect(
    fits: list[RMAResult | None],
    x: RMAResult,
    transf: Callable[..., Any] | None,
    targs: Any,
) -> dict[str, FloatArray]:
    n = len(fits)

    def get(attr: str) -> FloatArray:
        return np.array([np.nan if f is None else float(getattr(f, attr)) for f in fits], dtype=np.float64)

    out = {
        'estimate': np.array([np.nan if f is None else float(f.beta[0]) for f in fits]),
        'se': np.array([np.nan if f is None else float(f.se[0]) for f in fits]),
        'stat': np.array([np.nan if f is None else float(f.zval[0]) for f in fits]),
        'pval': np.array([np.nan if f is None else float(f.pval[0]) for f in fits]),
        'ci_lb': np.array([np.nan if f is None else float(f.ci_lb[0]) for f in fits]),
        'ci_ub': np.array([np.nan if f is None else float(f.ci_ub[0]) for f in fits]),
        'Q': get('QE'),
        'Qp': get('QEp'),
        'tau2': get('tau2'),
        'I2': get('I2'),
        'H2': get('H2'),
        'k': np.array([np.nan if f is None else float(f.k) for f in fits]),
    }
    if transf is not None:
        out['estimate'] = apply_transf(out['estimate'], transf, targs, x.tau2)
        out['se'] = np.full(n, np.nan)
        out['ci_lb'] = apply_transf(out['ci_lb'], transf, targs, x.tau2)
        out['ci_ub'] = apply_transf(out['ci_ub'], transf, targs, x.tau2)
    out['ci_lb'], out['ci_ub'] = psort(out['ci_lb'], out['ci_ub'])
    return out


@leave1out.register
def _leave1out_rma_uni(
    x: RMAResult,
    cluster: ArrayLike | str | None = None,
    digits: int = 4,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    *,
    na_action: NAAction = 'omit',
) -> Leave1outResult:
    """Refit an intercept-only ``rma.uni`` model leaving out each study (or each ``cluster`` of studies).

    ``transf`` is applied to the estimates and CI bounds (the integral and mode
    transformations default to the model's ``tau2``); the standard errors are
    then NaN. ``cluster`` gives one value per study supplied (or a column name
    in the data the model was fitted to) and leaves out all studies of one
    cluster at a time. With ``na_action='exclude'``/``'pass'`` (and no
    ``cluster``) the output has one row per supplied study.
    """
    check_na_action(na_action)
    if not x.int_only:
        raise ValueError('Method only applicable to models without moderators.')
    if x.k == 1:
        raise ValueError('Stopped because k = 1.')
    miss_cluster = cluster is None
    if miss_cluster:
        labels_k = np.arange(x.k)
    else:
        labels_k = _cluster_values(x, cluster)
    ids = list(dict.fromkeys(labels_k.tolist()))
    if not miss_cluster:
        ids = sorted(ids)
    fits = [refit(x, labels_k != cid) for cid in ids]
    col = _collect(fits, x, transf, targs)
    slab: tuple[str, ...]
    if miss_cluster:
        slab = tuple(f'-{s}' for s in x.slab)
    else:
        slab = tuple(f'-{cid}' for cid in ids)
    if na_action == 'fail' and not x.data.keep.all():
        raise ValueError('Missing values in results.')
    if miss_cluster and na_action in ('exclude', 'pass'):
        keys = ['estimate', 'se', 'stat', 'pval', 'ci_lb', 'ci_ub', 'Q', 'Qp', 'tau2', 'I2', 'H2']
        padded, full_slab = pad_to_supplied(x, *[col[key] for key in keys])
        col.update(dict(zip(keys, padded, strict=True)))
        slab = tuple(f'-{s}' for s in full_slab)
    fe = x.method in FIXED_EFFECTS
    return Leave1outResult(
        estimate=col['estimate'],
        se=col['se'],
        stat=col['stat'],
        pval=col['pval'],
        ci_lb=col['ci_lb'],
        ci_ub=col['ci_ub'],
        Q=col['Q'],
        Qp=col['Qp'],
        tau2=None if fe else col['tau2'],
        I2=col['I2'],
        H2=col['H2'],
        slab=slab,
        stat_name='tval' if x.test in ('knha', 'adhoc', 't') else 'zval',
        digits=digits,
        transf=transf is not None,
    )


def _cluster_values(x: RMAResult, values: ArrayLike | str | None) -> NDArray[Any]:
    """Resolve a per-study variable (array over the supplied rows or a column of the model's data) to the studies used."""
    if isinstance(values, str):
        if x.data.data is None or values not in x.data.data.columns:
            raise ValueError(f'cannot find the variable {values!r} in the data')
        arr = x.data.data[values].to_numpy()
    else:
        arr = np.asarray(values)
    keep = x.data.keep
    if arr.shape[0] != keep.shape[0]:
        raise ValueError(f'Length of the variable ({arr.shape[0]}) does not match the length of the data ({keep.shape[0]}).')
    return arr[keep]


@singledispatch
def cumul(x: Any, *args: Any, **kwargs: Any) -> CumulResult:
    """Cumulative meta-analysis of a fitted model; see :func:`_cumul_rma_uni`."""
    raise TypeError(f'cumul() not implemented for {type(x).__name__}')


@cumul.register
def _cumul_rma_uni(
    x: RMAResult,
    order: ArrayLike | str | None = None,
    digits: int = 4,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    collapse: bool = False,
    *,
    decreasing: bool = False,
) -> CumulResult:
    """Fit the intercept-only model to the first ``i`` studies for ``i = 1, ..., k``.

    ``order`` is a variable (one value per supplied study, or a column name of
    the model's data) giving the order in which studies are added; without it
    the original order is used. With ``collapse=True`` studies sharing the same
    value of ``order`` are added together. Studies dropped for missing values
    in the fit are skipped (the ``na.omit`` behaviour).
    """
    if not x.int_only:
        raise ValueError('Method only applicable to models without moderators.')
    k = x.k
    if order is None:
        orvar = pd.Series(np.arange(k))
        collapse = False
    else:
        orvar = pd.Series(_cluster_values(x, order))
    # R: order(orvar, decreasing=, na.last=NA); the radix sort is stable in both directions
    sub = orvar.dropna().sort_values(ascending=not decreasing, kind='stable')
    srt = sub.index.to_numpy()
    ov = orvar.iloc[srt].to_numpy()
    uor = np.array(list(dict.fromkeys(ov.tolist())), dtype=ov.dtype) if collapse else ov
    n = len(uor)
    fits: list[RMAResult | None] = []
    for i in range(n):
        incl = srt[np.isin(ov, uor[: i + 1])] if collapse else srt[: i + 1]
        mask = np.zeros(k, dtype=np.bool_)
        mask[incl] = True
        fits.append(refit(x, mask))
    col = _collect(fits, x, transf, targs)
    slab = tuple(_label(u) for u in uor) if collapse else tuple(np.asarray(x.slab, dtype=object)[srt].tolist())
    fe = x.method in FIXED_EFFECTS
    return CumulResult(
        estimate=col['estimate'],
        se=col['se'],
        stat=col['stat'],
        pval=col['pval'],
        ci_lb=col['ci_lb'],
        ci_ub=col['ci_ub'],
        Q=col['Q'],
        Qp=col['Qp'],
        tau2=None if fe else col['tau2'],
        I2=col['I2'],
        H2=col['H2'],
        slab=slab,
        stat_name='tval' if x.test in ('knha', 'adhoc', 't') else 'zval',
        digits=digits,
        transf=transf is not None,
        k=col['k'],
        order=uor,
    )


def _label(value: Any) -> str:
    """An ordering value as R's ``as.character`` would print it (``1948`` rather than ``1948.0``)."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)
