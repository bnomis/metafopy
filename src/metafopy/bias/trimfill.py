"""Trim-and-fill method (metafor's ``trimfill``).

Duval & Tweedie (2000), doi:10.1111/j.0006-341X.2000.00455.x and
doi:10.1080/01621459.2000.10473905. See
https://wviechtb.github.io/metafor/reference/trimfill.html.
"""

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from metafopy._data import MetaData
from metafopy.bias._inputs import fp, fx
from metafopy.models.rma_uni import RMAResult, fit_rma_uni

__all__ = ['TrimfillResult', 'trimfill']

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class TrimfillResult(RMAResult):
    """The model refitted to the observed and the filled-in studies, with the trim-and-fill details.

    ``fill`` flags the imputed studies (they are last); ``k0`` is the estimated number of missing studies
    (``se_k0`` its standard error), ``side`` the side of the funnel plot they are missing from and
    ``p_k0`` the p-value of the R0 test (NaN for other estimators).
    """

    fill: BoolArray = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))
    k0: int = 0
    se_k0: float = math.nan
    side: str = 'left'
    k0_est: str = 'L0'
    p_k0: float = math.nan

    def summary(self, digits: int = 4) -> str:
        block = [f'Estimated number of missing studies on the {self.side} side: {self.k0} (SE = {fx(self.se_k0, digits)})']
        if self.k0_est == 'R0':
            gap = ' ' * len(str(self.k0))
            p_text = fp(self.p_k0, digits, equal=True, sep=True, pname='p-val ')
            block.append(f'Test of H0: no missing studies on the {self.side} side:     {gap}{p_text}')
        return '\n' + '\n'.join(block) + '\n' + super().summary(digits)

    def __str__(self) -> str:
        return self.summary()


def _fit(
    res: RMAResult,
    yi: FloatArray,
    vi: FloatArray,
    weights: FloatArray | None,
    ni: FloatArray | None,
    *,
    mods: FloatArray | None = None,
    use_control: bool,
    kw: dict[str, Any],
) -> RMAResult:
    k = yi.shape[0]
    X = np.ones((k, 1)) if mods is None else np.column_stack([np.ones(k), mods])
    md = MetaData(
        yi=yi,
        vi=vi,
        X=X,
        X_names=('intrcpt',) if mods is None else ('intrcpt', 'mods'),
        slab=tuple(str(i + 1) for i in range(k)),
        keep=np.ones(k, dtype=np.bool_),
        weights=weights,
        ni=ni,
        intercept=True,
    )
    from dataclasses import asdict

    return fit_rma_uni(md, method=res.method, weighted=res.weighted, control=asdict(res.control) if use_control else None, _skip_r2=True, **kw)


def trimfill(
    x: RMAResult,
    side: Literal['left', 'right'] | None = None,
    estimator: Literal['L0', 'R0', 'Q0'] = 'L0',
    maxiter: int = 100,
    ilim: tuple[float, float] | None = None,
    **kwargs: Any,
) -> TrimfillResult:
    """Trim-and-fill for a model without moderators.

    ``kwargs`` (``level``, ``test``) are passed to the model fits, as the ``...`` of metafor's ``trimfill``.
    """
    if not x.int_only:
        raise ValueError('Trim-and-fill method only applicable to models without moderators.')
    if estimator not in ('L0', 'R0', 'Q0'):
        raise ValueError(f'unknown estimator {estimator!r}')
    if x.k == 1:
        raise ValueError('Stopped because k = 1.')
    kw = {k: v for k, v in kwargs.items() if k in ('level', 'test')}

    yi, vi, wi, ni = x.yi, x.vi, x.weights, x.data.ni
    if side is None:
        with_se = _fit(x, yi, vi, wi, ni, mods=np.sqrt(vi), use_control=True, kw=kw)
        side = 'right' if with_se.beta[1] < 0 else 'left'
    elif side not in ('left', 'right'):
        raise ValueError(f'unknown side {side!r}')
    if side == 'right':
        yi = -1 * yi
    ix = np.argsort(yi, kind='stable')
    yi, vi = yi[ix], vi[ix]
    wi = None if wi is None else wi[ix]
    ni = None if ni is None else ni[ix]
    k = yi.shape[0]

    k0_prev = -1
    k0 = 0
    iteration = 0
    se_k0 = math.nan
    yi_c = yi
    beta = math.nan
    while abs(k0 - k0_prev) > 0:
        k0_prev = k0
        iteration += 1
        if iteration > maxiter:
            raise ValueError('Trim and fill algorithm did not converge.')
        sel = slice(0, k - k0)
        fit = _fit(x, yi[sel], vi[sel], None if wi is None else wi[sel], None, use_control=True, kw=kw)
        beta = float(fit.beta[0])
        yi_c = yi - beta
        ranks = np.empty(k)
        ranks[np.argsort(np.abs(yi_c), kind='stable')] = np.arange(1, k + 1)
        signed = np.sign(yi_c) * ranks
        if estimator == 'R0':
            neg = signed[signed < 0]
            if neg.size == 0:
                raise ValueError('Trim and fill algorithm failed: no study to the left of the pooled estimate.')
            k0f = (k - float(np.max(-1 * neg))) - 1
            se_k0 = math.sqrt(2 * max(0.0, k0f) + 2)
        else:
            sr = float(np.sum(signed[signed > 0]))
            if estimator == 'L0':
                k0f = (4 * sr - k * (k + 1)) / (2 * k - 1)
            else:
                k0f = k - 1 / 2 - math.sqrt(2 * k**2 - 4 * sr + 1 / 4)
            var_sr = 1 / 24 * (k * (k + 1) * (2 * k + 1) + 10 * k0f**3 + 27 * k0f**2 + 17 * k0f - 18 * k * k0f**2 - 18 * k * k0f + 6 * k**2 * k0f)
            if estimator == 'L0':
                se_k0 = 4 * math.sqrt(var_sr) / (2 * k - 1)
            else:
                se_k0 = 2 * math.sqrt(var_sr) / math.sqrt((k - 1 / 2) ** 2 - k0f * (2 * k - k0f - 1))
        k0 = max(0, round(k0f))
        se_k0 = max(0.0, se_k0)

    md = x.data
    if k0 > 0:
        yi_c = -1 * (yi_c - beta) if side == 'right' else yi_c - beta
        tail = slice(k - k0, k)
        yi_fill = np.concatenate([x.yi, -1 * yi_c[tail]])
        if ilim is not None:
            lo, hi = sorted(ilim)
            yi_fill = np.clip(yi_fill, lo, hi)
        vi_fill = np.concatenate([x.vi, vi[tail]])
        wi_fill = None if x.weights is None or wi is None else np.concatenate([x.weights, wi[tail]])
        ni_fill = None if md.ni is None or ni is None else np.concatenate([md.ni, ni[tail]])
        ids = np.flatnonzero(md.keep) + 1
        default_slab = x.slab == tuple(str(i) for i in ids)
        slab = (tuple(f'Study {i}' for i in ids) if default_slab else tuple(x.slab)) + tuple(f'Filled {i + 1}' for i in range(k0))
        mdf = MetaData(
            yi=yi_fill,
            vi=vi_fill,
            X=np.ones((yi_fill.shape[0], 1)),
            X_names=tuple(x.coef_names),
            slab=slab,
            keep=np.concatenate([md.keep, np.ones(k0, dtype=np.bool_)]),
            weights=wi_fill,
            ni=ni_fill,
            intercept=True,
        )
        res = fit_rma_uni(mdf, method=x.method, weighted=x.weighted, **kw)
        res = replace(res, measure=x.measure)
        fill = np.concatenate([np.zeros(x.k, dtype=np.bool_), np.ones(k0, dtype=np.bool_)])
    else:
        res = x
        fill = np.zeros(k, dtype=np.bool_)

    p_k0 = math.nan
    if estimator == 'R0':
        p_k0 = 1 - sum(math.comb(m + 1, m + 1) * 0.5 ** (m + 2) for m in range(-1, k0))
    values = {f.name: getattr(res, f.name) for f in fields(RMAResult)}
    return TrimfillResult(**values, fill=fill, k0=k0, se_k0=se_k0, side=side, k0_est=estimator, p_k0=p_k0)
