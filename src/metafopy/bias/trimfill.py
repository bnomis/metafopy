"""Trim-and-fill method (metafor's ``trimfill``).

Duval & Tweedie (2000), doi:10.1111/j.0006-341X.2000.00455.x and
doi:10.1080/01621459.2000.10473905. See
https://wviechtb.github.io/metafor/reference/trimfill.html.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing

import metafopy.bias.inputs
import metafopy.core.verbose
import metafopy.errors
import metafopy.formatters
import metafopy.metadata
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.rma_uni

__all__ = ['TrimfillResult', 'trimfill']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TrimfillResult(metafopy.models.rma_uni.RMAResult):
    """The model refitted to the observed and the filled-in studies, with the trim-and-fill details.

    ``fill`` flags the imputed studies (they are last); ``k0`` is the estimated number of missing studies
    (``se_k0`` its standard error), ``side`` the side of the funnel plot they are missing from and
    ``p_k0`` the p-value of the R0 test (NaN for other estimators).
    """

    fill: BoolArray = dataclasses.field(default_factory=lambda: numpy.zeros(0, dtype=numpy.bool_))
    k0: int = 0
    se_k0: float = math.nan
    side: str = 'left'
    k0_est: str = 'L0'
    p_k0: float = math.nan

    def summary(
        self,
        digits: int | collections.abc.Mapping[str, int] | None = None,
        showfit: bool = True,
        *,
        signif_stars: bool | None = None,
        signif_legend: bool | None = None,
    ) -> str:
        digits = metafopy.mfopt.option_digits(digits, self.digits)
        block = [f'Estimated number of missing studies on the {self.side} side: {self.k0} (SE = {metafopy.bias.inputs.fx(self.se_k0, digits)})']
        if self.k0_est == 'R0':
            gap = ' ' * len(str(self.k0))
            p_text = metafopy.bias.inputs.fp(self.p_k0, digits, equal=True, sep=True, pname='p-val ')
            block.append(f'Test of H0: no missing studies on the {self.side} side:     {gap}{p_text}')
        return '\n' + '\n'.join(block) + '\n' + super().summary(digits, showfit, signif_stars=signif_stars, signif_legend=signif_legend)

    def __str__(self) -> str:
        return self.summary(showfit=False)


def trimfill_fit(
    res: metafopy.models.rma_uni.RMAResult,
    yi: FloatArray,
    vi: FloatArray,
    weights: FloatArray | None,
    ni: FloatArray | None,
    *,
    mods: FloatArray | None = None,
    use_control: bool,
    kw: dict[str, typing.Any],
) -> metafopy.models.rma_uni.RMAResult:
    k = yi.shape[0]
    X = numpy.ones((k, 1)) if mods is None else numpy.column_stack([numpy.ones(k), mods])
    md = metafopy.metadata.MetaData(
        yi=yi,
        vi=vi,
        X=X,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        X_names=('intrcpt',) if mods is None else ('intrcpt', 'mods'),
        slab=tuple(str(i + 1) for i in range(k)),
        keep=numpy.ones(k, dtype=numpy.bool_),
        weights=weights,
        ni=ni,
        intercept=True,
    )
    return metafopy.models.rma_uni.fit_rma_uni(
        md, method=res.method, weighted=res.weighted, control=dataclasses.asdict(res.control) if use_control else None, skip_r2=True, **kw
    )


def sqrt_nan(x: float) -> float:
    """R's ``sqrt``: NaN with a warning for a negative number."""
    if x < 0:
        warnings.warn('NaNs produced', stacklevel=3)
        return math.nan
    return math.sqrt(x)


def trimfill(
    x: metafopy.models.rma_uni.RMAResult,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    side: typing.Literal['left', 'right'] | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    estimator: typing.Literal['L0', 'R0', 'Q0'] = 'L0',
    maxiter: int = 100,
    verbose: bool = False,
    ilim: tuple[float, float] | None = None,
    **kwargs: typing.Any,
) -> TrimfillResult:
    """Trim-and-fill for a model without moderators.

    ``kwargs`` (``level``, ``test``) are passed to the model fits, as the ``...`` of metafor's ``trimfill``.
    ``verbose`` prints metafor's iteration log (the number of missing studies and the estimate) on stdout.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.fitview.check_method(x, 'trimfill')
    metafopy.methods.fitview.check_class(x, must='rma.uni', notav=('robust.rma', 'rma.ls', 'rma.gen', 'rma.uni.selmodel'))
    if not x.int_only:
        raise metafopy.errors.InputError('Trim-and-fill method only applicable to models without moderators.')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if estimator not in ('L0', 'R0', 'Q0'):
        raise metafopy.errors.InputError(f'unknown estimator {estimator!r}')
    if x.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    kw = {k: v for k, v in kwargs.items() if k in ('level', 'test')}

    yi, vi, wi, ni = x.yi, x.vi, x.weights, x.data.ni
    if side is None:
        with_se = trimfill_fit(x, yi, vi, wi, ni, mods=numpy.sqrt(vi), use_control=True, kw=kw)
        side = 'right' if with_se.beta[1] < 0 else 'left'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif side not in ('left', 'right'):
        raise metafopy.errors.InputError(f'unknown side {side!r}')
    if side == 'right':
        yi = -1 * yi
    ix = numpy.argsort(yi, kind='stable')
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
    est_digits = metafopy.mfopt.option_digits(None)
    if verbose:
        metafopy.core.verbose.cat('\n')
    while abs(k0 - k0_prev) > 0:
        k0_prev = k0
        iteration += 1
        if iteration > maxiter:
            raise metafopy.errors.InputError('Trim and fill algorithm did not converge.')
        sel = slice(0, k - k0)
        fit = trimfill_fit(x, yi[sel], vi[sel], None if wi is None else wi[sel], None, use_control=True, kw=kw)
        beta = float(fit.beta[0])
        yi_c = yi - beta
        ranks = numpy.empty(k)
        ranks[numpy.argsort(numpy.abs(yi_c), kind='stable')] = numpy.arange(1, k + 1)
        signed = numpy.sign(yi_c) * ranks
        if estimator == 'R0':
            neg = signed[signed < 0]
            if neg.size == 0:
                raise metafopy.errors.InputError('Trim and fill algorithm failed: no study to the left of the pooled estimate.')
            k0f = (k - float(numpy.max(-1 * neg))) - 1
            se_k0 = math.sqrt(2 * max(0.0, k0f) + 2)
        else:
            sr = float(numpy.sum(signed[signed > 0]))
            if estimator == 'L0':
                k0f = (4 * sr - k * (k + 1)) / (2 * k - 1)
            else:
                k0f = k - 1 / 2 - math.sqrt(2 * k**2 - 4 * sr + 1 / 4)
            var_sr = 1 / 24 * (k * (k + 1) * (2 * k + 1) + 10 * k0f**3 + 27 * k0f**2 + 17 * k0f - 18 * k * k0f**2 - 18 * k * k0f + 6 * k**2 * k0f)
            if estimator == 'L0':
                se_k0 = 4 * sqrt_nan(var_sr) / (2 * k - 1)
            else:
                se_k0 = 2 * sqrt_nan(var_sr) / sqrt_nan((k - 1 / 2) ** 2 - k0f * (2 * k - k0f - 1))
        k0 = max(0, round(k0f))
        se_k0 = se_k0 if math.isnan(se_k0) else max(0.0, se_k0)
        if verbose:
            it_text = metafopy.formatters.fmtx(iteration, 0, flag='-', addwidth=len(str(maxiter)))
            k0_text = metafopy.formatters.fmtx(k0, 0, flag='-', addwidth=len(str(k)))
            beta_text = metafopy.formatters.fmtx(-beta if side == 'right' else beta, est_digits)
            metafopy.core.verbose.cat(f'Iteration: {it_text}  missing = {k0_text}  beta = {beta_text}\n')

    md = x.data
    if k0 > 0:
        yi_c = -1 * (yi_c - beta) if side == 'right' else yi_c - beta
        tail = slice(k - k0, k)
        yi_fill = numpy.concatenate([x.yi, -1 * yi_c[tail]])
        if ilim is not None:
            lo, hi = sorted(ilim)
            yi_fill = numpy.clip(yi_fill, lo, hi)
        vi_fill = numpy.concatenate([x.vi, vi[tail]])
        wi_fill = None if x.weights is None or wi is None else numpy.concatenate([x.weights, wi[tail]])
        ni_fill = None if md.ni is None or ni is None else numpy.concatenate([md.ni, ni[tail]])
        ids = numpy.flatnonzero(md.keep) + 1
        default_slab = x.slab == tuple(str(i) for i in ids)
        slab = (tuple(f'Study {i}' for i in ids) if default_slab else tuple(x.slab)) + tuple(f'Filled {i + 1}' for i in range(k0))
        mdf = metafopy.metadata.MetaData(
            yi=yi_fill,
            vi=vi_fill,
            X=numpy.ones((yi_fill.shape[0], 1)),
            X_names=tuple(x.coef_names),
            slab=slab,
            keep=numpy.concatenate([md.keep, numpy.ones(k0, dtype=numpy.bool_)]),
            weights=wi_fill,
            ni=ni_fill,
            intercept=True,
            slab_given=True,
            ids_all=numpy.concatenate([x.ids, int(x.ids.max()) + 1 + numpy.arange(k0)]).astype(numpy.intp),
        )
        res = metafopy.models.rma_uni.fit_rma_uni(mdf, method=x.method, weighted=x.weighted, **kw)
        refit = {'yi': yi_fill, 'vi': vi_fill, 'weights': wi_fill, 'ni': ni_fill}
        # in metafor the measure travels as an attribute of the yi in the call
        arguments = {name: value for name, value in refit.items() if value is not None} | {
            'measure': x.measure,
            'method': x.method,
            'weighted': x.weighted,
            **kw,
        }
        res = dataclasses.replace(
            res, measure=x.measure, digits=x.digits, call=metafopy.models.call.Call(function=metafopy.models.rma_uni.rma_uni, arguments=arguments)
        )
        fill = numpy.concatenate([numpy.zeros(x.k, dtype=numpy.bool_), numpy.ones(k0, dtype=numpy.bool_)])
    else:
        res = x
        fill = numpy.zeros(k, dtype=numpy.bool_)

    p_k0 = math.nan
    if estimator == 'R0':
        p_k0 = 1 - sum(math.comb(m + 1, m + 1) * 0.5 ** (m + 2) for m in range(-1, k0))
    values = {f.name: getattr(res, f.name) for f in dataclasses.fields(metafopy.models.rma_uni.RMAResult)}
    return TrimfillResult(**values, fill=fill, k0=k0, se_k0=se_k0, side=side, k0_est=estimator, p_k0=p_k0)
