"""Confidence intervals, residuals, leave-one-out and cumulative analyses for ``rma_mh`` and ``rma_peto`` fits.

These mirror metafor's ``confint.rma.mh``, ``rstandard.rma.mh``,
``rstudent.rma.mh``, ``leave1out.rma.mh`` and ``cumul.rma.mh`` (and their
``rma.peto`` counterparts); the diagnostics refit the model with the
corresponding tables left out.
"""

import collections.abc
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.confint
import metafopy.methods.influence
import metafopy.methods.leave1out
import metafopy.models.binary
import metafopy.models.rma_mh
import metafopy.models.rma_peto
import metafopy.models.rma_uni
import metafopy.transf

__all__ = ['refit_tables']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Transf = collections.abc.Callable[..., typing.Any] | bool | None


def refit_tables(
    x: metafopy.models.binary.FixedBinaryResult,
    cells: dict[str, FloatArray],
    subset: numpy.typing.NDArray[numpy.bool_] | None,
) -> metafopy.models.binary.FixedBinaryResult | None:
    """Refit ``x``'s model to (a subset of) the given cell counts; ``None`` if the fit fails, as R's ``try()`` does."""
    level = 1 - x.level
    cell_args: dict[str, typing.Any] = dict(cells)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if isinstance(x, metafopy.models.rma_mh.MHResult):
                return metafopy.models.rma_mh.rma_mh(
                    **cell_args, measure=x.measure, subset=subset, add=x.add, to=x.to, drop00=x.drop00, correct=x.correct, level=level
                )
            return metafopy.models.rma_peto.rma_peto(**cell_args, subset=subset, add=x.add, to=x.to, drop00=x.drop00, level=level)
    except metafopy.methods.common.REFIT_ERRORS:
        return None


def resolve_transf(x: metafopy.models.binary.FixedBinaryResult, transf: Transf) -> collections.abc.Callable[..., typing.Any] | None:
    if transf is True:
        return numpy.exp if x.measure in ('OR', 'RR', 'IRR', 'PETO') else None
    if transf is False or transf is None:
        return None
    return transf


@metafopy.methods.confint.confint.register
def confint_fixed_binary(
    model: metafopy.models.binary.FixedBinaryResult,
    *,
    level: float | None = None,
    transf: Transf = None,
    targs: typing.Any = None,
) -> metafopy.methods.confint.ConfintResult:
    """Wald-type confidence interval of the pooled effect (``confint.rma.mh`` / ``confint.rma.peto``); ``transf=True`` exponentiates."""
    alpha = model.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    crit = float(scipy.stats.norm.isf(alpha / 2))
    beta = model.beta
    lb = beta - crit * model.se
    ub = beta + crit * model.se
    fn = resolve_transf(model, transf)
    if fn is not None:
        beta = metafopy.transf.apply_transf(beta, fn, targs)
        lb = metafopy.transf.apply_transf(lb, fn, targs)
        ub = metafopy.transf.apply_transf(ub, fn, targs)
    lb, ub = metafopy.methods.common.psort(lb, ub)
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    fixed = pandas.DataFrame({'estimate': beta, 'ci.lb': lb, 'ci.ub': ub}, index=[''])
    return metafopy.methods.confint.ConfintResult(fixed=fixed, random=None)


# -- residuals --------------------------------------------------------------------------------------


def residual_result(
    x: metafopy.models.binary.FixedBinaryResult,
    resid: FloatArray,
    seresid: FloatArray,
    digits: int,
    na_action: str,
) -> metafopy.methods.influence.ResidualResult:
    resid = metafopy.methods.common.tzero(resid)
    z = resid / seresid
    t = x.tables
    if na_action == 'fail' and not t.not_na_yivi.all():
        raise metafopy.errors.InputError('Missing values in results.')
    slab = numpy.asarray(t.slab, dtype=object)
    if na_action == 'omit':
        sel = t.not_na_yivi
        return metafopy.methods.influence.ResidualResult(resid=resid[sel], se=seresid[sel], z=z[sel], slab=tuple(slab[sel].tolist()), digits=digits)
    return metafopy.methods.influence.ResidualResult(resid=resid, se=seresid, z=z, slab=tuple(slab.tolist()), digits=digits)


@metafopy.methods.influence.rstandard.register
def rstandard_fixed_binary(
    model: metafopy.models.binary.FixedBinaryResult,
    digits: int | None = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.influence.ResidualResult:
    """Standardised residuals ``(yi - beta) / sqrt(vi)`` of the individual tables."""
    metafopy.methods.common.check_na_action(na_action)
    t = model.tables
    with numpy.errstate(all='ignore'):
        resid = t.yi_f - model.beta[0]
        se = numpy.sqrt(t.vi_f)
    return residual_result(model, resid, se, model.digits if digits is None else digits, na_action)


@metafopy.methods.influence.rstudent.register
def rstudent_fixed_binary(
    model: metafopy.models.binary.FixedBinaryResult,
    digits: int | None = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.influence.ResidualResult:
    """Externally standardised residuals: each table is compared with the pooled estimate from the other tables."""
    metafopy.methods.common.check_na_action(na_action)
    t = model.tables
    delpred = numpy.full(t.k_f, numpy.nan)
    vdelpred = numpy.full(t.k_f, numpy.nan)
    for i in range(t.k_f):
        if not t.not_na[i]:
            continue
        keep = numpy.ones(t.k_f, dtype=numpy.bool_)
        keep[i] = False
        fit = refit_tables(model, t.cells_f, keep)
        if fit is None:
            continue
        delpred[i] = fit.beta[0]
        vdelpred[i] = fit.vb[0, 0]
    with numpy.errstate(all='ignore'):
        resid = t.yi_f - delpred
        se = numpy.sqrt(t.vi_f + vdelpred)
    return residual_result(model, resid, se, model.digits if digits is None else digits, na_action)


# -- leave-one-out / cumulative -----------------------------------------------------------------------


def collect(
    fits: list[metafopy.models.binary.FixedBinaryResult | None],
    x: metafopy.models.binary.FixedBinaryResult,
    transf: collections.abc.Callable[..., typing.Any] | None,
    targs: typing.Any,
) -> dict[str, FloatArray]:
    def get(fn: collections.abc.Callable[[metafopy.models.binary.FixedBinaryResult], float]) -> FloatArray:
        return numpy.array([numpy.nan if f is None else fn(f) for f in fits], dtype=numpy.float64)

    out = {
        'estimate': get(lambda f: float(f.beta[0])),
        'se': get(lambda f: float(f.se[0])),
        'stat': get(lambda f: float(f.zval[0])),
        'pval': get(lambda f: float(f.pval[0])),
        'ci_lb': get(lambda f: float(f.ci_lb[0])),
        'ci_ub': get(lambda f: float(f.ci_ub[0])),
        'Q': get(lambda f: f.QE),
        'Qp': get(lambda f: f.QEp),
        'I2': get(lambda f: f.I2),
        'H2': get(lambda f: f.H2),
        'k': get(lambda f: float(f.k)),
    }
    if transf is not None:
        out['estimate'] = metafopy.transf.apply_transf(out['estimate'], transf, targs)
        out['se'] = numpy.full(len(fits), numpy.nan)
        out['ci_lb'] = metafopy.transf.apply_transf(out['ci_lb'], transf, targs)
        out['ci_ub'] = metafopy.transf.apply_transf(out['ci_ub'], transf, targs)
    out['ci_lb'], out['ci_ub'] = metafopy.methods.common.psort(out['ci_lb'], out['ci_ub'])
    return out


def per_table_variable(x: metafopy.models.binary.FixedBinaryResult, values: numpy.typing.ArrayLike | str, what: str) -> numpy.typing.NDArray[typing.Any]:
    """A variable given for all supplied rows (or a column of the model's data), restricted to the subset used.

    Raises:
        metafopy.errors.InputError: the column is not in the data or the length does not match the data.
    """
    if isinstance(values, str):
        if x.data is None or values not in x.data.columns:
            raise metafopy.errors.InputError(f'cannot find the variable {values!r} (for {what}) in the data')
        arr = x.data[values].to_numpy()
    else:
        arr = numpy.asarray(values)
    t = x.tables
    if arr.shape[0] != t.k_all:
        raise metafopy.errors.InputError(f'Length of the {what} variable ({arr.shape[0]}) does not match the length of the data ({t.k_all}).')
    return arr[t.ids]


@metafopy.methods.leave1out.leave1out.register
def leave1out_fixed_binary(
    x: metafopy.models.binary.FixedBinaryResult,
    cluster: numpy.typing.ArrayLike | str | None = None,
    digits: int | None = None,
    transf: Transf = None,
    targs: typing.Any = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.leave1out.Leave1outResult:
    """Refit the model leaving out each table (or each ``cluster`` of tables) in turn.

    Raises:
        metafopy.errors.InputError: the data are not suitable (``k = 1``) or ``cluster`` is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    t = x.tables
    if t.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    if na_action == 'fail' and not t.not_na.all():
        raise metafopy.errors.InputError('Missing values in results.')
    miss_cluster = cluster is None
    if cluster is None:
        cluster_k = numpy.arange(t.k)
    else:
        cluster_f = per_table_variable(x, cluster, 'cluster')
        cluster_k = cluster_f[t.not_na]
    ids = list(dict.fromkeys(cluster_k.tolist()))
    if not miss_cluster:
        ids = sorted(ids)
    fits = [refit_tables(x, t.cells, cluster_k != cid) for cid in ids]
    fn = resolve_transf(x, transf)
    col = collect(fits, x, fn, targs)
    slab_all = numpy.asarray(t.slab, dtype=object)
    if miss_cluster:
        slab = tuple(f'-{s}' for s in slab_all[t.not_na])
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        if na_action in ('exclude', 'pass'):
            keys = ['estimate', 'se', 'stat', 'pval', 'ci_lb', 'ci_ub', 'Q', 'Qp', 'I2', 'H2']
            for key in keys:
                full = numpy.full(t.k_f, numpy.nan)
                full[t.not_na] = col[key]
                col[key] = full
            slab = tuple(f'-{s}' for s in slab_all)
    else:
        slab = tuple(f'-{cid}' for cid in ids)
    return metafopy.methods.leave1out.Leave1outResult(
        estimate=col['estimate'],
        se=col['se'],
        stat=col['stat'],
        pval=col['pval'],
        ci_lb=col['ci_lb'],
        ci_ub=col['ci_ub'],
        Q=col['Q'],
        Qp=col['Qp'],
        tau2=None,
        I2=col['I2'],
        H2=col['H2'],
        slab=slab,
        stat_name='zval',
        digits=x.digits if digits is None else digits,
        transf=fn is not None or transf is True,
    )


@metafopy.methods.leave1out.cumul.register
def cumul_fixed_binary(
    x: metafopy.models.binary.FixedBinaryResult,
    order: numpy.typing.ArrayLike | str | None = None,
    digits: int | None = None,
    transf: Transf = None,
    targs: typing.Any = None,
    collapse: bool = False,
    *,
    decreasing: bool = False,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.leave1out.CumulResult:
    """Fit the model to the first ``i`` tables for ``i = 1, ..., k`` (tables with missing counts are skipped).

    Raises:
        metafopy.errors.InputError: ``order`` is invalid or tables are missing and ``na_action='fail'``.
    """
    metafopy.methods.common.check_na_action(na_action)
    t = x.tables
    if na_action == 'fail' and not t.not_na.all():
        raise metafopy.errors.InputError('Missing values in data.')
    if order is None:
        orvar = pandas.Series(numpy.arange(t.k_f))
        collapse = False
    else:
        orvar = pandas.Series(per_table_variable(x, order, 'order'))
    sub = orvar.dropna().sort_values(ascending=not decreasing, kind='stable')
    srt = sub.index.to_numpy()
    ov = orvar.iloc[srt].to_numpy()
    uor = numpy.array(list(dict.fromkeys(ov.tolist())), dtype=ov.dtype) if collapse else ov
    cells_o = {n: v[srt] for n, v in t.cells_f.items()}
    not_na_o = t.not_na[srt]
    slab_o = numpy.asarray(t.slab, dtype=object)[srt]
    n = len(uor)
    fits: list[metafopy.models.binary.FixedBinaryResult | None] = [None] * n
    show = numpy.ones(n, dtype=numpy.bool_)
    for i in range(n):
        if collapse:
            members = numpy.isin(ov, uor[: i + 1])
            if not not_na_o[numpy.isin(ov, uor[i : i + 1])].any():
                show[i] = na_action != 'omit'
                continue
            incl = members
        else:
            if not not_na_o[i]:
                show[i] = na_action != 'omit'
                continue
            incl = numpy.arange(n) <= i
        fits[i] = refit_tables(x, cells_o, incl)
    fn = resolve_transf(x, transf)
    col = collect(fits, x, fn, targs)
    slab = tuple(metafopy.methods.leave1out.label(u) for u in uor[show]) if collapse else tuple(slab_o[show].tolist())
    return metafopy.methods.leave1out.CumulResult(
        estimate=col['estimate'][show],
        se=col['se'][show],
        stat=col['stat'][show],
        pval=col['pval'][show],
        ci_lb=col['ci_lb'][show],
        ci_ub=col['ci_ub'][show],
        Q=col['Q'][show],
        Qp=col['Qp'][show],
        tau2=None,
        I2=col['I2'][show],
        H2=col['H2'][show],
        slab=slab,
        stat_name='zval',
        digits=x.digits if digits is None else digits,
        transf=fn is not None or transf is True,
        k=col['k'][show],
        order=uor[show],
    )
