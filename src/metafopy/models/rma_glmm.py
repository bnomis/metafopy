"""Meta-analysis with generalized linear mixed-effects models (metafor's ``rma.glmm``).

Odds ratios (``measure='OR'``) of 2x2 tables and incidence rate ratios
(``'IRR'``) of two-group rate data are modelled with unconditional or
conditional models, single-group log odds (``'PLO'``) and log incidence rates
(``'IRLN'``) with logistic and Poisson mixed-effects models, following
Stijnen, Hamza and Ozdemir (2010), doi:10.1002/sim.3767. See
https://wviechtb.github.io/metafor/reference/rma.glmm.html and the source at
https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r.

metafor calls ``lme4::glmer()`` for the mixed models; here
:mod:`metafopy.core.glmm` maximises the same likelihood (adaptive
Gauss-Hermite quadrature or the Laplace approximation), so estimates agree
with metafor's to the (loose) convergence tolerance of lme4's optimiser.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.linalg
import metafopy.errors
import metafopy.metadata
import metafopy.models.glmm_fit
import metafopy.models.glmm_result
import metafopy.models.rma_uni
import metafopy.models.tables

__all__ = ['rma_glmm']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
SUPPORTED = ('OR', 'IRR', 'PLO', 'IRLN')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
EXPERIMENTAL = ('PR', 'RR', 'RD', 'PLN')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
MODELS = ('UM.FS', 'UM.RS', 'CM.EL', 'CM.AL')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
METHODS = ('FE', 'EE', 'CE', 'ML')


@dataclasses.dataclass(frozen=True, slots=True)
class Design:
    """Model matrices at the three stages of the NA handling (all tables, tables used in fitting, tables with valid ``yi``/``vi``)."""

    X: FloatArray
    X_f: FloatArray
    X_yi: FloatArray
    names: tuple[str, ...]
    int_incl: bool


def make_control(control: metafopy.models.glmm_fit.GLMMControl | collections.abc.Mapping[str, typing.Any] | None) -> metafopy.models.glmm_fit.GLMMControl:
    """Build the control settings; a mapping uses metafor's names (``.`` may be written as ``_``).

    Raises:
        metafopy.errors.InputError: an unknown control setting is given.
    """
    if control is None:
        return metafopy.models.glmm_fit.GLMMControl()
    if isinstance(control, metafopy.models.glmm_fit.GLMMControl):
        return control
    names = {f.name for f in dataclasses.fields(metafopy.models.glmm_fit.GLMMControl)}
    clean = {str(key).replace('.', '_').replace('scaleX', 'scale_x'): value for key, value in control.items()}
    unknown = set(clean) - names
    if unknown:
        raise metafopy.errors.InputError(f'unknown control settings: {sorted(unknown)}')
    return metafopy.models.glmm_fit.GLMMControl(**clean)


def first_of(value: typing.Any) -> typing.Any:
    """metafor uses only the first of several ``add``/``to`` values in ``rma.glmm``."""
    if isinstance(value, str) or not isinstance(value, collections.abc.Sequence):
        return value
    return value[0]


def check_arguments(measure: str | None, model: str | None, method: str, coding: float, test: str) -> str:
    """Validate the argument values and return the model after the ``CM.AL``/``IRR`` and one-group adjustments.

    Raises:
        metafopy.errors.InputError: an argument has an unknown value.
        metafopy.errors.NotSupportedError: the measure is one of the experimental ones that metafopy does not port.
    """
    if measure is None:
        raise metafopy.errors.InputError("Must specify the 'measure' argument.")
    if measure in EXPERIMENTAL:
        raise metafopy.errors.NotSupportedError(
            f"measure={measure!r} is experimental in metafor's rma.glmm() and not implemented; use 'OR', 'IRR', 'PLO' or 'IRLN'."
        )
    if measure not in SUPPORTED:
        raise metafopy.errors.InputError("Unknown 'measure' specified.")
    if method not in METHODS:
        raise metafopy.errors.InputError("Unknown 'method' specified.")
    if coding not in (0.5, 1, 0):
        raise metafopy.errors.InputError("Unknown 'coding' option specified.")
    if test not in ('z', 't'):
        raise metafopy.errors.InputError("Unknown option specified for the 'test' argument.")
    if model is not None and model not in MODELS:
        raise metafopy.errors.InputError("Unknown 'model' specified.")
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    if measure in ('PLO', 'IRLN'):
        if model is not None:
            warnings.warn("Argument 'model' not relevant for this outcome measure.", stacklevel=3)
        return 'UM.FS'
    resolved = model or 'UM.FS'
    return 'CM.EL' if (resolved == 'CM.AL' and measure == 'IRR') else resolved


def build_design(tab: metafopy.models.tables.TableSet, mods_all: FloatArray, names_all: tuple[str, ...]) -> Design:
    """Model matrices for the three NA stages, with redundant columns dropped and the intercept moved first (as ``rma.glmm`` does).

    Raises:
        metafopy.errors.InputError: no columns remain.
    """
    sub = mods_all if tab.subset is None else mods_all[tab.subset]
    X_f = sub
    X = X_f[tab.not_na]
    X_yi = X_f[tab.not_na_yivi]
    names = list(names_all)
    drop = metafopy.core.linalg.drop_redundant(X)
    if drop.any():
        warnings.warn('Redundant predictors dropped from the model.', stacklevel=3)
        X, X_f = X[:, ~drop], X_f[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]
    drop_yi = metafopy.core.linalg.drop_redundant(X_yi) if X_yi.shape[0] > 0 else numpy.zeros(X_yi.shape[1], dtype=numpy.bool_)
    X_yi = X_yi[:, ~drop_yi]
    names, X, X_f, int_incl = move_intercept(names, X, X_f)
    if X_yi.shape[0] > 0:
        _, X_yi, _, _ = move_intercept([f'x{j}' for j in range(X_yi.shape[1])], X_yi, X_yi.copy())
    return Design(X=X, X_f=X_f, X_yi=X_yi, names=tuple(names), int_incl=int_incl)


def move_intercept(names: list[str], X: FloatArray, X_f: FloatArray) -> tuple[list[str], FloatArray, FloatArray, bool]:
    """Put a single intercept column first, removing duplicates (``cbind(intrcpt=1, X[, -int.indx])``)."""
    is_int = numpy.array([metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])], dtype=numpy.bool_)
    if not is_int.any():
        return names, X, X_f, False
    k, k_f = X.shape[0], X_f.shape[0]
    new_names = ['intrcpt', *[n for n, i in zip(names, is_int, strict=True) if not i]]
    return new_names, numpy.column_stack([numpy.ones(k), X[:, ~is_int]]), numpy.column_stack([numpy.ones(k_f), X_f[:, ~is_int]]), True


def heterogeneity_tests(fit: metafopy.models.glmm_fit.ModelFit, k: int, p: int) -> tuple[float, float, float, float, int | None]:
    """Wald-type and likelihood ratio tests of the saturated model's extra coefficients: ``(Wld, p, LRT, p, df)``."""
    nan = math.nan
    if not (k > 1 and fit.qe_converged):
        return nan, nan, nan, nan, None
    if fit.qe_wld is not None:
        wld = fit.qe_wld
    elif fit.vb2_qe.shape[0] > 0:
        wld = wald_statistic(fit.b2_qe, fit.vb2_qe)
    else:
        wld = 0.0
    lrt = -2 * (fit.ll_fe - fit.ll_qe)
    wld = 0.0 if wld <= 0 else wld
    lrt = 0.0 if lrt <= 0 else lrt
    df = k - p
    if df > 0:
        return wld, float(scipy.stats.chi2.sf(wld, df)), lrt, float(scipy.stats.chi2.sf(lrt, df)), df
    return wld, 1.0, lrt, 1.0, df


def wald_statistic(b2: FloatArray, vb2: FloatArray) -> float:
    try:
        chol = numpy.linalg.cholesky(vb2)
    except numpy.linalg.LinAlgError:
        warnings.warn('Cannot invert the Hessian for the saturated model.', stacklevel=4)
        return math.nan
    z = numpy.linalg.solve(chol, b2)
    return float(z @ z)


def i2_h2(tab: metafopy.models.tables.TableSet, X_yi: FloatArray, tau2: float, p: int, i2def: str) -> tuple[float, float, float]:
    """``I^2``, ``H^2`` and the typical sampling variance ``vt`` from the ``yi``/``vi`` of the tables."""
    k_yi = tab.k_yi
    with numpy.errstate(all='ignore'):
        wi = 1.0 / tab.vi
        if i2def == '2':
            vt = float(1.0 / wi.mean())
        else:
            WX = wi[:, None] * X_yi
            P = numpy.diag(wi) - WX @ metafopy.core.linalg.invcalc(X_yi, wi) @ WX.T
            vt = (k_yi - p) / float(numpy.trace(P))
    return 100 * tau2 / (vt + tau2), tau2 / vt + 1, vt


def scale_design(X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, numpy.typing.NDArray[numpy.bool_]]:
    """Rescale the non-dummy columns (``scale()``); returns the new matrix, column means, standard deviations and the dummy flags."""
    is_d = numpy.array([bool(numpy.all((numpy.abs(X[:, j]) < 1e-8) | (numpy.abs(X[:, j] - 1) < 1e-8))) for j in range(X.shape[1])], dtype=numpy.bool_)
    mean_x = X[:, 1:].mean(axis=0)
    sd_x = X[:, 1:].std(axis=0, ddof=1)
    out = X.copy()
    cols = ~is_d
    out[:, cols] = (X[:, cols] - X[:, cols].mean(axis=0)) / X[:, cols].std(axis=0, ddof=1)
    return out, mean_x, sd_x, is_d


def back_transform(
    beta: FloatArray, vb: FloatArray, mean_x: FloatArray, sd_x: FloatArray, is_d: numpy.typing.NDArray[numpy.bool_]
) -> tuple[FloatArray, FloatArray]:
    """Undo the rescaling of the model matrix on the coefficients and their covariance matrix."""
    p = beta.shape[0]
    dummy = is_d[1:]
    mX = numpy.zeros((p, p))
    mX[0, 0] = 1.0
    mX[0, 1:] = -numpy.where(dummy, 0.0, mean_x / sd_x)
    mX[1:, 1:] = numpy.diag(numpy.where(dummy, 1.0, 1.0 / sd_x))
    return mX @ beta, mX @ vb @ mX.T


def omnibus_test(beta: FloatArray, vb: FloatArray, btt: tuple[int, ...]) -> float:
    idx = numpy.array(btt)
    try:
        chol = numpy.linalg.cholesky(vb[numpy.ix_(idx, idx)])
    except numpy.linalg.LinAlgError:
        warnings.warn('Cannot invert the Hessian for the QM-test.', stacklevel=3)
        return math.nan
    z = numpy.linalg.solve(chol, beta[idx])
    return float(z @ z)


def fit_statistics(ll: float, ll_qe: float, parms: int, k_eff: int) -> metafopy.models.rma_uni.FitStats:
    dev = -2 * (ll - ll_qe)
    big = max(k_eff, parms + 2)
    return metafopy.models.rma_uni.FitStats(
        ll=ll,
        dev=dev,
        AIC=-2 * ll + 2 * parms,
        BIC=-2 * ll + parms * math.log(k_eff),
        AICc=-2 * ll + 2 * parms * big / (big - parms - 1),
    )


def rma_glmm(
    *,
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    xi: metafopy.metadata.Column | None = None,
    mi: metafopy.metadata.Column | None = None,
    ti: metafopy.metadata.Column | None = None,
    ni: metafopy.metadata.Column | None = None,
    mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None = None,
    measure: str | None = None,
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | collections.abc.Sequence[float] = 0.5,
    to: str | collections.abc.Sequence[str] = 'only0',
    drop00: bool | collections.abc.Sequence[bool] = True,
    intercept: bool = True,
    model: str | None = None,
    method: str = 'ML',
    coding: float = 0.5,
    cor: bool = False,
    test: str = 'z',
    level: float = 95,
    btt: metafopy.models.rma_uni.Btt = None,
    nAGQ: int = 7,
    digits: int = 4,
    control: metafopy.models.glmm_fit.GLMMControl | collections.abc.Mapping[str, typing.Any] | None = None,
    na_action: metafopy.metadata.NAAction = 'omit',
    i2def: str = '1',
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> metafopy.models.glmm_result.GLMMResult:
    """Fit a generalized linear mixed-effects model to binary or count data.

    Mirrors metafor's ``rma.glmm()``; see
    https://wviechtb.github.io/metafor/reference/rma.glmm.html. ``measure`` is
    ``'OR'`` (2x2 tables: ``ai``, ``bi``, ``ci``, ``di`` or ``ai``, ``ci``,
    ``n1i``, ``n2i``), ``'IRR'`` (``x1i``, ``x2i``, ``t1i``, ``t2i``), ``'PLO'``
    (``xi`` and ``mi`` or ``ni``) or ``'IRLN'`` (``xi``, ``ti``). ``model`` is
    ``'UM.FS'`` (the default), ``'UM.RS'``, ``'CM.AL'`` or ``'CM.EL'`` and
    ``method`` ``'ML'`` or ``'FE'`` (``'EE'``, ``'CE'``). Only the first of
    several ``add`` and ``to`` values is used.

    The experimental measures ``'PR'``, ``'RR'``, ``'RD'`` and ``'PLN'`` of
    metafor, the ``GLMMadaptive``/``glmmTMB`` back-ends, the ``clogit`` and
    ``clogistic`` optimisers and ``link``/``family`` overrides are not
    available.

    Examples
    --------
    >>> from metafopy import datasets, rma_glmm
    >>> dat = datasets.load("bcg")  # doctest: +SKIP
    >>> rma_glmm(ai="tpos", bi="tneg", ci="cpos", di="cneg", data=dat, measure="OR")  # doctest: +SKIP

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the measure is not implemented.
        metafopy.errors.EstimationError: a model cannot be fitted.
    """
    model_name = check_arguments(measure, model, method, coding, test)
    if measure is None:
        raise metafopy.errors.InputError("Must specify the 'measure' argument.")
    con = make_control(control)
    add1, to1, drop1 = float(first_of(add)), str(first_of(to)), bool(first_of(drop00))
    if to1 not in metafopy.models.tables.TO_CHOICES:
        raise metafopy.errors.InputError("Unknown 'to' argument specified.")
    nagq = nAGQ
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    if measure in ('OR', 'IRR') and model_name == 'UM.RS' and method == 'ML' and nagq > 1:
        warnings.warn("Not possible to fit RE/ME model='UM.RS' with nAGQ > 1 with glmer(). nAGQ automatically set to 1.", stacklevel=2)
        nagq = 1

    # -- model matrix and table data --------------------------------------
    k_all = length_of(measure, ai, x1i, xi, data)
    if mods is None and not intercept:
        warnings.warn('Must either include an intercept and/or moderators in the model.\nCoerced an intercept into the model.', stacklevel=2)
    mods_all, names_all, mods_missing, _ = metafopy.metadata.MetaData.design(mods, intercept, data, k_all)
    tab = metafopy.models.tables.prepare_tables(
        measure,
        ai=ai,
        bi=bi,
        ci=ci,
        di=di,
        n1i=n1i,
        n2i=n2i,
        x1i=x1i,
        x2i=x2i,
        t1i=t1i,
        t2i=t2i,
        xi=xi,
        mi=mi,
        ni=ni,
        ti=ti,
        extra_missing=mods_missing,
        data=data,
        slab=slab,
        subset=subset,
        add=(add1, 0.0),
        to=(typing.cast(metafopy.models.tables.To, to1), 'none'),
        drop00=(drop1, drop1),
        onlyo1=onlyo1,
        addyi=addyi,
        addvi=addvi,
        na_action=na_action,
    )
    design = build_design(tab, mods_all, names_all)
    X, X_f, X_yi = design.X, design.X_f, design.X_yi
    k, p = X.shape
    names = metafopy.metadata.make_unique(design.names)
    int_incl = design.int_incl
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    if (method in ('FE', 'EE', 'CE') and p > k) or (method == 'ML' and p + 1 > k):
        raise metafopy.errors.InputError('Number of parameters to be estimated is larger than the number of observations.')
    btt_idx = metafopy.models.rma_uni.set_btt(btt, p, int_incl, names)
    m = len(btt_idx)
    if not metafopy.core.linalg.is_positive_definite(X.T @ X, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')

    scaled = not int_only and int_incl and con.scale_x
    mean_x = sd_x = numpy.zeros(0)
    is_d = numpy.zeros(0, dtype=numpy.bool_)
    X_fit = X
    if scaled:
        X_fit, mean_x, sd_x, is_d = scale_design(X)

    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    spec = metafopy.models.glmm_fit.FitSpec(measure, model_name, method, X_fit, tab.cells, coding, cor, nagq, alpha, con)
    fit = metafopy.models.glmm_fit.fit_model(spec)
    used_method = 'ML' if fit.method == 'ML' else method

    # -- tests and heterogeneity ----------------------------------------------
    qe_wld, qep_wld, qe_lrt, qep_lrt, qe_df = heterogeneity_tests(fit, k, p)
    i2, h2, vt = i2_h2(tab, X_yi, fit.tau2, p, i2def)
    beta, vb = fit.beta, fit.vb
    qm = omnibus_test(beta, vb, btt_idx)
    if scaled:
        beta, vb = back_transform(beta, vb, mean_x, sd_x, is_d)
    ddf = k - p if test == 't' else None
    ve = numpy.diag(vb)
    se = numpy.where(ve >= 0, numpy.sqrt(numpy.where(ve >= 0, ve, 0.0)), math.nan)
    zval = beta / se
    if ddf is not None:
        qm = qm / m
        qmdf: tuple[int, int | None] = (m, ddf)
        qmp = float(scipy.stats.f.sf(qm, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * scipy.stats.t.sf(numpy.abs(zval), ddf) if ddf > 0 else numpy.full(p, math.nan)
        crit = float(scipy.stats.t.isf(alpha / 2, ddf)) if ddf > 0 else math.nan
    else:
        qmdf = (m, None)
        qmp = float(scipy.stats.chi2.sf(qm, m))
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(alpha / 2))

    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    ll_ml = fit.ll_fe if method in ('FE', 'EE', 'CE') else fit.ll_ml
    stats_ml = fit_statistics(ll_ml, fit.ll_qe, fit.parms, fit.k_eff)
    nan_stats = metafopy.models.rma_uni.FitStats(math.nan, math.nan, math.nan, math.nan, math.nan)
    return metafopy.models.glmm_result.GLMMResult(
        beta=beta,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ci_lb=beta - crit * se,
        ci_ub=beta + crit * se,
        vb=vb,
        tau2=fit.tau2,
        se_tau2=fit.se_tau2,
        sigma2=fit.sigma2,
        rho=fit.rho,
        ci_lb_tau2=fit.ci_lb_tau2,
        ci_ub_tau2=fit.ci_ub_tau2,
        I2=i2,
        H2=h2,
        vt=vt,
        QE_Wld=qe_wld,
        QEp_Wld=qep_wld,
        QE_LRT=qe_lrt,
        QEp_LRT=qep_lrt,
        QE_df=qe_df,
        QM=qm,
        QMdf=qmdf,
        QMp=qmp,
        k=k,
        k_f=tab.k_f,
        k_yi=tab.k_yi,
        k_eff=fit.k_eff,
        k_all=tab.k_all,
        p=p,
        p_eff=fit.p_eff,
        parms=fit.parms,
        int_only=int_only,
        int_incl=int_incl,
        intercept=int_incl,
        X=X,
        X_f=X_f,
        coef_names=names,
        tables=tab,
        measure=measure,
        method=used_method,
        model=model_name,
        weighted=True,
        test=test,
        ddf=ddf,
        btt=btt_idx,
        m=m,
        digits=digits,
        level=alpha,
        add=add1,
        to=to1,
        drop00=drop1,
        fit_stats_ml=stats_ml,
        fit_stats_reml=nan_stats,
        se_warn=fit.se_warn,
        data=data,
    )


def length_of(
    measure: str,
    ai: metafopy.metadata.Column | None,
    x1i: metafopy.metadata.Column | None,
    xi: metafopy.metadata.Column | None,
    data: pandas.DataFrame | None,
) -> int:
    """Number of tables supplied (the length of the first count argument), needed to build the model matrix.

    Raises:
        metafopy.errors.InputError: the count arguments for the measure are missing.
    """
    first = {'OR': ai, 'IRR': x1i}.get(measure, xi)
    values = metafopy.metadata.column(first, data, 'ai' if measure == 'OR' else ('x1i' if measure == 'IRR' else 'xi'))
    if values is None:
        raise metafopy.errors.InputError('Must specify the table or rate data arguments for this measure.')
    return int(values.shape[0])
