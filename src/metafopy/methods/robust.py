"""Cluster-robust (sandwich) inference for ``rma.uni`` and ``rma.mv`` fits (metafor's ``robust()``).

``robust(x, cluster)`` replaces the model-based standard errors, tests and confidence intervals of a fit by cluster-robust ones
(Sidik and Jonkman, 2005, doi:10.1002/sim.2072; Hedges, Tipton and Johnson, 2010, doi:10.1002/jrsm.5). With
``club_sandwich=True`` the small-sample corrections of Tipton and Pustejovsky (2015, doi:10.3102/1076998615606099) and
Pustejovsky and Tipton (2018, doi:10.1080/07350015.2016.1247004) are used as implemented in the R package ``clubSandwich``
(see :mod:`metafopy.core.clubsandwich`). Reference: https://github.com/wviechtb/metafor/blob/master/R/robust.rma.uni.r
"""

import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse
import scipy.stats

import metafopy.core.clubsandwich
import metafopy.core.linalg
import metafopy.errors
import metafopy.methods.common
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.models.robust_info

__all__ = ['robust']

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.intp]
Cluster = str | collections.abc.Sequence[typing.Any] | numpy.typing.NDArray[typing.Any] | pandas.Series
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
COEF_TESTS = ('z', 'naive-t', 'naive-tp', 'Satterthwaite', 'saddlepoint')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
CONF_TESTS = ('z', 'naive-t', 'naive-tp', 'Satterthwaite')


@functools.singledispatch
def robust(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Cluster-robust inference for a fitted model.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'robust() not implemented for {type(model).__name__}')


def cluster_values(
    cluster: Cluster,
    data: pandas.DataFrame | None,
    k_all: int,
    used: numpy.typing.NDArray[numpy.bool_],
) -> numpy.typing.NDArray[numpy.object_]:
    """The cluster variable restricted to the rows used in fitting (``used`` has length ``k_all``).

    Raises:
        metafopy.errors.InputError: the variable cannot be found, has the wrong length or has missing values.
    """
    if isinstance(cluster, str):
        if data is None or cluster not in data.columns:
            raise metafopy.errors.InputError(f"Cannot find the 'cluster' variable {cluster!r}.")
        values = data[cluster].to_numpy()
    else:
        values = numpy.asarray(cluster)
    if values.shape[0] == int(used.sum()) and values.shape[0] != k_all:
        out = values
    elif values.shape[0] == k_all:
        out = values[used]
    else:
        raise metafopy.errors.InputError(
            f"Length of the variable specified via 'cluster' ({values.shape[0]}) does not correspond to the size of the original dataset ({k_all})."
        )
    if out.shape[0] == 0:
        raise metafopy.errors.InputError("Cannot find 'cluster' variable (or it has zero length).")
    if pandas.isna(out).any():
        raise metafopy.errors.InputError("No missing values allowed in 'cluster' variable.")
    return numpy.asarray(out, dtype=object)


def factorize(values: numpy.typing.NDArray[numpy.object_]) -> tuple[IntArray, int]:
    codes, uniques = pandas.factorize(pandas.Series(values), sort=True)
    return numpy.asarray(codes, dtype=numpy.intp), len(uniques)


def check_options(
    vcov: str | None,
    coef_test: str | None,
    conf_test: str | None,
    wald_test: str | None,
) -> tuple[str, str, str, str]:
    vcov = 'CR2' if vcov is None else vcov
    if vcov not in metafopy.core.clubsandwich.CR_TYPES:
        raise metafopy.errors.InputError(f"Unknown 'vcov' type {vcov!r}; use one of {', '.join(metafopy.core.clubsandwich.CR_TYPES)}.")
    coef_test = 'Satterthwaite' if coef_test is None else coef_test
    if coef_test not in COEF_TESTS:
        raise metafopy.errors.InputError(f"Unknown 'coef_test' {coef_test!r}.")
    if conf_test is None:
        conf_test = coef_test
        if conf_test == 'saddlepoint':
            conf_test = 'Satterthwaite'
            warnings.warn("Cannot use 'saddlepoint' for conf_test() - using 'Satterthwaite' instead.", stacklevel=4)
    elif conf_test not in CONF_TESTS:
        raise metafopy.errors.InputError(f"Unknown 'conf_test' {conf_test!r}.")
    wald_test = 'HTZ' if wald_test is None else wald_test
    if wald_test not in metafopy.core.clubsandwich.WALD_TESTS:
        raise metafopy.errors.InputError(f"Unknown 'wald_test' {wald_test!r}.")
    return vcov, coef_test, conf_test, wald_test


def sandwich_pieces(
    X: FloatArray,
    resid: FloatArray,
    beta: FloatArray,
    vcov_type: str,
    coef_test: str,
    conf_test: str,
    wald_test: str,
    alpha: float,
    btt: tuple[int, ...],
    int_only: bool,
    sw: metafopy.core.clubsandwich.Sandwich,
) -> dict[str, typing.Any]:
    """Coefficients' standard errors, tests, intervals and omnibus test from a clubSandwich variance estimator."""
    se = numpy.sqrt(numpy.diag(sw.vcov))
    tstat = beta / se
    if coef_test == 'saddlepoint':
        df_coef = numpy.full(len(beta), numpy.nan)
        pval = metafopy.core.clubsandwich.saddlepoint_p(sw, tstat)
    else:
        df_coef = metafopy.core.clubsandwich.coef_df(sw, coef_test)
        pval = numpy.array([2 * scipy.stats.norm.sf(abs(t)) if math.isinf(d) else 2 * scipy.stats.t.sf(abs(t), d) for t, d in zip(tstat, df_coef, strict=True)])
    df_conf = df_coef if conf_test == coef_test else metafopy.core.clubsandwich.coef_df(sw, conf_test)
    crit = numpy.array([scipy.stats.norm.isf(alpha / 2) if math.isinf(d) else scipy.stats.t.isf(alpha / 2, d) for d in df_conf])
    zval = numpy.where(numpy.isinf(tstat), numpy.nan, tstat)
    if int_only:
        QM = max(0.0, float(zval[0] ** 2))
        QMdf: tuple[int, float | None] = (1, float(df_coef[0]))
        QMp = float(pval[0])
    else:
        C = numpy.eye(len(beta))[list(btt)]
        try:
            res = metafopy.core.clubsandwich.wald_test(sw, beta, C, wald_test)
            QM, QMdf, QMp = max(0.0, res.Fstat), (res.df_num, max(0.0, res.df_denom)), res.p_val
        except numpy.linalg.LinAlgError, ValueError, ArithmeticError:
            warnings.warn('Could not obtain the cluster-robust omnibus Wald test.', stacklevel=4)
            QM, QMdf, QMp = math.nan, (len(btt), math.nan), math.nan
    dfs = numpy.where(numpy.isnan(df_coef), numpy.nan, df_coef)
    return {
        'se': se,
        'zval': zval,
        'pval': pval,
        'ci_lb': beta - crit * se,
        'ci_ub': beta + crit * se,
        'QM': QM,
        'QMdf': QMdf,
        'QMp': QMp,
        'dfs': dfs,
        'vb': sw.vcov,
    }


def cluster_counts(codes: IntArray, n: int) -> IntArray:
    return numpy.bincount(codes, minlength=n).astype(numpy.intp)


def default_inference(
    bread: FloatArray,
    resid: FloatArray,
    codes: IntArray,
    n: int,
    p: int,
    k: int,
    adjust: bool | str,
    alpha: float,
    beta: FloatArray,
    btt: tuple[int, ...],
    m: int,
) -> tuple[dict[str, typing.Any], str]:
    """metafor's default robust estimator ``bread %*% meat %*% t(bread)`` with the chosen small-sample adjustment.

    Raises:
        metafopy.errors.InputError: the adjustment is unknown.
    """
    dfs = n - p
    comps = numpy.column_stack([bread[:, codes == j] @ resid[codes == j] for j in range(n)])
    vb = comps @ comps.T
    vbest = 'CR0'
    if adjust is True:
        vb = (n / dfs) * vb
        vbest = 'CR1'
    elif isinstance(adjust, str):
        if adjust in ('Stata', 'Stata1'):
            vb = (n / (n - 1) * (k - 1) / (k - p)) * vb
            vbest = 'CR1.S1'
        elif adjust == 'Stata2':
            vb = (n / (n - 1)) * vb
            vbest = 'CR1.S2'
        else:
            raise metafopy.errors.InputError(f"Unknown 'adjust' value {adjust!r}.")
    is0 = numpy.diag(vb) < 100 * numpy.finfo(float).eps
    vb[is0, :] = numpy.nan
    vb[:, is0] = numpy.nan
    se = numpy.sqrt(numpy.diag(vb))
    zval = beta / se
    pval = 2 * scipy.stats.t.sf(numpy.abs(zval), dfs)
    crit = float(scipy.stats.t.isf(alpha / 2, dfs))
    sub = numpy.array(btt)
    try:
        sub_vb = vb[numpy.ix_(sub, sub)]
        c = numpy.linalg.cholesky(sub_vb)
        z = scipy.linalg.solve_triangular(c, beta[sub], lower=True)
        QM = float(z @ z)
    except numpy.linalg.LinAlgError:
        QM = math.nan
    if math.isnan(QM):
        warnings.warn('Could not obtain the cluster-robust omnibus Wald test.', stacklevel=4)
    QM /= m
    QMp = float(scipy.stats.f.sf(QM, m, dfs)) if not math.isnan(QM) else math.nan
    return {
        'se': se,
        'zval': zval,
        'pval': pval,
        'ci_lb': beta - crit * se,
        'ci_ub': beta + crit * se,
        'QM': QM,
        'QMdf': (m, float(dfs)),
        'QMp': QMp,
        'dfs': numpy.full(p, float(dfs)),
        'vb': vb,
    }, vbest


def check_adjust(adjust: bool | str) -> None:
    if isinstance(adjust, str) and adjust not in ('Stata', 'Stata1', 'Stata2'):
        raise metafopy.errors.InputError(f"Unknown 'adjust' value {adjust!r}.")


@robust.register
def robust_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    cluster: Cluster,
    *,
    adjust: bool | str = True,
    club_sandwich: bool | None = None,
    vcov: str | None = None,
    coef_test: str | None = None,
    conf_test: str | None = None,
    wald_test: str | None = None,
    clubSandwich: bool | None = None,
) -> metafopy.models.rma_uni.RMAResult:
    """Cluster-robust inference for an ``rma.uni`` fit.

    ``cluster`` is a column name of the data given to the model or a vector with one value per original row. ``adjust`` is
    ``True`` (CR1: ``n/(n-p)`` factor), ``False`` (CR0), ``'Stata1'``/``'Stata'`` or ``'Stata2'``. With ``club_sandwich=True``,
    ``vcov`` (``CR0``, ``CR1``, ``CR1p``, ``CR1S``, ``CR2``, ``CR3``; default ``CR2``), ``coef_test`` (``z``, ``naive-t``,
    ``naive-tp``, ``Satterthwaite``, ``saddlepoint``; the latter gives p-values only, the confidence intervals then use
    ``Satterthwaite`` unless ``conf_test`` says otherwise), ``conf_test`` (as ``coef_test`` without ``saddlepoint``) and
    ``wald_test`` (``chi-sq``, ``Naive-F``, ``Naive-Fp``, ``HTA``, ``HTB``, ``HTZ``, ``EDF``, ``EDT``) select the method.
    ``clubSandwich`` is R's spelling of ``club_sandwich``; giving both with different values is an error.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    club_sandwich = bool(metafopy.methods.common.resolve_alias(club_sandwich, clubSandwich, 'club_sandwich', 'clubSandwich'))
    x = model
    if x.data.data is None and isinstance(cluster, str):
        raise metafopy.errors.InputError("Cannot find the 'cluster' variable: the model was fitted without 'data'.")
    values = cluster_values(cluster, x.data.data, x.data.k_all, x.data.keep)
    codes, n = factorize(values)
    check_adjust(adjust)
    p, k = x.p, x.k
    dfs = n - p
    if not club_sandwich and dfs <= 0:
        raise metafopy.errors.InputError(f'Number of clusters ({n}) must be larger than the number of fixed effects ({p}).')
    X, resid = x.X, x.yi - x.X @ x.beta
    if club_sandwich:
        vcov_t, coef_t, conf_t, wald_t = check_options(vcov, coef_test, conf_test, wald_test)
        vi_tau = x.vi + x.tau2
        if x.weighted:
            wi = vi_tau**-1 if x.weights is None else x.weights.astype(numpy.float64)
            XWX = X.T @ (wi[:, None] * X)
        else:
            wi = numpy.ones(k)
            XWX = X.T @ X
        bread = metafopy.core.clubsandwich.chol2inv(XWX) * k
        w_scale = float(wi.mean())
        members = [numpy.flatnonzero(codes == j) for j in range(n)]
        W_list = [numpy.diag(wi[idx] / w_scale) for idx in members]
        inverse_var = x.weights is None and x.weighted
        theta = None if inverse_var else [numpy.diag(vi_tau[idx]) for idx in members]
        sw = metafopy.core.clubsandwich.vcov_cr(X, resid, codes, W_list, theta, bread, float(k), vcov_t, inverse_var, w_scale)
        pieces = sandwich_pieces(X, resid, x.beta, vcov_t, coef_t, conf_t, wald_t, x.level, x.btt, x.int_only, sw)
        vbest = vcov_t
        extra = {'coef_test': coef_t, 'conf_test': conf_t, 'wald_test': wald_t}
    else:
        if x.weighted:
            if x.weights is None:
                bread = x.vb @ (X.T * (x.vi + x.tau2) ** -1)
            else:
                A = x.weights.astype(numpy.float64)
                bread = metafopy.core.linalg.invcalc(X, A) @ (X.T * A)
        else:
            bread = metafopy.core.linalg.invcalc(X, numpy.ones(k)) @ X.T
        pieces, vbest = default_inference(bread, resid, codes, n, p, k, adjust, x.level, x.beta, x.btt, x.m)
        extra = {}
    info = metafopy.models.robust_info.RobustInfo(
        n=n,
        tcl=cluster_counts(codes, n),
        vbest=vbest,
        robumethod='clubSandwich' if club_sandwich else 'default',
        dfs=pieces['dfs'],
        cluster=values,
        **extra,
    )
    ddf: int | None = None if club_sandwich else dfs
    return dataclasses.replace(
        x,
        ddf=ddf,
        vb=pieces['vb'],
        se=pieces['se'],
        zval=pieces['zval'],
        pval=pieces['pval'],
        ci_lb=pieces['ci_lb'],
        ci_ub=pieces['ci_ub'],
        QM=pieces['QM'],
        QMdf=pieces['QMdf'],
        QMp=pieces['QMp'],
        test='t',
        s2w=1.0,
        robust=info,
    )


def nested_in_cluster(x: metafopy.models.mv_result.RMAMVResult, codes: IntArray) -> bool:
    """True when every random-effects grouping factor of the model is nested within the clusters."""
    factors: list[IntArray] = []
    factors += [numpy.asarray(f.codes) for f in x.s_factors]
    for grp in (x.group_g, x.group_h):
        if grp is not None:
            factors.append(numpy.asarray(grp.outer.codes))
    for f in factors:
        for level in numpy.unique(f[f >= 0]):
            if len(numpy.unique(codes[f == level])) > 1:
                return False
    return True


@robust.register
def robust_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    cluster: Cluster,
    *,
    adjust: bool | str = True,
    club_sandwich: bool | None = None,
    vcov: str | None = None,
    coef_test: str | None = None,
    conf_test: str | None = None,
    wald_test: str | None = None,
    clubSandwich: bool | None = None,
) -> metafopy.models.mv_result.RMAMVResult:
    """Cluster-robust inference for an ``rma.mv`` fit; the arguments are those of the ``rma.uni`` method.

    With ``club_sandwich=True`` (or R's spelling ``clubSandwich``) the random effects must be nested within the clusters and ``R=`` matrices are not allowed.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    club_sandwich = bool(metafopy.methods.common.resolve_alias(club_sandwich, clubSandwich, 'club_sandwich', 'clubSandwich'))
    x = model
    data = x.spec.get('data')
    used = numpy.zeros(x.k_all, dtype=numpy.bool_)
    used[numpy.array(x.ids, dtype=numpy.intp)[x.not_na]] = True
    values = cluster_values(cluster, data if isinstance(data, pandas.DataFrame) else None, x.k_all, used)
    codes, n = factorize(values)
    check_adjust(adjust)
    p, k = x.p, x.k
    dfs = n - p
    if not club_sandwich and dfs <= 0:
        raise metafopy.errors.InputError(f'Number of clusters ({n}) must be larger than the number of fixed effects ({p}).')
    X = x.X
    resid = x.yi - X @ x.beta
    alpha = metafopy.models.rma_uni.level_to_alpha(x.level)
    M = numpy.asarray(x.M.toarray() if scipy.sparse.issparse(x.M) else x.M, dtype=numpy.float64)
    A = None if x.W is None else numpy.asarray(x.W.toarray() if scipy.sparse.issparse(x.W) else x.W, dtype=numpy.float64)
    if club_sandwich:
        if any(x.Rfix) or x.with_r:
            raise metafopy.errors.InputError('vcovCR.rma.mv() does not work with fixed correlation matrices in the R argument.')
        if not nested_in_cluster(x, codes):
            raise metafopy.errors.InputError('Random effects are not nested within clustering variable.')
        vcov_t, coef_t, conf_t, wald_t = check_options(vcov, coef_test, conf_test, wald_test)
        members = [numpy.flatnonzero(codes == j) for j in range(n)]
        M_list = [M[numpy.ix_(idx, idx)] for idx in members]
        if A is None:
            W_list = [metafopy.core.clubsandwich.chol2inv(m_) for m_ in M_list]
            bread = k * x.vb / x.s2w
        else:
            W_list = [A[numpy.ix_(idx, idx)] for idx in members]
            bread = metafopy.core.clubsandwich.chol2inv(X.T @ A @ X) * k
        sw = metafopy.core.clubsandwich.vcov_cr(X, resid, codes, W_list, M_list, bread, float(k), vcov_t, A is None, 1.0)
        pieces = sandwich_pieces(X, resid, x.beta, vcov_t, coef_t, conf_t, wald_t, alpha, x.btt, x.int_only, sw)
        vbest = vcov_t
        extra = {'coef_test': coef_t, 'conf_test': conf_t, 'wald_test': wald_t}
    else:
        if A is None:
            try:
                bread = x.vb @ scipy.linalg.cho_solve(scipy.linalg.cho_factor(M), X).T
            except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError:
                raise metafopy.errors.EstimationError('Cannot invert marginal var-cov matrix.') from None
        else:
            bread = metafopy.core.clubsandwich.chol2inv(X.T @ A @ X) @ (X.T @ A)
        pieces, vbest = default_inference(bread, resid, codes, n, p, k, adjust, alpha, x.beta, x.btt, x.m)
        extra = {}
    info = metafopy.models.robust_info.RobustInfo(
        n=n,
        tcl=cluster_counts(codes, n),
        vbest=vbest,
        robumethod='clubSandwich' if club_sandwich else 'default',
        dfs=pieces['dfs'],
        cluster=values,
        **extra,
    )
    return dataclasses.replace(
        x,
        ddf=pieces['dfs'],
        dfs=float(dfs) if not club_sandwich else tuple(float(v) for v in pieces['dfs']),
        vb=pieces['vb'],
        se=pieces['se'],
        zval=pieces['zval'],
        pval=pieces['pval'],
        ci_lb=pieces['ci_lb'],
        ci_ub=pieces['ci_ub'],
        QM=pieces['QM'],
        QMdf=pieces['QMdf'],
        QMp=pieces['QMp'],
        test='t',
        s2w=1.0,
        robust=info,
    )
