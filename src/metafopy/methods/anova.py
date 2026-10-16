"""Wald tests of model coefficients and likelihood-ratio tests between models."""

import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.clubsandwich
import metafopy.core.linalg
import metafopy.core.padjust
import metafopy.errors
import metafopy.methods.common
import metafopy.mfopt
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.models.robust_info
import metafopy.models.tau2
import metafopy.printing

__all__ = ['LRTResult', 'MVLRTResult', 'WaldResult', 'anova']

FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class WaldResult:
    """Omnibus Wald test (``QM``) and, for linear combinations, individual tests."""

    QM: float
    QMdf: tuple[int, float | None]
    QMp: float
    btt: tuple[int, ...] | None = None
    Xb: FloatArray | None = None
    se: FloatArray | None = None
    zval: FloatArray | None = None
    pval: FloatArray | None = None
    ddf: FloatArray | None = None

    def __str__(self) -> str:
        m, ddf = self.QMdf
        head = (
            f'Test of Moderators (coefficient{"s" if m > 1 else ""} {metafopy.models.rma_uni.coef_span(self.btt)}):'
            if self.btt is not None
            else 'Omnibus Test of Hypotheses:'
        )
        p = metafopy.printing.fmt_pval(self.QMp)
        p = f'< {p[1:]}' if p.startswith('<') else f'= {p}'
        if ddf is None:
            body = f'QM(df = {m}) = {metafopy.printing.fmt_num(self.QM)}, p-val {p}'
        else:
            body = f'F(df1 = {m}, df2 = {metafopy.printing.fmt_df_scalar(ddf)}) = {metafopy.printing.fmt_num(self.QM)}, p-val {p}'
        return f'{head}\n{body}'


@dataclasses.dataclass(frozen=True, slots=True)
class LRTResult:
    """Likelihood-ratio test of a full against a reduced model."""

    df_full: int
    df_reduced: int
    LRT: float
    pval: float
    fit_stats_full: tuple[float, float, float, float, float]
    fit_stats_reduced: tuple[float, float, float, float, float]
    QE_full: float
    QE_reduced: float
    tau2_full: float
    tau2_reduced: float
    R2: float

    def to_frame(self) -> pandas.DataFrame:
        cols = ['df', 'AIC', 'BIC', 'AICc', 'logLik', 'LRT', 'pval', 'QE', 'tau^2', 'R^2']
        full = [
            self.df_full,
            self.fit_stats_full[2],
            self.fit_stats_full[3],
            self.fit_stats_full[4],
            self.fit_stats_full[0],
            math.nan,
            math.nan,
            self.QE_full,
            self.tau2_full,
            math.nan,
        ]
        red = [
            self.df_reduced,
            self.fit_stats_reduced[2],
            self.fit_stats_reduced[3],
            self.fit_stats_reduced[4],
            self.fit_stats_reduced[0],
            self.LRT,
            self.pval,
            self.QE_reduced,
            self.tau2_reduced,
            self.R2,
        ]
        return pandas.DataFrame([full, red], index=['Full', 'Reduced'], columns=cols)

    def __str__(self) -> str:
        frame = self.to_frame()
        digits = metafopy.mfopt.option_digits(None)

        def cell(column: str, value: float) -> str:
            if math.isnan(value):
                return ''
            if column == 'df':
                return str(int(value))
            if column == 'pval':
                return metafopy.printing.fmt_pval(value, digits)
            return metafopy.printing.fmt_num(value, digits) + ('%' if column == 'R^2' else '')

        rows = [[cell(c, float(v)) for c, v in zip(frame.columns, row, strict=True)] for row in frame.to_numpy()]
        return metafopy.printing.fmt_wrapped_table([str(c) for c in frame.columns], rows, [str(i) for i in frame.index])


@functools.singledispatch
def anova(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Wald or likelihood-ratio tests for fitted models.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'anova() not implemented for {type(model).__name__}')


@anova.register
def anova_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    model2: metafopy.models.rma_uni.RMAResult | None = None,
    *,
    object2: metafopy.models.rma_uni.RMAResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    adjust: bool | str | None = None,
    refit: bool = False,
    L: numpy.typing.ArrayLike | None = None,
    fixed: bool = False,
    test: str | None = None,
    df: int | None = None,
    verbose: bool = False,
) -> WaldResult | LRTResult | list[WaldResult]:
    """Test coefficients of one model, or compare two nested models.

    ``anova(res, btt=...)`` tests a subset of coefficients (0-based indices, a
    name pattern, or a list of these for one test each); ``anova(res, X=L)``
    tests linear combinations ``L beta = rhs``; ``anova(full, reduced)`` gives
    a likelihood-ratio test. ``adjust`` (``True`` for ``'bonferroni'`` or one of
    :data:`metafopy.core.padjust.P_ADJUST_METHODS`) adjusts the p-values of
    several tests as R's ``p.adjust``; ``refit=True`` refits REML models that
    differ in their fixed effects with ML before the likelihood-ratio test. ``object2`` is R's name for ``model2``.

    ``L`` is the old spelling of ``X`` and ``fixed=True`` matches a character ``btt`` literally instead of as a regular expression. For two
    models, ``test='Wald'`` tests the coefficients that only the full model has (with the Wald test of that model) instead of the
    likelihood-ratio test, and ``df`` replaces the difference in the number of parameters in the likelihood-ratio test. ``verbose`` adds
    the reason to the error of a cluster-robust test that fails.

    For a model made robust with ``club_sandwich=True`` the omnibus test is the cluster-robust Wald test with the small-sample
    approximation chosen by ``wald_test`` of :func:`metafopy.robust` (``HTZ`` by default: an F test with df1 and df2 from
    ``clubSandwich::Wald_test``), and the tests of linear combinations use the degrees of freedom of ``coef_test``. As in R,
    ``rhs`` cannot be used with ``X`` for such a model (a warning is given and zero is used).

    R source: https://github.com/wviechtb/metafor/blob/master/R/anova.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/anova.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.InternalError: an internal invariant does not hold.
    """
    x = model
    tdist = x.ddf is not None
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    model2 = metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2')
    X = merge_x(X, L)
    rhs_given = rhs is not None
    rhs = 0.0 if rhs is None else rhs
    if model2 is not None:
        if check_test(test) == 'Wald':
            return wald_between(x, model2, df)
        return lrt(x, model2, refit, df)
    club = club_info(x.robust)
    if is_btt_list(btt):
        return adjust_list([typing.cast(WaldResult, anova_rma_uni(x, btt=b, fixed=fixed)) for b in typing.cast(list[typing.Any], btt)], method)

    if X is not None and club is not None:
        return club_wald_x(x.beta, x.vb, club, X, x.p, x.int_incl, rhs_given, method, verbose)

    if X is not None:
        L = numpy.atleast_2d(numpy.asarray(X, dtype=numpy.float64))
        if x.int_incl and L.shape[1] == x.p - 1:
            L = numpy.column_stack([numpy.ones(L.shape[0]), L])
        if L.shape[1] != x.p:
            raise metafopy.errors.InputError(f'X must have {x.p} columns')
        m = L.shape[0]
        Xb = L @ x.beta - numpy.broadcast_to(numpy.asarray(rhs, dtype=numpy.float64), (m,))
        vXb = L @ x.vb @ L.T
        se = numpy.sqrt(numpy.diag(vXb))
        zval = Xb / se
        if tdist:
            if x.ddf is None:
                raise metafopy.errors.InternalError('x.ddf must not be None')
            pval = 2 * scipy.stats.t.sf(numpy.abs(zval), x.ddf)
        else:
            pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        QM = QMp = math.nan
        if numpy.linalg.matrix_rank(L) == m:
            try:
                QM = float(Xb @ numpy.linalg.solve(vXb, Xb))
            except numpy.linalg.LinAlgError:
                QM = math.nan
            if tdist:
                if x.ddf is None:
                    raise metafopy.errors.InternalError('x.ddf must not be None')
                QM /= m
                QMp = float(scipy.stats.f.sf(QM, m, x.ddf))
            else:
                QMp = float(scipy.stats.chi2.sf(QM, m))
        pval = numpy.asarray(pval, dtype=numpy.float64)
        if method is not None:
            pval = metafopy.core.padjust.p_adjust(pval, method)
        return WaldResult(
            QM=QM,
            QMdf=(m, x.ddf),
            QMp=QMp,
            Xb=Xb,
            se=se,
            zval=zval,
            pval=pval,
            ddf=None if x.ddf is None else numpy.full(m, float(x.ddf)),
        )

    idx = select_btt(btt, x.p, x.int_incl, x.coef_names, x.btt, fixed)
    sub = numpy.array(idx)
    m = len(idx)
    b = x.beta[sub] - btt_rhs(rhs, m)
    if club is not None:
        beta = x.beta.copy()
        beta[sub] = b
        out = club_wald_btt(beta, club, idx, verbose)
        if method is not None:
            out = dataclasses.replace(out, QMp=float(metafopy.core.padjust.p_adjust([out.QMp], method)[0]))
        return out
    try:
        QM = float(b @ numpy.linalg.solve(x.vb[numpy.ix_(sub, sub)], b))
    except numpy.linalg.LinAlgError:
        QM = math.nan
    if tdist:
        if x.ddf is None:
            raise metafopy.errors.InternalError('x.ddf must not be None')
        QM /= m
        QMp = float(scipy.stats.f.sf(QM, m, x.ddf)) if x.ddf > 0 else math.nan
    else:
        QMp = float(scipy.stats.chi2.sf(QM, m))
    if method is not None:
        QMp = float(metafopy.core.padjust.p_adjust([QMp], method)[0])
    return WaldResult(QM=QM, QMdf=(m, x.ddf), QMp=QMp, btt=idx)


def merge_x(X: numpy.typing.ArrayLike | None, L: numpy.typing.ArrayLike | None) -> numpy.typing.ArrayLike | None:
    """``L`` (the old spelling of ``X``) takes precedence over ``X``, as in R."""
    return X if L is None else L


def check_test(test: str | None) -> str:
    """R's ``match.arg(ddd$test, c('LRT', 'Wald'))``.

    Raises:
        metafopy.errors.ArgumentError: ``test`` is neither ``'LRT'`` nor ``'Wald'``.
    """
    if test is None:
        return 'LRT'
    for choice in ('LRT', 'Wald'):
        if test and choice.startswith(test):
            return choice
    raise metafopy.errors.ArgumentError('\'arg\' should be one of "LRT", "Wald"')


def select_btt(
    btt: metafopy.models.rma_uni.Btt, p: int, int_incl: bool, names: collections.abc.Sequence[str], default: tuple[int, ...], fixed: bool
) -> tuple[int, ...]:
    """The coefficients ``btt`` selects (``.set.btt``); ``fixed`` matches a character ``btt`` literally."""
    if btt is None:
        return default
    if fixed and isinstance(btt, str):
        found = tuple(i for i, n in enumerate(names) if btt in n)
        if not found:
            raise metafopy.errors.InputError("Cannot identify coefficient(s) corresponding to the specified 'btt' string.")
        return found
    return metafopy.models.rma_uni.set_btt(btt, p, int_incl, names)


def club_info(info: metafopy.models.robust_info.RobustInfo | None) -> metafopy.models.robust_info.RobustInfo | None:
    """The robust information of a fit made robust with ``club_sandwich=True`` (else ``None``)."""
    if info is None or info.robumethod != 'clubSandwich' or info.sandwich is None:
        return None
    return info


def club_wald_btt(beta: FloatArray, info: metafopy.models.robust_info.RobustInfo, idx: tuple[int, ...], verbose: bool) -> WaldResult:
    """The cluster-robust Wald test of the coefficients ``idx`` (``clubSandwich::Wald_test`` with ``constrain_zero``) of ``anova.rma``.

    Raises:
        metafopy.errors.InputError: the test cannot be computed.
    """
    if info.sandwich is None:
        raise metafopy.errors.InternalError('a clubSandwich fit must keep its variance estimator')
    C = numpy.eye(len(beta))[list(idx)]
    try:
        res = metafopy.core.clubsandwich.wald_test(info.sandwich, beta, C, info.wald_test or 'HTZ')
    except (numpy.linalg.LinAlgError, ValueError, ArithmeticError) as err:
        raise metafopy.errors.InputError(
            'Could not obtain the cluster-robust Wald test' + (f' ({err}).' if verbose else ' (use verbose=True for more details).')
        ) from err
    return WaldResult(QM=max(0.0, res.Fstat), QMdf=(res.df_num, res.df_denom), QMp=res.p_val, btt=idx)


def club_wald_x(
    beta: FloatArray,
    vb: FloatArray,
    info: metafopy.models.robust_info.RobustInfo,
    X: numpy.typing.ArrayLike,
    p: int,
    int_incl: bool,
    rhs_given: bool,
    method: str | None,
    verbose: bool,
) -> WaldResult:
    """Tests of linear combinations of a ``clubSandwich`` fit: ``clubSandwich::linear_contrast`` and ``Wald_test`` of ``anova.rma``.

    Raises:
        metafopy.errors.InputError: the contrasts do not fit the model or the tests cannot be computed.
    """
    sw = info.sandwich
    if sw is None:
        raise metafopy.errors.InternalError('a clubSandwich fit must keep its variance estimator')
    C = numpy.atleast_2d(numpy.asarray(X, dtype=numpy.float64))
    if int_incl and C.shape[1] == p - 1:
        C = numpy.column_stack([numpy.ones(C.shape[0]), C])
    if C.shape[1] != p:
        raise metafopy.errors.InputError(f"Length or number of columns of 'X' ({C.shape[1]}) does not match the number of model coefficients ({p}).")
    m = C.shape[0]
    try:
        ddf = metafopy.core.clubsandwich.contrast_df(sw, C, info.coef_test or 'Satterthwaite')
    except metafopy.errors.InputError as err:
        raise metafopy.errors.InputError(
            'Could not obtain the cluster-robust test(s)' + (f' ({err}).' if verbose else ' (use verbose=True for more details).')
        ) from err
    if rhs_given:
        warnings.warn("Cannot use 'rhs' argument for 'robust.rma' objects based on 'clubSandwich'.", stacklevel=3)
    Xb = C @ beta
    se = numpy.sqrt(numpy.diag(C @ vb @ C.T))
    zval = Xb / se
    pval = numpy.array([2 * scipy.stats.norm.sf(abs(t)) if math.isinf(d) else 2 * scipy.stats.t.sf(abs(t), d) for t, d in zip(zval, ddf, strict=True)])
    QM = QMp = math.nan
    QMdf: tuple[int, float | None] = (m, None)
    if numpy.linalg.matrix_rank(C) == m:
        try:
            res = metafopy.core.clubsandwich.wald_test(sw, beta, C, info.wald_test or 'HTZ')
        except (numpy.linalg.LinAlgError, ValueError, ArithmeticError) as err:
            raise metafopy.errors.InputError(
                'Could not obtain the cluster-robust omnibus Wald test' + (f' ({err}).' if verbose else ' (use verbose=True for more details).')
            ) from err
        QM, QMdf, QMp = max(0.0, res.Fstat), (res.df_num, res.df_denom), res.p_val
    if method is not None:
        pval = metafopy.core.padjust.p_adjust(pval, method)
    return WaldResult(QM=QM, QMdf=QMdf, QMp=QMp, Xb=Xb, se=se, zval=zval, pval=numpy.asarray(pval, dtype=numpy.float64), ddf=ddf)


def wald_between(
    a: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    b: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    df: int | None,
) -> WaldResult:
    """``anova(full, reduced, test='Wald')``: the Wald test of the coefficients that only the full model has.

    Raises:
        metafopy.errors.InputError: the models cannot be compared.
    """
    if type(a) is not type(b):
        raise metafopy.errors.InputError("Class of 'object' must be the same as class of 'object2'.")
    if df is None and a.parms == b.parms:
        raise metafopy.errors.InputError('Models have the same number of parameters.')
    full, red = (a, b) if df is not None or a.parms > b.parms else (b, a)
    if full.k != red.k or not numpy.allclose(full.yi, red.yi):
        raise metafopy.errors.InputError('The observed outcomes and/or sampling variances are not equal in the full and reduced model.')
    if full.method != red.method:
        raise metafopy.errors.InputError("Full and reduced model must use the same 'method' for the model fitting.")
    extra = [n for n in full.coef_names if n not in red.coef_names]
    if not extra:
        raise metafopy.errors.InputError('Full and reduced models appear to contain the same moderators.')
    if any(n not in full.coef_names for n in red.coef_names):
        raise metafopy.errors.InputError('There are coefficients in the reduced model that are not in the full model.')
    idx = [full.coef_names.index(n) for n in extra]
    out = anova(full, btt=idx)
    return typing.cast(WaldResult, out)


def is_btt_list(btt: typing.Any) -> bool:
    """Whether ``btt`` asks for one test per element (metafor's list ``btt``, or a character vector of length > 1)."""
    if not isinstance(btt, list | tuple):
        return False
    if all(isinstance(b, str) for b in btt):
        return len(btt) > 1
    return any(not isinstance(b, int | numpy.integer) for b in btt)


def adjust_list(results: list[WaldResult], method: str | None) -> list[WaldResult]:
    if method is None:
        return results
    adjusted = metafopy.core.padjust.p_adjust([r.QMp for r in results], method)
    return [dataclasses.replace(r, QMp=float(p)) for r, p in zip(results, adjusted, strict=True)]


def btt_rhs(rhs: numpy.typing.ArrayLike | float, m: int) -> FloatArray:
    """``rhs`` recycled to the ``m`` tested coefficients.

    Raises:
        metafopy.errors.InputError: the length of ``rhs`` does not match.
    """
    r = numpy.atleast_1d(numpy.asarray(rhs, dtype=numpy.float64))
    if r.size == 1:
        r = numpy.full(m, float(r[0]))
    if r.size != m:
        raise metafopy.errors.InputError(f"Length of 'rhs' ({r.size}) does not match the number of coefficients tested ({m}).")
    return r


def refit_ml(res: metafopy.models.rma_uni.RMAResult) -> metafopy.models.rma_uni.RMAResult:
    """``res`` refitted with ML, as ``anova.rma(..., refit=TRUE)`` does for ``rma.uni`` models.

    Raises:
        metafopy.errors.InputError: the refit failed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = metafopy.models.rma_uni.rma_uni(
                yi=res.yi,
                vi=res.vi,
                weights=res.weights,
                mods=res.X,
                intercept=False,
                method='ML',
                weighted=res.weighted,
                test=res.test,
                level=res.level,
                tau2=res.tau2 if res.tau2_fix else None,
                control=res.control,
            )
    except (metafopy.errors.MetaforError, numpy.linalg.LinAlgError, ValueError) as err:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.') from err
    return out


def lrt(a: metafopy.models.rma_uni.RMAResult, b: metafopy.models.rma_uni.RMAResult, refit: bool = False, df_given: int | None = None) -> LRTResult:
    if df_given is None and a.parms == b.parms:
        raise metafopy.errors.InputError('models do not differ in the number of parameters')
    full, red = (a, b) if df_given is not None or a.parms > b.parms else (b, a)
    if full.k != red.k or not numpy.allclose(full.yi, red.yi):
        raise metafopy.errors.InputError('models are not fitted to the same data')
    same_X = full.X.shape == red.X.shape and numpy.allclose(full.X, red.X)
    if full.method == 'REML' and not same_X:
        if refit:
            full, red = refit_ml(full), refit_ml(red)
        else:
            warnings.warn(
                "REML comparisons not meaningful for models with different fixed effects (use 'refit=True' to refit both models based on ML estimation).",
                stacklevel=3,
            )
    use_reml = full.method == 'REML'
    fs_f = full.fit_stats_reml if use_reml else full.fit_stats_ml
    fs_r = red.fit_stats_reml if use_reml else red.fit_stats_ml
    LRT = max(0.0, fs_r.dev - fs_f.dev)
    df = full.parms - red.parms if df_given is None else df_given
    parms_full = full.parms + (0 if df_given is None else df_given)
    pval = float(scipy.stats.chi2.sf(LRT, df))

    if full.method in metafopy.models.tau2.FIXED_EFFECTS and red.method in metafopy.models.tau2.FIXED_EFFECTS:
        s2f = sigma2(full)
        s2r = sigma2(red)
        R2 = 100 * max(0.0, (s2r - s2f) / s2r)
    elif red.tau2 == 0:
        R2 = 0.0
    else:
        R2 = 100 * max(0.0, (red.tau2 - full.tau2) / red.tau2)

    return LRTResult(
        df_full=parms_full,
        df_reduced=red.parms,
        LRT=LRT,
        pval=pval,
        fit_stats_full=fs_f.as_tuple(),
        fit_stats_reduced=fs_r.as_tuple(),
        QE_full=full.QE,
        QE_reduced=red.QE,
        tau2_full=full.tau2,
        tau2_reduced=red.tau2,
        R2=R2,
    )


def sigma2(res: metafopy.models.rma_uni.RMAResult) -> float:
    """Residual variance of ``lm(yi ~ X, weights)`` as used for EE model R^2 in metafor."""
    n = res.k
    w = (1.0 / res.vi if res.weights is None else res.weights) if res.weighted else numpy.ones(n)
    Xi = numpy.column_stack([numpy.ones(n), res.X])
    Xi = Xi[:, ~metafopy.core.linalg.drop_redundant(Xi)]
    sw = numpy.sqrt(w)
    coef, *_ = numpy.linalg.lstsq(Xi * sw[:, None], res.yi * sw, rcond=None)
    e = res.yi - Xi @ coef
    return float(numpy.sum(w * e**2) / (n - Xi.shape[1]))


@dataclasses.dataclass(frozen=True, slots=True)
class MVLRTResult:
    """Likelihood-ratio test of a full against a reduced ``rma.mv`` model."""

    df_full: int
    df_reduced: int
    LRT: float
    pval: float
    fit_stats_full: tuple[float, float, float, float, float]
    fit_stats_reduced: tuple[float, float, float, float, float]
    QE_full: float
    QE_reduced: float

    def __str__(self) -> str:
        header = ['df', 'AIC', 'BIC', 'AICc', 'logLik', 'LRT', 'pval', 'QE']
        fs_f, fs_r = self.fit_stats_full, self.fit_stats_reduced
        fmt = metafopy.printing.fmt_num
        rows = {
            'Full': [str(self.df_full), fmt(fs_f[2]), fmt(fs_f[3]), fmt(fs_f[4]), fmt(fs_f[0]), '', '', fmt(self.QE_full)],
            'Reduced': [
                str(self.df_reduced),
                fmt(fs_r[2]),
                fmt(fs_r[3]),
                fmt(fs_r[4]),
                fmt(fs_r[0]),
                fmt(self.LRT),
                metafopy.printing.fmt_pval(self.pval),
                fmt(self.QE_reduced),
            ],
        }
        width = [max(len(h), *(len(r[i]) for r in rows.values())) for i, h in enumerate(header)]
        lines = [' ' * 7 + ' ' + ' '.join(h.rjust(w) for h, w in zip(header, width, strict=True))]
        lines += [f'{name:<7} ' + ' '.join(c.rjust(w) for c, w in zip(cells, width, strict=True)) for name, cells in rows.items()]
        return '\n'.join(lines)


def mv_wald(x: metafopy.models.mv_result.RMAMVResult, idx: tuple[int, ...], rhs: FloatArray | None = None) -> WaldResult:
    sub = numpy.array(idx)
    m = len(idx)
    b = x.beta[sub] if rhs is None else x.beta[sub] - rhs
    try:
        QM = float(b @ numpy.linalg.solve(x.vb[numpy.ix_(sub, sub)], b))
    except numpy.linalg.LinAlgError:
        QM = math.nan
    if x.test == 'z':
        return WaldResult(QM=QM, QMdf=(m, None), QMp=float(scipy.stats.chi2.sf(QM, m)), btt=idx)
    ddf = int(numpy.min(x.ddf[sub]))
    QM /= m
    return WaldResult(QM=QM, QMdf=(m, ddf), QMp=float(scipy.stats.f.sf(QM, m, ddf)) if ddf > 0 else math.nan, btt=idx)


@anova.register
def anova_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    model2: metafopy.models.mv_result.RMAMVResult | None = None,
    *,
    object2: metafopy.models.mv_result.RMAMVResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    adjust: bool | str | None = None,
    refit: bool = False,
    L: numpy.typing.ArrayLike | None = None,
    fixed: bool = False,
    test: str | None = None,
    df: int | None = None,
    verbose: bool = False,
) -> WaldResult | MVLRTResult | list[WaldResult]:
    """Test coefficients of one ``rma.mv`` model, or compare two nested models by a likelihood-ratio test.

    ``adjust``, ``refit``, ``L``, ``fixed``, ``test``, ``df``, ``verbose`` and the handling of ``club_sandwich=True`` fits are as in
    :func:`anova_rma_uni`; the ML refit uses :func:`metafopy.update`. ``object2`` is R's name for ``model2``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    model2 = metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2')
    X = merge_x(X, L)
    rhs_given = rhs is not None
    rhs = 0.0 if rhs is None else rhs
    club = club_info(x.robust)
    if model2 is not None:
        if check_test(test) == 'Wald':
            return wald_between(x, model2, df)
        if df is None and x.parms == model2.parms:
            raise metafopy.errors.InputError('models do not differ in the number of parameters')
        full, red = (x, model2) if df is not None or x.parms > model2.parms else (model2, x)
        if full.k != red.k or not numpy.allclose(full.yi, red.yi):
            raise metafopy.errors.InputError('models are not fitted to the same data')
        use_reml = full.method == 'REML'
        if use_reml and not (full.X.shape == red.X.shape and numpy.allclose(full.X, red.X)):
            if refit:
                full, red = refit_ml_mv(full), refit_ml_mv(red)
                use_reml = False
            else:
                warnings.warn(
                    "REML comparisons not meaningful for models with different fixed effects (use 'refit=True' to refit both models based on ML estimation).",
                    stacklevel=2,
                )
        fs_f = full.fit_stats_reml if use_reml else full.fit_stats_ml
        fs_r = red.fit_stats_reml if use_reml else red.fit_stats_ml
        LRT = max(0.0, fs_r.dev - fs_f.dev)
        df_diff = full.parms - red.parms if df is None else df
        return MVLRTResult(
            df_full=full.parms + (0 if df is None else df),
            df_reduced=red.parms,
            LRT=LRT,
            pval=float(scipy.stats.chi2.sf(LRT, df_diff)),
            fit_stats_full=fs_f.as_tuple(),
            fit_stats_reduced=fs_r.as_tuple(),
            QE_full=full.QE,
            QE_reduced=red.QE,
        )

    if X is not None and club is not None:
        return club_wald_x(x.beta, x.vb, club, X, x.p, x.int_incl, rhs_given, method, verbose)

    if X is not None:
        L = numpy.atleast_2d(numpy.asarray(X, dtype=numpy.float64))
        if x.int_incl and L.shape[1] == x.p - 1:
            L = numpy.column_stack([numpy.ones(L.shape[0]), L])
        if L.shape[1] != x.p:
            raise metafopy.errors.InputError(f'X must have {x.p} columns')
        m = L.shape[0]
        Xb = L @ x.beta - numpy.broadcast_to(numpy.asarray(rhs, dtype=numpy.float64), (m,))
        vXb = L @ x.vb @ L.T
        se = numpy.sqrt(numpy.diag(vXb))
        zval = Xb / se
        ddf = numpy.array([float(numpy.min(x.ddf[row != 0])) if numpy.any(row != 0) else math.nan for row in L])
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval)) if x.test == 'z' else 2 * scipy.stats.t.sf(numpy.abs(zval), ddf)
        if method is not None:
            pval = metafopy.core.padjust.p_adjust(numpy.asarray(pval, dtype=numpy.float64), method)
        QM = QMp = math.nan
        df2: int | None = None if x.test == 'z' else int(numpy.nanmin(ddf))
        if numpy.linalg.matrix_rank(L) == m:
            QM = float(Xb @ numpy.linalg.solve(vXb, Xb))
            if df2 is None:
                QMp = float(scipy.stats.chi2.sf(QM, m))
            else:
                QM /= m
                QMp = float(scipy.stats.f.sf(QM, m, df2))
        return WaldResult(
            QM=QM, QMdf=(m, df2), QMp=QMp, Xb=Xb, se=se, zval=zval, pval=numpy.asarray(pval, dtype=numpy.float64), ddf=None if x.test == 'z' else ddf
        )

    if is_btt_list(btt):
        return adjust_list([typing.cast(WaldResult, anova_rma_mv(x, btt=b, fixed=fixed)) for b in typing.cast(list[typing.Any], btt)], method)
    idx = select_btt(btt, x.p, x.int_incl, x.coef_names, x.btt, fixed)
    rhs_btt = btt_rhs(rhs, len(idx))
    if club is not None:
        beta = x.beta.copy()
        beta[numpy.array(idx)] -= rhs_btt
        out = club_wald_btt(beta, club, idx, verbose)
    else:
        out = mv_wald(x, idx, rhs_btt)
    if method is not None:
        out = dataclasses.replace(out, QMp=float(metafopy.core.padjust.p_adjust([out.QMp], method)[0]))
    return out


def refit_ml_mv(res: metafopy.models.mv_result.RMAMVResult) -> metafopy.models.mv_result.RMAMVResult:
    """``res`` refitted with ML via ``update(res, method='ML')``, as ``anova.rma(..., refit=TRUE)`` does.

    Raises:
        metafopy.errors.InputError: the model has no recorded call or the refit failed.
    """
    if res.call is None:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed (no recorded call).')
    try:
        out = res.call(method='ML')
    except (metafopy.errors.MetaforError, numpy.linalg.LinAlgError, ValueError) as err:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.') from err
    if not isinstance(out, metafopy.models.mv_result.RMAMVResult):
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.')
    return out
