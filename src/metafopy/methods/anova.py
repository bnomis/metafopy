"""Wald tests of model coefficients and likelihood-ratio tests between models."""

import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.linalg
import metafopy.core.padjust
import metafopy.errors
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

__all__ = ['LRTResult', 'MVLRTResult', 'WaldResult', 'anova']

FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class WaldResult:
    """Omnibus Wald test (``QM``) and, for linear combinations, individual tests."""

    QM: float
    QMdf: tuple[int, int | None]
    QMp: float
    btt: tuple[int, ...] | None = None
    Xb: FloatArray | None = None
    se: FloatArray | None = None
    zval: FloatArray | None = None
    pval: FloatArray | None = None

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
            body = f'F(df1 = {m}, df2 = {ddf}) = {metafopy.printing.fmt_num(self.QM)}, p-val {p}'
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
        return self.to_frame().round(4).to_string()


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
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float = 0.0,
    adjust: bool | str | None = None,
    refit: bool = False,
) -> WaldResult | LRTResult | list[WaldResult]:
    """Test coefficients of one model, or compare two nested models.

    ``anova(res, btt=...)`` tests a subset of coefficients (0-based indices, a
    name pattern, or a list of these for one test each); ``anova(res, X=L)``
    tests linear combinations ``L beta = rhs``; ``anova(full, reduced)`` gives
    a likelihood-ratio test. ``adjust`` (``True`` for ``'bonferroni'`` or one of
    :data:`metafopy.core.padjust.P_ADJUST_METHODS`) adjusts the p-values of
    several tests as R's ``p.adjust``; ``refit=True`` refits REML models that
    differ in their fixed effects with ML before the likelihood-ratio test.

    R source: https://github.com/wviechtb/metafor/blob/master/R/anova.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/anova.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.InternalError: an internal invariant does not hold.
    """
    x = model
    tdist = x.ddf is not None
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    if model2 is not None:
        return lrt(x, model2, refit)
    if is_btt_list(btt):
        return adjust_list([typing.cast(WaldResult, anova_rma_uni(x, btt=b)) for b in typing.cast(list[typing.Any], btt)], method)

    if X is not None:
        L = numpy.atleast_2d(numpy.asarray(X, dtype=numpy.float64))
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
        )

    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names) if btt is not None else x.btt
    sub = numpy.array(idx)
    m = len(idx)
    b = x.beta[sub] - btt_rhs(rhs, m)
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


def lrt(a: metafopy.models.rma_uni.RMAResult, b: metafopy.models.rma_uni.RMAResult, refit: bool = False) -> LRTResult:
    if a.parms == b.parms:
        raise metafopy.errors.InputError('models do not differ in the number of parameters')
    full, red = (a, b) if a.parms > b.parms else (b, a)
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
    df = full.parms - red.parms
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
        df_full=full.parms,
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
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float = 0.0,
    adjust: bool | str | None = None,
    refit: bool = False,
) -> WaldResult | MVLRTResult | list[WaldResult]:
    """Test coefficients of one ``rma.mv`` model, or compare two nested models by a likelihood-ratio test.

    ``adjust`` and ``refit`` are as in :func:`anova_rma_uni`; the ML refit uses :func:`metafopy.update`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    if model2 is not None:
        if x.parms == model2.parms:
            raise metafopy.errors.InputError('models do not differ in the number of parameters')
        full, red = (x, model2) if x.parms > model2.parms else (model2, x)
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
        df = full.parms - red.parms
        return MVLRTResult(
            df_full=full.parms,
            df_reduced=red.parms,
            LRT=LRT,
            pval=float(scipy.stats.chi2.sf(LRT, df)),
            fit_stats_full=fs_f.as_tuple(),
            fit_stats_reduced=fs_r.as_tuple(),
            QE_full=full.QE,
            QE_reduced=red.QE,
        )

    if X is not None:
        L = numpy.atleast_2d(numpy.asarray(X, dtype=numpy.float64))
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
        return WaldResult(QM=QM, QMdf=(m, df2), QMp=QMp, Xb=Xb, se=se, zval=zval, pval=numpy.asarray(pval, dtype=numpy.float64))

    if is_btt_list(btt):
        return adjust_list([typing.cast(WaldResult, anova_rma_mv(x, btt=b)) for b in typing.cast(list[typing.Any], btt)], method)
    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names) if btt is not None else x.btt
    out = mv_wald(x, idx, btt_rhs(rhs, len(idx)))
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
