"""Wald tests and likelihood-ratio tests for location-scale models (metafor's ``anova.rma`` for ``rma.ls`` objects).

Wald tests of subsets of the location coefficients (``btt``), of the scale coefficients (``att``) or of linear combinations
of either (``X`` and ``Z``), and likelihood-ratio tests of nested models (Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1564).
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.ls_common
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['LSWaldResult', 'anova_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Sequences = collections.abc.Sequence[typing.Any]


@dataclasses.dataclass(frozen=True, slots=True)
class LSWaldResult:
    """Wald test in a location-scale model.

    ``part`` is ``'location'`` (``btt`` or ``X``) or ``'scale'`` (``att`` or ``Z``); ``stat`` is the omnibus statistic (``QM`` or
    ``QS``) with degrees of freedom ``df`` and p-value ``pval_omnibus``. For linear combinations (``X``, ``Z``) ``hyp`` describes
    each hypothesis and ``estimate``, ``se``, ``zval`` and ``pval`` hold the individual tests (``ddf`` the degrees of freedom of
    the t-tests); for coefficient subsets ``idx`` holds the 0-based indices tested.
    """

    part: str
    stat: float
    df: tuple[int, float | None]
    pval_omnibus: float
    test: str
    m: int
    idx: tuple[int, ...] | None = None
    hyp: tuple[str, ...] | None = None
    estimate: FloatArray | None = None
    se: FloatArray | None = None
    zval: FloatArray | None = None
    pval: FloatArray | None = None
    ddf: FloatArray | None = None

    @property
    def QM(self) -> float:
        return self.stat if self.part == 'location' else math.nan

    @property
    def QMp(self) -> float:
        return self.pval_omnibus if self.part == 'location' else math.nan

    @property
    def QS(self) -> float:
        return self.stat if self.part == 'scale' else math.nan

    @property
    def QSp(self) -> float:
        return self.pval_omnibus if self.part == 'scale' else math.nan

    def stat_line(self, digits: int = 4) -> str:
        name = 'QM' if self.part == 'location' else 'QS'
        p = metafopy.models.rma_uni.pfmt(self.pval_omnibus, digits)
        if self.df[1] is not None:
            return f'F(df1 = {self.df[0]}, df2 = {int(self.df[1])}) = {metafopy.printing.fmt_num(self.stat, digits)}, p-val {p}'
        return f'{name}(df = {self.df[0]}) = {metafopy.printing.fmt_num(self.stat, digits)}, p-val {p}'

    def __str__(self) -> str:
        digits = 4
        plural = 's' if self.m > 1 else ''
        if self.idx is not None:
            what = 'Location' if self.part == 'location' else 'Scale'
            return f'\nTest of {what} Coefficients (coefficient{plural} {metafopy.models.rma_uni.coef_span(self.idx)}):\n{self.stat_line(digits)}\n'
        if self.hyp is None or self.estimate is None or self.se is None or self.zval is None or self.pval is None:
            raise metafopy.errors.InternalError('a test of linear combinations needs its individual tests')
        width = max(len(h) for h in self.hyp)
        names = [f'{j + 1}:' for j in range(self.m)]
        lines = ['Hypothesis:' if self.m == 1 else 'Hypotheses:']
        lines += [f'{n} {h.rjust(width)} ' for n, h in zip(names, self.hyp, strict=True)]
        t_test = self.ddf is not None
        header = ['estimate', 'se', 'tval' if t_test else 'zval']
        cols = [
            [metafopy.printing.fmt_num(float(v), digits) for v in self.estimate],
            [metafopy.printing.fmt_num(float(v), digits) for v in self.se],
            [metafopy.printing.fmt_num(float(v), digits) for v in self.zval],
        ]
        if self.ddf is not None:
            header.append('df')
            cols.append(metafopy.printing.fmt_df(self.ddf))
        header.append('pval')
        cols.append([metafopy.printing.fmt_pval(float(v), digits) for v in self.pval])
        header.append('')
        cols.append(metafopy.methods.ls_common.center_stars(float(v) for v in self.pval))
        widths = [max(len(header[j]), *(len(c) for c in cols[j])) for j in range(len(header))]
        name_w = len(names[-1]) if names else 0
        lines2 = ['', 'Results:']
        lines2.append(' ' * name_w + ' ' + ' '.join(header[j].rjust(widths[j]) for j in range(len(header))))
        for i in range(self.m):
            lines2.append(names[i].ljust(name_w) + ' ' + ' '.join(cols[j][i].rjust(widths[j]) for j in range(len(header))))
        out = '\n'.join(lines + lines2)
        if not math.isnan(self.stat):
            out += '\n\n' + ('Test of Hypothesis:' if self.m == 1 else 'Omnibus Test of Hypotheses:') + '\n' + self.stat_line(digits)
        return '\n' + out + '\n'


def solve_quad(b: FloatArray, v: FloatArray) -> float:
    """``b' V^-1 b`` (NaN when ``V`` is not positive definite)."""
    try:
        chol = numpy.linalg.cholesky(v)
    except numpy.linalg.LinAlgError:
        return math.nan
    z = numpy.linalg.solve(chol, b)
    return float(z @ z)


def omnibus(stat: float, m: int, ddf_den: float | None) -> tuple[float, tuple[int, float | None], float]:
    """Scale ``stat`` and compute the p-value (F-test with ``ddf_den`` denominator degrees of freedom, else chi-square)."""
    if ddf_den is not None:
        stat /= m
        p = float(scipy.stats.f.sf(stat, m, ddf_den)) if ddf_den > 0 else math.nan
        return stat, (m, ddf_den), p
    return stat, (m, None), float(scipy.stats.chi2.sf(stat, m))


def rhs_vector(rhs: numpy.typing.ArrayLike | float | None, m: int, what: str) -> FloatArray:
    if rhs is None:
        return numpy.zeros(m)
    arr = numpy.broadcast_to(numpy.asarray(rhs, dtype=numpy.float64).reshape(-1), (m,)) if numpy.size(rhs) in (1, m) else None
    if arr is None:
        raise metafopy.errors.InputError(f"Length of 'rhs' ({numpy.size(rhs)}) does not match the number of {what} ({m}).")
    return numpy.array(arr)


def hypotheses(L: FloatArray, names: collections.abc.Sequence[str], rhs: FloatArray, digits: int = 4) -> tuple[str, ...]:
    """The text of each linear hypothesis, as in metafor (``'ni100 - imag = 0'``)."""
    out: list[str] = []
    for row in L:
        terms = [f'{numpy.round(c, digits):g}*{names[j]}' for j, c in enumerate(row) if numpy.round(c, digits) != 0]
        text = ' + '.join(terms).replace('1*', '').replace('+ -', '- ')
        out.append(text)
    if bool(numpy.all(rhs == 0)):
        return tuple(f'{h} = 0' for h in out)
    if len(set(rhs.tolist())) == 1:
        return tuple(f'{h} = {numpy.round(rhs[0], digits):g}' for h in out)
    return tuple(f'{h} = {metafopy.printing.fmt_num(float(r), digits)}' for h, r in zip(out, rhs, strict=True))


def lin_comb(
    coef: FloatArray,
    vcov: FloatArray,
    L: numpy.typing.ArrayLike,
    rhs: numpy.typing.ArrayLike | float | None,
    *,
    names: collections.abc.Sequence[str],
    int_incl: bool,
    n_coef: int,
    ddf: int | None,
    part: str,
    test: str,
    label: str,
) -> LSWaldResult:
    """Wald tests of linear combinations ``L coef = rhs`` (``X`` for the location, ``Z`` for the scale part)."""
    M = numpy.atleast_2d(numpy.asarray(L, dtype=numpy.float64))
    if int_incl and M.shape[1] == n_coef - 1:
        M = numpy.column_stack([numpy.ones(M.shape[0]), M])
    if M.shape[1] != n_coef:
        raise metafopy.errors.InputError(
            f"Length or number of columns of '{label}' ({M.shape[1]}) does not match the number of {'location' if part == 'location' else 'scale'} coefficients ({n_coef})."
        )
    m = M.shape[0]
    rhs_v = rhs_vector(rhs, m, 'linear combinations')
    est = M @ coef - rhs_v
    v = M @ vcov @ M.T
    se = numpy.sqrt(numpy.diag(v))
    zval = est / se
    t_test = test in ('knha', 'adhoc', 't')
    if t_test:
        d = float(ddf) if ddf is not None else math.nan
        pval = numpy.full(m, math.nan) if not d > 0 else 2 * scipy.stats.t.sf(numpy.abs(zval), d)
        ddf_arr: FloatArray | None = numpy.full(m, d)
    else:
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        ddf_arr = None
    stat = pval_omni = math.nan
    df: tuple[int, float | None] = (m, None)
    if numpy.linalg.matrix_rank(M) == m:
        stat0 = solve_quad(est, v)
        stat, df, pval_omni = omnibus(stat0, m, float(ddf) if (t_test and ddf is not None) else None)
    return LSWaldResult(
        part=part,
        stat=stat,
        df=df,
        pval_omnibus=pval_omni,
        test=test,
        m=m,
        hyp=hypotheses(M, names, rhs_v),
        estimate=est,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ddf=ddf_arr,
    )


def wald_subset(
    coef: FloatArray,
    vcov: FloatArray,
    idx: tuple[int, ...],
    rhs: numpy.typing.ArrayLike | float | None,
    *,
    ddf: int | None,
    part: str,
    test: str,
) -> LSWaldResult:
    sub = numpy.array(idx)
    m = len(idx)
    b = coef[sub] - rhs_vector(rhs, m, 'coefficients tested')
    stat0 = solve_quad(b, vcov[numpy.ix_(sub, sub)])
    t_test = test in ('knha', 'adhoc', 't')
    stat, df, p = omnibus(stat0, m, float(ddf) if (t_test and ddf is not None) else None)
    return LSWaldResult(part=part, stat=stat, df=df, pval_omnibus=p, test=test, m=m, idx=idx)


def lrt(a: metafopy.models.rma_ls_result.RMALSResult, b: metafopy.models.rma_ls_result.RMALSResult, refit: bool) -> metafopy.methods.anova.MVLRTResult:
    """Likelihood-ratio test of two nested location-scale models (the one with more parameters is the full model).

    Raises:
        metafopy.errors.InputError: the models cannot be compared.
    """
    full, red = (a, b) if a.parms >= b.parms else (b, a)
    if full.parms == red.parms:
        raise metafopy.errors.InputError('Models have the same number of parameters.')
    if full.k != red.k or not numpy.array_equal(full.yi, red.yi) or not numpy.array_equal(full.vi, red.vi):
        raise metafopy.errors.InputError('The observed outcomes and/or sampling variances are not equal in the full and reduced model.')
    if (full.method == 'ML' and red.method == 'REML') or (red.method == 'ML' and full.method == 'REML'):
        raise metafopy.errors.InputError(f'Mismatch between the use of {full.method} and {red.method} estimation in the full versus reduced model.')
    same_X = full.X.shape == red.X.shape and bool(numpy.allclose(full.X, red.X))
    if full.method == 'REML' and not same_X:
        if refit:
            fitted_f = metafopy.methods.ls_common.refit_ls(full, alpha=numpy.where(full.alpha_fix, full.alpha, numpy.nan), skiphes=False, method='ML')
            fitted_r = metafopy.methods.ls_common.refit_ls(red, alpha=numpy.where(red.alpha_fix, red.alpha, numpy.nan), skiphes=False, method='ML')
            if fitted_f is None:
                raise metafopy.errors.InputError('Refitting the full model with ML estimation failed.')
            if fitted_r is None:
                raise metafopy.errors.InputError('Refitting the reduced model with ML estimation failed.')
            full, red = fitted_f, fitted_r
        else:
            warnings.warn(
                "REML comparisons not meaningful for models with different fixed effects\n(use 'refit=True' to refit both models based on ML estimation).",
                stacklevel=3,
            )
    use_reml = full.method == 'REML'
    fs_f = full.fit_stats_reml if use_reml else full.fit_stats_ml
    fs_r = red.fit_stats_reml if use_reml else red.fit_stats_ml
    stat = max(0.0, fs_r.dev - fs_f.dev)
    return metafopy.methods.anova.MVLRTResult(
        df_full=full.parms,
        df_reduced=red.parms,
        LRT=stat,
        pval=float(scipy.stats.chi2.sf(stat, full.parms - red.parms)),
        fit_stats_full=fs_f.as_tuple(),
        fit_stats_reduced=fs_r.as_tuple(),
        QE_full=full.QE,
        QE_reduced=red.QE,
    )


@metafopy.methods.anova.anova.register
def anova_rma_ls(
    model: metafopy.models.rma_ls_result.RMALSResult,
    model2: metafopy.models.rma_ls_result.RMALSResult | None = None,
    *,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    att: metafopy.models.rma_uni.Btt | Sequences = None,
    Z: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    refit: bool = False,
) -> LSWaldResult | metafopy.methods.anova.MVLRTResult | list[LSWaldResult]:
    """Wald tests (``btt``, ``att``, ``X``, ``Z``) or, given a second model, a likelihood-ratio test.

    ``btt`` and ``att`` (0-based indices or a regular expression matching coefficient names) test subsets of the location and scale coefficients; ``X`` and
    ``Z`` test linear combinations of the location and scale coefficients against ``rhs``; ``att`` may also be a list of
    subsets, which returns a list of tests. With ``model2`` the model with more parameters is the full model; ``refit=True``
    refits REML fits with ML when their location models differ.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    x = model
    if model2 is not None:
        if not isinstance(model2, metafopy.models.rma_ls_result.RMALSResult):
            raise metafopy.errors.InputError("Class of 'object' must be the same as class of 'object2'.")
        return lrt(x, model2, refit)
    if X is not None and Z is not None:
        raise metafopy.errors.InputError("Can only specify either 'X' or 'Z', but not both.")
    if att is not None and btt is not None:
        raise metafopy.errors.InputError("Can only specify either 'btt' or 'att', but not both.")
    if Z is not None:
        return lin_comb(
            x.alpha, x.va, Z, rhs, names=x.alpha_names, int_incl=x.Z_int_incl, n_coef=x.q, ddf=x.ddf_alpha, part='scale', test=x.test, label='Z'
        )
    if X is not None:
        return lin_comb(x.beta, x.vb, X, rhs, names=x.coef_names, int_incl=x.int_incl, n_coef=x.p, ddf=x.ddf, part='location', test=x.test, label='X')
    if att is not None:
        if isinstance(att, collections.abc.Sequence) and not isinstance(att, str) and len(att) > 0 and all(isinstance(a, collections.abc.Sequence) and not isinstance(a, str) for a in att):
            if rhs is not None:
                raise metafopy.errors.InputError("Cannot use 'rhs' argument when specifying a list for 'att'.")
            return [typing.cast(LSWaldResult, anova_rma_ls(x, att=a)) for a in att]
        idx = metafopy.models.rma_uni.set_btt(att, x.q, x.Z_int_incl, x.alpha_names)
        return wald_subset(x.alpha, x.va, idx, rhs, ddf=x.ddf_alpha, part='scale', test=x.test)
    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names) if btt is not None else x.btt
    return wald_subset(x.beta, x.vb, idx, rhs, ddf=x.ddf, part='location', test=x.test)
