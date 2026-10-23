"""Wald tests and likelihood-ratio tests for location-scale models (metafor's ``anova.rma`` for ``rma.ls`` objects).

Wald tests of subsets of the location coefficients (``btt``), of the scale coefficients (``att``) or of linear combinations
of either (``X`` and ``Z``), and likelihood-ratio tests of nested models (Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1562).
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.stats

import metafopy.core.padjust
import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.common
import metafopy.methods.ls_common
import metafopy.methods.pairmat
import metafopy.mfopt
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
    digits: int = 4
    k: int | None = None
    p: int | None = None
    q: int | None = None
    class_: tuple[str, ...] | None = None

    @property
    def Xb(self) -> FloatArray:
        """The estimates of the linear combinations of the location coefficients (``X``), metafor's ``Xb``.

        Raises:
            AttributeError: the test is not of linear combinations of the location coefficients.
        """
        if self.part != 'location' or self.idx is not None or self.estimate is None:
            raise AttributeError('Xb')
        return self.estimate

    @property
    def Za(self) -> FloatArray:
        """The estimates of the linear combinations of the scale coefficients (``Z``), metafor's ``Za``.

        Raises:
            AttributeError: the test is not of linear combinations of the scale coefficients.
        """
        if self.part != 'scale' or self.idx is not None or self.estimate is None:
            raise AttributeError('Za')
        return self.estimate

    @property
    def type(self) -> str:
        """The kind of test: ``'Wald.btt'`` or ``'Wald.Xb'`` (location part), ``'Wald.att'`` or ``'Wald.Za'`` (scale part), metafor's ``type``."""
        suffix = ('btt' if self.part == 'location' else 'att') if self.idx is not None else ('Xb' if self.part == 'location' else 'Za')
        return f'Wald.{suffix}'

    @property
    def QMdf(self) -> tuple[int, float | None]:
        """The degrees of freedom of the test of the location coefficients, metafor's ``QMdf``."""
        return self.df

    @property
    def QSdf(self) -> tuple[int, float | None]:
        """The degrees of freedom of the test of the scale coefficients, metafor's ``QSdf``."""
        return self.df

    @property
    def btt(self) -> tuple[int, ...] | None:
        """The location coefficients tested (0-based), metafor's ``btt``."""
        return self.idx if self.part == 'location' else None

    @property
    def att(self) -> tuple[int, ...] | None:
        """The scale coefficients tested (0-based), metafor's ``att``."""
        return self.idx if self.part == 'scale' else None

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
        digits = metafopy.mfopt.option_digits(None, self.digits)
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
    """Wald tests of linear combinations ``L coef = rhs`` (``X`` for the location, ``Z`` for the scale part).

    Raises:
        metafopy.errors.InputError: ``L`` does not match the number of coefficients or ``rhs`` has the wrong length.
    """
    M = numpy.atleast_2d(numpy.asarray(L, dtype=numpy.float64))
    if int_incl and M.shape[1] == n_coef - 1:
        M = numpy.column_stack([numpy.ones(M.shape[0]), M])
    if M.shape[1] != n_coef:
        kind = 'location' if part == 'location' else 'scale'
        raise metafopy.errors.InputError(
            f"Length or number of columns of '{label}' ({M.shape[1]}) does not match the number of {kind} coefficients ({n_coef})."
        )
    m = M.shape[0]
    rhs_v = rhs_vector(rhs, m, 'linear combinations')
    est = M @ coef - rhs_v
    v = M @ vcov @ M.T
    se = numpy.sqrt(numpy.diag(v))
    zval = est / se
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
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
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    t_test = test in ('knha', 'adhoc', 't')
    stat, df, p = omnibus(stat0, m, float(ddf) if (t_test and ddf is not None) else None)
    return LSWaldResult(part=part, stat=stat, df=df, pval_omnibus=p, test=test, m=m, idx=idx)


def lrt(
    a: metafopy.models.rma_ls_result.RMALSResult, b: metafopy.models.rma_ls_result.RMALSResult, refit: bool, df_given: int | None = None
) -> metafopy.methods.anova.MVLRTResult:
    """Likelihood-ratio test of two nested location-scale models (the one with more parameters is the full model).

    Raises:
        metafopy.errors.InputError: the models cannot be compared.
    """
    full, red = (a, b) if df_given is not None or a.parms >= b.parms else (b, a)
    if df_given is None and full.parms == red.parms:
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
        df_full=full.parms + (0 if df_given is None else df_given),
        df_reduced=red.parms,
        LRT=stat,
        pval=float(scipy.stats.chi2.sf(stat, full.parms - red.parms if df_given is None else df_given)),
        fit_stats_full=fs_f.as_tuple(),
        fit_stats_reduced=fs_r.as_tuple(),
        QE_full=full.QE,
        QE_reduced=red.QE,
        method=full.method,
    )


def wald_between(a: metafopy.models.rma_ls_result.RMALSResult, b: metafopy.models.rma_ls_result.RMALSResult, df: int | None, digits: int) -> LSWaldResult:
    """``anova(full, reduced, test='Wald')``: the Wald test of the location coefficients that only the full model has.

    Raises:
        metafopy.errors.InputError: the models cannot be compared.
    """
    if df is None and a.parms == b.parms:
        raise metafopy.errors.InputError('Models have the same number of parameters.')
    full, red = (a, b) if df is not None or a.parms > b.parms else (b, a)
    if full.k != red.k or not numpy.array_equal(full.yi, red.yi) or not numpy.array_equal(full.vi, red.vi):
        raise metafopy.errors.InputError('The observed outcomes and/or sampling variances are not equal in the full and reduced model.')
    if full.method != red.method:
        raise metafopy.errors.InputError("Full and reduced model must use the same 'method' for the model fitting.")
    extra = [n for n in full.coef_names if n not in red.coef_names]
    if not extra:
        raise metafopy.errors.InputError('Full and reduced models appear to contain the same moderators.')
    idx = tuple(full.coef_names.index(n) for n in extra)
    return dataclasses.replace(wald_subset(full.beta, full.vb, idx, None, ddf=full.ddf, part='location', test=full.test), digits=digits)


def adjust_individual(res: LSWaldResult, method: str | None, digits: int) -> LSWaldResult:
    """Apply ``p.adjust`` to the individual tests of a linear-combination result and store ``digits``."""
    if method is not None and res.pval is not None:
        res = dataclasses.replace(res, pval=numpy.asarray(metafopy.core.padjust.p_adjust(res.pval, method), dtype=numpy.float64))
    return dataclasses.replace(res, digits=digits)


@metafopy.methods.anova.anova.register
@metafopy.methods.anova.with_model_info
def anova_rma_ls(
    model: metafopy.models.rma_ls_result.RMALSResult,
    model2: metafopy.models.rma_ls_result.RMALSResult | None = None,
    *,
    object2: metafopy.models.rma_ls_result.RMALSResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | metafopy.methods.pairmat.PairmatSpec | None = None,
    att: metafopy.models.rma_uni.Btt | Sequences = None,
    Z: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    adjust: bool | str | None = None,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    refit: bool = False,
    abbrev: bool = False,
    L: numpy.typing.ArrayLike | None = None,
    fixed: bool = False,
    test: str | None = None,
    df: int | None = None,
    verbose: bool = False,
) -> LSWaldResult | metafopy.methods.anova.MVLRTResult | list[LSWaldResult]:
    """Wald tests of the location and scale coefficients of a location-scale model or, given a second model, a likelihood-ratio test (metafor's
    ``anova.rma``).

    R source: <https://github.com/wviechtb/metafor/blob/master/R/anova.rma.r>

    Args:
        model: Location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
        model2: A second, nested location-scale model fitted to the same data for a likelihood-ratio test. The model with more parameters is the full
            model; when ``df`` is given, ``model`` is the full model. Default: none, which gives a Wald test of the coefficients of ``model``.
        object2: R's name for ``model2``; giving both with different values raises ``InputError``.
        btt: Location coefficients to test jointly: 0-based indices (sorted and without duplicates) or a string matched with ``re.search`` against the
            coefficient names (see ``fixed``; metafor uses ``grep``). A list of several tests is not supported here. Default: the coefficients tested
            by the omnibus test of the model, all but the intercept if there is one. Cannot be combined with ``att``.
        X: Linear combinations of the location coefficients to test, as for [anova_rma_uni][metafopy.methods.anova.anova_rma_uni]. Cannot be combined
            with ``Z``.
        att: Scale coefficients to test jointly: 0-based indices or a string matched against the coefficient names of the scale model, or a list of
            index lists (``[[0], [1]]``), which returns a list of tests (``rhs`` cannot be used with a list).
        Z: Linear combinations of the scale coefficients to test (a 1-d or 2-d array, with all ``q`` columns or without the intercept column, tested
            against ``rhs``).
        rhs: Right-hand side of the hypotheses of ``X`` and ``Z`` (and of the tests of ``btt`` and ``att``): a number or one value per combination or
            coefficient (a different length raises ``InputError``). Default: 0.
        adjust: Adjust the p-values of the individual tests (``X``, ``Z``) or of a list of tests (``att``) for multiple testing, as R's ``p.adjust``:
            ``True`` for ``'bonferroni'``, ``False`` for ``'none'``, or one of ``'holm'``, ``'hochberg'``, ``'hommel'``, ``'bonferroni'``, ``'BH'``,
            ``'BY'``, ``'fdr'`` and ``'none'`` (a unique prefix is accepted). Default: no adjustment.
        digits: Number of decimals of the printout. Default: the ``digits`` option, else that of the model.
        refit: With ``model2``, refit REML fits with ML when their location models differ (otherwise a warning is given).
        abbrev: Remove ``factor(`` and ``)`` from the text of the hypotheses of ``X`` and ``Z``.
        L: Old spelling of ``X``; if both are given, ``L`` is used.
        fixed: Match a string ``btt`` or ``att`` as a literal substring of the coefficient names instead of as a regular expression.
        test: With ``model2``: ``'LRT'`` (default) for the likelihood-ratio test, or ``'Wald'`` to test the location coefficients that only the full
            model has, with the Wald test of that model (the models must use the same ``method``). A unique prefix is accepted; anything else raises
            ``ArgumentError``.
        df: With ``model2``: the degrees of freedom of the likelihood-ratio test, instead of the difference in the number of parameters (it is also
            added to the number of parameters of the full model).
        verbose: Accepted for metafor's cluster-robust tests, which location-scale models do not have; it has no effect.

    Returns:
        A Wald test result, an ``LSWaldResult``: for location coefficients (``btt``, ``X``) the statistic is in ``QM``, ``QMdf`` and ``QMp``, for scale
        coefficients (``att``, ``Z``) in ``QS``, ``QSdf`` and ``QSp`` (the other statistic is NaN); ``stat``, ``df`` and ``pval_omnibus`` give whichever
        applies, and tests of linear combinations also hold ``hyp``, ``estimate``, ``se``, ``zval``, ``pval`` and ``ddf``. A list of them for a list
        ``att``. With ``model2``, a likelihood-ratio test result, an ``MVLRTResult`` (fields ``df_full``, ``df_reduced``, ``LRT``, ``pval``,
        ``fit_stats_full``, ``fit_stats_reduced``, ``QE_full``, ``QE_reduced`` and ``method``), or an ``LSWaldResult`` with ``test='Wald'``.

    Raises:
        metafopy.errors.InputError: an argument is invalid (for example both ``X`` and ``Z`` or both ``btt`` and ``att``, ``rhs`` of the wrong length or
            with a list ``att``, ``X`` or ``Z`` with the wrong number of columns, ``btt`` or ``att`` selecting nothing, an unknown ``adjust``, or
            models that differ in class, data or number of parameters or cannot be compared).
        metafopy.errors.ArgumentError: ``test`` is neither ``'LRT'`` nor ``'Wald'``.

    Metafor reference:
        [anova.rma](https://wviechtb.github.io/metafor/reference/anova.rma.html)
    """
    X = metafopy.methods.pairmat.resolve(X, model)
    if abbrev:
        return metafopy.methods.anova.abbreviate_hyp(anova_rma_ls(**{**locals(), 'abbrev': False}), True)
    x = model
    if L is not None:
        X = L
    d = metafopy.mfopt.option_digits(digits, x.digits)
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    model2 = metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2')
    if model2 is not None:
        if not isinstance(model2, metafopy.models.rma_ls_result.RMALSResult):
            raise metafopy.errors.InputError("Class of 'object' must be the same as class of 'object2'.")
        if metafopy.methods.anova.check_test(test) == 'Wald':
            return wald_between(x, model2, df, d)
        return lrt(x, model2, refit, df)
    if X is not None and Z is not None:
        raise metafopy.errors.InputError("Can only specify either 'X' or 'Z', but not both.")
    if att is not None and btt is not None:
        raise metafopy.errors.InputError("Can only specify either 'btt' or 'att', but not both.")
    if Z is not None:
        res = lin_comb(x.alpha, x.va, Z, rhs, names=x.alpha_names, int_incl=x.Z_int_incl, n_coef=x.q, ddf=x.ddf_alpha, part='scale', test=x.test, label='Z')
        return adjust_individual(res, method, d)
    if X is not None:
        res = lin_comb(x.beta, x.vb, X, rhs, names=x.coef_names, int_incl=x.int_incl, n_coef=x.p, ddf=x.ddf, part='location', test=x.test, label='X')
        return adjust_individual(res, method, d)
    if att is not None:
        if (
            isinstance(att, collections.abc.Sequence)
            and not isinstance(att, str)
            and len(att) > 0
            and all(isinstance(a, collections.abc.Sequence) and not isinstance(a, str) for a in att)
        ):
            if rhs is not None:
                raise metafopy.errors.InputError("Cannot use 'rhs' argument when specifying a list for 'att'.")
            tests = [typing.cast(LSWaldResult, anova_rma_ls(x, att=a, digits=d, fixed=fixed)) for a in att]
            if method is None:
                return tests
            adjusted = metafopy.core.padjust.p_adjust([t.pval_omnibus for t in tests], method)
            return [dataclasses.replace(t, pval_omnibus=float(p)) for t, p in zip(tests, adjusted, strict=True)]
        idx = metafopy.models.rma_uni.set_btt(att, x.q, x.Z_int_incl, x.alpha_names, fixed)
        return dataclasses.replace(wald_subset(x.alpha, x.va, idx, rhs, ddf=x.ddf_alpha, part='scale', test=x.test), digits=d)
    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names, fixed) if btt is not None else x.btt
    return dataclasses.replace(wald_subset(x.beta, x.vb, idx, rhs, ddf=x.ddf, part='location', test=x.test), digits=d)
