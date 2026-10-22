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
import metafopy.formatters
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.pairmat
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.models.robust_info
import metafopy.models.tau2
import metafopy.printing

__all__ = ['LRTResult', 'ListAnovaResult', 'MVLRTResult', 'WaldResult', 'anova']

FloatArray = numpy.typing.NDArray[numpy.float64]


def wald_test_line(name: str, QM: float, QMdf: tuple[int, float | None], QMp: float, digits: int) -> str:
    """``fmtt()`` of an omnibus Wald test: ``QM(df = 2) = 12.4826, p-val = 0.0019``, or the F form when the denominator degrees of freedom are known."""
    m, ddf = QMdf
    p = metafopy.models.rma_uni.pfmt(QMp, digits)
    if ddf is None:
        return f'{name}(df = {m}) = {metafopy.printing.fmt_num(QM, digits)}, p-val {p}'
    return f'F(df1 = {m}, df2 = {metafopy.printing.fmt_df_scalar(ddf)}) = {metafopy.printing.fmt_num(QM, digits)}, p-val {p}'


@dataclasses.dataclass(frozen=True, slots=True)
class WaldResult(metafopy.printing.HtmlRepr):
    """Omnibus Wald test (``QM``) and, for linear combinations, individual tests, as returned by [anova][metafopy.anova] (metafor's ``anova.rma`` object).

    Attributes:
        QM: Test statistic of the omnibus test: the chi-square statistic, or the F statistic (the chi-square statistic divided by ``m``) when the
            degrees of freedom of the denominator are known. NaN when the test cannot be computed (``X`` not of full row rank, singular covariance
            matrix).
        QMdf: Degrees of freedom of the omnibus test: ``(m, None)`` for a chi-square test, ``(m, ddf)`` for an F test.
        QMp: p-value of the omnibus test (adjusted if ``adjust`` was given for a test of moderators).
        btt: Tested coefficients (0-based, sorted) of a test of moderators; None for a test of linear combinations.
        Xb: Estimates of the linear combinations (less ``rhs``); None for a test of moderators.
        se: Standard errors of the combinations.
        zval: Test statistics of the combinations (t values when ``ddf`` is not None).
        pval: p-values of the combinations (adjusted if requested).
        ddf: Degrees of freedom of the t tests of the combinations (one per combination), None for z tests.
        hyp: Hypotheses as R prints them (for example ``'intrcpt + 30*ablat = 0'``).
        digits: Number of decimals of the printout.
        k: Number of studies of the model (None if not filled in).
        p: Number of coefficients of the model.
        test: The test of the model's coefficients, ``'z'``, ``'t'``, ``'knha'`` or ``'adhoc'``.
        class_: The class vector of the model a test of moderators was made for (metafor's ``class``), for example ``('rma.uni', 'rma')``.
    """

    QM: float
    QMdf: tuple[int, float | None]
    QMp: float
    btt: tuple[int, ...] | None = None
    Xb: FloatArray | None = None
    se: FloatArray | None = None
    zval: FloatArray | None = None
    pval: FloatArray | None = None
    ddf: FloatArray | None = None
    hyp: tuple[str, ...] | None = None
    digits: int = 4
    k: int | None = None
    p: int | None = None
    test: str | None = None
    class_: tuple[str, ...] | None = None
    """The class vector of the model a test of moderators was made for (the ``class`` element of metafor's result), e.g. ``('rma.uni', 'rma')``."""

    @property
    def m(self) -> int:
        """The number of coefficients or linear combinations tested, metafor's ``m``."""
        return self.QMdf[0]

    @property
    def type(self) -> str:
        """The kind of test, ``'Wald.btt'`` (coefficients) or ``'Wald.Xb'`` (linear combinations), metafor's ``type``."""
        return 'Wald.btt' if self.Xb is None else 'Wald.Xb'

    def to_frame(self) -> pandas.DataFrame:
        """The test as a data frame (metafor's ``as.data.frame.anova.rma``).

        A test of moderators gives one row with the columns ``coefs``, ``QM``, ``df`` and ``pval`` (``Fval``, ``df1``, ``df2`` and ``pval`` for
        F tests); tests of linear combinations give one row per hypothesis (row names ``'1:'``, ``'2:'``, ...) with ``hyp``, ``estimate``,
        ``se``, ``zval`` (``tval`` and ``df`` for t tests) and ``pval``.
        """
        m, df2 = self.QMdf
        if self.Xb is None or self.se is None or self.zval is None or self.pval is None:
            coefs = metafopy.methods.vif.format_btt(i + 1 for i in (self.btt or ()))
            if df2 is None:
                return pandas.DataFrame({'coefs': [coefs], 'QM': [self.QM], 'df': [round(m, 2)], 'pval': [self.QMp]})
            return pandas.DataFrame({'coefs': [coefs], 'Fval': [self.QM], 'df1': [round(m, 2)], 'df2': [round(float(df2), 2)], 'pval': [self.QMp]})
        n = self.Xb.shape[0]
        cols: dict[str, typing.Any] = {
            'hyp': list(self.hyp) if self.hyp is not None else [''] * n,
            'estimate': self.Xb,
            'se': self.se,
            'zval' if self.ddf is None else 'tval': self.zval,
        }
        if self.ddf is not None:
            cols['df'] = numpy.round(self.ddf, 2)
        cols['pval'] = self.pval
        return pandas.DataFrame(cols, index=[f'{j + 1}:' for j in range(n)])

    def summary(self, digits: int | collections.abc.Mapping[str, int] | None = None, *, signif_stars: bool | None = None) -> str:
        """The printout of metafor's ``print.anova.rma``; ``digits`` defaults to the ``digits`` option, else to that of the test."""
        d = metafopy.mfopt.option_digits(digits, self.digits)
        stars, _ = metafopy.mfopt.option_signif(signif_stars, None)
        m = self.QMdf[0]
        if self.Xb is None or self.se is None or self.zval is None or self.pval is None:
            head = (
                f'Test of Moderators (coefficient{"s" if m > 1 else ""} {metafopy.methods.vif.format_btt(i + 1 for i in self.btt)}):'
                if self.btt is not None
                else 'Omnibus Test of Hypotheses:'
            )
            return f'{head}\n{wald_test_line("QM", self.QM, self.QMdf, self.QMp, d)}'
        n = self.Xb.shape[0]
        names = [f'{j + 1}:' for j in range(n)]
        out: list[str] = ['Hypothesis:' if n == 1 else 'Hypotheses:']
        if self.hyp is not None:
            width = max(len(h) for h in self.hyp)
            out += [f'{name} {h.rjust(width)}' for name, h in zip(names, self.hyp, strict=True)]
        out += ['', 'Results:']
        header = ['estimate', 'se', 'zval' if self.ddf is None else 'tval']
        cells = [[metafopy.printing.fmt_num(float(v), d) for v in col] for col in (self.Xb, self.se, self.zval)]
        if self.ddf is not None:
            header.append('df')
            cells.append(metafopy.printing.fmt_df(self.ddf))
        header += ['pval', '']
        cells.append([metafopy.printing.fmt_pval(float(v), d) for v in self.pval])
        cells.append([metafopy.printing.signif_stars(float(v)) for v in self.pval])
        rows = [[col[j] for col in cells] for j in range(n)]
        header, rows = metafopy.printing.drop_stars(header, rows, stars)
        out.append(metafopy.printing.fmt_table(header, rows, names, gap=1))
        if not math.isnan(self.QM):
            out += ['', 'Test of Hypothesis:' if n == 1 else 'Omnibus Test of Hypotheses:', wald_test_line('QM', self.QM, self.QMdf, self.QMp, d)]
        return '\n'.join(out)

    def __str__(self) -> str:
        return self.summary()


class ListAnovaResult(list[WaldResult]):
    """The tests of a list of ``btt`` specifications (metafor's ``list.anova.rma``): a list of :class:`WaldResult`, one per specification.

    ``names`` holds the specifications as R names the elements (``'2'``, ``'1:2'`` or a coefficient name).
    """

    def __init__(self, results: collections.abc.Iterable[WaldResult], names: collections.abc.Sequence[str]) -> None:
        super().__init__(results)
        self.names = tuple(names)

    def to_frame(self) -> pandas.DataFrame:
        """The tests as one data frame (metafor's ``as.data.frame.list.anova.rma``), one row per specification.

        The columns are ``spec`` (left out when all specifications are numeric), ``coefs``, ``QM``, ``df`` and ``pval`` (``Fval``, ``df1``,
        ``df2`` and ``pval`` for F tests).
        """
        frames = [r.to_frame() for r in self]
        tab = pandas.concat(frames, ignore_index=True)
        if not all(name[:1] in '123456789' for name in self.names):
            tab.insert(0, 'spec', list(self.names))
        return tab

    def summary(self, digits: int | collections.abc.Mapping[str, int] | None = None, *, signif_stars: bool | None = None) -> str:
        """The printout of metafor's ``print.list.anova.rma``."""
        d = metafopy.mfopt.option_digits(digits, self[0].digits if self else 4)
        stars, _ = metafopy.mfopt.option_signif(signif_stars, None)
        tab = self.to_frame()
        header = [str(c) for c in tab.columns]
        cells: list[list[str]] = []
        for col in tab.columns:
            values = tab[col]
            if col in ('spec', 'coefs'):
                cells.append([str(v) for v in values])
            elif col in ('df', 'df1', 'df2'):
                cells.append(metafopy.printing.fmt_df(values.to_numpy(dtype=numpy.float64)))
            elif col == 'pval':
                cells.append([metafopy.printing.fmt_pval(float(v), d) for v in values])
            else:
                cells.append([metafopy.printing.fmt_num(float(v), d) for v in values])
        header.append('')
        cells.append([metafopy.printing.signif_stars(float(v)) for v in tab['pval']])
        rows = [[col[j] for col in cells] for j in range(len(tab))]
        header, rows = metafopy.printing.drop_stars(header, rows, stars)
        return metafopy.printing.fmt_table(header, rows, [str(j + 1) for j in range(len(tab))], gap=1)

    def __str__(self) -> str:
        return self.summary()


@dataclasses.dataclass(frozen=True, slots=True)
class LRTResult(metafopy.printing.HtmlRepr):
    """Likelihood-ratio test of a full against a reduced model, as returned by [anova][metafopy.anova] for two models.

    Attributes:
        df_full: Number of parameters of the full model (plus ``df`` if that was given).
        df_reduced: Number of parameters of the reduced model.
        LRT: Likelihood-ratio test statistic.
        pval: p-value of the test.
        fit_stats_full: Fit statistics of the full model (log-likelihood, deviance, AIC, BIC, AICc), from the REML likelihood if the models were fitted
            with REML and the ML likelihood otherwise.
        fit_stats_reduced: Fit statistics of the reduced model, in the same order.
        QE_full: Test statistic for residual heterogeneity of the full model.
        QE_reduced: Test statistic for residual heterogeneity of the reduced model.
        tau2_full: Estimate of ``tau^2`` of the full model.
        tau2_reduced: Estimate of ``tau^2`` of the reduced model.
        R2: Percentage reduction in ``tau^2`` (or in the residual variance for fixed-effects models) from the reduced to the full model; 0 if ``tau^2``
            of the reduced model is 0.
        digits: Number of decimals of the printout.
        method: Estimation method of the full model.
        class_f: Class vector of the full model (metafor's ``class.f``).
    """

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
    digits: int = 4
    method: str | None = None
    class_f: tuple[str, ...] | None = None
    """The class vector of the full model (metafor's ``class.f``)."""

    @property
    def type(self) -> str:
        """The kind of test, ``'LRT'`` (metafor's ``type``)."""
        return 'LRT'

    @property
    def fit_stats_f(self) -> tuple[float, float, float, float, float]:
        """The fit statistics of the full model, metafor's ``fit.stats.f``."""
        return self.fit_stats_full

    @property
    def fit_stats_r(self) -> tuple[float, float, float, float, float]:
        """The fit statistics of the reduced model, metafor's ``fit.stats.r``."""
        return self.fit_stats_reduced

    @property
    def parms_f(self) -> int:
        """The number of parameters of the full model, metafor's ``parms.f``."""
        return self.df_full

    @property
    def parms_r(self) -> int:
        """The number of parameters of the reduced model, metafor's ``parms.r``."""
        return self.df_reduced

    @property
    def QE_f(self) -> float:
        """The test statistic for residual heterogeneity of the full model, metafor's ``QE.f``."""
        return self.QE_full

    @property
    def QE_r(self) -> float:
        """The test statistic for residual heterogeneity of the reduced model, metafor's ``QE.r``."""
        return self.QE_reduced

    @property
    def tau2_f(self) -> float:
        """The amount of heterogeneity of the full model, metafor's ``tau2.f``."""
        return self.tau2_full

    @property
    def tau2_r(self) -> float:
        """The amount of heterogeneity of the reduced model, metafor's ``tau2.r``."""
        return self.tau2_reduced

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
        digits = metafopy.mfopt.option_digits(None, self.digits)

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
    """Wald tests of coefficients or likelihood-ratio tests of nested models for fitted models (metafor's ``anova``).

    This is a single-dispatch function: the implementation is chosen by the class of ``model``. See [anova_rma_uni][metafopy.methods.anova.anova_rma_uni],
    [anova_rma_mv][metafopy.methods.anova.anova_rma_mv] and [anova_rma_ls][metafopy.methods.ls_anova.anova_rma_ls].

    Args:
        model: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation (a second model first).
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        A [WaldResult][metafopy.WaldResult] (Wald test), an [LRTResult][metafopy.LRTResult] (likelihood-ratio test of two ``rma_uni`` models), a
        ``MVLRTResult`` (likelihood-ratio test of two ``rma_mv`` or location-scale models), an ``LSWaldResult`` (Wald test for a location-scale
        model), or a list of Wald results for a list of tests.

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``model``. An argument the implementation does not have is a
            ``TypeError`` from Python, as for any function.
    """
    raise metafopy.errors.ArgumentError(f'anova() not implemented for {type(model).__name__}')


def with_model_info[F: collections.abc.Callable[..., typing.Any]](function: F) -> F:
    """Record ``k``, ``p`` and ``test`` of the model in the Wald tests ``function`` returns (the fields metafor's ``anova.rma`` object has)."""

    @functools.wraps(function)
    def wrapper(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        result = function(model, *args, **kwargs)
        return add_model_info(result, model)

    return typing.cast(F, wrapper)


def add_model_info(result: typing.Any, model: typing.Any) -> typing.Any:
    """``result`` with the ``k``, ``p``, ``q`` and ``test`` of ``model`` filled in where it has such fields (lists of tests element by element)."""
    if isinstance(result, ListAnovaResult):
        return ListAnovaResult([add_model_info(item, model) for item in result], result.names)
    if isinstance(result, list):
        return [add_model_info(item, model) for item in result]
    if not dataclasses.is_dataclass(result) or isinstance(result, type):
        return result
    names = {f.name for f in dataclasses.fields(result)}
    changes: dict[str, typing.Any] = {name: int(getattr(model, name)) for name in ('k', 'p', 'q') if name in names and hasattr(model, name)}
    if 'test' in names and typing.cast(typing.Any, result).test is None:
        changes['test'] = str(model.test)
    if 'class_' in names and typing.cast(typing.Any, result).hyp is None and typing.cast(typing.Any, result).class_ is None:
        changes['class_'] = metafopy.methods.fitview.r_class(model)
    return dataclasses.replace(result, **changes) if changes else result


@anova.register
@with_model_info
def anova_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    model2: metafopy.models.rma_uni.RMAResult | None = None,
    *,
    object2: metafopy.models.rma_uni.RMAResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | metafopy.methods.pairmat.PairmatSpec | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    adjust: bool | str | None = None,
    refit: bool = False,
    L: numpy.typing.ArrayLike | None = None,
    fixed: bool = False,
    test: str | None = None,
    df: int | None = None,
    verbose: bool = False,
    abbrev: bool = False,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    att: typing.Any = None,
    Z: typing.Any = None,
) -> WaldResult | LRTResult | ListAnovaResult:
    """Test coefficients of one [rma_uni][metafopy.rma_uni] model, or compare two nested models (metafor's ``anova.rma``).

    ``anova(res, btt=...)`` tests a subset of coefficients, ``anova(res, X=L)`` tests linear combinations ``L beta = rhs`` and ``anova(full, reduced)``
    gives a likelihood-ratio test.

    For a model made robust with ``club_sandwich=True`` the omnibus test is the cluster-robust Wald test with the small-sample approximation chosen by
    ``wald_test`` of [robust][metafopy.robust] (``HTZ`` by default: an F test with df1 and df2 from ``clubSandwich::Wald_test``), and the tests of linear
    combinations use the degrees of freedom of ``coef_test``. As in R, ``rhs`` cannot be used with ``X`` for such a model (a warning is given and zero is
    used).

    R source: <https://github.com/wviechtb/metafor/blob/master/R/anova.rma.r>; reference page:

    Args:
        model: Model fitted by [rma_uni][metafopy.rma_uni].
        model2: A second, nested model fitted to the same data, of the same class, for a likelihood-ratio test. The model with more parameters is the
            full model; when ``df`` is given, ``model`` is the full model. Default: none, which gives a Wald test of the coefficients of ``model``.
        object2: R's name for ``model2``; giving both with different values raises ``InputError``.
        btt: Coefficients to test jointly: 0-based indices (sorted and without duplicates; an index outside the model raises ``InputError``), a
            string matched with ``re.search`` against the coefficient names (see ``fixed``; metafor uses ``grep``), or a list of several strings or
            index lists for one test each (a list result, named by the specifications). Default: the coefficients tested by the omnibus test of the
            model, all but the intercept if there is one.
        X: Linear combinations of the coefficients to test: a 1-d array (one combination) or a 2-d array with one row per combination, with all ``p``
            columns or without the intercept column (a column of ones is then added). Each row is tested against ``rhs``; the result holds one test
            per row and, if ``X`` has full row rank, the omnibus test of all of them (``QM`` is otherwise NaN). Also accepted: a
            [pairmat][metafopy.pairmat] specification. Default: none.
        rhs: Right-hand side of the hypotheses: a number or one value per combination (or per tested coefficient with ``btt``; a different length
            raises ``InputError``). Default: 0.
        adjust: Adjust the p-values for multiple testing, as R's ``p.adjust``: ``True`` for ``'bonferroni'``, ``False`` for ``'none'``, or one of
            ``'holm'``, ``'hochberg'``, ``'hommel'``, ``'bonferroni'``, ``'BH'``, ``'BY'``, ``'fdr'`` and ``'none'`` (a unique prefix is accepted).
            It applies to the p-values of the individual tests of the linear combinations, and to the omnibus p-value of a test of moderators (across
            the elements for a list ``btt``). Default: no adjustment.
        refit: For a likelihood-ratio test of REML models whose fixed effects differ, refit them with ML first (otherwise a warning is given).
        L: Old spelling of ``X``; if both are given, ``L`` is used.
        fixed: Match a string ``btt`` as a literal substring of the coefficient names instead of as a regular expression.
        test: With ``model2``: ``'LRT'`` (default) for the likelihood-ratio test, or ``'Wald'`` to test the coefficients that only the full model has,
            with the Wald test of that model (the models must use the same ``method``). A unique prefix is accepted; anything else raises
            ``ArgumentError``.
        df: With ``model2``: the degrees of freedom of the likelihood-ratio test, instead of the difference in the number of parameters (it is also
            added to the number of parameters of the full model).
        verbose: Add the reason to the error of a cluster-robust test that fails.
        abbrev: Remove ``factor(`` and ``)`` from the text of the hypotheses of a test of linear combinations (not described on metafor's reference
            page, but accepted by its source).
        digits: Number of decimals of the printout of the result (also used for the hypotheses of a test of linear combinations), or a mapping as for
            [predict][metafopy.predict]. Default: the ``digits`` option, else that of the model.
        att: For location-scale models only; giving it here raises ``InputError``.
        Z: For location-scale models only; giving it here raises ``InputError``.

    Returns:
        For a Wald test, a [WaldResult][metafopy.WaldResult] with the omnibus statistic ``QM``, its degrees of freedom ``QMdf`` and p-value ``QMp`` and,
        for linear combinations, the estimates ``Xb``, ``se``, ``zval``, ``pval`` and the hypotheses ``hyp``. For two models, an
        [LRTResult][metafopy.LRTResult] (or, with ``test='Wald'``, a ``WaldResult``). For a list ``btt``, a ``ListAnovaResult`` (a list of Wald results).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid (for example ``att`` or ``Z`` is given, ``X`` has the wrong number of columns,
            ``btt`` selects nothing, ``adjust`` is unknown, the models differ in class or data or in the number of parameters, or a cluster-robust test
            cannot be computed).
        metafopy.errors.ArgumentError: ``test`` is neither ``'LRT'`` nor ``'Wald'``.
        metafopy.errors.InternalError: an internal invariant does not hold.

    Metafor reference:
        [anova.rma](https://wviechtb.github.io/metafor/reference/anova.rma.html)
    """
    X = metafopy.methods.pairmat.resolve(X, model)
    check_scale_arguments(att, Z)
    if abbrev:
        return abbreviate_hyp(anova_rma_uni(**{**locals(), 'abbrev': False}), True)
    x = model
    d = metafopy.mfopt.option_digits(digits, x.digits)
    tdist = x.ddf is not None
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    model2 = metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2')
    X = merge_x(X, L)
    rhs_given = rhs is not None
    rhs = 0.0 if rhs is None else rhs
    if model2 is not None:
        check_same_class(x, model2)
        if check_test(test) == 'Wald':
            return dataclasses.replace(wald_between(x, model2, df), digits=d)
        return dataclasses.replace(lrt(x, model2, refit, df), digits=d)
    club = club_info(x.robust)
    if is_btt_list(btt):
        specs = typing.cast(list[typing.Any], btt)
        return make_list([typing.cast(WaldResult, anova_rma_uni(x, btt=b, fixed=fixed, digits=d)) for b in specs], specs, method)

    if X is not None and club is not None:
        return club_wald_x(x.beta, x.vb, club, X, x.p, x.int_incl, rhs_given, method, verbose, x.coef_names, d)

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
            hyp=hypotheses(L, x.coef_names, numpy.broadcast_to(numpy.asarray(rhs, dtype=numpy.float64), (m,)), d),
            digits=d,
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
        return dataclasses.replace(out, digits=d)
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
    return WaldResult(QM=QM, QMdf=(m, x.ddf), QMp=QMp, btt=idx, digits=d)


def abbreviate_hyp[T](result: T, abbrev: bool) -> T:
    """``anova(..., abbrev=TRUE)``: ``factor(`` and ``)`` are removed from the text of the hypotheses of a test of linear combinations."""
    if not abbrev:
        return result
    if isinstance(result, list):
        return typing.cast(T, [abbreviate_hyp(item, abbrev) for item in result])
    hyp = getattr(result, 'hyp', None)
    if hyp is None:
        return result
    return typing.cast(T, dataclasses.replace(typing.cast(typing.Any, result), hyp=tuple(h.replace('factor(', '').replace(')', '') for h in hyp)))


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
    names: collections.abc.Sequence[str],
    digits: int,
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
    return WaldResult(
        QM=QM,
        QMdf=QMdf,
        QMp=QMp,
        Xb=Xb,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ddf=ddf,
        hyp=hypotheses(C, names, numpy.zeros(m), digits),
        digits=digits,
    )


def check_same_class(a: object, b: object) -> None:
    """R's ``identical(class(object), class(object2))``: a robust, selection model or plain fit is not compared with a fit of another class.

    Raises:
        metafopy.errors.InputError: the classes differ.
    """
    if metafopy.methods.fitview.r_class(a) != metafopy.methods.fitview.r_class(b):
        raise metafopy.errors.InputError("Class of 'object' must be the same as class of 'object2'.")


def wald_between(
    a: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    b: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    df: int | None,
) -> WaldResult:
    """``anova(full, reduced, test='Wald')``: the Wald test of the coefficients that only the full model has.

    Raises:
        metafopy.errors.InputError: the models cannot be compared.
    """
    check_same_class(a, b)
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


def make_list(results: list[WaldResult], specs: collections.abc.Sequence[typing.Any], method: str | None) -> ListAnovaResult:
    """The results of a list ``btt`` as metafor's ``list.anova.rma``, named by the specifications (``.format.btt``, 1-based like R)."""
    names = [metafopy.methods.vif.format_spec(spec) for spec in specs]
    return ListAnovaResult(adjust_list(results, method), names)


def check_scale_arguments(att: typing.Any, Z: typing.Any) -> None:
    """``att`` and ``Z`` are for location-scale models; metafor stops when they are given for any other model.

    Raises:
        metafopy.errors.InputError: ``att`` or ``Z`` is given.
    """
    if att is not None:
        raise metafopy.errors.InputError("Can only specify 'att' for location-scale models.")
    if Z is not None:
        raise metafopy.errors.InputError("Can only specify 'Z' for location-scale models.")


def hypotheses(L: FloatArray, names: collections.abc.Sequence[str], rhs: FloatArray, digits: int) -> tuple[str, ...]:
    """The hypotheses ``anova.rma`` prints for ``X``: ``'intrcpt + 30*ablat = 0'`` (coefficients rounded to ``digits`` decimals; as in R the
    literal ``1*`` is removed everywhere, so ``-1*x`` reads ``-x``)."""
    terms: list[str] = []
    for row in L:
        coefs = numpy.round(row, digits)
        text = ' + '.join(f'{metafopy.formatters.r_as_character(float(v))}*{name}' for v, name in zip(coefs, names, strict=True) if v != 0)
        terms.append(text.replace('1*', '').replace('+ -', '- '))
    if bool(numpy.all(rhs == 0)):
        return tuple(f'{t} = 0' for t in terms)
    if len(set(rhs.tolist())) == 1:
        return tuple(f'{t} = {metafopy.formatters.r_as_character(round(float(rhs[0]), digits))}' for t in terms)
    return tuple(f'{t} = {metafopy.printing.fmt_num(float(r), digits)}' for t, r in zip(terms, rhs, strict=True))


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


def check_lrt_models(
    full: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    red: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
) -> None:
    """The checks ``anova.rma`` makes before a likelihood-ratio test of ``full`` against ``red`` (stops or warns as R does).

    Raises:
        metafopy.errors.InputError: the full model is a fixed-effects and the reduced a random/mixed-effects model, or one model was fitted with
            ML and the other with REML.
    """
    if full.method in metafopy.models.tau2.FIXED_EFFECTS and red.method not in metafopy.models.tau2.FIXED_EFFECTS:
        raise metafopy.errors.InputError('Full model uses a fixed- and reduced model uses a random/mixed-effects model.')
    if {full.method, red.method} == {'ML', 'REML'}:
        raise metafopy.errors.InputError(f'Mismatch between the use of {full.method} and {red.method} estimation in the full versus reduced model.')
    lrt_methods = (*metafopy.models.tau2.FIXED_EFFECTS, 'ML', 'REML')
    if full.method not in lrt_methods or red.method not in lrt_methods:
        warnings.warn('LRTs should be based on ML/REML estimation.', stacklevel=4)


def warn_not_nested(
    full: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
    red: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
) -> None:
    """The warning of ``anova.rma`` when the fixed effects differ and the model matrices are not nested."""
    same_X = full.X.shape == red.X.shape and numpy.allclose(full.X, red.X)
    if not same_X and not metafopy.core.linalg.is_nested(full.X, red.X):
        warnings.warn('The models being compared appear not to be nested.', stacklevel=4)


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
    except (metafopy.errors.MetafopyError, numpy.linalg.LinAlgError, ValueError) as err:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.') from err
    return out


def lrt(a: metafopy.models.rma_uni.RMAResult, b: metafopy.models.rma_uni.RMAResult, refit: bool = False, df_given: int | None = None) -> LRTResult:
    if df_given is None and a.parms == b.parms:
        raise metafopy.errors.InputError('models do not differ in the number of parameters')
    full, red = (a, b) if df_given is not None or a.parms > b.parms else (b, a)
    if full.k != red.k or not numpy.allclose(full.yi, red.yi) or not numpy.allclose(full.vi, red.vi):
        raise metafopy.errors.InputError('The observed outcomes and/or sampling variances are not equal in the full and reduced model.')
    check_lrt_models(full, red)
    same_X = full.X.shape == red.X.shape and numpy.allclose(full.X, red.X)
    if full.method == 'REML' and not same_X:
        if refit:
            full, red = refit_ml(full), refit_ml(red)
        else:
            warnings.warn(
                "REML comparisons not meaningful for models with different fixed effects (use 'refit=True' to refit both models based on ML estimation).",
                stacklevel=3,
            )
    warn_not_nested(full, red)
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
        method=full.method,
        class_f=metafopy.methods.fitview.r_class(full),
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
class MVLRTResult(metafopy.printing.HtmlRepr):
    """Likelihood-ratio test of a full against a reduced ``rma.mv`` model."""

    df_full: int
    df_reduced: int
    LRT: float
    pval: float
    fit_stats_full: tuple[float, float, float, float, float]
    fit_stats_reduced: tuple[float, float, float, float, float]
    QE_full: float
    QE_reduced: float
    digits: int = 4
    method: str | None = None
    """The estimation method of the full model (metafor's ``method``)."""

    @property
    def type(self) -> str:
        """The kind of test, ``'LRT'`` (metafor's ``type``)."""
        return 'LRT'

    @property
    def fit_stats_f(self) -> tuple[float, float, float, float, float]:
        """The fit statistics of the full model, metafor's ``fit.stats.f``."""
        return self.fit_stats_full

    @property
    def fit_stats_r(self) -> tuple[float, float, float, float, float]:
        """The fit statistics of the reduced model, metafor's ``fit.stats.r``."""
        return self.fit_stats_reduced

    @property
    def parms_f(self) -> int:
        """The number of parameters of the full model, metafor's ``parms.f``."""
        return self.df_full

    @property
    def parms_r(self) -> int:
        """The number of parameters of the reduced model, metafor's ``parms.r``."""
        return self.df_reduced

    @property
    def QE_f(self) -> float:
        """The test statistic for residual heterogeneity of the full model, metafor's ``QE.f``."""
        return self.QE_full

    @property
    def QE_r(self) -> float:
        """The test statistic for residual heterogeneity of the reduced model, metafor's ``QE.r``."""
        return self.QE_reduced

    @property
    def tau2_f(self) -> float:
        """``NaN``: metafor stores ``NA`` for models with several variance components (``tau2.f``)."""
        return math.nan

    @property
    def tau2_r(self) -> float:
        """``NaN``: metafor stores ``NA`` for models with several variance components (``tau2.r``)."""
        return math.nan

    @property
    def R2(self) -> float:
        """``NaN``: metafor stores ``NA`` for these models."""
        return math.nan

    def __str__(self) -> str:
        header = ['df', 'AIC', 'BIC', 'AICc', 'logLik', 'LRT', 'pval', 'QE']
        fs_f, fs_r = self.fit_stats_full, self.fit_stats_reduced
        d = metafopy.mfopt.option_digits(None, self.digits)

        def fmt(x: float) -> str:
            return metafopy.printing.fmt_num(x, d)

        rows = {
            'Full': [str(self.df_full), fmt(fs_f[2]), fmt(fs_f[3]), fmt(fs_f[4]), fmt(fs_f[0]), '', '', fmt(self.QE_full)],
            'Reduced': [
                str(self.df_reduced),
                fmt(fs_r[2]),
                fmt(fs_r[3]),
                fmt(fs_r[4]),
                fmt(fs_r[0]),
                fmt(self.LRT),
                metafopy.printing.fmt_pval(self.pval, d),
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
@with_model_info
def anova_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    model2: metafopy.models.mv_result.RMAMVResult | None = None,
    *,
    object2: metafopy.models.mv_result.RMAMVResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | metafopy.methods.pairmat.PairmatSpec | None = None,
    rhs: numpy.typing.ArrayLike | float | None = None,
    adjust: bool | str | None = None,
    refit: bool = False,
    L: numpy.typing.ArrayLike | None = None,
    fixed: bool = False,
    test: str | None = None,
    df: int | None = None,
    verbose: bool = False,
    abbrev: bool = False,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    att: typing.Any = None,
    Z: typing.Any = None,
) -> WaldResult | MVLRTResult | ListAnovaResult:
    """Test coefficients of one [rma_mv][metafopy.rma_mv] model, or compare two nested models by a likelihood-ratio test (metafor's ``anova.rma``).

    The arguments and the handling of ``club_sandwich=True`` fits are those of [anova_rma_uni][metafopy.methods.anova.anova_rma_uni]; the ML refit
    repeats the recorded call of the fit with ``method='ML'`` (and raises ``InputError`` if no call was recorded). Only the observed outcomes (not the
    sampling variances) are compared for equality in a likelihood-ratio test.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/anova.rma.r>; reference page:

    Args:
        model: Model fitted by [rma_mv][metafopy.rma_mv].
        model2: A second, nested model fitted to the same data, of the same class, for a likelihood-ratio test. The model with more parameters is the
            full model; when ``df`` is given, ``model`` is the full model. Default: none, which gives a Wald test of the coefficients of ``model``.
        object2: R's name for ``model2``; giving both with different values raises ``InputError``.
        btt: Coefficients to test jointly: 0-based indices (sorted and without duplicates; an index outside the model raises ``InputError``), a
            string matched with ``re.search`` against the coefficient names (see ``fixed``; metafor uses ``grep``), or a list of several strings or
            index lists for one test each (a list result, named by the specifications). Default: the coefficients tested by the omnibus test of the
            model, all but the intercept if there is one.
        X: Linear combinations of the coefficients to test: a 1-d array (one combination) or a 2-d array with one row per combination, with all ``p``
            columns or without the intercept column (a column of ones is then added). Each row is tested against ``rhs``; the result holds one test
            per row and, if ``X`` has full row rank, the omnibus test of all of them (``QM`` is otherwise NaN). Also accepted: a
            [pairmat][metafopy.pairmat] specification. Default: none.
        rhs: Right-hand side of the hypotheses: a number or one value per combination (or per tested coefficient with ``btt``; a different length
            raises ``InputError``). Default: 0.
        adjust: Adjust the p-values for multiple testing, as R's ``p.adjust``: ``True`` for ``'bonferroni'``, ``False`` for ``'none'``, or one of
            ``'holm'``, ``'hochberg'``, ``'hommel'``, ``'bonferroni'``, ``'BH'``, ``'BY'``, ``'fdr'`` and ``'none'`` (a unique prefix is accepted).
            It applies to the p-values of the individual tests of the linear combinations, and to the omnibus p-value of a test of moderators (across
            the elements for a list ``btt``). Default: no adjustment.
        refit: For a likelihood-ratio test of REML models whose fixed effects differ, refit them with ML first (otherwise a warning is given).
        L: Old spelling of ``X``; if both are given, ``L`` is used.
        fixed: Match a string ``btt`` as a literal substring of the coefficient names instead of as a regular expression.
        test: With ``model2``: ``'LRT'`` (default) for the likelihood-ratio test, or ``'Wald'`` to test the coefficients that only the full model has,
            with the Wald test of that model (the models must use the same ``method``). A unique prefix is accepted; anything else raises
            ``ArgumentError``.
        df: With ``model2``: the degrees of freedom of the likelihood-ratio test, instead of the difference in the number of parameters (it is also
            added to the number of parameters of the full model).
        verbose: Add the reason to the error of a cluster-robust test that fails.
        abbrev: Remove ``factor(`` and ``)`` from the text of the hypotheses of a test of linear combinations (not described on metafor's reference
            page, but accepted by its source).
        digits: Number of decimals of the printout of the result (also used for the hypotheses of a test of linear combinations), or a mapping as for
            [predict][metafopy.predict]. Default: the ``digits`` option, else that of the model.
        att: For location-scale models only; giving it here raises ``InputError``.
        Z: For location-scale models only; giving it here raises ``InputError``.

    Returns:
        For a Wald test, a [WaldResult][metafopy.WaldResult]; for two models, an ``MVLRTResult`` with the fields ``df_full``, ``df_reduced``, ``LRT``,
        ``pval``, ``fit_stats_full``, ``fit_stats_reduced``, ``QE_full``, ``QE_reduced``, ``digits`` and ``method`` of [LRTResult][metafopy.LRTResult]
        (its ``tau2_f``, ``tau2_r`` and ``R2`` are NaN, as metafor stores NA); with ``test='Wald'``, a ``WaldResult``; for a list ``btt``, a
        ``ListAnovaResult`` (a list of Wald results).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid (for example ``att`` or ``Z`` is given, ``X`` has the wrong number of columns,
            ``btt`` selects nothing, ``adjust`` is unknown, the models differ in class, data or number of parameters, an ML refit fails, or a
            cluster-robust test cannot be computed).
        metafopy.errors.ArgumentError: ``test`` is neither ``'LRT'`` nor ``'Wald'``.

    Metafor reference:
        [anova.rma](https://wviechtb.github.io/metafor/reference/anova.rma.html)
    """
    X = metafopy.methods.pairmat.resolve(X, model)
    check_scale_arguments(att, Z)
    if abbrev:
        return abbreviate_hyp(anova_rma_mv(**{**locals(), 'abbrev': False}), True)
    x = model
    d = metafopy.mfopt.option_digits(digits, x.digits)
    method = None if adjust is None else metafopy.core.padjust.match_adjust(adjust)
    model2 = metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2')
    X = merge_x(X, L)
    rhs_given = rhs is not None
    rhs = 0.0 if rhs is None else rhs
    club = club_info(x.robust)
    if model2 is not None:
        check_same_class(x, model2)
        if check_test(test) == 'Wald':
            return dataclasses.replace(wald_between(x, model2, df), digits=d)
        if df is None and x.parms == model2.parms:
            raise metafopy.errors.InputError('models do not differ in the number of parameters')
        full, red = (x, model2) if df is not None or x.parms > model2.parms else (model2, x)
        if full.k != red.k or not numpy.allclose(full.yi, red.yi):
            raise metafopy.errors.InputError('The observed outcomes are not equal in the full and reduced model.')
        check_lrt_models(full, red)
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
        warn_not_nested(full, red)
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
            digits=d,
            method=full.method,
        )

    if X is not None and club is not None:
        return club_wald_x(x.beta, x.vb, club, X, x.p, x.int_incl, rhs_given, method, verbose, x.coef_names, d)

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
            QM=QM,
            QMdf=(m, df2),
            QMp=QMp,
            Xb=Xb,
            se=se,
            zval=zval,
            pval=numpy.asarray(pval, dtype=numpy.float64),
            ddf=None if x.test == 'z' else ddf,
            hyp=hypotheses(L, x.coef_names, numpy.broadcast_to(numpy.asarray(rhs, dtype=numpy.float64), (m,)), d),
            digits=d,
        )

    if is_btt_list(btt):
        specs = typing.cast(list[typing.Any], btt)
        return make_list([typing.cast(WaldResult, anova_rma_mv(x, btt=b, fixed=fixed, digits=d)) for b in specs], specs, method)
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
    return dataclasses.replace(out, digits=d)


def refit_ml_mv(res: metafopy.models.mv_result.RMAMVResult) -> metafopy.models.mv_result.RMAMVResult:
    """``res`` refitted with ML via ``update(res, method='ML')``, as ``anova.rma(..., refit=TRUE)`` does.

    Raises:
        metafopy.errors.InputError: the model has no recorded call or the refit failed.
    """
    if res.call is None:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed (no recorded call).')
    try:
        out = res.call(method='ML')
    except (metafopy.errors.MetafopyError, numpy.linalg.LinAlgError, ValueError) as err:
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.') from err
    if not isinstance(out, metafopy.models.mv_result.RMAMVResult):
        raise metafopy.errors.InputError('Refitting the model with ML estimation failed.')
    return out
