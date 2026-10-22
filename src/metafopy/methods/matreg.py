"""Regression models fitted to a correlation or covariance matrix (metafor's ``matreg``).

R sources: https://github.com/wviechtb/metafor/blob/master/R/matreg.r,
https://github.com/wviechtb/metafor/blob/master/R/methods.matreg.r,
https://github.com/wviechtb/metafor/blob/master/R/predict.matreg.r and
https://github.com/wviechtb/metafor/blob/master/R/print.matreg.r. Documentation:
https://wviechtb.github.io/metafor/reference/matreg.html.

With ``n``, the standard errors are those of a regression on raw data with ``n`` observations. With ``V``, the
variance-covariance matrix of the coefficients is obtained by the multivariate delta method as described in Becker (1992),
https://doi.org/10.3102/10769986017004341 (with the corrections of Becker, 1995, https://doi.org/10.3102/10769986020001100),
the references of metafor's documentation (which also cites Becker & Aloe, 2019, Model-based meta-analysis and related approaches,
in *The Handbook of Research Synthesis and Meta-Analysis*, 3rd ed., pp. 339-363).
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.sparse
import scipy.stats

import metafopy.core.linalg
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.confint
import metafopy.methods.predict
import metafopy.mfopt
import metafopy.models.fit_methods
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['MatregConfintResult', 'MatregPredictResult', 'MatregResult', 'matreg']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Matrix = numpy.typing.ArrayLike | pandas.DataFrame
type Transf = collections.abc.Callable[..., typing.Any]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type NAAction = typing.Literal['omit', 'exclude', 'pass', 'fail']

LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
# auditme: accept[non-alphabetical-literal] order is metafor's and is meaningful (rows of fit.stats)
FIT_ROWS = ('ll', 'dev', 'AIC', 'BIC', 'AICc')


def chol_inverse(x: FloatArray) -> FloatArray | None:
    """R's ``chol2inv(chol(x))``; ``None`` when ``x`` is not positive definite."""
    try:
        L = numpy.linalg.cholesky(x)
    except numpy.linalg.LinAlgError:
        return None
    Linv = numpy.linalg.solve(L, numpy.eye(x.shape[0]))
    return numpy.asarray(Linv.T @ Linv, dtype=numpy.float64)


def fmt(x: float, digits: int) -> str:
    return metafopy.printing.fmt_num(float(x), digits)


@dataclasses.dataclass(frozen=True)
class MatregConfintResult:
    """Confidence intervals of a ``matreg`` fit: ``tab`` has the columns ``estimate``, ``ci.lb`` and ``ci.ub``.

    Attributes:
        tab: Data frame with one row per coefficient and the columns ``estimate``, ``ci.lb`` and ``ci.ub``.
        digits: Number of decimals used when the result is printed.
    """

    tab: pandas.DataFrame
    digits: int = 4

    def __str__(self) -> str:
        cols = [(name, [fmt(v, self.digits) for v in self.tab[name]]) for name in self.tab.columns]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, [str(i) for i in self.tab.index]), ''])


@dataclasses.dataclass(frozen=True, slots=True)
class MatregPredictResult(metafopy.methods.predict.PredictResult):
    """Predicted values of a ``matreg`` fit (``predict.matreg``), without prediction intervals (``pi_lb`` and ``pi_ub`` are NaN).

    ``X`` is the matrix of predictor values (``addx=True``), ``vcov`` the variance-covariance matrix of the predicted
    values (``vcov=True`` without ``transf``), ``ddf`` the degrees of freedom of the t-distribution (``None`` for a
    model fitted with ``V``) and ``transf`` whether a transformation was applied (``se`` is then NaN, as in metafor).

    The attributes ``pred``, ``se``, ``ci_lb``, ``ci_ub``, ``pi_lb`` and ``pi_ub`` are those of the [predict][metafopy.predict] result.

    Attributes:
        X: Predictor values of the predictions as a data frame, or ``None`` (set with ``addx=True``).
        vcov: Variance-covariance matrix of the predicted values, or ``None`` (set with ``vcov=True`` without ``transf``).
        ddf: Degrees of freedom of the t-distribution, or ``None`` for a model fitted with ``V``.
        digits: Number of decimals used when the result is printed.
        transf: Whether a transformation was applied to the predictions.
    """

    X: pandas.DataFrame | None = None
    vcov: pandas.DataFrame | None = None
    ddf: float | None = None
    digits: int = 4
    transf: bool = False

    def __str__(self) -> str:
        # print.list.rma (predict.matreg has no 'method' element): pred, se, ci.lb, ci.ub and the first column of X are
        # formatted with 'digits', the other columns of X as R prints a numeric data.frame column
        listed: list[tuple[str, FloatArray]] = [('pred', self.pred)]
        if not self.transf:
            listed.append(('se', self.se))
        listed += [('ci.lb', self.ci_lb), ('ci.ub', self.ci_ub)]
        cols = [(name, [fmt(v, self.digits) for v in vals]) for name, vals in listed]
        if self.X is not None:
            for j, name in enumerate(self.X.columns):
                vals = self.X[name].to_numpy(dtype=numpy.float64)
                cells = [fmt(v, self.digits) for v in vals] if j == 0 else metafopy.methods.predict.r_numeric_column(vals)
                cols.append((f'X.{name}', cells))
        # auditme: accept[literal-duplication] R's blank row label of a single prediction
        slab = list(self.slab) if self.slab is not None else [''] * len(self.pred)
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, slab), ''])


@dataclasses.dataclass(frozen=True)
class MatregResult:
    """A regression model fitted to a correlation or covariance matrix (metafor's ``matreg`` object).

    ``tab`` holds the columns ``beta``, ``se``, ``tval`` and ``df`` (``test == 't'``, a model fitted with ``n``) or
    ``zval`` (``test == 'z'``, a model fitted with ``V``), ``pval``, ``ci.lb`` and ``ci.ub``. The ``F`` test
    (``F``, ``Fdf``, ``Fp``), ``R2adj``, the residual variances and ``fit_stats`` exist only for ``test == 't'``;
    the Wald test (``QM``, ``QMdf``, ``QMp``) only for ``test == 'z'``. ``level`` is the alpha level (``0.05`` for 95%)
    and ``intercept`` is true for a covariance matrix input (``cov=True``).

    Attributes:
        tab: Table of coefficients, one row per coefficient (see above for the columns).
        vb: Variance-covariance matrix of the coefficients.
        R2: Proportion of variance of the outcome explained by the predictors.
        p: Number of estimated coefficients for a model fitted with ``n`` (the intercept included when ``cov=True``); the number of predictors
            for a model fitted with ``V``.
        parms: Number of estimated parameters: ``p + 1`` for a model fitted with ``n`` (the residual variance counts), ``p`` for one fitted
            with ``V``.
        digits: Number of decimals used when the result is printed.
        test: ``'t'`` for a model fitted with ``n``, ``'z'`` for one fitted with ``V``.
        level: The alpha level of the confidence intervals.
        intercept: Whether the model has an intercept.
        R2adj: Adjusted ``R^2`` (``test == 't'`` only).
        F: The F test of the predictors as a tuple (statistic, numerator degrees of freedom, denominator degrees of freedom) (``test == 't'``
            only).
        Fdf: Degrees of freedom of the F test (``test == 't'`` only).
        Fp: P-value of the F test (``test == 't'`` only).
        df_residual: Residual degrees of freedom (``test == 't'`` only).
        sigma2_ml: Maximum-likelihood estimate of the residual variance (``test == 't'`` only).
        sigma2_reml: Restricted maximum-likelihood estimate of the residual variance (``test == 't'`` only).
        nobs: Number of observations (``test == 't'`` only).
        deviance: Deviance of the model (``test == 't'`` only).
        fit_stats: Log-likelihood, deviance, AIC, BIC and AICc under ML and REML (``test == 't'`` only).
        QM: Wald-type test statistic of the coefficients (``test == 'z'`` only).
        QMdf: Degrees of freedom of the Wald test (``test == 'z'`` only).
        QMp: P-value of the Wald test (``test == 'z'`` only).
    """

    tab: pandas.DataFrame
    vb: pandas.DataFrame
    R2: float
    p: int
    parms: int
    digits: int
    test: typing.Literal['t', 'z']
    level: float
    intercept: bool
    R2adj: float | None = None
    F: tuple[float, float, float] | None = None
    Fdf: tuple[float, float] | None = None
    Fp: float | None = None
    df_residual: float | None = None
    sigma2_ml: float | None = None
    sigma2_reml: float | None = None
    nobs: float | None = None
    deviance: float | None = None
    fit_stats: pandas.DataFrame | None = None
    QM: float | None = None
    QMdf: tuple[int, None] | None = None
    QMp: float | None = None

    # -- accessors ----------------------------------------------------------------

    def coef(self) -> pandas.Series:
        """The estimated coefficients (``coef.matreg``)."""
        return pandas.Series(self.tab['beta'].to_numpy(dtype=numpy.float64), index=list(self.tab.index), name='estimate')

    def vcov(self) -> pandas.DataFrame:
        """The variance-covariance matrix of the coefficients (``vcov.matreg``)."""
        return self.vb

    def sigma(self, REML: bool = True) -> float | None:
        """The residual standard deviation (``sigma.matreg``); ``None`` with a warning for a model fitted with ``V``."""
        if self.test != 't':
            warnings.warn("Model does not contain a 'sigma' estimate.", stacklevel=2)
            return None
        return metafopy.core.linalg.rsqrt(typing.cast(float, self.sigma2_reml if REML else self.sigma2_ml))

    def stat(self, row: str, REML: bool) -> float:
        if self.fit_stats is None:
            raise metafopy.errors.InternalError('fit_stats is missing for a model fitted with n')
        return float(typing.cast(float, self.fit_stats.at[row, 'REML' if REML else 'ML']))

    def logLik(self, REML: bool = False) -> metafopy.models.fit_methods.LogLik | None:
        """The (restricted) log-likelihood (``logLik.matreg``), printed as R prints a ``logLik``; ``None`` with a warning for a model fitted with ``V``."""
        if self.test != 't':
            warnings.warn('Cannot compute log-likelihood for this type of model object.', stacklevel=2)
            return None
        return metafopy.models.fit_methods.LogLik(self.stat('ll', REML), self.parms)

    def AIC(self, *others: MatregResult, k: float = 2, correct: bool = False, REML: bool = False) -> float | pandas.DataFrame | None:
        """AIC (or AICc with ``correct=True``) of this fit (``AIC.matreg``).

        With further fits in ``others`` a frame with the columns ``df`` (number of parameters) and ``AIC`` (or ``AICc``),
        one row per fit. As in metafor, ``k`` is accepted but not used.
        """
        del k
        if self.test != 't':
            warnings.warn('Cannot compute AIC for this type of model object.', stacklevel=2)
            return None
        row = 'AICc' if correct else 'AIC'
        if not others:
            return self.stat(row, REML)
        fits = [self, *others]
        return pandas.DataFrame({'df': [f.parms for f in fits], row: [f.stat(row, REML) for f in fits]})

    def BIC(self, *others: MatregResult, REML: bool = False) -> float | pandas.DataFrame | None:
        """BIC of this fit (``BIC.matreg``); with further fits in ``others`` a frame with the columns ``df`` and ``BIC``."""
        if self.test != 't':
            warnings.warn('Cannot compute BIC for this type of model object.', stacklevel=2)
            return None
        if not others:
            return self.stat('BIC', REML)
        fits = [self, *others]
        return pandas.DataFrame({'df': [f.parms for f in fits], 'BIC': [f.stat('BIC', REML) for f in fits]})

    def confint(self, *args: typing.Any, **kwargs: typing.Any) -> MatregConfintResult:
        return confint_matreg(self, *args, **kwargs)

    def predict(self, *args: typing.Any, **kwargs: typing.Any) -> MatregPredictResult:
        return predict_matreg(self, *args, **kwargs)

    # -- printing -------------------------------------------------------------------

    def table_lines(self, digits: int, signif_stars: bool, signif_legend: bool) -> list[str]:
        """``print.matreg``."""
        tab = self.tab
        beta = tab['beta'].to_numpy(dtype=numpy.float64)
        stat = 'tval' if self.test == 't' else 'zval'
        cols: list[tuple[str, list[str]]] = [
            ('estimate', [fmt(v, digits) for v in beta]),
            ('se', [fmt(v, digits) for v in tab['se']]),
            (stat, [fmt(v, digits) for v in tab[stat]]),
        ]
        if self.test == 't':
            df = tab['df'].to_numpy(dtype=numpy.float64).copy()
            df[numpy.isnan(beta)] = numpy.nan
            finite = ~numpy.isnan(df)
            cells = numpy.array(['NA'] * len(df), dtype=object)
            if finite.any():
                cells[finite] = metafopy.printing.fmt_df(df[finite])
            cols.append(('df', [str(c) for c in cells]))
        cols += [
            ('pval', [metafopy.printing.fmt_pval(float(v), digits) for v in tab['pval']]),
            ('ci.lb', [fmt(v, digits) for v in tab['ci.lb']]),
            ('ci.ub', [fmt(v, digits) for v in tab['ci.ub']]),
        ]
        if signif_stars:
            cols.append(('', [metafopy.printing.signif_stars(float(v)) or ' ' for v in tab['pval']]))
        lines = ['', *metafopy.methods.blup.format_r_frame(cols, [str(i) for i in tab.index], gap=2, width=10**6)]
        if signif_legend:
            lines += ['', '---', LEGEND]
        return [*lines, '']

    def summary(self, digits: int | None = None, signif_stars: bool = True, signif_legend: bool | None = None) -> str:
        """``print(summary(x))``: the coefficient table followed by the fit statistics."""
        d = metafopy.mfopt.stored_digits(digits, self.digits)
        lines = self.table_lines(d, signif_stars, signif_stars if signif_legend is None else signif_legend)
        if self.test == 't':
            F = typing.cast(tuple[float, float, float], self.F)
            Fdf = typing.cast(tuple[float, float], self.Fdf)
            p_text = metafopy.printing.fmt_pval(typing.cast(float, self.Fp), d)
            df1, df2 = (metafopy.printing.fmt_df_scalar(v) for v in Fdf)
            rse = fmt(metafopy.core.linalg.rsqrt(typing.cast(float, self.sigma2_reml)), d)
            lines += [
                f'Residual standard error: {rse} on {df2} degrees of freedom',
                f'Multiple R-squared: {fmt(self.R2, d)},  Adjusted R-squared: {fmt(typing.cast(float, self.R2adj), d)}',
                f'F-statistic: {fmt(F[0], d)} on {df1} and {df2} DF,  p-value: {p_text}',
            ]
        else:
            qm = typing.cast(float, self.QM)
            qm_p = metafopy.models.rma_uni.pfmt(typing.cast(float, self.QMp), d)
            lines.append(f'R^2: {fmt(self.R2, d)}, QM(df = {self.p}) = {fmt(qm, d)}, p-val {qm_p}')
        return '\n'.join([*lines, ''])

    def __str__(self) -> str:
        return '\n'.join(self.table_lines(self.digits, True, True))


def pmatch(name: str, choices: collections.abc.Sequence[str]) -> int | None:
    """R's ``pmatch()`` for one name: an exact match, else the unique choice that starts with ``name``."""
    if name in choices:
        return choices.index(name)
    hits = [i for i, c in enumerate(choices) if name and c.startswith(name)]
    return hits[0] if len(hits) == 1 else None


def as_square(x: Matrix, what: str) -> tuple[FloatArray, list[str] | None]:
    """The values of a square matrix and its dimension names (column names, else row names).

    Raises:
        metafopy.errors.InputError: ``x`` is not a square matrix.
    """
    names: list[str] | None = None
    if isinstance(x, pandas.DataFrame):
        values = x.to_numpy(dtype=numpy.float64)
        names = index_labels(x.columns)
        if names is None:
            names = index_labels(x.index)
    elif scipy.sparse.issparse(x):
        values = numpy.asarray(typing.cast(typing.Any, x).toarray(), dtype=numpy.float64)
    else:
        values = numpy.array(x, dtype=numpy.float64)
        given = getattr(x, 'colnames', None) or getattr(x, 'rownames', None)  # a matrix with dimnames (metafopy.core.labelled.LabelledMatrix)
        if given is not None and values.ndim == 2 and len(given) == values.shape[1]:
            names = [str(n) for n in given]
    if values.ndim == 0:
        values = values.reshape(1, 1)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise metafopy.errors.InputError(f"Argument '{what}' must be a square matrix.")
    return values, names


def lower_to_upper(x: FloatArray) -> FloatArray:
    """``x[upper.tri(x)] <- t(x)[upper.tri(x)]``: the lower triangle mirrored into the upper one."""
    out = numpy.tril(x)
    return out + numpy.tril(x, -1).T


def resolve_y(y: int | str, names: list[str] | None, p: int) -> int:
    """
    Raises:
        metafopy.errors.InputError: ``y`` cannot be found in ``R`` or is out of range.
    """
    if isinstance(y, str):
        if names is None:
            raise metafopy.errors.InputError("'R' must have dimension names when specifying a variable name for 'y'.")
        if len(set(names)) != len(names):
            raise metafopy.errors.InputError("Dimension names of 'R' must be unique.")
        pos = pmatch(y, names)
        if pos is None:
            raise metafopy.errors.InputError(f"Could not find variable '{y}' in the 'R' matrix.")
        return pos
    yi = round(y)
    if yi < 0 or yi > p - 1:
        raise metafopy.errors.InputError(f"Index 'y' must be >= 0 or <= {p - 1}.")
    return yi


def resolve_x(x: collections.abc.Sequence[int | str] | int | str, names: list[str] | None, p: int) -> list[int]:
    """
    Raises:
        metafopy.errors.InputError: an element of ``x`` cannot be found in ``R``, is out of range or repeated.
    """
    xs: list[int | str] = [x] if isinstance(x, int | str) else list(x)
    if xs and all(isinstance(v, str) for v in xs):
        if names is None:
            raise metafopy.errors.InputError("'R' must have dimension names when specifying variable names for 'x'.")
        if len(set(names)) != len(names):
            raise metafopy.errors.InputError("Dimension names of 'R' must be unique.")
        found = [pmatch(str(v), names) for v in xs]
        missing = [str(v) for v, pos in zip(xs, found, strict=True) if pos is None]
        if missing:
            raise metafopy.errors.InputError(f"Could not find variable{'s' if len(missing) > 1 else ''} '{', '.join(missing)}' in the 'R' matrix.")
        return [typing.cast(int, pos) for pos in found]
    out = [round(typing.cast(float, v)) for v in xs]
    if len(set(out)) != len(out):
        raise metafopy.errors.InputError("Argument 'x' should not contain duplicated elements.")
    if any(v < 0 or v > p - 1 for v in out):
        raise metafopy.errors.InputError(f"Indices in 'x' must be >= 0 or <= {p - 1}.")
    return out


def parse_formula(formula: str) -> tuple[str, list[str]]:
    """``y ~ x1 + x2``: the outcome and the (unique) predictor names.

    Raises:
        metafopy.errors.InputError: the formula has no left-hand side.
    """
    lhs, _, rhs = formula.partition('~')
    if not lhs.strip():
        raise metafopy.errors.InputError("The formula specified via the 'y' argument does not have a left-hand side.")
    xs = list(dict.fromkeys(v.strip() for v in rhs.split('+')))
    return lhs.strip(), xs


def matreg(
    y: int | str,
    x: collections.abc.Sequence[int | str] | int | str | None = None,
    R: Matrix | None = None,
    n: float | None = None,
    V: Matrix | None = None,
    cov: bool = False,
    means: numpy.typing.ArrayLike | None = None,
    ztor: bool = False,
    nearpd: bool = False,
    level: float = 95,
    digits: int | None = None,
) -> MatregResult:
    """Fit a regression model to a correlation or covariance matrix (metafor's ``matreg``).

    ``y`` is the outcome: a 0-based index into ``R``, a variable name (``R`` must then be a frame with dimension
    names) or a formula string such as ``'perf ~ acog + asom + conf'`` (which also gives ``x``). ``x`` are the
    predictors (indices or names; by default every other variable). Give either ``n``, the sample size the matrix is
    based on, or ``V``, the variance-covariance matrix of the elements of the lower triangle of ``R`` (column-wise,
    with the diagonal when ``cov=True``), for instance from :func:`metafopy.rcalc` or the ``vcov()`` of a multivariate
    model of correlations. ``cov=True`` treats ``R`` as a covariance matrix (the model then has an intercept, which
    needs ``means``), ``ztor=True`` back-transforms Fisher r-to-z transformed correlations (and their ``V``) and
    ``nearpd=True`` replaces a non-positive-definite ``R[x, x]`` by the nearest correlation matrix
    (:func:`metafopy.core.linalg.near_pd`, a port of ``Matrix::nearPD``).

    Without names, the coefficients are labelled ``x1``, ``x2``, ... by their 1-based position in ``R``, as in R.
    See <https://wviechtb.github.io/metafor/reference/matreg.html>.

    Args:
        y: The outcome variable: a 0-based index into ``R``, a variable name (``R`` must then be a data frame with dimension names) or a
            formula string such as ``'perf ~ acog + asom + conf'``, which also gives ``x``.
        x: The predictor variables: a sequence of 0-based indices or names (or a single one). Ignored when ``y`` is a formula. ``None`` (the
            default) uses every variable other than ``y``.
        R: The correlation or covariance matrix (an array or data frame; required). Only its lower triangle (including the diagonal) is used.
        n: The sample size the matrix is based on; gives the standard errors of a regression on raw data (t tests). Give ``n`` or ``V``, not both.
        V: The variance-covariance matrix of the elements of the lower triangle of ``R``, column-wise (with the diagonal when ``cov=True``),
            for instance from [rcalc][metafopy.rcalc] or the ``vcov()`` of a multivariate model of correlations. Gives the standard errors by the
            multivariate delta method (z tests). Give ``n`` or ``V``, not both.
        cov: Treat ``R`` as a covariance matrix; the model then has an intercept, which needs ``means`` (default ``False``, a correlation matrix).
        means: The means of the variables, for the intercept when ``cov=True``. Without them the intercept estimate is NaN.
        ztor: ``R`` (and ``V``) hold Fisher r-to-z transformed correlations, which are back-transformed to raw correlations first (default
            ``False``).
        nearpd: Replace an ``R[x, x]`` that cannot be inverted (is not positive definite) by the nearest correlation matrix (a port of
            ``Matrix::nearPD``) instead of failing (default ``False``). Treat the results with caution. metafor's undocumented ``nearPD`` alias
            of this argument is not accepted.
        level: Confidence interval level in percent (default 95), or a value below 1 read as in metafor.
        digits: Number of decimals used when the result is printed (default 4).

    Returns:
        A [MatregResult][metafopy.MatregResult]: the table ``tab`` of (standardised, for a correlation matrix) regression coefficients with
        their standard errors, test statistics, p-values and confidence interval bounds, the variance-covariance matrix ``vb`` of the
        coefficients, ``R2`` and further fit statistics. It has ``coef``, ``vcov``, ``confint`` and ``predict`` methods.

    Raises:
        metafopy.errors.InputError: an argument is missing, invalid or inconsistent with ``R``.
        metafopy.errors.ArgumentError: a logical argument is not a bool.
        metafopy.errors.EstimationError: ``R[x, x]`` cannot be inverted, or ``n`` does not exceed the number of regression coefficients.
    """
    if digits is None:
        digits = 4
    alpha = metafopy.models.rma_uni.level_to_alpha(level)

    if R is None:
        raise metafopy.errors.InputError("Must specify the 'R' argument.")
    Rm, names = as_square(R, 'R')
    p = Rm.shape[0]
    if p <= 1:
        raise metafopy.errors.InputError("The 'R' matrix must be at least of size 2x2.")

    if isinstance(y, str) and '~' in y:
        y_name, x_names = parse_formula(y)
        yi = resolve_y(y_name, names, p)
        xs = resolve_x(x_names, names, p)
    else:
        if not isinstance(y, int | str) or isinstance(y, bool):
            raise metafopy.errors.InputError("Argument 'y' should specify a single variable.")
        yi = resolve_y(y, names, p)
        xs = resolve_x([i for i in range(p) if i != yi] if x is None else x, names, p)
    if yi in xs:
        raise metafopy.errors.InputError("Variable 'y' should not be an element of 'x'.")

    if V is None and n is None:
        raise metafopy.errors.InputError("Either 'V' or 'n' must be specified.")
    if V is not None and n is not None:
        raise metafopy.errors.InputError("Either 'V' or 'n' must be specified, not both.")
    for flag, label in ((cov, 'cov'), (ztor, 'ztor')):
        if not isinstance(flag, bool):
            raise metafopy.errors.ArgumentError(f"Argument '{label}' must be either TRUE or FALSE.")
    if cov and ztor:
        raise metafopy.errors.InputError("Cannot use a covariance matrix as input when 'ztor=TRUE'.")

    m = len(xs)
    Rm = lower_to_upper(Rm)
    s = p * (p + 1) // 2 if cov else p * (p - 1) // 2

    Vm: FloatArray | None = None
    if V is not None:
        Vm, _ = as_square(V, 'V')
        Vm = lower_to_upper(Vm)
        if Vm.shape[0] != s:
            raise metafopy.errors.InputError(f"Dimensions of 'V' ({Vm.shape[0]}x{Vm.shape[1]}) do not match the number of elements in 'R' ({s}).")

    if ztor:
        if Vm is not None:
            # R[lower.tri(R)], i.e. the lower triangle column by column
            rows, cols = numpy.triu_indices(p, 1)
            zij = Rm[cols, rows]
            Dmat = numpy.diag(2 / (numpy.cosh(2 * zij) + 1))
            Vm = Dmat @ Vm @ Dmat
        Rm = numpy.tanh(Rm)
        numpy.fill_diagonal(Rm, 1.0)

    S = Rm
    if cov:
        sd = numpy.sqrt(numpy.diag(S))
        Rm = S / numpy.outer(sd, sd)
        sdy = float(sd[yi])
        sdx = sd[xs]
    else:
        if numpy.any(numpy.abs(Rm[~numpy.isnan(Rm)]) > 1):
            raise metafopy.errors.InputError("Argument 'R' must be a correlation matrix, but contains values outside [-1,1].")
        Rm = Rm.copy()
        numpy.fill_diagonal(Rm, 1.0)
        sdy = 1.0
        sdx = numpy.ones(m)

    Rxy = Rm[numpy.ix_(xs, [yi])]
    Rxx = Rm[numpy.ix_(xs, xs)]
    invRxx = chol_inverse(Rxx)
    if invRxx is None:
        if not nearpd:
            raise metafopy.errors.EstimationError('Cannot invert R[x,x] matrix.')
        warnings.warn('Cannot invert R[x,x] matrix. Using nearPD(). Treat results with caution.', stacklevel=2)
        Rxx = numpy.asarray(metafopy.core.linalg.near_pd(Rxx, corr=True).mat, dtype=numpy.float64)
        invRxx = chol_inverse(Rxx)
        if invRxx is None:
            raise metafopy.errors.EstimationError('Still cannot invert R[x,x] matrix.')

    b = invRxx @ Rxy
    rnames = [names[i] for i in xs] if names is not None else [f'x{i + 1}' for i in xs]

    has_means = False
    mvec = numpy.zeros(p)
    if cov and means is not None:
        mvec = numpy.asarray(means, dtype=numpy.float64).ravel()
        if mvec.shape[0] != p:
            raise metafopy.errors.InputError(f"Length of 'means' ({mvec.shape[0]}) does not match the dimensions of 'R' ({p}x{p}).")
        has_means = True

    if Vm is None:
        if numpy.size(typing.cast(typing.Any, n)) != 1:
            raise metafopy.errors.InputError("Argument 'n' should be a single number.")
        return fit_with_n(
            float(numpy.asarray(n, dtype=numpy.float64).reshape(-1)[0]), b, Rxy, Rxx, invRxx, S, xs, yi, sdy, sdx, mvec, has_means, cov, rnames, alpha, digits
        )
    return fit_with_v(Vm, b, Rxy, invRxx, S, xs, yi, sdy, sdx, mvec, has_means, cov, rnames, p, s, alpha, digits)


def fit_with_n(
    n: float,
    b: FloatArray,
    Rxy: FloatArray,
    Rxx: FloatArray,
    invRxx: FloatArray,
    S: FloatArray,
    xs: list[int],
    yi: int,
    sdy: float,
    sdx: FloatArray,
    mvec: FloatArray,
    has_means: bool,
    cov: bool,
    rnames: list[str],
    alpha: float,
    digits: int,
) -> MatregResult:
    """The ``n`` branch of ``matreg``: standard errors, t-tests, the F-test and the fit statistics of a regression on raw data.

    Raises:
        metafopy.errors.EstimationError: ``n`` does not exceed the number of coefficients.
    """
    m = len(xs)
    df = n - m - (1 if cov else 0)
    if df <= 0:
        raise metafopy.errors.EstimationError("Cannot fit model when 'n' is equal to or less than the number of regression coefficients.")
    sse = 1 - float((b.T @ Rxy)[0, 0])
    mse = sse / df
    vb = mse * invRxx
    R2 = 1 - sse
    R2adj = 1 - (1 - R2) * ((n - (1 if cov else 0)) / df)
    Fval = (R2 / m) / mse
    Fp = float(scipy.stats.f.sf(Fval, m, df))
    mse = sdy**2 * (n - 1) * (1 - R2) / df

    beta = b[:, 0].copy()
    if cov:
        beta = beta * sdy / sdx
        beta = numpy.concatenate([[mvec[yi] - float(mvec[xs] @ beta)], beta])
        rnames = ['intrcpt', *rnames]
        Sxx = S[numpy.ix_(xs, xs)]
        bd = numpy.zeros((m + 1, m + 1))
        bd[1:, 1:] = Sxx
        c1 = numpy.concatenate([[1.0], mvec[xs]])
        XtX = (n - 1) * bd + n * numpy.outer(c1, c1)
        invXtX = chol_inverse(XtX)
        if invXtX is None:
            vb = numpy.full((m + 1, m + 1), numpy.nan)
            warnings.warn('Cannot obtain var-cov matrix of the regression coefficients.', stacklevel=3)
        else:
            vb = mse * invXtX
        if not has_means:
            beta[0] = numpy.nan
            vb[0, :] = numpy.nan
            vb[:, 0] = numpy.nan
    else:
        XtX = Rxx * (n - 1)

    with numpy.errstate(invalid='ignore'):
        se = numpy.sqrt(numpy.diag(vb))
    tval = beta / se
    pval = 2 * scipy.stats.t.sf(numpy.abs(tval), df)
    crit = float(scipy.stats.t.isf(alpha / 2, df))
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    p = int(numpy.sum(~numpy.isnan(beta)))
    parms = p + 1
    deviance = mse * df
    sigma2_ml = mse * df / n
    sigma2_reml = mse
    with numpy.errstate(invalid='ignore', divide='ignore'):
        ll_ml = float(-n / 2 * numpy.log(2 * math.pi * sigma2_ml) - 1 / 2 * deviance / sigma2_ml)
        ll_reml = float(-df / 2 * numpy.log(2 * math.pi * sigma2_reml) - 1 / 2 * deviance / sigma2_reml - 1 / 2 * numpy.linalg.slogdet(XtX)[1])
    big_ml = max(n, parms + 2)
    big_reml = max(n - p, parms + 2)
    fit_stats = pandas.DataFrame(
        {
            'ML': [ll_ml, deviance, -2 * ll_ml + 2 * parms, -2 * ll_ml + parms * math.log(n), -2 * ll_ml + 2 * parms * big_ml / (big_ml - parms - 1)],
            'REML': [
                ll_reml,
                -2 * ll_reml,
                -2 * ll_reml + 2 * parms,
                -2 * ll_reml + parms * math.log(n - p),
                -2 * ll_reml + 2 * parms * big_reml / (big_reml - parms - 1),
            ],
        },
        index=list(FIT_ROWS),
    )
    tab = pandas.DataFrame(
        {'beta': beta, 'se': se, 'tval': tval, 'df': numpy.full(len(beta), df), 'pval': pval, 'ci.lb': ci_lb, 'ci.ub': ci_ub},
        index=rnames,
    )
    return MatregResult(
        tab=tab,
        vb=pandas.DataFrame(vb, index=rnames, columns=rnames),
        R2=R2,
        p=p,
        parms=parms,
        digits=metafopy.mfopt.reduce_digits(digits),
        test='t',
        level=alpha,
        intercept=cov,
        R2adj=R2adj,
        F=(Fval, float(m), df),
        Fdf=(float(m), df),
        Fp=Fp,
        df_residual=df,
        sigma2_ml=sigma2_ml,
        sigma2_reml=sigma2_reml,
        nobs=n,
        deviance=deviance,
        fit_stats=fit_stats,
    )


def fit_with_v(
    V: FloatArray,
    b: FloatArray,
    Rxy: FloatArray,
    invRxx: FloatArray,
    S: FloatArray,
    xs: list[int],
    yi: int,
    sdy: float,
    sdx: FloatArray,
    mvec: FloatArray,
    has_means: bool,
    cov: bool,
    rnames: list[str],
    p: int,
    s: int,
    alpha: float,
    digits: int,
) -> MatregResult:
    """The ``V`` branch of ``matreg``: the delta-method variance-covariance matrix ``A V A'`` of the coefficients and z-tests."""
    m = len(xs)
    R2 = float((b.T @ Rxy)[0, 0])
    if cov:
        b = b * sdy / sdx[:, None]
        Rxy = S[numpy.ix_(xs, [yi])]
        Dinv = numpy.diag(1 / sdx)
        invRxx = Dinv @ invRxx @ Dinv

    # U[i, j]: the position of R[i, j] in the column-wise lower triangle (with the diagonal for a covariance matrix)
    U = numpy.full((p, p), -1, dtype=numpy.intp)
    pos = 0
    for j in range(p):
        for i in range(j if cov else j + 1, p):
            U[i, j] = U[j, i] = pos
            pos += 1
    Uxx = U[numpy.ix_(xs, xs)]
    Uxy = U[xs, yi]

    A = numpy.zeros((m, s))
    for a in range(s):
        hits = numpy.argwhere(Uxx == a)
        if hits.size:
            r, c = hits[0]
            J = numpy.zeros((m, m))
            J[r, c] = J[c, r] = 1.0
            A[:, a] = -(invRxx @ J @ invRxx @ Rxy)[:, 0]
        rows = numpy.flatnonzero(Uxy == a)
        if rows.size:
            A[:, a] = invRxx[:, rows[0]]
    vb = A @ V @ A.T

    beta = b[:, 0].copy()
    if cov:
        beta = numpy.concatenate([[mvec[yi] - float(mvec[xs] @ beta)], beta])
        rnames = ['intrcpt', *rnames]
        X = numpy.vstack([mvec[xs][None, :], numpy.eye(m)])
        vb = X @ vb @ X.T
        if not has_means:
            beta[0] = numpy.nan
            vb[0, :] = numpy.nan
            vb[:, 0] = numpy.nan

    with numpy.errstate(invalid='ignore'):
        se = numpy.sqrt(numpy.diag(vb))
    zval = beta / se
    pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
    crit = float(scipy.stats.norm.isf(alpha / 2))
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    sub = slice(1, None) if cov else slice(None)
    inv = chol_inverse(vb[sub, sub])
    QM = math.nan if inv is None else float(beta[sub] @ inv @ beta[sub])
    QMp = float(scipy.stats.chi2.sf(QM, m))

    tab = pandas.DataFrame({'beta': beta, 'se': se, 'zval': zval, 'pval': pval, 'ci.lb': ci_lb, 'ci.ub': ci_ub}, index=rnames)
    return MatregResult(
        tab=tab,
        vb=pandas.DataFrame(vb, index=rnames, columns=rnames),
        R2=R2,
        p=m,
        parms=m,
        digits=metafopy.mfopt.reduce_digits(digits),
        test='z',
        level=alpha,
        intercept=cov,
        QM=QM,
        QMdf=(m, None),
        QMp=QMp,
    )


@metafopy.methods.confint.confint.register
def confint_matreg(model: MatregResult, parm: typing.Any = None, level: float | None = None, digits: int | None = None) -> MatregConfintResult:
    """Confidence intervals for the coefficients of a ``matreg`` fit (``confint.matreg``); ``parm`` is ignored (with a warning), as in metafor."""
    if parm is not None:
        warnings.warn("Argument 'parm' (currently) ignored.", stacklevel=2)
    alpha = model.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    if model.test == 't':
        crit = float(scipy.stats.t.isf(alpha / 2, typing.cast(float, model.df_residual)))
    else:
        crit = float(scipy.stats.norm.isf(alpha / 2))
    beta = model.tab['beta'].to_numpy(dtype=numpy.float64)
    se = model.tab['se'].to_numpy(dtype=numpy.float64)
    tab = pandas.DataFrame({'estimate': beta, 'ci.lb': beta - crit * se, 'ci.ub': beta + crit * se}, index=list(model.tab.index))
    return MatregConfintResult(tab=tab, digits=metafopy.mfopt.stored_digits(digits, model.digits))


def subsequence_distance(name: str, target: str) -> float:
    """``adist(name, target, costs=c(ins=1, sub=Inf, del=Inf))``: the number of insertions that turn ``name`` into ``target``."""
    it = iter(target)
    if all(ch in it for ch in name):
        return float(len(target) - len(name))
    return math.inf


def match_columns(colnames: list[str], model_names: list[str]) -> list[int]:
    """The model coefficient each column of ``newmods`` belongs to (``predict.matreg``'s name matching).

    Raises:
        metafopy.errors.InputError: a name has no unique match, is matched twice, or a coefficient has no column.
    """
    pos: list[int] = []
    for colname in colnames:
        d = numpy.array([subsequence_distance(colname, target) for target in model_names])
        if numpy.all(numpy.isinf(d)):
            raise metafopy.errors.InputError(f"Could not find variable '{colname}' in the model.")
        best = numpy.flatnonzero(d == d.min())
        if best.size > 1:
            raise metafopy.errors.InputError(f"Could not match up variable '{colname}' uniquely to a variable in the model.")
        pos.append(int(best[0]))
    if len(set(pos)) != len(pos):
        seen: set[int] = set()
        dups: list[str] = []
        for colname, ps in zip(colnames, pos, strict=True):
            if ps in seen and colname not in dups:
                dups.append(colname)
            seen.add(ps)
        raise metafopy.errors.InputError(f'Found multiple matches for the same variable name ({", ".join(dups)}).')
    if len(pos) != len(model_names):
        no_match = [name for i, name in enumerate(model_names) if i not in pos]
        if len(no_match) > 3:
            raise metafopy.errors.InputError(f"Argument 'newmods' does not specify values for these variables: {', '.join(no_match[:3])}, ...")
        if len(no_match) > 1:
            raise metafopy.errors.InputError(f"Argument 'newmods' does not specify values for these variables: {', '.join(no_match)}")
        raise metafopy.errors.InputError(f"Argument 'newmods' does not specify values for this variable: {no_match[0]}")
    return pos


def index_labels(index: typing.Any) -> list[str] | None:
    """The labels of a pandas index as strings; ``None`` for a default ``RangeIndex`` (R: no names)."""
    return None if isinstance(index, pandas.RangeIndex) else [str(v) for v in index]


def newmods_matrix(newmods: typing.Any) -> tuple[FloatArray, bool, list[str] | None, list[str] | None]:
    """``newmods`` as values, whether it is a vector (not a matrix), its column names and its row names.

    Raises:
        metafopy.errors.InputError: ``newmods`` is not numeric or has more than two dimensions.
    """
    colnames: list[str] | None = None
    rownames: list[str] | None = None
    values: numpy.typing.NDArray[typing.Any]
    if isinstance(newmods, collections.abc.Mapping):
        newmods = pandas.Series(newmods)
    if isinstance(newmods, pandas.DataFrame):
        colnames = index_labels(newmods.columns)
        rownames = index_labels(newmods.index)
        values, is_vector = newmods.to_numpy(), False
    elif isinstance(newmods, pandas.Series):
        colnames = index_labels(newmods.index)
        values, is_vector = newmods.to_numpy(), True
    else:
        values = numpy.asarray(newmods)
        is_vector = values.ndim < 2
    try:
        arr = numpy.asarray(values, dtype=numpy.float64)
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError("Argument 'newmods' should only contain numeric variables.") from err
    if arr.ndim > 2:
        raise metafopy.errors.InputError("Argument 'newmods' should be a vector or matrix.")
    return arr, is_vector, colnames, rownames


@metafopy.methods.predict.predict.register
def predict_matreg(
    model: MatregResult,
    newmods: typing.Any = None,
    *,
    intercept: bool | None = None,
    addx: bool = False,
    level: float | None = None,
    adjust: bool = False,
    digits: int | None = None,
    transf: Transf | None = None,
    targs: typing.Any = None,
    vcov: bool = False,
    na_action: NAAction = 'omit',
) -> MatregPredictResult:
    """Predicted values for a ``matreg`` fit (``predict.matreg``).

    ``newmods`` holds the predictor values: a vector (one prediction, or one per value for a model with a single
    predictor), a matrix with one row per prediction, or a :class:`pandas.Series`/:class:`pandas.DataFrame` whose
    names are matched to the coefficients as in metafor. For a model with an intercept (``cov=True``) the intercept
    column is added (with 0 for ``intercept=False``) unless ``newmods`` already has a column per coefficient.
    ``adjust=True`` applies a Bonferroni correction to the intervals; ``vcov=True`` adds the variance-covariance
    matrix of the predictions; ``transf`` (with ``targs``) transforms the predictions and interval bounds.
    ``na_action`` corresponds to R's ``na.action`` option: with ``'omit'`` (the default) predictions that are NA are dropped.

    Raises:
        metafopy.errors.InputError: ``newmods`` is missing or does not match the model.
    """
    x = model
    metafopy.methods.common.check_na_action(na_action)
    if newmods is None:
        raise metafopy.errors.InputError("Argument 'newmods' must be specified.")
    int_spec = intercept is not None
    if intercept is None:
        intercept = x.intercept
    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    digits = metafopy.mfopt.stored_digits(digits, x.digits)
    model_names = list(x.tab.index)
    p = len(model_names)

    arr, is_vector, colnames, rownames = newmods_matrix(newmods)
    ncol1 = arr.ndim < 2 or arr.shape[1] == 1
    singlemod = ncol1 and ((not x.intercept and p == 1) or (x.intercept and p == 2))
    rnames: list[str] | None
    if singlemod:
        X_new = arr.reshape(-1, 1)
        rnames = colnames if is_vector else rownames
        colnames = None
    else:
        if arr.ndim < 2 or arr.shape[0] == 1:
            X_new = arr.reshape(1, -1)
            rnames = None if is_vector else rownames
        else:
            X_new = arr
            rnames = rownames
        if colnames is not None and all(c != '' for c in colnames):
            mod_names = model_names[1:] if x.intercept else model_names
            pos = match_columns(colnames, mod_names)
            X_new = X_new[:, numpy.argsort(pos, kind='stable')]
            colnames = mod_names

    if not singlemod and X_new.shape[1] == p:
        if int_spec:
            warnings.warn("Arguments 'intercept' ignored when 'newmods' includes 'p' columns.", stacklevel=2)
    elif x.intercept:
        X_new = numpy.column_stack([numpy.full(X_new.shape[0], 1.0 if intercept else 0.0), X_new])
    if X_new.shape[1] != p:
        raise metafopy.errors.InputError(f"Dimensions of 'newmods' ({X_new.shape[1]}) do not the match dimensions of the model ({p}).")

    k_new = X_new.shape[0]
    beta_all = x.tab['beta'].to_numpy(dtype=numpy.float64)
    vb_all = x.vb.to_numpy(dtype=numpy.float64)
    pred = numpy.empty(k_new)
    vpred = numpy.empty(k_new)
    for i in range(k_new):
        xi = X_new[i]
        zero = xi == 0
        beta = numpy.where(zero, 0.0, beta_all)
        vb = vb_all.copy()
        vb[zero, :] = 0.0
        vb[:, zero] = 0.0
        pred[i] = xi @ beta
        vpred[i] = xi @ vb @ xi

    div = 2 * k_new if adjust else 2
    if x.test == 't':
        crit = float(scipy.stats.t.isf(alpha / div, typing.cast(float, x.df_residual)))
    else:
        crit = float(scipy.stats.norm.isf(alpha / div))
    vpred[vpred < 0] = numpy.nan
    se = numpy.sqrt(vpred)
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se

    vcovpred: FloatArray | None = None
    if vcov:
        full = X_new @ vb_all @ X_new.T
        vcovpred = (full + full.T) / 2
        numpy.fill_diagonal(vcovpred, se**2)

    do_transf = transf is not None
    if transf is not None:
        pred = metafopy.methods.common.apply_each(pred, transf, targs)
        se = numpy.full(k_new, numpy.nan)
        ci_lb = metafopy.methods.common.apply_each(ci_lb, transf, targs)
        ci_ub = metafopy.methods.common.apply_each(ci_ub, transf, targs)
    ci_lb, ci_ub = metafopy.methods.common.psort(ci_lb, ci_ub)

    slab = [str(i + 1) for i in range(k_new)] if rnames is None else rnames
    vcov_names = list(slab)
    if k_new == 1 and rnames is None:
        # auditme: accept[literal-duplication] R's blank row label of a single prediction
        slab = ['']

    not_na = ~numpy.isnan(pred) if na_action == 'omit' else numpy.ones(k_new, dtype=bool)
    X_out = pandas.DataFrame(X_new[not_na], columns=model_names) if addx else None
    vcov_out: pandas.DataFrame | None = None
    if vcovpred is not None and not do_transf:
        keep = numpy.flatnonzero(not_na)
        kept_names = [vcov_names[i] for i in keep]
        vcov_out = pandas.DataFrame(vcovpred[numpy.ix_(keep, keep)], index=kept_names, columns=kept_names)
    nan = numpy.full(int(not_na.sum()), numpy.nan)
    return MatregPredictResult(
        pred=pred[not_na],
        se=se[not_na],
        ci_lb=ci_lb[not_na],
        ci_ub=ci_ub[not_na],
        pi_lb=nan,
        pi_ub=nan.copy(),
        slab=tuple(s for s, keep_i in zip(slab, not_na, strict=True) if keep_i),
        X=X_out,
        vcov=vcov_out,
        ddf=x.df_residual if x.test != 'z' else None,
        digits=metafopy.mfopt.reduce_digits(digits),
        transf=do_transf,
    )
