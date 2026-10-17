"""Results shared by the fixed-effects binary-data models ``rma_mh`` and ``rma_peto``.

Both fit an intercept-only equal-effects model to 2x2 tables (or two-group
rate data) and report the same quantities as the corresponding ``rma.mh`` and
``rma.peto`` objects of metafor, see
https://wviechtb.github.io/metafor/reference/rma.mh.html and
https://wviechtb.github.io/metafor/reference/rma.peto.html.
"""

import abc
import collections.abc
import dataclasses
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.rma_uni
import metafopy.models.tables
import metafopy.printing

__all__ = ['FixedBinaryResult', 'fit_statistics', 'format_header', 'format_test_line', 'positive_max']

type FloatArray = numpy.typing.NDArray[numpy.float64]


def positive_max(x: float) -> float:
    """R's ``max(0, x)``, which is ``NA`` for ``NA`` (Python's ``max`` would return 0)."""
    return math.nan if math.isnan(x) else max(0.0, x)


def fit_statistics(yi: FloatArray, vi: FloatArray, fit: float, k_yi: int) -> tuple[metafopy.models.rma_uni.FitStats, metafopy.models.rma_uni.FitStats]:
    """ML and REML fit statistics of the equal-effects model; ``fit`` is ``QE`` (rma.mh) or ``RSS`` (rma.peto)."""
    nan_stats = metafopy.models.rma_uni.FitStats(math.nan, math.nan, math.nan, math.nan, math.nan)
    if k_yi < 1:
        return nan_stats, nan_stats
    with numpy.errstate(all='ignore'):
        wi = 1.0 / vi
        k = k_yi
        ll_ml = float(-0.5 * k * numpy.log(2 * math.pi) - 0.5 * numpy.sum(numpy.log(vi)) - 0.5 * fit)
        ll_reml = float(
            -0.5 * (k - 1) * numpy.log(2 * math.pi) + 0.5 * numpy.log(k) - 0.5 * numpy.sum(numpy.log(vi)) - 0.5 * numpy.log(numpy.sum(wi)) - 0.5 * fit
        )
        if numpy.any(vi <= 0):
            dev_ml = -2 * ll_ml
        else:
            dev_ml = float(-2 * (ll_ml + 0.5 * numpy.sum(numpy.log(2 * math.pi * vi))))
        big_ml = max(k, 3)
        big_reml = max(k - 1, 3)
        ml = metafopy.models.rma_uni.FitStats(
            ll=ll_ml,
            dev=dev_ml,
            AIC=-2 * ll_ml + 2,
            BIC=float(-2 * ll_ml + numpy.log(k)),
            AICc=-2 * ll_ml + 2 * big_ml / (big_ml - 2),
        )
        reml = metafopy.models.rma_uni.FitStats(
            ll=ll_reml,
            dev=-2 * ll_reml,
            AIC=-2 * ll_reml + 2,
            BIC=float(-2 * ll_reml + numpy.log(k - 1)),
            AICc=-2 * ll_reml + 2 * big_reml / (big_reml - 2),
        )
    return ml, reml


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FixedBinaryResult(metafopy.models.fit_methods.TableFields, metafopy.models.fit_methods.FitMethods, abc.ABC):
    """An intercept-only equal-effects model for 2x2 tables or rate data.

    ``beta``, ``se``, ``zval``, ``pval``, ``ci_lb``, ``ci_ub`` and ``vb`` hold a
    single element (the pooled log odds ratio, log risk ratio, risk difference,
    ...). ``tables`` keeps the cell counts, the ``yi``/``vi`` of the individual
    tables and the bookkeeping that the diagnostics need. ``level`` is the
    significance level ``alpha`` (0.05 for 95% intervals), as in metafor.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    tau2: float
    I2: float
    H2: float
    QE: float
    QEp: float
    k: int
    k_f: int
    k_yi: int
    k_all: int
    k_pos: int | None
    p: int
    parms: int
    measure: str
    level: float
    add: tuple[float, float]
    to: tuple[str, str]
    drop00: tuple[bool, bool]
    fit_stats_ml: metafopy.models.rma_uni.FitStats
    fit_stats_reml: metafopy.models.rma_uni.FitStats
    tables: metafopy.models.tables.TableSet = dataclasses.field(repr=False)
    data: pandas.DataFrame | None = dataclasses.field(default=None, repr=False)
    digits: int = 4
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    coef_names: tuple[str, ...] = ('intrcpt',)
    method: str = 'FE'
    test: str = 'z'
    int_only: bool = True
    weighted: bool = True
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)

    # -- accessors ---------------------------------------------------------

    @property
    def yi(self) -> FloatArray:
        return self.tables.yi

    @property
    def vi(self) -> FloatArray:
        return self.tables.vi

    @property
    def slab(self) -> tuple[str, ...]:
        return self.tables.slab

    def coef(self, type: str = 'beta') -> pandas.Series:
        """Estimated coefficient, indexed by name; ``type`` is accepted for metafor's ``coef(x, type=)`` and only location-scale fits use it."""
        return pandas.Series(self.beta, index=list(self.coef_names), name='estimate')

    @property
    def k_eff(self) -> int:
        """Number of effective observations (the tables used in the fit), metafor's ``k.eff``."""
        return int(self.k)

    @property
    def p_eff(self) -> int:
        """Number of effective model coefficients, metafor's ``p.eff`` (always 1)."""
        return 1

    @property
    def X_f(self) -> FloatArray:
        """The model matrix of the tables after subsetting, a column of ones, metafor's ``X.f``."""
        return numpy.ones((self.k_f, 1))

    @property
    def tau2_f(self) -> float:
        """The amount of heterogeneity, metafor's ``tau2.f`` (always 0 for these equal-effects models)."""
        return 0.0

    @property
    def intercept(self) -> bool:
        """Whether the model has an intercept (always ``True``), metafor's ``intercept``."""
        return True

    @property
    def coef_na(self) -> tuple[bool, ...]:
        """Whether the coefficient could not be estimated (always ``(False,)``), metafor's ``coef.na``."""
        return (False,)

    @property
    def btt(self) -> tuple[int, ...]:
        """The coefficients tested (the only one, 0-based), metafor's ``btt``."""
        return (0,)

    @property
    def m(self) -> int:
        """The number of coefficients tested (1), metafor's ``m``."""
        return 1

    @property
    def ddf(self) -> None:
        """The degrees of freedom of the test (``None``: a z-test), metafor's ``ddf``."""
        return None

    @property
    def dfs(self) -> None:
        """The degrees of freedom of the test (``None``: a z-test), metafor's ``dfs``."""
        return None

    @property
    def ni(self) -> FloatArray:
        """Sample sizes of the tables used in the fit (``ai + bi + ci + di``, or the person-times of rate data), metafor's ``ni``."""
        return self.tables.ni

    def vcov(self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> pandas.DataFrame:
        """Variance-covariance matrix of the pooled estimate, or with ``type='fitted'`` of the fitted values; see :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov_frame(self, type, na_action)

    def fit_stats_for(self, REML: bool | None = None) -> metafopy.models.rma_uni.FitStats:
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def fitted(self, *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """The pooled estimate repeated for each table used in fitting (``'exclude'`` and ``'pass'``: for each table supplied)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        return metafopy.methods.fitted.fitted_values(self, na_action)

    def residuals(self, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def plot(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Draw the forest, funnel, radial and residual plots with :func:`metafopy.plot.plot_rma`."""
        # auditme: accept[in-function-import] metafopy.plot imports the models, so the import is deferred to avoid a cycle
        import metafopy.plot.rma

        return metafopy.plot.rma.plot_rma(self, *args, **kwargs)

    @abc.abstractmethod
    def weight_values(self) -> FloatArray:
        """Un-normalised weights of the tables used in fitting."""

    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    def weights_(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', na_action: metafopy.metadata.NAAction = 'omit') -> FloatArray:
        """Percentage weights of the tables (``type='diagonal'``) or the diagonal weight matrix (metafor's ``weights()``).

        Raises:
            metafopy.errors.InputError: ``type`` or ``na_action`` is not valid or tables are missing and ``na_action='fail'``.
        """
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        if type not in ('diagonal', 'matrix'):
            raise metafopy.errors.InputError("'type' must be 'diagonal' or 'matrix'.")
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        if na_action not in ('omit', 'exclude', 'pass', 'fail'):
            raise metafopy.errors.InputError("Unknown 'na_action' specified.")
        wi = self.weight_values()
        not_na = self.tables.not_na
        if na_action == 'fail' and not not_na.all():
            raise metafopy.errors.InputError('Missing values in weights.')
        if type == 'diagonal':
            full = numpy.full(self.k_f, numpy.nan)
            full[not_na] = wi / wi.sum() * 100
            return full[not_na] if na_action == 'omit' else full
        matrix = numpy.full((self.k_f, self.k_f), numpy.nan)
        matrix[numpy.ix_(not_na, not_na)] = numpy.diag(wi)
        return matrix[numpy.ix_(not_na, not_na)] if na_action == 'omit' else matrix

    def summary(self, digits: int | None = None, showfit: bool = True, *, signif_stars: bool | None = None, signif_legend: bool | None = None) -> str:
        """The printout with the fit statistics table, as metafor's ``summary()``; ``str(res)`` leaves it out like ``print()``.

        ``signif_stars`` and ``signif_legend`` are accepted as in metafor, whose table for these models has no significance stars.
        """
        return metafopy.models.fit_methods.SummaryText(
            metafopy.mfopt.option_space(self.render(metafopy.mfopt.option_digits(digits, self.digits), showfit)),
            self,
            metafopy.mfopt.option_digits(digits, self.digits),
        )

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.measure} k={self.k}>'

    @abc.abstractmethod
    def render(self, digits: int, showfit: bool) -> str:
        """The metafor-style printout."""


def format_header(res: FixedBinaryResult, digits: int, showfit: bool, df: int) -> list[str]:
    """Title, optional fit statistics, I^2/H^2 and the heterogeneity test (shared by the ``rma.mh`` and ``rma.peto`` printouts)."""
    out = ['', f'Equal-Effects Model (k = {res.k})']
    if showfit:
        fs = res.fit_stats_ml
        out.append('')
        out.append(
            metafopy.printing.fmt_named_vector(
                # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
                ['logLik', 'deviance', 'AIC', 'BIC', 'AICc'],
                [metafopy.printing.fmt_num(v, digits) for v in fs.as_tuple()],
            )
        )
    out.append('')
    if not math.isnan(res.I2):
        out.append(f'I^2 (total heterogeneity / total variability):  {res.I2:.2f}%')
    if not math.isnan(res.H2):
        out.append(f'H^2 (total variability / sampling variability): {res.H2:.2f}')
    if not math.isnan(res.QE):
        out.append('')
        out.append('Test for Heterogeneity:')
        out.append(format_test_line('Q', res.QE, res.QEp, df, digits))
    if not (math.isnan(res.I2) and math.isnan(res.H2) and math.isnan(res.QE)):
        out.append('')
    return out


def format_test_line(name: str, value: float, p: float, df: int, digits: int) -> str:
    return f'{name}(df = {df}) = {metafopy.printing.fmt_num(value, digits)}, p-val {metafopy.models.rma_uni.pfmt(p, digits)}'


def result_rows(res: FixedBinaryResult, digits: int, scale: str | None) -> list[str]:
    """The ``Model Results`` tables (log scale and exponentiated scale, or one table for RD/IRD)."""
    beta = float(res.beta[0])
    row = [
        metafopy.printing.fmt_num(beta, digits),
        metafopy.printing.fmt_num(float(res.se[0]), digits),
        metafopy.printing.fmt_num(float(res.zval[0]), digits),
        metafopy.printing.fmt_pval(float(res.pval[0]), digits),
        metafopy.printing.fmt_num(float(res.ci_lb[0]), digits),
        metafopy.printing.fmt_num(float(res.ci_ub[0]), digits),
    ]
    # auditme: accept[literal-duplication] the column headers of metafor's printed tables
    header = ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub']
    if scale is None:
        return ['Model Results:', '', metafopy.printing.fmt_table(header, [row])]
    with numpy.errstate(all='ignore'):
        exp_row = [metafopy.printing.fmt_num(float(numpy.exp(v)), digits) for v in (beta, float(res.ci_lb[0]), float(res.ci_ub[0]))]
    return [
        'Model Results (log scale):',
        '',
        metafopy.printing.fmt_table(header, [row]),
        '',
        f'Model Results ({scale} scale):',
        '',
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        metafopy.printing.fmt_table(['estimate', 'ci.lb', 'ci.ub'], [exp_row]),
    ]


def argument_pairs(
    add: float | collections.abc.Sequence[float],
    to: str | collections.abc.Sequence[str],
    drop00: bool | collections.abc.Sequence[bool],
) -> tuple[tuple[float, float], tuple[metafopy.models.tables.To, metafopy.models.tables.To], tuple[bool, bool]]:
    """Expand and check the ``add``, ``to`` and ``drop00`` arguments of ``rma_mh`` / ``rma_peto``.

    Raises:
        metafopy.errors.InputError: an argument has more than two elements or an unknown value.
    """
    add2 = metafopy.models.tables.pair(add, 0.0, 'add')
    to2 = metafopy.models.tables.pair(to, 'none', 'to')
    drop2 = metafopy.models.tables.pair(drop00, False, 'drop00')
    metafopy.models.tables.check_to(to2)
    return (float(add2[0]), float(add2[1])), typing.cast(tuple[metafopy.models.tables.To, metafopy.models.tables.To], to2), (bool(drop2[0]), bool(drop2[1]))
