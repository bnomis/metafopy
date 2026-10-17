"""The result of ``rma_glmm`` (metafor's ``rma.glmm`` object) and its printed form (``print.rma.glmm``).

See https://wviechtb.github.io/metafor/reference/rma.glmm.html and
https://github.com/wviechtb/metafor/blob/master/R/print.rma.glmm.r.
"""

import dataclasses
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.labelled
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.rma_uni
import metafopy.models.tables
import metafopy.printing

__all__ = ['GLMMResult']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
MODEL_TYPES = {
    'UM.FS': 'Unconditional Model with Fixed Study Effects',
    'UM.RS': 'Unconditional Model with Random Study Effects',
    'CM.AL': 'Conditional Model with Approximate Likelihood',
    'CM.EL': 'Conditional Model with Exact Likelihood',
}
# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
FE_TITLES = {'FE': 'Fixed-Effects Model', 'EE': 'Equal-Effects Model', 'CE': 'Common-Effects Model'}
# auditme: accept option and column lists repeat as metafor repeats them (literal-duplication, non-alphabetical-literal)
FIXED_METHODS = ('FE', 'EE', 'CE')


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class GLMMResult(metafopy.models.fit_methods.TableFields, metafopy.models.fit_methods.FitMethods):
    """A fitted ``rma.glmm`` model: a generalized linear mixed-effects model for binary or count data.

    ``beta``, ``se``, ``zval``, ``pval``, ``ci_lb``, ``ci_ub`` and ``vb`` hold one element per coefficient
    (``coef_names``). ``QE_Wld`` and ``QE_LRT`` are the Wald-type and likelihood ratio tests for (residual)
    heterogeneity, ``QM`` the omnibus test of the coefficients ``btt`` (0-based). ``level`` is the
    significance level ``alpha`` (0.05 for 95% intervals), as in metafor. ``tables`` keeps the counts
    and the ``yi``/``vi`` used for ``I2``/``H2``.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    tau2: float
    se_tau2: float
    sigma2: float
    rho: float
    ci_lb_tau2: float
    ci_ub_tau2: float
    I2: float
    H2: float
    vt: float
    QE_Wld: float
    QEp_Wld: float
    QE_LRT: float
    QEp_LRT: float
    QE_df: int | None
    QM: float
    QMdf: tuple[int, int | None]
    QMp: float
    k: int
    k_f: int
    k_yi: int
    k_eff: int
    k_all: int
    p: int
    p_eff: int
    parms: int
    int_only: bool
    int_incl: bool
    intercept: bool
    X: FloatArray
    X_f: FloatArray
    coef_names: tuple[str, ...]
    tables: metafopy.models.tables.TableSet = dataclasses.field(repr=False)
    measure: str
    method: str
    model: str
    weighted: bool
    test: str
    ddf: int | None
    btt: tuple[int, ...]
    m: int
    digits: int
    level: float
    add: float
    to: str
    drop00: bool
    fit_stats_ml: metafopy.models.rma_uni.FitStats
    fit_stats_reml: metafopy.models.rma_uni.FitStats
    se_warn: bool
    data: pandas.DataFrame | None = dataclasses.field(default=None, repr=False)
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)
    control: typing.Any = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)

    outdat_names: typing.ClassVar[tuple[str, ...]] = (*metafopy.models.fit_methods.COUNT_NAMES, 'xi', 'mi', 'ni', 'ti')

    def __post_init__(self) -> None:
        # R names the rows of the model matrices by the rows of the data and their columns by the coefficients
        names = [str(int(i) + 1) for i in self.tables.ids]
        used = [n for n, flag in zip(names, self.tables.not_na, strict=True) if flag]
        object.__setattr__(self, 'X', metafopy.core.labelled.design_matrix(self.X, used, self.coef_names))
        object.__setattr__(self, 'X_f', metafopy.core.labelled.design_matrix(self.X_f, names, self.coef_names))

    # -- accessors ---------------------------------------------------------

    @property
    def outdat_f(self) -> dict[str, FloatArray | None]:
        """The counts of the tables after subsetting; ``ni`` holds the sample sizes of those tables, as in metafor's ``outdat.f``.

        See ``rma.glmm`` in <https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r>.
        """
        cells = self.tables.cells_f
        out: dict[str, FloatArray | None] = {name: cells.get(name) for name in self.outdat_names}
        out['ni'] = self.tables.ni_f
        return out

    @property
    def dfs(self) -> int | None:
        """The degrees of freedom of the tests of the coefficients (``None`` for the z-test), metafor's ``dfs``."""
        return self.ddf

    @property
    def ni(self) -> FloatArray:
        """Sample sizes of the tables used in the fit, metafor's ``ni``."""
        return self.tables.ni

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
        """Estimated coefficients, indexed by name; ``type`` is accepted for metafor's ``coef(x, type=)`` and only location-scale fits use it."""
        return pandas.Series(self.beta, index=list(self.coef_names), name='estimate')

    def vcov(self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> pandas.DataFrame:
        """Variance-covariance matrix of the coefficients, or with ``type='fitted'`` of the fitted values; see :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov_frame(self, type, na_action)

    def fitted(self, *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Fitted values ``X beta`` of the tables used in fitting (metafor's ``fitted.rma``; ``na_action`` as for ``rma_uni`` fits)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        return metafopy.methods.fitted.fitted_values(self, na_action)

    def weights_(self, type: str = 'diagonal', na_action: str = 'omit') -> FloatArray:
        """Not available (metafor's ``weights()`` stops for this class).

        Raises:
            metafopy.errors.NotSupportedError: always.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.weights

        metafopy.methods.weights.weights_unsupported(self)

    def residuals(self, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def fit_stats_for(self, REML: bool | None = None) -> metafopy.models.rma_uni.FitStats:
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def summary(self, digits: int | None = None, showfit: bool = True, *, signif_stars: bool | None = None, signif_legend: bool | None = None) -> str:
        """The printout with the fit statistics table, as metafor's ``summary()``; ``str(res)`` leaves it out like ``print()``.

        ``signif_stars`` (default: the ``signif_stars`` option, ``True``) adds the significance stars, ``signif_legend`` (default: ``signif_stars``)
        their legend.
        """
        stars, legend = metafopy.mfopt.option_signif(signif_stars, signif_legend)
        return metafopy.models.fit_methods.SummaryText(
            metafopy.mfopt.option_space(self.render(metafopy.mfopt.option_digits(digits, self.digits), showfit, stars, legend)),
            self,
            metafopy.mfopt.option_digits(digits, self.digits),
        )

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<GLMMResult {self.measure} {self.model} {self.method} k={self.k}>'

    # -- printing ----------------------------------------------------------

    def title_lines(self, digits: int, showfit: bool) -> list[str]:
        """Title, model type and optional fit statistics (``print.rma.glmm``)."""
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        out = ['']
        if self.method in FIXED_METHODS:
            title = FE_TITLES[self.method] if self.int_only else 'Fixed-Effects with Moderators Model'
            out.append(f'{title} (k = {self.k})')
        else:
            title = 'Random-Effects Model' if self.int_only else 'Mixed-Effects Model'
            out.append(f'{title} (k = {self.k}; tau^2 estimator: {self.method})')
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        if self.measure in ('OR', 'IRR'):
            out.append(f'Model Type: {MODEL_TYPES[self.model]}')
        out.append('')
        if showfit:
            out.append(
                metafopy.printing.fmt_named_vector(
                    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
                    ['logLik', 'deviance', 'AIC', 'BIC', 'AICc'],
                    [metafopy.printing.fmt_num(v, digits) for v in self.fit_stats_ml.as_tuple()],
                )
            )
            out.append('')
        return out

    def heterogeneity_lines(self, digits: int) -> list[str]:
        out: list[str] = []
        if self.method not in FIXED_METHODS:
            tau2 = metafopy.printing.zero_below(self.tau2, digits)
            if not math.isnan(self.se_tau2):
                tau2 += f' (SE = {metafopy.printing.fmt_num(self.se_tau2, digits)})'
            if self.int_only:
                rows = [
                    ('tau^2 (estimated amount of total heterogeneity):', tau2),
                    ('tau (square root of estimated tau^2 value):', metafopy.printing.zero_below(math.sqrt(self.tau2), digits)),
                    ('I^2 (total heterogeneity / total variability):', f'{self.I2:.2f}%'),
                    ('H^2 (total variability / sampling variability):', f'{self.H2:.2f}'),
                ]
            else:
                rows = [
                    ('tau^2 (estimated amount of residual heterogeneity):', tau2),
                    ('tau (square root of estimated tau^2 value):', metafopy.printing.zero_below(math.sqrt(self.tau2), digits)),
                    ('I^2 (residual heterogeneity / unaccounted variability):', f'{self.I2:.2f}%'),
                    ('H^2 (unaccounted variability / sampling variability):', f'{self.H2:.2f}'),
                ]
            width = max(len(label) for label, _ in rows)
            out += [f'{label.ljust(width)} {value}' for label, value in rows]
            out.append('')
        if not math.isnan(self.sigma2):
            out.append(f'sigma^2 (estimated amount of study level variability): {metafopy.printing.zero_below(self.sigma2, digits)}')
            out.append(f'sigma (square root of estimated sigma^2 value):        {metafopy.printing.zero_below(math.sqrt(self.sigma2), digits)}')
            out.append('')
        if not (math.isnan(self.QE_Wld) and math.isnan(self.QE_LRT)):
            wld, lrt = metafopy.printing.fmt_num(self.QE_Wld, digits), metafopy.printing.fmt_num(self.QE_LRT, digits)
            width = max(len(wld), len(lrt))
            df = self.QE_df if self.QE_df is not None else 0
            out.append('Tests for Heterogeneity:' if self.int_only else 'Tests for Residual Heterogeneity:')
            out.append(f'Wld(df = {df}) = {wld.rjust(width)}, p-val {metafopy.models.rma_uni.pfmt(self.QEp_Wld, digits)}')
            out.append(f'LRT(df = {df}) = {lrt.rjust(width)}, p-val {metafopy.models.rma_uni.pfmt(self.QEp_LRT, digits)}')
            out.append('')
        return out

    def render(self, digits: int, showfit: bool, stars: bool = True, legend: bool = True) -> str:
        out = self.title_lines(digits, showfit)
        out += self.heterogeneity_lines(digits)
        if self.p > 1 and not math.isnan(self.QM):
            coefs = metafopy.models.rma_uni.coef_span(self.btt)
            out.append(f'Test of Moderators (coefficient{"s" if self.m > 1 else ""} {coefs}):')
            qm = metafopy.printing.fmt_num(self.QM, digits)
            pv = metafopy.models.rma_uni.pfmt(self.QMp, digits)
            if self.test == 't':
                out.append(f'F(df1 = {self.QMdf[0]}, df2 = {self.QMdf[1]}) = {qm}, p-val {pv}')
            else:
                out.append(f'QM(df = {self.QMdf[0]}) = {qm}, p-val {pv}')
            out.append('')
        out.append('Model Results:')
        out.append('')
        stat = 'tval' if self.test == 't' else 'zval'
        header = ['estimate', 'se', stat]
        if self.test == 't':
            header.append('df')
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        header += ['pval', 'ci.lb', 'ci.ub', '']
        rows: list[list[str]] = []
        for j in range(self.p):
            row = [
                metafopy.printing.fmt_num(float(self.beta[j]), digits),
                metafopy.printing.fmt_num(float(self.se[j]), digits),
                metafopy.printing.fmt_num(float(self.zval[j]), digits),
            ]
            if self.test == 't':
                row.append(str(self.ddf))
            row += [
                metafopy.printing.fmt_pval(float(self.pval[j]), digits),
                metafopy.printing.fmt_num(float(self.ci_lb[j]), digits),
                metafopy.printing.fmt_num(float(self.ci_ub[j]), digits),
                metafopy.printing.signif_stars(float(self.pval[j])),
            ]
            rows.append(row)
        header, rows = metafopy.printing.drop_stars(header, rows, stars)
        out.append(
            metafopy.printing.fmt_table(header, rows, row_names=None if self.int_only else list(self.coef_names), width=metafopy.printing.console_width())
        )
        out += metafopy.printing.signif_footer(legend)
        return '\n'.join(out)
