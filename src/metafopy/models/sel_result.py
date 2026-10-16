"""The fitted selection model (:class:`SelmodelResult`) and its metafor-style printout."""

import dataclasses
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.rma_uni
import metafopy.models.sel_spec
import metafopy.models.tau2
import metafopy.printing

__all__ = ['SelmodelResult', 'format_selmodel']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

# auditme: accept[literal-duplication] the columns of metafor's coefficient tables
TABLE_HEADER = ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub', '']  # auditme: accept[non-alphabetical-literal] metafor's column order
NA_FIT = metafopy.models.rma_uni.FitStats(ll=math.nan, dev=math.nan, AIC=math.nan, BIC=math.nan, AICc=math.nan)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class SelmodelResult(metafopy.models.fit_methods.FitMethods):
    """A selection model fitted by :func:`metafopy.selmodel` (metafor's ``rma.uni.selmodel`` object).

    ``base`` is the ML (or equal/fixed-effects) :class:`~metafopy.RMAResult` the selection model was fitted from; the
    data (``yi``, ``vi``, ``X``, ...) come from it. All other names follow metafor with ``.`` replaced by ``_``
    (``se_delta``, ``ci_lb_delta``, ``LRT_tau2``). ``beta``, ``se``, ``vb`` and ``tau2`` are the estimates under the
    selection model; ``delta`` holds the selection parameters, ``vd`` their covariance matrix.
    """

    base: metafopy.models.rma_uni.RMAResult
    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    betaspec: bool
    tau2: float
    se_tau2: float
    ci_lb_tau2: float
    ci_ub_tau2: float
    tau2_fix: bool
    QM: float
    QMp: float
    delta: FloatArray
    vd: FloatArray
    se_delta: FloatArray
    zval_delta: FloatArray
    pval_delta: FloatArray
    ci_lb_delta: FloatArray
    ci_ub_delta: FloatArray
    deltas: int
    delta_fix: tuple[bool, ...]
    hessian: FloatArray | None
    hest: tuple[bool, ...]
    ll: float
    ll0: float
    LRT: float
    LRTdf: int
    LRTp: float
    LRT_tau2: float
    LRTp_tau2: float
    parms: int
    fit_stats_ml: metafopy.models.rma_uni.FitStats
    fit_stats_reml: metafopy.models.rma_uni.FitStats
    pvals: FloatArray
    spec: metafopy.models.sel_spec.SelSpec
    type: str
    steps: FloatArray
    decreasing: bool
    stepsspec: bool
    pgrp: numpy.typing.NDArray[numpy.intp]
    ptable: pandas.DataFrame | None
    subset: BoolArray
    k0: int
    k1: int
    alternative: str
    pval_min: float
    prec: str | None
    precspec: bool
    precis: dict[str, float]
    scaleprec: bool | None
    delta_lb: FloatArray
    delta_ub: FloatArray
    delta_lb_excl: tuple[bool, ...]
    delta_ub_excl: tuple[bool, ...]
    delta_min: FloatArray
    delta_max: FloatArray
    tau2_max: float
    control: dict[str, typing.Any]
    defmap: bool | None
    mapfun: tuple[str | None, ...] | None
    mapinvfun: tuple[str | None, ...] | None
    digits: int = 4
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)

    # -- data of the underlying model --------------------------------------------

    @property
    def k_all(self) -> int:
        """Number of rows supplied, including those dropped for missing values."""
        return self.base.k_all

    @property
    def k_eff(self) -> int:
        """Number of effective observations."""
        return self.base.k_eff

    @property
    def p_eff(self) -> int:
        """Number of effective model coefficients."""
        return self.base.p_eff

    @property
    def ni(self) -> FloatArray | None:
        """Sample sizes of the studies used in the fit (``None`` when unknown)."""
        return self.base.ni

    @property
    def M(self) -> FloatArray:
        """The marginal variance-covariance matrix of the observed effects, ``diag(vi + tau2)`` with the ``tau2`` of the selection model."""
        return numpy.diag(self.base.vi + self.tau2)

    @property
    def I2(self) -> float:
        """Not computed for selection models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def H2(self) -> float:
        """Not computed for selection models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def vt(self) -> float:
        """Not computed for selection models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def QE(self) -> float:
        """Not computed for selection models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def QEp(self) -> float:
        """Not computed for selection models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def yi(self) -> FloatArray:
        return self.base.yi

    @property
    def vi(self) -> FloatArray:
        return self.base.vi

    @property
    def X(self) -> FloatArray:
        return self.base.X

    @property
    def k(self) -> int:
        return self.base.k

    @property
    def p(self) -> int:
        return self.base.p

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def btt(self) -> tuple[int, ...]:
        return self.base.btt

    @property
    def method(self) -> str:
        return self.base.method

    @property
    def int_only(self) -> bool:
        return self.base.int_only

    @property
    def int_incl(self) -> bool:
        return self.base.int_incl

    @property
    def coef_names(self) -> tuple[str, ...]:
        return self.base.coef_names

    @property
    def slab(self) -> tuple[str, ...]:
        return self.base.slab

    @property
    def level(self) -> float:
        return self.base.level

    @property
    def test(self) -> str:
        return 'z'

    @property
    def ddf(self) -> None:
        return None

    @property
    def delta_names(self) -> tuple[str, ...]:
        return ('delta',) if self.deltas == 1 else tuple(f'delta.{j + 1}' for j in range(self.deltas))

    # -- accessors ----------------------------------------------------------------

    @typing.overload
    def coef(self, type: typing.Literal['beta', 'delta', 'all']) -> pandas.Series: ...

    @typing.overload
    def coef(self, type: None = None) -> dict[str, pandas.Series]: ...

    def coef(self, type: typing.Literal['beta', 'delta', 'all'] | None = None) -> pandas.Series | dict[str, pandas.Series]:
        """The coefficients and selection parameters as ``{'beta': ..., 'delta': ...}`` (as R's ``coef()``), or one of them with ``type``.

        ``type='beta'`` or ``'delta'`` gives that part as a series and ``'all'`` both concatenated into one series.
        """
        beta = pandas.Series(self.beta, index=list(self.coef_names), name='estimate')
        delta = pandas.Series(self.delta, index=list(self.delta_names), name='estimate')
        if type == 'beta':
            return beta
        if type == 'delta':
            return delta
        if type == 'all':
            return pandas.concat([beta, delta])
        return {'beta': beta, 'delta': delta}

    @typing.overload
    def vcov(
        self,
        type: typing.Literal['beta', 'alpha', 'delta', 'obs', 'marginal', 'fitted', 'resid'],
        *,
        na_action: metafopy.models.rma_uni.NAActionAll = 'omit',
    ) -> pandas.DataFrame: ...

    @typing.overload
    def vcov(
        self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit'
    ) -> pandas.DataFrame | dict[str, pandas.DataFrame]: ...

    def vcov(
        self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit'
    ) -> pandas.DataFrame | dict[str, pandas.DataFrame]:
        """Variance-covariance matrices; ``'fixed'`` (the default) gives ``{'beta': ..., 'delta': ...}``, the other types follow :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov(self, type, na_action=na_action)

    def fitted(self, *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Fitted values ``X beta`` (``na_action`` is accepted for symmetry: the selection model was fitted to the studies without missing values)."""
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

    def fit_stats(self, REML: bool | None = None) -> metafopy.models.rma_uni.FitStats:
        return self.fit_stats_reml if REML else self.fit_stats_ml

    # -- delegated methods (defined in metafopy.methods.selmodel) ------------------

    def predict(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.selmodel

        return metafopy.methods.selmodel.predict(self, *args, **kwargs)

    def confint(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.selmodel

        return metafopy.methods.selmodel.confint(self, *args, **kwargs)

    def profile(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.selmodel

        return metafopy.methods.selmodel.profile(self, *args, **kwargs)

    def anova(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.selmodel

        return metafopy.methods.selmodel.anova(self, *args, **kwargs)

    def plot(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Draw the selection function with :func:`metafopy.plot.plot_selmodel`."""
        # auditme: accept[in-function-import] metafopy.plot imports the methods, so the import is deferred to avoid a cycle
        import metafopy.plot.selmodel

        return metafopy.plot.selmodel.plot_selmodel(self, *args, **kwargs)

    def vif(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vif

        return metafopy.methods.vif.vif(self, *args, **kwargs)

    # -- printing -------------------------------------------------------------------

    def summary(self, digits: int | None = None, showfit: bool = True) -> str:
        """The printout with the fit statistics table, as metafor's ``summary()``; ``str(res)`` leaves it out like ``print()``."""
        return metafopy.mfopt.option_space(format_selmodel(self, metafopy.mfopt.option_digits(digits, self.digits), showfit))

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<SelmodelResult {self.type} {self.method} k={self.k} p={self.p}>'


def delta_table(res: SelmodelResult, digits: int) -> tuple[list[str], list[list[str]], list[str] | None]:
    """Header, rows and row names of the table of selection parameters."""
    stepwise = res.type in metafopy.models.sel_spec.STEPWISE
    header = ['k'] if stepwise else []
    header += TABLE_HEADER
    rows: list[list[str]] = []
    for j in range(res.deltas):
        na = '---' if res.delta_fix[j] else 'NA'
        cells = [
            metafopy.printing.fmt_num(res.delta[j], digits),
            metafopy.printing.fmt_num(res.se_delta[j], digits),
            metafopy.printing.fmt_num(res.zval_delta[j], digits),
            metafopy.printing.fmt_pval(res.pval_delta[j], digits),
            metafopy.printing.fmt_num(res.ci_lb_delta[j], digits),
            metafopy.printing.fmt_num(res.ci_ub_delta[j], digits),
        ]
        cells = [na if c == 'NA' else c for c in cells]
        cells.append(metafopy.printing.signif_stars(res.pval_delta[j]))
        if stepwise and res.ptable is not None:
            cells.insert(0, str(int(res.ptable['k'].iloc[j])))
        rows.append(cells)
    if stepwise and res.ptable is not None:
        names = [str(n) for n in res.ptable.index]
    elif res.deltas == 1:
        names = None
    else:
        names = list(res.delta_names)
    return header, rows, names


def format_selmodel(res: SelmodelResult, digits: int = 4, showfit: bool = False) -> str:
    """The output of metafor's ``print.rma.uni`` for a selection model."""
    fmt = metafopy.printing.fmt_num
    pfmt = metafopy.models.rma_uni.pfmt
    # auditme: accept[literal-duplication] print.rma starts with a blank line
    out: list[str] = ['']
    fe = res.method in metafopy.models.tau2.FIXED_EFFECTS
    if fe:
        if res.int_only:
            kind = {'EE': 'Equal-Effects', 'FE': 'Fixed-Effects', 'CE': 'Common-Effects'}[res.method]
            out.append(f'{kind} Model (k = {res.k})')
        else:
            out.append(f'Fixed-Effects with Moderators Model (k = {res.k})')
    else:
        kind = 'Random-Effects' if res.int_only else 'Mixed-Effects'
        how = 'user-specified tau^2 value' if res.tau2_fix else f'tau^2 estimator: {res.method}'
        out.append(f'{kind} Model (k = {res.k}; {how})')
    out.append('')
    if showfit:
        out.append(
            metafopy.printing.fmt_named_vector(
                # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
                ['logLik', 'deviance', 'AIC', 'BIC', 'AICc'],
                [metafopy.printing.fmt_num(v, digits) for v in res.fit_stats_ml.as_tuple()],
            )
        )
        out.append('')

    if not fe:
        het = 'total' if res.int_only else 'residual'
        what = 'specified' if res.tau2_fix else 'estimated'
        tau2_line = metafopy.printing.zero_below(res.tau2, digits)
        if not math.isnan(res.se_tau2):
            tau2_line += f' (SE = {fmt(res.se_tau2, digits)})'
        rows = [
            (f'tau^2 ({what} amount of {het} heterogeneity):', tau2_line),
            (f'tau (square root of {what} tau^2 value):', metafopy.printing.zero_below(math.sqrt(res.tau2), digits)),
        ]
        width = max(len(r[0]) for r in rows)
        out += [f'{label.ljust(width)} {value}' for label, value in rows]
        out.append('')

    if not math.isnan(res.LRT_tau2):
        out.append('Test for Heterogeneity:' if res.int_only else 'Test for Residual Heterogeneity:')
        out.append(f'LRT(df = 1) = {fmt(res.LRT_tau2, digits)}, p-val {pfmt(res.LRTp_tau2, digits)}')
        out.append('')

    if res.p > 1 and not math.isnan(res.QM):
        coefs = metafopy.models.rma_uni.coef_span(res.btt)
        out.append(f'Test of Moderators (coefficient{"s" if res.m > 1 else ""} {coefs}):')
        out.append(f'QM(df = {res.m}) = {fmt(res.QM, digits)}, p-val {pfmt(res.QMp, digits)}')
        out.append('')

    out.append('Model Results:')
    out.append('')
    header = TABLE_HEADER
    rows_: list[list[str]] = []
    for j in range(res.p):
        rows_.append(
            [
                fmt(res.beta[j], digits),
                fmt(res.se[j], digits),
                fmt(res.zval[j], digits),
                metafopy.printing.fmt_pval(res.pval[j], digits),
                fmt(res.ci_lb[j], digits),
                fmt(res.ci_ub[j], digits),
                metafopy.printing.signif_stars(res.pval[j]),
            ]
        )
    out.append(metafopy.printing.fmt_table(header, rows_, row_names=None if res.int_only else list(res.coef_names)))

    if not math.isnan(res.LRT):
        out.append('')
        out.append('Test for Selection Model Parameters:')
        out.append(f'LRT(df = {res.LRTdf}) = {fmt(res.LRT, digits)}, p-val {pfmt(res.LRTp, digits)}')

    out.append('')
    out.append('Selection Model Results:' if res.k == res.k1 else f'Selection Model Results (k_subset = {res.k1}):')
    out.append('')
    dheader, drows, dnames = delta_table(res, digits)
    out.append(metafopy.printing.fmt_table(dheader, drows, row_names=dnames))
    out.append('')
    out.append('---')
    out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    out.append('')
    return '\n'.join(out)
