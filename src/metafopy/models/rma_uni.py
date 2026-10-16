"""Equal-, fixed- and random/mixed-effects meta-analysis models (metafor's ``rma.uni``).

The model is ``y_i = x_i' beta + u_i + e_i`` with ``u_i ~ N(0, tau^2)`` and
``e_i ~ N(0, v_i)`` for known sampling variances ``v_i``. ``tau^2`` is
estimated by one of the estimators in :mod:`metafopy.models.tau2`; ``beta``
by weighted least squares given ``tau^2``. Inference follows Viechtbauer
(2010), doi:10.18637/jss.v036.i03, including the Knapp & Hartung (2003)
adjustment (doi:10.1002/sim.1482).
"""

import collections.abc
import dataclasses
import math
import re
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.linalg
import metafopy.core.verbose
import metafopy.errors
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.robust_info
import metafopy.models.robust_print
import metafopy.models.tau2
import metafopy.printing

if typing.TYPE_CHECKING:
    import metafopy.models.rma_ls
    import metafopy.models.rma_ls_result

__all__ = ['FitStats', 'RMAResult', 'rma', 'rma_uni']

type FloatArray = numpy.typing.NDArray[numpy.float64]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type Test = typing.Literal['z', 't', 'knha', 'adhoc']
type Btt = int | collections.abc.Sequence[int] | str | None
type ResidualType = typing.Literal['response', 'rstandard', 'rstudent', 'pearson', 'cholesky']
type NAActionAll = typing.Literal['omit', 'exclude', 'pass', 'fail']
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
type VcovType = typing.Literal['fixed', 'beta', 'alpha', 'delta', 'obs', 'marginal', 'fitted', 'resid']

DBL_EPS = numpy.finfo(float).eps


@dataclasses.dataclass(frozen=True, slots=True)
class FitStats:
    """Log-likelihood, deviance and information criteria (metafor's ``fit.stats``)."""

    ll: float
    dev: float
    AIC: float
    BIC: float
    AICc: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.ll, self.dev, self.AIC, self.BIC, self.AICc)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RMAResult(metafopy.models.fit_methods.DataFields, metafopy.models.fit_methods.FitMethods):
    """A fitted ``rma.uni`` model.

    Array attributes follow metafor's names with ``.`` replaced by ``_``
    (``ci_lb``, ``se_tau2``). ``btt`` holds 0-based coefficient indices.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    coef_names: tuple[str, ...]
    tau2: float
    se_tau2: float
    k: int
    p: int
    m: int
    btt: tuple[int, ...]
    QE: float
    QEp: float
    QM: float
    QMp: float
    QMdf: tuple[int, float | None]
    I2: float
    H2: float
    R2: float | None
    vt: float
    fit_stats_ml: FitStats
    fit_stats_reml: FitStats
    parms: int
    method: str
    test: Test
    level: float
    ddf: int | None
    s2w: float
    weighted: bool
    weights: FloatArray | None
    tau2_fix: bool
    int_only: bool
    int_incl: bool
    allvipos: bool
    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    slab: tuple[str, ...]
    data: metafopy.metadata.MetaData = dataclasses.field(repr=False)
    control: metafopy.models.tau2.Control = dataclasses.field(repr=False)
    iterations: int = 0
    measure: str = 'GEN'
    robust: metafopy.models.robust_info.RobustInfo | None = None
    digits: int = 4
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)

    # -- accessors ---------------------------------------------------------

    @property
    def dfs(self) -> float | tuple[float, ...] | None:
        """The degrees of freedom of a model made robust with :func:`metafopy.robust` (metafor's ``dfs``), ``None`` otherwise.

        A single number (the number of clusters minus ``p``) for the default method and one value per coefficient for ``club_sandwich=True``,
        for which ``ddf`` stays ``None`` because it holds a single number.
        """
        if self.robust is None:
            return None
        if self.robust.robumethod == 'clubSandwich':
            return tuple(float(v) for v in self.robust.dfs)
        return float(self.robust.dfs[0])

    def coef(self) -> pandas.Series:
        """Estimated coefficients, indexed by name."""
        return pandas.Series(self.beta, index=list(self.coef_names), name='estimate')

    def vcov(self, type: VcovType = 'fixed', *, na_action: NAActionAll = 'omit') -> pandas.DataFrame:
        """Variance-covariance matrix of the coefficients, or with ``type`` of the observed effects; see :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov_frame(self, type, na_action)

    def fitted(self, *, na_action: NAActionAll = 'omit') -> FloatArray:
        """Fitted values ``X beta``.

        ``na_action`` is ``'omit'`` (the studies used in fitting), ``'exclude'``, ``'pass'`` or ``'fail'``, as in :func:`metafopy.residuals`.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        return metafopy.methods.fitted.fitted_values(self, na_action)

    def residuals(self, type: ResidualType = 'response', *, na_action: NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def fit_stats(self, REML: bool | None) -> FitStats:
        if REML is None:
            REML = self.method == 'REML'
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def weights_(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', na_action: NAActionAll = 'omit') -> FloatArray:
        """Percentage weights of the estimates in the model fit (``type='diagonal'``) or the weight matrix (``'matrix'``), metafor's ``weights()``.

        ``na_action`` is ``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'`` as in :func:`metafopy.residuals`.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.weights

        return metafopy.methods.weights.weights_uni(self, type, na_action)

    def hat_matrix(self) -> FloatArray:
        """The full hat matrix ``X (X'WX)^-1 X'W``."""
        X = self.X
        if self.weighted:
            w = 1.0 / (self.vi + self.tau2) if self.weights is None else self.weights
            return X @ metafopy.core.linalg.invcalc(X, w) @ (X.T * w)
        return X @ metafopy.core.linalg.invcalc(X, numpy.ones(self.k)) @ X.T

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    def hatvalues(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', *, na_action: NAActionAll = 'omit') -> FloatArray:
        """Diagonal of the hat matrix (``type='diagonal'``) or the hat matrix itself (``'matrix'``), as metafor's ``hatvalues()``.

        With ``na_action='exclude'`` or ``'pass'`` the studies left out because of missing values are NaN rows (and columns); ``'fail'`` stops if any were.

        Raises:
            metafopy.errors.InputError: ``type`` is not ``'diagonal'`` or ``'matrix'``, ``na_action`` is unknown, or studies are missing with ``'fail'``.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if type not in ('diagonal', 'matrix'):
            raise metafopy.errors.InputError("type must be 'diagonal' or 'matrix'")
        H = self.hat_matrix()
        if type == 'matrix':
            return metafopy.methods.fitted.hat_values(self, H, na_action)
        h = numpy.diag(H).copy()
        h[h > 1 - 10 * DBL_EPS] = 1.0
        return metafopy.methods.fitted.hat_values(self, h, na_action)

    # -- delegated methods (defined in metafopy.methods) ----------------------

    def predict(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.predict

        return metafopy.methods.predict.predict(self, *args, **kwargs)

    def confint(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.confint

        return metafopy.methods.confint.confint(self, *args, **kwargs)

    def plot(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Draw the forest, funnel, radial and residual plots with :func:`metafopy.plot.plot_rma`."""
        # auditme: accept[in-function-import] metafopy.plot imports the models, so the import is deferred to avoid a cycle
        import metafopy.plot.rma

        return metafopy.plot.rma.plot_rma(self, *args, **kwargs)

    def anova(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.anova

        return metafopy.methods.anova.anova(self, *args, **kwargs)

    def blup(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.blup

        return metafopy.methods.blup.blup(self, *args, **kwargs)

    def ranef(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.ranef

        return metafopy.methods.ranef.ranef(self, *args, **kwargs)

    def vif(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vif

        return metafopy.methods.vif.vif(self, *args, **kwargs)

    # -- printing ----------------------------------------------------------

    def summary(
        self,
        digits: int | collections.abc.Mapping[str, int] | None = None,
        showfit: bool = True,
        *,
        signif_stars: bool | None = None,
        signif_legend: bool | None = None,
    ) -> str:
        """The printout with the fit statistics table, as metafor's ``summary()``; ``str(res)`` leaves it out like ``print()``.

        ``digits`` defaults to the ``digits`` option, else to the ``digits`` the model was fitted with. ``signif_stars`` (default: the
        ``signif_stars`` option, ``True``) adds the significance stars and ``signif_legend`` (default: ``signif_stars``) the legend of the codes.
        """
        stars, legend = metafopy.mfopt.option_signif(signif_stars, signif_legend)
        return metafopy.mfopt.option_space(format_result(self, metafopy.mfopt.option_digits(digits, self.digits), showfit, stars, legend))

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<RMAResult {self.method} k={self.k} p={self.p}>'


def make_control(control: metafopy.models.tau2.Control | collections.abc.Mapping[str, typing.Any] | None) -> metafopy.models.tau2.Control:
    if control is None:
        return metafopy.models.tau2.Control()
    if isinstance(control, metafopy.models.tau2.Control):
        return control
    names = {f.name for f in dataclasses.fields(metafopy.models.tau2.Control)}
    kwargs: dict[str, typing.Any] = {}
    for key, value in control.items():
        name = key.replace('.', '_')
        if name not in names:
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        kwargs[name] = value
    return metafopy.models.tau2.Control(**kwargs)


def set_btt(btt: Btt, p: int, int_incl: bool, names: collections.abc.Sequence[str], fixed: bool = False) -> tuple[int, ...]:
    if btt is None:
        if p > 1:
            return tuple(range(1, p)) if int_incl else tuple(range(p))
        return (0,)
    if isinstance(btt, str):
        found = tuple(i for i, n in enumerate(names) if (btt in n if fixed else re.search(btt, n)))
        if not found:
            raise metafopy.errors.InputError("cannot identify coefficient(s) matching the 'btt' pattern")
        return found
    idx = [btt] if isinstance(btt, int) else list(btt)
    out = sorted({int(i) for i in idx})
    if any(i < 0 or i >= p for i in out):
        raise metafopy.errors.InputError(f"'btt' indices must lie in 0..{p - 1}")
    return tuple(out)


def level_to_alpha(level: float) -> float:
    """metafor's ``.level()``: 95 (or 0.95) -> 0.05."""
    return 1 - level / 100 if level > 1 else 1 - level


def mad(x: FloatArray) -> float:
    med = float(numpy.median(x))
    return 1.4826 * float(numpy.median(numpy.abs(x - med)))


# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
ESCALC_ARGS = (
    'ai',
    'bi',
    'ci',
    'di',
    'n1i',
    'n2i',
    'x1i',
    'x2i',
    't1i',
    't2i',
    'm1i',
    'm2i',
    'sd1i',
    'sd2i',
    'xi',
    'mi',
    'ri',
    'ti',
    'fi',
    'pi',
    'sdi',
    'r2i',
    'mini',
    'maxi',
)


@typing.overload
def rma_uni(
    yi: metafopy.metadata.Column | pandas.DataFrame | None = None,
    vi: metafopy.metadata.Column | None = None,
    *,
    sei: metafopy.metadata.Column | None = None,
    weights: metafopy.metadata.Column | None = None,
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    m1i: metafopy.metadata.Column | None = None,
    m2i: metafopy.metadata.Column | None = None,
    sd1i: metafopy.metadata.Column | None = None,
    sd2i: metafopy.metadata.Column | None = None,
    xi: metafopy.metadata.Column | None = None,
    mi: metafopy.metadata.Column | None = None,
    ri: metafopy.metadata.Column | None = None,
    ti: metafopy.metadata.Column | None = None,
    fi: metafopy.metadata.Column | None = None,
    pi: metafopy.metadata.Column | None = None,
    sdi: metafopy.metadata.Column | None = None,
    r2i: metafopy.metadata.Column | None = None,
    mini: metafopy.metadata.Column | None = None,
    maxi: metafopy.metadata.Column | None = None,
    ni: metafopy.metadata.Column | None = None,
    mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None = None,
    measure: str = 'GEN',
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    to: typing.Literal['all', 'only0', 'if0all', 'none'] = 'only0',
    drop00: bool = False,
    intercept: bool = True,
    method: str | collections.abc.Sequence[str] = 'REML',
    weighted: bool = True,
    test: str = 'z',
    level: float = 95,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    btt: Btt = None,
    scale: None = None,
    tau2: float | None = None,
    control: metafopy.models.tau2.Control | collections.abc.Mapping[str, typing.Any] | None = None,
    verbose: bool | int = False,
    na_action: metafopy.metadata.NAAction = 'omit',
    vtype: str | collections.abc.Sequence[str] = 'LS',
    correct: bool = True,
    cutoff: float | None = None,
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> RMAResult: ...


@typing.overload
def rma_uni(
    yi: metafopy.metadata.Column | pandas.DataFrame | None = None,
    vi: metafopy.metadata.Column | None = None,
    *,
    scale: metafopy.models.rma_ls.ScaleSpec,
    **kwargs: typing.Any,
) -> metafopy.models.rma_ls_result.RMALSResult: ...


@metafopy.models.call.records_call  # type: ignore[misc]  # the scale overload forwards the remaining keywords
def rma_uni(
    yi: metafopy.metadata.Column | pandas.DataFrame | None = None,
    vi: metafopy.metadata.Column | None = None,
    *,
    sei: metafopy.metadata.Column | None = None,
    weights: metafopy.metadata.Column | None = None,
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    m1i: metafopy.metadata.Column | None = None,
    m2i: metafopy.metadata.Column | None = None,
    sd1i: metafopy.metadata.Column | None = None,
    sd2i: metafopy.metadata.Column | None = None,
    xi: metafopy.metadata.Column | None = None,
    mi: metafopy.metadata.Column | None = None,
    ri: metafopy.metadata.Column | None = None,
    ti: metafopy.metadata.Column | None = None,
    fi: metafopy.metadata.Column | None = None,
    pi: metafopy.metadata.Column | None = None,
    sdi: metafopy.metadata.Column | None = None,
    r2i: metafopy.metadata.Column | None = None,
    mini: metafopy.metadata.Column | None = None,
    maxi: metafopy.metadata.Column | None = None,
    ni: metafopy.metadata.Column | None = None,
    mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None = None,
    measure: str = 'GEN',
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    to: typing.Literal['all', 'only0', 'if0all', 'none'] = 'only0',
    drop00: bool = False,
    intercept: bool = True,
    method: str | collections.abc.Sequence[str] = 'REML',
    weighted: bool = True,
    test: str = 'z',
    level: float = 95,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    btt: Btt = None,
    att: Btt = None,
    scale: metafopy.models.rma_ls.ScaleSpec | None = None,
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    link: typing.Literal['log', 'identity'] = 'log',
    optbeta: bool = False,
    alpha: numpy.typing.ArrayLike | None = None,
    beta: numpy.typing.ArrayLike | None = None,
    tau2: float | None = None,
    control: metafopy.models.tau2.Control | metafopy.models.rma_ls_result.LSControl | collections.abc.Mapping[str, typing.Any] | None = None,
    verbose: bool | int = False,
    na_action: metafopy.metadata.NAAction = 'omit',
    vtype: str | collections.abc.Sequence[str] = 'LS',
    correct: bool = True,
    cutoff: float | None = None,
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> RMAResult | metafopy.models.rma_ls_result.RMALSResult:
    """Fit an equal-, fixed-, random- or mixed-effects meta-analytic model.

    Mirrors metafor's ``rma.uni()`` (alias ``rma``); see
    https://wviechtb.github.io/metafor/reference/rma.uni.html. Inputs can be

    - ``yi`` with ``vi`` or ``sei`` (arrays or column names in ``data``);
    - a model formula in ``yi`` (``'yi ~ ablat + year'``), which then also
      supplies the moderators;
    - an :func:`~metafopy.escalc` data frame as ``yi`` or ``data``;
    - raw data (``ai``/``bi``/``ci``/``di``, ``m1i``/``sd1i``/``n1i``, ...)
      with ``measure=``, in which case the effect sizes are computed by
      :func:`~metafopy.escalc` (with ``add``, ``to``, ``drop00``, ``vtype``,
      ``correct``, ``cutoff``, ``onlyo1``, ``addyi`` and ``addvi``).

    ``method`` may be a single estimator or a sequence tried in order until
    one succeeds. ``test`` is ``'z'``, ``'t'``, ``'knha'`` (alias ``'hksj'``) or
    ``'adhoc'``. ``btt`` takes 0-based coefficient indices (or a regex matched
    against coefficient names); printed output reports them 1-based, as
    metafor does.

    ``scale`` (a one-sided formula such as ``'~ ni100'`` or a model matrix) turns the
    model into a location-scale model (Viechtbauer & Lopez-Lopez, 2022,
    doi:10.1002/jrsm.1562) and the result is a
    :class:`~metafopy.models.rma_ls_result.RMALSResult`; ``link`` (``'log'`` or
    ``'identity'``), ``optbeta``, ``alpha`` (fixed values of the scale coefficients),
    ``beta`` (fixed values of the location coefficients, with ``optbeta``) and ``att``
    (the scale coefficients to test) only apply to such models.

    ``digits`` (a number of decimals, or a mapping with the keys ``est``, ``se``, ... of which ``est`` is used) is stored in the result
    and used by its printout unless ``digits`` is passed to ``summary()`` or set with :func:`metafopy.setmfopt`.

    ``verbose`` prints metafor's iteration log of the tau^2 estimators to stdout
    (and with ``verbose=2`` or more its progress messages to stderr); it has no
    effect for location-scale models.

    Examples
    --------
    >>> from metafopy import datasets, rma
    >>> dat = datasets.load("dat.bcg")  # doctest: +SKIP
    >>> rma(measure="RR", ai="tpos", bi="tneg", ci="cpos", di="cneg", data=dat)  # doctest: +SKIP

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    vlevel = int(verbose)
    if vlevel:
        metafopy.core.verbose.cat('\n')
    if vlevel > 1:
        metafopy.core.verbose.message('Extracting/computing the yi/vi values ...')
        if weights is not None:
            metafopy.core.verbose.message('Extracting custom weights ...')
        metafopy.core.verbose.message('Creating the model matrix ...')
        if subset is not None:
            metafopy.core.verbose.message('Subsetting ...')
        metafopy.core.verbose.message('Generating/extracting the study labels ...')
        if subset is not None:
            metafopy.core.verbose.message('Subsetting ...')
    test_l = test.lower()
    if test_l == 'hksj':
        test_l = 'knha'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test_l not in ('z', 't', 'knha', 'adhoc'):
        raise metafopy.errors.InputError("Unknown option specified for the 'test' argument.")

    if isinstance(yi, pandas.DataFrame):
        data = yi
        yi = None

    if isinstance(yi, str) and '~' in yi:
        if mods is not None:
            warnings.warn("Argument 'mods' ignored when 'yi' is a model formula.", stacklevel=2)
        if data is None:
            raise metafopy.errors.InputError('a model formula in yi needs data')
        lhs, rhs = (part.strip() for part in yi.split('~', 1))
        if not lhs:
            raise metafopy.errors.InputError('the model formula in yi must have a left-hand side')
        yi = lhs if lhs in data.columns else numpy.asarray(data.eval(lhs), dtype=numpy.float64)
        mods = '~ ' + rhs
        intercept = False

    if measure == 'GEN' and data is not None and data.attrs.get('measure'):
        measure = str(data.attrs['measure'])

    raw = {
        name: val
        for name, val in zip(
            ESCALC_ARGS,
            (ai, bi, ci, di, n1i, n2i, x1i, x2i, t1i, t2i, m1i, m2i, sd1i, sd2i, xi, mi, ri, ti, fi, pi, sdi, r2i, mini, maxi),
            strict=True,
        )
        if val is not None
    }

    if yi is None and not (data is not None and 'yi' in data.columns and not raw):
        if measure == 'GEN':
            raise metafopy.errors.InputError("Specify the desired outcome measure via the 'measure' argument.")
        md = metadata_from_escalc(
            measure,
            raw,
            ni=ni,
            data=data,
            slab=slab,
            subset=subset,
            mods=mods,
            weights=weights,
            intercept=intercept,
            na_action=na_action,
            escalc_opts={
                'add': add,
                'to': to,
                'drop00': drop00,
                'vtype': vtype,
                'correct': correct,
                'cutoff': cutoff,
                'onlyo1': onlyo1,
                'addyi': addyi,
                'addvi': addvi,
            },
        )
    else:
        md = metafopy.metadata.MetaData.from_inputs(
            yi,
            vi,
            sei=sei,
            weights=weights,
            ni=ni,
            mods=mods,
            intercept=intercept,
            data=data,
            slab=slab,
            subset=subset,
            na_action=na_action,
        )
    digits_ = metafopy.mfopt.reduce_digits(digits) if digits is not None else 4
    if scale is not None:
        return dataclasses.replace(
            fit_location_scale(
                md,
                scale,
                method=method,
                weighted=weighted,
                test=typing.cast(Test, test_l),
                level=level,
                btt=btt,
                att=att,
                tau2=tau2,
                link=link,
                optbeta=optbeta,
                alpha=alpha,
                beta=beta,
                control=control,
                na_action=na_action,
                measure=measure,
            ),
            digits=digits_,
        )
    if md.na_rows is not None and md.na_rows.any():
        if vlevel > 1:
            metafopy.core.verbose.message('Handling NAs ...')
        warn_na_omitted(int(md.na_rows.sum()))
    if optbeta or att is not None:
        warnings.warn("Arguments 'optbeta' and 'att' are only relevant for location-scale models and hence ignored.", stacklevel=2)
    res = fit_rma_uni(
        md,
        method=method,
        weighted=weighted,
        test=typing.cast(Test, test_l),
        level=level,
        btt=btt,
        tau2=tau2,
        control=typing.cast(metafopy.models.tau2.Control | collections.abc.Mapping[str, typing.Any] | None, control),
        verbose=vlevel,
    )
    if vlevel:
        metafopy.core.verbose.cat('\n')
    return dataclasses.replace(res, measure=measure, digits=digits_)


def warn_na_omitted(n: int) -> None:
    """metafor's warning that ``n`` studies were left out of the model fit because of missing values."""
    warnings.warn(f'{n} {"studies" if n > 1 else "study"} with NAs omitted from model fitting.', stacklevel=3)


def fit_location_scale(
    md: metafopy.metadata.MetaData, scale: metafopy.models.rma_ls.ScaleSpec, **kwargs: typing.Any
) -> metafopy.models.rma_ls_result.RMALSResult:
    """Fit the location-scale model of ``rma_uni(..., scale=...)``; see :func:`metafopy.models.rma_ls.fit_scale`."""
    # auditme: accept[in-function-import] metafopy.models.rma_ls imports this module, so the import is deferred to avoid a cycle
    import metafopy.models.rma_ls

    return metafopy.models.rma_ls.fit_scale(md, scale, **kwargs)


def metadata_from_escalc(
    measure: str,
    raw: dict[str, typing.Any],
    *,
    ni: metafopy.metadata.Column | None,
    data: pandas.DataFrame | None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None,
    subset: numpy.typing.ArrayLike | str | None,
    mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None,
    weights: metafopy.metadata.Column | None,
    intercept: bool,
    na_action: metafopy.metadata.NAAction,
    escalc_opts: dict[str, typing.Any],
) -> metafopy.metadata.MetaData:
    """Compute ``yi``/``vi`` with escalc after subsetting, as ``rma.uni`` does for raw data.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
    import metafopy.effects

    if not raw:
        raise metafopy.errors.InputError('Cannot compute outcomes. Check that all of the required information is specified via the appropriate arguments.')
    if data is not None:
        k_all = len(data)
    else:
        first = next(v for v in raw.values() if not isinstance(v, str))
        k_all = int(numpy.atleast_1d(numpy.asarray(first)).shape[0])
    mask = numpy.ones(k_all, dtype=numpy.bool_) if subset is None else metafopy.metadata.subset_mask(subset, k_all, data)
    if ni is not None:
        raw = {**raw, 'ni': ni}
    es = metafopy.effects.escalc(measure, **raw, data=data, subset=mask, **escalc_opts)
    idx = numpy.flatnonzero(mask)

    def sub(value: typing.Any) -> typing.Any:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, pandas.DataFrame):
            return value.iloc[idx].reset_index(drop=True)
        arr = numpy.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == k_all:
            return arr[idx]
        return value

    data_sub = None if data is None else data.iloc[idx].reset_index(drop=True)
    ni_es = es.attrs.get('ni')
    count_names = ('ai', 'bi', 'ci', 'di', 'x1i', 'x2i', 't1i', 't2i')
    counts_all = {
        name: arr[idx] for name in count_names if name in raw and (arr := metafopy.metadata.column(raw[name], data, name)) is not None and arr.shape[0] == k_all
    }
    for count, group_size, other in (('bi', 'n1i', 'ai'), ('di', 'n2i', 'ci')):
        if count not in counts_all and other in counts_all and group_size in raw:
            sizes = metafopy.metadata.column(raw[group_size], data, group_size)
            if sizes is not None and sizes.shape[0] == k_all:
                counts_all[count] = sizes[idx] - counts_all[other]
    md = metafopy.metadata.MetaData.from_inputs(
        es['yi'].to_numpy(dtype=numpy.float64),
        es['vi'].to_numpy(dtype=numpy.float64),
        weights=sub(weights),
        ni=None if ni_es is None else numpy.asarray(ni_es, dtype=numpy.float64),
        mods=sub(mods),
        intercept=intercept,
        data=data_sub,
        slab=[str(i + 1) for i in idx] if slab is None else sub(slab),
        na_action=na_action,
    )
    return dataclasses.replace(md, counts_all=counts_all or None)


rma = rma_uni


def fit_rma_uni(
    md: metafopy.metadata.MetaData,
    *,
    method: str | collections.abc.Sequence[str] = 'REML',
    weighted: bool = True,
    test: Test = 'z',
    level: float = 95,
    btt: Btt = None,
    tau2: float | None = None,
    control: metafopy.models.tau2.Control | collections.abc.Mapping[str, typing.Any] | None = None,
    skip_r2: bool = False,
    verbose: int = 0,
) -> RMAResult:
    """Fit ``rma.uni`` to already normalised inputs.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: every requested tau^2 estimator failed.
    """
    methods = [method] if isinstance(method, str) else list(method)
    for m_ in methods:
        if m_ not in metafopy.models.tau2.ESTIMATORS:
            raise metafopy.errors.InputError(f'unknown tau^2 estimator {m_!r}')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test not in ('z', 't', 'knha', 'adhoc'):
        raise metafopy.errors.InputError(f'unknown test {test!r}')

    yi = md.yi.copy()
    vi = md.vi.copy()
    X = md.X.copy()
    names = list(md.X_names)
    k = md.k
    user_w = md.weights
    if k < 1:
        raise metafopy.errors.InputError('processing terminated since k = 0')

    if numpy.any(vi <= 0):
        allvipos = False
        if numpy.any(vi < 0):
            warnings.warn('Negative sampling variances constrained to 0.', stacklevel=2)
            vi[vi < 0] = 0.0
    else:
        allvipos = True

    if k == 1 and test != 'z':
        warnings.warn('Setting argument test="z" since k=1.', stacklevel=2)
        test = 'z'

    drop = metafopy.core.linalg.drop_redundant(X)
    if drop.any():
        warnings.warn('Redundant predictors dropped from the model.', stacklevel=2)
        X = X[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]

    is_int = numpy.array([metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])])
    int_incl = bool(is_int.any())
    if int_incl:
        X = numpy.column_stack([numpy.ones(k), X[:, ~is_int]])
        names = ['intrcpt', *[n for n, i in zip(names, is_int, strict=True) if not i]]
    p = X.shape[1]
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])

    tau2_fix = tau2 is not None and not math.isnan(tau2) and methods[0] not in metafopy.models.tau2.FIXED_EFFECTS
    if not (int_only and k == 1):
        need = p if (methods[0] in metafopy.models.tau2.FIXED_EFFECTS or tau2_fix) else p + 1
        if need > k:
            raise metafopy.errors.InputError('Number of parameters to be estimated is larger than the number of observations.')

    btt_idx = set_btt(btt, p, int_incl, names)
    m = len(btt_idx)

    con = make_control(control)
    if control is None or 'tau2_max' not in {key.replace('.', '_') for key in (control if isinstance(control, collections.abc.Mapping) else {})}:
        con = dataclasses.replace(con, tau2_max=max(con.tau2_max, 10 * mad(yi) ** 2))

    if not metafopy.core.linalg.is_positive_definite(X.T @ X, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')

    # -- heterogeneity ---------------------------------------------------
    used = methods[0]
    tau2_hat = 0.0
    se_tau2 = math.nan
    iterations = 0
    if verbose > 1 and not tau2_fix and used not in metafopy.models.tau2.FIXED_EFFECTS:
        metafopy.core.verbose.message('Estimating the tau^2 value ...\n')
    if used not in metafopy.models.tau2.FIXED_EFFECTS:
        errors: list[str] = []
        for cand in methods:
            used = cand
            if cand in metafopy.models.tau2.FIXED_EFFECTS:
                break
            try:
                fit = metafopy.models.tau2.estimate_tau2(
                    cand,
                    yi,
                    vi,
                    X,
                    weights=user_w,
                    tau2_fixed=float(tau2) if tau2_fix else None,  # type: ignore[arg-type]
                    allvipos=allvipos,
                    con=con,
                    verbose=verbose,
                    digits=metafopy.mfopt.option_digits(None),
                )
            except metafopy.errors.EstimationError as exc:
                errors.append(str(exc))
                continue
            tau2_hat, se_tau2, iterations = fit.tau2, fit.se_tau2, fit.iterations
            break
        else:
            raise metafopy.errors.EstimationError(errors[-1])
    if used in metafopy.models.tau2.FIXED_EFFECTS:
        tau2_hat = 0.0
        se_tau2 = math.nan

    # -- fixed effects -----------------------------------------------------
    if verbose > 1:
        metafopy.core.verbose.message('\nModel fitting ...')
    alpha = level_to_alpha(level)
    wi = 1.0 / (vi + tau2_hat)
    beta, vb, RSS_f, s2w = fixed_effects(yi, vi, X, tau2_hat, user_w, weighted, test)

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    tdist = test in ('knha', 'adhoc', 't')
    ddf = k - p if tdist else None

    if verbose > 1:
        metafopy.core.verbose.message('Conducting the tests of the fixed effects ...')
    sub = numpy.array(btt_idx)
    try:
        vsub = vb[numpy.ix_(sub, sub)]
        L = numpy.linalg.cholesky(vsub)
        b = beta[sub]
        z = numpy.linalg.solve(L, b)
        QM = float(z @ z)
    except numpy.linalg.LinAlgError:
        QM = math.nan

    se = numpy.sqrt(numpy.diag(vb))
    zval = beta / se
    if tdist:
        if ddf is None:
            raise metafopy.errors.InternalError('ddf must not be None')
        QM = QM / m
        QMdf: tuple[int, int | None] = (m, ddf)
        QMp = float(scipy.stats.f.sf(QM, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * scipy.stats.t.sf(numpy.abs(zval), ddf) if ddf > 0 else numpy.full(p, math.nan)
        crit = float(scipy.stats.t.isf(alpha / 2, ddf)) if ddf > 0 else math.nan
    else:
        QMdf = (m, None)
        QMp = float(scipy.stats.chi2.sf(QM, m))
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(alpha / 2))
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    # -- heterogeneity statistics ----------------------------------------------
    if verbose > 1:
        metafopy.core.verbose.message('Conducting the heterogeneity test ...')
    QE = QEp = I2 = H2 = vt = math.nan
    if allvipos:
        if k > p:
            w_fe = 1.0 / vi
            WX = w_fe[:, None] * X
            P = numpy.diag(w_fe) - WX @ metafopy.core.linalg.invcalc(X, w_fe) @ WX.T
            ymc = yi - yi.mean()
            QE = max(0.0, float(ymc @ P @ ymc))
            QEp = float(scipy.stats.chi2.sf(QE, k - p))
            vt = (k - p) / float(numpy.trace(P))
            if used in metafopy.models.tau2.FIXED_EFFECTS:
                I2 = max(0.0, 100 * (QE - (k - p)) / QE)
                H2 = QE / (k - p)
            else:
                I2 = 100 * tau2_hat / (vt + tau2_hat)
                H2 = tau2_hat / vt + 1
        else:
            QE, QEp, I2, H2, vt = 0.0, 1.0, 0.0, 1.0, 0.0

    # -- fit statistics -----------------------------------------------------
    if verbose > 1:
        metafopy.core.verbose.message('Computing fit statistics and log-likelihood ...')
    parms = p + (0 if (used in metafopy.models.tau2.FIXED_EFFECTS or tau2_fix) else 1)
    lv = numpy.log(vi + tau2_hat)
    ll_ml = -0.5 * k * math.log(2 * math.pi) - 0.5 * float(lv.sum()) - 0.5 * RSS_f
    ll_reml = (
        -0.5 * (k - p) * math.log(2 * math.pi)
        + (0.5 * metafopy.core.linalg.logdet(X.T @ X) if con.REMLf else 0.0)
        - 0.5 * float(lv.sum())
        - 0.5 * metafopy.core.linalg.logdet((X.T * wi) @ X)
        - 0.5 * RSS_f
    )
    if k > p:
        if allvipos:
            dev_ml = -2 * (ll_ml - float(numpy.sum(-0.5 * numpy.log(2 * math.pi * vi))))
        else:
            dev_ml = -2 * ll_ml
    else:
        dev_ml = 0.0
    fs_ml = FitStats(
        ll=ll_ml,
        dev=dev_ml,
        AIC=-2 * ll_ml + 2 * parms,
        BIC=-2 * ll_ml + parms * math.log(k),
        AICc=aicc(ll_ml, parms, k),
    )
    fs_reml = FitStats(
        ll=ll_reml,
        dev=-2 * ll_reml,
        AIC=-2 * ll_reml + 2 * parms,
        BIC=-2 * ll_reml + parms * math.log(k - p) if k > p else math.nan,
        AICc=aicc(ll_reml, parms, k - p),
    )

    # -- pseudo R^2 ----------------------------------------------------------
    R2: float | None = None
    if not int_only and metafopy.core.linalg.is_nested(X, numpy.ones((k, 1))) and not skip_r2:
        if verbose > 1:
            metafopy.core.verbose.message('Computing R^2 ...')
        R2 = pseudo_r2(
            md,
            yi=yi,
            X=X,
            wi=wi,
            user_w=user_w,
            weighted=weighted,
            used=used,
            methods=methods,
            test=test,
            con=con,
            tau2_hat=tau2_hat,
            verbose=verbose,
        )

    if verbose > 1:
        metafopy.core.verbose.message('Preparing the output ...')
    return RMAResult(
        beta=beta,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vb=vb,
        coef_names=tuple(names),
        tau2=tau2_hat,
        se_tau2=se_tau2,
        k=k,
        p=p,
        m=m,
        btt=btt_idx,
        QE=QE,
        QEp=QEp,
        QM=QM,
        QMp=QMp,
        QMdf=QMdf,
        I2=I2,
        H2=H2,
        R2=R2,
        vt=vt,
        fit_stats_ml=fs_ml,
        fit_stats_reml=fs_reml,
        parms=parms,
        method=used,
        test=test,
        level=alpha,
        ddf=ddf,
        s2w=s2w,
        weighted=weighted,
        weights=user_w,
        tau2_fix=tau2_fix,
        int_only=int_only,
        int_incl=int_incl,
        allvipos=allvipos,
        yi=yi,
        vi=vi,
        X=X,
        slab=md.slab,
        data=md,
        control=con,
        iterations=iterations,
    )


def fixed_effects(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    tau2: float,
    user_w: FloatArray | None,
    weighted: bool,
    test: Test,
) -> tuple[FloatArray, FloatArray, float, float]:
    """``beta``, its (Knapp-Hartung scaled) covariance, ``RSS_f`` and ``s2w``.

    Follows metafor: with user weights the estimate uses them but RSS (and the
    Knapp-Hartung factor) still use inverse-variance weights.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    k, p = X.shape
    wi = 1.0 / (vi + tau2)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    knha = test in ('knha', 'adhoc')
    s2w = 1.0
    if weighted:
        if numpy.any(numpy.isinf(wi)) and (user_w is None or knha):
            raise metafopy.errors.InputError('Division by zero when computing the inverse variance weights.')
        vb = metafopy.core.linalg.invcalc(X, wi)
        beta = vb @ (X.T @ (wi * yi))
        RSS_knha = float(numpy.sum(wi * (yi - X @ beta) ** 2))
        if user_w is not None:
            stXAX = metafopy.core.linalg.invcalc(X, user_w)
            beta = stXAX @ (X.T @ (user_w * yi))
            XA = X.T * user_w
            vb = stXAX @ XA @ numpy.diag(vi + tau2) @ XA.T @ stXAX
        RSS_f = float(numpy.sum(wi * (yi - X @ beta) ** 2))
    else:
        stXX = metafopy.core.linalg.invcalc(X, numpy.ones(k))
        beta = stXX @ (X.T @ yi)
        vb = stXX @ X.T @ numpy.diag(vi + tau2) @ X @ stXX
        RSS_f = float(numpy.sum(wi * (yi - X @ beta) ** 2))
        RSS_knha = math.nan
        if knha:
            if numpy.any(numpy.isinf(wi)):
                raise metafopy.errors.InputError('Division by zero when computing the inverse variance weights.')
            b_k = metafopy.core.linalg.invcalc(X, wi) @ (X.T @ (wi * yi))
            RSS_knha = float(numpy.sum(wi * (yi - X @ b_k) ** 2))
    if knha:
        s2w = 0.0 if RSS_knha <= DBL_EPS else RSS_knha / (k - p)
    if test == 'adhoc':
        s2w = max(s2w, 1.0)
    return beta, s2w * vb, RSS_f, s2w


def aicc(ll: float, parms: int, n: int) -> float:
    big = max(n, parms + 2)
    return -2 * ll + 2 * parms * big / (big - parms - 1)


def pseudo_r2(
    md: metafopy.metadata.MetaData,
    *,
    yi: FloatArray,
    X: FloatArray,
    wi: FloatArray,
    user_w: FloatArray | None,
    weighted: bool,
    used: str,
    methods: list[str],
    test: Test,
    con: metafopy.models.tau2.Control,
    tau2_hat: float,
    verbose: int = 0,
) -> float | None:
    k = yi.shape[0]
    if used in metafopy.models.tau2.FIXED_EFFECTS:
        if numpy.var(yi) == 0:
            return 0.0
        w = (wi if user_w is None else user_w) if weighted else numpy.ones(k)
        return max(0.0, 100 * adj_r2(yi, X, w))
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    md0 = dataclasses.replace(md, yi=yi, X=numpy.ones((k, 1)), X_names=('intrcpt',), intercept=True)
    if verbose:
        # metafor refits the intercept-only model with verbose=TRUE, which logs its iterations between blank lines
        metafopy.core.verbose.cat('\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res0 = fit_rma_uni(
                md0,
                method=methods,
                weighted=weighted,
                test=test,
                control=con,
                skip_r2=True,
                verbose=1 if verbose else 0,
            )
    except metafopy.errors.EstimationError, ValueError:
        return math.nan
    finally:
        if verbose:
            metafopy.core.verbose.cat('\n')
    tau2_re = res0.tau2
    if tau2_re == 0:
        return 0.0
    return max(0.0, 100 * (tau2_re - tau2_hat) / tau2_re)


def adj_r2(y: FloatArray, X: FloatArray, w: FloatArray) -> float:
    """Adjusted R^2 of ``lm(y ~ X, weights=w)`` (with R's added intercept)."""
    n = y.shape[0]
    Xi = numpy.column_stack([numpy.ones(n), X])
    keep = ~metafopy.core.linalg.drop_redundant(Xi)
    Xi = Xi[:, keep]
    sw = numpy.sqrt(w)
    coef, *_ = numpy.linalg.lstsq(Xi * sw[:, None], y * sw, rcond=None)
    resid = y - Xi @ coef
    rss = float(numpy.sum(w * resid**2))
    mean = float(numpy.sum(w * y) / numpy.sum(w))
    tss = float(numpy.sum(w * (y - mean) ** 2))
    r2 = 1 - rss / tss
    rdf = n - Xi.shape[1]
    return float(1 - (1 - r2) * ((n - 1) / rdf))


# -- printing -----------------------------------------------------------------


def format_result(res: RMAResult, digits: int, showfit: bool = False, signif_stars: bool = True, signif_legend: bool = True) -> str:
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    out: list[str] = ['']
    fe = res.method in metafopy.models.tau2.FIXED_EFFECTS
    if fe:
        if res.int_only:
            kind = 'Equal-Effects' if res.method == 'EE' else ('Fixed-Effects' if res.method == 'FE' else 'Common-Effects')
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
                [metafopy.printing.fmt_num(v, digits) for v in res.fit_stats(None).as_tuple()],
            )
        )
        out.append('')

    if not fe:
        het = 'total' if res.int_only else 'residual'
        tau2_line = metafopy.printing.fmt_num(res.tau2, digits)
        if not math.isnan(res.se_tau2):
            tau2_line += f' (SE = {metafopy.printing.fmt_num(res.se_tau2, digits)})'
        est = 'specified' if res.tau2_fix else 'estimated'
        rows = [
            (f'tau^2 ({est} amount of {het} heterogeneity):', tau2_line),
            (f'tau (square root of {est} tau^2 value):', metafopy.printing.fmt_num(math.sqrt(res.tau2), digits)),
        ]
        if not math.isnan(res.I2):
            if res.int_only:
                rows += [
                    ('I^2 (total heterogeneity / total variability):', f'{res.I2:.2f}%'),
                    ('H^2 (total variability / sampling variability):', f'{res.H2:.2f}'),
                ]
            else:
                rows += [
                    ('I^2 (residual heterogeneity / unaccounted variability):', f'{res.I2:.2f}%'),
                    ('H^2 (unaccounted variability / sampling variability):', f'{res.H2:.2f}'),
                ]
        if res.R2 is not None:
            rows.append(('R^2 (amount of heterogeneity accounted for):', f'{res.R2:.2f}%'))
        width = max(len(r[0]) for r in rows)
        out += [f'{label.ljust(width)} {value}' for label, value in rows]
        out.append('')
    elif not math.isnan(res.I2):
        if res.int_only:
            out.append(f'I^2 (total heterogeneity / total variability):   {res.I2:.2f}%')
            out.append(f'H^2 (total variability / sampling variability):  {res.H2:.2f}')
        else:
            out.append(f'I^2 (residual heterogeneity / unaccounted variability): {res.I2:.2f}%')
            out.append(f'H^2 (unaccounted variability / sampling variability):   {res.H2:.2f}')
            if res.R2 is not None:
                out.append(f'R^2 (amount of heterogeneity accounted for):            {res.R2:.2f}%')
        out.append('')

    if not math.isnan(res.QE):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        name, label = ('Q', 'Heterogeneity') if res.int_only else ('QE', 'Residual Heterogeneity')
        out.append(f'Test for {label}:')
        out.append(f'{name}(df = {res.k - res.p}) = {metafopy.printing.fmt_num(res.QE, digits)}, p-val {pfmt(res.QEp, digits)}')
        out.append('')

    robust = res.robust
    if robust is not None:
        out += metafopy.models.robust_print.counts_lines(robust, res.k)

    if not res.int_only:
        coefs = coef_span(res.btt)
        mark = metafopy.models.robust_print.FOOT_MARK if robust is not None else ''
        out.append(f'Test of Moderators (coefficient{"s" if res.m > 1 else ""} {coefs}):{mark}')
        if robust is not None:
            df2 = metafopy.printing.fmt_df_scalar(res.QMdf[1])
            out.append(f'F(df1 = {res.m}, df2 = {df2}) = {metafopy.printing.fmt_num(res.QM, digits)}, p-val {pfmt(res.QMp, digits)}')
        elif res.ddf is not None:
            out.append(f'F(df1 = {res.m}, df2 = {res.ddf}) = {metafopy.printing.fmt_num(res.QM, digits)}, p-val {pfmt(res.QMp, digits)}')
        else:
            out.append(f'QM(df = {res.m}) = {metafopy.printing.fmt_num(res.QM, digits)}, p-val {pfmt(res.QMp, digits)}')
        out.append('')

    out.append('Model Results:')
    out.append('')
    use_t = res.ddf is not None or robust is not None
    stat = 'tval' if use_t else 'zval'
    header = ['estimate', 'se', stat]
    if use_t:
        header.append('df')
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    header += ['pval', 'ci.lb', 'ci.ub', '']
    rows_: list[list[str]] = []
    for j in range(res.p):
        row = [
            metafopy.printing.fmt_num(res.beta[j], digits),
            metafopy.printing.fmt_num(res.se[j], digits),
            metafopy.printing.fmt_num(res.zval[j], digits),
        ]
        if robust is not None:
            row.append(metafopy.printing.fmt_df_scalar(float(robust.dfs[j])))
        elif res.ddf is not None:
            row.append(str(res.ddf))
        row += [
            metafopy.printing.fmt_pval(res.pval[j], digits),
            metafopy.printing.fmt_num(res.ci_lb[j], digits),
            metafopy.printing.fmt_num(res.ci_ub[j], digits),
            metafopy.printing.signif_stars(res.pval[j]),
        ]
        rows_.append(row)
    if robust is not None:
        header, rows_ = metafopy.models.robust_print.decorate(header, rows_)
    header, rows_ = metafopy.printing.drop_stars(header, rows_, signif_stars)
    out.append(metafopy.printing.fmt_table(header, rows_, row_names=None if res.int_only else list(res.coef_names)))
    out += metafopy.printing.signif_footer(signif_legend, metafopy.models.robust_print.legend_lines(robust, res.int_only) if robust is not None else ())
    return '\n'.join(out)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def pfmt(p: float, digits: int) -> str:
    s = metafopy.printing.fmt_pval(p, digits)
    return f'< {s[1:]}' if s.startswith('<') else f'= {s}'


def coef_span(idx: collections.abc.Sequence[int]) -> str:
    one = [i + 1 for i in idx]
    if len(one) > 1 and one == list(range(one[0], one[-1] + 1)):
        return f'{one[0]}:{one[-1]}'
    return ','.join(str(i) for i in one)
