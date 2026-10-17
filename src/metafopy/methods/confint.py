"""Confidence intervals for model coefficients and variance components.

For ``rma.uni`` fits, the CI for tau^2 uses the Q-profile method (Viechtbauer
2007, doi:10.1002/sim.2514) by default, with I^2, H^2 and tau intervals derived
from its bounds, exactly as metafor's ``confint.rma.uni()`` does.
"""

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

import metafopy.core.linalg
import metafopy.core.verbose
import metafopy.core.zeroin
import metafopy.errors
import metafopy.formatters
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.control_names
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_mv
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing
import metafopy.transf

__all__ = ['ConfintList', 'ConfintResult', 'confint']

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
SPATIAL = ('SPEXP', 'SPGAU', 'SPLIN', 'SPRAT', 'SPSPH', 'PHYPL', 'PHYPD')

FloatArray = numpy.typing.NDArray[numpy.float64]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
CIType = typing.Literal['qp', 'genq', 'pl', 'ht', 'wald', 'wald.log', 'wald.sqrt']


@dataclasses.dataclass(frozen=True, slots=True)
class ConfintResult:
    """``fixed`` and/or ``random`` tables with ``estimate``, ``ci.lb``, ``ci.ub`` columns."""

    fixed: pandas.DataFrame | None
    random: pandas.DataFrame | None
    lb_sign: str = ''
    ub_sign: str = ''
    digits: int | None = None

    def to_frame(self) -> pandas.DataFrame:
        """The ``fixed`` and ``random`` tables stacked into one data frame (the columns are ``estimate``, ``ci.lb`` and ``ci.ub``)."""
        tables = [t for t in (self.fixed, self.random) if t is not None]
        if not tables:
            return pandas.DataFrame(columns=ci_columns())
        return pandas.concat(tables)

    def summary(self, digits: int | collections.abc.Mapping[str, int] | None = None) -> str:
        """The printout of metafor's ``print.confint.rma``: the table of the coefficients, then that of the variance components."""
        d = metafopy.mfopt.option_digits(digits, 4 if self.digits is None else self.digits)
        parts = []
        for table, lb_sign, ub_sign in ((self.fixed, '', ''), (self.random, self.lb_sign, self.ub_sign)):
            if table is not None:
                cells = [[metafopy.printing.fmt_num(float(v), d) for v in row] for row in table.to_numpy()]
                for row in cells:
                    row[1], row[2] = lb_sign + row[1], ub_sign + row[2]
                parts.append(metafopy.printing.fmt_table(ci_columns(), cells, [str(n) for n in table.index], gap=1))
        return '\n\n'.join(parts)

    def __str__(self) -> str:
        return self.summary()


def ci_columns() -> list[str]:
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    return ['estimate', 'ci.lb', 'ci.ub']


class ConfintList(list[ConfintResult]):
    """The confidence intervals of several variance components of an ``rma.mv`` fit (metafor's ``list.confint.rma``): one :class:`ConfintResult`
    per component, in the order of ``sigma2``, ``tau2``, ``rho``, ``gamma2`` and ``phi``."""

    def __init__(self, results: collections.abc.Iterable[ConfintResult], digits: int | None = None) -> None:
        super().__init__(results)
        self.digits = digits

    def to_frame(self) -> pandas.DataFrame:
        """The tables of all components stacked into one data frame (the columns are ``estimate``, ``ci.lb`` and ``ci.ub``)."""
        return pandas.concat([r.to_frame() for r in self])

    def summary(self, digits: int | collections.abc.Mapping[str, int] | None = None) -> str:
        """The printout of metafor's ``print.list.confint.rma``: the tables of the components, separated by blank lines."""
        d = metafopy.mfopt.option_digits(digits, 4 if self.digits is None else self.digits)
        return '\n\n'.join(r.summary(d) for r in self)

    def __str__(self) -> str:
        return self.summary()


@functools.singledispatch
def confint(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Confidence intervals for a fitted model.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'confint() not implemented for {type(model).__name__}')


def qe(tau2: float, y: FloatArray, vi: FloatArray, X: FloatArray) -> float:
    if numpy.any(tau2 + vi < 0):
        raise metafopy.errors.InputError('Some marginal variances are negative.')
    w = 1.0 / (vi + tau2)
    WX = w[:, None] * X
    P = numpy.diag(w) - WX @ metafopy.core.linalg.invcalc(X, w) @ WX.T
    return float(y @ P @ y)


@confint.register
def confint_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    *,
    fixed: bool = False,
    random: bool = True,
    type: CIType | None = None,
    level: float | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    tau2_min: float | None = None,
    tau2_max: float | None = None,
    tol: float = metafopy.core.zeroin.DEFAULT_TOL,
    maxiter: int = 1000,
    verbose: bool | int = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    xlim: float | collections.abc.Sequence[float] | None = None,
) -> ConfintResult:
    """CIs for an ``rma.uni`` fit (metafor's ``confint.rma.uni``).

    ``type`` is ``'qp'`` (Q-profile; Viechtbauer, 2007, doi:10.1002/sim.2514),
    ``'genq'`` (generalised Q; Jackson, 2013, doi:10.1002/jrsm.1081; the default
    for ``method='GENQ'``/``'GENQM'`` fits), ``'pl'`` (profile likelihood),
    ``'ht'`` (Higgins & Thompson, 2002, doi:10.1002/sim.1186) or one of the
    Wald types. ``transf`` (with optional ``targs``) is applied to the fixed
    effects and their bounds. ``verbose`` prints the steps of the root finding
    (on stdout, as metafor's ``cat()``), with ``digits`` decimals. ``control`` is metafor's list of root finding settings
    (``tau2_min``, ``tau2_max``, ``tol`` and ``maxiter``, written with ``.`` or ``_``); its entries replace the keyword arguments of the same name.
    ``xlim`` (metafor's ``...`` argument) is the interval ``(tau2_min, tau2_max)`` searched for the bounds (a single value is the upper
    limit and the lower one is 0) and wins over ``control``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    vlevel = int(verbose)
    ndig = metafopy.mfopt.option_digits(digits, x.digits)
    con = control_values(control, {'tau2_min': tau2_min, 'tau2_max': tau2_max, 'tol': tol, 'maxiter': maxiter})
    tau2_min, tau2_max, tol, maxiter = con['tau2_min'], con['tau2_max'], con['tol'], con['maxiter']
    if xlim is not None:
        limits = [float(v) for v in numpy.atleast_1d(numpy.asarray(xlim, dtype=numpy.float64))]
        if len(limits) == 1:
            limits = [0.0, limits[0]]
        if len(limits) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 1 or 2.")
        tau2_min, tau2_max = limits
    if type is None:
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        type = 'genq' if x.method in ('GENQ', 'GENQM') else 'qp'
    else:
        type = typing.cast(CIType, type.lower())
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if type not in ('qp', 'genq', 'pl', 'ht', 'wald', 'wald.log', 'wald.sqrt'):
            raise metafopy.errors.InputError(f'unknown CI type {type!r}')
    if not isinstance(fixed, bool | numpy.bool_) or not isinstance(random, bool | numpy.bool_):
        raise metafopy.errors.InputError("Arguments 'fixed' and 'random' must be logicals.")
    if not fixed and not random:
        raise metafopy.errors.InputError("at least one of 'fixed' and 'random' must be True")
    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)

    res_random: pandas.DataFrame | None = None
    lb_sign = ub_sign = ''
    if random:
        k, p = x.k, x.p
        if k == 1:
            raise metafopy.errors.InputError('stopped because k = 1')
        if x.method in metafopy.models.tau2.FIXED_EFFECTS:
            raise metafopy.errors.InputError('model does not contain a random-effects component')
        if x.tau2_fix:
            raise metafopy.errors.InputError('model does not contain an estimated random-effects component')
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if type == 'genq' and x.method not in ('GENQ', 'GENQM'):
            raise metafopy.errors.InputError('model must be fitted with method="GENQ" or method="GENQM" to use this option')

        lo = min(0.0, x.tau2) if tau2_min is None else tau2_min
        hi = max(100.0, x.tau2 * 10, lo * 10) if tau2_max is None else tau2_max
        tau2_lb = tau2_ub = math.nan
        lb_conv = ub_conv = False

        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        if type == 'qp':
            if not x.allvipos:
                raise metafopy.errors.InputError('cannot compute CI for tau^2 when there are non-positive sampling variances')
            y, vi, X = x.yi, x.vi, x.X
            crit_u = float(scipy.stats.chi2.isf(alpha / 2, k - p))
            crit_l = float(scipy.stats.chi2.ppf(alpha / 2, k - p))
            qe_max = qe(hi, y, vi, X)
            # as metafor: try(.QE.func(tau2.min)); a try-error compares as false with a number, so the roots are searched (and fail)
            try:
                qe_min = qe(lo, y, vi, X)
            except metafopy.errors.InputError:
                qe_min = math.nan

            if qe_min < crit_l:
                tau2_lb = tau2_ub = lo
                lb_sign = ub_sign = '<'
                lb_conv = ub_conv = True
            elif qe_max > crit_l:
                tau2_ub = hi
                ub_sign = '>'
                ub_conv = True
            else:
                tau2_ub, ub_conv = find_root(lambda t: qe_log(t, y, vi, X, crit_l, vlevel, ndig), lo, hi, tol, maxiter)

            if qe_max > crit_u:
                tau2_lb = tau2_ub = hi
                lb_sign = ub_sign = '>'
                lb_conv = ub_conv = True
            elif qe_min < crit_u:
                tau2_lb = lo
                lb_conv = True
                if lo > 0:
                    lb_sign = '<'
            else:
                tau2_lb, lb_conv = find_root(lambda t: qe_log(t, y, vi, X, crit_u, vlevel, ndig), lo, hi, tol, maxiter)
        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        elif type == 'genq':
            if x.weights is None:
                raise metafopy.errors.InputError('cannot extract weights from the model object')
            y, vi, X = x.yi, x.vi, x.X
            A = x.weights
            AX = A[:, None] * X
            P = numpy.diag(A) - AX @ metafopy.core.linalg.invcalc(X, A) @ AX.T
            Q = float(y @ P @ y)

            def g(t: float, level_: float, getlower: bool) -> float:
                return metafopy.models.tau2.genq_func(t, P, vi, Q, level_, k, p, getlower=getlower)

            def g_log(t: float, getlower: bool) -> float:
                return metafopy.models.tau2.log_objective(t, g(t, alpha / 2, getlower), '  objective =', vlevel, ndig)

            g_max = g(hi, 0.0, True)
            g_min = g(lo, 0.0, True)
            if g_min > 1 - alpha / 2:
                tau2_lb = tau2_ub = lo
                lb_sign = ub_sign = '<'
                lb_conv = ub_conv = True
            elif g_max < 1 - alpha / 2:
                tau2_ub = hi
                ub_sign = '>'
                ub_conv = True
            else:
                # metafor calls uniroot() here with its default tol and maxiter
                tau2_ub, ub_conv = find_root(lambda t: g_log(t, False), lo, hi, metafopy.core.zeroin.DEFAULT_TOL, 1000)

            if g_max < alpha / 2:
                tau2_lb = tau2_ub = hi
                lb_sign = ub_sign = '>'
                lb_conv = ub_conv = True
            elif g_min > alpha / 2:
                tau2_lb = lo
                lb_conv = True
                if lo > 0:
                    lb_sign = '<'
            else:
                tau2_lb, lb_conv = find_root(lambda t: g_log(t, True), lo, hi, metafopy.core.zeroin.DEFAULT_TOL, 1000)
        elif type == 'pl':
            if lo > x.tau2:
                raise metafopy.errors.InputError('lower bound of interval to be searched must be <= actual value of component')
            if hi < x.tau2:
                raise metafopy.errors.InputError('upper bound of interval to be searched must be >= actual value of component')
            objective = float(scipy.stats.chi2.ppf(1 - alpha, 1))

            def prof(t: float) -> float:
                try:
                    out = profile_lrt(x, t) - objective
                except ValueError, ArithmeticError, numpy.linalg.LinAlgError, metafopy.errors.MetaforError:
                    if vlevel:
                        metafopy.core.verbose.cat(f'tau2 = {metafopy.formatters.fmtx(t, ndig, addwidth=4)}   LRT - objective = NA \n')
                    raise
                if vlevel:
                    metafopy.core.verbose.cat(
                        f'tau2 = {metafopy.formatters.fmtx(t, ndig, addwidth=4)}   LRT - objective = {metafopy.formatters.fmtx(out, ndig, addwidth=4)} \n'
                    )
                return out

            val = try_profile(prof, lo)
            if not math.isnan(val):
                if val < 0:
                    tau2_lb = lo
                    lb_conv = True
                    if lo > 0:
                        lb_sign = '<'
                else:
                    tau2_lb, lb_conv = find_root(prof, lo, x.tau2, tol, maxiter, check_conv=True)
            val = try_profile(prof, hi)
            if not math.isnan(val):
                if val < 0:
                    tau2_ub = hi
                    ub_conv = True
                    ub_sign = '>'
                else:
                    tau2_ub, ub_conv = find_root(prof, x.tau2, hi, tol, maxiter, check_conv=True)
        elif type == 'ht':
            if not x.int_only:
                raise metafopy.errors.InputError('method only applicable to models without moderators')
            if k <= 2:
                raise metafopy.errors.InputError('method only applicable when k > 2')
            if k < x.QE:
                se_lnH = 0.5 * (math.log(x.QE) - math.log(k - 1)) / (math.sqrt(2 * x.QE) - math.sqrt(2 * k - 3))
            else:
                se_lnH = math.sqrt(1 / (2 * (k - 2)) * (1 - 1 / (3 * (k - 2) ** 2)))
            crit = float(scipy.stats.norm.isf(alpha / 2))
            lb_conv = ub_conv = True
            H2_lb = math.exp(math.log(x.H2) - crit * 2 * se_lnH)
            H2_ub = math.exp(math.log(x.H2) + crit * 2 * se_lnH)
            I2_lb = (H2_lb - 1) / H2_lb
            I2_ub = (H2_ub - 1) / H2_ub
            tau2_lb = max(0.0, I2_lb * x.vt / (1 - I2_lb))
            tau2_ub = I2_ub * x.vt / (1 - I2_ub)
        else:
            crit = float(scipy.stats.norm.isf(alpha / 2))
            lb_conv = ub_conv = True
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            if type == 'wald':
                tau2_lb = max(0.0 if tau2_min is None else tau2_min, x.tau2 - crit * x.se_tau2)
                tau2_ub = x.tau2 + crit * x.se_tau2
            elif type == 'wald.log':
                if x.tau2 >= 0:
                    tau2_lb = math.exp(math.log(x.tau2) - crit * x.se_tau2 / x.tau2)
                    tau2_ub = max(x.tau2, math.exp(math.log(x.tau2) + crit * x.se_tau2 / x.tau2))
            elif type == 'wald.sqrt':
                if x.tau2 >= 0:
                    st = math.sqrt(x.tau2)
                    tau2_lb = max(0.0, st - crit * x.se_tau2 / (2 * st)) ** 2
                    tau2_ub = (st + crit * x.se_tau2 / (2 * st)) ** 2
            else:
                raise metafopy.errors.InputError(f'unknown CI type {type!r}')

        if not lb_conv:
            warnings.warn('Error in iterative search for the lower bound.', stacklevel=3)
        if not ub_conv:
            warnings.warn('Error in iterative search for the upper bound.', stacklevel=3)

        vt = x.vt

        def sq(v: float) -> float:
            return math.sqrt(v) if v >= 0 else math.nan

        res_random = pandas.DataFrame(
            {
                'estimate': [x.tau2, sq(x.tau2), x.I2, x.H2],
                'ci.lb': [tau2_lb, sq(tau2_lb), 100 * tau2_lb / (vt + tau2_lb), tau2_lb / vt + 1],
                'ci.ub': [tau2_ub, sq(tau2_ub), 100 * tau2_ub / (vt + tau2_ub), tau2_ub / vt + 1],
            },
            index=['tau^2', 'tau', 'I^2(%)', 'H^2'],
        )

    res_fixed: pandas.DataFrame | None = None
    if fixed:
        if x.ddf is not None:
            crit = float(scipy.stats.t.isf(alpha / 2, x.ddf)) if x.ddf > 0 else math.nan
        else:
            crit = float(scipy.stats.norm.isf(alpha / 2))
        est = x.beta.copy()
        lb = x.beta - crit * x.se
        ub = x.beta + crit * x.se
        if transf is not None:
            est, lb, ub = (metafopy.transf.apply_transf(a, transf, targs, x.tau2) for a in (est, lb, ub))
        lb, ub = numpy.minimum(lb, ub), numpy.maximum(lb, ub)
        res_fixed = pandas.DataFrame({'estimate': est, 'ci.lb': lb, 'ci.ub': ub}, index=list(x.coef_names))

    return ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign, digits=ndig)


def control_values(control: collections.abc.Mapping[str, typing.Any] | None, current: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """The settings in ``current`` with the entries of metafor's ``control`` list applied (``tau2.min`` and ``tau2_min`` both work).

    Raises:
        metafopy.errors.InputError: ``control`` has an entry that is not one of the settings.
    """
    out = dict(current)
    for key, value in (control or {}).items():
        name = key.replace('.', '_')
        if name not in out:
            if metafopy.models.control_names.read_elsewhere(key):
                continue
            raise metafopy.errors.InputError(f'unknown control parameter {key!r}; the settings are {", ".join(out)}')
        out[name] = value
    return out


def find_root(f: collections.abc.Callable[[float], float], lo: float, hi: float, tol: float, maxiter: int, *, check_conv: bool = False) -> tuple[float, bool]:
    """``try(uniroot(...)$root)``: ``(root, converged)``, or ``(nan, False)`` on error."""
    try:
        res = metafopy.core.zeroin.uniroot(f, lo, hi, tol=tol, maxiter=maxiter)
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError, metafopy.errors.EstimationError:
        return math.nan, False
    if check_conv and not res.converged:
        return math.nan, False
    return res.root, True


def try_profile(f: collections.abc.Callable[[float], float], v: float) -> float:
    try:
        return f(v)
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError, metafopy.errors.EstimationError:
        return math.nan


def qe_log(tau2: float, y: FloatArray, vi: FloatArray, X: FloatArray, objective: float, verbose: int, digits: int) -> float:
    """metafor's ``.QE.func``: ``QE(tau2) - objective``, logged when ``verbose``."""
    return metafopy.models.tau2.log_objective(tau2, qe(tau2, y, vi, X) - objective, '  RSS - objective =', verbose, digits)


def loglik(res: metafopy.models.rma_uni.RMAResult) -> float:
    return res.fit_stats_reml.ll if res.method == 'REML' else res.fit_stats_ml.ll


def profile_lrt(x: metafopy.models.rma_uni.RMAResult, tau2: float) -> float:
    """``-2 (ll(tau2) - ll(tau2_hat))`` from a refit with tau^2 fixed (metafor's ``.profile.rma.uni``)."""
    md = metafopy.metadata.MetaData.from_inputs(x.yi, x.vi, weights=x.weights, mods=x.X, intercept=False)
    md = dataclasses.replace(md, X_names=tuple(x.coef_names))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.models.rma_uni.fit_rma_uni(
            md,
            method=x.method,
            weighted=x.weighted,
            test=x.test,
            level=1 - x.level,
            control=x.control,
            tau2=tau2,
            skip_r2=True,
        )
    return -2 * (loglik(res) - loglik(x))


def mv_search_limits(x: metafopy.models.mv_result.RMAMVResult, comp: str, vc: float) -> tuple[float, float]:
    """Default interval ``(vc.min, vc.max)`` searched for the bounds of a variance component's profile-likelihood CI."""
    eps = float(numpy.finfo(float).eps)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if comp in ('sigma2', 'tau2', 'gamma2'):
        return 0.0, max(10.0 if vc <= math.sqrt(eps) else max(10.0, vc * 100), 0.0)
    struct = x.struct[0] if comp == 'rho' else x.struct[1]
    lo, hi = -1.0, 1.0
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if struct in ('AR', 'HAR', 'CAR'):
        lo = min(0.0, vc)
    if struct in SPATIAL[:5]:
        lo, hi = 0.0, max(10.0, vc * 10)
    if struct in SPATIAL[5:]:
        lo, hi = 0.0, max(2.0, vc * 2)
    return lo, hi


def mv_refit_logLik(x: metafopy.models.mv_result.RMAMVResult, comp: str, pos: int, val: float) -> float:
    """Log-likelihood of ``x`` refitted with one variance component fixed at ``val`` (``.profile.rma.mv``)."""
    return mv_refit_fixed(x, comp, pos, val).logLik()


def mv_refit_fixed(x: metafopy.models.mv_result.RMAMVResult, comp: str, pos: int, val: float) -> metafopy.models.mv_result.RMAMVResult:
    """``x`` refitted with one variance component fixed at ``val`` and the others as in ``x`` (``.profile.rma.mv``)."""
    spec = dict(x.spec)
    # metafor's .profile.rma.mv() passes 'vccon' on to the refits but not 'optbeta', 'beta', 'lambda1/2' or 'cvvc'
    for dropped in ('optbeta', 'beta', 'lambda1', 'lambda2', 'cvvc', 'restart'):
        spec.pop(dropped, None)
    for name in metafopy.models.mv_structs.VC_NAMES:
        flags = x.vc_fix[name]
        if flags is None:
            continue
        current = numpy.asarray(getattr(x, name), dtype=numpy.float64)
        arg = numpy.where(flags, current, numpy.nan)
        if name == comp:
            arg[pos] = val
        spec[name] = arg
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.models.rma_mv.rma_mv(**spec)
        # metafor's nlminb and SciPy's BFGS can stop at different local maxima of the restricted likelihood (BFGS from the
        # default starting values falls short on some BLAS builds); the profile is the maximum over the free components, so
        # a second refit started at the estimates of x is tried and the higher log-likelihood is kept
        warm = metafopy.models.rma_mv.make_control(spec.get('control'))
        inits: dict[str, typing.Any] = {
            f'{name}_init': numpy.asarray(getattr(x, name), dtype=numpy.float64) for name in metafopy.models.mv_structs.VC_NAMES if x.vc_fix[name] is not None
        }
        if not inits:
            return fit
        try:
            alt = metafopy.models.rma_mv.rma_mv(**(spec | {'control': dataclasses.replace(warm, **inits)}))
        except metafopy.errors.MetaforError:
            return fit
        return alt if alt.logLik() > fit.logLik() else fit


@confint.register
def confint_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    *,
    fixed: bool = False,
    sigma2: int | None = None,
    tau2: int | None = None,
    rho: int | None = None,
    gamma2: int | None = None,
    phi: int | None = None,
    level: float | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    vc_min: float | None = None,
    vc_max: float | None = None,
    tol: float = metafopy.core.zeroin.DEFAULT_TOL,
    maxiter: int = 1000,
    eptries: int = 10,
    verbose: bool | int = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
) -> ConfintResult | ConfintList:
    """CIs for an ``rma.mv`` fit (metafor's ``confint.rma.mv``).

    ``sigma2``, ``tau2``, ``rho``, ``gamma2`` or ``phi`` (a 0-based position; only one may be given) request a
    profile-likelihood CI (Hardy & Thompson, 1996, doi:10.1002/(SICI)1097-0258(19960430)15:8<619::AID-SIM188>3.0.CO;2-A)
    for that component; ``fixed=True`` gives Wald-type CIs for the coefficients. With neither, a profile-likelihood CI is
    computed for every estimated variance/correlation component and a :class:`ConfintList` (a list of results) is returned (a single result if there is one).
    ``verbose`` prints the profile steps of the root finding on stdout, with ``digits`` decimals. ``control`` is metafor's list of settings
    (``vc_min``, ``vc_max``, ``tol``, ``maxiter`` and ``eptries``, written with ``.`` or ``_``); its entries replace the keyword arguments of the same name.
    ``xlim`` (metafor's ``...`` argument) is the interval ``(vc_min, vc_max)`` searched for the bounds and wins over ``control``.

    Raises:
        metafopy.errors.InputError: an argument is invalid or the model has no such component.
    """
    x = model
    vlevel = int(verbose)
    ndig = metafopy.mfopt.option_digits(digits, x.digits)
    con = control_values(control, {'vc_min': vc_min, 'vc_max': vc_max, 'tol': tol, 'maxiter': maxiter, 'eptries': eptries})
    vc_min, vc_max, tol, maxiter, eptries = con['vc_min'], con['vc_max'], con['tol'], con['maxiter'], con['eptries']
    if xlim is not None:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        vc_min, vc_max = float(xlim[0]), float(xlim[1])
    alpha = metafopy.models.rma_uni.level_to_alpha(x.level if level is None else level)
    chosen = {name: v for name, v in zip(metafopy.models.mv_structs.VC_NAMES, (sigma2, tau2, rho, gamma2, phi), strict=True) if v is not None}
    if len(chosen) > 1:
        raise metafopy.errors.InputError("Must specify only one of the arguments 'sigma2', 'tau2', 'rho', 'gamma2', or 'phi'.")

    if not fixed and not chosen:
        results: list[ConfintResult] = []
        for name in metafopy.models.mv_structs.VC_NAMES:
            flags = x.vc_fix[name]
            if flags is None:
                continue
            for index in numpy.flatnonzero(~flags):
                if vlevel:
                    metafopy.core.verbose.cat(f'\nObtaining CI for {name} = {index + 1} \n')
                results.append(
                    typing.cast(
                        ConfintResult,
                        confint_rma_mv(
                            x,
                            level=level,
                            **typing.cast(dict[str, typing.Any], {name: int(index)}),
                            vc_min=vc_min,
                            vc_max=vc_max,
                            tol=tol,
                            maxiter=maxiter,
                            eptries=eptries,
                            verbose=verbose,
                            digits=digits,
                        ),
                    )
                )
        if not results:
            raise metafopy.errors.InputError('No components for which a CI can be obtained.')
        return results[0] if len(results) == 1 else ConfintList(results, ndig)

    res_random: pandas.DataFrame | None = None
    lb_sign = ub_sign = ''
    if chosen:
        comp, pos = next(iter(chosen.items()))
        flags = x.vc_fix[comp]
        if flags is None or bool(numpy.all(flags)):
            raise metafopy.errors.InputError(f"Model does not contain any (estimated) '{comp}' components.")
        if not 0 <= pos < len(flags):
            raise metafopy.errors.InputError(f"No such '{comp}' component in the model.")
        if flags[pos]:
            raise metafopy.errors.InputError(f"Specified '{comp}' component was fixed.")
        vc = float(numpy.asarray(getattr(x, comp), dtype=numpy.float64)[pos])
        lo, hi = mv_search_limits(x, comp, vc)
        lo = lo if vc_min is None else vc_min
        hi = hi if vc_max is None else vc_max
        if lo > vc:
            raise metafopy.errors.InputError('Lower bound of interval to be searched must be <= estimated value of component.')
        if hi < vc:
            raise metafopy.errors.InputError('Upper bound of interval to be searched must be >= estimated value of component.')
        objective = float(scipy.stats.chi2.isf(alpha, 1))
        ll_obj = x.logLik()

        def diff(val: float) -> float:
            try:
                out = -2 * (mv_refit_logLik(x, comp, pos, val) - ll_obj) - objective
            except ValueError, ArithmeticError, numpy.linalg.LinAlgError, metafopy.errors.MetaforError:
                if vlevel:
                    metafopy.core.verbose.cat(f'val = {metafopy.formatters.fmtx(val, ndig, addwidth=4)}   LRT - objective = NA \n')
                raise
            if vlevel:
                metafopy.core.verbose.cat(
                    f'val = {metafopy.formatters.fmtx(val, ndig, addwidth=4)}   LRT - objective = {metafopy.formatters.fmtx(out, ndig, addwidth=4)} \n'
                )
            return out

        struct = x.struct[0] if comp == 'rho' else x.struct[1]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        spatial = comp in ('rho', 'phi') and struct in SPATIAL
        vc_lb = vc_ub = math.nan
        lb_conv = ub_conv = False

        step = abs(lo - vc) / eptries
        for _ in range(eptries):
            res = try_profile(diff, lo)
            if not math.isnan(res):
                if res < 0:
                    vc_lb, lb_conv = lo, True
                    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                    if (comp in ('sigma2', 'tau2', 'gamma2') and lo > 0) or (comp in ('rho', 'phi') and lo > -1) or (spatial and lo > 0):
                        lb_sign = '<'
                else:
                    root, ok = find_root(diff, lo, vc, tol, maxiter, check_conv=True)
                    if ok:
                        vc_lb, lb_conv = root, True
                break
            lo += step

        if vlevel:
            metafopy.core.verbose.cat('\n')

        step = abs(hi - vc) / eptries
        for _ in range(eptries):
            res = try_profile(diff, hi)
            if not math.isnan(res):
                if res < 0:
                    vc_ub, ub_conv = hi, True
                    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                    if comp in ('sigma2', 'tau2', 'gamma2') or (comp in ('rho', 'phi') and hi < 1) or spatial:
                        ub_sign = '>'
                else:
                    root, ok = find_root(diff, vc, hi, tol, maxiter, check_conv=True)
                    if ok:
                        vc_ub, ub_conv = root, True
                break
            hi -= step

        if not lb_conv:
            warnings.warn('Cannot obtain lower bound of profile likelihood CI due to convergence problems.', stacklevel=2)
        if not ub_conv:
            warnings.warn('Cannot obtain upper bound of profile likelihood CI due to convergence problems.', stacklevel=2)

        values = numpy.array([vc, vc_lb, vc_ub])
        n_comp = len(flags)
        suffix = '' if n_comp == 1 else f'.{pos + 1}'
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if comp in ('sigma2', 'tau2', 'gamma2'):
            stem = {'sigma2': ('sigma^2', 'sigma'), 'tau2': ('tau^2', 'tau'), 'gamma2': ('gamma^2', 'gamma')}[comp]
            root_vals = numpy.sqrt(numpy.where(values >= 0, values, numpy.nan))
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            res_random = pandas.DataFrame([values, root_vals], index=[stem[0] + suffix, stem[1] + suffix], columns=['estimate', 'ci.lb', 'ci.ub'])
        else:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            res_random = pandas.DataFrame([values], index=[comp + suffix], columns=['estimate', 'ci.lb', 'ci.ub'])

    res_fixed: pandas.DataFrame | None = None
    if fixed:
        if x.test == 'z':
            crit = numpy.full(x.p, float(scipy.stats.norm.isf(alpha / 2)))
        else:
            crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else math.nan for d in x.ddf])
        beta, lb, ub = x.beta, x.beta - crit * x.se, x.beta + crit * x.se
        if transf is not None:
            beta, lb, ub = (metafopy.transf.apply_transf(a, transf, targs, 0.0) for a in (beta, lb, ub))
        lo_b, hi_b = numpy.minimum(lb, ub), numpy.maximum(lb, ub)
        res_fixed = pandas.DataFrame({'estimate': beta, 'ci.lb': lo_b, 'ci.ub': hi_b}, index=list(x.coef_names))

    return ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign, digits=ndig)
