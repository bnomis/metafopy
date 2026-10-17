"""``predict``, ``confint``, ``profile`` and ``anova`` for selection models (:class:`~metafopy.models.sel_result.SelmodelResult`).

Ports of ``predict.rma``, ``confint.rma.uni.selmodel``, ``profile.rma.uni.selmodel`` and ``anova.rma`` as metafor 5.2-1
applies them to ``rma.uni.selmodel`` objects. The profile-likelihood intervals and the profiles refit the selection
model with one component fixed, see https://wviechtb.github.io/metafor/reference/selmodel.html and
https://wviechtb.github.io/metafor/reference/confint.rma.html.
"""

import collections.abc
import concurrent.futures
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.parallel
import metafopy.core.timing
import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.common
import metafopy.methods.confint
import metafopy.methods.fitted
import metafopy.methods.fitview
import metafopy.methods.predict
import metafopy.methods.profile
import metafopy.models.control_names
import metafopy.models.rma_uni
import metafopy.models.sel_result
import metafopy.models.selmodel
import metafopy.models.tau2
import metafopy.printing
import metafopy.transf

__all__ = ['SelProfileResult', 'anova', 'confint', 'predict', 'profile']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Sel = metafopy.models.sel_result.SelmodelResult

NOT_IMPLEMENTED = 'Method not currently implemented for this type of model.'


def check_supported(x: Sel, what: str) -> None:
    if x.betaspec:
        raise metafopy.errors.InputError(f'Cannot {what} when one or more beta values were fixed.')
    if x.decreasing or x.type == 'stepcon':
        raise metafopy.errors.NotSupportedError(NOT_IMPLEMENTED)


def has_tau2(x: Sel) -> bool:
    return x.method not in metafopy.models.tau2.FIXED_EFFECTS and not x.tau2_fix


def as_rma(x: Sel) -> metafopy.models.rma_uni.RMAResult:
    """The underlying model with the estimates of the selection model (normal-theory tests), for the methods that only need those."""
    return dataclasses.replace(x.base, beta=x.beta, se=x.se, vb=x.vb, tau2=x.tau2, test='z', ddf=None, robust=None)


# -- predict ---------------------------------------------------------------------------------------------


def predict(
    model: Sel,
    newmods: numpy.typing.ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
    verbose: bool = False,
) -> metafopy.methods.predict.PredictResult:
    """Predicted values under the selection model (metafor's ``predict.rma``); the arguments are those of :func:`metafopy.predict`.

    The coefficients, their covariance matrix and ``tau^2`` are the estimates of the selection model; intervals use the
    normal distribution. The selection model is fitted to the studies without missing values, so ``na_action`` makes no difference to the fitted values.
    """
    base = dataclasses.replace(as_rma(model), parms=model.parms)
    return metafopy.methods.predict.with_digits(
        metafopy.methods.predict.location_prediction(
            metafopy.methods.predict.uni_location_model(base, metafopy.methods.fitted.rows_of(model)),
            newmods,
            intercept,
            transf,
            targs,
            level,
            hetvar=hetvar,
            prob=prob,
            addx=addx,
            adjust=adjust,
            vcov=vcov,
            predtype=predtype,
            pi_type=pi_type,
            newvi=newvi,
            na_action=na_action,
        ),
        digits,
        model.digits,
    )


# -- refits ----------------------------------------------------------------------------------------------


def refit(obj: Sel, comp: str, val: float, delta_pos: int, *, confint_: bool) -> Sel | None:
    """metafor's ``.profile.rma.uni.selmodel``: the selection model with one component fixed at ``val`` (``None`` if the fit fails)."""
    tau2_arg = obj.tau2 if not has_tau2(obj) else math.nan
    delta_arg = numpy.where(numpy.asarray(obj.delta_fix), obj.delta, math.nan).astype(numpy.float64)
    if comp == 'tau2':
        tau2_arg = val
    else:
        delta_arg[delta_pos] = val
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = metafopy.models.selmodel.selmodel(
                obj,
                obj.type,
                alternative=obj.alternative,
                prec=obj.prec,
                scaleprec=obj.scaleprec,
                tau2=tau2_arg,
                delta=delta_arg,
                steps=obj.steps if obj.stepsspec else None,
                decreasing=obj.decreasing,
                verbose=False,
                control=obj.control,
                skiphes=confint_,
                skiphet=True,
                defmap=obj.defmap,
                mapfun=obj.mapfun,
                mapinvfun=obj.mapinvfun,
            )
    except metafopy.errors.MetaforError, ArithmeticError, numpy.linalg.LinAlgError:
        return None
    assert isinstance(fit, metafopy.models.sel_result.SelmodelResult)
    return fit


def component(x: Sel, tau2: bool | None, delta: int | None) -> tuple[str, float, int]:
    """Validate the ``tau2``/``delta`` arguments: the component name, its estimate and the 0-based position of a ``delta``.

    Raises:
        metafopy.errors.InputError: the component does not exist or was not estimated.
    """
    if tau2 is not None and delta is not None:
        raise metafopy.errors.InputError("Must specify only one of the 'tau2' or 'delta' arguments.")
    if tau2 is not None:
        if not has_tau2(x):
            raise metafopy.errors.InputError("Model does not contain an (estimated) 'tau2' component.")
        if not tau2:
            raise metafopy.errors.InputError("No such 'tau2' component in the model.")
        return 'tau2', x.tau2, -1
    assert delta is not None
    if all(x.delta_fix):
        raise metafopy.errors.InputError("Model does not contain any estimated 'delta' components.")
    if isinstance(delta, bool) or not isinstance(delta, int | numpy.integer):
        raise metafopy.errors.InputError("Must specify a number for the 'delta' component.")
    if delta < 0 or delta >= x.deltas:
        raise metafopy.errors.InputError("No such 'delta' component in the model.")
    if x.delta_fix[delta]:
        raise metafopy.errors.InputError("Specified 'delta' component was fixed.")
    return 'delta', float(x.delta[delta]), int(delta)


# -- confint ---------------------------------------------------------------------------------------------

CONFINT_CONTROL = ('eptries', 'maxiter', 'tol', 'vc_max', 'vc_min', 'verbose')
# auditme: accept[literal-duplication] the columns of metafor's interval tables
CI_COLUMNS = ['estimate', 'ci.lb', 'ci.ub']  # auditme: accept[non-alphabetical-literal] metafor's column order
TAU2_NAME = 'tau^2'


@metafopy.core.timing.timed
def confint(
    model: Sel,
    *,
    fixed: bool = False,
    tau2: bool | None = None,
    delta: int | None = None,
    level: float | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    verbose: bool = False,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    extint: bool = False,
    digits: int | None = None,
    parm: typing.Any = None,
) -> metafopy.methods.confint.ConfintResult | metafopy.methods.confint.ConfintList:
    """Confidence intervals for the coefficients (``fixed=True``, Wald-type) and for ``tau^2`` or a selection parameter (profile likelihood).

    Without ``fixed``, ``tau2`` and ``delta`` the intervals of every estimated variance and selection parameter are
    returned as a list (a single result if there is just one). ``tau2=True`` or ``delta=j`` (0-based) select one
    component. ``control`` takes ``tol``, ``maxiter``, ``verbose``, ``eptries``, ``vc_min`` and ``vc_max``; ``xlim`` sets
    the interval searched and ``extint`` lets the search interval grow (R's ``extendInt``). ``digits`` is the number of decimals of the printout and
    ``parm`` is ignored with a warning, as in metafor.

    Raises:
        metafopy.errors.InputError: an argument or the model is not suitable.
        metafopy.errors.NotSupportedError: the model is a decreasing step function or a ``'stepcon'`` model.
    """
    metafopy.methods.confint.warn_parm(parm)
    x = model
    check_supported(x, 'obtain confidence intervals')
    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    ndig = metafopy.mfopt.option_digits(digits, x.digits)
    ctrl = dict(control or {})
    if xlim is not None:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        ctrl['vc_min'], ctrl['vc_max'] = xlim[0], xlim[1]

    random = tau2 is not None or delta is not None
    if not fixed and not random:
        comps = int(has_tau2(x)) + int(sum(not f for f in x.delta_fix))
        if comps == 0:
            raise metafopy.errors.InputError('No components for which a CI can be obtained.')
        out: list[metafopy.methods.confint.ConfintResult] = []
        if has_tau2(x):
            if verbose:
                print('\nObtaining CI for tau2\n')
            out.append(confint_one(x, False, True, None, alpha, ctrl, transf, targs, verbose, extint, ndig))
        for pos in range(x.deltas):
            if x.delta_fix[pos]:
                continue
            if verbose:
                print(f'\nObtaining CI for delta = {pos + 1}\n')
            out.append(confint_one(x, False, None, pos, alpha, ctrl, transf, targs, verbose, extint, ndig))
        return out[0] if len(out) == 1 else metafopy.methods.confint.ConfintList(out)
    return confint_one(x, fixed, tau2, delta, alpha, ctrl, transf, targs, verbose, extint, ndig)


def confint_one(
    x: Sel,
    fixed: bool,
    tau2: bool | None,
    delta: int | None,
    alpha: float,
    ctrl: collections.abc.Mapping[str, typing.Any],
    transf: collections.abc.Callable[..., typing.Any] | None,
    targs: typing.Any,
    verbose: bool,
    extint: bool,
    ndig: int,
) -> metafopy.methods.confint.ConfintResult:
    res_random: pandas.DataFrame | None = None
    lb_sign = ub_sign = ''
    if tau2 is not None or delta is not None:
        res_random, lb_sign, ub_sign = profile_ci(x, tau2, delta, alpha, ctrl, verbose=verbose, extint=extint)

    res_fixed: pandas.DataFrame | None = None
    if fixed:
        crit = float(scipy.stats.norm.isf(alpha / 2))
        est, lb, ub = x.beta, x.beta - crit * x.se, x.beta + crit * x.se
        if transf is not None:
            est, lb, ub = (metafopy.transf.apply_transf(a, transf, targs, x.tau2) for a in (est, lb, ub))
        lb, ub = numpy.minimum(lb, ub), numpy.maximum(lb, ub)
        res_fixed = pandas.DataFrame({'estimate': est, 'ci.lb': lb, 'ci.ub': ub}, index=list(x.coef_names))
    return metafopy.methods.confint.ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign, digits=ndig)


def profile_ci(
    x: Sel,
    tau2: bool | None,
    delta: int | None,
    alpha: float,
    control: collections.abc.Mapping[str, typing.Any],
    *,
    verbose: bool,
    extint: bool,
) -> tuple[pandas.DataFrame, str, str]:
    """The profile-likelihood interval of one component and the ``<``/``>`` signs marking a bound that hit the search limit.

    Raises:
        metafopy.errors.InputError: the component or the search interval is invalid.
    """
    comp, vc, pos = component(x, tau2, delta)
    eps_sqrt = math.sqrt(2.220446049250313e-16)
    con: dict[str, typing.Any] = {'tol': metafopy.core.zeroin.DEFAULT_TOL, 'maxiter': 1000, 'verbose': False, 'eptries': 10}
    if comp == 'tau2':
        con['vc_min'] = 0.0
        con['vc_max'] = min(max(10.0 if vc <= eps_sqrt else max(10.0, vc * 100), con['vc_min']), x.tau2_max)
    else:
        con['vc_min'] = max(0.0, float(x.delta_min[pos]))
        con['vc_max'] = min(max(10.0 if vc <= eps_sqrt else max(10.0, vc * 10), con['vc_min']), float(x.delta_max[pos]))
    for key, value in control.items():
        name = key.replace('.', '_')
        if name not in CONFINT_CONTROL:
            if metafopy.models.control_names.read_elsewhere(key):
                continue
            raise metafopy.errors.InputError(f'unknown control element {key!r}')
        con[name] = value
    if verbose:
        con['verbose'] = True
    show = bool(con['verbose'])
    vc_min, vc_max = float(con['vc_min']), float(con['vc_max'])
    if vc_min > vc:
        raise metafopy.errors.InputError('Lower bound of interval to be searched must be <= estimated value of component.')
    if vc_max < vc:
        raise metafopy.errors.InputError('Upper bound of interval to be searched must be >= estimated value of component.')

    objective = float(scipy.stats.chi2.isf(alpha, 1))
    ll_hat = x.logLik()

    def lrt(val: float, check: bool = False) -> float:
        fit = refit(x, comp, val, pos, confint_=True)
        if fit is None:
            if show:
                print(f'val = {val:.4f}  LRT - objective = NA')
            raise metafopy.errors.EstimationError('refit failed')
        out = -2 * (fit.logLik() - ll_hat) - objective
        if show:
            print(f'val = {val:.4f}  LRT - objective = {out:.4f}')
        return out

    def root(lo: float, hi: float, direction: typing.Literal['downX', 'upX']) -> float | None:
        try:
            res = metafopy.core.zeroin.uniroot(lrt, lo, hi, tol=float(con['tol']), maxiter=int(con['maxiter']), extend_int=direction if extint else 'no')
        except metafopy.errors.MetaforError, ArithmeticError, numpy.linalg.LinAlgError:
            return None
        return res.root if res.converged else None

    def first_valid(start: float, step: float) -> tuple[float, float] | None:
        """Walk from ``start`` in steps of ``step`` until the profile LRT can be evaluated (R's ``eptries`` loop)."""
        val = start
        for _ in range(int(con['eptries'])):
            try:
                res = lrt(val)
            except metafopy.errors.MetaforError, ArithmeticError, numpy.linalg.LinAlgError:
                res = math.nan
            if not math.isnan(res):
                return val, res
            val += step
        return None

    vc_lb = vc_ub = math.nan
    lb_conv = ub_conv = False
    lb_sign = ub_sign = ''
    epdiff = abs(vc_min - vc) / int(con['eptries'])
    hit = first_valid(vc_min, epdiff)
    if hit is not None:
        low, res = hit
        if not extint and res < 0:
            vc_lb, lb_conv = low, True
            if low > 0:
                lb_sign = '<'
        else:
            found = root(low, vc, 'downX')
            if found is not None:
                vc_lb, lb_conv = found, True
    if show:
        print()
    epdiff = abs(vc_max - vc) / int(con['eptries'])
    hit = first_valid(vc_max, -epdiff)
    if hit is not None:
        high, res = hit
        if not extint and res < 0:
            vc_ub, ub_conv = high, True
            ub_sign = '>'
        else:
            found = root(vc, high, 'upX')
            if found is not None:
                vc_ub, ub_conv = found, True
    if not lb_conv:
        warnings.warn('Cannot obtain lower bound of profile likelihood CI due to convergence problems.', stacklevel=3)
    if not ub_conv:
        warnings.warn('Cannot obtain upper bound of profile likelihood CI due to convergence problems.', stacklevel=3)

    values = numpy.array([vc, vc_lb, vc_ub])
    if comp == 'tau2':
        with numpy.errstate(invalid='ignore'):
            sq = numpy.sqrt(numpy.where(values >= 0, values, numpy.nan))
        table = pandas.DataFrame([values, sq], index=[TAU2_NAME, 'tau'], columns=CI_COLUMNS)
    else:
        table = pandas.DataFrame([values], index=['delta' if x.deltas == 1 else f'delta.{pos + 1}'], columns=CI_COLUMNS)
    return table, lb_sign, ub_sign


# -- profile ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SelProfileResult(metafopy.methods.profile.ProfileResult):
    """Profile log-likelihood of ``tau^2`` or of one selection parameter.

    As :class:`~metafopy.ProfileResult`, with ``tau2`` holding the grid of the profiled component (``comp`` is
    ``'tau2'`` or ``'delta'``; ``delta_pos`` the 0-based position of the selection parameter and ``deltas`` their
    number). ``I2`` and ``H2`` are empty.
    """

    comp: str = 'tau2'
    delta_pos: int = 0
    deltas: int = 1

    @property
    def grid(self) -> FloatArray:
        return self.tau2

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({self.comp: self.tau2, 'll': self.ll})


@metafopy.core.timing.timed
def profile(
    fitted: Sel,
    *,
    tau2: bool | None = None,
    delta: int | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    steps: int | numpy.typing.ArrayLike = 20,
    lltol: float = 0.001,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    plot: bool = False,
    exp: bool = False,
    ax: typing.Any = None,
    LB: bool = False,
    **plot_args: typing.Any,
) -> SelProfileResult | list[SelProfileResult]:
    """Profile the likelihood of ``tau^2`` (``tau2=True``) or of a selection parameter (``delta=j``, 0-based).
    ``LB`` is accepted as in metafor and changes no result.

    Without ``tau2`` and ``delta`` every estimated component is profiled and a list is returned (a single result if
    there is just one). ``steps`` is the number of equally spaced values in ``xlim`` or the values themselves.
    ``plot=True`` draws the profile with :func:`metafopy.plot.plot_profile`. Further keyword arguments (``pch``, ``cline``, ``refline``, ``xlab``,
    ``ylab``, ``main``, ``log``, ``cex``, ``lwd``) are passed on to :func:`metafopy.plot.plot_profile`, as metafor's ``...`` is to
    ``plot.profile.rma``.

    As in metafor, ``parallel`` (``'snow'`` or ``'multicore'``) fits the models for the grid values in a process pool of ``ncpus``
    workers, or on the :class:`concurrent.futures.Executor` given as ``cl`` (``ncpus > 1`` implies ``'snow'``); the fits are
    independent and deterministic, so the result equals that of the serial run. ``progbar=True`` prints a progress bar to stderr.
    (``confint`` has no such arguments in metafor: its profile-likelihood searches are sequential.)

    Raises:
        metafopy.errors.InputError: an argument or the model is not suitable.
        metafopy.errors.NotSupportedError: the model is a decreasing step function or a ``'stepcon'`` model.
    """
    x = fitted
    check_supported(x, 'profile')
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    pool: dict[str, typing.Any] = {'progbar': progbar, 'parallel': parallel, 'ncpus': ncpus, 'cl': cl}
    st = numpy.atleast_1d(numpy.asarray(steps, dtype=numpy.float64))
    if numpy.isnan(st).any():
        raise metafopy.errors.InputError("No missing values allowed in 'steps' argument.")
    step_seq = st.shape[0] >= 2
    if step_seq:
        if xlim is None:
            xlim = (float(st.min()), float(st.max()))
    elif st[0] < 2:
        raise metafopy.errors.InputError("Argument 'steps' must be >= 2.")

    if tau2 is None and delta is None:
        comps = int(has_tau2(x)) + int(sum(not f for f in x.delta_fix))
        if comps == 0:
            raise metafopy.errors.InputError('No components in the model for which a profile likelihood can be constructed.')
        sav: list[SelProfileResult] = []
        if has_tau2(x):
            sav.append(profile_one(x, 'tau2', -1, xlim, ylim, st, step_seq, lltol, exp, **pool))
        sav.extend(profile_one(x, 'delta', pos, xlim, ylim, st, step_seq, lltol, exp, **pool) for pos in range(x.deltas) if not x.delta_fix[pos])
        if plot:
            plot_all(sav, ax, plot_args)
        return sav[0] if len(sav) == 1 else sav

    comp, _, pos = component(x, tau2, delta)
    out = profile_one(x, comp, pos, xlim, ylim, st, step_seq, lltol, exp, **pool)
    if plot:
        plot_all([out], ax, plot_args)
    return out


def plot_all(results: list[SelProfileResult], ax: typing.Any, plot_args: collections.abc.Mapping[str, typing.Any]) -> None:
    # auditme: accept[in-function-import] metafopy.plot.profile imports the methods, so the import is deferred to avoid a cycle
    import metafopy.plot.profile

    metafopy.plot.profile.plot_profile(results[0] if len(results) == 1 else results, ax=ax, **plot_args)


def profile_fit(x: Sel, comp: str, pos: int, val: float) -> tuple[float, FloatArray, FloatArray, FloatArray] | None:
    """The log-likelihood and coefficients (estimates and CI bounds) of the model with one component fixed at ``val``."""
    fit = refit(x, comp, val, pos, confint_=False)
    if fit is None:
        return None
    return fit.logLik(), fit.beta, fit.ci_lb, fit.ci_ub


def profile_one(
    x: Sel,
    comp: str,
    pos: int,
    xlim: collections.abc.Sequence[float] | None,
    ylim: collections.abc.Sequence[float] | None,
    st: FloatArray,
    step_seq: bool,
    lltol: float,
    exp: bool,
    *,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
) -> SelProfileResult:
    crit = float(scipy.stats.norm.ppf(0.995))
    if comp == 'tau2':
        vc = x.tau2
        lo_bound, hi_bound = 0.0, x.tau2_max
        se = x.se_tau2
    else:
        vc = float(x.delta[pos])
        lo_bound, hi_bound = float(x.delta_min[pos]), float(x.delta_max[pos])
        se = float(x.se_delta[pos])
    if xlim is None:
        if math.isnan(se):
            vc_lb = max(0.0, vc / 4, lo_bound)
            vc_ub = min(max(0.1, vc * 4), hi_bound)
        else:
            vc_lb = max(0.0, vc - crit * se, lo_bound)
            vc_ub = min(max(0.1, vc + crit * se), hi_bound)
        if math.isnan(vc_lb) or math.isnan(vc_ub):
            raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")
        xl = (vc_lb, vc_ub)
    else:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        xl = (float(min(xlim)), float(max(xlim)))
        if xl[0] < lo_bound:
            raise metafopy.errors.InputError(f'Lower bound for profiling must be >= {lo_bound:g}.')
        if comp == 'delta' and xl[1] > hi_bound:
            raise metafopy.errors.InputError(f'Upper bound for profiling must be <= {hi_bound:g}.')
    vcs = st if step_seq else numpy.linspace(xl[0], xl[1], int(st[0]))
    if vcs.shape[0] <= 1:
        raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")

    names = list(x.coef_names)
    lls = numpy.full(len(vcs), numpy.nan)
    beta = numpy.full((len(vcs), x.p), numpy.nan)
    ci_lb = numpy.full_like(beta, numpy.nan)
    ci_ub = numpy.full_like(beta, numpy.nan)
    one = functools.partial(profile_fit, x, comp, pos)
    fits = metafopy.core.parallel.map_sims(one, [float(v) for v in vcs], parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        lls[i], beta[i], ci_lb[i], ci_ub[i] = fit[0], fit[1], fit[2], fit[3]
    maxll = float(x.logLik())
    if numpy.any(lls >= maxll + lltol):
        warnings.warn('At least one profiled log-likelihood value is larger than the log-likelihood of the fitted model.', stacklevel=3)
    if numpy.all(numpy.isnan(lls)):
        warnings.warn('All model fits failed. Cannot draw profile likelihood plot.', stacklevel=3)
    if exp:
        lls = numpy.exp(lls)
        maxll = math.exp(maxll)

    finite = lls[numpy.isfinite(lls)]
    if ylim is None:
        if finite.size:
            yl = (min(maxll, finite.min()), max(maxll, finite.max())) if xl[0] <= vc <= xl[1] else (finite.min(), finite.max())
        else:
            yl = (maxll, maxll)
        if not exp:
            yl = (yl[0] - 0.1, yl[1] + 0.1)
    else:
        if len(ylim) != 2:
            raise metafopy.errors.InputError("Argument 'ylim' should be a vector of length 2.")
        yl = (float(min(ylim)), float(max(ylim)))
    empty = numpy.full(len(vcs), numpy.nan)
    return SelProfileResult(
        tau2=vcs,
        ll=lls,
        beta=pandas.DataFrame(beta, columns=names),
        ci_lb=pandas.DataFrame(ci_lb, columns=names),
        ci_ub=pandas.DataFrame(ci_ub, columns=names),
        xlim=xl,
        ylim=(float(yl[0]), float(yl[1])),
        method=x.method,
        vc=vc,
        maxll=maxll,
        I2=empty,
        H2=empty,
        exp=exp,
        sqrt=False,
        comp=comp,
        delta_pos=max(pos, 0),
        deltas=x.deltas,
    )


# -- anova -----------------------------------------------------------------------------------------------


def anova(
    model: Sel,
    model2: Sel | metafopy.models.rma_uni.RMAResult | None = None,
    *,
    object2: Sel | metafopy.models.rma_uni.RMAResult | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    X: numpy.typing.ArrayLike | None = None,
    rhs: numpy.typing.ArrayLike | float = 0.0,
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
) -> metafopy.methods.anova.WaldResult | metafopy.methods.anova.LRTResult | metafopy.methods.anova.ListAnovaResult:
    """Wald-type tests of the coefficients (``btt`` 0-based, or linear combinations ``X``) or, given a second model, a likelihood-ratio test.

    ``adjust`` adjusts the p-values of several tests (see :func:`metafopy.methods.anova.anova_rma_uni`). ``object2`` is R's name for ``model2``.
    ``L``, ``fixed``, ``verbose``, ``abbrev`` and ``digits`` are those of that function; for the likelihood-ratio test of two models ``df`` replaces the
    difference in the number of parameters; ``refit`` and ``test`` are accepted, but the models are compared as fitted (selection models are fitted by ML).

    The likelihood-ratio test compares the ML log-likelihoods of two nested models (selection models or ``rma_uni`` fits
    to the same data).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    model2 = typing.cast(Sel | metafopy.models.rma_uni.RMAResult | None, metafopy.methods.common.resolve_alias(model2, object2, 'model2', 'object2'))
    if model2 is not None:
        return lrt(model, model2, df)
    out = metafopy.methods.anova.anova_rma_uni(
        as_rma(model), btt=btt, X=X, rhs=rhs, adjust=adjust, L=L, fixed=fixed, verbose=verbose, abbrev=abbrev, digits=digits, att=att, Z=Z
    )
    if isinstance(out, metafopy.methods.anova.WaldResult) and out.class_ is not None:
        out = dataclasses.replace(out, class_=metafopy.methods.fitview.r_class(model))
    return out


def lrt(a: Sel, b: Sel | metafopy.models.rma_uni.RMAResult, df_given: int | None = None) -> metafopy.methods.anova.LRTResult:
    """The likelihood-ratio test of two nested models fitted by ML (metafor's ``anova.rma`` with two models).

    Raises:
        metafopy.errors.InputError: the models are not nested fits to the same data.
    """
    if df_given is None and a.parms == b.parms:
        raise metafopy.errors.InputError('models do not differ in the number of parameters')
    full, red = (a, b) if df_given is not None or a.parms > b.parms else (b, a)
    if full.k != red.k or not numpy.allclose(full.yi, red.yi):
        raise metafopy.errors.InputError('models are not fitted to the same data')
    fs_f, fs_r = full.fit_stats_ml, red.fit_stats_ml
    if math.isnan(fs_f.dev) or math.isnan(fs_r.dev):
        raise metafopy.errors.InputError('models must be fitted by ML to be compared')
    LRT = max(0.0, fs_r.dev - fs_f.dev)
    df = full.parms - red.parms if df_given is None else df_given
    # metafor stores QE = NA in a selection model
    qe_f = math.nan if isinstance(full, metafopy.models.sel_result.SelmodelResult) else full.QE
    qe_r = math.nan if isinstance(red, metafopy.models.sel_result.SelmodelResult) else red.QE
    if full.method in metafopy.models.tau2.FIXED_EFFECTS and red.method in metafopy.models.tau2.FIXED_EFFECTS:
        s2f = metafopy.methods.anova.sigma2(full.base if isinstance(full, metafopy.models.sel_result.SelmodelResult) else full)
        s2r = metafopy.methods.anova.sigma2(red.base if isinstance(red, metafopy.models.sel_result.SelmodelResult) else red)
        R2 = 100 * max(0.0, (s2r - s2f) / s2r)
    elif red.tau2 == 0:
        R2 = 0.0
    else:
        R2 = 100 * max(0.0, (red.tau2 - full.tau2) / red.tau2)
    return metafopy.methods.anova.LRTResult(
        df_full=full.parms + (0 if df_given is None else df_given),
        df_reduced=red.parms,
        LRT=LRT,
        pval=float(scipy.stats.chi2.sf(LRT, df)),
        fit_stats_full=fs_f.as_tuple(),
        fit_stats_reduced=fs_r.as_tuple(),
        QE_full=qe_f,
        QE_reduced=qe_r,
        tau2_full=full.tau2,
        tau2_reduced=red.tau2,
        R2=R2,
        class_f=metafopy.methods.fitview.r_class(full),
    )


metafopy.methods.predict.predict.register(metafopy.models.sel_result.SelmodelResult, predict)
metafopy.methods.confint.confint.register(metafopy.models.sel_result.SelmodelResult, confint)
# auditme: accept[type-ignore] singledispatch types every registered method as returning ProfileResult, a selection model may return a list
metafopy.methods.profile.profile.register(metafopy.models.sel_result.SelmodelResult, profile)  # type: ignore[arg-type]
metafopy.methods.anova.anova.register(metafopy.models.sel_result.SelmodelResult, anova)
