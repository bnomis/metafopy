"""Profile likelihood of the variance and correlation components of an ``rma.mv`` fit (metafor's ``profile.rma.mv``).

The model is refitted with one of the ``sigma2``, ``tau2``, ``rho``, ``gamma2`` or ``phi`` components fixed at each value
of a grid (the other components re-estimated) and the (restricted) log-likelihood of each fit is recorded, see
Viechtbauer (2010), doi:10.18637/jss.v036.i03. Ports https://github.com/wviechtb/metafor/blob/master/R/profile.rma.mv.r and
``.profile.rma.mv`` in ``misc.func.hidden.profile.r``; see https://wviechtb.github.io/metafor/reference/profile.rma.html.
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

import metafopy.core.parallel
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.confint
import metafopy.methods.mv_diagnostics
import metafopy.methods.profile
import metafopy.models.mv_result
import metafopy.models.mv_structs

__all__ = ['MVProfileResult', 'profile_rma_mv']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type MV = metafopy.models.mv_result.RMAMVResult
type Fit = tuple[float, FloatArray, FloatArray, FloatArray] | None

VARIANCES = ('gamma2', 'sigma2', 'tau2')
# structures whose correlation parameter is not bounded by 1 (the PHY* structures are disabled in metafor 5.2-1)
NONNEG_CORR = ('CAR', *metafopy.models.mv_structs.SPATIAL)


@dataclasses.dataclass(frozen=True, slots=True)
class MVProfileResult(metafopy.methods.profile.ProfileResult):
    """Profile log-likelihood of one variance or correlation component of an [rma_mv][metafopy.rma_mv] fit. It has the attributes of
    [ProfileResult][metafopy.ProfileResult], with ``tau2`` holding the grid of the profiled component (whatever it is, also through the property
    named like the component: ``sigma2``, ``rho``, ``gamma2`` or ``phi``; the others raise AttributeError); ``I2`` and ``H2`` are NaN.

    Attributes:
        comp: Name of the profiled component: ``'sigma2'``, ``'tau2'``, ``'rho'``, ``'gamma2'`` or ``'phi'``.
        comp_pos: 0-based position of the profiled component.
        comp_n: Number of components of that kind in the model.
    """

    comp: str = 'sigma2'
    comp_pos: int = 0
    comp_n: int = 1

    @property
    def grid(self) -> FloatArray:
        return self.tau2

    def component_grid(self, name: str) -> FloatArray:
        if self.comp != name:
            raise AttributeError(f'this profile is of {self.comp}, not {name}')
        return self.tau2

    @property
    def sigma2(self) -> FloatArray:
        """The grid of a ``sigma2`` profile (metafor stores it under the name of the component)."""
        return self.component_grid('sigma2')

    @property
    def rho(self) -> FloatArray:
        """The grid of a ``rho`` profile."""
        return self.component_grid('rho')

    @property
    def gamma2(self) -> FloatArray:
        """The grid of a ``gamma2`` profile."""
        return self.component_grid('gamma2')

    @property
    def phi(self) -> FloatArray:
        """The grid of a ``phi`` profile."""
        return self.component_grid('phi')

    @property
    def xlab(self) -> str:
        """The label of the x axis of the profile plot."""
        return f'{self.comp} Value'

    @property
    def title(self) -> str:
        """The title of the profile plot."""
        return f'Profile Plot for {self.comp}'

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({self.comp: self.tau2, 'll': self.ll})


def profile_fit(x: MV, comp: str, pos: int, val: float) -> Fit:
    """The log-likelihood and coefficients (estimates and CI bounds) of the model with one component fixed at ``val``."""
    try:
        fit = metafopy.methods.confint.mv_refit_fixed(x, comp, pos, val)
    except metafopy.methods.mv_diagnostics.REFIT_ERRORS:
        return None
    return fit.logLik(), fit.beta, fit.ci_lb, fit.ci_ub


def auto_xlim(x: MV, comp: str, vc: float) -> tuple[float, float]:
    """metafor's default profiling range for a component."""
    if comp in VARIANCES:
        struct = x.struct[0] if comp == 'tau2' else x.struct[1]
        if comp != 'sigma2' and struct in metafopy.models.mv_structs.SPATIAL:
            return max(0.0, vc / 2), max(0.1, vc * 2)
        return max(0.0, vc / 4), max(0.1, vc * 4)
    struct = x.struct[0] if comp == 'rho' else x.struct[1]
    if struct == 'CAR':
        return max(0.0, vc - 0.5), min(0.99999, vc + 0.5)
    if struct in metafopy.models.mv_structs.SPATIAL:
        return vc / 2, vc * 2
    return max(-0.99999, vc - 0.5), min(0.99999, vc + 0.5)


def check_xlim(x: MV, comp: str, xl: tuple[float, float]) -> None:
    """metafor's bounds for a user-supplied profiling range.

    Raises:
        metafopy.errors.InputError: the range extends beyond the parameter space.
    """
    if comp in VARIANCES:
        if xl[0] < 0:
            raise metafopy.errors.InputError('Lower bound for profiling must be >= 0.')
        return
    struct = x.struct[0] if comp == 'rho' else x.struct[1]
    if struct in NONNEG_CORR and xl[0] < 0:
        raise metafopy.errors.InputError('Lower bound for profiling must be >= 0.')
    if xl[0] < -1:
        raise metafopy.errors.InputError('Lower bound for profiling must be >= -1.')
    if struct not in NONNEG_CORR and xl[1] > 1:
        raise metafopy.errors.InputError('Upper bound for profiling must be <= 1.')


def component(x: MV, chosen: dict[str, int | None]) -> tuple[str, int]:
    """The component to profile and its 0-based position, checked as metafor does.

    Raises:
        metafopy.errors.InputError: more than one component is given, or the model has no such (estimated) component.
    """
    given = {name: pos for name, pos in chosen.items() if pos is not None}
    if len(given) > 1:
        raise metafopy.errors.InputError("Must specify only one of the arguments 'sigma2', 'tau2', 'rho', 'gamma2', or 'phi'.")
    comp, pos = next(iter(given.items()))
    flags = x.vc_fix[comp]
    if flags is None or bool(numpy.all(flags)):
        raise metafopy.errors.InputError(f"Model does not contain any (estimated) '{comp}' components.")
    if isinstance(pos, bool):
        raise metafopy.errors.InputError(f"Must specify a number for the '{comp}' component.")
    pos = round(pos)
    if not 0 <= pos < len(flags):
        raise metafopy.errors.InputError(f"No such '{comp}' component in the model.")
    if flags[pos]:
        raise metafopy.errors.InputError(f"Specified '{comp}' component was fixed.")
    return comp, pos


# auditme: accept[duplicate-call-sequence] mirrors profile.rma.mv, which repeats the steps of the other profile methods
def profile_one(
    x: MV,
    comp: str,
    pos: int,
    xlim: collections.abc.Sequence[float] | None,
    ylim: collections.abc.Sequence[float] | None,
    st: FloatArray,
    step_seq: bool,
    lltol: float,
    exp: bool,
    pool: dict[str, typing.Any],
) -> MVProfileResult:
    vc = float(numpy.asarray(getattr(x, comp), dtype=numpy.float64)[pos])
    if xlim is None:
        xl = auto_xlim(x, comp, vc)
        if math.isnan(xl[0]) or math.isnan(xl[1]):
            raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")
    else:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        xl = (float(min(xlim)), float(max(xlim)))
        check_xlim(x, comp, xl)
    vcs = st if step_seq else numpy.linspace(xl[0], xl[1], int(st[0]))
    if vcs.shape[0] <= 1:
        raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")

    names = list(x.coef_names)
    lls = numpy.full(len(vcs), numpy.nan)
    beta = numpy.full((len(vcs), x.p), numpy.nan)
    ci_lb = numpy.full_like(beta, numpy.nan)
    ci_ub = numpy.full_like(beta, numpy.nan)
    one = functools.partial(profile_fit, x, comp, pos)
    fits = metafopy.core.parallel.map_sims(one, [float(v) for v in vcs], **pool)
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
    flags = x.vc_fix[comp]
    return MVProfileResult(
        tau2=numpy.asarray(vcs, dtype=numpy.float64),
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
        comp_pos=pos,
        comp_n=0 if flags is None else len(flags),
    )


# auditme: accept[duplicate-call-sequence] mirrors profile.rma.mv, which repeats the steps of the other profile methods
@metafopy.core.timing.timed
def profile_rma_mv(
    fitted: MV,
    sigma2: int | None = None,
    tau2: int | None = None,
    rho: int | None = None,
    gamma2: int | None = None,
    phi: int | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    steps: int | numpy.typing.ArrayLike = 20,
    lltol: float = 0.001,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    plot: bool = False,
    *,
    exp: bool = False,
    ax: typing.Any = None,
    startmethod: typing.Any = None,
    LB: bool = False,
    **plot_args: typing.Any,
) -> MVProfileResult | list[MVProfileResult]:
    """Profile the likelihood of an [rma_mv][metafopy.rma_mv] fit over one variance or correlation component (metafor's ``profile.rma.mv``).

    Without any of ``sigma2``, ``tau2``, ``rho``, ``gamma2`` and ``phi`` every estimated component is profiled and a list is returned (a single result if
    there is just one; several are a ``ProfileList``). ``xlim`` defaults to metafor's range around the estimate: for a variance a quarter to four times
    the estimate (half to twice for ``tau2`` and ``gamma2`` with a spatial structure), at least up to 0.1; for a correlation the estimate +- 0.5 within
    (-1, 1) (from 0 for ``'CAR'``, half to twice the estimate for the spatial structures).

    R source: <https://github.com/wviechtb/metafor/blob/master/R/profile.rma.mv.r>; reference page:
    <https://wviechtb.github.io/metafor/reference/profile.rma.html>.

    Args:
        fitted: Model fitted by [rma_mv][metafopy.rma_mv].
        sigma2: 0-based position of the ``sigma2`` component to profile (metafor counts from 1). Only one of ``sigma2``, ``tau2``, ``rho``, ``gamma2``
            and ``phi`` may be given.
        tau2: 0-based position of the ``tau2`` component to profile.
        rho: 0-based position of the ``rho`` component to profile.
        gamma2: 0-based position of the ``gamma2`` component to profile.
        phi: 0-based position of the ``phi`` component to profile.
        xlim: Range ``(lower, upper)`` of the profiled values of the component, within the parameter space (a variance at least 0, a correlation
            within -1 and 1, at least 0 for ``'CAR'`` and the spatial structures). Default: metafor's range around the estimate.
        ylim: Range ``(lower, upper)`` of the profile log-likelihood stored in the result (and drawn by the plot). Default: the range of the profile and
            the maximum.
        steps: Number of equally spaced values in ``xlim`` to profile over (a single number, at least 2), or a sequence of two or more values to profile
            at (then ``xlim`` defaults to their range). Missing values are not allowed.
        lltol: Tolerance when checking whether a profiled log-likelihood exceeds that of the fitted model (a warning is given if it does).
        progbar: Print a progress bar to stderr while the models of the grid are fitted (default False; metafor's default is ``TRUE``).
        parallel: Parallel fitting of the grid: ``'no'`` (default), ``'snow'`` or ``'multicore'``; both use a process pool ('multicore' forks the
            workers where the platform allows). As in metafor, ``'no'`` with ``ncpus > 1`` (or with ``cl``) is ``'snow'``, and ``'snow'`` or
            ``'multicore'`` with ``ncpus < 2`` and no ``cl`` runs serially. The result equals that of the serial run. Any other value raises
            an InputError.
        ncpus: Number of worker processes (at least 1).
        cl: Executor to run the fits on instead of creating a process pool of ``ncpus`` workers.
        plot: Draw the profile with [plot_profile][metafopy.plot_profile] (default False; metafor's default is ``TRUE``).
        exp: Report the likelihood instead of the log-likelihood (exponentiate it; ``ylim`` is then not widened by 0.1).
        ax: Matplotlib axes to draw on (with ``plot=True``). Default: a new figure.
        startmethod: Deprecated in metafor; giving it only warns.
        LB: Accepted as in metafor, where it chooses ``parLapplyLB``; it changes no result.
        **plot_args: Further keyword arguments passed on to [plot_profile][metafopy.plot_profile] (``pch``, ``cline``, ``refline``, ``xlab``, ``ylab``,
            ``main``, ``log``, ``cex``, ``lwd``), as metafor's ``...`` is to ``plot.profile.rma``; they are not used (and not checked) with ``plot=False``.

    Returns:
        The profile, a [MVProfileResult][metafopy.methods.mv_profile.MVProfileResult], or a list of them when several components are profiled.

    Raises:
        metafopy.errors.InputError: more than one component is given; the model has no such (estimated) component or the position does not exist or is
            fixed; no component can be profiled; ``steps`` has missing values or is a single number below 2; ``xlim`` or ``ylim`` does not have length
            2 or ``xlim`` leaves the parameter space; ``parallel`` is not one of the accepted values or ``ncpus`` is below 1.
    """
    if startmethod is not None:
        warnings.warn("Argument 'startmethod' has been deprecated.", stacklevel=2)
    x = fitted
    if plot_args.pop('startmethod', None) is not None:
        warnings.warn("Argument 'startmethod' has been deprecated.", stacklevel=2)
    mode = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    pool: dict[str, typing.Any] = {'progbar': progbar, 'parallel': mode, 'ncpus': ncpus, 'cl': cl}
    st = numpy.atleast_1d(numpy.asarray(steps, dtype=numpy.float64))
    if numpy.isnan(st).any():
        raise metafopy.errors.InputError("No missing values allowed in 'steps' argument.")
    step_seq = st.shape[0] >= 2
    if step_seq:
        if xlim is None:
            xlim = (float(st.min()), float(st.max()))
    elif st[0] < 2:
        raise metafopy.errors.InputError("Argument 'steps' must be >= 2.")

    chosen = dict(zip(metafopy.models.mv_structs.VC_NAMES, (sigma2, tau2, rho, gamma2, phi), strict=True))
    if all(v is None for v in chosen.values()):
        todo = [(name, int(pos)) for name in metafopy.models.mv_structs.VC_NAMES if (flags := x.vc_fix[name]) is not None for pos in numpy.flatnonzero(~flags)]
        if not todo:
            raise metafopy.errors.InputError('No components in the model for which a profile likelihood can be constructed.')
        sav = [profile_one(x, comp, pos, xlim, ylim, st, step_seq, lltol, exp, pool) for comp, pos in todo]
        if plot:
            plot_all(sav, ax, plot_args)
        return sav[0] if len(sav) == 1 else metafopy.methods.profile.ProfileList(sav)

    comp, pos = component(x, chosen)
    out = profile_one(x, comp, pos, xlim, ylim, st, step_seq, lltol, exp, pool)
    if plot:
        plot_all([out], ax, plot_args)
    return out


def plot_all(results: list[MVProfileResult], ax: typing.Any, plot_args: collections.abc.Mapping[str, typing.Any]) -> None:
    """Draw each profile; several profiles go into a grid of panels (metafor's ``par(mfrow = n2mfrow(comps))``)."""
    # auditme: accept[in-function-import] metafopy.plot imports the methods, so the import is deferred to avoid a cycle
    import metafopy.plot.profile

    metafopy.plot.profile.plot_profile(results[0] if len(results) == 1 else results, ax=ax, **plot_args)


metafopy.methods.profile.profile.register(metafopy.models.mv_result.RMAMVResult, profile_rma_mv)  # type: ignore[arg-type]
