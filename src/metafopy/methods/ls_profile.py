"""Profile likelihood for the scale coefficients of location-scale models (metafor's ``profile.rma.ls``).

The model is refitted with one scale coefficient fixed at each value of a grid and the log-likelihood of each fit is recorded
(Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1562).
"""

import collections.abc
import concurrent.futures
import dataclasses
import functools
import math
import typing
import warnings

import matplotlib.axes
import numpy
import numpy.typing
import pandas
import scipy.optimize
import scipy.stats

import metafopy.core.labelled
import metafopy.core.parallel
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.ls_common
import metafopy.methods.profile
import metafopy.models.rma_ls_result

__all__ = ['LSProfileResult', 'profile_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]

LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
FEASIBLE_WEIGHTS = (0.25, 0.5, 0.75)


@dataclasses.dataclass(frozen=True, slots=True)
class LSProfileResult:
    """Profile log-likelihood of one scale coefficient.

    ``alpha`` holds the grid and ``ll`` the log-likelihoods (the REML ones for REML fits); ``beta``, ``ci_lb`` and
    ``ci_ub`` hold the location coefficients and their CI bounds at each grid value (one row per grid value), ``ylim`` the range of the
    log-likelihood for the plot, ``vc`` the estimate of the coefficient, ``maxll`` the log-likelihood of the fitted model and ``exp`` whether ``ll`` and
    ``maxll`` are exponentiated. ``position`` is the 0-based coefficient position and ``alphas`` the number of scale coefficients. Unlike
    [ProfileResult][metafopy.ProfileResult] it has no ``tau2``, ``I2`` or ``H2``.
    """

    alpha: FloatArray
    ll: FloatArray
    beta: pandas.DataFrame
    ci_lb: pandas.DataFrame
    ci_ub: pandas.DataFrame
    ylim: tuple[float, float]
    method: str
    vc: float
    maxll: float
    position: int
    exp: bool = False
    alphas: int = 1
    sub1: bool = False

    @property
    def xlim(self) -> tuple[float, float]:
        """The range of the grid, which is the horizontal range metafor's ``plot.profile.rma`` draws (the object has no ``xlim`` element)."""
        grid = self.alpha[numpy.isfinite(self.alpha)]
        return float(grid.min()), float(grid.max())

    @property
    def comps(self) -> int:
        """metafor's ``comps``: the number of components profiled in this profile (one)."""
        return 1

    @property
    def xlab(self) -> str:
        """The label of the x axis of the profile plot."""
        return f'alpha[{self.position + (0 if self.sub1 else 1)}] Value'

    @property
    def title(self) -> str:
        """The title of the profile plot."""
        return f'Profile Plot for alpha[{self.position + (0 if self.sub1 else 1)}]'

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'alpha': self.alpha, 'll': self.ll})

    def __str__(self) -> str:
        return metafopy.core.labelled.frame_text(self.to_frame())


def feasible_start(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> FloatArray | None:
    """A start of the identity-link refit at ``alpha[pos] = val`` whose smallest ``tau2_i`` is as large as it can be, or ``None`` if none is positive.

    Solves the linear programme: maximise ``t`` over the other coefficients ``a`` subject to ``Z[:, free] a + Z[:, pos] val >= t`` for every study and
    ``t <= mean(Z @ x.alpha)`` (1 where that mean is not positive), with ``scipy.optimize.linprog``. Returns ``x.alpha`` with ``alpha[pos] = val`` and
    the solution's ``a``, or ``None`` when ``pos`` has no other coefficient, the programme fails, or its ``t`` is not positive.
    """
    free = [i for i in range(x.q) if i != pos]
    if not free:
        return None
    n = len(free)
    cap = float(numpy.mean(x.Z @ x.alpha))
    cap = cap if cap > 0 else 1.0
    result = scipy.optimize.linprog(
        c=numpy.append(numpy.zeros(n), -1.0),
        A_ub=numpy.column_stack([-x.Z[:, free], numpy.ones(x.Z.shape[0])]),
        b_ub=x.Z[:, pos] * val,
        bounds=[(None, None)] * n + [(None, cap)],
    )
    if result.status != 0 or result.x is None or result.x[n] <= 0:
        return None
    point = numpy.array(x.alpha, dtype=numpy.float64)
    point[pos] = val
    point[free] = result.x[:n]
    return point


def profile_starts(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> list[FloatArray | None]:
    """The starting values of the refits at ``alpha[pos] = val``: the default start (``None``), then points from the fit's ``alpha`` along the line start.

    The fitted line start sets the other coefficients by least squares so that ``Z alpha`` stays as near ``Z alpha_hat`` as it can with ``alpha[pos]``
    at ``val``. Far from the estimate the optimiser stops at the first start whenever the likelihood is nearly flat there, so the starts between the
    two (a quarter, a half and three quarters of the way) are tried as well, and two beyond the line start (one and a half and twice the way): the
    profile optimum can lie past the line, where every start inside it ends on a ``tau2 = 0`` plateau. A start that leaves a ``tau2_i`` negative under
    the identity link is left out; if that leaves only the default start, ``feasible_start`` and the feasible points from it towards the fit's ``alpha``
    are added.
    """
    own = numpy.array(x.alpha, dtype=numpy.float64)
    own[pos] = val
    starts: list[FloatArray | None] = [None, own]
    free = [i for i in range(x.q) if i != pos]
    if free:
        line = own.copy()
        line[free] = numpy.linalg.lstsq(x.Z[:, free], x.Z @ x.alpha - x.Z[:, pos] * val, rcond=None)[0]
        starts.extend(own + weight * (line - own) for weight in LINE_WEIGHTS)
    if x.link == 'identity':
        starts = [s for s in starts if s is None or float(numpy.min(x.Z @ s)) >= 0]
        if all(s is None for s in starts):
            start = feasible_start(x, pos, val)
            if start is not None:
                starts.append(start)
                starts.extend(point for point in (start + w * (own - start) for w in FEASIBLE_WEIGHTS) if float(numpy.min(x.Z @ point)) >= 0)
    return starts


def profile_fit(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> tuple[float, FloatArray, FloatArray, FloatArray] | None:
    """The refit of ``x`` with scale coefficient ``pos`` fixed at ``val``: log-likelihood, coefficients and CI bounds (``None`` if every start fails).

    The model is refitted from each start of [profile_starts][metafopy.methods.ls_profile.profile_starts] and the fit with the highest log-likelihood is
    kept, so that the profile is the maximum over the other coefficients (Viechtbauer & Lopez-Lopez, 2022, section 2.5.3) and not a local maximum that
    one start ends at.
    """
    arg = numpy.where(x.alpha_fix, x.alpha, numpy.nan)
    arg[pos] = val
    best: metafopy.models.rma_ls_result.RMALSResult | None = None
    for start in profile_starts(x, pos, val):
        fit = metafopy.methods.ls_common.refit_ls(x, alpha=arg) if start is None else metafopy.methods.ls_common.refit_ls(x, alpha=arg, alpha_init=start)
        if fit is not None and (best is None or fit.logLik() > best.logLik()):
            best = fit
    if best is None:
        return None
    return best.logLik(), best.beta, best.ci_lb, best.ci_ub


@metafopy.methods.profile.profile.register  # type: ignore[arg-type]  # returns a result class of its own
@metafopy.core.timing.timed
# auditme: accept[duplicate-call-sequence] the profile of a location-scale model follows the same steps as profile.rma in metafor
def profile_rma_ls(
    fitted: metafopy.models.rma_ls_result.RMALSResult,
    alpha: int | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    steps: int | numpy.typing.ArrayLike = 20,
    lltol: float = 0.001,
    *,
    exp: bool = False,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    plot: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    sub1: bool = False,
    LB: bool = False,
    **plot_args: typing.Any,
) -> LSProfileResult | list[LSProfileResult]:
    """Profile the likelihood of a location-scale model over a scale coefficient (metafor's ``profile.rma.ls``).

    Without ``alpha`` every estimated scale coefficient is profiled and a list is returned (a single result if there is one). ``xlim`` defaults to the
    estimate plus and minus 2.576 standard errors, limited by ``alpha_min`` and ``alpha_max`` of the control settings.

    Each refit at a grid value starts from several points and keeps the highest log-likelihood, as the profile likelihood of Viechtbauer &
    Lopez-Lopez (2022, section 2.5.3) maximises over the other coefficients; metafor's ``.profile.rma.ls`` refits once from its default start. See the
    subsection "Profile-likelihood bounds of the scale coefficients" of ``docs/differences.md``.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/profile.rma.ls.r>

    Args:
        fitted: Location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
        alpha: 0-based position of the scale coefficient to profile (0 is the first, usually the intercept; metafor counts from 1). Default: all
            estimated scale coefficients.
        xlim: Range ``(lower, upper)`` of the profiled values of the coefficient. Default: the estimate +- 2.576 standard errors, limited by the control
            settings.
        ylim: Range ``(lower, upper)`` of the profile log-likelihood stored in the result (and drawn by the plot). Default: the range of the profile and
            the maximum.
        steps: Number of equally spaced values in ``xlim`` to profile over (a single number, at least 2), or a sequence of two or more values to profile
            at (then ``xlim`` defaults to their range). Missing values are not allowed.
        lltol: Tolerance when checking whether a profiled log-likelihood exceeds that of the fitted model (a warning is given if it does).
        exp: Report the likelihood instead of the log-likelihood (exponentiate it; ``ylim`` is then not widened by 0.1).
        progbar: Print a progress bar to stderr while the models of the grid are fitted (default False; metafor's default is ``TRUE``).
        parallel: Parallel fitting of the grid: ``'no'`` (default), ``'snow'`` or ``'multicore'``; both use a process pool ('multicore' forks the
            workers where the platform allows). As in metafor, ``'no'`` with ``ncpus > 1`` (or with ``cl``) is ``'snow'``, and ``'snow'`` or
            ``'multicore'`` with ``ncpus < 2`` and no ``cl`` runs serially. The result equals that of the serial run. Any other value raises
            an InputError.
        ncpus: Number of worker processes (at least 1).
        cl: Executor to run the fits on instead of creating a process pool of ``ncpus`` workers.
        plot: Draw the profile with [plot_profile][metafopy.plot_profile] (default False; metafor's default is ``TRUE``). Several profiles are drawn in a
            grid of panels.
        ax: Matplotlib axes to draw on (with ``plot=True``). Default: a new figure.
        sub1: Number the scale coefficients in the axis label and title of the plot from 0, so that the first is ``alpha[0]`` instead of ``alpha[1]``
            (metafor's undocumented ``sub1`` argument).
        LB: Accepted as in metafor, where it chooses ``parLapplyLB``; it changes no result.
        **plot_args: Further keyword arguments passed on to [plot_profile][metafopy.plot_profile] (``pch``, ``cline``, ``refline``, ``xlab``, ``ylab``,
            ``main``, ``log``, ``cex``, ``lwd``), as metafor's ``...`` is to ``plot.profile.rma``; they are not used (and not checked) with ``plot=False``.

    Returns:
        The profile (an ``LSProfileResult`` with the grid ``alpha``, the log-likelihoods ``ll``, ``beta``, ``ci_lb``, ``ci_ub``, ``ylim``, ``vc``,
        ``maxll`` and ``position``), or a list of them (a ``ProfileList``) when several scale coefficients are profiled.

    Raises:
        metafopy.errors.InputError: the model was fitted with ``optbeta=True`` or has no estimated scale coefficient; ``alpha`` is out of range or
            fixed; ``steps`` has missing values or is a single number below 2; ``xlim`` or ``ylim`` does not have length 2; ``parallel`` is not one
            of the accepted values or ``ncpus`` is below 1.
        metafopy.errors.InternalError: a refit of the location-scale model found an internal invariant broken.

    Metafor reference:
        [profile.rma](https://wviechtb.github.io/metafor/reference/profile.rma.html)
    """
    x = fitted
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    if x.optbeta:
        raise metafopy.errors.InputError("Profiling not yet implemented for models fitted with 'optbeta=True'.")
    st = numpy.atleast_1d(numpy.asarray(steps, dtype=numpy.float64))
    if numpy.isnan(st).any():
        raise metafopy.errors.InputError("No missing values allowed in 'steps' argument.")
    step_seq = st.shape[0] >= 2
    if step_seq:
        if xlim is None:
            xlim = (float(st.min()), float(st.max()))
    elif st[0] < 2:
        raise metafopy.errors.InputError("Argument 'steps' must be >= 2.")

    if alpha is None:
        free = numpy.flatnonzero(~x.alpha_fix)
        if free.size == 0:
            raise metafopy.errors.InputError('No components in the model for which a profile likelihood can be constructed.')
        results = [
            profile_rma_ls(x, int(pos), xlim, ylim, steps, lltol, exp=exp, progbar=progbar, parallel=parallel, ncpus=ncpus, cl=cl, sub1=sub1) for pos in free
        ]
        out = [r for r in results if isinstance(r, LSProfileResult)]
        if plot and out:
            metafopy.methods.profile.plot_all(out, ax, plot_args)
        return out[0] if len(out) == 1 else metafopy.methods.profile.ProfileList(out)

    if bool(numpy.all(x.alpha_fix)):
        raise metafopy.errors.InputError("Model does not contain any estimated 'alpha' components.")
    if not 0 <= alpha < x.q:
        raise metafopy.errors.InputError("No such 'alpha' component in the model.")
    if x.alpha_fix[alpha]:
        raise metafopy.errors.InputError("Specified 'alpha' component was fixed.")
    vc = float(x.alpha[alpha])

    if xlim is None:
        se = float(x.se_alpha[alpha])
        if math.isnan(se):
            lb, ub = vc - 4 * abs(vc), vc + 4 * abs(vc)
        else:
            crit = float(scipy.stats.norm.ppf(0.995))
            lb, ub = vc - crit * se, vc + crit * se
        if math.isnan(lb) or math.isnan(ub):
            raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")
        amin = numpy.broadcast_to(numpy.asarray(x.control.alpha_min, dtype=numpy.float64), (x.q,))
        amax = numpy.broadcast_to(numpy.asarray(x.control.alpha_max, dtype=numpy.float64), (x.q,))
        lb = max(lb, float(amin[alpha]))
        ub = min(ub, float(amax[alpha]))
        xl = (min(lb, ub), max(lb, ub))
    else:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        xl = (float(min(xlim)), float(max(xlim)))

    vcs = st if step_seq else numpy.linspace(xl[0], xl[1], int(st[0]))
    if vcs.shape[0] <= 1:
        raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")

    names = list(x.coef_names)
    lls = numpy.full(len(vcs), numpy.nan)
    beta = numpy.full((len(vcs), x.p), numpy.nan)
    ci_lb = numpy.full_like(beta, numpy.nan)
    ci_ub = numpy.full_like(beta, numpy.nan)
    one = functools.partial(profile_fit, x, alpha)
    fits = metafopy.core.parallel.map_sims(one, [float(v) for v in vcs], parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        lls[i], beta[i], ci_lb[i], ci_ub[i] = fit

    maxll = float(x.logLik())
    if numpy.any(lls >= maxll + lltol):
        warnings.warn('At least one profiled log-likelihood value is larger than the log-likelihood of the fitted model.', stacklevel=2)
    if numpy.all(numpy.isnan(lls)):
        warnings.warn('All model fits failed. Cannot draw profile likelihood plot.', stacklevel=2)
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

    result = LSProfileResult(
        alpha=vcs,
        ll=lls,
        beta=pandas.DataFrame(beta, columns=names),
        ci_lb=pandas.DataFrame(ci_lb, columns=names),
        ci_ub=pandas.DataFrame(ci_ub, columns=names),
        ylim=(float(yl[0]), float(yl[1])),
        method=x.method,
        vc=vc,
        maxll=maxll,
        position=alpha,
        exp=exp,
        alphas=x.q,
        sub1=sub1,
    )
    if plot:
        metafopy.methods.profile.plot_all([result], ax, plot_args)
    return result
