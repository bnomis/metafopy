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


def profile_fit(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float) -> tuple[float, FloatArray, FloatArray, FloatArray] | None:
    """The refit of ``x`` with scale coefficient ``pos`` fixed at ``val``: log-likelihood, coefficients and CI bounds (``None`` if it fails)."""
    arg = numpy.where(x.alpha_fix, x.alpha, numpy.nan)
    arg[pos] = val
    fit = metafopy.methods.ls_common.refit_ls(x, alpha=arg)
    if fit is None:
        return None
    return fit.logLik(), fit.beta, fit.ci_lb, fit.ci_ub


@metafopy.methods.profile.profile.register  # type: ignore[arg-type]  # returns a result class of its own
# auditme: accept[duplicate-call-sequence] the profile of a location-scale model follows the same steps as profile.rma in metafor
@metafopy.core.timing.timed
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

    R source: <https://github.com/wviechtb/metafor/blob/master/R/profile.rma.ls.r>; reference page:

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
            plot_all(out, ax, plot_args)
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
        plot_all([result], ax, plot_args)
    return result


def plot_all(results: list[LSProfileResult], ax: matplotlib.axes.Axes | None, plot_args: collections.abc.Mapping[str, typing.Any]) -> None:
    # auditme: accept[in-function-import] metafopy.plot.profile imports the methods, so the import is deferred to avoid a cycle
    import metafopy.plot.profile

    metafopy.plot.profile.plot_profile(results[0] if len(results) == 1 else results, ax=ax, **plot_args)
