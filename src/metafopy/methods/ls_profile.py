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
    ``ci_ub`` hold the location coefficients and their CI bounds at each grid value, ``vc`` the estimate of the
    coefficient and ``maxll`` the log-likelihood of the fitted model. ``position`` is the 0-based coefficient position.
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
        return self.to_frame().to_string()


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
    """Profile the likelihood over a scale coefficient (``alpha``, a 0-based position) on ``steps`` equally spaced values in ``xlim``.

    ``steps`` may also be the vector of values. Without ``alpha`` every estimated scale coefficient is profiled and a list is
    returned (a single result if there is one). ``xlim`` defaults to the estimate plus and minus 2.576 standard errors,
    limited by ``alpha_min`` and ``alpha_max`` of the control settings. As in metafor, ``parallel`` (``'snow'`` or
    ``'multicore'``) refits the models in a process pool of ``ncpus`` workers (or on the executor ``cl``) and
    ``progbar=True`` prints a progress bar to stderr; the result equals that of the serial run. ``plot=True`` draws the profile(s) with
    :func:`metafopy.plot.plot_profile` (metafor's default is ``plot=TRUE``), several in a grid of panels. Further keyword arguments (``pch``,
    ``cline``, ``refline``, ``xlab``, ``ylab``, ``main``, ``log``, ``cex``, ``lwd``) are passed on to :func:`metafopy.plot.plot_profile`, as metafor's
    ``...`` is to ``plot.profile.rma``. ``sub1=True`` labels the profile of the scale coefficients one lower in the plots (so that the
    intercept is ``alpha_0``); ``LB`` is accepted as in metafor and changes nothing.

    Raises:
        metafopy.errors.InputError: an argument is invalid or the model has no such coefficient.
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
