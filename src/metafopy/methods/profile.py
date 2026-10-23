"""Profile likelihood plots for ``tau^2`` (metafor's ``profile.rma.uni``).

The model is refitted with ``tau^2`` fixed at each value of a grid and the
log-likelihood of each fit is recorded (the REML log-likelihood for models
fitted with REML, the ML one otherwise), see Viechtbauer (2010),
doi:10.18637/jss.v036.i03, and
https://wviechtb.github.io/metafor/reference/profile.rma.html.
"""

import collections.abc
import concurrent.futures
import dataclasses
import functools
import importlib
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.labelled
import metafopy.core.parallel
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.confint
import metafopy.methods.predict
import metafopy.models.rma_uni
import metafopy.models.tau2

__all__ = ['ProfileList', 'ProfileResult', 'profile']

type FloatArray = numpy.typing.NDArray[numpy.float64]


class ProfileList(list[typing.Any]):
    """The profiles of several components (metafor returns a list of profiles with the number of components in ``comps``)."""

    @property
    def comps(self) -> int:
        """The number of components profiled."""
        return len(self)

    def __str__(self) -> str:
        """The list as R prints it: ``[[1]]``, ``[[2]]``, ... above the profiles."""
        return '\n'.join(f'[[{i + 1}]]\n{item}\n' for i, item in enumerate(self))


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileResult:
    """Profile log-likelihood of ``tau^2`` (or ``tau`` with ``sqrt=True``), as returned by [profile][metafopy.profile]. ``to_frame()`` gives the grid and the
    log-likelihoods.

    Attributes:
        tau2: The grid of profiled values (``tau`` values for ``sqrt=True``, which are then also in the ``tau`` property; reading ``tau`` otherwise
            raises AttributeError).
        ll: Profile (log-)likelihood at each grid value.
        beta: Coefficient estimates at each grid value (one row per grid value, one column per coefficient).
        ci_lb: Lower confidence interval bounds of the coefficients at each grid value.
        ci_ub: Upper confidence interval bounds of the coefficients at each grid value.
        xlim: Range ``(lower, upper)`` of the profiled values.
        ylim: Range ``(lower, upper)`` of the (log-)likelihood used for the plot.
        method: Estimation method of the model.
        vc: The model's own estimate of the profiled parameter (``tau`` with ``sqrt=True``).
        maxll: (Restricted) log-likelihood of the fitted model (its exponential with ``exp=True``).
        I2: Estimates of ``I^2`` at each grid value.
        H2: Estimates of ``H^2`` at each grid value.
        exp: Whether ``ll`` is the likelihood rather than the log-likelihood.
        sqrt: Whether the profile is over ``tau`` rather than ``tau^2``.
        pred: With ``pred=True``: predicted values at each grid value (one row per prediction, one column per grid value).
        pred_ci_lb: With ``pred=True``: lower confidence bounds of the predictions.
        pred_ci_ub: With ``pred=True``: upper confidence bounds of the predictions.
        pred_pi_lb: With ``pred=True``: lower prediction interval bounds.
        pred_pi_ub: With ``pred=True``: upper prediction interval bounds.
        blup: With ``blup=True``: BLUPs at each grid value (one row per study).
        blup_se: With ``blup=True``: standard errors of the BLUPs.
        blup_pi_lb: With ``blup=True``: lower prediction interval bounds of the BLUPs.
        blup_pi_ub: With ``blup=True``: upper prediction interval bounds of the BLUPs.
    """

    tau2: FloatArray
    ll: FloatArray
    beta: pandas.DataFrame
    ci_lb: pandas.DataFrame
    ci_ub: pandas.DataFrame
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    method: str
    vc: float
    maxll: float
    I2: FloatArray
    H2: FloatArray
    exp: bool = False
    sqrt: bool = False
    pred: pandas.DataFrame | None = None
    pred_ci_lb: pandas.DataFrame | None = None
    pred_ci_ub: pandas.DataFrame | None = None
    pred_pi_lb: pandas.DataFrame | None = None
    pred_pi_ub: pandas.DataFrame | None = None
    blup: pandas.DataFrame | None = None
    blup_se: pandas.DataFrame | None = None
    blup_pi_lb: pandas.DataFrame | None = None
    blup_pi_ub: pandas.DataFrame | None = None

    @property
    def comps(self) -> int:
        """metafor's ``comps``: the number of variance components profiled (one)."""
        return 1

    @property
    def xlab(self) -> str:
        """The label of the x axis of the profile plot."""
        return 'tau Value' if self.sqrt else 'tau^2 Value'

    @property
    def title(self) -> str:
        """The title of the profile plot."""
        return 'Profile Plot for tau' if self.sqrt else 'Profile Plot for tau^2'

    @property
    def tau(self) -> FloatArray:
        """The grid of ``tau`` values (metafor names the grid ``tau`` for ``sqrt=TRUE``).

        Raises:
            AttributeError: the profile is of ``tau^2``.
        """
        if not self.sqrt:
            raise AttributeError('tau')
        return self.tau2

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'tau' if self.sqrt else 'tau2': self.tau2, 'll': self.ll})

    def __str__(self) -> str:
        return metafopy.core.labelled.frame_text(self.to_frame())


@functools.singledispatch
def profile(fitted: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> ProfileResult:
    """Profile likelihood of a variance component of a fitted model (metafor's ``profile``).

    This is a single-dispatch function: the implementation is chosen by the class of ``fitted``. See
    [profile_rma_uni][metafopy.methods.profile.profile_rma_uni], [profile_rma_ls][metafopy.methods.ls_profile.profile_rma_ls],
    [profile_rma_mv][metafopy.methods.mv_profile.profile_rma_mv]; selection models ([selmodel][metafopy.selmodel]) are also supported.

    Args:
        fitted: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation.
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        The profile: a [ProfileResult][metafopy.ProfileResult] for an [rma_uni][metafopy.rma_uni] fit, an ``LSProfileResult`` for a location-scale fit,
        a [MVProfileResult][metafopy.methods.mv_profile.MVProfileResult] for an [rma_mv][metafopy.rma_mv] fit and a
        [SelProfileResult][metafopy.SelProfileResult] for a selection model; a list of them (a ``ProfileList``) when several components are profiled.

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``fitted``.
        metafopy.errors.InputError: raised by the implementation for an invalid argument.
    """
    raise metafopy.errors.ArgumentError(f'profile() not implemented for {type(fitted).__name__}')


type Extras = tuple[dict[str, FloatArray], dict[str, FloatArray]]
type FitResult = tuple[float, FloatArray, FloatArray, FloatArray, float, float, Extras]

PRED_KEYS = ('pred', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub')
BLUP_KEYS = ('pred', 'se', 'pi_lb', 'pi_ub')


def profile_fit(x: metafopy.models.rma_uni.RMAResult, tau2: float, pred: bool = False, blup: bool = False, newmods: typing.Any = None) -> FitResult | None:
    """The refit of ``x`` with ``tau^2`` fixed at ``tau2``: log-likelihood, coefficients, CI bounds, I^2, H^2 and, if asked, predictions and BLUPs.

    ``None`` if the refit fails.
    """
    fit = metafopy.methods.common.refit_arrays(x, tau2=tau2)
    if fit is None:
        return None
    pred_out: dict[str, FloatArray] = {}
    blup_out: dict[str, FloatArray] = {}
    if pred:
        predres = metafopy.methods.predict.predict(fit, newmods=newmods)
        pred_out = {key: getattr(predres, key) for key in PRED_KEYS}
    if blup:
        blupres = metafopy.methods.blup.blup(fit)
        blup_out = {key: getattr(blupres, key) for key in BLUP_KEYS}
    return fit.logLik(), fit.beta, fit.ci_lb, fit.ci_ub, fit.I2, fit.H2, (pred_out, blup_out)


def grid_frame(columns: list[FloatArray | None], labels: collections.abc.Sequence[str]) -> pandas.DataFrame:
    """One column per grid value (NaN where the refit failed), one row per label."""
    out = numpy.full((len(labels), len(columns)), numpy.nan)
    for j, col in enumerate(columns):
        if col is not None:
            out[:, j] = col
    return pandas.DataFrame(out, index=list(labels))


@profile.register
@metafopy.core.timing.timed
def profile_rma_uni(
    fitted: metafopy.models.rma_uni.RMAResult,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    steps: int | numpy.typing.ArrayLike = 20,
    lltol: float = 0.001,
    plot: bool = False,
    *,
    sqrt: bool = False,
    exp: bool = False,
    ax: typing.Any = None,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    pred: bool = False,
    blup: bool = False,
    newmods: numpy.typing.ArrayLike | collections.abc.Mapping[str, float] | pandas.Series | pandas.DataFrame | None = None,
    LB: bool = False,
    **plot_args: typing.Any,
) -> ProfileResult:
    """Profile the likelihood of an [rma_uni][metafopy.rma_uni] fit over ``tau^2`` (metafor's ``profile.rma.uni``).

    The model is refitted with ``tau^2`` fixed at each value of a grid and the log-likelihood (and the coefficients) recorded. ``xlim`` defaults to the
    range spanned by the Q-profile CI for ``tau^2`` and the estimate (the upper end at least 0.1). Models with equal or fixed effects cannot be profiled.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/profile.rma.uni.r>

    Args:
        fitted: Random- or mixed-effects model fitted by [rma_uni][metafopy.rma_uni] (not an equal- or fixed-effects model).
        xlim: Range ``(lower, upper)`` of ``tau^2`` values (``tau`` values with ``sqrt=True``). Default: the range spanned by the Q-profile CI for
            ``tau^2`` and the estimate.
        ylim: Range ``(lower, upper)`` of the profile log-likelihood stored in the result (and drawn by the plot). Default: the range of the profile and
            the maximum.
        steps: Number of equally spaced values in ``xlim`` to profile over (a single number, at least 2), or a sequence of two or more values to profile
            at (then ``xlim`` defaults to their range). Missing values are not allowed.
        lltol: Tolerance when checking whether a profiled log-likelihood exceeds that of the fitted model (a warning is given if it does).
        plot: Draw the profile with [plot_profile][metafopy.plot_profile] (default False; metafor's default is ``TRUE``).
        sqrt: Profile over ``tau`` instead of ``tau^2`` (the grid and ``xlim`` are then on the ``tau`` scale).
        exp: Report the likelihood instead of the log-likelihood (exponentiate it; ``ylim`` is then not widened by 0.1).
        ax: Matplotlib axes to draw on (with ``plot=True``). Default: a new figure.
        progbar: Print a progress bar to stderr while the models of the grid are fitted (default False; metafor's default is ``TRUE``).
        parallel: Parallel fitting of the grid: ``'no'`` (default), ``'snow'`` or ``'multicore'``; both use a process pool ('multicore' forks the
            workers where the platform allows). As in metafor, ``'no'`` with ``ncpus > 1`` (or with ``cl``) is ``'snow'``, and ``'snow'`` or
            ``'multicore'`` with ``ncpus < 2`` and no ``cl`` runs serially. The result equals that of the serial run. Any other value raises
            an InputError.
        ncpus: Number of worker processes (at least 1).
        cl: Executor to run the fits on instead of creating a process pool of ``ncpus`` workers.
        pred: Also store the predicted values of each refit (for ``newmods``, or the fitted values without it, as [predict][metafopy.predict]); metafor's
            undocumented argument.
        blup: Also store the BLUPs ([blup][metafopy.blup]) of each refit; metafor's undocumented argument.
        newmods: Moderator values for the predictions stored with ``pred=True``, as for [predict][metafopy.predict].
        LB: Accepted as in metafor, where it chooses ``parLapplyLB``; it changes no result.
        **plot_args: Further keyword arguments passed on to [plot_profile][metafopy.plot_profile] (``pch``, ``cline``, ``refline``, ``xlab``, ``ylab``,
            ``main``, ``log``, ``cex``, ``lwd``), as metafor's ``...`` is to ``plot.profile.rma``; they are not used (and not checked) with ``plot=False``.

    Returns:
        The profile, a [ProfileResult][metafopy.ProfileResult] with the grid ``tau2``, the log-likelihoods ``ll``, the coefficients and their CI bounds at
        each grid value, and the estimate ``vc`` and maximum log-likelihood ``maxll`` of the model.

    Raises:
        metafopy.errors.InputError: the model uses an equal- or fixed-effects method; ``steps`` has missing values or is a single number below 2;
            ``xlim`` or ``ylim`` does not have length 2; ``xlim`` cannot be set automatically; ``parallel`` is not one of the accepted values or
            ``ncpus`` is below 1; with ``pred=True`` no predictions can be computed.

    Metafor reference:
        [profile.rma](https://wviechtb.github.io/metafor/reference/profile.rma.html)
    """
    x = fitted
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    if x.method in metafopy.models.tau2.FIXED_EFFECTS:
        raise metafopy.errors.InputError('Cannot profile tau^2 parameter for equal/fixed-effects models.')
    pred_slab: tuple[str, ...] | None = None
    if pred:
        predres = metafopy.methods.predict.predict(x, newmods=newmods)
        if predres.pred.shape[0] == 0:
            raise metafopy.errors.InputError('Cannot compute predicted values.')
        n_pred = predres.pred.shape[0]
        pred_slab = predres.slab if predres.slab is not None else tuple(str(i + 1) if n_pred > 1 else '' for i in range(n_pred))
    st = numpy.atleast_1d(numpy.asarray(steps, dtype=numpy.float64))
    if numpy.isnan(st).any():
        raise metafopy.errors.InputError("No missing values allowed in 'steps' argument.")
    step_seq = st.shape[0] >= 2
    if step_seq:
        if xlim is None:
            xlim = (float(st.min()), float(st.max()))
    elif st[0] < 2:
        raise metafopy.errors.InputError("Argument 'steps' must be >= 2.")

    if xlim is None:
        lb: float
        ub: float
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ci = metafopy.methods.confint.confint(x)
            assert ci.random is not None
            lb = min(x.tau2, float(numpy.asarray(ci.random.iloc[0, 1], dtype=numpy.float64)))
            ub = max(0.1, x.tau2, float(numpy.asarray(ci.random.iloc[0, 2], dtype=numpy.float64)))
        except ValueError, ArithmeticError, numpy.linalg.LinAlgError, AssertionError:
            lb = ub = math.nan
        if math.isnan(lb) or math.isnan(ub):
            crit = float(scipy.stats.norm.ppf(0.995))
            lb = max(0.0, x.tau2 - crit * x.se_tau2)
            ub = max(0.1, x.tau2 + crit * x.se_tau2)
        if math.isnan(lb) or math.isnan(ub):
            lb = max(0.0, x.tau2 / 4)
            ub = max(0.1, x.tau2 * 4)
        if math.isnan(lb) or math.isnan(ub):
            raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")
        xl = (math.sqrt(lb), math.sqrt(ub)) if sqrt else (lb, ub)
    else:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        xl = (float(min(xlim)), float(max(xlim)))

    vcs = st if step_seq else numpy.linspace(xl[0], xl[1], int(st[0]))
    if vcs.shape[0] <= 1:
        raise metafopy.errors.InputError("Cannot set 'xlim' automatically. Please set this argument manually.")
    tau2_grid = vcs**2 if sqrt else vcs

    names = list(x.coef_names)
    lls = numpy.full(len(tau2_grid), numpy.nan)
    beta = numpy.full((len(tau2_grid), x.p), numpy.nan)
    ci_lb = numpy.full_like(beta, numpy.nan)
    ci_ub = numpy.full_like(beta, numpy.nan)
    I2 = numpy.full(len(tau2_grid), numpy.nan)
    H2 = numpy.full(len(tau2_grid), numpy.nan)
    one = functools.partial(profile_fit, x, pred=pred, blup=blup, newmods=newmods)
    fits = metafopy.core.parallel.map_sims(one, [float(t2) for t2 in tau2_grid], parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
    pred_cols: list[dict[str, FloatArray] | None] = []
    blup_cols: list[dict[str, FloatArray] | None] = []
    for i, fit in enumerate(fits):
        if fit is None:
            pred_cols.append(None)
            blup_cols.append(None)
            continue
        lls[i], beta[i], ci_lb[i], ci_ub[i], I2[i], H2[i], (pred_i, blup_i) = fit
        pred_cols.append(pred_i)
        blup_cols.append(blup_i)

    vc = math.sqrt(x.tau2) if sqrt else x.tau2
    maxll = float(x.logLik())
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if x.method in ('ML', 'REML') and numpy.any(lls >= maxll + lltol):
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

    extra: dict[str, pandas.DataFrame] = {}
    if pred and pred_slab is not None:
        for key in PRED_KEYS:
            extra['pred' if key == 'pred' else f'pred_{key}'] = grid_frame([None if c is None else c[key] for c in pred_cols], pred_slab)
    if blup:
        for key in BLUP_KEYS:
            extra['blup' if key == 'pred' else f'blup_{key}'] = grid_frame([None if c is None else c[key] for c in blup_cols], x.slab)

    out = ProfileResult(
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
        I2=I2,
        H2=H2,
        exp=exp,
        sqrt=sqrt,
        **extra,
    )
    if plot:
        # metafopy.plot.profile imports this module, so metafopy.plot is looked up when it is needed
        importlib.import_module('metafopy.plot').plot_profile(out, ax=ax, **plot_args)
    return out
