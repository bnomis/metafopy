"""Profile likelihood plots for ``tau^2`` (metafor's ``profile.rma.uni``).

The model is refitted with ``tau^2`` fixed at each value of a grid and the
log-likelihood of each fit is recorded (the REML log-likelihood for models
fitted with REML, the ML one otherwise), see Viechtbauer (2010),
doi:10.18637/jss.v036.i03, and
https://wviechtb.github.io/metafor/reference/profile.rma.html.
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

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.confint
import metafopy.models.rma_uni
import metafopy.models.tau2

__all__ = ['ProfileResult', 'profile']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileResult:
    """Profile log-likelihood of ``tau^2`` (or ``tau`` with ``sqrt=True``).

    ``tau2`` holds the grid (``tau`` values for ``sqrt=True``), ``ll`` the
    (log-)likelihoods, ``beta``/``ci_lb``/``ci_ub`` the coefficient estimates and CI bounds
    at each grid value, ``vc`` the model's own estimate and ``maxll`` its log-likelihood.
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

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'tau' if self.sqrt else 'tau2': self.tau2, 'll': self.ll})

    def __str__(self) -> str:
        return self.to_frame().to_string()


@functools.singledispatch
def profile(fitted: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> ProfileResult:
    """Profile likelihood of a variance component; see :func:`_profile_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'profile() not implemented for {type(fitted).__name__}')


@profile.register
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
) -> ProfileResult:
    """Profile the likelihood over ``tau^2`` (``steps`` equally spaced values in ``xlim``, or the values given as ``steps``).

    ``xlim`` defaults to the range spanned by the Q-profile CI for ``tau^2`` and
    the estimate (at least up to 0.1); with ``sqrt=True`` the grid is on the ``tau``
    scale. With ``plot=True`` the profile is drawn with :func:`metafopy.plot.plot_profile`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = fitted
    if x.method in metafopy.models.tau2.FIXED_EFFECTS:
        raise metafopy.errors.InputError('Cannot profile tau^2 parameter for equal/fixed-effects models.')
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
    for i, t2 in enumerate(tau2_grid):
        fit = metafopy.methods.common.refit_arrays(x, tau2=float(t2))
        if fit is None:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            continue
        lls[i] = fit.logLik()
        beta[i], ci_lb[i], ci_ub[i] = fit.beta, fit.ci_lb, fit.ci_ub
        I2[i], H2[i] = fit.I2, fit.H2

    vc = math.sqrt(x.tau2) if sqrt else x.tau2
    maxll = x.logLik()
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
    )
    if plot:
        # auditme: accept[in-function-import] metafopy.plot.profile imports this module, so the import is deferred to avoid a cycle
        from metafopy.plot import plot_profile

        plot_profile(out, ax=ax)
    return out
