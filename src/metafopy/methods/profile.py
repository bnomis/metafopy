"""Profile likelihood plots for ``tau^2`` (metafor's ``profile.rma.uni``).

The model is refitted with ``tau^2`` fixed at each value of a grid and the
log-likelihood of each fit is recorded (the REML log-likelihood for models
fitted with REML, the ML one otherwise), see Viechtbauer (2010),
doi:10.18637/jss.v036.i03, and
https://wviechtb.github.io/metafor/reference/profile.rma.html.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.methods._common import refit_arrays
from metafopy.methods.confint import confint
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult

__all__ = ['ProfileResult', 'profile']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Profile log-likelihood of ``tau^2`` (or ``tau`` with ``sqrt=True``).

    ``tau2`` holds the grid (``tau`` values for ``sqrt=True``), ``ll`` the
    (log-)likelihoods, ``beta``/``ci_lb``/``ci_ub`` the coefficient estimates and CI bounds
    at each grid value, ``vc`` the model's own estimate and ``maxll`` its log-likelihood.
    """

    tau2: FloatArray
    ll: FloatArray
    beta: pd.DataFrame
    ci_lb: pd.DataFrame
    ci_ub: pd.DataFrame
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    method: str
    vc: float
    maxll: float
    I2: FloatArray
    H2: FloatArray
    exp: bool = False
    sqrt: bool = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'tau' if self.sqrt else 'tau2': self.tau2, 'll': self.ll})

    def __str__(self) -> str:
        return self.to_frame().to_string()


@singledispatch
def profile(fitted: Any, *args: Any, **kwargs: Any) -> ProfileResult:
    """Profile likelihood of a variance component; see :func:`_profile_rma_uni`."""
    raise TypeError(f'profile() not implemented for {type(fitted).__name__}')


@profile.register
def _profile_rma_uni(
    fitted: RMAResult,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    steps: int | ArrayLike = 20,
    lltol: float = 0.001,
    plot: bool = False,
    *,
    sqrt: bool = False,
    exp: bool = False,
    ax: Any = None,
) -> ProfileResult:
    """Profile the likelihood over ``tau^2`` (``steps`` equally spaced values in ``xlim``, or the values given as ``steps``).

    ``xlim`` defaults to the range spanned by the Q-profile CI for ``tau^2`` and
    the estimate (at least up to 0.1); with ``sqrt=True`` the grid is on the ``tau``
    scale. With ``plot=True`` the profile is drawn with :func:`metafopy.plot.plot_profile`.
    """
    x = fitted
    if x.method in FIXED_EFFECTS:
        raise ValueError('Cannot profile tau^2 parameter for equal/fixed-effects models.')
    st = np.atleast_1d(np.asarray(steps, dtype=np.float64))
    if np.isnan(st).any():
        raise ValueError("No missing values allowed in 'steps' argument.")
    step_seq = st.shape[0] >= 2
    if step_seq:
        if xlim is None:
            xlim = (float(st.min()), float(st.max()))
    elif st[0] < 2:
        raise ValueError("Argument 'steps' must be >= 2.")

    if xlim is None:
        lb: float
        ub: float
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ci = confint(x)
            assert ci.random is not None
            lb = min(x.tau2, float(np.asarray(ci.random.iloc[0, 1], dtype=np.float64)))
            ub = max(0.1, x.tau2, float(np.asarray(ci.random.iloc[0, 2], dtype=np.float64)))
        except ValueError, ArithmeticError, np.linalg.LinAlgError, AssertionError:
            lb = ub = math.nan
        if math.isnan(lb) or math.isnan(ub):
            crit = float(stats.norm.ppf(0.995))
            lb = max(0.0, x.tau2 - crit * x.se_tau2)
            ub = max(0.1, x.tau2 + crit * x.se_tau2)
        if math.isnan(lb) or math.isnan(ub):
            lb = max(0.0, x.tau2 / 4)
            ub = max(0.1, x.tau2 * 4)
        if math.isnan(lb) or math.isnan(ub):
            raise ValueError("Cannot set 'xlim' automatically. Please set this argument manually.")
        xl = (math.sqrt(lb), math.sqrt(ub)) if sqrt else (lb, ub)
    else:
        if len(xlim) != 2:
            raise ValueError("Argument 'xlim' should be a vector of length 2.")
        xl = (float(min(xlim)), float(max(xlim)))

    vcs = st if step_seq else np.linspace(xl[0], xl[1], int(st[0]))
    if vcs.shape[0] <= 1:
        raise ValueError("Cannot set 'xlim' automatically. Please set this argument manually.")
    tau2_grid = vcs**2 if sqrt else vcs

    names = list(x.coef_names)
    lls = np.full(len(tau2_grid), np.nan)
    beta = np.full((len(tau2_grid), x.p), np.nan)
    ci_lb = np.full_like(beta, np.nan)
    ci_ub = np.full_like(beta, np.nan)
    I2 = np.full(len(tau2_grid), np.nan)
    H2 = np.full(len(tau2_grid), np.nan)
    for i, t2 in enumerate(tau2_grid):
        fit = refit_arrays(x, tau2=float(t2))
        if fit is None:
            continue
        lls[i] = fit.logLik()
        beta[i], ci_lb[i], ci_ub[i] = fit.beta, fit.ci_lb, fit.ci_ub
        I2[i], H2[i] = fit.I2, fit.H2

    vc = math.sqrt(x.tau2) if sqrt else x.tau2
    maxll = x.logLik()
    if x.method in ('ML', 'REML') and np.any(lls >= maxll + lltol):
        warnings.warn('At least one profiled log-likelihood value is larger than the log-likelihood of the fitted model.', stacklevel=2)
    if np.all(np.isnan(lls)):
        warnings.warn('All model fits failed. Cannot draw profile likelihood plot.', stacklevel=2)
    if exp:
        lls = np.exp(lls)
        maxll = math.exp(maxll)

    finite = lls[np.isfinite(lls)]
    if ylim is None:
        if finite.size:
            yl = (min(maxll, finite.min()), max(maxll, finite.max())) if xl[0] <= vc <= xl[1] else (finite.min(), finite.max())
        else:
            yl = (maxll, maxll)
        if not exp:
            yl = (yl[0] - 0.1, yl[1] + 0.1)
    else:
        if len(ylim) != 2:
            raise ValueError("Argument 'ylim' should be a vector of length 2.")
        yl = (float(min(ylim)), float(max(ylim)))

    out = ProfileResult(
        tau2=vcs,
        ll=lls,
        beta=pd.DataFrame(beta, columns=names),
        ci_lb=pd.DataFrame(ci_lb, columns=names),
        ci_ub=pd.DataFrame(ci_ub, columns=names),
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
        from metafopy.plot import plot_profile

        plot_profile(out, ax=ax)
    return out
