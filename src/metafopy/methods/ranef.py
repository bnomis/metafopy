"""Best linear unbiased predictions of the random effects (metafor's ``ranef.rma.uni``).

With ``lambda_i = tau^2 / (tau^2 + v_i)`` the BLUP of study ``i``'s random
effect is ``lambda_i (y_i - x_i' beta)`` with variance
``lambda_i v_i + lambda_i^2 x_i' Var(beta) x_i`` (Raudenbush & Bryk 1985,
appendix; doi:10.3102/10769986010002075); for equal/fixed/common-effects
models all random effects are 0. The computations follow
https://wviechtb.github.io/metafor/reference/ranef.html.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Any

from metafopy.methods.blup import BlupResult, NAAction, Transf, shrinkage_predictions
from metafopy.models.rma_uni import RMAResult

__all__ = ['RanefResult', 'ranef']


@dataclass(frozen=True, slots=True)
class RanefResult(BlupResult):
    """Predicted random effects with standard errors and prediction intervals."""


@singledispatch
def ranef(res: Any, *args: Any, **kwargs: Any) -> RanefResult:
    """BLUPs of the random effects for a fitted model; see :func:`_ranef_rma_uni`."""
    raise TypeError(f'ranef() not implemented for {type(res).__name__}')


@ranef.register
def _ranef_rma_uni(
    res: RMAResult,
    level: float | None = None,
    digits: int | None = None,
    transf: Transf | None = None,
    targs: Any = None,
    *,
    na_action: NAAction = 'omit',
) -> RanefResult:
    """BLUPs of the random effects of an ``rma.uni`` fit (arguments as in :func:`metafopy.methods.blup.blup`)."""
    out = shrinkage_predictions(res, 'ranef', level, digits, transf, targs, na_action)
    return RanefResult(pred=out.pred, se=out.se, pi_lb=out.pi_lb, pi_ub=out.pi_ub, slab=out.slab, digits=out.digits, transf=out.transf)
