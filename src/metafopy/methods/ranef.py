"""Best linear unbiased predictions of the random effects (metafor's ``ranef.rma.uni``).

With ``lambda_i = tau^2 / (tau^2 + v_i)`` the BLUP of study ``i``'s random
effect is ``lambda_i (y_i - x_i' beta)`` with variance
``lambda_i v_i + lambda_i^2 x_i' Var(beta) x_i`` (Raudenbush & Bryk 1985,
appendix; doi:10.3102/10769986010002075); for equal/fixed/common-effects
models all random effects are 0. The computations follow
https://wviechtb.github.io/metafor/reference/ranef.html.
"""

import dataclasses
import functools
import typing

import metafopy.errors
import metafopy.methods.blup
import metafopy.models.rma_uni

__all__ = ['RanefResult', 'ranef']


@dataclasses.dataclass(frozen=True, slots=True)
class RanefResult(metafopy.methods.blup.BlupResult):
    """Predicted random effects with standard errors and prediction intervals."""


@functools.singledispatch
def ranef(res: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> RanefResult:
    """BLUPs of the random effects for a fitted model; see :func:`_ranef_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'ranef() not implemented for {type(res).__name__}')


@ranef.register
def ranef_rma_uni(
    res: metafopy.models.rma_uni.RMAResult,
    level: float | None = None,
    digits: int | None = None,
    transf: metafopy.methods.blup.Transf | None = None,
    targs: typing.Any = None,
    *,
    na_action: metafopy.methods.blup.NAAction = 'omit',
) -> RanefResult:
    """BLUPs of the random effects of an ``rma.uni`` fit (arguments as in :func:`metafopy.methods.blup.blup`)."""
    out = metafopy.methods.blup.shrinkage_predictions(res, 'ranef', level, digits, transf, targs, na_action)
    return RanefResult(pred=out.pred, se=out.se, pi_lb=out.pi_lb, pi_ub=out.pi_ub, slab=out.slab, digits=out.digits, transf=out.transf)
