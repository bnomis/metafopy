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
import metafopy.models.glmm_result
import metafopy.models.rma_uni

__all__ = ['RanefResult', 'ranef']


@dataclasses.dataclass(frozen=True, slots=True)
class RanefResult(metafopy.methods.blup.BlupResult):
    """Predicted random effects with standard errors and prediction intervals.

    The attributes are those of [BlupResult][metafopy.BlupResult]: ``pred``, ``se``, ``pi_lb``, ``pi_ub``, ``slab``, ``digits`` and ``transf``.
    """


@functools.singledispatch
def ranef(res: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> RanefResult:
    """Best linear unbiased predictions (BLUPs) of the random effects for a fitted model.

    This is a single-dispatch function: the implementation is chosen from the class of ``res``. See
    [ranef_rma_uni][metafopy.methods.ranef.ranef_rma_uni] for [rma_uni][metafopy.rma_uni] fits,
    [ranef_rma_ls][metafopy.methods.ls_resid.ranef_rma_ls] for location-scale models and
    [ranef_rma_mv][metafopy.methods.mv_ranef.ranef_rma_mv] for [rma_mv][metafopy.rma_mv] fits.

    Args:
        res: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``res``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``res``.

    Returns:
        What the implementation returns: a [RanefResult][metafopy.RanefResult] (or a dict of data frames for `rma_mv` fits).

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [ranef](https://wviechtb.github.io/metafor/reference/ranef.html)
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
    """BLUPs of the random effects of an ``rma.uni`` fit (metafor's ``ranef.rma.uni``).

    The prediction for study ``i`` is the BLUP of its deviation from the model's fitted value (the random effect). For equal- and
    fixed-effects models all random effects are 0. The arguments are those of [blup_rma_uni][metafopy.methods.blup.blup_rma_uni].

    Args:
        res: A model fitted by [rma_uni][metafopy.rma_uni].
        level: Level of the prediction intervals in percent (e.g. 95), or a value below 1 read as in metafor (0.95 and 0.05 both give 95 percent).
            ``None`` (the default) uses the level of the model.
        digits: Number of decimals used when the result is printed. ``None`` uses the ``digits`` option, else the ``digits`` of the model.
        transf: A function applied to the predicted values and interval bounds (for example ``numpy.exp``); the standard errors are then NaN.
            ``None`` (the default) applies no transformation.
        targs: Extra argument passed to ``transf`` as ``transf(x, targs)``. ``None`` (the default) calls ``transf(x)``.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of NaN for each, so there is one row per supplied study) or ``'fail'`` (raise an error).

    Returns:
        A [RanefResult][metafopy.RanefResult] with the predicted random effects ``pred``, standard errors ``se`` and prediction interval
        bounds ``pi_lb`` and ``pi_ub``, one entry per study.

    Raises:
        metafopy.errors.InputError: the model has user-defined weights, ``na_action`` is unknown or values are missing with ``na_action='fail'``.

    Metafor reference:
        [ranef](https://wviechtb.github.io/metafor/reference/ranef.html)
    """
    out = metafopy.methods.blup.shrinkage_predictions(res, 'ranef', level, digits, transf, targs, na_action)
    return RanefResult(pred=out.pred, se=out.se, pi_lb=out.pi_lb, pi_ub=out.pi_ub, slab=out.slab, digits=out.digits, transf=out.transf)


@ranef.register
def ranef_rma_glmm(res: metafopy.models.glmm_result.GLMMResult, *args: typing.Any, **kwargs: typing.Any) -> RanefResult:
    """Not available: metafor has no ``ranef`` method for ``rma.glmm`` objects (only ``ranef.rma.uni`` and ``ranef.rma.mv``).

    Args:
        res: A model fitted by [rma_glmm][metafopy.rma_glmm].
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        Nothing; the function always raises.

    Raises:
        metafopy.errors.NotSupportedError: always, as in metafor ("no applicable method for 'ranef' applied to an object of class c('rma.glmm', 'rma')").
    """
    raise metafopy.errors.NotSupportedError(
        "no applicable method for 'ranef' applied to an object of class c('rma.glmm', 'rma'): metafor has none for rma.glmm models"
    )
