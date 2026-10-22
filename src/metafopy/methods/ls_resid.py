"""BLUPs, random effects and standardised residuals for location-scale models.

The heterogeneity ``tau_i^2`` of each study enters through the fitted values ``tau2`` of the model; otherwise the computations are
those of ``blup.rma.uni``, ``ranef.rma.uni`` and ``rstandard.rma.uni`` in metafor (Raudenbush & Bryk, 1985,
doi:10.3102/10769986010002075).
"""

import typing

import numpy
import numpy.typing

import metafopy.core.linalg
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.influence
import metafopy.methods.ranef
import metafopy.mfopt
import metafopy.models.rma_ls_result

__all__ = ['blup_rma_ls', 'ranef_rma_ls', 'rstandard_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = float(numpy.finfo(float).eps)


@metafopy.methods.blup.blup.register
def blup_rma_ls(
    res: metafopy.models.rma_ls_result.RMALSResult,
    level: float | None = None,
    digits: int | None = None,
    transf: metafopy.methods.blup.Transf | None = None,
    targs: typing.Any = None,
    *,
    na_action: metafopy.methods.blup.NAAction = 'omit',
) -> metafopy.methods.blup.BlupResult:
    """BLUPs of the true effects of a location-scale fit (metafor's ``blup.rma.uni`` for ``rma.ls`` objects).

    The arguments are those of [blup_rma_uni][metafopy.methods.blup.blup_rma_uni]; the shrinkage uses the study-specific ``tau^2`` of the
    scale model. See <https://wviechtb.github.io/metafor/reference/blup.html>.

    Args:
        res: A location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
        level: Level of the prediction intervals in percent (e.g. 95), or a value below 1 read as in metafor (0.95 and 0.05 both give 95 percent).
            ``None`` (the default) uses the level of the model.
        digits: Number of decimals used when the result is printed. ``None`` uses the ``digits`` option, else the ``digits`` of the model.
        transf: A function applied to the predicted values and interval bounds (for example ``numpy.exp``); the standard errors are then NaN.
            ``None`` (the default) applies no transformation.
        targs: Extra argument passed to ``transf`` as ``transf(x, targs)``. ``None`` (the default) calls ``transf(x)``.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of NaN for each, so there is one row per supplied study) or ``'fail'`` (raise an error).

    Returns:
        A [BlupResult][metafopy.BlupResult] with the predictions ``pred``, standard errors ``se`` and prediction interval bounds ``pi_lb``
        and ``pi_ub``, one entry per study.

    Raises:
        metafopy.errors.InputError: the model has user-defined weights, ``na_action`` is unknown or values are missing with ``na_action='fail'``.
    """
    return metafopy.methods.blup.shrinkage_predictions(res, 'blup', level, digits, transf, targs, na_action)


@metafopy.methods.ranef.ranef.register
def ranef_rma_ls(
    res: metafopy.models.rma_ls_result.RMALSResult,
    level: float | None = None,
    digits: int | None = None,
    transf: metafopy.methods.blup.Transf | None = None,
    targs: typing.Any = None,
    *,
    na_action: metafopy.methods.blup.NAAction = 'omit',
) -> metafopy.methods.ranef.RanefResult:
    """BLUPs of the random effects of a location-scale fit (metafor's ``ranef.rma.uni`` for ``rma.ls`` objects).

    The arguments are those of [blup_rma_uni][metafopy.methods.blup.blup_rma_uni]. See <https://wviechtb.github.io/metafor/reference/ranef.html>.

    Args:
        res: A location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
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
    """
    out = metafopy.methods.blup.shrinkage_predictions(res, 'ranef', level, digits, transf, targs, na_action)
    return metafopy.methods.ranef.RanefResult(pred=out.pred, se=out.se, pi_lb=out.pi_lb, pi_ub=out.pi_ub, slab=out.slab, digits=out.digits, transf=out.transf)


@metafopy.methods.influence.rstandard.register
def rstandard_rma_ls(
    model: metafopy.models.rma_ls_result.RMALSResult,
    digits: int = 4,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    type: typing.Literal['marginal', 'conditional'] = 'marginal',
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.influence.ResidualResult:
    """Internally standardised residuals of a location-scale fit (metafor's ``rstandard.rma.uni`` applied to an ``rma.ls`` object, which has no method of
    its own).

    With ``type='marginal'`` the marginal residuals are divided by their standard errors, with ``type='conditional'`` the conditional ones (the
    deviations of the estimates from the BLUPs), both with the study-specific ``tau^2`` of the model.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/rstandard.rma.uni.r>; reference page:
    <https://wviechtb.github.io/metafor/reference/residuals.rma.html>.

    Args:
        model: Location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
        digits: Number of decimals of the printout (default 4; unlike for [rma_uni][metafopy.rma_uni] fits the digits of the model are not used).
        type: ``'marginal'`` (default) for marginal residuals or ``'conditional'`` for conditional residuals. Exactly one of these two values.
        na_action: What to do with studies left out of the fit because of missing values: ``'omit'`` (default) drops them, ``'exclude'`` and ``'pass'``
            keep one row per study with NaN for those left out, ``'fail'`` raises an InputError if there are any. Corresponds to R's
            ``na.action`` option.

    Returns:
        A [ResidualResult][metafopy.ResidualResult] with the residuals ``resid``, their standard errors ``se`` and the standardised values ``z``, labelled
        with the study labels.

    Raises:
        metafopy.errors.InputError: ``type`` is not ``'marginal'`` or ``'conditional'``; ``type='conditional'`` for a model with weights given or
            fitted without weighting; ``na_action`` is not valid, or is ``'fail'`` and studies were left out of the fit.
    """
    metafopy.methods.common.check_na_action(na_action)
    x = model
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if type not in ('marginal', 'conditional'):
        raise metafopy.errors.InputError("type must be 'marginal' or 'conditional'")
    if type == 'conditional' and (x.weights is not None or not x.weighted):
        raise metafopy.errors.InputError('Extraction of conditional residuals not available for models with non-standard weights.')
    H = x.hat_matrix()
    ImH = numpy.eye(x.k) - H
    if type == 'marginal':
        ei = x.yi - x.X @ x.beta
        ei = numpy.where(numpy.abs(ei) < 100 * EPS, 0.0, ei)
        sei = numpy.sqrt(numpy.diag(ImH @ numpy.diag(x.vi + x.tau2) @ ImH.T))
    else:
        li = x.tau2 / (x.tau2 + x.vi)
        ei = x.yi - (li * x.yi + (1 - li) * (x.X @ x.beta))
        sei = numpy.sqrt(x.vi**2 / (x.vi + x.tau2) * (1 - numpy.diag(H)))
    if na_action == 'fail' and not x.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    z = ei / sei
    slab = x.slab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action in ('exclude', 'pass'):
        keep = x.data.keep
        idx = numpy.flatnonzero(keep)
        padded = []
        for arr in (ei, sei, z):
            full = numpy.full(keep.shape[0], numpy.nan)
            full[idx] = arr
            padded.append(full)
        ei, sei, z = padded
        labels = [str(i + 1) for i in range(keep.shape[0])]
        for pos, i in enumerate(idx):
            labels[i] = x.slab[pos]
        slab = tuple(labels)
    return metafopy.methods.influence.ResidualResult(resid=ei, se=sei, z=z, slab=slab, digits=metafopy.mfopt.reduce_digits(digits))
