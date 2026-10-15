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
    """BLUPs of the true effects of a location-scale fit (arguments as in :func:`metafopy.methods.blup.blup`)."""
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
    """BLUPs of the random effects of a location-scale fit (arguments as in :func:`metafopy.methods.blup.blup`)."""
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
    """Standardised residuals of a location-scale fit; ``type`` is ``'marginal'`` or ``'conditional'``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
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
    return metafopy.methods.influence.ResidualResult(resid=ei, se=sei, z=z, slab=slab, digits=digits)
