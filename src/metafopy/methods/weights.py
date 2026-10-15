"""Weights of fitted models (metafor's ``weights.rma.uni`` and ``weights.rma.mv``).

``type='diagonal'`` gives the percentage weights, ``'matrix'`` the weight matrix ``W`` (for ``rma_uni`` and location-scale fits diagonal,
with ``1 / (vi + tau2)`` on the diagonal; for ``rma_mv`` fits the user-supplied matrix or the inverse of the marginal variance-covariance
matrix) and, for intercept-only ``rma_mv`` fits, ``'rowsum'`` the percentage row sums of ``W``. ``rma_mh`` and ``rma_peto`` fits implement the
same in :meth:`metafopy.models.binary.FixedBinaryResult.weights_`. Selection-model and ``rma_glmm`` fits have no weights.

R sources: https://github.com/wviechtb/metafor/blob/master/R/weights.rma.uni.r, https://github.com/wviechtb/metafor/blob/master/R/weights.rma.mv.r;
reference page: https://wviechtb.github.io/metafor/reference/weights.rma.html.
"""

import typing

import numpy
import numpy.typing

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni

__all__ = ['weights_mv', 'weights_uni', 'weights_unsupported']

type FloatArray = numpy.typing.NDArray[numpy.float64]

UNI_TYPES = ('diagonal', 'matrix')
MV_TYPES = ('diagonal', 'matrix', 'rowsum')


def percent(wi: FloatArray, not_na: numpy.typing.NDArray[numpy.bool_], na_action: metafopy.methods.common.NAAction) -> FloatArray:
    full = numpy.full(not_na.shape[0], numpy.nan)
    full[not_na] = wi / wi.sum() * 100
    return metafopy.methods.common.apply_na_action(full, not_na, na_action, 'weights')


def matrix(W: FloatArray, not_na: numpy.typing.NDArray[numpy.bool_], na_action: metafopy.methods.common.NAAction) -> FloatArray:
    full = numpy.full((not_na.shape[0], not_na.shape[0]), numpy.nan)
    full[numpy.ix_(not_na, not_na)] = W
    return metafopy.methods.common.apply_na_action(full, not_na, na_action)


def weights_uni(
    x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult,
    type: typing.Literal['diagonal', 'matrix'] = 'diagonal',
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> FloatArray:
    """Percentage weights or the weight matrix of an ``rma_uni`` or location-scale fit; see :func:`metafopy.models.rma_uni.RMAResult.weights_`."""
    metafopy.methods.common.check_na_action(na_action)
    kind = metafopy.methods.common.match_arg(type, UNI_TYPES)
    if x.weighted:
        wi = numpy.asarray(1.0 / (x.vi + x.tau2) if x.weights is None else x.weights, dtype=numpy.float64)
    else:
        wi = numpy.full(x.k, 1.0 / x.k)
    not_na = metafopy.methods.common.uni_not_na(x)
    if kind == 'diagonal':
        return percent(wi, not_na, na_action)
    return matrix(numpy.diag(wi), not_na, na_action)


def weights_mv(
    x: metafopy.models.mv_result.RMAMVResult,
    type: typing.Literal['diagonal', 'matrix', 'rowsum'] = 'diagonal',
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> FloatArray:
    """Percentage weights, the weight matrix or the percentage row sums of an ``rma_mv`` fit; see :func:`metafopy.models.mv_result.RMAMVResult.weights_`."""
    metafopy.methods.common.check_na_action(na_action)
    kind = metafopy.methods.common.match_arg(type, MV_TYPES)
    W = x.weight_matrix()
    if kind == 'matrix':
        return matrix(W, x.not_na, na_action)
    if kind == 'rowsum':
        if not x.int_only:
            raise metafopy.errors.InputError('Row-sum weights are only meaningful for intercept-only models.')
        return percent(W.sum(axis=1), x.not_na, na_action)
    return percent(numpy.diag(W), x.not_na, na_action)


def weights_unsupported(x: typing.Any) -> typing.NoReturn:
    """Stop as metafor's ``weights.rma.glmm`` and ``weights.rma.uni`` do for the classes they are not available for.

    Raises:
        metafopy.errors.NotSupportedError: always.
    """
    metafopy.methods.fitview.check_class(x, notav=('rma.glmm', 'rma.uni.selmodel'))
    raise metafopy.errors.NotSupportedError(f'Method not available for objects of class "{type(x).__name__}".')
