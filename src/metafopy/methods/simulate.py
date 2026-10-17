"""``simulate()``: outcomes drawn from a fitted ``rma_uni``, location-scale or ``rma_mv`` model.

Port of metafor's ``simulate.rma`` (https://github.com/wviechtb/metafor/blob/master/R/simulate.rma.r;
reference page https://wviechtb.github.io/metafor/reference/simulate.rma.html) and of its ``.mvrnorm``
helper, the eigen-based draw of ``MASS::mvrnorm``. The normal deviates come from the R-compatible generator in
:mod:`metafopy.core.rng`, so ``seed=`` (or :func:`metafopy.set_seed`) reproduces R's draws.
"""

import collections.abc
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.core.rng
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['mvrnorm', 'simulate']

type FloatArray = numpy.typing.NDArray[numpy.float64]


def mvrnorm(n: int, mu: FloatArray, Sigma: FloatArray, rng: metafopy.core.rng.RRandom) -> FloatArray:
    """metafor's ``.mvrnorm`` (``MASS::mvrnorm`` without its checks): an ``n`` x ``p`` matrix of draws.

    The ``p * n`` deviates fill the rows in turn and are multiplied by the symmetric square root of ``Sigma``
    from its eigen decomposition, negative eigenvalues set to 0.
    """
    p = Sigma.shape[0]
    values, vectors = numpy.linalg.eigh(Sigma)
    root = vectors @ (vectors.T * numpy.sqrt(numpy.maximum(values, 0.0))[:, None])
    Z = rng.rnorm(p * n).reshape(n, p)
    out: FloatArray = Z @ root.T + mu[None, :]
    return out


def marginal_vcov(obj: typing.Any) -> FloatArray:
    """The marginal variance-covariance matrix ``M`` of the outcomes."""
    if isinstance(obj, metafopy.models.mv_result.RMAMVResult):
        M = obj.M
        return numpy.asarray(M.toarray() if scipy.sparse.issparse(M) else M, dtype=numpy.float64)
    return numpy.diag(numpy.asarray(obj.vi, dtype=numpy.float64) + numpy.asarray(obj.tau2, dtype=numpy.float64))


def simulate(
    object: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult | metafopy.models.mv_result.RMAMVResult,
    nsim: int = 1,
    seed: int | None = None,
    olim: collections.abc.Sequence[float] | None = None,
    *,
    withvb: bool = False,
    rng: metafopy.core.rng.RRandom | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.DataFrame:
    """Simulate ``nsim`` sets of outcomes from a fitted model, as metafor's ``simulate.rma``.

    The outcomes are multivariate normal with mean ``X @ beta`` and the model's marginal variance-covariance
    matrix (``diag(vi + tau2)`` for ``rma_uni`` and location-scale fits, ``M`` for ``rma_mv``); ``withvb=True``
    adds ``X vb X'`` (metafor's ``withvb`` passed through ``...``). ``olim`` clamps the values to an interval.
    With ``seed`` a generator seeded with it is used, leaving the global one (:func:`metafopy.set_seed`)
    untouched, as R restores ``.Random.seed``; without it the global generator (or ``rng``) advances. The columns are ``sim_1``, ``sim_2``, ...;
    ``attrs['seed']`` holds ``seed``. ``na_action`` stands for R's ``options(na.action=)``: with ``'omit'`` (default) the rows are the studies used in the
    fit, with ``'exclude'``, ``'pass'`` and ``'fail'`` (which R does not enforce here) one row for every study of the subset, NaN for the studies left out.

    Raises:
        metafopy.errors.InputError: the model type is not supported or an argument is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    # R's simulate.rma also receives fits it rejects, so the type is checked at run time
    fit: typing.Any = object
    unsupported = (
        metafopy.models.glmm_result.GLMMResult,
        metafopy.models.binary.FixedBinaryResult,
        metafopy.models.sel_result.SelmodelResult,
    )
    if isinstance(fit, unsupported):
        raise metafopy.errors.InputError(f'Method not available for objects of class {type(fit).__name__}.')
    if not isinstance(fit, (metafopy.models.rma_uni.RMAResult, metafopy.models.rma_ls_result.RMALSResult, metafopy.models.mv_result.RMAMVResult)):
        raise metafopy.errors.ArgumentError(f'simulate() is not available for {type(fit).__name__}')
    if nsim <= 0:
        raise metafopy.errors.InputError("Argument 'nsim' must be >= 1.")
    gen = metafopy.core.rng.RRandom(seed) if seed is not None else (rng if rng is not None else metafopy.core.rng.get_rng())
    X = numpy.asarray(fit.X, dtype=numpy.float64)
    pred = X @ numpy.asarray(fit.beta, dtype=numpy.float64)
    Sigma = marginal_vcov(fit)
    if withvb:
        XvX = X @ numpy.asarray(fit.vb, dtype=numpy.float64) @ X.T
        Sigma = Sigma + (XvX + XvX.T) / 2
    val = mvrnorm(nsim, pred, Sigma, gen).T
    if olim is not None:
        if len(olim) != 2:
            raise metafopy.errors.InputError("Argument 'olim' must be of length 2.")
        lo, hi = sorted(float(v) for v in olim)
        val = numpy.clip(val, lo, hi)
    slab = list(fit.slab)
    if na_action != 'omit':
        rows = metafopy.methods.fitted.rows_of(fit)
        padded = numpy.full((rows.not_na.shape[0], nsim), numpy.nan)
        padded[rows.not_na] = val
        val, slab = padded, list(rows.slab)
    index = slab if len(slab) == val.shape[0] else None
    out = pandas.DataFrame(val, index=index, columns=[f'sim_{j + 1}' for j in range(nsim)])
    out.attrs['seed'] = seed
    return out
