"""Helpers shared by the methods for location-scale models: refitting with some scale coefficients fixed."""

import dataclasses
import typing
import warnings

import numpy
import numpy.typing

import metafopy.errors
import metafopy.models.rma_ls_fit
import metafopy.models.rma_ls_result

type FloatArray = numpy.typing.NDArray[numpy.float64]

# what a failed refit looks like to callers (metafor wraps the refit in try() and records NA)
REFIT_ERRORS = (metafopy.errors.EstimationError, metafopy.errors.InputError, ValueError, ArithmeticError, numpy.linalg.LinAlgError)


def refit_ls(
    res: metafopy.models.rma_ls_result.RMALSResult,
    *,
    alpha: numpy.typing.ArrayLike | None = None,
    beta: numpy.typing.ArrayLike | None = None,
    skiphes: bool = True,
    optbeta: bool | None = None,
    weighted: bool | None = None,
    method: str | None = None,
    X: FloatArray | None = None,
    Z: FloatArray | None = None,
    **overrides: typing.Any,
) -> metafopy.models.rma_ls_result.RMALSResult | None:
    """Refit ``res`` with the same data and settings; ``alpha`` (``NaN`` for free coefficients) fixes scale coefficients.

    Mirrors the refits of ``.profile.rma.ls``, the permutation test and the re-estimated VIFs: the model matrices are those of
    the fit unless ``X`` or ``Z`` replace them, and the Hessian is skipped by default. Returns ``None`` when the fit fails (metafor records ``NA`` then).
    """
    md = res.data
    if X is not None:
        md = dataclasses.replace(
            md, yi=res.yi, vi=res.vi, X=X, X_names=tuple(f'X{j + 1}' for j in range(X.shape[1])), weights=res.weights, intercept=False
        )
    z_names = tuple(res.alpha_names) if Z is None else tuple(f'Z{j + 1}' for j in range(Z.shape[1]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return metafopy.models.rma_ls_fit.fit_rma_ls(
                md,
                res.Z if Z is None else Z,
                z_names,
                method=res.method if method is None else method,
                weighted=res.weighted if weighted is None else weighted,
                test=res.test,
                level=1 - res.level,
                btt=None,
                att=None,
                link=res.link,
                optbeta=res.optbeta if optbeta is None else optbeta,
                alpha=alpha,
                beta=beta,
                control=dataclasses.replace(res.control, **overrides) if overrides else res.control,
                skiphes=skiphes,
            )
    except REFIT_ERRORS:
        return None
