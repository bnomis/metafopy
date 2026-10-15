"""What :func:`metafopy.robust` records about a cluster-robust fit."""

import dataclasses

import numpy
import numpy.typing

import metafopy.core.clubsandwich

__all__ = ['RobustInfo']

FloatArray = numpy.typing.NDArray[numpy.float64]
IntArray = numpy.typing.NDArray[numpy.intp]


@dataclasses.dataclass(frozen=True, slots=True)
class RobustInfo:
    """Cluster-robust inference settings and cluster summary of a fit returned by :func:`metafopy.robust`.

    ``robumethod`` is ``'default'`` (the sandwich estimator of metafor) or ``'clubSandwich'``; ``vbest`` names the
    variance estimator (``CR0``, ``CR1``, ``CR1.S1``, ``CR2``, ...); ``dfs`` holds the degrees of freedom used for
    each coefficient and ``tcl`` the number of estimates in each cluster. A ``clubSandwich`` fit also keeps the variance estimator
    (``sandwich``) that ``predict()`` needs for the Satterthwaite degrees of freedom of linear contrasts. ``meat`` is the block diagonal
    matrix of the outer products of the residuals within the clusters (``None`` for ``clubSandwich`` fits, where metafor stores NA).
    """

    n: int
    tcl: IntArray
    vbest: str
    robumethod: str
    dfs: FloatArray
    cluster: numpy.typing.NDArray[numpy.object_]
    coef_test: str | None = None
    conf_test: str | None = None
    wald_test: str | None = None
    sandwich: metafopy.core.clubsandwich.Sandwich | None = None
    meat: FloatArray | None = None
