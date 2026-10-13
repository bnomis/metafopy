"""Small linear-algebra helpers mirroring metafor's internal utilities."""

import numpy
import numpy.typing

__all__ = [
    'drop_redundant',
    'invcalc',
    'is_intercept',
    'is_nested',
    'is_positive_definite',
    'logdet',
]

FloatArray = numpy.typing.NDArray[numpy.float64]


def invcalc(X: FloatArray, w: FloatArray) -> FloatArray:
    """``(X' W X)^-1`` for diagonal ``W = diag(w)`` via a QR decomposition.

    Mirrors metafor's ``.invcalc()``, which solves ``sqrt(W) X`` by QR rather
    than inverting ``X' W X`` directly for better numerical stability.
    """
    sWX = numpy.sqrt(w)[:, None] * X
    _, R = numpy.linalg.qr(sWX)
    Rinv = numpy.linalg.solve(R, numpy.eye(R.shape[0]))
    return Rinv @ Rinv.T


def is_intercept(x: FloatArray, eps: float = 1e-8) -> bool:
    """True when every element of ``x`` equals 1 (metafor's ``.is.intercept``)."""
    return bool(numpy.all(numpy.abs(x - 1.0) < eps))


def drop_redundant(X: FloatArray) -> numpy.typing.NDArray[numpy.bool_]:
    """Columns that are linear combinations of earlier ones.

    Equivalent to the ``NA`` coefficients R's ``lm(y ~ 0 + X)`` reports: the
    pivoted QR in ``lm`` keeps columns in order and drops each column that adds
    no rank (tolerance 1e-7 as in ``lm.fit``).
    """
    p = X.shape[1]
    drop = numpy.zeros(p, dtype=numpy.bool_)
    kept: list[int] = []
    rank = 0
    for j in range(p):
        cand = [*kept, j]
        r = rank_tol(X[:, cand])
        if r > rank:
            kept.append(j)
            rank = r
        else:
            drop[j] = True
    return drop


def rank_tol(A: FloatArray, tol: float = 1e-7) -> int:
    if A.shape[1] == 0:
        return 0
    # rank via QR with relative tolerance on |diag(R)|, as LINPACK dqrdc2 does
    _, R = numpy.linalg.qr(A)
    d = numpy.abs(numpy.diag(R))
    norms = numpy.linalg.norm(A, axis=0)
    scale = numpy.where(norms > 0, norms, 1.0)
    return int(numpy.sum(d / scale[: d.shape[0]] > tol))


def is_nested(X1: FloatArray, X2: FloatArray) -> bool:
    """True when the column space of ``X2`` lies within that of ``X1``."""
    r1 = numpy.linalg.matrix_rank(X1)
    r2 = numpy.linalg.matrix_rank(X2)
    return bool(max(r1, r2) == numpy.linalg.matrix_rank(numpy.column_stack([X1, X2])))


def is_positive_definite(A: FloatArray, tol: float = numpy.finfo(float).eps) -> bool:
    """True when every eigenvalue of symmetric ``A`` exceeds ``tol`` (metafor's ``.chkpd``)."""
    return bool(numpy.all(numpy.linalg.eigvalsh(A) > tol))


def logdet(A: FloatArray) -> float:
    """Log-determinant (modulus) of a square matrix."""
    _, value = numpy.linalg.slogdet(A)
    return float(value)
