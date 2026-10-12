"""Small linear-algebra helpers mirroring metafor's internal utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "drop_redundant",
    "invcalc",
    "is_intercept",
    "is_nested",
    "is_positive_definite",
    "logdet",
]

FloatArray = NDArray[np.float64]


def invcalc(X: FloatArray, w: FloatArray) -> FloatArray:
    """``(X' W X)^-1`` for diagonal ``W = diag(w)`` via a QR decomposition.

    Mirrors metafor's ``.invcalc()``, which solves ``sqrt(W) X`` by QR rather
    than inverting ``X' W X`` directly for better numerical stability.
    """
    sWX = np.sqrt(w)[:, None] * X
    _, R = np.linalg.qr(sWX)
    Rinv = np.linalg.solve(R, np.eye(R.shape[0]))
    return Rinv @ Rinv.T


def is_intercept(x: FloatArray, eps: float = 1e-8) -> bool:
    """True when every element of ``x`` equals 1 (metafor's ``.is.intercept``)."""
    return bool(np.all(np.abs(x - 1.0) < eps))


def drop_redundant(X: FloatArray) -> NDArray[np.bool_]:
    """Columns that are linear combinations of earlier ones.

    Equivalent to the ``NA`` coefficients R's ``lm(y ~ 0 + X)`` reports: the
    pivoted QR in ``lm`` keeps columns in order and drops each column that adds
    no rank (tolerance 1e-7 as in ``lm.fit``).
    """
    p = X.shape[1]
    drop = np.zeros(p, dtype=np.bool_)
    kept: list[int] = []
    rank = 0
    for j in range(p):
        cand = [*kept, j]
        r = _rank(X[:, cand])
        if r > rank:
            kept.append(j)
            rank = r
        else:
            drop[j] = True
    return drop


def _rank(A: FloatArray, tol: float = 1e-7) -> int:
    if A.shape[1] == 0:
        return 0
    # rank via QR with relative tolerance on |diag(R)|, as LINPACK dqrdc2 does
    _, R = np.linalg.qr(A)
    d = np.abs(np.diag(R))
    norms = np.linalg.norm(A, axis=0)
    scale = np.where(norms > 0, norms, 1.0)
    return int(np.sum(d / scale[: d.shape[0]] > tol))


def is_nested(X1: FloatArray, X2: FloatArray) -> bool:
    """True when the column space of ``X2`` lies within that of ``X1``."""
    r1 = np.linalg.matrix_rank(X1)
    r2 = np.linalg.matrix_rank(X2)
    return bool(max(r1, r2) == np.linalg.matrix_rank(np.column_stack([X1, X2])))


def is_positive_definite(A: FloatArray, tol: float = np.finfo(float).eps) -> bool:
    """True when every eigenvalue of symmetric ``A`` exceeds ``tol`` (metafor's ``.chkpd``)."""
    return bool(np.all(np.linalg.eigvalsh(A) > tol))


def logdet(A: FloatArray) -> float:
    """Log-determinant (modulus) of a square matrix."""
    _, value = np.linalg.slogdet(A)
    return float(value)
