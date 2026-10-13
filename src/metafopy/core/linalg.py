"""Small linear-algebra helpers mirroring metafor's internal utilities."""

import dataclasses
import warnings

import numpy
import numpy.typing

import metafopy.errors

__all__ = [
    'NearPDResult',
    'drop_redundant',
    'invcalc',
    'is_intercept',
    'is_nested',
    'is_positive_definite',
    'is_symmetric',
    'logdet',
    'near_pd',
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


def is_symmetric(A: FloatArray, tol: float = 100 * float(numpy.finfo(float).eps)) -> bool:
    """R's ``isSymmetric()`` for a numeric matrix: square and equal to its transpose up to a mean relative difference of ``tol``."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    At = A.T
    na = numpy.isnan(A)
    if bool(numpy.any(na != numpy.isnan(At))):
        return False
    keep = ~na
    if not bool(numpy.any(keep)):
        return True
    xy = float(numpy.mean(numpy.abs(A[keep] - At[keep])))
    xn = float(numpy.mean(numpy.abs(A[keep])))
    if numpy.isfinite(xn) and xn > tol:
        xy /= xn
    return bool(xy < tol)


@dataclasses.dataclass(frozen=True)
class NearPDResult:
    """Result of :func:`near_pd` (the fields of the list returned by ``Matrix::nearPD``)."""

    mat: FloatArray
    eigenvalues: FloatArray
    corr: bool
    norm_f: float
    iterations: int
    rel_tol: float
    converged: bool


def near_pd(
    x: FloatArray,
    corr: bool = False,
    keep_diag: bool = False,
    do2eigen: bool = True,
    do_sym: bool = False,
    do_dykstra: bool = True,
    ensure_symmetry: bool | None = None,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
    posd_tol: float = 1e-8,
    maxit: int = 100,
) -> NearPDResult:
    """Nearest positive definite matrix by Higham's alternating projections, a port of ``Matrix::nearPD``.

    Higham (2002), https://doi.org/10.1093/imanum/22.3.329. R source:
    https://github.com/cran/Matrix/blob/master/R/nearPD.R (CRAN package Matrix).
    The defaults are those of ``Matrix::nearPD``; the infinity norm is used for the convergence check.

    Raises:
        metafopy.errors.EstimationError: the matrix is negative semi-definite.
    """
    x = numpy.array(x, dtype=numpy.float64)
    if ensure_symmetry is None:
        ensure_symmetry = not is_symmetric(x)
    if ensure_symmetry:
        x = (x + x.T) / 2
    n = x.shape[1]
    diag_x0 = numpy.diag(x).copy()
    d_s = numpy.zeros_like(x)
    xx = x.copy()
    iteration = 0
    converged = False
    conv = float('inf')
    d = numpy.empty(0)
    while iteration < maxit and not converged:
        y = xx
        r = y - d_s if do_dykstra else y
        values, vectors = numpy.linalg.eigh(r)
        d = values[::-1]
        q = vectors[:, ::-1]
        p = d > eig_tol * d[0]
        if not bool(numpy.any(p)):
            raise metafopy.errors.EstimationError('Matrix seems negative semi-definite')
        q = q[:, p]
        xx = (q * d[p]) @ q.T
        if do_dykstra:
            d_s = xx - r
        if do_sym:
            xx = (xx + xx.T) / 2
        if corr:
            numpy.fill_diagonal(xx, 1.0)
        elif keep_diag:
            numpy.fill_diagonal(xx, diag_x0)
        conv = float(numpy.linalg.norm(y - xx, numpy.inf) / numpy.linalg.norm(y, numpy.inf))
        iteration += 1
        converged = conv <= conv_tol
    if not converged:
        warnings.warn(f"'nearPD()' did not converge in {iteration} iterations", stacklevel=2)
    if do2eigen:
        values, vectors = numpy.linalg.eigh(xx)
        d = values[::-1]
        q = vectors[:, ::-1]
        eps = posd_tol * abs(d[0])
        if d[n - 1] < eps:
            d = numpy.where(d < eps, eps, d)
            o_diag = numpy.diag(xx).copy()
            xx = (q * d) @ q.T
            dd = numpy.sqrt(numpy.maximum(eps, o_diag) / numpy.diag(xx))
            xx = dd[:, None] * xx * dd[None, :]
        if corr:
            numpy.fill_diagonal(xx, 1.0)
        elif keep_diag:
            numpy.fill_diagonal(xx, diag_x0)
    return NearPDResult(
        mat=xx,
        eigenvalues=d,
        corr=corr,
        norm_f=float(numpy.linalg.norm(x - xx, 'fro')),
        iterations=iteration,
        rel_tol=conv,
        converged=converged,
    )
