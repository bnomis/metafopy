"""Cluster-robust variance estimation with small-sample corrections (a port of what metafor uses from ``clubSandwich``).

The CR0, CR1, CR1p, CR1S, CR2 and CR3 variance estimators, Satterthwaite degrees of freedom for individual coefficients and
the ``chi-sq``, ``Naive-F``, ``Naive-Fp`` and ``HTZ`` Wald tests, following Pustejovsky and Tipton (2018),
doi:10.1080/07350015.2016.1247004, and the R package (https://github.com/jepusto/clubSandwich, GPL >= 2, version 0.7.0).

Everything works on per-cluster lists: ``X_list`` (``n_j x p``), ``W_list`` (the working weight matrices ``n_j x n_j``)
and ``theta_list`` (the target variance matrices ``n_j x n_j``).
"""

import dataclasses
import math

import numpy
import numpy.typing
import scipy.linalg
import scipy.stats

import metafopy.errors

__all__ = [
    'CR_TYPES',
    'Sandwich',
    'WaldResult',
    'coef_df',
    'matrix_power',
    'satterthwaite_df',
    'vcov_cr',
    'wald_test',
]

FloatArray = numpy.typing.NDArray[numpy.float64]
CR_TYPES = ('CR0', 'CR1', 'CR1p', 'CR1S', 'CR2', 'CR3')


@dataclasses.dataclass(frozen=True, slots=True)
class Sandwich:
    """A cluster-robust variance-covariance matrix with everything needed for degrees-of-freedom calculations."""

    vcov: FloatArray
    type: str
    bread: FloatArray
    v_scale: float
    w_scale: float
    est_mats: list[FloatArray]
    e_list: list[FloatArray]
    target: list[FloatArray]
    inverse_var: bool
    X_list: list[FloatArray]
    W_list: list[FloatArray]
    n_clusters: int


@dataclasses.dataclass(frozen=True, slots=True)
class WaldResult:
    """Result of one cluster-robust Wald test."""

    test: str
    Fstat: float
    delta: float
    df_num: int
    df_denom: float
    p_val: float


def matrix_power(x: FloatArray, p: float, tol: float = -12.0) -> FloatArray:
    """``clubSandwich:::matrix_power``: ``x^p`` of a symmetric matrix via its eigendecomposition (small eigenvalues become 0)."""
    values, vectors = numpy.linalg.eigh(x)
    # eigen() in R sorts decreasingly, which does not matter for the product
    powered = numpy.where(values > 10.0**tol, numpy.where(values > 0, values, 1.0) ** p, 0.0)
    return numpy.asarray((vectors * powered) @ vectors.T)


def total(mats: list[FloatArray]) -> FloatArray:
    """Elementwise sum of equally shaped matrices."""
    return numpy.asarray(numpy.sum(numpy.stack(mats), axis=0))


def chol2inv(a: FloatArray) -> FloatArray:
    """Inverse of a symmetric positive-definite matrix via Cholesky.

    Raises:
        numpy.linalg.LinAlgError: ``a`` is not positive definite.
    """
    c = numpy.linalg.cholesky(a)
    ci = scipy.linalg.solve_triangular(c, numpy.eye(a.shape[0]), lower=True)
    return numpy.asarray(ci.T @ ci)


def inverse_h(M: FloatArray, X_list: list[FloatArray], XW_list: list[FloatArray]) -> list[FloatArray]:
    """``I - X_j M X_j' W_j`` for each cluster (``IH_jj_list``)."""
    return [numpy.eye(x.shape[0]) - x @ M @ xw for x, xw in zip(X_list, XW_list, strict=True)]


def adjustments_for(
    type_: str,
    J: int,
    N: int,
    p: int,
    M_U: FloatArray | None,
    U_list: list[FloatArray],
    UW_list: list[FloatArray],
    theta_list: list[FloatArray],
    inverse_var: bool,
    X_list: list[FloatArray],
    XW_list: list[FloatArray],
) -> float | list[FloatArray] | None:
    if type_ == 'CR0':
        return None
    if type_ == 'CR1':
        return math.sqrt(J / (J - 1))
    if type_ == 'CR1p':
        return math.sqrt(J / (J - p))
    if type_ == 'CR1S':
        return math.sqrt(J * (N - 1) / ((J - 1) * (N - p)))
    if type_ == 'CR3':
        M = chol2inv(total([xw @ x for xw, x in zip(XW_list, X_list, strict=True)]))
        return [numpy.linalg.inv(ih) for ih in inverse_h(M, X_list, XW_list)]
    if M_U is None:
        raise metafopy.errors.InternalError('M_U is needed for CR2')
    chol = [numpy.linalg.cholesky(t).T for t in theta_list]  # R's chol() returns the upper triangle
    if inverse_var:
        ih = inverse_h(M_U, U_list, UW_list)
        G_list = [a @ i @ b @ a.T for a, b, i in zip(chol, theta_list, ih, strict=True)]
    else:
        H_jj = [u @ M_U @ uw for u, uw in zip(U_list, UW_list, strict=True)]
        uwTwu = [uw @ th @ uw.T for uw, th in zip(UW_list, theta_list, strict=True)]
        MUWTWUM = M_U @ total(uwTwu) @ M_U
        G_list = [v @ (thet - h @ thet - thet @ h.T + u @ MUWTWUM @ u.T) @ v.T for thet, h, u, v in zip(theta_list, H_jj, U_list, chol, strict=True)]
    return [v.T @ matrix_power((g + g.T) / 2, -0.5) @ v for v, g in zip(chol, G_list, strict=True)]


def adjust_est_mats(type_: str, est_mats: list[FloatArray], adjustments: float | list[FloatArray] | None) -> list[FloatArray]:
    if type_ == 'CR0':
        return est_mats
    if type_ in ('CR1', 'CR1p', 'CR1S') and isinstance(adjustments, float):
        return [e * adjustments for e in est_mats]
    if isinstance(adjustments, list):
        return [e @ a for e, a in zip(est_mats, adjustments, strict=True)]
    raise metafopy.errors.InternalError(f'unusable adjustment for {type_}')


def vcov_cr(
    X: FloatArray,
    resid: FloatArray,
    cluster: numpy.typing.NDArray[numpy.intp],
    W_list: list[FloatArray],
    theta_list: list[FloatArray] | None,
    bread: FloatArray,
    v_scale: float,
    type_: str = 'CR2',
    inverse_var: bool = False,
    w_scale: float = 1.0,
) -> Sandwich:
    """``clubSandwich:::vcov_CR`` for models without an augmented model matrix (as ``rma.uni`` and ``rma.mv``).

    ``cluster`` holds 0-based cluster codes of the ``N`` rows; ``W_list`` and ``theta_list`` follow the cluster order
    (the target variances default to the inverses of ``W_list`` when ``inverse_var`` is true).

    Raises:
        metafopy.errors.InputError: the type is unknown or there are fewer than two clusters.
    """
    if type_ not in CR_TYPES:
        raise metafopy.errors.InputError(f"Unknown 'vcov' type {type_!r}.")
    J = int(cluster.max()) + 1
    if J < 2:
        raise metafopy.errors.InputError('Cluster-robust variance estimation will not work when the data only includes a single cluster.')
    N, p = X.shape
    members = [numpy.flatnonzero(cluster == j) for j in range(J)]
    X_list = [X[idx] for idx in members]
    XW_list = [x.T @ w for x, w in zip(X_list, W_list, strict=True)]
    if theta_list is None:
        theta_list = [chol2inv(w) for w in W_list]
    M_U: FloatArray | None = None
    if type_ == 'CR2':
        M_U = matrix_power(total([xw @ x for xw, x in zip(XW_list, X_list, strict=True)]), -1.0)
    adjustments = adjustments_for(type_, J, N, p, M_U, X_list, XW_list, theta_list, inverse_var, X_list, XW_list)
    E_list = adjust_est_mats(type_, XW_list, adjustments)
    components = numpy.column_stack([e @ resid[idx] for e, idx in zip(E_list, members, strict=True)])
    meat = components @ components.T * w_scale**2 / v_scale
    vcov = bread @ meat @ bread / v_scale
    return Sandwich(
        vcov=vcov,
        type=type_,
        bread=bread,
        v_scale=v_scale,
        w_scale=w_scale,
        est_mats=XW_list,
        e_list=E_list,
        target=theta_list,
        inverse_var=inverse_var,
        X_list=X_list,
        W_list=W_list,
        n_clusters=J,
    )


def get_gh(sw: Sandwich) -> tuple[list[FloatArray], FloatArray]:
    """``clubSandwich:::get_GH`` (no augmented model matrix): the list ``G`` and the array ``H``."""
    M = sw.bread / sw.v_scale
    p = M.shape[0]
    J = sw.n_clusters
    M_U = sw.w_scale * M
    M_U_ct = numpy.linalg.cholesky((M_U + M_U.T) / 2)
    ME_list = [M @ e for e in sw.e_list]
    G_list = [me @ numpy.linalg.cholesky(theta) for me, theta in zip(ME_list, sw.target, strict=True)]
    if sw.inverse_var:
        return G_list, numpy.stack([me @ u @ M_U_ct for me, u in zip(ME_list, sw.X_list, strict=True)], axis=2)
    H = numpy.empty((3, p, p, J))
    MEU_list = [me @ u for me, u in zip(ME_list, sw.X_list, strict=True)]
    TWU_list = [t @ w @ u for t, w, u in zip(sw.target, sw.W_list, sw.X_list, strict=True)]
    for j in range(J):
        H[0, :, :, j] = MEU_list[j] @ M_U_ct
        H[1, :, :, j] = ME_list[j] @ TWU_list[j] @ M_U_ct
    uwtwu = total([x.T @ w @ twu for x, w, twu in zip(sw.X_list, sw.W_list, TWU_list, strict=True)])
    omega = M_U @ uwtwu @ M_U
    omega_ct = numpy.linalg.cholesky((omega + omega.T) / 2)
    for j in range(J):
        H[2, :, :, j] = MEU_list[j] @ omega_ct
    return G_list, H


def p_array(G_list: list[FloatArray], H: FloatArray) -> FloatArray:
    """``get_P_array(all_terms = FALSE)``: a ``J x J x p`` array."""
    J = len(G_list)
    if H.ndim == 3:
        p = H.shape[0]
        P = numpy.empty((J, J, p))
        for i in range(p):
            P[:, :, i] = -(H[i].T @ H[i])
    else:
        p = H.shape[1]
        P = numpy.empty((J, J, p))
        for i in range(p):
            uf = H[0, i].T @ H[1, i]
            P[:, :, i] = H[2, i].T @ H[2, i] - uf - uf.T
    diag = numpy.array([(g**2).sum(axis=1) for g in G_list]).T  # p x J
    for i in range(p):
        P[numpy.arange(J), numpy.arange(J), i] += diag[i]
    return P


def satterthwaite_df(P: FloatArray) -> FloatArray:
    """Satterthwaite degrees of freedom, one per coefficient, from the ``J x J x p`` array ``P``."""
    p = P.shape[2]
    V_coef = 2 * numpy.array([(P[:, :, i] ** 2).sum() for i in range(p)])
    E_coef = numpy.array([numpy.trace(P[:, :, i]) for i in range(p)])
    return numpy.asarray(2 * E_coef**2 / V_coef)


def coef_df(sw: Sandwich, test: str) -> FloatArray:
    """Degrees of freedom of the t-tests for all coefficients (``z``: inf, ``naive-t``: J-1, ``naive-tp``: J-p, ``Satterthwaite``)."""
    p = sw.vcov.shape[0]
    J = sw.n_clusters
    if test == 'z':
        return numpy.full(p, numpy.inf)
    if test == 'naive-t':
        return numpy.full(p, float(J - 1))
    if test == 'naive-tp':
        return numpy.full(p, float(J - p))
    if test == 'Satterthwaite':
        G_list, H = get_gh(sw)
        return satterthwaite_df(p_array(G_list, H))
    raise metafopy.errors.InputError(f"Unknown test {test!r}; use 'z', 'naive-t', 'naive-tp' or 'Satterthwaite'.")


def p_array_all(G_list: list[FloatArray], H: FloatArray) -> FloatArray:
    """``get_P_array(all_terms = TRUE)``: a ``q x q x J x J`` array for the (already contrast-transformed) ``G`` and ``H``."""
    J = len(G_list)
    if H.ndim == 3:
        q = H.shape[0]
        P = numpy.empty((q, q, J, J))
        for i in range(q):
            for j in range(i, q):
                tmp = -(H[i].T @ H[j])
                tmp[numpy.arange(J), numpy.arange(J)] += numpy.array([float(numpy.sum(g[i] * g[j])) for g in G_list])
                P[i, j] = tmp
                P[j, i] = tmp.T
    else:
        q = H.shape[1]
        P = numpy.empty((q, q, J, J))
        for i in range(q):
            for j in range(i, q):
                tmp = H[2, i].T @ H[2, j] - H[0, i].T @ H[1, j] - H[1, i].T @ H[0, j]
                tmp[numpy.arange(J), numpy.arange(J)] += numpy.array([float(numpy.sum(g[i] * g[j])) for g in G_list])
                P[i, j] = tmp
                P[j, i] = tmp.T
    return P


def hotelling(Q: float, q: int, nu: float) -> tuple[float, float, float, float]:
    """``Hotelling_Tsq``: ``(Fstat, delta, df, p_val)``."""
    delta = max((nu - q + 1) / nu, 0.0)
    df = nu - q + 1
    F = delta * Q / q
    p_val = float(scipy.stats.f.sf(F, q, df)) if df > 0 else math.nan
    return F, delta, df, p_val


def wald_test(sw: Sandwich, beta: FloatArray, C: FloatArray, test: str = 'HTZ', null_constant: float = 0.0) -> WaldResult:
    """Cluster-robust Wald test of ``C beta = null_constant`` (``test``: ``chi-sq``, ``Naive-F``, ``Naive-Fp`` or ``HTZ``).

    Raises:
        metafopy.errors.InputError: the test is not supported.
    """
    if test not in ('chi-sq', 'Naive-F', 'Naive-Fp', 'HTZ'):
        raise metafopy.errors.InputError(f"Wald test {test!r} is not supported; use 'chi-sq', 'Naive-F', 'Naive-Fp' or 'HTZ'.")
    q, p = C.shape
    J = sw.n_clusters
    cov = C @ sw.vcov @ C.T
    try:
        inv_cov = chol2inv((cov + cov.T) / 2)
    except numpy.linalg.LinAlgError:
        return WaldResult(test, math.nan, math.nan, q, math.nan, math.nan)
    c_beta = C @ beta - null_constant
    Q = float(c_beta @ inv_cov @ c_beta)
    if test == 'chi-sq':
        return WaldResult(test, Q / q, 1.0, q, math.inf, float(scipy.stats.chi2.sf(Q, q)))
    if test == 'Naive-F':
        return WaldResult(test, Q / q, 1.0, q, float(J - 1), float(scipy.stats.f.sf(Q / q, q, J - 1)))
    if test == 'Naive-Fp':
        return WaldResult(test, Q / q, 1.0, q, float(J - p), float(scipy.stats.f.sf(Q / q, q, J - p)))
    G_list, H = get_gh(sw)
    G_c = [C @ g for g in G_list]
    if H.ndim == 3:
        H_c = numpy.einsum('qp,puj->quj', C, H)
    else:
        H_c = numpy.stack([numpy.einsum('qp,puj->quj', C, H[i]) for i in range(3)], axis=0)
    P = p_array_all(G_c, H_c)
    omega = numpy.array([[numpy.trace(P[i, j].reshape(J, J)) for j in range(q)] for i in range(q)])
    omega_nsqrt = matrix_power((omega + omega.T) / 2, -0.5)
    B = numpy.einsum('as,stjk,tb->abjk', omega_nsqrt, P, omega_nsqrt)
    var_mat = numpy.empty((q, q))
    for s in range(q):
        for t in range(s + 1):
            var_mat[s, t] = var_mat[t, s] = float(numpy.sum(B[s, t] * B[t, s]) + numpy.sum(B[s, s] * B[t, t]))
    nu_Z = q * (q + 1) / float(var_mat.sum())
    F, delta, df, p_val = hotelling(Q, q, nu_Z)
    return WaldResult(test, F, delta, q, df, p_val)
