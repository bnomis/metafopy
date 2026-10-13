"""Cluster-robust variance estimation with small-sample corrections (a port of what metafor uses from ``clubSandwich``).

The CR0, CR1, CR1p, CR1S, CR2 and CR3 variance estimators, Satterthwaite degrees of freedom and saddlepoint p-values for
individual coefficients (Kuonen, 1999, doi:10.1093/biomet/86.4.929) and the ``chi-sq``, ``Naive-F``, ``Naive-Fp``, ``HTA``,
``HTB``, ``HTZ``, ``EDF`` and ``EDT`` Wald tests, following Pustejovsky and Tipton (2018), doi:10.1080/07350015.2016.1247004, and
the R package (https://github.com/jepusto/clubSandwich, GPL >= 2, version 0.7.0; ``R/saddlepoint.R`` and ``R/Wald-test.R``).

Everything works on per-cluster lists: ``X_list`` (``n_j x p``), ``W_list`` (the working weight matrices ``n_j x n_j``)
and ``theta_list`` (the target variance matrices ``n_j x n_j``).
"""

import collections.abc
import dataclasses
import math

import numpy
import numpy.typing
import scipy.linalg
import scipy.stats

import metafopy.errors

# auditme: accept[non-alphabetical-literal] ruff sorts the constants first
__all__ = [
    'CR_TYPES',
    'WALD_TESTS',
    'Sandwich',
    'WaldResult',
    'coef_df',
    'matrix_power',
    'saddlepoint_p',
    'saddlepoint_pval',
    'satterthwaite_df',
    'uniroot',
    'vcov_cr',
    'wald_test',
]

FloatArray = numpy.typing.NDArray[numpy.float64]
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
CR_TYPES = ('CR0', 'CR1', 'CR1p', 'CR1S', 'CR2', 'CR3')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
WALD_TESTS = ('chi-sq', 'Naive-F', 'Naive-Fp', 'HTA', 'HTB', 'HTZ', 'EDF', 'EDT')


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
    """Degrees of freedom of the t-tests for all coefficients (``z``: inf, ``naive-t``: J-1, ``naive-tp``: J-p, ``Satterthwaite``).

    Raises:
        metafopy.errors.InputError: the test is unknown.
    """
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


def f_sf(x: float, df1: float, df2: float) -> float:
    """Upper tail of the F distribution (``df2 = inf`` gives the scaled chi-square tail, as ``pf`` in R)."""
    if math.isinf(df2):
        return float(scipy.stats.chi2.sf(x * df1, df1))
    return float(scipy.stats.f.sf(x, df1, df2))


def covariance_array(B: FloatArray) -> FloatArray:
    """``clubSandwich:::covariance_array``: the ``q x q x q x q`` array of covariances from ``B[s, t, j, k]``."""
    return numpy.asarray(numpy.einsum('svjk,tujk->stuv', B, B) + numpy.einsum('sujk,tvjk->stuv', B, B))


def wald_test(sw: Sandwich, beta: FloatArray, C: FloatArray, test: str = 'HTZ', null_constant: float = 0.0) -> WaldResult:
    """Cluster-robust Wald test of ``C beta = null_constant`` (``clubSandwich:::Wald_testing``).

    ``test`` is one of ``chi-sq``, ``Naive-F``, ``Naive-Fp``, ``HTA``, ``HTB``, ``HTZ`` (Tipton and Pustejovsky, 2015), ``EDF`` or
    ``EDT`` (Pustejovsky and Tipton, 2018, doi:10.1080/07350015.2016.1247004). As in R, ``EDT`` ignores ``null_constant``.

    Raises:
        metafopy.errors.InputError: the test is not supported.
    """
    if test not in WALD_TESTS:
        raise metafopy.errors.InputError(f'Wald test {test!r} is not supported; use one of {", ".join(WALD_TESTS)}.')
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
        return WaldResult(test, Q / q, 1.0, q, float(J - 1), f_sf(Q / q, q, J - 1))
    if test == 'Naive-Fp':
        return WaldResult(test, Q / q, 1.0, q, float(J - p), f_sf(Q / q, q, J - p))
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
    if test == 'HTZ':
        var_mat = numpy.empty((q, q))
        for s in range(q):
            for t in range(s + 1):
                var_mat[s, t] = var_mat[t, s] = float(numpy.sum(B[s, t] * B[t, s]) + numpy.sum(B[s, s] * B[t, t]))
        F, delta, df, p_val = hotelling(Q, q, q * (q + 1) / float(var_mat.sum()))
        return WaldResult(test, F, delta, q, df, p_val)
    if test in ('HTA', 'HTB'):
        cov_arr = covariance_array(B)
        var_mat = numpy.einsum('abab->ab', cov_arr)
        if test == 'HTA':
            nu = 2 * float(var_mat.sum()) / float(numpy.sum(cov_arr**2))
        else:
            lower = numpy.zeros(cov_arr.shape, dtype=bool)
            for s in range(q):
                for t in range(s + 1):
                    for u in range(s + 1):
                        lower[s, t, u, : (t if u == s else u) + 1] = True
            nu = 2 * float(var_mat[numpy.tril_indices(q)].sum()) / float(numpy.sum(cov_arr[lower] ** 2))
        F, delta, df, p_val = hotelling(Q, q, nu)
        return WaldResult(test, F, delta, q, df, p_val)
    cvc = C @ sw.vcov @ C.T
    values, vectors = numpy.linalg.eigh(omega_nsqrt @ ((cvc + cvc.T) / 2) @ omega_nsqrt.T)
    xs = vectors.T @ omega_nsqrt
    df_eig = 1.0 / numpy.array([float(numpy.sum(numpy.einsum('a,abjk,b->jk', x, P, x) ** 2)) for x in xs])
    if test == 'EDF':
        df4 = numpy.maximum(df_eig, 4.1)
        EQ = float(numpy.sum(df4 / (df4 - 2)))
        VQ = float(2 * numpy.sum(df4**2 * (df4 - 1) / ((df4 - 2) ** 2 * (df4 - 4))))
        if q * VQ > 2 * EQ**2:
            delta = (EQ**2 * (q - 2) + 2 * q * VQ) / (EQ * (VQ + EQ**2))
            df = 4 + 2 * EQ**2 * (q + 2) / (q * VQ - 2 * EQ**2)
        else:
            delta = q / EQ
            df = math.inf
        Fstat = delta * Q / q
        return WaldResult(test, Fstat, delta, q, df, f_sf(Fstat, q, df))
    t_j = (vectors.T @ omega_nsqrt @ (C @ beta)) / numpy.sqrt(values)
    a_j = df_eig - 0.5
    b_j = 48 * a_j**2
    c_j = numpy.sqrt(a_j * numpy.log(1 + t_j**2 / df_eig))
    z_j = c_j + (c_j**3 + 3 * c_j) / b_j - (4 * c_j**7 + 33 * c_j**5 + 240 * c_j**3 + 855 * c_j) / (10 * b_j**2 + 8 * b_j * c_j**4 + 1000 * b_j)
    Fstat = float(numpy.mean(z_j**2))
    return WaldResult(test, Fstat, 1.0, q, math.inf, f_sf(Fstat, q, math.inf))


def uniroot(
    f: collections.abc.Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = numpy.finfo(float).eps ** 0.25,
    maxiter: int = 1000,
) -> float:
    """Root of ``f`` in ``[lower, upper]`` by Brent's method, a port of R's ``uniroot`` (``R_zeroin2``, tolerance ``eps^0.25``).

    Raises:
        metafopy.errors.EstimationError: ``f`` does not change sign over the interval, or the iteration limit is reached.
    """
    big = float(numpy.finfo(float).max)

    def evaluate(x: float) -> float:
        # R's zeroin2 replaces infinite function values by the largest finite value, keeping the sign
        return float(numpy.clip(f(x), -big, big))

    fa, fb = evaluate(lower), evaluate(upper)
    if math.isnan(fa) or math.isnan(fb):
        raise metafopy.errors.EstimationError('f() values at the end points of the interval are not finite.')
    if fa * fb > 0:
        raise metafopy.errors.EstimationError('f() values at the end points are not of opposite sign.')
    if fa == 0.0:
        return lower
    if fb == 0.0:
        return upper
    eps = float(numpy.finfo(float).eps)
    a, b, c, fc = lower, upper, lower, fa
    for _ in range(maxiter + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * eps * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                pp = cb * t1
                qq = 1.0 - t1
            else:
                qq = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                pp = t2 * (cb * qq * (qq - t1) - (b - a) * (t1 - 1.0))
                qq = (qq - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if pp > 0:
                qq = -qq
            else:
                pp = -pp
            if pp < (0.75 * cb * qq - abs(tol_act * qq) / 2) and pp < abs(prev_step * qq / 2):
                new_step = pp / qq
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = evaluate(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    raise metafopy.errors.EstimationError('uniroot() did not converge.')


def saddlepoint_pval(t: float, Q: FloatArray, eps: float = 1e-10) -> tuple[float, float]:
    """``clubSandwich:::saddlepoint_pval``: the saddlepoint approximation (Kuonen, 1999) to the two-sided p-value of a t-statistic.

    ``Q`` is the symmetric ``J x J`` matrix of the quadratic form in the variance estimator. Returns ``(s, p_val)`` with
    ``s`` the root of the saddlepoint equation (NaN when ``t`` is zero).
    """
    if abs(t) < eps:
        return math.nan, 1.0
    eig = numpy.maximum(0.0, numpy.linalg.eigvalsh((Q + Q.T) / 2))
    g = numpy.concatenate([[1.0], -(t**2) * eig / eig.sum()])

    def s_eq(s: float) -> float:
        with numpy.errstate(divide='ignore', invalid='ignore'):
            return float(numpy.sum(g / (1 - 2 * g * s)))

    s_range = (1 / (2 * float(g.min())), 0.0) if t**2 < 1 else (0.0, 1 / (2 * float(g.max())))
    s = uniroot(s_eq, *s_range)
    if abs(s) > 0.01:
        r = math.copysign(1.0, s) * math.sqrt(float(numpy.sum(numpy.log(1 - 2 * g * s))))
        q = s * math.sqrt(2 * float(numpy.sum(g**2 / (1 - 2 * g * s) ** 2)))
        p_val = 1 - float(scipy.stats.norm.cdf(r)) - float(scipy.stats.norm.pdf(r)) * (1 / r - 1 / q)
    else:
        p_val = 0.5 - float(numpy.sum(g**3)) / (3 * math.sqrt(math.pi) * float(numpy.sum(g**2)) ** 1.5)
    return s, p_val


def saddlepoint_p(sw: Sandwich, tstat: FloatArray) -> FloatArray:
    """Saddlepoint p-values (``coef_test(test = 'saddlepoint')``) for the t-statistics of all coefficients."""
    G_list, H = get_gh(sw)
    P = p_array(G_list, H)
    return numpy.array([saddlepoint_pval(float(t), P[:, :, i])[1] for i, t in enumerate(tstat)])
