"""(Restricted) log-likelihood of multivariate/multilevel models and its gradient.

Ports ``.ll.rma.mv()`` of metafor (https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.rma.mv.r)
for the marginal model ``y ~ N(X beta, M)`` with ``M = V + sum_j sigma2_j D_j + (Z_G E_G Z_G') * (Z_G2 Z_G2')
+ (Z_H E_H Z_H') * (Z_H2 Z_H2')`` (Konstantopoulos, 2011, doi:10.1002/jrsm.35). The gradient with respect to the
transformed parameters uses the exact trace formulas for the Gaussian (RE)ML score (Searle, Casella and McCulloch,
1992, *Variance Components*, section 6.6) combined with a central difference of the small structure matrices,
which replaces the derivative-free ``nlminb`` search of metafor.
"""

import dataclasses
import math

import numpy
import numpy.typing
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import metafopy.errors
import metafopy.models.mv_structs

__all__ = ['Fit', 'Layout', 'Problem', 'evaluate', 'final_fit', 'gradient', 'negative_loglik', 'quad_form']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]
type Mat = FloatArray | scipy.sparse.csc_array

LOG2PI = math.log(2 * math.pi)


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """Sizes of the five blocks of the parameter vector ``(sigma2, tau2, rho, gamma2, phi)``."""

    sigma2s: int
    tau2s: int
    rhos: int
    gamma2s: int
    phis: int

    @property
    def total(self) -> int:
        return self.sigma2s + self.tau2s + self.rhos + self.gamma2s + self.phis

    def split(self, par: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        a = self.sigma2s
        b = a + self.tau2s
        c = b + self.rhos
        d = c + self.gamma2s
        return par[:a], par[a:b], par[b:c], par[c:d], par[d : d + self.phis]


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """Everything fixed during the optimisation."""

    Y: FloatArray
    V: Mat
    A: Mat | None
    X: FloatArray
    D_S: tuple[Mat, ...]
    sigma2_arg: FloatArray
    g: metafopy.models.mv_structs.Term | None
    h: metafopy.models.mv_structs.Term | None
    ZZ_g: Mat | None
    ZZ_h: Mat | None
    layout: Layout
    with_s: bool
    reml_full: bool
    sparse: bool
    vccon: dict[str, IntArray] | None = None
    optbeta: bool = False
    beta_arg: FloatArray | None = None
    lambda1: float = 0.0
    lambda2: float = 0.0
    intercept: bool = True

    @property
    def k(self) -> int:
        return int(self.Y.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def offset(self) -> int:
        """Number of leading entries of the parameter vector that are coefficients (``optbeta=TRUE``)."""
        return self.p if self.optbeta else 0

    def fix_beta(self, beta: FloatArray) -> FloatArray:
        """Replace the coefficients that were fixed by the user (``beta`` argument of metafor's ``rma.mv``)."""
        return beta if self.beta_arg is None else numpy.where(numpy.isnan(self.beta_arg), beta, self.beta_arg)

    def penalty(self, beta: FloatArray) -> float:
        """The lasso/ridge penalty ``lambda1 * sum|b| + lambda2 * b'b`` (the intercept is not penalised)."""
        if not self.optbeta or (self.lambda1 <= 0 and self.lambda2 <= 0):
            return 0.0
        b = beta[1:] if self.intercept else beta
        return float(self.lambda1 * numpy.sum(numpy.abs(b)) + self.lambda2 * (b @ b))


@dataclasses.dataclass(frozen=True, slots=True)
class Fit:
    """Result of :func:`final_fit`."""

    beta: FloatArray
    vb: FloatArray
    M: Mat
    ll_ml: float
    ll_reml: float
    sigma2: FloatArray | None
    G: FloatArray | None
    tau2: FloatArray | None
    rho: FloatArray | None
    H: FloatArray | None
    gamma2: FloatArray | None
    phi: FloatArray | None


class Factorization:
    """Cholesky (dense) or sparse LU factorisation of a positive-definite matrix."""

    def __init__(self, M: Mat) -> None:
        self.sparse = scipy.sparse.issparse(M)
        self.ok = True
        self.logdet = math.nan
        if self.sparse:
            try:
                lu = scipy.sparse.linalg.splu(scipy.sparse.csc_array(M), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
            except RuntimeError:
                self.ok = False
                return
            pivots = lu.U.diagonal()
            if not numpy.all(pivots > 0):
                self.ok = False
                return
            self.lu = lu
            self.logdet = float(numpy.sum(numpy.log(pivots)))
        else:
            try:
                self.chol = scipy.linalg.cho_factor(numpy.asarray(M), lower=True, check_finite=False)
            except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError:
                self.ok = False
                return
            self.logdet = 2.0 * float(numpy.sum(numpy.log(numpy.diag(self.chol[0]))))

    def solve(self, B: FloatArray) -> FloatArray:
        if self.sparse:
            return numpy.asarray(self.lu.solve(B), dtype=numpy.float64)
        return scipy.linalg.cho_solve(self.chol, B, check_finite=False)


def hadamard(a: Mat, b: Mat) -> Mat:
    if scipy.sparse.issparse(a):
        return scipy.sparse.csc_array(a.multiply(b))
    return numpy.asarray(a * b)


def quad_form(Z: Mat, E: FloatArray, sparse: bool) -> Mat:
    """``Z E Z'``."""
    if sparse:
        return scipy.sparse.csc_array(Z @ scipy.sparse.csc_array(E) @ Z.T)
    return numpy.asarray(Z @ E @ Z.T)


def outer_cross(Z2: Mat) -> Mat:
    """``tcrossprod(Z2)``, the indicator of rows sharing the same outer level."""
    return scipy.sparse.csc_array(Z2 @ Z2.T) if scipy.sparse.issparse(Z2) else numpy.asarray(Z2 @ Z2.T)


def tie_mean(values: FloatArray, codes: IntArray) -> FloatArray:
    """Replace the entries of each group of ``codes`` by their mean (``vccon``: constraints equating variance components)."""
    out = values.copy()
    for code in numpy.unique(codes):
        sel = codes == code
        out[sel] = values[sel].mean()
    return out


def structure(
    problem: Problem,
    term: metafopy.models.mv_structs.Term,
    names: tuple[str, str],
    v_par: FloatArray,
    r_par: FloatArray,
    vctransf: bool,
    vccov: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``.con.E()`` followed by the ``vccon`` constraints on the variances and correlations named ``names``."""
    v, r, E = metafopy.models.mv_structs.con_e(v_par, r_par, term, vctransf, vccov)
    con = problem.vccon
    if con is None:
        return v, r, E
    if names[0] in con:
        v = tie_mean(v, con[names[0]])
    if names[1] in con:
        r = tie_mean(r, con[names[1]])
    return metafopy.models.mv_structs.con_e(v, r, dataclasses.replace(term, cholesky=False), False, vccov)


@dataclasses.dataclass(slots=True)
class Evaluated:
    M: Mat
    sigma2: FloatArray | None
    tau2: FloatArray | None
    rho: FloatArray | None
    G: FloatArray | None
    gamma2: FloatArray | None
    phi: FloatArray | None
    H: FloatArray | None


def evaluate(problem: Problem, par: FloatArray, vctransf: bool = True, vccov: bool = False) -> Evaluated:
    """Assemble the marginal variance-covariance matrix ``M`` for the parameters ``par`` (without leading coefficients)."""
    lay = problem.layout
    s_par, tau_par, rho_par, gam_par, phi_par = lay.split(par)
    M: Mat = problem.V
    sigma2: FloatArray | None = None
    if problem.with_s:
        raw = s_par
        sigma2 = numpy.where(numpy.isnan(problem.sigma2_arg), numpy.exp(raw) if vctransf else raw, problem.sigma2_arg)
        if not vctransf:
            sigma2 = numpy.where(sigma2 < 0, 0.0, sigma2)
        sigma2 = numpy.where(sigma2 <= metafopy.models.mv_structs.EPS10, 0.0, sigma2)
        if problem.vccon is not None and 'sigma2' in problem.vccon:
            sigma2 = tie_mean(sigma2, problem.vccon['sigma2'])
        for j, D in enumerate(problem.D_S):
            M = M + sigma2[j] * D
    tau2 = rho = G = gamma2 = phi = H = None
    if problem.g is not None and problem.ZZ_g is not None:
        tau2, rho, G = structure(problem, problem.g, metafopy.models.mv_structs.G_NAMES, tau_par, rho_par, vctransf, vccov)
        M = M + hadamard(quad_form(problem.g.Z1, G, problem.sparse), problem.ZZ_g)
    if problem.h is not None and problem.ZZ_h is not None:
        gamma2, phi, H = structure(problem, problem.h, metafopy.models.mv_structs.H_NAMES, gam_par, phi_par, vctransf, vccov)
        M = M + hadamard(quad_form(problem.h.Z1, H, problem.sparse), problem.ZZ_h)
    return Evaluated(M=M, sigma2=sigma2, tau2=tau2, rho=rho, G=G, gamma2=gamma2, phi=phi, H=H)


def gls(
    problem: Problem,
    fac: Factorization,
    X: FloatArray | None = None,
    beta_par: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """``beta``, ``(X'WX)^-1``, ``M^-1 X``, ``M^-1 (y - X beta)`` and the residuals for the weight matrix ``W = M^-1``.

    ``beta`` is the generalised least-squares estimate unless ``beta_par`` gives the coefficients being optimised (``optbeta``);
    coefficients fixed by the user replace either.
    """
    X = problem.X if X is None else X
    MiX = fac.solve(X)
    MiY = fac.solve(problem.Y[:, None])[:, 0]
    XtWX = X.T @ MiX
    stXWX = scipy.linalg.cho_solve(scipy.linalg.cho_factor(XtWX, lower=True, check_finite=False), numpy.eye(X.shape[1]), check_finite=False)
    beta = stXWX @ (X.T @ MiY) if beta_par is None else beta_par
    beta = problem.fix_beta(beta)
    resid = problem.Y - X @ beta
    MiR = MiY - MiX @ beta
    return beta, stXWX, MiX, MiR, resid


def loglik_parts(
    problem: Problem,
    fac: Factorization,
    dofit: bool,
    reml: bool,
    beta_par: FloatArray | None = None,
) -> tuple[float, float, FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(ll_ml, ll_reml, beta, stXWX, MiX, MiR)`` where unrequested log-likelihoods are NaN."""
    k, p = problem.k, problem.p
    beta, stXWX, MiX, MiR, resid = gls(problem, fac, beta_par=beta_par)
    rss = float(resid @ MiR) + problem.penalty(beta)
    ll_ml = ll_reml = math.nan
    if dofit or not reml:
        ll_ml = -0.5 * k * LOG2PI - 0.5 * fac.logdet - 0.5 * rss
    if dofit or reml:
        logdet_xwx = -float(numpy.linalg.slogdet(stXWX)[1])
        const = 0.5 * float(numpy.linalg.slogdet(problem.X.T @ problem.X)[1]) if problem.reml_full else 0.0
        ll_reml = -0.5 * (k - p) * LOG2PI + const - 0.5 * fac.logdet - 0.5 * logdet_xwx - 0.5 * rss
    return ll_ml, ll_reml, beta, stXWX, MiX, MiR


def negative_loglik(problem: Problem, par: FloatArray, reml: bool, vctransf: bool = True, vccov: bool = False) -> float:
    """``-1 *`` the log-likelihood (``+inf`` when ``M`` is not positive definite), as minimised by ``rma.mv``.

    With ``optbeta`` the first ``p`` entries of ``par`` are the coefficients. ``vctransf=False`` takes the variances and
    correlations (``vccov=True``: covariances) on their original scale, as metafor does for the Hessian of the variance components.
    """
    off = problem.offset
    with numpy.errstate(all='ignore'):
        ev = evaluate(problem, par[off:], vctransf, vccov)
        if not numpy.all(numpy.isfinite(ev.M.data if scipy.sparse.issparse(ev.M) else ev.M)):
            return math.inf
        fac = Factorization(ev.M)
        if not fac.ok:
            return math.inf
        ll_ml, ll_reml, *_ = loglik_parts(problem, fac, dofit=False, reml=reml, beta_par=par[:off] if problem.optbeta else None)
    val = ll_reml if reml else ll_ml
    return -val if math.isfinite(val) else math.inf


def final_fit(problem: Problem, par: FloatArray) -> Fit:
    """The ``dofit=TRUE`` branch of ``.ll.rma.mv()``: coefficients, their covariance matrix and both log-likelihoods.

    Raises:
        metafopy.errors.EstimationError: the final variance-covariance matrix is not positive definite.
    """
    off = problem.offset
    ev = evaluate(problem, par[off:])
    fac = Factorization(ev.M)
    if not fac.ok:
        raise metafopy.errors.EstimationError('Final variance-covariance matrix is not positive definite.')
    ll_ml, ll_reml, beta, stXWX, _, _ = loglik_parts(problem, fac, dofit=True, reml=True, beta_par=par[:off] if problem.optbeta else None)
    vb = stXWX
    if problem.A is not None:
        A = problem.A
        X = problem.X
        AX = A @ X
        stXAX = numpy.asarray(scipy.linalg.inv(numpy.asarray(X.T @ AX)), dtype=numpy.float64)
        beta = problem.fix_beta(stXAX @ (X.T @ (A @ problem.Y)))
        resid = problem.Y - X @ beta
        MiR = fac.solve(resid[:, None])[:, 0]
        rss = float(resid @ MiR)
        MAX = ev.M @ AX
        vb = stXAX @ numpy.asarray(AX.T @ MAX) @ stXAX
        k, p = problem.k, problem.p
        MiX = fac.solve(X)
        logdet_xwx = float(numpy.linalg.slogdet(X.T @ MiX)[1])
        ll_ml = -0.5 * k * LOG2PI - 0.5 * fac.logdet - 0.5 * rss
        const = 0.5 * float(numpy.linalg.slogdet(X.T @ X)[1]) if problem.reml_full else 0.0
        ll_reml = -0.5 * (k - p) * LOG2PI + const - 0.5 * fac.logdet - 0.5 * logdet_xwx - 0.5 * rss
    return Fit(
        beta=beta,
        vb=numpy.asarray(vb),
        M=ev.M,
        ll_ml=ll_ml,
        ll_reml=ll_reml,
        sigma2=ev.sigma2,
        G=ev.G,
        tau2=ev.tau2,
        rho=ev.rho,
        H=ev.H,
        gamma2=ev.gamma2,
        phi=ev.phi,
    )


def numeric_gradient(problem: Problem, par: FloatArray, reml: bool, free: numpy.typing.NDArray[numpy.intp]) -> FloatArray:
    """Central-difference gradient of ``-ll``.

    With some coefficients fixed by the user (``beta``) and the rest estimated by generalised least squares the coefficients are
    not at their conditional optimum, so the score formulas of :func:`gradient` do not apply.
    """
    grad = numpy.zeros(free.shape[0])
    for n, i in enumerate(free):
        step = 1e-6 * max(1.0, abs(par[i]))
        up = par.copy()
        dn = par.copy()
        up[i] += step
        dn[i] -= step
        grad[n] = (negative_loglik(problem, up, reml) - negative_loglik(problem, dn, reml)) / (2 * step)
    return grad


def gradient(problem: Problem, par: FloatArray, reml: bool, free: numpy.typing.NDArray[numpy.intp]) -> FloatArray:
    """Gradient of ``-ll`` with respect to the free entries of the (transformed) parameter vector.

    For a parameter ``psi`` with ``dM = dM/dpsi`` the (RE)ML score is ``-1/2 tr(P dM) + 1/2 r' W dM W r`` with ``P = W``
    (ML) or ``P = W - W X (X'WX)^-1 X'W`` (REML) and ``r = y - X beta``. ``dM`` is obtained exactly for the variances of
    the ``~ 1 | id`` terms and by central differences of the (small) structure matrices ``E`` for the others. With
    ``optbeta`` the leading entries of ``par`` are the coefficients, whose gradient is ``-X'W r`` plus that of the penalty.
    """
    off = problem.offset
    if not problem.optbeta and problem.beta_arg is not None and not numpy.all(numpy.isnan(problem.beta_arg)):
        return numeric_gradient(problem, par, reml, free)
    with numpy.errstate(all='ignore'):
        rest = par[off:]
        ev = evaluate(problem, rest)
        fac = Factorization(ev.M)
        if not fac.ok:
            return numpy.full(free.shape[0], numpy.nan)
        _, _, beta, stXWX, MiX, MiR = loglik_parts(problem, fac, dofit=False, reml=reml, beta_par=par[:off] if problem.optbeta else None)
        k = problem.k
        W = fac.solve(numpy.eye(k))
        P = W - MiX @ stXWX @ MiX.T if reml else W
        lay = problem.layout
        grad = numpy.zeros(par.shape[0])

        def score(dM: Mat) -> float:
            if scipy.sparse.issparse(dM):
                dMd = dM.toarray()
            else:
                dMd = dM
            return 0.5 * float(numpy.sum(P * dMd)) - 0.5 * float(MiR @ (dMd @ MiR))

        if problem.optbeta:
            grad[:off] = -(problem.X.T @ MiR)
            if problem.lambda1 > 0 or problem.lambda2 > 0:
                pen = 0.5 * problem.lambda1 * numpy.sign(beta) + problem.lambda2 * beta
                if problem.intercept:
                    pen[0] = 0.0
                grad[:off] += pen

        s_end = lay.sigma2s
        g_lo, g_hi = s_end, s_end + lay.tau2s + lay.rhos
        h_lo = g_hi
        h_hi = h_lo + lay.gamma2s + lay.phis
        free_set = set(free.tolist())
        if problem.with_s and ev.sigma2 is not None:
            tied = None if problem.vccon is None else problem.vccon.get('sigma2')
            for j in range(lay.sigma2s):
                if off + j not in free_set:
                    continue
                if tied is None:
                    if ev.sigma2[j] > 0:
                        grad[off + j] = score(ev.sigma2[j] * problem.D_S[j])
                    continue
                own = math.exp(rest[j])
                if own <= metafopy.models.mv_structs.EPS10:
                    continue
                members = numpy.flatnonzero(tied == tied[j])
                dM: Mat = problem.D_S[int(members[0])] * 0.0
                for m in members:
                    dM = dM + problem.D_S[int(m)]
                grad[off + j] = score(own / members.shape[0] * dM)
        for lo, hi, term, zz, names in (
            (g_lo, g_hi, problem.g, problem.ZZ_g, metafopy.models.mv_structs.G_NAMES),
            (h_lo, h_hi, problem.h, problem.ZZ_h, metafopy.models.mv_structs.H_NAMES),
        ):
            if term is None or zz is None:
                continue
            for i in range(lo, hi):
                if off + i not in free_set:
                    continue
                step = 1e-6 * max(1.0, abs(rest[i]))
                up = rest.copy()
                dn = rest.copy()
                up[i] += step
                dn[i] -= step
                s_u = lay.split(up)
                s_d = lay.split(dn)
                idx = (1, 2) if term is problem.g else (3, 4)
                E_up = structure(problem, term, names, s_u[idx[0]], s_u[idx[1]], True, False)[2]
                E_dn = structure(problem, term, names, s_d[idx[0]], s_d[idx[1]], True, False)[2]
                dE = (E_up - E_dn) / (2 * step)
                grad[off + i] = score(hadamard(quad_form(term.Z1, dE, problem.sparse), zz))
    return grad[free]
