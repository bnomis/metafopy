"""Binomial and Poisson mixed-effects models fitted as ``GLMMadaptive::mixed_model()`` fits them.

``rma.glmm`` (https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r) can fit its mixed models with
``GLMMadaptive`` (Rizopoulos, https://drizopoulos.github.io/GLMMadaptive/, the algorithm of
https://github.com/drizopoulos/GLMMadaptive/blob/master/R/mixed_fit.R): adaptive Gauss-Hermite quadrature of the
random effects of every cluster, centred at the posterior modes and scaled by a numerical Hessian there,
maximised by a fixed number of EM iterations that re-centre the nodes every ``update_GH_every`` iterations
and, unless the EM iterations met their relative tolerances, by quasi-Newton (BFGS) iterations.

The iterations stop at relative tolerances, and the EM iterations of a random-effects variance can stop far from
the maximum, so the estimates differ from the maximum likelihood estimates (and from lme4's) by more than an
optimiser's noise. This module therefore repeats the algorithm step by step (the BFGS routine is
:func:`metafopy.core.bfgs.optim_bfgs`, the port of R's ``optim()``) instead of maximising the likelihood to
convergence. The covariance matrix of the fixed effects is the inverse of the central-difference Hessian of the
score (with the quadrature nodes held fixed) over all parameters: the fixed effects and the log-Cholesky
parameters of the covariance matrix of the random effects.
"""

import collections.abc
import dataclasses
import functools
import math
import sys
import warnings

import numpy
import numpy.polynomial.hermite
import numpy.typing
import scipy.linalg
import scipy.special

import metafopy.core.bfgs
import metafopy.core.glm
import metafopy.core.glmm
import metafopy.errors

__all__ = ['AdaptiveControl', 'fit_adaptive']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]
type VectorFunction = collections.abc.Callable[[FloatArray], FloatArray]

EPS = sys.float_info.epsilon
LOGIT_LIMIT = 30.0
HESSIAN_WARNING = 'Hessian matrix at convergence is not positive definite; unstable solution.'
HESSIAN_NONFINITE = 'infinite or missing values in Hessian at convergence.'
TIE = 1e-13
LARGE_COEFFICIENT = 'A large coefficient value has been detected during the optimization.'
INITIAL_SCALE = math.sqrt(1.346)
FD_EPS = EPS**0.25
CD_EPS = 1e-3


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveControl:
    """The control list of ``mixed_model()`` (defaults of GLMMadaptive 0.9-7), with ``max_coef_value`` the bound on the coefficients."""

    iter_em: int = 30
    iter_qn_outer: int = 15
    iter_qn: int = 10
    iter_qn_incr: int = 10
    parscale_betas: float = 0.1
    parscale_d: float = 0.01
    tol1: float = 1e-4
    tol2: float = 1e-5
    tol3: float = 1e-8
    update_gh_every: int = 10
    max_coef_value: float = 12.0


def link_inverse(link: metafopy.core.glm.Link, eta: FloatArray) -> FloatArray:
    """``family$linkinv`` of R's ``binomial()`` and ``poisson()``, including their clamping at the extremes."""
    if link == 'logit':
        tmp = numpy.where(eta < -LOGIT_LIMIT, EPS, numpy.where(eta > LOGIT_LIMIT, 1.0 / EPS, numpy.exp(numpy.clip(eta, -LOGIT_LIMIT, LOGIT_LIMIT))))
        return numpy.asarray(tmp / (1.0 + tmp), dtype=numpy.float64)
    if link == 'log':
        return numpy.maximum(numpy.exp(eta), EPS)
    return numpy.asarray(eta, dtype=numpy.float64)


def mean_derivative(link: metafopy.core.glm.Link, eta: FloatArray) -> FloatArray:
    """``family$mu.eta``."""
    if link == 'logit':
        opexp = numpy.exp(numpy.clip(eta, -LOGIT_LIMIT, LOGIT_LIMIT))
        return numpy.where(numpy.abs(eta) > LOGIT_LIMIT, EPS, opexp / (1.0 + opexp) ** 2)
    if link == 'log':
        return numpy.maximum(numpy.exp(eta), EPS)
    return numpy.ones_like(eta)


@dataclasses.dataclass(frozen=True, slots=True)
class Observations:
    """Responses and design of some rows of the data (all of them, or one cluster's)."""

    X: FloatArray
    Z: FloatArray
    y: FloatArray
    size: FloatArray | None
    offset: FloatArray
    family: metafopy.core.glm.Family
    link: metafopy.core.glm.Link

    def take(self, rows: IntArray) -> Observations:
        return Observations(self.X[rows], self.Z[rows], self.y[rows], None if self.size is None else self.size[rows], self.offset[rows], self.family, self.link)

    def spread(self, values: FloatArray, eta: FloatArray) -> FloatArray:
        """``values`` (one per row) shaped to broadcast against ``eta`` (rows, or rows by nodes)."""
        return values.reshape(-1, *([1] * (eta.ndim - 1)))

    def log_density(self, eta: FloatArray) -> FloatArray:
        """Log-density of every observation at the linear predictors ``eta``; NaN where the mean leaves its range."""
        y = self.spread(self.y, eta)
        mu = link_inverse(self.link, eta)
        if self.family == 'binomial':
            assert self.size is not None
            n = self.spread(self.size, eta)
            with numpy.errstate(all='ignore'):
                out = scipy.special.gammaln(n + 1) - scipy.special.gammaln(y + 1) - scipy.special.gammaln(n - y + 1)
                out = out + scipy.special.xlogy(y, mu) + scipy.special.xlog1py(n - y, -mu)
            return numpy.asarray(numpy.where((mu < 0) | (mu > 1), numpy.nan, out), dtype=numpy.float64)
        return numpy.asarray(scipy.special.xlogy(y, mu) - mu - scipy.special.gammaln(y + 1), dtype=numpy.float64)

    def score_eta(self, eta: FloatArray) -> FloatArray:
        """Derivative of the log-density with respect to the linear predictor."""
        y = self.spread(self.y, eta)
        mu = link_inverse(self.link, eta)
        deriv = mean_derivative(self.link, eta)
        with numpy.errstate(all='ignore'):
            if self.family == 'binomial':
                assert self.size is not None
                n = self.spread(self.size, eta)
                return numpy.asarray((y - n * mu) * deriv / (mu * (1.0 - mu)), dtype=numpy.float64)
            return numpy.asarray((y - mu) * deriv / mu, dtype=numpy.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class Context:
    """The model: all observations, one :class:`Observations` per cluster and where each cluster's rows start (``rowsum()``)."""

    obs: Observations
    clusters: list[Observations]
    cluster: IntArray
    starts: IntArray
    structure: metafopy.core.glmm.Structure
    nagq: int
    control: AdaptiveControl

    @property
    def n(self) -> int:
        return len(self.clusters)

    @property
    def r(self) -> int:
        return int(self.obs.Z.shape[1])

    def cluster_sums(self, values: FloatArray) -> FloatArray:
        """Sums of the rows of each cluster (a NaN stays within its cluster)."""
        return numpy.asarray(numpy.add.reduceat(values, self.starts, axis=0), dtype=numpy.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class Nodes:
    """The adaptive quadrature nodes of every cluster (``GHfun``): locations ``b``, their products ``b2``, ``Z b``, weights and log-scales."""

    b: FloatArray
    b2: FloatArray
    ztb: FloatArray
    wgh: FloatArray
    log_wgh: FloatArray
    log_dets: FloatArray
    modes: FloatArray


def unpack_d(par: FloatArray, structure: metafopy.core.glmm.Structure) -> tuple[FloatArray, list[FloatArray]]:
    """The covariance matrix ``D`` of the random effects from its unconstrained parameters, and ``dD / d par`` for each parameter.

    Independent effects use ``log(diag(D))``; otherwise ``D = U'U`` with ``par`` the upper triangle of ``U`` (column by column)
    and the diagonal on the log scale (``chol_transf``).
    """
    if structure == 'independent':
        e = numpy.exp(par)
        return numpy.diag(e), [numpy.diag(numpy.eye(par.shape[0])[k] * e[k]) for k in range(par.shape[0])]
    r = round((-1 + math.sqrt(1 + 8 * par.shape[0])) / 2)
    rows, cols = numpy.tril_indices(r)
    upper = numpy.zeros((r, r))
    upper[cols, rows] = par
    upper[numpy.diag_indices(r)] = numpy.exp(numpy.diag(upper))
    slopes: list[FloatArray] = []
    for k, (a, b) in enumerate(zip(rows, cols, strict=True)):
        d_upper = numpy.zeros((r, r))
        d_upper[b, a] = math.exp(par[k]) if a == b else 1.0
        slopes.append(d_upper.T @ upper + upper.T @ d_upper)
    return upper.T @ upper, slopes


def pack_d(D: FloatArray, structure: metafopy.core.glmm.Structure) -> FloatArray:
    """Inverse of :func:`unpack_d`."""
    if structure == 'independent':
        return numpy.log(numpy.diag(D))
    upper = numpy.linalg.cholesky(D).T
    upper[numpy.diag_indices_from(upper)] = numpy.log(numpy.diag(upper))
    return numpy.asarray(upper.T[numpy.tril_indices_from(upper)], dtype=numpy.float64)


def d_entries(D: FloatArray, structure: metafopy.core.glmm.Structure) -> FloatArray:
    """The entries of ``D`` the EM iterations compare: its diagonal for independent effects, else ``D[lower.tri(D, TRUE)]``."""
    return numpy.diag(D).copy() if structure == 'independent' else numpy.asarray(D[numpy.tril_indices_from(D)], dtype=numpy.float64)


def quadrature_grid(points: int, dim: int) -> tuple[FloatArray, FloatArray]:
    """Nodes of the product Gauss-Hermite rule (weight ``exp(-x^2)``) and the weights ``wGH`` of ``GHfun`` that undo the density."""
    x, w = numpy.polynomial.hermite.hermgauss(points)
    mesh = numpy.meshgrid(*([x] * dim), indexing='ij')
    wmesh = numpy.meshgrid(*([w] * dim), indexing='ij')
    grid = numpy.column_stack([m.ravel() for m in mesh])
    weights = numpy.prod([m.ravel() for m in wmesh], axis=0)
    return grid, 2.0 ** (dim / 2.0) * weights * numpy.exp(numpy.sum(grid**2, axis=1))


def central_hessian(score: VectorFunction, x: FloatArray) -> FloatArray:
    """``cd_vec()``: the symmetrised central difference of the vector function ``score`` with relative step 1e-3."""
    n = x.shape[0]
    res = numpy.zeros((n, n))
    ex = numpy.maximum(numpy.abs(x), 1.0)
    for i in range(n):
        x1, x2 = x.copy(), x.copy()
        x1[i] = x[i] + CD_EPS * ex[i]
        x2[i] = x[i] - CD_EPS * ex[i]
        res[:, i] = (score(x1) - score(x2)) / (x1[i] - x2[i])
    return 0.5 * (res + res.T)


def forward_hessian(score: VectorFunction, x: FloatArray) -> FloatArray:
    """``fd_vec()``: the symmetrised forward difference of the vector function ``score``."""
    n = x.shape[0]
    res = numpy.zeros((n, n))
    ex = numpy.maximum(numpy.abs(x), 1.0)
    f0 = score(x)
    for i in range(n):
        x1 = x.copy()
        x1[i] = x[i] + FD_EPS * ex[i]
        res[:, i] = (score(x1) - f0) / (x1[i] - x[i])
    return 0.5 * (res + res.T)


def near_pd(M: FloatArray, eig_tol: float = 1e-6, conv_tol: float = 1e-7, posd_tol: float = 1e-8, maxits: int = 100) -> FloatArray:
    """GLMMadaptive's ``nearPD()`` (Higham's alternating projections)."""

    def inorm(x: FloatArray) -> float:
        return float(numpy.max(numpy.sum(numpy.abs(x), axis=1)))

    n = M.shape[0]
    U = numpy.zeros((n, n))
    X = M.copy()
    iteration = 0
    converged = False
    while iteration < maxits and not converged:
        Y = X
        T = Y - U
        d, Q = numpy.linalg.eigh(Y)
        d, Q = d[::-1], Q[:, ::-1]
        keep = d > eig_tol * d[0]
        QQ = Q[:, keep]
        X = QQ @ numpy.diag(d[keep]) @ QQ.T
        U = X - T
        X = (X + X.T) / 2
        converged = inorm(Y - X) / inorm(Y) <= conv_tol
        iteration += 1
    X = (X + X.T) / 2
    d, Q = numpy.linalg.eigh(X)
    d, Q = d[::-1], Q[:, ::-1]
    floor = posd_tol * abs(d[0])
    if d[n - 1] < floor:
        d = numpy.where(d < floor, floor, d)
        diag = numpy.diag(X).copy()
        X = Q @ (d[:, None] * Q.T)
        scale = numpy.sqrt(numpy.maximum(floor, diag) / numpy.diag(X))
        X = scale[:, None] * X * scale[None, :]
    return numpy.asarray((X + X.T) / 2, dtype=numpy.float64)


def find_mode(ctx: Context, i: int, start: FloatArray, beta: FloatArray, inv_d: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Posterior mode of the random effects of cluster ``i`` (``optim()`` BFGS) and the central-difference Hessian of its negative log-posterior there."""
    o = ctx.clusters[i]
    fixed = o.X @ beta + o.offset

    def neg_log_posterior(b: FloatArray) -> float:
        return float(-numpy.nansum(o.log_density(fixed + o.Z @ b)) + 0.5 * b @ inv_d @ b)

    def score(b: FloatArray) -> FloatArray:
        return numpy.asarray(-o.Z.T @ o.score_eta(fixed + o.Z @ b) + inv_d @ b, dtype=numpy.float64)

    with numpy.errstate(all='ignore'):
        mode = metafopy.core.bfgs.optim_bfgs(neg_log_posterior, start, gr=score).par
        return mode, central_hessian(score, mode)


def make_nodes(ctx: Context, post_modes: FloatArray, beta: FloatArray, D: FloatArray) -> Nodes:
    """``GHfun()``: centre and scale the quadrature nodes of every cluster.

    Raises:
        metafopy.errors.EstimationError: a Hessian of the log-posterior is not positive definite.
    """
    inv_d = numpy.linalg.inv(D)
    grid, wgh = quadrature_grid(ctx.nagq, ctx.r)
    modes = numpy.empty_like(post_modes)
    b = numpy.empty((ctx.n, grid.shape[0], ctx.r))
    log_dets = numpy.empty(ctx.n)
    for i in range(ctx.n):
        modes[i], hess = find_mode(ctx, i, post_modes[i], beta, inv_d)
        try:
            lower = numpy.linalg.cholesky(hess)
        except numpy.linalg.LinAlgError as exc:
            raise metafopy.errors.EstimationError('the Hessian of the posterior of the random effects is not positive definite') from exc
        b[i] = math.sqrt(2.0) * scipy.linalg.solve_triangular(lower.T, grid.T, lower=False).T + modes[i]
        log_dets[i] = -float(numpy.sum(numpy.log(numpy.diag(lower))))
    ztb = numpy.einsum('rm,rgm->rg', ctx.obs.Z, b[ctx.cluster])
    products = numpy.einsum('nga,ngb->ngab', b, b).transpose(0, 1, 3, 2).reshape(ctx.n, grid.shape[0], ctx.r * ctx.r)
    return Nodes(b=b, b2=products, ztb=ztb, wgh=wgh, log_wgh=numpy.log(wgh), log_dets=log_dets, modes=modes)


def log_density_re(b: FloatArray, D: FloatArray) -> FloatArray:
    """Log-density of the random effects at the nodes ``b`` (clusters by nodes by dimension)."""
    inv = numpy.linalg.inv(D)
    quad = numpy.einsum('ngr,rs,ngs->ng', b, inv, b)
    return numpy.asarray(-0.5 * (D.shape[0] * math.log(2.0 * math.pi) + numpy.linalg.slogdet(D)[1] + quad), dtype=numpy.float64)


def joint(ctx: Context, nodes: Nodes, beta: FloatArray, D: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``log p(y_i, b) + log p(b)`` at the nodes (clusters by nodes) and ``log p(y_i)`` (the log-likelihood contributions without the log-scales)."""
    eta = (ctx.obs.X @ beta + ctx.obs.offset)[:, None] + nodes.ztb
    with numpy.errstate(all='ignore'):
        lyb = ctx.cluster_sums(ctx.obs.log_density(eta)) + log_density_re(nodes.b, D)
        lpy = scipy.special.logsumexp(lyb + nodes.log_wgh, axis=1)
    return lyb, numpy.asarray(lpy, dtype=numpy.float64)


def negative_loglik(ctx: Context, nodes: Nodes, theta: FloatArray, q: int) -> float:
    """``logLik_mixed()``: minus the log-likelihood with the nodes held fixed."""
    D = unpack_d(theta[q:], ctx.structure)[0]
    _, lpy = joint(ctx, nodes, theta[:q], D)
    return float(-numpy.nansum(lpy + nodes.log_dets))


def gradient(ctx: Context, nodes: Nodes, theta: FloatArray, q: int) -> FloatArray:
    """``score_mixed()``: the derivative of :func:`negative_loglik`."""
    beta = theta[:q]
    D, slopes = unpack_d(theta[q:], ctx.structure)
    lyb, lpy = joint(ctx, nodes, beta, D)
    with numpy.errstate(all='ignore'):
        post = numpy.exp(lyb - lpy[:, None]) * nodes.wgh
    g_beta = expected_score(ctx, nodes, post, beta)
    inv = numpy.linalg.inv(D)
    second = numpy.einsum('ig,igc->ic', post, nodes.b2).reshape(ctx.n, ctx.r, ctx.r, order='F')
    g_d = numpy.empty(len(slopes))
    for k, dD in enumerate(slopes):
        m = inv @ dD @ inv
        g_d[k] = 0.5 * (ctx.n * float(numpy.trace(inv @ dD)) - numpy.nansum(numpy.einsum('rs,isr->i', m, second)))
    return numpy.concatenate([g_beta, g_d])


def expected_score(ctx: Context, nodes: Nodes, post: FloatArray, beta: FloatArray) -> FloatArray:
    """The EM score of the fixed effects with the posterior weights ``post`` (clusters by nodes) held fixed."""
    eta = (ctx.obs.X @ beta + ctx.obs.offset)[:, None] + nodes.ztb
    with numpy.errstate(all='ignore'):
        se = ctx.obs.score_eta(eta)
        per_cluster = ctx.cluster_sums(se[:, :, None] * ctx.obs.X[:, None, :])
        return numpy.asarray(-numpy.nansum(numpy.einsum('igq,ig->iq', per_cluster, post), axis=0), dtype=numpy.float64)


def check_coefficients(beta: FloatArray, control: AdaptiveControl) -> None:
    if bool(numpy.any(numpy.abs(beta[1:]) > control.max_coef_value)):
        raise metafopy.errors.EstimationError(LARGE_COEFFICIENT)


def expectation_maximisation(ctx: Context, beta: FloatArray, D: FloatArray, post_modes: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    """The EM iterations of ``mixed_fit()``; returns the coefficients, ``D``, the posterior modes and whether the tolerances were met."""
    con = ctx.control
    structure = ctx.structure
    nodes = make_nodes(ctx, post_modes, beta, D)
    post_modes = nodes.modes
    params = numpy.zeros((con.iter_em, beta.shape[0] + d_entries(D, structure).shape[0]))
    loglik = numpy.zeros(con.iter_em)
    for it in range(1, con.iter_em + 1):
        if it % con.update_gh_every == 0:
            nodes = make_nodes(ctx, post_modes, beta, D)
            post_modes = nodes.modes
        params[it - 1] = numpy.concatenate([beta, d_entries(D, structure)])
        lyb, lpy = joint(ctx, nodes, beta, D)
        with numpy.errstate(all='ignore'):
            post = numpy.exp(lyb - lpy[:, None])
        weighted = post * nodes.wgh
        post_b2 = numpy.einsum('ig,igc->ic', weighted, nodes.b2)
        total = lpy + nodes.log_dets
        loglik[it - 1] = float(numpy.sum(total[numpy.isfinite(total)]))
        # an increase of a few units in the last place is a tie, which rounding differences between platforms would otherwise turn into a stop
        if it > 4 and loglik[it - 1] - loglik[it - 2] > TIE * abs(loglik[it - 2]):
            before, after = params[it - 2], params[it - 1]
            check1 = float(numpy.max(numpy.abs(after - before) / (numpy.abs(before) + con.tol1))) < con.tol2
            check2 = (loglik[it - 1] - loglik[it - 2]) < con.tol3 * (abs(loglik[it - 2]) + con.tol3)
            if check1 or check2:
                return beta, D, post_modes, True
        Dn = numpy.nanmean(post_b2, axis=0).reshape(ctx.r, ctx.r, order='F')
        D = 0.5 * (Dn + Dn.T)
        if structure == 'independent':
            D = numpy.diag(numpy.diag(D))
        hessian = forward_hessian(functools.partial(expected_score, ctx, nodes, weighted), beta)
        step = numpy.linalg.solve(near_pd(hessian), expected_score(ctx, nodes, weighted, beta))
        beta = beta - step
        check_coefficients(beta, con)
    return beta, D, post_modes, False


def quasi_newton(ctx: Context, beta: FloatArray, D: FloatArray, post_modes: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, bool]:
    """The quasi-Newton iterations of ``mixed_fit()``, each started with freshly centred nodes."""
    con = ctx.control
    q = beta.shape[0]
    theta = numpy.concatenate([beta, pack_d(D, ctx.structure)])
    parscale = numpy.concatenate([numpy.full(q, con.parscale_betas), numpy.full(theta.shape[0] - q, con.parscale_d)])
    iterations = con.iter_qn
    for _ in range(con.iter_qn_outer):
        nodes = make_nodes(ctx, post_modes, theta[:q], unpack_d(theta[q:], ctx.structure)[0])
        with numpy.errstate(all='ignore'):
            opt = metafopy.core.bfgs.optim_bfgs(
                functools.partial(negative_loglik, ctx, nodes, q=q),
                theta,
                gr=functools.partial(gradient, ctx, nodes, q=q),
                parscale=parscale,
                maxit=iterations,
                reltol=con.tol3,
            )
        theta = opt.par
        post_modes = nodes.modes
        check_coefficients(theta[:q], con)
        if opt.convergence == 0:
            return theta[:q], unpack_d(theta[q:], ctx.structure)[0], post_modes, True
        iterations += con.iter_qn_incr
    return theta[:q], unpack_d(theta[q:], ctx.structure)[0], post_modes, False


def covariance_factor(D: FloatArray, structure: metafopy.core.glmm.Structure) -> FloatArray:
    """The ``theta`` of :class:`metafopy.core.glmm.GLMMFit` (``D = Lambda Lambda'``) for the covariance matrix ``D``."""
    if structure == 'independent':
        return numpy.sqrt(numpy.diag(D))
    lower = numpy.linalg.cholesky(D)
    return numpy.asarray(lower[numpy.tril_indices_from(lower)], dtype=numpy.float64)


def fit_adaptive(
    prob: metafopy.core.glmm.GLMMProblem, *, nagq: int, d0: FloatArray | None = None, control: AdaptiveControl | None = None
) -> metafopy.core.glmm.GLMMFit:
    """Fit ``prob`` as ``GLMMadaptive::mixed_model()`` does, with ``nagq`` quadrature points per random effect and the start value ``d0`` for ``D``.

    Raises:
        metafopy.errors.EstimationError: a coefficient exceeds ``control.max_coef_value`` in absolute value, a Hessian is not positive
            definite or the covariance matrix of the estimates cannot be computed.
    """
    con = control or AdaptiveControl()
    n = prob.n_clusters
    cluster = prob.cluster.astype(numpy.intp)
    obs = Observations(prob.X, prob.Z, prob.y, prob.size, prob.offset, prob.family, prob.link or metafopy.core.glm.canonical_link(prob.family))
    if not bool(numpy.all(numpy.diff(cluster) >= 0)) or set(cluster.tolist()) != set(range(n)):
        raise metafopy.errors.InternalError('the clusters must be numbered from 0 and sorted')
    rows = [numpy.flatnonzero(cluster == i) for i in range(n)]
    ctx = Context(obs, [obs.take(r) for r in rows], cluster, numpy.array([r[0] for r in rows], dtype=numpy.intp), prob.structure, nagq, con)
    start = metafopy.core.glm.glm_fit(prob.X, prob.y, family=prob.family, size=prob.size, offset=prob.offset, link=prob.link)
    beta = numpy.where(start.aliased, 0.0, start.coef) * INITIAL_SCALE
    q = beta.shape[0]
    D = numpy.eye(prob.Z.shape[1]) if d0 is None else d0.copy()
    post_modes = numpy.zeros((n, prob.Z.shape[1]))

    beta, D, post_modes, converged = expectation_maximisation(ctx, beta, D, post_modes)
    if not converged and con.iter_qn_outer > 0:
        beta, D, post_modes, converged = quasi_newton(ctx, beta, D, post_modes)

    theta = numpy.concatenate([beta, pack_d(D, prob.structure)])
    nodes = make_nodes(ctx, post_modes, beta, D)
    loglik = -negative_loglik(ctx, nodes, theta, q)
    hessian = central_hessian(lambda t: gradient(ctx, nodes, t, q), theta)
    if not bool(numpy.all(numpy.isfinite(hessian))):
        warnings.warn(HESSIAN_NONFINITE, stacklevel=3)
    else:
        ev = numpy.linalg.eigvalsh(hessian)[::-1]
        if not bool(numpy.all(ev >= -1e-6 * abs(ev[0]))):
            warnings.warn(HESSIAN_WARNING, stacklevel=3)
    try:
        covariance = numpy.linalg.inv(hessian)[:q, :q]
    except numpy.linalg.LinAlgError as exc:
        raise metafopy.errors.EstimationError('the Hessian of the GLMMadaptive fit is singular') from exc
    return metafopy.core.glmm.GLMMFit(
        beta=beta,
        theta=covariance_factor(D, prob.structure),
        loglik=loglik,
        cond_deviance=math.nan,
        vcov=numpy.asarray(covariance, dtype=numpy.float64),
        modes=nodes.modes,
        converged=converged,
        evaluations=0,
    )
