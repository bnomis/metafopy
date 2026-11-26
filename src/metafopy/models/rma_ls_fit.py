"""Estimation of location-scale models, after the ``rma.ls`` branches of metafor's ``rma.uni``.

The model is ``y_i = x_i' beta + u_i + e_i`` with ``u_i ~ N(0, tau_i^2)`` and ``e_i ~ N(0, v_i)``,
where ``log(tau_i^2) = z_i' alpha`` (log link) or ``tau_i^2 = z_i' alpha`` (identity link); see Viechtbauer and
Lopez-Lopez (2022), doi:10.1002/jrsm.1562, and
https://github.com/wviechtb/metafor/blob/master/R/rma.uni.r and
https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.rma.ls.r.

The (restricted) log-likelihood is maximised over ``alpha`` (and over ``beta`` when ``optbeta=True``); the
covariance matrix of the scale coefficients comes from the numerical Hessian of ``numDeriv::hessian``.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.stats

import metafopy.core.linalg
import metafopy.core.numderiv
import metafopy.errors
import metafopy.metadata
import metafopy.models.control_names
import metafopy.models.rma_ls_opt
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_opt

__all__ = ['LS_METHODS']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

# auditme: accept[literal-duplication] the option values metafor lists, repeated as in the R source
LS_METHODS: tuple[str, ...] = ('ML', 'REML')
DBL_EPS = float(numpy.finfo(float).eps)
LEVERAGE_ONE = 1e-10  # a study with 1 - h_ii below this has a zero residual in exact arithmetic
SCIPY_METHODS = ('BFGS', 'CG', 'L-BFGS-B', 'Nelder-Mead')
CONSTRAINED = ('constrOptim',)
# identity link: the optimisers metafor accepts for the constraint tau_i^2 >= 0; solnp and nloptr are also allowed with the log link
INEQUALITY = ('auglag', 'constrOptim', 'constrOptim.nl', 'nloptr', 'solnp')
NO_LOG_LINK = ('auglag', 'constrOptim', 'constrOptim.nl')
OPTIMIZERS = (*metafopy.models.sel_opt.OPTIMIZERS, 'constrOptim')
# auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
OPTIMIZER_SETTINGS = ('abstol', 'maxit', 'ndeps', 'reltol', 'tol')
# how far inside alpha_min and alpha_max, as a fraction of the box's width, a start clipped into the box is placed: on a bound the logistic map
# puts the start at infinity, where its gradient is zero and the optimiser cannot leave it
BOX_START_MARGIN = 1e-4
RESTART_STEP = 4.0  # a restart moves each slope towards and past zero by this many standard deviations of its column of Z, inverted
RESTART_GAIN = 1e-6  # a restart replaces the fit only when its log-likelihood is higher by more than this
RESTART_RATIO = 1e-5  # ... and its information on the standardised Z has min/max eigenvalue ratio above this


def make_ls_control(
    control: metafopy.models.rma_ls_result.LSControl | dict[str, object] | None,
) -> metafopy.models.rma_ls_result.LSControl:
    """Build an :class:`~metafopy.models.rma_ls_result.LSControl` from a mapping with metafor's (``.`` or ``_``) names.

    Raises:
        metafopy.errors.ArgumentError: an unknown control argument was given.
        metafopy.errors.InputError: ``restarts`` is not a ``bool``.
    """
    if control is None:
        return metafopy.models.rma_ls_result.LSControl()
    if isinstance(control, metafopy.models.rma_ls_result.LSControl):
        return control
    names = {f.name for f in dataclasses.fields(metafopy.models.rma_ls_result.LSControl)}
    kwargs: dict[str, object] = {}
    opt_ctrl: dict[str, object] = {}
    ignored: dict[str, object] = {}
    for key, value in control.items():
        name = key.replace('.', '_')
        if name in OPTIMIZER_SETTINGS:
            opt_ctrl[name] = value
            ignored[key] = value
        elif name not in names:
            if metafopy.models.control_names.read_elsewhere(key):
                ignored[key] = value
                continue
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        else:
            if name == 'restarts' and not isinstance(value, bool):
                raise metafopy.errors.InputError(f"control['restarts'] must be a bool, not {value!r}")
            kwargs[name] = value
    if ignored:
        kwargs['ignored'] = ignored
    if opt_ctrl:
        kwargs['opt_ctrl'] = {**dict(kwargs.get('opt_ctrl', {})), **opt_ctrl}  # type: ignore[call-overload]
    return metafopy.models.rma_ls_result.LSControl(**kwargs)  # type: ignore[arg-type]


def resolve_optimizer(con: metafopy.models.rma_ls_result.LSControl, link: str) -> str:
    """metafor's optimiser bookkeeping of ``rma.uni`` for location-scale models; returns the name of the optimiser to run.

    ``optim`` stands for ``optmethod`` (``BFGS`` unless the control says otherwise); ``ncpus > 1`` selects ``optimParallel`` and
    ``alabama`` is ``constrOptim.nl``. With the identity link the default ``nlminb`` becomes ``constrOptim`` and an optimiser that
    cannot honour the constraint is reset to it with a warning.

    Raises:
        metafopy.errors.InputError: the optimiser is unknown or cannot be used with the log link.
        metafopy.errors.NotSupportedError: ``optmethod`` names an ``optim`` method with no SciPy counterpart.
    """
    optimizer = metafopy.models.sel_opt.partial_match(con.optimizer, OPTIMIZERS, 'optimizer')
    optmethod = metafopy.models.sel_opt.partial_match(con.optmethod, metafopy.models.sel_opt.OPTMETHODS, 'optmethod')
    if optimizer in metafopy.models.sel_opt.OPTMETHODS:
        optmethod, optimizer = optimizer, 'optim'
    if con.ncpus > 1:
        optimizer = 'optimParallel'
    if optimizer == 'alabama':
        optimizer = 'constrOptim.nl'
    if link == 'identity':
        if optimizer == 'nlminb':
            optimizer = 'constrOptim'
        elif optimizer not in INEQUALITY:
            warnings.warn(
                "Can only use optimizers 'constrOptim', 'solnp', 'nloptr', 'constrOptim.nl', or 'auglag' when link='identity' (resetting to 'constrOptim').",
                stacklevel=4,
            )
            optimizer = 'constrOptim'
    elif optimizer in NO_LOG_LINK:
        raise metafopy.errors.InputError(f"Cannot use '{optimizer}' optimizer when using a log link.")
    elif optimizer == 'nlminb':
        optimizer = 'BFGS'
    if optimizer == 'optim':
        if optmethod not in SCIPY_METHODS:
            raise metafopy.errors.NotSupportedError(f"optim(method='{optmethod}') is not available for location-scale models.")
        optimizer = optmethod
    return optimizer


# auditme: accept[raises-section-drift] minimise_standin checks the start value is finite first, so the InputError of sel_opt.minimize never arises
def minimise_standin(
    objective: collections.abc.Callable[[FloatArray], float],
    start: FloatArray,
    choice: metafopy.models.sel_opt.OptimizerChoice,
    con: metafopy.models.rma_ls_result.LSControl,
    ui: FloatArray | None,
) -> FloatArray:
    """Run an optimiser of metafor that has no exact counterpart here as :func:`metafopy.models.sel_opt.minimize` maps it.

    ``ui`` (the matrix of ``constrOptim``) turns ``ui @ x >= 0`` into the inequality constraint of the constrained optimisers.

    Raises:
        metafopy.errors.EstimationError: the optimiser did not converge or could not start.
        metafopy.errors.NotSupportedError: the optimiser has no counterpart here.
    """
    if not math.isfinite(objective(start)):
        raise metafopy.errors.EstimationError('Cannot evaluate the log-likelihood at the initial values.')
    ineq = None if ui is None else (lambda x: numpy.asarray(ui @ x, dtype=numpy.float64))
    return metafopy.models.sel_opt.minimize(objective, start, choice, {'maxit': con.maxiter, **con.opt_ctrl}, ineq)


def mapfun(x: FloatArray, lb: FloatArray, ub: FloatArray) -> FloatArray:
    """Map the unconstrained scale coefficients to ``(lb, ub)``; unbounded coefficients stay as they are."""
    bounded = numpy.isfinite(lb) & numpy.isfinite(ub)
    out = x.copy()
    with numpy.errstate(over='ignore'):
        out[bounded] = lb[bounded] + (ub[bounded] - lb[bounded]) / (1 + numpy.exp(-x[bounded]))
    return out


def mapinvfun(x: FloatArray, lb: FloatArray, ub: FloatArray) -> FloatArray:
    bounded = numpy.isfinite(lb) & numpy.isfinite(ub)
    out = x.copy()
    with numpy.errstate(divide='ignore', invalid='ignore'):
        out[bounded] = numpy.log((x[bounded] - lb[bounded]) / (ub[bounded] - x[bounded]))
    return out


def clip_into_box(x: FloatArray, lb: FloatArray, ub: FloatArray) -> FloatArray:
    """Clip the starting scale coefficients into ``[lb, ub]``, keeping bounded ones ``BOX_START_MARGIN`` of the width inside each bound.

    metafor clips onto the bound (``pmin(pmax(alpha.init, alpha.min), alpha.max)`` in ``rma.uni``), where its logistic map sends the start to
    infinity, so a box that does not bind can end at a non-maximum; see ``docs/differences.md``. A start further inside than the margin is unchanged.
    """
    bounded = numpy.isfinite(lb) & numpy.isfinite(ub)
    margin = numpy.where(bounded, BOX_START_MARGIN * (ub - lb), 0.0)
    return numpy.minimum(numpy.maximum(x, lb + margin), ub - margin)


@dataclasses.dataclass(frozen=True, slots=True)
class LSProblem:
    """Data of the (restricted) log-likelihood of a location-scale model (metafor's ``.ll.rma.ls``)."""

    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    Z: FloatArray
    reml: bool
    REMLf: bool
    link: str
    alpha_min: FloatArray
    alpha_max: FloatArray
    tau2_min: float
    tau2_max: float
    optbeta: bool

    def loglik(self, par: FloatArray, alpha_arg: FloatArray, beta_arg: FloatArray, *, alpha_transf: bool) -> float:
        """The log-likelihood at ``par`` (``-inf`` where it cannot be evaluated); fixed values override ``par``."""
        yi, vi, X = self.yi, self.vi, self.X
        k, p = X.shape
        if self.optbeta:
            beta: FloatArray | None = numpy.where(numpy.isnan(beta_arg), par[:p], beta_arg)
            alpha = par[p:]
        else:
            beta = None
            alpha = par
        if alpha_transf:
            alpha = mapfun(alpha, self.alpha_min, self.alpha_max)
        alpha = numpy.where(numpy.isnan(alpha_arg), alpha, alpha_arg)
        with numpy.errstate(all='ignore'):
            lin = self.Z @ alpha
            tau2 = numpy.exp(lin) if self.link == 'log' else lin
            if numpy.any(numpy.isnan(tau2)) or numpy.any(tau2 < self.tau2_min) or numpy.any(tau2 > self.tau2_max) or numpy.any(tau2 < 0):
                return -math.inf
            wi = 1.0 / (vi + tau2)
            if beta is None:
                try:
                    beta = metafopy.core.linalg.invcalc(X, wi) @ (X.T @ (wi * yi))
                except numpy.linalg.LinAlgError:
                    return -math.inf
            rss = float(numpy.sum(wi * (yi - X @ beta) ** 2))
            sum_log = float(numpy.sum(numpy.log(vi + tau2)))
            if not self.reml:
                val = -0.5 * k * math.log(2 * math.pi) - 0.5 * sum_log - 0.5 * rss
            else:
                val = (
                    -0.5 * (k - p) * math.log(2 * math.pi)
                    + (0.5 * metafopy.core.linalg.logdet(X.T @ X) if self.REMLf else 0.0)
                    - 0.5 * sum_log
                    - 0.5 * metafopy.core.linalg.logdet((X.T * wi) @ X)
                    - 0.5 * rss
                )
        return val if math.isfinite(val) else -math.inf

    def negloglik(self, par: FloatArray, alpha_arg: FloatArray, beta_arg: FloatArray, *, alpha_transf: bool) -> float:
        return -self.loglik(par, alpha_arg, beta_arg, alpha_transf=alpha_transf)

    def score_parts(self, alpha: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``d``, ``M`` and ``e = P y`` of the identity-link score at ``alpha`` (``tau2 = Z alpha``, ``W = diag(1 / (v + tau2))``).

        ``P = W - W X (X' W X)^-1 X' W``. ``d`` and ``M`` are ``diag(W)`` and ``W`` under ML and ``diag(P)`` and ``P`` under REML.
        """
        return self.tau2_parts(self.Z @ alpha)

    def tau2_parts(self, tau2: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """:meth:`score_parts` at the heterogeneities ``tau2``."""
        P = self.projection(tau2)
        e = P @ self.yi
        if self.reml:
            return numpy.diag(P).copy(), P, e
        w = 1.0 / (self.vi + tau2)
        return w, numpy.diag(w), e

    def log_score(self, alpha: FloatArray) -> FloatArray:
        """Score ``-1/2 Z' ((d - e^2) tau2)`` of the scale coefficients under the log link, ``tau2 = exp(Z alpha)``: the gradient of :meth:`loglik`."""
        tau2 = numpy.exp(self.Z @ alpha)
        d, _, e = self.tau2_parts(tau2)
        return numpy.asarray(-0.5 * self.Z.T @ ((d - e**2) * tau2), dtype=numpy.float64)

    def log_information(self, alpha: FloatArray) -> FloatArray:
        """Observed information (minus the Hessian of :meth:`loglik`) of the scale coefficients under the log link.

        With ``D_j = diag(tau2_i z_ij)`` and ``T`` the ``M`` of :meth:`score_parts`, the entry ``(j, l)`` is
        ``-1/2 tr(T D_l T D_j) + 1/2 tr(T D_jl) + y'P D_l P D_j P y - 1/2 y'P D_jl P y`` with ``D_jl = diag(tau2_i z_ij z_il)``, from differentiating
        :meth:`log_score` once more with ``dW = -W dM W`` and ``dP = -P dM P`` (Harville, 1977, <https://doi.org/10.1080/01621459.1977.10480998>).
        """
        tau2 = numpy.exp(self.Z @ alpha)
        d, T, e = self.tau2_parts(tau2)
        P = self.projection(tau2)
        D = self.Z * tau2[:, None]
        De = D * e[:, None]
        cross = D.T @ (T * T) @ D
        second = (self.Z * (d * tau2)[:, None]).T @ self.Z
        quad = De.T @ P @ De
        data = (self.Z * (e**2 * tau2)[:, None]).T @ self.Z
        return numpy.asarray(-0.5 * cross + 0.5 * second + quad - 0.5 * data, dtype=numpy.float64)

    def projection(self, tau2: FloatArray) -> FloatArray:
        """``P = W - W X (X' W X)^-1 X' W`` at the heterogeneities ``tau2``."""
        w = 1.0 / (self.vi + tau2)
        XtW = self.X.T * w
        return numpy.asarray(numpy.diag(w) - XtW.T @ metafopy.core.linalg.invcalc(self.X, w) @ XtW, dtype=numpy.float64)

    def ls_score(self, alpha: FloatArray) -> FloatArray:
        """Score ``-1/2 Z' (d - e^2)`` of the scale coefficients with the identity link, the gradient of :meth:`loglik`.

        Viechtbauer and Lopez-Lopez (2022, equations 7 to 9, <https://doi.org/10.1002/jrsm.1564>) give the (restricted) likelihood; the
        derivative follows Harville (1977, <https://doi.org/10.1080/01621459.1977.10480998>).
        """
        d, _, e = self.score_parts(alpha)
        return numpy.asarray(-0.5 * self.Z.T @ (d - e**2), dtype=numpy.float64)

    def ls_information(self, alpha: FloatArray) -> FloatArray:
        """Expected information ``1/2 Z' (M o M) Z`` of the scale coefficients with the identity link (``o`` the elementwise product)."""
        _, M, _ = self.score_parts(alpha)
        return numpy.asarray(0.5 * self.Z.T @ (M * M) @ self.Z, dtype=numpy.float64)


def polish_log_fit(problem: LSProblem, theta: FloatArray, alpha_arg: FloatArray, beta_arg: FloatArray) -> FloatArray:
    """Finish a log-link fit on the analytic score (:func:`metafopy.models.rma_ls_opt.polish_log`), in the unconstrained ``theta`` of the optimiser.

    Where ``alpha_min`` and ``alpha_max`` bound a coefficient, ``alpha = mapfun(theta)`` and the score and information of ``theta`` follow by the
    chain rule: ``J g`` and ``J I J - diag(g alpha'')`` with ``J = diag(d alpha / d theta)``. A coefficient fixed by ``alpha_arg`` has ``J = 0``.
    """
    lb, ub = problem.alpha_min, problem.alpha_max
    bounded = numpy.isfinite(lb) & numpy.isfinite(ub)
    fixed = ~numpy.isnan(alpha_arg)

    def to_alpha(x: FloatArray) -> FloatArray:
        return numpy.where(fixed, alpha_arg, mapfun(x, lb, ub))

    def derivatives(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        with numpy.errstate(over='ignore'):
            sig = numpy.where(bounded, 1.0 / (1.0 + numpy.exp(-x)), 0.0)
        width = numpy.where(bounded, ub - lb, 0.0)
        d1 = numpy.where(bounded, width * sig * (1 - sig), 1.0)
        d2 = numpy.where(bounded, width * sig * (1 - sig) * (1 - 2 * sig), 0.0)
        return numpy.where(fixed, 0.0, d1), numpy.where(fixed, 0.0, d2)

    def score(x: FloatArray) -> FloatArray:
        d1, _ = derivatives(x)
        return numpy.asarray(d1 * problem.log_score(to_alpha(x)), dtype=numpy.float64)

    def information(x: FloatArray) -> FloatArray:
        d1, d2 = derivatives(x)
        return numpy.asarray(
            d1[:, None] * problem.log_information(to_alpha(x)) * d1[None, :] - numpy.diag(d2 * problem.log_score(to_alpha(x))), dtype=numpy.float64
        )

    return metafopy.models.rma_ls_opt.polish_log(
        theta,
        lambda x: problem.loglik(x, alpha_arg, beta_arg, alpha_transf=True),
        score,
        information,
        numpy.isnan(alpha_arg),
    )


def restart_fit(
    problem: LSProblem,
    par_opt: FloatArray,
    optimizer: str,
    con: metafopy.models.rma_ls_result.LSControl,
    alpha_arg: FloatArray,
    beta_arg: FloatArray,
) -> FloatArray:
    """Refit a log-link model with every scale coefficient free from each of :func:`restart_starts`, through the optimiser and polish of the first fit.

    The end point with the highest log-likelihood replaces ``par_opt`` when it beats it by more than ``RESTART_GAIN`` and
    :func:`is_regular_maximum` holds there. A start whose fit fails or is not finite is skipped.
    """

    def objective(x: FloatArray) -> float:
        return problem.negloglik(x, alpha_arg, beta_arg, alpha_transf=True)

    best = par_opt
    best_ll = problem.loglik(par_opt, alpha_arg, beta_arg, alpha_transf=True)
    for start in restart_starts(par_opt, problem.Z):
        try:
            if optimizer in SCIPY_METHODS:
                end = metafopy.models.rma_ls_opt.minimise(
                    objective, start, method=optimizer, maxiter=con.maxiter, polish_converged=con.polish_converged
                ).par.copy()
            else:
                end = minimise_standin(objective, start, metafopy.models.sel_opt.OptimizerChoice(optimizer, 'BFGS'), con, None)
            if optimizer == 'BFGS' and metafopy.models.sel_opt.partial_match(con.optimizer, OPTIMIZERS, 'optimizer') == 'nlminb':
                end = polish_log_fit(problem, end, alpha_arg, beta_arg)
        except metafopy.errors.EstimationError, numpy.linalg.LinAlgError:
            continue
        ll = problem.loglik(end, alpha_arg, beta_arg, alpha_transf=True)
        if math.isfinite(ll) and ll > best_ll + RESTART_GAIN and is_regular_maximum(problem, end):
            best, best_ll = end, ll
    return best


def restart_starts(alpha: FloatArray, Z: FloatArray) -> list[FloatArray]:
    """One start per non-constant column ``j`` of ``Z``: ``alpha`` with ``alpha[j] - sign(alpha[j]) * RESTART_STEP / sd(Z[:, j])`` (``sign(0)`` is 1).

    A log-link likelihood can have a second interior maximum with the slope of the opposite sign (Viechtbauer and Lopez-Lopez, 2022, section 2.5,
    <https://doi.org/10.1002/jrsm.1562>); the start moves the slope across zero to find it.
    """
    spread = numpy.std(Z, axis=0, ddof=1) if Z.shape[0] > 1 else numpy.zeros(Z.shape[1])
    starts: list[FloatArray] = []
    for j in range(Z.shape[1]):
        if spread[j] > 0.0:
            start = alpha.copy()
            start[j] = alpha[j] - (1.0 if alpha[j] >= 0 else -1.0) * RESTART_STEP / spread[j]
            starts.append(start)
    return starts


def is_regular_maximum(problem: LSProblem, alpha: FloatArray) -> bool:
    """The observed information of the log link (:meth:`LSProblem.log_information`) at ``alpha``, taken on the standardised ``Z`` (each non-constant
    column scaled, and centred where a constant column absorbs the mean), is finite and has min/max eigenvalue ratio above ``RESTART_RATIO``.

    An end point on the ridge where ``tau2`` tends to 0 in one group has a singular information, and an interior maximum a well-conditioned one.
    """
    Z = problem.Z
    q = Z.shape[1]
    spread = numpy.std(Z, axis=0, ddof=1) if Z.shape[0] > 1 else numpy.zeros(q)
    constant = [j for j in range(q) if spread[j] == 0.0]
    T = numpy.eye(q)
    for j in range(q):
        if spread[j] > 0.0:
            T[j, j] = spread[j]
            if constant:
                T[constant[0], j] = float(numpy.mean(Z[:, j]))
    scaled = dataclasses.replace(problem, Z=Z @ numpy.linalg.inv(T))
    with numpy.errstate(all='ignore'):
        info = scaled.log_information(T @ alpha)
    if not numpy.all(numpy.isfinite(info)):
        return False
    eigenvalues = numpy.linalg.eigvalsh(info)
    return bool(eigenvalues[0] > 0 and eigenvalues[0] / eigenvalues[-1] > RESTART_RATIO)


def identity_restart_starts(alpha: FloatArray, Z: FloatArray) -> list[FloatArray]:
    """Starts for the identity link from the end point ``alpha``, each with ``Z alpha >= 0``.

    One per distinct non-zero row ``r`` of ``Z``: the nearest ``alpha`` (in the metric ``Z'Z``) with ``r' alpha = 0``, kept only when it is feasible
    on every row. One per non-constant column ``j`` when ``Z`` has a constant column: the slope reflected about the column's midrange
    (``alpha[j] -> -alpha[j]``, the intercept gains ``2 alpha[j] c_j``), which swaps the ``tau2`` at the column's two ends; kept when feasible.

    The identity-link likelihood can have its maximum on a face ``tau2_i = 0``, which the single start of metafor's ``constrOptim`` can miss
    (Viechtbauer and Lopez-Lopez, 2022, after equation 8, <https://doi.org/10.1002/jrsm.1562>; the fit is ``rma.uni`` in
    <https://github.com/wviechtb/metafor/blob/master/R/rma.uni.r>). Entries of a face start below ``1e-12 * max(1, max|a|)`` are set to exactly 0,
    since a negative residue evaluates to a log-likelihood of ``-inf``. A deficit of ``Z a`` within ``ACTIVE_TOL * max(1, max|Z a|)`` is added to
    the intercept.
    """
    q = Z.shape[1]
    spread = numpy.std(Z, axis=0, ddof=1) if Z.shape[0] > 1 else numpy.zeros(q)
    constant = [j for j in range(q) if spread[j] == 0.0 and Z[0, j] != 0.0]

    def feasible(a: FloatArray, *, clean: bool) -> FloatArray | None:
        a = a.copy()
        if clean:
            a[numpy.abs(a) < 1e-12 * max(1.0, float(numpy.max(numpy.abs(a))))] = 0.0
        za = Z @ a
        deficit = float(numpy.min(za))
        if deficit < 0 and constant and -deficit <= metafopy.models.rma_ls_opt.ACTIVE_TOL * max(1.0, float(numpy.max(numpy.abs(za)))):
            a[constant[0]] += -deficit / Z[0, constant[0]]
        return a if float(numpy.min(Z @ a)) >= 0.0 else None

    starts: list[FloatArray] = []
    G = numpy.linalg.pinv(Z.T @ Z)
    for r in numpy.unique(Z[numpy.any(Z != 0.0, axis=1)], axis=0):
        gr = G @ r
        denom = float(r @ gr)
        if denom > 0.0:
            start = feasible(alpha - gr * float(r @ alpha) / denom, clean=True)
            if start is not None:
                starts.append(start)
    if constant:
        c0 = constant[0]
        for j in range(q):
            if spread[j] > 0.0:
                mid = (float(numpy.min(Z[:, j])) + float(numpy.max(Z[:, j]))) / 2
                reflected = alpha.copy()
                reflected[j] = -alpha[j]
                reflected[c0] = alpha[c0] + 2 * alpha[j] * mid / Z[0, c0]
                start = feasible(reflected, clean=False)
                if start is not None:
                    starts.append(start)
    return starts


def is_face_maximum(problem: LSProblem, alpha: FloatArray) -> bool:
    """The analytic information ``problem.ls_information`` at ``alpha``, projected on the null space of the active rows (``z_i' alpha`` within
    ``rma_ls_opt.ACTIVE_TOL`` of zero, relative to the largest), is finite and has min/max eigenvalue ratio above ``RESTART_RATIO``. A face of
    dimension 0 (the active rows pin every coefficient) passes.

    The identity-link counterpart of :func:`is_regular_maximum`, on the face where the maximum lies (Viechtbauer and Lopez-Lopez, 2022,
    <https://doi.org/10.1002/jrsm.1562>).
    """
    Z = problem.Z
    za = Z @ alpha
    rows = za <= metafopy.models.rma_ls_opt.ACTIVE_TOL * max(1.0, float(numpy.max(numpy.abs(za))))
    if rows.any():
        _, sv, vt = numpy.linalg.svd(Z[rows])
        rank = int(numpy.sum(sv > 1e-10 * sv[0]))
        N = vt[rank:].T
    else:
        N = numpy.eye(Z.shape[1])
    if N.shape[1] == 0:
        return True
    with numpy.errstate(all='ignore'):
        info = N.T @ problem.ls_information(alpha) @ N
    if not numpy.all(numpy.isfinite(info)):
        return False
    eigenvalues = numpy.linalg.eigvalsh(info)
    return bool(eigenvalues[0] > 0 and eigenvalues[0] / eigenvalues[-1] > RESTART_RATIO)


def identity_restart_fit(problem: LSProblem, par_opt: FloatArray, alpha_arg: FloatArray) -> FloatArray:
    """Polish from each of :func:`identity_restart_starts` with ``rma_ls_opt.polish_identity`` and return the end point of highest log-likelihood
    when it beats ``par_opt`` by more than ``RESTART_GAIN`` and :func:`is_face_maximum` holds there; otherwise ``par_opt``.

    Every coefficient is free and no optimiser runs: the barrier stand-in needs an interior start and drifts off a face maximum (Viechtbauer and
    Lopez-Lopez, 2022, <https://doi.org/10.1002/jrsm.1562>). A start whose polish fails or ends non-finite is skipped.
    """
    free = numpy.isnan(alpha_arg)

    def loglik(a: FloatArray) -> float:
        return problem.loglik(a, alpha_arg, numpy.full(problem.X.shape[1], numpy.nan), alpha_transf=False)

    best = par_opt
    best_ll = loglik(par_opt)
    for start in identity_restart_starts(par_opt, problem.Z):
        try:
            end = metafopy.models.rma_ls_opt.polish_identity(start, loglik, problem.ls_score, problem.ls_information, problem.Z, free)
        except metafopy.errors.EstimationError, numpy.linalg.LinAlgError:
            continue
        ll = loglik(end)
        if math.isfinite(ll) and ll > best_ll + RESTART_GAIN and is_face_maximum(problem, end):
            best, best_ll = end, ll
    return best


def pos_chol_inverse(A: FloatArray) -> FloatArray | None:
    """``chol2inv(chol(A))``, or ``None`` when ``A`` is not positive definite or not finite."""
    if not numpy.all(numpy.isfinite(A)):
        return None
    try:
        L = numpy.linalg.cholesky(numpy.triu(A) + numpy.triu(A, 1).T)
    except numpy.linalg.LinAlgError:
        return None
    Linv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
    return Linv.T @ Linv


def quad_form(coef: FloatArray, V: FloatArray, idx: tuple[int, ...]) -> float:
    """``coef[idx]' V[idx, idx]^-1 coef[idx]``, NaN when the submatrix cannot be inverted."""
    sub = numpy.array(idx)
    inv = pos_chol_inverse(V[numpy.ix_(sub, sub)])
    if inv is None:
        return math.nan
    b = coef[sub]
    return float(b @ inv @ b)


def marginal_residuals(yi: FloatArray, vi: FloatArray, X: FloatArray, tau2: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Raw residuals and their standard errors for the weighted fit with heterogeneity ``tau2`` (metafor's ``rstandard``).

    A residual under ``100 * DBL_EPS``, or of a study with ``1 - h_ii < LEVERAGE_ONE``, is 0: the latter is 0 in exact arithmetic and only rounding
    noise in floating point, which differs between platforms.
    """
    w = 1.0 / (vi + tau2)
    inv = metafopy.core.linalg.invcalc(X, w)
    beta = inv @ (X.T @ (w * yi))
    H = X @ inv @ (X.T * w)
    ei = yi - X @ beta
    ei[numpy.abs(ei) < 100 * DBL_EPS] = 0.0
    ei[1.0 - numpy.diag(H) < LEVERAGE_ONE] = 0.0
    ImH = numpy.eye(yi.shape[0]) - H
    ve = ImH @ numpy.diag(vi + tau2) @ ImH.T
    return ei, numpy.sqrt(numpy.diag(ve))


# auditme: accept[raises-section-drift] fit_rma_uni is called here without control, so its ArgumentError for an unknown control argument never arises
def start_alpha(md: metafopy.metadata.MetaData, Z: FloatArray, link: str, is_int: BoolArray) -> FloatArray:
    """Starting values for the scale coefficients: a regression of the transformed squared residuals of a HE fit on ``Z``.

    Raises:
        metafopy.errors.EstimationError: no suitable starting values were found.
        metafopy.errors.InputError: the heterogeneity (HE) fit of the location model that gives the starting values rejects the data.
        metafopy.errors.InternalError: a weight matrix an estimator needs is missing.
    """
    md0 = dataclasses.replace(md, weights=None, intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.models.rma_uni.fit_rma_uni(md0, method='HE', weighted=True, test='z', skip_r2=True)
    ei, sei = marginal_residuals(md.yi, md.vi, md.X, numpy.full(md.k, fit.tau2))
    with numpy.errstate(all='ignore'):
        if link == 'log':
            y = numpy.log(ei**2)
            v = 4 / ei**2 * sei**2
        else:
            y = ei**2
            v = sei**2
    w = 1.0 / v
    alpha = metafopy.core.linalg.invcalc(Z, w) @ (Z.T @ (w * y))
    if link == 'identity':
        if numpy.any(Z @ alpha < 0):
            alpha = numpy.where(is_int, fit.tau2 + 0.01, 0.0)
        if numpy.any(Z @ alpha < 0):
            raise metafopy.errors.EstimationError('Unable to find suitable starting values for the scale parameters.')
    return numpy.asarray(alpha, dtype=numpy.float64)


def location_estimates(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    tau2: FloatArray,
    user_w: FloatArray | None,
    weighted: bool,
    test: str,
    opt: tuple[FloatArray, FloatArray] | None,
) -> tuple[FloatArray, FloatArray, float, float]:
    """``beta``, its (Knapp-Hartung scaled) covariance, ``RSS_f`` and ``s2w``; ``opt`` holds ``beta`` and ``vb`` from the joint optimisation.

    Raises:
        metafopy.errors.InputError: an inverse-variance weight is infinite.
    """
    if opt is None:
        return metafopy.models.rma_uni.fixed_effects(yi, vi, X, tau2, user_w, weighted, test)  # type: ignore[arg-type]
    beta, vb = opt
    k, p = X.shape
    wi = 1.0 / (vi + tau2)
    if numpy.any(numpy.isinf(wi)):
        raise metafopy.errors.InputError('Division by zero when computing the inverse variance weights.')
    rss_f = float(numpy.sum(wi * (yi - X @ beta) ** 2))
    rss_knha = rss_f
    if user_w is not None:
        stXAX = metafopy.core.linalg.invcalc(X, user_w)
        beta = stXAX @ (X.T @ (user_w * yi))
        XA = X.T * user_w
        vb = stXAX @ XA @ numpy.diag(vi + tau2) @ XA.T @ stXAX
        rss_f = float(numpy.sum(wi * (yi - X @ beta) ** 2))
    s2w = 1.0
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    if test in ('knha', 'adhoc'):
        s2w = 0.0 if rss_knha <= DBL_EPS else rss_knha / (k - p)
    if test == 'adhoc':
        s2w = max(s2w, 1.0)
    return beta, s2w * vb, rss_f, s2w


# auditme: accept[raises-section-drift] fit_rma_uni is called here without control, so its ArgumentError for an unknown control argument never arises
def fit_rma_ls(
    md: metafopy.metadata.MetaData,
    Z: FloatArray,
    z_names: tuple[str, ...],
    *,
    method: str,
    weighted: bool,
    test: metafopy.models.rma_uni.Test,
    level: float,
    btt: metafopy.models.rma_uni.Btt,
    att: metafopy.models.rma_uni.Btt,
    link: str,
    optbeta: bool,
    alpha: numpy.typing.ArrayLike | None,
    beta: numpy.typing.ArrayLike | None,
    control: metafopy.models.rma_ls_result.LSControl,
    skiphes: bool = False,
    formula_scale: str | None = None,
    Z_f: FloatArray | None = None,
    dfs: float | collections.abc.Sequence[float] | None = None,
    abbrev: bool = False,
) -> metafopy.models.rma_ls_result.RMALSResult:
    """Fit a location-scale model to normalised inputs: ``md`` (``yi``, ``vi``, ``X``) and the scale model matrix ``Z``.

    With ``link='identity'`` the optimum of every optimiser is polished on the analytic score within ``Z alpha >= 0``
    (:func:`metafopy.models.rma_ls_opt.polish_identity`): R's ``constrOptim`` stops about 1e-5 short of the maximum, and metafor does not polish
    (see "Location-scale fits with link=\"identity\"" in ``docs/differences.md``). Fits with ``optbeta=True`` are not polished.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: the optimisation failed.
        metafopy.errors.NotSupportedError: the optimiser chosen in ``control`` has no counterpart here.
        metafopy.errors.InternalError: a weight matrix an estimator needs is missing.
    """
    if method not in LS_METHODS:
        raise metafopy.errors.InputError('Location-scale models can only be fitted with ML or REML estimation.')
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    if link not in ('log', 'identity'):
        raise metafopy.errors.InputError("The 'link' argument must be either 'log' or 'identity'.")
    if optbeta and md.weights is not None:
        raise metafopy.errors.InputError("Cannot use custom weights when 'optbeta=True'.")
    if optbeta and not weighted:
        raise metafopy.errors.InputError("Must use 'weighted=TRUE' when 'optbeta=TRUE'.")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test not in ('z', 't', 'knha', 'adhoc'):
        raise metafopy.errors.InputError(f'unknown test {test!r}')
    con = control

    yi = md.yi.copy()
    vi = md.vi.copy()
    X = md.X.copy()
    names = list(md.X_names)
    k = md.k
    user_w = md.weights
    if k < 1:
        raise metafopy.errors.InputError('processing terminated since k = 0')
    if k != Z.shape[0]:
        raise metafopy.errors.InputError('the scale model matrix must have one row per study')

    if numpy.any(vi <= 0):
        allvipos = False
        if numpy.any(vi < 0):
            warnings.warn('Negative sampling variances constrained to 0.', stacklevel=3)
            vi[vi < 0] = 0.0
    else:
        allvipos = True

    if k == 1 and test != 'z':
        warnings.warn('Setting argument test="z" since k=1.', stacklevel=3)
        test = 'z'

    drop = metafopy.core.linalg.drop_redundant(X)
    X_f = md.X_all[md.rows_f] if md.X_all is not None and md.X_all.shape[1] == drop.shape[0] else X
    if drop.any():
        warnings.warn('Redundant predictors dropped from the model.', stacklevel=3)
        X = X[:, ~drop]
        X_f = X_f[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]
    is_int_x = numpy.array([metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])])
    int_incl = bool(is_int_x.any())
    if int_incl:
        X = numpy.column_stack([numpy.ones(k), X[:, ~is_int_x]])
        X_f = numpy.column_stack([numpy.ones(X_f.shape[0]), X_f[:, ~is_int_x]])
        names = ['intrcpt', *[n for n, i in zip(names, is_int_x, strict=True) if not i]]
    p = X.shape[1]
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])
    if not (int_only and k == 1) and p + 1 > k:
        raise metafopy.errors.InputError('Number of parameters to be estimated is larger than the number of observations.')
    btt_idx = metafopy.models.rma_uni.set_btt(btt, p, int_incl, names)
    m = len(btt_idx)
    if abbrev:
        names = metafopy.models.rma_uni.abbreviate_names(names)

    if not metafopy.core.linalg.is_positive_definite(X.T @ X, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')

    # -- scale model matrix ---------------------------------------------------
    znames = list(z_names)
    drop_z = metafopy.core.linalg.drop_redundant(Z)
    Z_full = Z_f if Z_f is not None and Z_f.shape[1] == drop_z.shape[0] else None
    if drop_z.any():
        warnings.warn('Redundant predictors dropped from the scale model.', stacklevel=3)
        Z = Z[:, ~drop_z]
        Z_full = None if Z_full is None else Z_full[:, ~drop_z]
        znames = [n for n, d in zip(znames, drop_z, strict=True) if not d]
    is_int_z = numpy.array([metafopy.core.linalg.is_intercept(Z[:, j]) for j in range(Z.shape[1])])
    Z_int_incl = bool(is_int_z.any())
    if Z_int_incl:
        Z = numpy.column_stack([numpy.ones(k), Z[:, ~is_int_z]])
        Z_full = None if Z_full is None else numpy.column_stack([numpy.ones(Z_full.shape[0]), Z_full[:, ~is_int_z]])
        znames = ['intrcpt', *[n for n, i in zip(znames, is_int_z, strict=True) if not i]]
    q = Z.shape[1]
    if not metafopy.core.linalg.is_positive_definite(Z.T @ Z, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix for scale part of the model not of full rank. Cannot fit model.')
    Z_int_only = q == 1 and metafopy.core.linalg.is_intercept(Z[:, 0])
    is_int = numpy.array([metafopy.core.linalg.is_intercept(Z[:, j]) for j in range(q)])

    alpha_arg = fixed_values(alpha, q, 'alpha')
    beta_arg = fixed_values(beta, p, 'beta') if optbeta else numpy.full(p, math.nan)

    alpha_min = expand_bound(con.alpha_min, q, 'alpha_min')
    alpha_max = expand_bound(con.alpha_max, q, 'alpha_max')
    if numpy.any(numpy.isinf(alpha_min) != numpy.isinf(alpha_max)):
        raise metafopy.errors.InputError('Constraints on the scale coefficients must be placed on both the lower and upper bound.')
    if link == 'identity' and (numpy.any(alpha_min != -numpy.inf) or numpy.any(alpha_max != numpy.inf)):
        raise metafopy.errors.InputError('Cannot use constraints on scale coefficients when using an identity link.')

    optimizer = resolve_optimizer(con, link)

    # -- rescale Z (as metafor does for models with scale moderators) -----------
    mZ: FloatArray | None = None
    imZ: FloatArray | None = None
    Zsave = Z
    if (
        not Z_int_only
        and Z_int_incl
        and con.scaleZ
        and math.isnan(alpha_arg[0])
        and numpy.all(numpy.isinf(alpha_min))
        and numpy.all(numpy.isinf(alpha_max))
        and not optbeta
    ):
        meanZ = Z[:, 1:].mean(axis=0)
        sdZ = Z[:, 1:].std(axis=0, ddof=1)
        is_d = numpy.array([metafopy.core.linalg.is_dummy(Z[:, j]) for j in range(q)])
        with numpy.errstate(all='ignore'):
            top = numpy.concatenate([[1.0], -1 * numpy.where(is_d[1:], 0.0, meanZ / sdZ)])
            body = numpy.column_stack([numpy.zeros(q - 1), numpy.diag(numpy.where(is_d[1:], 1.0, 1 / sdZ))])
        mZ = numpy.vstack([top, body])
        try:
            imZ = numpy.linalg.inv(mZ)
        except numpy.linalg.LinAlgError:
            imZ = None
        Z = Z.copy()
        for j in range(q):
            if not is_d[j]:
                Z[:, j] = (Z[:, j] - Z[:, j].mean()) / Z[:, j].std(ddof=1)
        if numpy.any(~numpy.isnan(alpha_arg)):
            if imZ is None:
                raise metafopy.errors.InputError('Unable to rescale starting values for the scale parameters.')
            alpha_arg = numpy.diag(imZ) * alpha_arg

    # -- starting values ------------------------------------------------------
    md_fit = dataclasses.replace(md, yi=yi, vi=vi, X=X, X_names=tuple(names), intercept=int_incl)
    if k == 1 and Z_int_only:
        alpha_init = numpy.array([-10000.0 if link == 'log' else 0.00001])
    elif con.alpha_init is None:
        alpha_init = start_alpha(md_fit, Z, link, is_int)
    else:
        alpha_init = numpy.atleast_1d(numpy.asarray(con.alpha_init, dtype=numpy.float64))
        if mZ is not None:
            if imZ is None:
                raise metafopy.errors.InputError('Unable to rescale the starting values for the scale parameters.')
            alpha_init = imZ @ alpha_init
        if link == 'identity' and numpy.any(Z @ alpha_init < 0):
            raise metafopy.errors.InputError('Starting values for the scale parameters lead to one or more negative tau^2 values.')
    if alpha_init.shape[0] != q:
        raise metafopy.errors.InputError(f"Length of the 'alpha_init' argument ({alpha_init.shape[0]}) does not match the actual number of parameters ({q}).")
    if numpy.any(numpy.isnan(alpha_init)):
        raise metafopy.errors.InputError("No missing values allowed in 'alpha_init'.")

    beta_init: FloatArray | None = None
    if optbeta:
        if con.beta_init is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                beta_init = metafopy.models.rma_uni.fit_rma_uni(
                    dataclasses.replace(md_fit, weights=None, intercept=False),
                    method='HE',
                    weighted=True,
                    test='z',
                    skip_r2=True,
                ).beta
        else:
            beta_init = numpy.atleast_1d(numpy.asarray(con.beta_init, dtype=numpy.float64))
            if beta_init.shape[0] != p:
                raise metafopy.errors.InputError(
                    f"Length of the 'beta_init' argument ({beta_init.shape[0]}) does not match the actual number of parameters ({p})."
                )
            if numpy.any(numpy.isnan(beta_init)):
                raise metafopy.errors.InputError("No missing values allowed in 'beta_init'.")

    alpha_init = mapinvfun(clip_into_box(alpha_init, alpha_min, alpha_max), alpha_min, alpha_max)

    tau2_min = con.tau2_min
    problem = LSProblem(
        yi=yi,
        vi=vi,
        X=X,
        Z=Z,
        reml=method == 'REML',
        REMLf=con.REMLf,
        link=link,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        tau2_min=max(tau2_min, 0.0),
        tau2_max=con.tau2_max,
        optbeta=optbeta,
    )

    # -- maximise the likelihood ------------------------------------------------
    par0 = numpy.concatenate([beta_init, alpha_init]) if beta_init is not None else alpha_init
    free = numpy.isnan(numpy.concatenate([beta_arg, alpha_arg])) if optbeta else numpy.isnan(alpha_arg)

    def objective(x: FloatArray) -> float:
        par = par0.copy()
        par[free] = x
        return problem.negloglik(par, alpha_arg, beta_arg, alpha_transf=True)

    if optimizer in INEQUALITY and link == 'identity':
        ui = numpy.column_stack([numpy.zeros((k, p)), Z]) if optbeta else Z

        def full_objective(x: FloatArray) -> float:
            return problem.negloglik(x, alpha_arg, beta_arg, alpha_transf=True)

        if optimizer == 'constrOptim':
            res = metafopy.models.rma_ls_opt.constr_optim(par0, full_objective, ui, numpy.zeros(k))
            par_opt = res.par.copy()
        else:
            choice = metafopy.models.sel_opt.OptimizerChoice(optimizer, 'BFGS')
            par_opt = minimise_standin(full_objective, par0, choice, con, ui)
    elif optimizer in SCIPY_METHODS:
        res = metafopy.models.rma_ls_opt.minimise(objective, par0[free], method=optimizer, maxiter=con.maxiter, polish_converged=con.polish_converged)
        par_opt = par0.copy()
        par_opt[free] = res.par
    else:
        choice = metafopy.models.sel_opt.OptimizerChoice(optimizer, 'BFGS')
        par_opt = par0.copy()
        par_opt[free] = minimise_standin(objective, par0[free], choice, con, None)

    if link == 'identity' and not optbeta:
        full = numpy.where(numpy.isnan(alpha_arg), par_opt, alpha_arg)
        par_opt = numpy.where(
            numpy.isnan(alpha_arg),
            metafopy.models.rma_ls_opt.polish_identity(
                full,
                lambda a: problem.loglik(a, alpha_arg, beta_arg, alpha_transf=False),
                problem.ls_score,
                problem.ls_information,
                Z,
                numpy.isnan(alpha_arg),
            ),
            alpha_arg,
        )
        if con.restarts and user_w is None and numpy.all(numpy.isnan(alpha_arg)) and numpy.ptp(Z, axis=0).max() > 0.0:
            par_opt = identity_restart_fit(problem, par_opt, alpha_arg)

    if link == 'log' and not optbeta and optimizer == 'BFGS' and metafopy.models.sel_opt.partial_match(con.optimizer, OPTIMIZERS, 'optimizer') == 'nlminb':
        par_opt = polish_log_fit(problem, par_opt, alpha_arg, beta_arg)

    if (
        link == 'log'
        and con.restarts
        and not optbeta
        and user_w is None
        and numpy.all(numpy.isnan(alpha_arg))
        and numpy.all(numpy.isinf(alpha_min))
        and numpy.all(numpy.isinf(alpha_max))
    ):
        par_opt = restart_fit(problem, par_opt, optimizer, con, alpha_arg, beta_arg)

    pos_beta = numpy.arange(p) if optbeta else numpy.arange(0)
    pos_alpha = numpy.arange(p, p + q) if optbeta else numpy.arange(q)
    par_opt[pos_alpha] = mapfun(par_opt[pos_alpha], alpha_min, alpha_max)
    if optbeta:
        par_opt[pos_beta] = numpy.where(numpy.isnan(beta_arg), par_opt[pos_beta], beta_arg)
    par_opt[pos_alpha] = numpy.where(numpy.isnan(alpha_arg), par_opt[pos_alpha], alpha_arg)

    beta_hat = par_opt[pos_beta]
    alpha_hat = par_opt[pos_alpha]

    # -- covariance matrix from the numerical Hessian ---------------------------
    beta_hes = beta_hat if con.hes_beta_fix else beta_arg
    alpha_hes = alpha_hat if con.hes_alpha_fix else alpha_arg
    if optbeta:
        vcovmat = numpy.full((p + q, p + q), math.nan)
        hest = numpy.concatenate([numpy.isnan(beta_hes), numpy.isnan(alpha_hes)])
        hesspars = numpy.concatenate([beta_hat, alpha_hat])
    else:
        vcovmat = numpy.full((q, q), math.nan)
        hest = numpy.isnan(alpha_hes)
        hesspars = alpha_hat.copy()
    H: FloatArray | None = None
    if hest.any() and not skiphes:
        try:
            H = metafopy.core.numderiv.hessian(
                lambda x: problem.negloglik(x, alpha_hes, beta_hes, alpha_transf=False),
                hesspars,
                d=con.hessian_d,
                eps=con.hessian_eps,
                r=con.hessian_r,
            )
        except ValueError, ArithmeticError, numpy.linalg.LinAlgError:
            warnings.warn('Error when trying to compute the Hessian.', stacklevel=3)
        if H is not None:
            inv = pos_chol_inverse(H[numpy.ix_(hest, hest)])
            if inv is None:
                warnings.warn('Error when trying to invert the Hessian.', stacklevel=3)
            else:
                vcovmat[numpy.ix_(hest, hest)] = inv
    if optbeta:
        vb_opt: FloatArray | None = vcovmat[:p, :p]
        va = vcovmat[p:, p:].copy()
    else:
        vb_opt = None
        va = vcovmat.copy()
    hest_a = hest[p:] if optbeta else hest

    if numpy.any(alpha_hat <= alpha_min + 10 * DBL_EPS**0.25) or numpy.any(alpha_hat >= alpha_max - 10 * DBL_EPS**0.25):
        warnings.warn(
            "One or more 'alpha' estimates are (almost) equal to their lower or upper bound.\n"
            "Treat results with caution (or consider adjusting 'alpha_min' and/or 'alpha_max').",
            stacklevel=3,
        )

    if mZ is not None:
        alpha_hat = mZ @ alpha_hat
        va[~hest_a, :] = 0.0
        va[:, ~hest_a] = 0.0
        va = mZ @ va @ mZ.T
        va[~hest_a, :] = math.nan
        va[:, ~hest_a] = math.nan
        Z = Zsave

    att_idx = metafopy.models.rma_uni.set_btt(att, q, Z_int_incl, znames)
    if abbrev:
        znames = metafopy.models.rma_uni.abbreviate_names(znames)
    m_alpha = len(att_idx)
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    tdist = test in ('knha', 'adhoc', 't')
    ddf_alpha = k - q if tdist else None
    QS = quad_form(alpha_hat, va, att_idx)
    with numpy.errstate(invalid='ignore'):
        se_alpha = numpy.sqrt(numpy.diag(va))
        zval_alpha = alpha_hat / se_alpha
    crit_level = metafopy.models.rma_uni.level_to_alpha(level)
    if test == 'knha' or test == 'adhoc' or test == 't':
        assert ddf_alpha is not None  # auditme: accept[assert-in-product-code] narrowed by tdist above
        QS = QS / m_alpha
        QSdf: tuple[int, float | None] = (m_alpha, ddf_alpha)
        QSp = float(scipy.stats.f.sf(QS, m_alpha, ddf_alpha)) if ddf_alpha > 0 else math.nan
        pval_alpha = 2 * scipy.stats.t.sf(numpy.abs(zval_alpha), ddf_alpha) if ddf_alpha > 0 else numpy.full(q, math.nan)
        crit_alpha = float(scipy.stats.t.isf(crit_level / 2, ddf_alpha)) if ddf_alpha > 0 else math.nan
    elif test == 'z':
        QSdf = (m_alpha, None)
        QSp = float(scipy.stats.chi2.sf(QS, m_alpha)) if not math.isnan(QS) else math.nan
        pval_alpha = 2 * scipy.stats.norm.sf(numpy.abs(zval_alpha))
        crit_alpha = float(scipy.stats.norm.isf(crit_level / 2))
    else:
        typing.assert_never(test)
    ci_lb_alpha = alpha_hat - crit_alpha * se_alpha
    ci_ub_alpha = alpha_hat + crit_alpha * se_alpha

    with numpy.errstate(over='ignore'):
        tau2 = numpy.exp(Z @ alpha_hat) if link == 'log' else Z @ alpha_hat

    # -- location coefficients ----------------------------------------------------
    opt_loc = (beta_hat.copy(), vb_opt.copy()) if (optbeta and vb_opt is not None) else None
    beta_out, vb, rss_f, s2w = location_estimates(yi, vi, X, tau2, user_w, weighted, test, opt_loc)
    ddf = (k - p if dfs is None else metafopy.models.rma_uni.first_df(dfs)) if tdist else None

    QM = quad_form(beta_out, vb, btt_idx)
    with numpy.errstate(invalid='ignore'):
        se = numpy.sqrt(numpy.diag(vb))
        zval = beta_out / se
    if test == 'knha' or test == 'adhoc' or test == 't':
        assert ddf is not None  # auditme: accept[assert-in-product-code] narrowed by tdist above
        QM = QM / m
        QMdf: tuple[int, float | None] = (m, ddf)
        QMp = float(scipy.stats.f.sf(QM, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * scipy.stats.t.sf(numpy.abs(zval), ddf) if ddf > 0 else numpy.full(p, math.nan)
        crit = float(scipy.stats.t.isf(crit_level / 2, ddf)) if ddf > 0 else math.nan
    elif test == 'z':
        QMdf = (m, None)
        QMp = float(scipy.stats.chi2.sf(QM, m)) if not math.isnan(QM) else math.nan
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(crit_level / 2))
    else:
        typing.assert_never(test)
    ci_lb = beta_out - crit * se
    ci_ub = beta_out + crit * se

    # -- heterogeneity statistics ---------------------------------------------------
    QE = QEp = vt = math.nan
    I2 = numpy.full(k, math.nan)
    H2 = numpy.full(k, math.nan)
    if allvipos:
        if k > p:
            w_fe = 1.0 / vi
            WX = w_fe[:, None] * X
            P = numpy.diag(w_fe) - WX @ metafopy.core.linalg.invcalc(X, w_fe) @ WX.T
            QE = max(0.0, float(yi @ P @ yi))
            QEp = float(scipy.stats.chi2.sf(QE, k - p))
            vt = (k - p) / float(numpy.trace(P))
            I2 = 100 * tau2 / (vt + tau2)
            H2 = tau2 / vt + 1
        else:
            QE, QEp, vt = 0.0, 1.0, 0.0
            I2 = numpy.zeros(k)
            H2 = numpy.ones(k)

    # -- fit statistics -------------------------------------------------------------
    q_est = int(numpy.sum(numpy.isnan(alpha_arg)))
    parms = (int(numpy.sum(numpy.isnan(beta_arg))) if optbeta else p) + q_est
    wi = 1.0 / (vi + tau2)
    lv = numpy.log(vi + tau2)
    ll_ml = -0.5 * k * math.log(2 * math.pi) - 0.5 * float(lv.sum()) - 0.5 * rss_f
    ll_reml = (
        -0.5 * (k - p) * math.log(2 * math.pi)
        + (0.5 * metafopy.core.linalg.logdet(X.T @ X) if con.REMLf else 0.0)
        - 0.5 * float(lv.sum())
        - 0.5 * metafopy.core.linalg.logdet((X.T * wi) @ X)
        - 0.5 * rss_f
    )
    if k > p:
        dev_ml = -2 * (ll_ml - float(numpy.sum(-0.5 * numpy.log(2 * math.pi * vi)))) if allvipos else -2 * ll_ml
    else:
        dev_ml = 0.0
    fs_ml = metafopy.models.rma_uni.fit_stats_from(ll_ml, dev_ml, parms, k)
    fs_reml = metafopy.models.rma_uni.fit_stats_from(ll_reml, -2 * ll_reml, parms, k - p)

    return metafopy.models.rma_ls_result.RMALSResult(
        beta=beta_out,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vb=vb,
        coef_names=tuple(names),
        alpha=alpha_hat,
        va=va,
        se_alpha=se_alpha,
        zval_alpha=zval_alpha,
        pval_alpha=numpy.asarray(pval_alpha, dtype=numpy.float64),
        ci_lb_alpha=ci_lb_alpha,
        ci_ub_alpha=ci_ub_alpha,
        alpha_names=tuple(znames),
        alpha_fix=~numpy.isnan(alpha_arg),
        tau2=tau2,
        k=k,
        p=p,
        q=q,
        m=m,
        m_alpha=m_alpha,
        btt=btt_idx,
        att=att_idx,
        QE=QE,
        QEp=QEp,
        QM=QM,
        QMp=QMp,
        QMdf=QMdf,
        QS=QS,
        QSp=QSp,
        QSdf=QSdf,
        I2=I2,
        H2=H2,
        vt=vt,
        fit_stats_ml=fs_ml,
        fit_stats_reml=fs_reml,
        parms=parms,
        method=method,
        test=test,
        level=crit_level,
        ddf=ddf,
        ddf_alpha=ddf_alpha,
        s2w=s2w,
        weighted=weighted,
        weights=user_w,
        link=link,
        optbeta=optbeta,
        beta_fix=~numpy.isnan(beta_arg),
        int_only=int_only,
        int_incl=int_incl,
        Z_int_only=Z_int_only,
        Z_int_incl=Z_int_incl,
        allvipos=allvipos,
        yi=yi,
        vi=vi,
        X=X,
        Z=Z,
        slab=md.slab,
        data=md,
        control=con,
        vcovmat=vcovmat if optbeta else None,
        H=H,
        formula_scale=formula_scale,
        Z_f=Z_full,
        coef_na=tuple(bool(d) for d in drop),
        coef_na_Z=tuple(bool(d) for d in drop_z),
        X_f=X_f,
    )


def fixed_values(values: numpy.typing.ArrayLike | None, n: int, what: str) -> FloatArray:
    """Fixed values of the coefficients (NaN for those to be estimated), recycled to length ``n``.

    Raises:
        metafopy.errors.InputError: the length does not match or a value is infinite.
    """
    if values is None:
        return numpy.full(n, math.nan)
    arr = numpy.atleast_1d(numpy.asarray(values, dtype=numpy.float64))
    if numpy.all(numpy.isnan(arr)):
        return numpy.full(n, math.nan)
    if numpy.any(numpy.isinf(arr)):
        raise metafopy.errors.InputError(f"Infinite values in '{what}' argument not allowed.")
    if arr.shape[0] == 1:
        arr = numpy.full(n, arr[0])
    if arr.shape[0] != n:
        raise metafopy.errors.InputError(f"Length of the '{what}' argument ({arr.shape[0]}) does not match the actual number of parameters ({n}).")
    return arr


def expand_bound(value: numpy.typing.ArrayLike, n: int, what: str) -> FloatArray:
    arr = numpy.atleast_1d(numpy.asarray(value, dtype=numpy.float64))
    if arr.shape[0] == 1:
        arr = numpy.full(n, arr[0])
    if arr.shape[0] != n:
        raise metafopy.errors.InputError(f"Length of the '{what}' argument ({arr.shape[0]}) does not match the actual number of parameters ({n}).")
    return arr
