"""Estimation of location-scale models, after the ``rma.ls`` branches of metafor's ``rma.uni``.

The model is ``y_i = x_i' beta + u_i + e_i`` with ``u_i ~ N(0, tau_i^2)`` and ``e_i ~ N(0, v_i)``,
where ``log(tau_i^2) = z_i' alpha`` (log link) or ``tau_i^2 = z_i' alpha`` (identity link); see Viechtbauer and
Lopez-Lopez (2022), doi:10.1002/jrsm.1562, and
https://github.com/wviechtb/metafor/blob/master/R/rma.uni.r and
https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.rma.ls.r.

The (restricted) log-likelihood is maximised over ``alpha`` (and over ``beta`` when ``optbeta=True``); the
covariance matrix of the scale coefficients comes from the numerical Hessian of ``numDeriv::hessian``.
"""

import dataclasses
import math
import warnings

import numpy
import numpy.typing
import scipy.stats

import metafopy.core.linalg
import metafopy.core.numderiv
import metafopy.errors
import metafopy.metadata
import metafopy.models.rma_ls_opt
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni

__all__ = ['LSProblem', 'fit_rma_ls', 'make_ls_control']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

DBL_EPS = float(numpy.finfo(float).eps)
SCIPY_METHODS = ('BFGS', 'CG', 'L-BFGS-B', 'Nelder-Mead')
CONSTRAINED = ('constrOptim',)


def make_ls_control(
    control: metafopy.models.rma_ls_result.LSControl | dict[str, object] | None,
) -> metafopy.models.rma_ls_result.LSControl:
    """Build an :class:`~metafopy.models.rma_ls_result.LSControl` from a mapping with metafor's (``.`` or ``_``) names.

    Raises:
        metafopy.errors.ArgumentError: an unknown control argument was given.
    """
    if control is None:
        return metafopy.models.rma_ls_result.LSControl()
    if isinstance(control, metafopy.models.rma_ls_result.LSControl):
        return control
    names = {f.name for f in dataclasses.fields(metafopy.models.rma_ls_result.LSControl)}
    kwargs: dict[str, object] = {}
    for key, value in control.items():
        name = key.replace('.', '_')
        if name not in names:
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        kwargs[name] = value
    return metafopy.models.rma_ls_result.LSControl(**kwargs)  # type: ignore[arg-type]


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
    """Raw residuals and their standard errors for the weighted fit with heterogeneity ``tau2`` (metafor's ``rstandard``)."""
    w = 1.0 / (vi + tau2)
    beta = metafopy.core.linalg.invcalc(X, w) @ (X.T @ (w * yi))
    H = X @ metafopy.core.linalg.invcalc(X, w) @ (X.T * w)
    ei = yi - X @ beta
    ei[numpy.abs(ei) < 100 * DBL_EPS] = 0.0
    ImH = numpy.eye(yi.shape[0]) - H
    ve = ImH @ numpy.diag(vi + tau2) @ ImH.T
    return ei, numpy.sqrt(numpy.diag(ve))


def start_alpha(md: metafopy.metadata.MetaData, Z: FloatArray, link: str, is_int: BoolArray) -> FloatArray:
    """Starting values for the scale coefficients: a regression of the transformed squared residuals of a HE fit on ``Z``.

    Raises:
        metafopy.errors.EstimationError: no suitable starting values were found.
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
) -> metafopy.models.rma_ls_result.RMALSResult:
    """Fit a location-scale model to normalised inputs: ``md`` (``yi``, ``vi``, ``X``) and the scale model matrix ``Z``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: the optimisation failed.
    """
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    if method not in ('ML', 'REML'):
        raise metafopy.errors.InputError('Location-scale models can only be fitted with ML or REML estimation.')
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    if link not in ('log', 'identity'):
        raise metafopy.errors.InputError("The 'link' argument must be either 'log' or 'identity'.")
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
    if drop.any():
        warnings.warn('Redundant predictors dropped from the model.', stacklevel=3)
        X = X[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]
    is_int_x = numpy.array([metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])])
    int_incl = bool(is_int_x.any())
    if int_incl:
        X = numpy.column_stack([numpy.ones(k), X[:, ~is_int_x]])
        names = ['intrcpt', *[n for n, i in zip(names, is_int_x, strict=True) if not i]]
    p = X.shape[1]
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])
    if not (int_only and k == 1) and p + 1 > k:
        raise metafopy.errors.InputError('Number of parameters to be estimated is larger than the number of observations.')
    btt_idx = metafopy.models.rma_uni.set_btt(btt, p, int_incl, names)
    m = len(btt_idx)

    if not metafopy.core.linalg.is_positive_definite(X.T @ X, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')

    # -- scale model matrix ---------------------------------------------------
    znames = list(z_names)
    drop_z = metafopy.core.linalg.drop_redundant(Z)
    if drop_z.any():
        warnings.warn('Redundant predictors dropped from the scale model.', stacklevel=3)
        Z = Z[:, ~drop_z]
        znames = [n for n, d in zip(znames, drop_z, strict=True) if not d]
    is_int_z = numpy.array([metafopy.core.linalg.is_intercept(Z[:, j]) for j in range(Z.shape[1])])
    Z_int_incl = bool(is_int_z.any())
    if Z_int_incl:
        Z = numpy.column_stack([numpy.ones(k), Z[:, ~is_int_z]])
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

    optimizer = con.optimizer
    if link == 'identity':
        if optimizer == 'nlminb':
            optimizer = 'constrOptim'
        elif optimizer not in CONSTRAINED:
            warnings.warn("Can only use optimizer 'constrOptim' when link='identity' (resetting to 'constrOptim').", stacklevel=3)
            optimizer = 'constrOptim'
    elif optimizer == 'nlminb':
        optimizer = 'BFGS'
    elif optimizer in CONSTRAINED:
        raise metafopy.errors.InputError(f"Cannot use '{optimizer}' optimizer when using a log link.")
    elif optimizer not in SCIPY_METHODS:
        raise metafopy.errors.InputError(f'unknown optimizer {optimizer!r}')

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

    alpha_init = numpy.minimum(numpy.maximum(alpha_init, alpha_min), alpha_max)
    alpha_init = mapinvfun(alpha_init, alpha_min, alpha_max)

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

    if optimizer == 'constrOptim':
        ui = numpy.column_stack([numpy.zeros((k, p)), Z]) if optbeta else Z
        res = metafopy.models.rma_ls_opt.constr_optim(
            par0,
            lambda x: problem.negloglik(x, alpha_arg, beta_arg, alpha_transf=True),
            ui,
            numpy.zeros(k),
        )
        par_opt = res.par.copy()
    else:
        res = metafopy.models.rma_ls_opt.minimise(objective, par0[free], method=optimizer, maxiter=con.maxiter)
        par_opt = par0.copy()
        par_opt[free] = res.par

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
    m_alpha = len(att_idx)
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    tdist = test in ('knha', 'adhoc', 't')
    ddf_alpha = k - q if tdist else None
    QS = quad_form(alpha_hat, va, att_idx)
    with numpy.errstate(invalid='ignore'):
        se_alpha = numpy.sqrt(numpy.diag(va))
        zval_alpha = alpha_hat / se_alpha
    crit_level = metafopy.models.rma_uni.level_to_alpha(level)
    if tdist:
        assert ddf_alpha is not None  # auditme: accept[assert-in-product-code] narrowed by tdist above
        QS = QS / m_alpha
        QSdf: tuple[int, float | None] = (m_alpha, ddf_alpha)
        QSp = float(scipy.stats.f.sf(QS, m_alpha, ddf_alpha)) if ddf_alpha > 0 else math.nan
        pval_alpha = 2 * scipy.stats.t.sf(numpy.abs(zval_alpha), ddf_alpha) if ddf_alpha > 0 else numpy.full(q, math.nan)
        crit_alpha = float(scipy.stats.t.isf(crit_level / 2, ddf_alpha)) if ddf_alpha > 0 else math.nan
    else:
        QSdf = (m_alpha, None)
        QSp = float(scipy.stats.chi2.sf(QS, m_alpha)) if not math.isnan(QS) else math.nan
        pval_alpha = 2 * scipy.stats.norm.sf(numpy.abs(zval_alpha))
        crit_alpha = float(scipy.stats.norm.isf(crit_level / 2))
    ci_lb_alpha = alpha_hat - crit_alpha * se_alpha
    ci_ub_alpha = alpha_hat + crit_alpha * se_alpha

    with numpy.errstate(over='ignore'):
        tau2 = numpy.exp(Z @ alpha_hat) if link == 'log' else Z @ alpha_hat

    # -- location coefficients ----------------------------------------------------
    opt_loc = (beta_hat.copy(), vb_opt.copy()) if (optbeta and vb_opt is not None) else None
    beta_out, vb, rss_f, s2w = location_estimates(yi, vi, X, tau2, user_w, weighted, test, opt_loc)
    ddf = k - p if tdist else None

    QM = quad_form(beta_out, vb, btt_idx)
    with numpy.errstate(invalid='ignore'):
        se = numpy.sqrt(numpy.diag(vb))
        zval = beta_out / se
    if tdist:
        assert ddf is not None  # auditme: accept[assert-in-product-code] narrowed by tdist above
        QM = QM / m
        QMdf: tuple[int, float | None] = (m, ddf)
        QMp = float(scipy.stats.f.sf(QM, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * scipy.stats.t.sf(numpy.abs(zval), ddf) if ddf > 0 else numpy.full(p, math.nan)
        crit = float(scipy.stats.t.isf(crit_level / 2, ddf)) if ddf > 0 else math.nan
    else:
        QMdf = (m, None)
        QMp = float(scipy.stats.chi2.sf(QM, m)) if not math.isnan(QM) else math.nan
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(crit_level / 2))
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
            ymc = yi - yi.mean()
            QE = max(0.0, float(ymc @ P @ ymc))
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
    fs_ml = metafopy.models.rma_uni.FitStats(
        ll=ll_ml,
        dev=dev_ml,
        AIC=-2 * ll_ml + 2 * parms,
        BIC=-2 * ll_ml + parms * math.log(k),
        AICc=metafopy.models.rma_uni.aicc(ll_ml, parms, k),
    )
    fs_reml = metafopy.models.rma_uni.FitStats(
        ll=ll_reml,
        dev=-2 * ll_reml,
        AIC=-2 * ll_reml + 2 * parms,
        BIC=-2 * ll_reml + parms * math.log(k - p) if k > p else math.nan,
        AICc=metafopy.models.rma_uni.aicc(ll_reml, parms, k - p),
    )

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
