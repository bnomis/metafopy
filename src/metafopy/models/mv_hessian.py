"""Numerical Hessians of the log-likelihood of :func:`metafopy.rma_mv` (``optbeta`` and ``cvvc``).

metafor obtains the covariance matrix of the coefficients when they are optimised along with the variance components
(``optbeta=TRUE``) and the covariance matrix of the variance components (``cvvc``) from the Hessian of ``-ll``, computed with
``numDeriv::hessian`` (Richardson extrapolation, ``r = 8``) and inverted by a Cholesky decomposition. Ports the relevant parts of
https://github.com/wviechtb/metafor/blob/master/R/rma.mv.r; the numerical derivative is :func:`metafopy.core.numderiv.hessian`.
"""

import dataclasses
import math
import warnings

import numpy
import numpy.typing

import metafopy.core.numderiv
import metafopy.models.mv_lik
import metafopy.models.mv_structs

__all__ = ['HessianArgs', 'HessianResult', 'coefficient_hessian', 'variance_hessian']

UNNAMED = ''

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]


@dataclasses.dataclass(frozen=True, slots=True)
class HessianArgs:
    """``hessianCtrl`` of metafor's ``numDeriv`` branch and the tolerance below which a row of the Hessian counts as fixed."""

    d: float
    eps: float
    r: int
    tol: float


@dataclasses.dataclass(frozen=True, slots=True)
class HessianResult:
    """The Hessian of ``-ll`` (fixed parameters removed), its inverse and the parameter names; ``None`` where metafor gives ``NA``."""

    hessian: FloatArray | None
    vvc: FloatArray | None
    names: tuple[str, ...]
    keep: BoolArray


def chol2inv(hessian: FloatArray) -> FloatArray | None:
    """``chol2inv(chol(hessian))``, or ``None`` when the matrix is not positive definite or the result is not finite."""
    if hessian.shape[0] == 0:
        return numpy.zeros((0, 0))
    if not numpy.all(numpy.isfinite(hessian)):
        return None
    try:
        chol = numpy.linalg.cholesky(hessian)
    except numpy.linalg.LinAlgError:
        return None
    inv_chol = numpy.linalg.inv(chol)
    out = numpy.asarray(inv_chol.T @ inv_chol, dtype=numpy.float64)
    return out if numpy.all(numpy.isfinite(out)) else None


def richardson(problem: metafopy.models.mv_lik.Problem, x: FloatArray, reml: bool, args: HessianArgs, vctransf: bool, vccov: bool) -> FloatArray:
    def func(par: FloatArray) -> float:
        return metafopy.models.mv_lik.negative_loglik(problem, par, reml, vctransf, vccov)

    with numpy.errstate(all='ignore'):
        return metafopy.core.numderiv.hessian(func, x, d=args.d, eps=args.eps, r=args.r)


def fixed_rows(hessian: FloatArray, tol: float) -> BoolArray:
    """Rows that are not (essentially) all zero, i.e. the parameters that were actually estimated."""
    with numpy.errstate(invalid='ignore'):
        return numpy.asarray(~numpy.all(numpy.abs(hessian) <= tol, axis=1), dtype=numpy.bool_)


def coefficient_hessian(
    problem: metafopy.models.mv_lik.Problem,
    par: FloatArray,
    reml: bool,
    args: HessianArgs,
    int_incl: bool,
) -> HessianResult:
    """The Hessian over the coefficients and the transformed variance components at the optimum (``optbeta=TRUE``)."""
    p = problem.p
    H = richardson(problem, par, reml, args, True, False)
    first = 0 if int_incl else 1
    names = [f'beta{first + j}' for j in range(p)] + [UNNAMED] * (par.shape[0] - p)
    keep = fixed_rows(H, args.tol)
    kept = tuple(n for n, flag in zip(names, keep, strict=True) if flag)
    return HessianResult(hessian=H[numpy.ix_(keep, keep)], vvc=None, names=kept, keep=keep)


def inner_names(prefix: str, count: int) -> list[str]:
    """Names of the ``count`` correlations/covariances of an unstructured term, in the order of the lower triangle."""
    if count == 1:
        return [prefix]
    nlevels = round((1 + math.sqrt(1 + 8 * count)) / 2)
    return [f'{prefix}.{i + 1}.{j + 1}' for j in range(nlevels) for i in range(j + 1, nlevels)]


def numbered(prefix: str, count: int) -> list[str]:
    return [prefix] if count == 1 else [f'{prefix}.{i + 1}' for i in range(count)]


def lower_triangle_order(n: int) -> list[int]:
    """Positions of the entries of ``(variances, covariances)`` in the order of the lower triangle (with diagonal) of an ``n x n`` matrix."""
    pos = numpy.full((n, n), -1)
    numpy.fill_diagonal(pos, numpy.arange(n))
    low = numpy.tril_indices(n, -1)
    order = numpy.lexsort((low[0], low[1]))
    pos[low[0][order], low[1][order]] = numpy.arange(n, n * (n + 1) // 2)
    keep = numpy.tril_indices(n)
    order2 = numpy.lexsort((keep[0], keep[1]))
    return [int(v) for v in pos[keep[0][order2], keep[1][order2]]]


def variance_hessian(
    problem: metafopy.models.mv_lik.Problem,
    reml: bool,
    cvvc: str,
    par: FloatArray,
    fitted: metafopy.models.mv_lik.Fit,
    args: HessianArgs,
    structs: tuple[str, str],
    any_fixed: bool,
) -> HessianResult:
    """The Hessian of ``-ll`` for the variance components and its inverse (metafor's ``cvvc``).

    ``cvvc`` is ``'varcor'`` (variances and correlations on their original scale), ``'varcov'`` (variances and covariances;
    only for ``UN`` and ``GEN`` structures without fixed components) or ``'transf'`` (the transformed parameters the optimiser
    works on). Parameters that are fixed or do not occur in the model (all-zero rows) are dropped.
    """
    lay = problem.layout
    with_s, with_g, with_h = problem.with_s, problem.g is not None, problem.h is not None
    if cvvc == 'varcov' and any_fixed:
        warnings.warn("Cannot use cvvc='varcov' when one or more components are fixed. Setting cvvc='varcor'.", stacklevel=4)
        cvvc = 'varcor'
    if cvvc == 'varcov' and any(s not in metafopy.models.mv_structs.UNSTRUCTURED for s in structs):
        warnings.warn("Cannot use cvvc='varcov' for the specified structure(s). Setting cvvc='varcor'.", stacklevel=4)
        cvvc = 'varcor'

    sig = fitted.sigma2 if fitted.sigma2 is not None else numpy.zeros(lay.sigma2s)
    tau = fitted.tau2 if fitted.tau2 is not None else numpy.zeros(lay.tau2s)
    rho = fitted.rho if fitted.rho is not None else numpy.zeros(lay.rhos)
    gam = fitted.gamma2 if fitted.gamma2 is not None else numpy.zeros(lay.gamma2s)
    phi = fitted.phi if fitted.phi is not None else numpy.zeros(lay.phis)
    if cvvc == 'varcov':
        if fitted.G is not None:
            rho = metafopy.models.mv_structs.lower_tri(fitted.G)
        if fitted.H is not None:
            phi = metafopy.models.mv_structs.lower_tri(fitted.H)
    if cvvc == 'transf':
        x = par[problem.offset :].copy()
        sub = problem
    else:
        x = numpy.concatenate([sig, tau, rho, gam, phi])
        sub = dataclasses.replace(
            problem,
            g=None if problem.g is None else dataclasses.replace(problem.g, cholesky=False),
            h=None if problem.h is None else dataclasses.replace(problem.h, cholesky=False),
        )
    H = richardson(sub, x, reml, args, cvvc == 'transf', cvvc == 'varcov')

    term_rho, term_phi = (
        ('cov1' if with_h else 'cov', 'cov2') if cvvc == 'varcov' else (metafopy.models.mv_structs.G_NAMES[1], metafopy.models.mv_structs.H_NAMES[1])
    )
    names = (
        numbered('sigma^2', lay.sigma2s)
        + numbered('tau^2', lay.tau2s)
        + inner_names(term_rho, lay.rhos)
        + numbered('gamma^2', lay.gamma2s)
        + inner_names(term_phi, lay.phis)
    )
    present = [with_s] * lay.sigma2s + [with_g] * (lay.tau2s + lay.rhos) + [with_h] * (lay.gamma2s + lay.phis)
    if not any(present):
        return HessianResult(hessian=None, vvc=None, names=(), keep=numpy.zeros(0, dtype=numpy.bool_))
    sel = numpy.flatnonzero(present)
    H = H[numpy.ix_(sel, sel)]
    names = [names[i] for i in sel]

    if cvvc == 'varcov' and with_g:
        pos_g = lower_triangle_order(lay.tau2s)
        pos = list(range(lay.sigma2s)) + [lay.sigma2s + v for v in pos_g] if with_s else list(pos_g)
        if with_h:
            top = max(pos) + 1
            pos += [top + v for v in lower_triangle_order(lay.gamma2s)]
        H = H[numpy.ix_(pos, pos)]
        names = [names[i] for i in pos]

    keep = fixed_rows(H, args.tol)
    H = H[numpy.ix_(keep, keep)]
    kept = tuple(n for n, flag in zip(names, keep, strict=True) if flag)
    return HessianResult(hessian=H, vvc=chol2inv(H), names=kept, keep=keep)
