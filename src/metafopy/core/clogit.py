"""Conditional logistic regression for stratified 2x2 tables: the ``clogit`` and ``clogistic`` optimisers of ``rma.glmm``.

With ``control=list(optimizer='clogit')`` or ``'clogistic'`` and ``model='CM.EL'``, ``rma.glmm``
(https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r) fits the fixed-effects model of odds ratios by
expanding the tables to individual patient data and calling ``survival::clogit()`` (Therneau, the Cox model with
``ties='exact'``, https://github.com/therneau/survival/blob/master/src/coxexact.c) or ``Epi::clogistic()``
(https://github.com/cran/Epi/blob/master/src/clogit.c). Both maximise the conditional (exact) likelihood of the
studies, i.e. the likelihood of the non-central hypergeometric distribution of ``ai`` given the margins, by
Newton-Raphson. Here the individual patient data are never built: the subjects of a study are exchangeable within
the two groups, so the sums over all case sets of the exact method are the sums over the number ``J`` of cases
in group 1, ``P(J = j)`` proportional to ``choose(n1, j) * choose(n2, m - j) * exp(j * eta)``, and the score and
information of a study are ``(a - E[J]) * x`` and ``Var(J) * x x'``.

Both routines repeat the Newton-Raphson iterations of the originals step by step (the Cholesky decomposition
``cholesky2`` that detects aliased columns, step halving, the convergence rules and their tolerances), so that
they stop at the same iterate, also when a coefficient is infinite (tables with zero cells in the saturated
model). ``coxph_exact`` rescales the columns of the design as ``coxexact.fit`` does. Differences from the C code:
no overflow of ``exp()`` (the sums are evaluated in the log scale), so the halving that the originals do for
non-finite values is not needed.
"""

import collections.abc
import dataclasses
import functools
import math
import sys
import typing

import numpy
import numpy.typing
import scipy.special

import metafopy.core.nchg
import metafopy.errors

__all__ = [
    'ConditionalFit',
    'Strata',
    'cholesky2',
    'chsolve2',
    'clogistic',
    'clogistic_settings',
    'clogit',
    'clogit_settings',
    'coxph_exact',
    'coxph_ties',
]

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

COX_EPS = 1e-9
COX_TOLER_CHOL = sys.float_info.epsilon**0.75
COX_TOLER_INF = math.sqrt(COX_EPS)
COX_ITER_MAX = 20
EPI_EPS = 1e-6
EPI_TOLER_CHOL = math.sqrt(sys.float_info.epsilon)
EPI_ITER_MAX = 20
NOT_CONVERGED = 1000


@dataclasses.dataclass(frozen=True, slots=True)
class Strata:
    """The studies as strata: the non-central hypergeometric support of each table and the number of cases in group 1 (``ai``).

    ``n1`` and ``n2`` are the group sizes ``ai + bi`` and ``ci + di`` and ``deaths`` the number of events ``ai + ci`` of a study.
    """

    tables: metafopy.core.nchg.Tables
    cases: FloatArray
    n1: FloatArray
    n2: FloatArray
    deaths: FloatArray

    @classmethod
    def from_counts(cls, ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> Strata:
        """Strata of the tables ``ai, bi, ci, di``.

        Raises:
            metafopy.errors.InputError: a count is negative or not an integer.
        """
        tables = metafopy.core.nchg.Tables.from_counts(ai, bi, ci, di)
        as_float = functools.partial(numpy.asarray, dtype=numpy.float64)
        return cls(tables, as_float(ai), as_float(ai + bi), as_float(ci + di), as_float(ai + ci))


@dataclasses.dataclass(frozen=True, slots=True)
class ConditionalFit:
    """A fitted conditional logistic regression.

    ``coef`` holds NaN for aliased columns, ``vcov`` covers the other columns in order (as ``coef()`` and ``vcov()``
    of the R fits do). ``loglik`` is the maximised conditional (partial) log-likelihood, ``flag`` the value of the
    last ``cholesky2`` rank (``NOT_CONVERGED`` when the iterations ran out). ``warnings`` are the messages R issues.
    """

    coef: FloatArray
    vcov: FloatArray
    loglik: float
    iterations: int
    flag: int
    warnings: tuple[str, ...]

    @property
    def aliased(self) -> BoolArray:
        return numpy.asarray(numpy.isnan(self.coef), dtype=numpy.bool_)


def cholesky2(matrix: FloatArray, toler: float) -> int:
    """Cholesky decomposition ``F D F'`` of a non-negative definite matrix, in place in the lower triangle (``cholesky2.c`` of survival).

    A column whose pivot is below ``toler`` times the largest diagonal element is deemed redundant and its diagonal set
    to zero. Returns the rank, negative if the matrix is not non-negative definite.
    """
    n = matrix.shape[0]
    eps = 0.0
    for i in range(n):
        if matrix[i, i] > eps:
            eps = matrix[i, i]
        for j in range(i + 1, n):
            matrix[j, i] = matrix[i, j]
    eps = toler if eps == 0 else eps * toler
    rank = 0
    nonneg = 1
    for i in range(n):
        pivot = matrix[i, i]
        if not math.isfinite(pivot) or pivot < eps:
            matrix[i, i] = 0.0
            if pivot < -8 * eps:
                nonneg = -1
        else:
            rank += 1
            for j in range(i + 1, n):
                temp = matrix[j, i] / pivot
                matrix[j, i] = temp
                matrix[j, j] -= temp * temp * pivot
                for k in range(j + 1, n):
                    matrix[k, j] -= temp * matrix[k, i]
    return rank * nonneg


def chsolve2(matrix: FloatArray, y: FloatArray) -> None:
    """Solve ``A b = y`` in place from the decomposition of :func:`cholesky2` (``chsolve2.c``); a zero pivot gives ``b = 0``."""
    n = matrix.shape[0]
    for i in range(n):
        temp = y[i]
        for j in range(i):
            temp -= y[j] * matrix[i, j]
        y[i] = temp
    for i in range(n - 1, -1, -1):
        if matrix[i, i] == 0:
            y[i] = 0.0
        else:
            temp = y[i] / matrix[i, i]
            for j in range(i + 1, n):
                temp -= y[j] * matrix[j, i]
            y[i] = temp


def chinv2(matrix: FloatArray) -> FloatArray:
    """Inverse from the decomposition of :func:`cholesky2` (``chinv2.c``), returned as a full symmetric matrix; aliased rows and columns are zero."""
    a = matrix.copy()
    n = a.shape[0]
    for i in range(n):
        if a[i, i] > 0:
            a[i, i] = 1 / a[i, i]
            for j in range(i + 1, n):
                a[j, i] = -a[j, i]
                for k in range(i):
                    a[j, k] += a[j, i] * a[i, k]
    for i in range(n):
        if a[i, i] == 0:
            for j in range(i):
                a[j, i] = 0.0
            for j in range(i, n):
                a[i, j] = 0.0
        else:
            for j in range(i + 1, n):
                temp = a[j, i] * a[j, j]
                a[i, j] = temp
                for k in range(i, j):
                    a[i, k] += temp * a[j, k]
    for i in range(1, n):
        for j in range(i):
            a[i, j] = a[j, i]
    return a


def conditional_terms(strata: Strata, X: FloatArray, beta: FloatArray) -> tuple[float, FloatArray, FloatArray]:
    """Partial log-likelihood, score and information of the exact conditional model at ``beta``.

    The log-likelihood of a study is ``a * eta - log(sum_j choose(n1, j) choose(n2, m - j) exp(j * eta))`` with
    ``eta = X[i] @ beta``, the form that ``coxph(ties='exact')`` and ``clogistic`` maximise (it differs from the
    log-probability of ``ai`` by a constant).
    """
    eta = X @ beta
    p = X.shape[1]
    loglik = 0.0
    score = numpy.zeros(p)
    info = numpy.zeros((p, p))
    tables = strata.tables
    for i in range(tables.k):
        support = tables.support[i]
        w = tables.log_base[i] + support * eta[i]
        norm = float(scipy.special.logsumexp(w))
        prob = numpy.exp(w - norm)
        mean = float(prob @ support)
        var = float(prob @ (support - mean) ** 2)
        loglik += strata.cases[i] * eta[i] - norm
        score += (strata.cases[i] - mean) * X[i]
        info += var * numpy.outer(X[i], X[i])
    return loglik, score, info


def column_scale(strata: Strata, X: FloatArray) -> FloatArray:
    """``scale()`` factors of the columns of the individual patient data design (``group1 * X``), 1 for the columns that ``coxph`` leaves alone.

    The columns whose values are all in ``{-1, 0, 1}`` (``nocenter``) are not rescaled. Centering does not change
    the conditional likelihood, only the scale matters here.
    """
    n1, n2 = strata.n1, strata.n2
    total = float(numpy.sum(n1 + n2))
    mean = (n1 @ X) / total
    ss = (n1 @ (X - mean) ** 2) + float(numpy.sum(n2)) * mean**2
    sd = numpy.sqrt(ss / (total - 1))
    present = n1 > 0
    keep = numpy.all(numpy.isin(X[present], (-1.0, 0.0, 1.0)), axis=0)
    return numpy.where(keep, 1.0, sd)


def coxph_exact(
    strata: Strata,
    X: FloatArray,
    *,
    init: FloatArray | None = None,
    eps: float = COX_EPS,
    toler_chol: float = COX_TOLER_CHOL,
    iter_max: int = COX_ITER_MAX,
    toler_inf: float = COX_TOLER_INF,
) -> ConditionalFit:
    """``survival::clogit(event ~ group1 * X + strata(study))`` with ``method='exact'`` (``coxexact.fit`` and ``coxexact.c``).

    ``X`` is the ``k x p`` design of the studies (its rows multiply the group 1 indicator).

    Raises:
        metafopy.errors.EstimationError: ``init`` has the wrong length.
    """
    nvar = X.shape[1]
    rescale = column_scale(strata, X)
    xs = X / rescale
    beta = start_values(init, nvar) * rescale
    loglik0, u, imat = conditional_terms(strata, xs, beta)
    loglik1 = loglik0
    fac = imat.copy()
    flag = cholesky2(fac, toler_chol)
    chsolve2(fac, u)
    iterations = 0
    old_beta = beta.copy()
    done = iter_max == 0 or not math.isfinite(loglik0)
    if not done:
        old_beta = beta.copy()
        beta = beta + u
        halving = False
        score = u
        for iterations in range(1, iter_max + 1):
            newlk, score, imat = conditional_terms(strata, xs, beta)
            fac = imat.copy()
            flag = cholesky2(fac, toler_chol)
            notfinite = not (math.isfinite(newlk) and bool(numpy.all(numpy.isfinite(score))) and bool(numpy.all(numpy.isfinite(imat))))
            if not notfinite and abs(1 - loglik1 / newlk) <= eps and not halving:
                loglik1 = newlk
                done = True
                break
            if iterations == iter_max:
                break
            if notfinite or newlk < loglik1:
                halving = True
                beta = (old_beta + beta) / 2
            else:
                halving = False
                loglik1 = newlk
                step = score.copy()
                chsolve2(fac, step)
                old_beta = beta.copy()
                beta = beta + step
        if not done:
            if iter_max > 1:
                beta = old_beta
                loglik1, score, imat = conditional_terms(strata, xs, beta)
                fac = numpy.triu(imat)
            else:
                loglik1 = newlk
            flag = NOT_CONVERGED
        u = score
    else:
        score = conditional_terms(strata, xs, beta)[1]
        u = score
    var = chinv2(fac)
    return finish_cox(beta, u, var, loglik1, iterations, flag, rescale, (eps, toler_inf, iter_max))


def finish_cox(
    beta: FloatArray, u: FloatArray, var: FloatArray, loglik: float, iterations: int, flag: int, rescale: FloatArray, tolerances: tuple[float, float, int]
) -> ConditionalFit:
    """Undo the rescaling, mark the aliased coefficients and collect the warnings of ``coxexact.fit``."""
    eps, toler_inf, iter_max = tolerances
    nvar = beta.shape[0]
    singular = numpy.diag(var) == 0 if flag < nvar else numpy.zeros(nvar, dtype=numpy.bool_)
    messages: list[str] = []
    if iter_max > 1:
        if flag == NOT_CONVERGED:
            messages.append('Ran out of iterations and did not converge')
        else:
            with numpy.errstate(all='ignore'):
                infs = numpy.abs(u @ var)
                bad = (infs > eps) & (infs > toler_inf * numpy.abs(beta))
            if bool(numpy.any(bad)):
                which = ','.join(str(j + 1) for j in numpy.flatnonzero(bad))
                messages.append(f'Loglik converged before variable {which} ; beta may be infinite. ')
    coef = beta / rescale
    coef[singular] = numpy.nan
    cov = var / numpy.outer(rescale, rescale)
    keep = ~singular
    return ConditionalFit(coef, cov[numpy.ix_(keep, keep)], loglik, iterations, flag, tuple(messages))


type Ties = typing.Literal['breslow', 'efron', 'exact']


@dataclasses.dataclass(frozen=True, slots=True)
class TiesDesign:
    """The centred and scaled covariates of the individual patient data as ``coxph.fit`` (``coxfit6.c``) builds them.

    ``z1`` are the rows of the studies' group 1 subjects, ``z0`` the row shared by all group 2 subjects (their covariates are 0 before centering).
    """

    z1: FloatArray
    z0: FloatArray
    means: FloatArray
    scale: FloatArray


def ties_design(strata: Strata, X: FloatArray) -> TiesDesign:
    """Centre each column that ``coxph`` does not leave alone (``nocenter``) at its mean and divide it by its mean absolute deviation."""
    n1, n2 = strata.n1, strata.n2
    total = float(numpy.sum(n1 + n2))
    present = n1 > 0
    keep = numpy.all(numpy.isin(X[present], (-1.0, 0.0, 1.0)), axis=0)
    mean = numpy.where(keep, 0.0, (n1 @ X) / total)
    deviation = (n1 @ numpy.abs(X - mean) + float(numpy.sum(n2)) * numpy.abs(mean)) / total
    scale = numpy.where(keep | (deviation <= 0), 1.0, 1.0 / numpy.where(deviation > 0, deviation, 1.0))
    return TiesDesign((X - mean) * scale, -mean * scale, mean, scale)


def ties_terms(strata: Strata, design: TiesDesign, beta: FloatArray, ties: Ties) -> tuple[float, FloatArray, FloatArray]:
    """Log-likelihood, score and information of the Breslow or Efron partial likelihood, all subjects of a study being tied at one time.

    The risk set of a study holds ``n1`` subjects with covariates ``z1`` and ``n2`` with ``z0``; the ``a`` cases of group 1 and the
    ``m - a`` cases of group 2 die. Efron's approximation removes the fraction ``j / m`` of the deaths from the risk set for ``j < m``.
    """
    nvar = beta.shape[0]
    loglik = 0.0
    score = numpy.zeros(nvar)
    info = numpy.zeros((nvar, nvar))
    z0 = design.z0
    r0 = math.exp(float(z0 @ beta))
    out0 = numpy.outer(z0, z0)
    for i in range(strata.tables.k):
        ndead = int(strata.deaths[i])
        if ndead == 0:
            continue
        n1, n2, a = strata.n1[i], strata.n2[i], strata.cases[i]
        c = ndead - a
        z1 = design.z1[i]
        r1 = math.exp(float(z1 @ beta))
        out1 = numpy.outer(z1, z1)
        s0, s1, s2 = n1 * r1 + n2 * r0, n1 * r1 * z1 + n2 * r0 * z0, n1 * r1 * out1 + n2 * r0 * out0
        d0, d1, d2 = a * r1 + c * r0, a * r1 * z1 + c * r0 * z0, a * r1 * out1 + c * r0 * out0
        loglik += a * float(z1 @ beta) + c * float(z0 @ beta)
        score += a * z1 + c * z0
        if ties == 'breslow' or ndead == 1:
            loglik -= ndead * math.log(s0)
            score -= ndead * s1 / s0
            info += ndead * (s2 / s0 - numpy.outer(s1, s1) / s0**2)
        else:
            frac = numpy.arange(1, ndead + 1) / ndead
            denom = (s0 - d0) + frac * d0
            avec = (s1 - d1)[None, :] + frac[:, None] * d1[None, :]
            cmat = (s2 - d2)[None, :, :] + frac[:, None, None] * d2[None, :, :]
            loglik -= float(numpy.sum(numpy.log(denom)))
            mean = avec / denom[:, None]
            score -= mean.sum(axis=0)
            info += numpy.sum(cmat / denom[:, None, None], axis=0) - mean.T @ mean
    return loglik, score, info


def coxph_ties(
    strata: Strata,
    X: FloatArray,
    *,
    ties: Ties = 'efron',
    init: FloatArray | None = None,
    eps: float = COX_EPS,
    toler_chol: float = COX_TOLER_CHOL,
    iter_max: int = COX_ITER_MAX,
    toler_inf: float = COX_TOLER_INF,
) -> ConditionalFit:
    """``survival::clogit(..., method='efron')`` or ``'breslow'`` (``'approximate'``): ``coxph.fit`` and ``coxfit6.c`` with all subjects of a study tied.

    The Efron and Breslow partial likelihoods approximate the exact conditional likelihood, so the estimates differ somewhat from those of
    :func:`coxph_exact`.

    Raises:
        metafopy.errors.EstimationError: ``init`` has the wrong length.
    """
    nvar = X.shape[1]
    design = ties_design(strata, X)
    scale = design.scale
    beta = start_values(init, nvar) / scale
    loglik0, u, imat = ties_terms(strata, design, beta, ties)
    loglik1 = loglik0
    fac = imat.copy()
    flag = cholesky2(fac, toler_chol)
    step = u.copy()
    chsolve2(fac, step)
    done = iter_max == 0 or not math.isfinite(loglik0)
    iterations = 0
    best = beta.copy()
    newbeta = beta + step
    if not done:
        halving = 0
        for iteration in range(1, iter_max + 1):
            iterations = iteration
            newlk, u, imat = ties_terms(strata, design, newbeta, ties)
            fac = imat.copy()
            flag = cholesky2(fac, toler_chol)
            notfinite = not (math.isfinite(newlk) and bool(numpy.all(numpy.isfinite(u))) and bool(numpy.all(numpy.isfinite(imat))))
            if not notfinite and abs(1 - loglik1 / newlk) <= eps:
                loglik1 = newlk
                best = newbeta
                if halving:
                    flag = -2
                done = True
                break
            if notfinite or newlk < loglik1:
                halving += 1
                newbeta = (newbeta + halving * best) / (halving + 1.0)
            else:
                halving = 0
                loglik1 = newlk
                step = u.copy()
                chsolve2(fac, step)
                best = newbeta
                newbeta = newbeta + step
        if not done:
            if iter_max > 1:
                loglik1, u, imat = ties_terms(strata, design, best, ties)
                fac = numpy.triu(imat)
            flag = NOT_CONVERGED
    var = chinv2(fac)
    rescale = numpy.outer(scale, scale)
    coef = best * scale
    return finish_ties(coef, u / scale, var * rescale, loglik1, iterations, flag, (strata, design), (eps, toler_inf, iter_max))


def finish_ties(
    coef: FloatArray,
    u: FloatArray,
    var: FloatArray,
    loglik: float,
    iterations: int,
    flag: int,
    data: tuple[Strata, TiesDesign],
    tolerances: tuple[float, float, int],
) -> ConditionalFit:
    """Mark the aliased coefficients and collect the warnings of ``coxph.fit``."""
    strata, design = data
    eps, toler_inf, iter_max = tolerances
    nvar = coef.shape[0]
    singular = numpy.diag(var) == 0 if flag < nvar else numpy.zeros(nvar, dtype=numpy.bool_)
    messages: list[str] = []
    if iter_max > 1:
        with numpy.errstate(all='ignore'):
            infs = numpy.abs(u @ var)
        if flag == NOT_CONVERGED:
            messages.append('Ran out of iterations and did not converge')
            offset = float(numpy.sum(coef * design.means))
            lp = numpy.concatenate([(design.z1 / design.scale + design.means) @ coef - offset, [-offset]])
            if float(numpy.max(lp)) > 500 or not bool(numpy.all(numpy.isfinite(infs))):
                messages.append('one or more coefficients may be infinite')
        else:
            with numpy.errstate(all='ignore'):
                bad = ~numpy.isfinite(u) | ((infs > eps) & (infs > toler_inf * numpy.abs(coef)))
            if bool(numpy.any(bad)):
                which = ','.join(str(j + 1) for j in numpy.flatnonzero(bad))
                messages.append(f'Loglik converged before variable {which} ; coefficient may be infinite. ')
    coefficients = coef.copy()
    coefficients[singular] = numpy.nan
    keep = ~singular
    del strata
    return ConditionalFit(coefficients, var[numpy.ix_(keep, keep)], loglik, iterations, flag, tuple(messages))


def clogistic(
    strata: Strata,
    X: FloatArray,
    *,
    init: FloatArray | None = None,
    iter_max: int = EPI_ITER_MAX,
    eps: float = EPI_EPS,
    toler_chol: float = EPI_TOLER_CHOL,
) -> ConditionalFit:
    """``Epi::clogistic(event ~ group1 * X, strata = study)`` (``clogistic.R`` and ``clogit.c`` of Epi), without rescaling of the covariates.

    Raises:
        metafopy.errors.EstimationError: the information matrix is not positive definite or ``init`` has the wrong length.
        metafopy.errors.InputError: ``eps`` or ``toler_chol`` is not positive, or ``toler_chol`` exceeds ``eps``.
    """
    if iter_max < 0:
        raise metafopy.errors.InputError("'iter.max' must be non-negative")
    if eps <= 0:
        raise metafopy.errors.InputError("'eps' must be positive")
    if toler_chol <= 0:
        raise metafopy.errors.InputError("'toler.chol' must be positive")
    if eps < toler_chol:
        raise metafopy.errors.InputError("'toler.chol' must be smaller than 'eps'")
    nvar = X.shape[1]
    beta = start_values(init, nvar)
    loglik, u, info = conditional_terms(strata, X, beta)
    fac = info.copy()
    flag = 0
    old_beta = beta.copy()
    maxiter = iter_max
    if maxiter > 0:
        flag = cholesky2(fac, toler_chol)
        if flag > 0:
            chsolve2(fac, u)
            old_beta = beta.copy()
            beta = beta + u
        else:
            maxiter = 0
    iterations = 0
    halving = False
    for iterations in range(1, maxiter + 1):
        oldlik = loglik
        loglik, u, info = conditional_terms(strata, X, beta)
        fac = info.copy()
        if abs(1 - oldlik / loglik) <= eps and not halving:
            break
        if iterations == maxiter:
            flag = NOT_CONVERGED
            break
        if loglik < oldlik:
            halving = True
            beta = (beta + old_beta) / 2
        else:
            halving = False
            flag = cholesky2(fac, toler_chol)
            if flag > 0:
                chsolve2(fac, u)
                old_beta = beta.copy()
                beta = beta + u
            else:
                break
    if flag <= 0:
        raise metafopy.errors.EstimationError('Information matrix is not positive definite')
    fac = info.copy()
    cholesky2(fac, toler_chol)
    var = chinv2(fac)
    messages = ('Iteration limit exceeded',) if flag == NOT_CONVERGED else ()
    singular = numpy.diag(var) == 0 if flag < nvar else numpy.zeros(nvar, dtype=numpy.bool_)
    coef = beta.copy()
    coef[singular] = numpy.nan
    keep = ~singular
    return ConditionalFit(coef, var[numpy.ix_(keep, keep)], loglik, iterations, flag, messages)


def clean_keys(ctrl: collections.abc.Mapping[str, typing.Any], allowed: tuple[str, ...], what: str) -> dict[str, typing.Any]:
    """The entries of a control list with R's ``.`` in the names written as ``_``.

    Raises:
        metafopy.errors.InputError: an entry is not one of ``allowed``.
    """
    clean = {str(key).replace('.', '_'): value for key, value in ctrl.items()}
    unknown = sorted(set(clean) - set(allowed))
    if unknown:
        raise metafopy.errors.InputError(f'unknown {what} entries: {unknown}; use {list(allowed)}')
    return clean


def start_values(init: FloatArray | None, nvar: int) -> FloatArray:
    """The start values of the Newton iterations (zeros by default).

    Raises:
        metafopy.errors.EstimationError: the length of ``init`` is not ``nvar`` (R stops, and metafor reports that the model cannot be fitted).
    """
    if init is None:
        return numpy.zeros(nvar)
    values = numpy.atleast_1d(numpy.asarray(init, dtype=numpy.float64))
    if values.shape[0] != nvar:
        raise metafopy.errors.EstimationError('Wrong length for inital values')
    return values.copy()


# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
CLOGIT_KEYS = ('method', 'init', 'control')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
COXPH_CONTROL_KEYS = ('eps', 'toler_chol', 'iter_max', 'toler_inf', 'outer_max', 'timefix')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
TIES_METHODS = ('exact', 'approximate', 'efron', 'breslow')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
CLOGISTIC_KEYS = ('init', 'iter_max', 'eps', 'toler_chol')


def clogit_settings(ctrl: collections.abc.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Keyword arguments of :func:`clogit` from metafor's ``clogitCtrl`` list.

    The entries are those of ``clogit()``: ``method`` (``'exact'``, the default, ``'efron'``, ``'breslow'`` or ``'approximate'``
    for Breslow's approximation), ``init`` and ``control``, a mapping with the entries of ``survival::coxph.control()`` (``eps``,
    ``toler_chol``, ``iter_max``, ``toler_inf``; ``outer_max`` and ``timefix`` have no effect here).

    Raises:
        metafopy.errors.InputError: an entry is unknown or ``method`` is not one of the methods of ``clogit()``.
    """
    clean = clean_keys(ctrl, CLOGIT_KEYS, "'clogitCtrl'")
    method = clean.get('method', 'exact')
    if method not in TIES_METHODS:
        raise metafopy.errors.InputError(f'clogitCtrl: method must be one of {", ".join(repr(m) for m in TIES_METHODS)}, not {method!r}')
    control = clean_keys(clean.get('control') or {}, COXPH_CONTROL_KEYS, "'clogitCtrl' control")
    out: dict[str, typing.Any] = {key: control[key] for key in ('eps', 'toler_chol', 'iter_max', 'toler_inf') if key in control}
    out['ties'] = 'breslow' if method == 'approximate' else method
    out['init'] = None if clean.get('init') is None else numpy.atleast_1d(numpy.asarray(clean['init'], dtype=numpy.float64))
    return out


def clogit(strata: Strata, X: FloatArray, *, ties: Ties = 'exact', **settings: typing.Any) -> ConditionalFit:
    """``survival::clogit()`` of the stratified tables: exact via :func:`coxph_exact`, Efron and Breslow via :func:`coxph_ties`."""
    if ties == 'exact':
        return coxph_exact(strata, X, **settings)
    return coxph_ties(strata, X, ties=ties, **settings)


def clogistic_settings(ctrl: collections.abc.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Keyword arguments of :func:`clogistic` from metafor's ``clogisticCtrl`` list (``init``, ``iter_max``, ``eps``, ``toler_chol`` of ``Epi::clogistic()``).

    Raises:
        metafopy.errors.InputError: an entry is unknown.
    """
    clean = clean_keys(ctrl, CLOGISTIC_KEYS, "'clogisticCtrl'")
    out: dict[str, typing.Any] = {key: clean[key] for key in ('iter_max', 'eps', 'toler_chol') if key in clean}
    out['init'] = None if clean.get('init') is None else numpy.atleast_1d(numpy.asarray(clean['init'], dtype=numpy.float64))
    return out
