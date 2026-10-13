"""Log-likelihoods of the selection models (metafor's ``.selmodel.ll.cont``, ``.selmodel.ll.stepfun`` and ``.selmodel.ll.trunc``).

Under selection the density of an observed estimate is ``w(p_i) f(y_i) / A_i`` with ``f`` the normal density of the
random-effects model and ``A_i = integral of w(p(y)) f(y) dy`` the normalising constant. For the continuous selection
functions ``A_i`` is computed by numerical integration (:func:`metafopy.core.integrate.integrate`); for step functions and
the truncated models it has a closed form in terms of normal probabilities. See
https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.selmodel.r.
"""

import dataclasses
import math
import sys

import numpy
import numpy.typing
import scipy.special

import metafopy.core.integrate
import metafopy.errors
import metafopy.models.sel_spec

__all__ = ['Evaluation', 'Problem', 'evaluate', 'map_delta', 'map_inverse', 'negative_loglik']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type IntArray = numpy.typing.NDArray[numpy.intp]

EPS = sys.float_info.epsilon
INV_SQRT2 = 1 / math.sqrt(2)
SQRT_2PI = math.sqrt(2 * math.pi)
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
ALTERNATIVES = {'greater': 0, 'less': 1, 'two.sided': 2}


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """Everything the likelihood needs (the arguments of metafor's ``.selmodel.ll.*``; ``nan`` marks a free value in the ``*_arg`` fields)."""

    spec: metafopy.models.sel_spec.SelSpec
    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    preci: FloatArray
    subset: BoolArray
    pvals: FloatArray
    pgrp: IntArray
    delta_arg: FloatArray
    tau2_arg: float
    beta_arg: FloatArray
    mapfun: tuple[str | None, ...]
    delta_min: FloatArray
    delta_max: FloatArray
    decreasing: bool
    tau2_max: float
    alternative: str
    int_lower: float
    int_upper: float
    int_subdivisions: int
    int_rel_tol: float


@dataclasses.dataclass(frozen=True, slots=True)
class Evaluation:
    ll: float
    beta: FloatArray
    tau2: float
    delta: FloatArray


def plogis(x: float) -> float:
    return float(scipy.special.expit(x))


def qlogis(p: float) -> float:
    return float(scipy.special.logit(p))


def apply_named(fun: str, x: float) -> float:
    """The functions metafor lets ``mapfun``/``mapinvfun`` name: ``I``, ``exp``, ``log``, ``plogis`` and ``qlogis``.

    Raises:
        metafopy.errors.InputError: the function is not one of those.
    """
    if fun == 'I':
        return x
    if fun == 'exp':
        return math.exp(x) if x < 709.0 else math.inf
    if fun == 'log':
        return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)
    if fun == 'plogis':
        return plogis(x)
    if fun == 'qlogis':
        return qlogis(x)
    raise metafopy.errors.InputError(f'unknown mapping function {fun!r} (use one of I, exp, log, plogis, qlogis)')


def map_delta(x: float, lb: float, ub: float, fun: str | None) -> float:
    """metafor's ``.mapfun``: map the unconstrained value ``x`` to ``(lb, ub)`` (logistic by default) or by the named function, clipped to ``[lb, ub]``."""
    if fun is None:
        if lb == 0 and ub == 1:
            return plogis(x)
        return lb + (ub - lb) * plogis(x)
    return min(max(apply_named(fun, x), lb), ub)


def map_inverse(x: float, lb: float, ub: float, fun: str | None) -> float:
    """metafor's ``.mapinvfun``: the inverse of :func:`map_delta` (no clipping for a named function)."""
    if fun is None:
        if lb == 0 and ub == 1:
            return qlogis(x)
        return math.log((x - lb) / (ub - x))
    return apply_named(fun, x)


def exp_safe(x: float) -> float:
    return math.exp(x) if x < 709.0 else math.inf


def dnorm_log(y: FloatArray, mean: FloatArray, sd: FloatArray) -> FloatArray:
    z = (y - mean) / sd
    return numpy.asarray(-(LOG_SQRT_2PI + 0.5 * z * z + numpy.log(sd)))


def cont_integrand(
    y: float,
    yhat: float,
    sv: float,
    st: float,
    preci: float,
    spec: metafopy.models.sel_spec.SelSpec,
    delta: tuple[float, ...],
    steps: tuple[float, ...],
    alt: int,
    pval_min: float,
) -> float:
    """``w(p(y)) * dnorm(y, yhat, sqrt(vi + tau2))``: metafor's ``.selmodel.int`` for one study.

    Raises:
        metafopy.errors.InternalError: the type of selection model has no scalar selection function.
    """
    z = y / sv
    if alt == 0:
        p = 0.5 * math.erfc(z * INV_SQRT2)
    elif alt == 1:
        p = 0.5 * math.erfc(-z * INV_SQRT2)
    else:
        p = math.erfc(abs(z) * INV_SQRT2)
    if p < pval_min:
        p = pval_min
    elif p > 1 - pval_min:
        p = 1 - pval_min
    scalar = spec.scalar
    if scalar is None or spec.weight is None:
        raise metafopy.errors.InternalError('no scalar selection function')
    try:
        w = scalar(p, delta, preci, steps)
    except ArithmeticError:
        w = float(spec.weight(numpy.array([p]), numpy.array(delta), numpy.zeros(1), numpy.zeros(1), preci, 'greater', numpy.array(steps))[0])
    u = (y - yhat) / st
    return w * math.exp(-0.5 * u * u) / (st * SQRT_2PI)


def transform_delta(prob: Problem, delta: FloatArray, delta_transf: bool) -> FloatArray:
    spec = prob.spec
    if prob.decreasing and spec.kind == 'stepfun':
        if delta_transf:
            e = numpy.exp(delta)
            delta = numpy.exp(numpy.cumsum(numpy.concatenate([[0.0], -e[1:]])))
    elif delta_transf:
        delta = numpy.array(
            [map_delta(float(d), float(lo), float(hi), f) for d, lo, hi, f in zip(delta, prob.delta_min, prob.delta_max, prob.mapfun, strict=True)]
        )
    fixed = ~numpy.isnan(prob.delta_arg)
    delta = numpy.where(fixed, prob.delta_arg, delta)
    if prob.decreasing and spec.kind == 'stepfun' and numpy.any(fixed[1:]):
        delta = numpy.maximum.accumulate(delta[::-1])[::-1]
    return numpy.asarray(delta, dtype=numpy.float64)


def evaluate(prob: Problem, par: FloatArray, *, delta_transf: bool, tau2_transf: bool) -> Evaluation:
    """The log-likelihood at ``par`` = (beta, tau2, delta), the latter two on the transformed scale when requested.

    Raises:
        metafopy.errors.EstimationError: the normalising integral of a study cannot be computed.
    """
    spec = prob.spec
    pX = prob.X.shape[1]
    beta = numpy.where(numpy.isnan(prob.beta_arg), par[:pX], prob.beta_arg)
    tau2 = exp_safe(float(par[pX])) if tau2_transf else float(par[pX])
    if not math.isnan(prob.tau2_arg):
        tau2 = prob.tau2_arg
    if tau2 < EPS * 10:
        tau2 = 0.0
    if tau2 > prob.tau2_max:
        tau2 = prob.tau2_max
    delta = transform_delta(prob, numpy.asarray(par[pX + 1 : pX + 1 + spec.deltas], dtype=numpy.float64), delta_transf)
    yhat = prob.X @ beta
    if spec.kind == 'cont':
        ll = ll_cont(prob, yhat, tau2, delta)
    elif spec.kind == 'stepfun':
        ll = ll_stepfun(prob, yhat, tau2, delta)
    else:
        ll = ll_trunc(prob, yhat, tau2, delta)
    return Evaluation(ll=ll, beta=numpy.asarray(beta, dtype=numpy.float64), tau2=tau2, delta=delta)


def ll_cont(prob: Problem, yhat: FloatArray, tau2: float, delta: FloatArray) -> float:
    spec = prob.spec
    sub = prob.subset
    k = prob.yi.shape[0]
    alt = ALTERNATIVES[prob.alternative]
    delta_t = tuple(float(d) for d in delta)
    steps_t = tuple(float(s) for s in spec.steps)
    sv = numpy.sqrt(prob.vi).tolist()
    st = numpy.sqrt(prob.vi + tau2).tolist()
    yh = yhat.tolist()
    pc = prob.preci.tolist()
    Ai = numpy.full(k, numpy.nan)
    for i in numpy.flatnonzero(sub):
        try:
            Ai[i] = metafopy.core.integrate.integrate(
                cont_integrand,
                prob.int_lower,
                prob.int_upper,
                args=(yh[i], sv[i], st[i], pc[i], spec, delta_t, steps_t, alt, spec.pval_min),
                subdivisions=prob.int_subdivisions,
                rel_tol=prob.int_rel_tol,
            )
        except metafopy.errors.EstimationError as exc:
            raise metafopy.errors.EstimationError(f'Could not integrate over density in study {int(i) + 1}.') from exc
    sei = numpy.sqrt(prob.vi + tau2)
    ll0 = float(dnorm_log(prob.yi[~sub], yhat[~sub], sei[~sub]).sum())
    if spec.weight is None:
        raise metafopy.errors.InternalError('no selection function')
    with numpy.errstate(all='ignore'):
        w = spec.weight(prob.pvals[sub], delta, prob.yi[sub], prob.vi[sub], prob.preci[sub], prob.alternative, spec.steps)
        ll1 = float((numpy.log(w) + dnorm_log(prob.yi[sub], yhat[sub], sei[sub]) - numpy.log(Ai[sub])).sum())
    return ll0 + ll1


def tail_up(t: FloatArray, m: FloatArray, s: FloatArray) -> FloatArray:
    """``pnorm(t, m, s, lower.tail=FALSE)``."""
    return numpy.asarray(scipy.special.ndtr((m - t) / s))


def tail_low(t: FloatArray, m: FloatArray, s: FloatArray) -> FloatArray:
    """``pnorm(t, m, s)``."""
    return numpy.asarray(scipy.special.ndtr((t - m) / s))


def ll_stepfun(prob: Problem, yhat: FloatArray, tau2: float, delta: FloatArray) -> float:
    spec = prob.spec
    sub = prob.subset
    steps = spec.steps
    N = steps.shape[0]
    if N < 2:
        raise metafopy.errors.InputError("The 'steps' argument must contain at least one value below 1 for this type of selection model.")
    sd = numpy.sqrt(prob.vi[sub])
    sei = numpy.sqrt(prob.vi + tau2)
    m = yhat[sub]
    s = sei[sub]
    pr = prob.preci[sub]
    alt = prob.alternative

    def thr_up(step: float) -> FloatArray:
        return -sd * float(scipy.special.ndtri(step))

    def thr_low(step: float) -> FloatArray:
        return sd * float(scipy.special.ndtri(step))

    with numpy.errstate(all='ignore'):
        if alt == 'greater':
            A = tail_up(thr_up(steps[0]), m, s)
            for j in range(1, N):
                if j < N - 1:
                    A = A + delta[j] / pr * (tail_up(thr_up(steps[j]), m, s) - tail_up(thr_up(steps[j - 1]), m, s))
                else:
                    A = A + delta[j] / pr * tail_low(thr_up(steps[j - 1]), m, s)
        elif alt == 'less':
            A = tail_low(thr_low(steps[0]), m, s)
            for j in range(1, N):
                if j < N - 1:
                    A = A + delta[j] / pr * (tail_low(thr_low(steps[j]), m, s) - tail_low(thr_low(steps[j - 1]), m, s))
                else:
                    A = A + delta[j] / pr * tail_up(thr_low(steps[j - 1]), m, s)
        else:
            A = tail_up(thr_up(steps[0] / 2), m, s) + tail_low(thr_low(steps[0] / 2), m, s)
            for j in range(1, N):
                if j < N - 1:
                    A = A + delta[j] / pr * (
                        (tail_up(thr_up(steps[j] / 2), m, s) - tail_up(thr_up(steps[j - 1] / 2), m, s))
                        + (tail_low(thr_low(steps[j] / 2), m, s) - tail_low(thr_low(steps[j - 1] / 2), m, s))
                    )
                else:
                    A = A + delta[j] / pr * (tail_low(thr_up(steps[j - 1] / 2), m, s) - tail_low(thr_low(steps[j - 1] / 2), m, s))
        ll0 = float(dnorm_log(prob.yi[~sub], yhat[~sub], sei[~sub]).sum())
        ll1 = float((numpy.log(delta[prob.pgrp[sub]] / pr) + dnorm_log(prob.yi[sub], m, s) - numpy.log(A)).sum())
    return ll0 + ll1


def ll_trunc(prob: Problem, yhat: FloatArray, tau2: float, delta: FloatArray) -> float:
    spec = prob.spec
    sub = prob.subset
    sei = numpy.sqrt(prob.vi + tau2)
    cut = float(delta[1]) if math.isnan(float(spec.steps[0])) else float(spec.steps[0])
    ll0 = dnorm_log(prob.yi[~sub], yhat[~sub], sei[~sub])
    y, m, s = prob.yi[sub], yhat[sub], sei[sub]
    with numpy.errstate(all='ignore'):
        if prob.alternative == 'greater':
            sel = numpy.where(y > cut, 0.0, math.log(delta[0]) if delta[0] > 0 else -math.inf)
            ll1 = sel + dnorm_log(y, m, s) - numpy.log(1 - (1 - delta[0]) * tail_low(numpy.full_like(m, cut), m, s))
        else:
            sel = numpy.where(y < cut, 0.0, math.log(delta[0]) if delta[0] > 0 else -math.inf)
            ll1 = sel + dnorm_log(y, m, s) - numpy.log(1 - (1 - delta[0]) * tail_up(numpy.full_like(m, cut), m, s))
    return float(ll0.sum() + ll1.sum())


def negative_loglik(prob: Problem, par: FloatArray, *, delta_transf: bool, tau2_transf: bool) -> float:
    """``-1 * ll``, the objective that is minimised and whose Hessian gives the covariance matrix."""
    return -evaluate(prob, par, delta_transf=delta_transf, tau2_transf=tau2_transf).ll
