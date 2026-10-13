"""Definitions of the selection models of :func:`metafopy.selmodel`.

Every ``type`` fixes the number of selection parameters ``delta``, their parameter space, the starting values, the
selection function ``w(p)`` and which likelihood is maximised. These are the blocks of metafor's
``selmodel.rma.uni()`` that set ``deltas``, ``delta.lb``, ``wi.fun`` and so on, see
https://github.com/wviechtb/metafor/blob/master/R/selmodel.rma.uni.r and
https://wviechtb.github.io/metafor/reference/selmodel.html.

Each continuous selection function exists twice: a vectorised one (``weight``) used wherever many p-values are
evaluated at once, and a scalar one (``scalar``) that avoids NumPy's per-call overhead inside the numerical
integration, which evaluates the function many thousands of times.
"""

import collections.abc
import dataclasses
import math
import re
import typing
import warnings

import numpy
import numpy.typing

import metafopy.errors

# auditme: accept[non-alphabetical-literal] constants first, as ruff's RUF022 orders __all__
__all__ = ['STEPWISE', 'TRUNCATED', 'TYPE_OPTIONS', 'SelSpec', 'build_spec', 'match_type']

STEPWISE = ('stepcon', 'stepfun')
TRUNCATED = ('trunc', 'truncest')
EXP = 'exp'
LOG = 'log'

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Weight = collections.abc.Callable[[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray | float, str, FloatArray], FloatArray]
type Scalar = collections.abc.Callable[[float, tuple[float, ...], float, tuple[float, ...]], float]
type Kind = typing.Literal['cont', 'stepfun', 'trunc']

# auditme: accept[non-alphabetical-literal] the order is metafor's, and partial matching takes the first hit
TYPE_OPTIONS = (
    'beta',
    'halfnorm',
    'negexp',
    'logistic',
    'power',
    'negexppow',
    'halfnorm1',
    'negexp1',
    'logistic1',
    'power1',
    'halfnorm2',
    'negexp2',
    'logistic2',
    'power2',
    'stepfun',
    'stepcon',
    'trunc',
    'truncest',
)

EPS = float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class SelSpec:
    """Parameter space and selection function of one selection model type (names follow metafor with ``.`` replaced by ``_``)."""

    type: str
    kind: Kind
    deltas: int
    transf_fun: tuple[str, ...]
    transf_inv: tuple[str, ...]
    delta_lb: FloatArray
    delta_ub: FloatArray
    delta_lb_excl: tuple[bool, ...]
    delta_ub_excl: tuple[bool, ...]
    delta_init: FloatArray
    delta_min: FloatArray
    delta_max: FloatArray
    delta_max_check: bool
    h0_delta: FloatArray
    delta_lrt: tuple[bool, ...]
    pval_min: float
    steps: FloatArray
    stepsspec: bool
    preci: FloatArray
    defmap: bool
    weight: Weight | None
    scalar: Scalar | None


def match_type(type_: str) -> str:
    """metafor's ``type.options[grep(type, type.options)[1]]``: the first option the (regular expression) ``type`` matches.

    Raises:
        metafopy.errors.InputError: no option matches.
    """
    for option in TYPE_OPTIONS:
        if re.search(type_, option):
            return option
    raise metafopy.errors.InputError("Unknown 'type' specified (see 'help(selmodel)' for options).")


# -- vectorised selection functions: wi.fun(x, delta, yi, vi, preci, alternative, steps) ----------------------


def w_beta(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    xt = numpy.minimum(numpy.maximum(s[0], x), s[1])
    return numpy.asarray(xt ** (d[0] - 1) * (1 - xt) ** (d[1] - 1), dtype=numpy.float64)


def w_halfnorm(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, numpy.exp(-d[0] * preci * x**2) / numpy.exp(-d[0] * preci * s[0] ** 2))


def w_negexp(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, numpy.exp(-d[0] * preci * x) / numpy.exp(-d[0] * preci * s[0]))


def logistic_kernel(d: float | FloatArray, preci: FloatArray | float, x: FloatArray | float) -> FloatArray:
    e = numpy.exp(-d * preci * x)
    return numpy.asarray(2 * e / (1 + e))


def w_logistic(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, logistic_kernel(d[0], preci, x) / logistic_kernel(d[0], preci, s[0]))


def w_power(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (1 - x) ** (preci * d[0]) / (1 - s[0]) ** (preci * d[0]))


def w_negexppow(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, numpy.exp(-d[0] * preci * x ** (1 / d[1])) / numpy.exp(-d[0] * preci * s[0] ** (1 / d[1])))


def rescaled(x: FloatArray, s: FloatArray) -> FloatArray:
    return numpy.asarray((x - s[0]) / (1 - s[0]))


def w_halfnorm1(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, numpy.exp(-d[0] * preci * rescaled(x, s) ** 2))


def w_negexp1(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, numpy.exp(-d[0] * preci * rescaled(x, s)))


def w_logistic1(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, logistic_kernel(d[0], preci, rescaled(x, s)))


def w_power1(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (1 - rescaled(x, s)) ** (preci * d[0]))


def w_halfnorm2(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (d[0] + numpy.exp(-d[1] * preci * x**2) / numpy.exp(-d[1] * preci * s[0] ** 2)) / (1 + d[0]))


def w_negexp2(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (d[0] + numpy.exp(-d[1] * preci * x) / numpy.exp(-d[1] * preci * s[0])) / (1 + d[0]))


def w_logistic2(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (d[0] + logistic_kernel(d[1], preci, x) / logistic_kernel(d[1], preci, s[0])) / (1 + d[0]))


def w_power2(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    return numpy.where(x <= s[0], 1.0, (d[0] + (1 - x) ** (preci * d[1]) / (1 - s[0]) ** (preci * d[1])) / (1 + d[0]))


def w_step(x: FloatArray, d: FloatArray, preci: FloatArray | float, s: FloatArray) -> FloatArray:
    idx = numpy.searchsorted(s, x, side='left')
    ok = (idx < s.shape[0]) & ~numpy.isnan(x)
    return numpy.where(ok, d[numpy.minimum(idx, d.shape[0] - 1)] / preci, numpy.nan)


VECTOR: dict[str, collections.abc.Callable[[FloatArray, FloatArray, FloatArray | float, FloatArray], FloatArray]] = {
    'beta': w_beta,
    'halfnorm': w_halfnorm,
    'halfnorm1': w_halfnorm1,
    'halfnorm2': w_halfnorm2,
    'logistic': w_logistic,
    'logistic1': w_logistic1,
    'logistic2': w_logistic2,
    'negexp': w_negexp,
    'negexp1': w_negexp1,
    'negexp2': w_negexp2,
    'negexppow': w_negexppow,
    'power': w_power,
    'power1': w_power1,
    'power2': w_power2,
    'stepcon': w_step,
    'stepfun': w_step,
}


# -- scalar selection functions ---------------------------------------------------------------------------


def s_beta(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    xt = min(max(s[0], p), s[1])
    return float(xt ** (d[0] - 1) * (1 - xt) ** (d[1] - 1))


def s_halfnorm(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else math.exp(-d[0] * preci * p * p) / math.exp(-d[0] * preci * s[0] * s[0])


def s_negexp(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else math.exp(-d[0] * preci * p) / math.exp(-d[0] * preci * s[0])


def s_logistic_kernel(d: float, preci: float, x: float) -> float:
    e = math.exp(-d * preci * x)
    return 2 * e / (1 + e)


def s_logistic(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else s_logistic_kernel(d[0], preci, p) / s_logistic_kernel(d[0], preci, s[0])


def s_power(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (1 - p) ** (preci * d[0]) / (1 - s[0]) ** (preci * d[0])


def s_negexppow(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else math.exp(-d[0] * preci * p ** (1 / d[1])) / math.exp(-d[0] * preci * s[0] ** (1 / d[1]))


def s_halfnorm1(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else math.exp(-d[0] * preci * ((p - s[0]) / (1 - s[0])) ** 2)


def s_negexp1(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else math.exp(-d[0] * preci * ((p - s[0]) / (1 - s[0])))


def s_logistic1(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else s_logistic_kernel(d[0], preci, (p - s[0]) / (1 - s[0]))


def s_power1(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (1 - (p - s[0]) / (1 - s[0])) ** (preci * d[0])


def s_halfnorm2(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (d[0] + math.exp(-d[1] * preci * p * p) / math.exp(-d[1] * preci * s[0] * s[0])) / (1 + d[0])


def s_negexp2(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (d[0] + math.exp(-d[1] * preci * p) / math.exp(-d[1] * preci * s[0])) / (1 + d[0])


def s_logistic2(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (d[0] + s_logistic_kernel(d[1], preci, p) / s_logistic_kernel(d[1], preci, s[0])) / (1 + d[0])


def s_power2(p: float, d: tuple[float, ...], preci: float, s: tuple[float, ...]) -> float:
    return 1.0 if p <= s[0] else (d[0] + (1 - p) ** (preci * d[1]) / (1 - s[0]) ** (preci * d[1])) / (1 + d[0])


SCALAR: dict[str, Scalar] = {
    'beta': s_beta,
    'halfnorm': s_halfnorm,
    'halfnorm1': s_halfnorm1,
    'halfnorm2': s_halfnorm2,
    'logistic': s_logistic,
    'logistic1': s_logistic1,
    'logistic2': s_logistic2,
    'negexp': s_negexp,
    'negexp1': s_negexp1,
    'negexp2': s_negexp2,
    'negexppow': s_negexppow,
    'power': s_power,
    'power1': s_power1,
    'power2': s_power2,
}


def make_weight(name: str) -> Weight:
    fn = VECTOR[name]

    def weight(x: FloatArray, delta: FloatArray, yi: FloatArray, vi: FloatArray, preci: FloatArray | float, alternative: str, steps: FloatArray) -> FloatArray:
        with numpy.errstate(all='ignore'):
            return numpy.asarray(fn(numpy.asarray(x, dtype=numpy.float64), numpy.asarray(delta, dtype=numpy.float64), preci, numpy.asarray(steps)))

    return weight


def vec(*values: float) -> FloatArray:
    return numpy.array(values, dtype=numpy.float64)


def build_spec(
    type_: str,
    *,
    alternative: str,
    steps: FloatArray,
    stepsspec: bool,
    precspec: bool,
    decreasing: bool,
    preci: FloatArray,
    yi: FloatArray,
    defmap: bool,
) -> SelSpec:
    """The parameter space and selection function of selection model ``type_`` (``type_`` already matched by :func:`match_type`).

    Raises:
        metafopy.errors.InputError: ``steps`` does not suit the type of selection model.
    """
    k = yi.shape[0]
    kind: Kind = 'cont'
    pval_min = 0.0
    delta_max_check = True
    deltas = 1
    transf: tuple[str, ...] = (EXP,)
    inverse: tuple[str, ...] = (LOG,)
    lb = vec(0.0)
    ub = vec(math.inf)
    lb_excl: tuple[bool, ...] = (False,)
    ub_excl: tuple[bool, ...] = (False,)
    init = vec(1.0)
    dmin = vec(0.0)
    dmax = vec(100.0)
    h0 = vec(0.0)
    lrt: tuple[bool, ...] = (True,)

    if type_ == 'beta':
        if stepsspec:
            if steps.shape[0] != 2:
                raise metafopy.errors.InputError("The 'steps' argument for this type of selection model should be of length 2.")
        else:
            steps = vec(0.0, 1.0)
        if precspec:
            warnings.warn("Argument 'prec' ignored (not applicable to this type of selection model).", stacklevel=3)
        deltas = 2
        transf, inverse = (EXP,) * 2, (LOG,) * 2
        lb, ub = vec(0.0, 0.0), vec(math.inf, math.inf)
        lb_excl, ub_excl = (True, True), (False, False)
        init, dmin, dmax, h0 = vec(1.0, 1.0), vec(1e-05, 1e-05), vec(100.0, 100.0), vec(1.0, 1.0)
        lrt = (True, True)
        pval_min = 1e-5

    elif type_ in ('halfnorm', 'negexp', 'logistic', 'power') or type_ in ('halfnorm1', 'negexp1', 'logistic1', 'power1'):
        if stepsspec:
            if steps.shape[0] != 2:
                raise metafopy.errors.InputError("Can only specify a single value for the 'steps' argument for this type of selection model.")
        else:
            steps = vec(0.0)
        pval_min = 1e-5 if type_ == 'power' else 0.0

    elif type_ in ('negexppow', 'halfnorm2', 'negexp2', 'logistic2', 'power2'):
        if stepsspec:
            if steps.shape[0] != 2:
                raise metafopy.errors.InputError("Can only specify a single value for the 'steps' argument for this type of selection model.")
        else:
            steps = vec(0.0)
        deltas = 2
        transf, inverse = (EXP,) * 2, (LOG,) * 2
        lb, ub = vec(0.0, 0.0), vec(math.inf, math.inf)
        lb_excl, ub_excl = (False, False), (False, False)
        init = vec(1.0, 1.0) if type_ == 'negexppow' else vec(1.0, 0.25)
        dmin, dmax, h0 = vec(0.0, 0.0), vec(100.0, 100.0), vec(0.0, 0.0)
        lrt = (True, True)

    elif type_ in STEPWISE:
        kind = 'stepfun'
        if not stepsspec:
            raise metafopy.errors.InputError("Must specify the 'steps' argument for this type of selection model.")
        if precspec:
            if type_ == 'stepcon' or decreasing:
                warnings.warn("Argument 'prec' ignored (not applicable to this type of selection model).", stacklevel=3)
                preci = numpy.ones(k)
            else:
                warnings.warn('Adding a precision measure to this selection model is undocumented and experimental.', stacklevel=3)
        deltas = steps.shape[0]
        lb_excl, ub_excl = (False,) * deltas, (False,) * deltas
        h0 = numpy.ones(deltas)
        lrt = (True,) * deltas
        if type_ == 'stepcon':
            transf, inverse = ('plogis',) * deltas, ('qlogis',) * deltas
            lb, ub = numpy.zeros(deltas), numpy.ones(deltas)
            init = numpy.linspace(1, 0.5, deltas)
            dmin, dmax = numpy.zeros(deltas), numpy.ones(deltas)
            delta_max_check = False
        elif decreasing:
            transf, inverse = ('I',) * deltas, ('I',) * deltas
            defmap = True
            lb = numpy.concatenate([[0.0], numpy.full(deltas - 1, -math.inf)])
            ub = numpy.concatenate([[1.0], numpy.full(deltas - 1, math.inf)])
            init = numpy.concatenate([[1.0], numpy.full(deltas - 1, -2.0)])
            dmin, dmax = numpy.zeros(deltas), numpy.ones(deltas)
            delta_max_check = False
        else:
            transf, inverse = (EXP,) * deltas, (LOG,) * deltas
            lb, ub = numpy.zeros(deltas), numpy.full(deltas, math.inf)
            init = numpy.linspace(1, 0.8, deltas)
            dmin, dmax = numpy.zeros(deltas), numpy.full(deltas, 100.0)

    elif type_ == 'trunc':
        kind = 'trunc'
        if steps.shape[0] != 1:
            raise metafopy.errors.InputError("Can only specify a single value for the 'steps' argument for this type of selection model.")
        if precspec:
            warnings.warn("Argument 'prec' ignored (not applicable to this type of selection model).", stacklevel=3)
        h0 = vec(1.0)

    elif type_ == 'truncest':
        kind = 'trunc'
        if stepsspec:
            warnings.warn("Argument 'steps' ignored (not applicable to this type of selection model).", stacklevel=3)
        stepsspec = False
        steps = vec(math.nan)
        if precspec:
            warnings.warn("Argument 'prec' ignored (not applicable to this type of selection model).", stacklevel=3)
        sd = float(numpy.std(yi, ddof=1)) if k > 1 else math.nan
        deltas = 2
        transf, inverse = (EXP, 'I'), (LOG, 'I')
        lb, ub = vec(0.0, -math.inf), vec(math.inf, math.inf)
        lb_excl, ub_excl = (False, False), (False, False)
        init = vec(1.0, float(numpy.mean(yi)))
        dmin = vec(0.0, float(yi.min()) - sd if alternative == 'greater' else float(yi.min()))
        dmax = vec(100.0, float(yi.max()) if alternative == 'greater' else float(yi.max()) + sd)
        h0 = vec(1.0, 0.0)
        lrt = (True, False)

    weight: Weight | None = None
    scalar: Scalar | None = None
    if type_ in VECTOR:
        weight = make_weight(type_)
        scalar = SCALAR.get(type_)

    return SelSpec(
        type=type_,
        kind=kind,
        deltas=deltas,
        transf_fun=transf,
        transf_inv=inverse,
        delta_lb=lb,
        delta_ub=ub,
        delta_lb_excl=lb_excl,
        delta_ub_excl=ub_excl,
        delta_init=init,
        delta_min=dmin,
        delta_max=dmax,
        delta_max_check=delta_max_check,
        h0_delta=h0,
        delta_lrt=lrt,
        pval_min=pval_min,
        steps=steps,
        stepsspec=stepsspec,
        preci=preci,
        defmap=defmap,
        weight=weight,
        scalar=scalar,
    )
