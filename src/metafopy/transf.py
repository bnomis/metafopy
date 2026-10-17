"""Transformation functions (metafor's ``transf.*``).

A port of ``R/transf.r`` from metafor 5.2-1. metafor's ``transf.rtoz`` is
``transf_rtoz`` here, ``transf.ipft.hm`` is ``transf_ipft_hm``, and so on
(``.`` becomes ``_``); argument names and defaults are metafor's. The functions
are documented on metafor's help page, ``?transf``
(https://wviechtb.github.io/metafor/reference/transf.html). Sources cited there:

- Fisher (1921), r-to-z, https://hdl.handle.net/2440/15169;
- Freeman & Tukey (1950), double arcsine and incidence-rate transformations,
  doi:10.1214/aoms/1177729756, inverted as in Miller (1978),
  doi:10.1080/00031305.1978.10479283 (``transf_ipft_hm`` uses the harmonic
  mean of the sample sizes; see Barendregt et al. 2013,
  doi:10.1136/jech-2013-203104, on back-transforming pooled proportions);
- Hakstian & Whalen (1976), doi:10.1007/BF02291840, and Bonett (2002),
  doi:10.3102/10769986027004335, for coefficient alpha;
- Olkin & Finn (1995), doi:10.1037/0033-2909.118.1.155, for R-squared;
- Cohen (1988) U1-U3, Inman & Bradley (1989), doi:10.1080/03610928908830127,
  and Grice & Barrett (2014), doi:10.2466/03.pr0.115c29z4, for overlap;
  McGraw & Wong (1992), doi:10.1037/0033-2909.111.2.361, for the CLES;
  Rosenthal & Rubin (1982), doi:10.1037/0022-0663.74.2.166, for the BESD;
- Jacobs & Viechtbauer (2017), doi:10.1002/jrsm.1218, for (point-)biserial
  correlations;
- Cox & Snell (1989) and Chinn (2000),
  doi:10.1002/1097-0258(20001130)19:22<3127::aid-sim784>3.0.co;2-m, for log
  odds ratios to SMDs; Pearson (1900), doi:10.1098/rsta.1900.0022, and Digby
  (1983), doi:10.2307/2531104, for tetrachoric approximations.

Every function accepts array-likes or scalars (NaN or ``None`` for R's NA) and
returns a float ndarray that is at least 1-d (R has no scalars), with the
broadcast shape of the inputs.

The ``*_int`` functions integrate with SciPy's QUADPACK ``qags``/``qagi``,
the routines behind R's ``integrate()``, using R's defaults (``rel.tol =
abs.tol = .Machine$double.eps^0.25``, 100 subdivisions); where R's
``integrate()`` stops with an error, metafor returns NA and so do these.
``transf_ilogit_mode`` uses the line-for-line port of R's ``uniroot()`` and
the other ``*_mode`` functions the port of R's ``optimize()``.

Behaviour kept from R, including its quirks:

- ``transf_iarcsin_int``, ``transf_iahw_int`` and ``transf_iabt_int`` bound
  the integral by ``max(0, xi - 10*tau)`` (and ``min(...)``) taken over *all*
  ``xi``, so a vector call can differ from elementwise calls, and one NaN in
  ``xi`` makes every value NaN;
- ``transf_ztor_mode(NaN)`` and similar return a number, as R's
  ``optimize()`` replaces non-finite values by the largest double.

Differences from R:

- with ``tau2 = 0`` the ``*_mode`` functions return the plain back-transform
  elementwise (in R a vector ``xi`` then yields every value ``length(xi)``
  times);
- ``targs`` is a mapping with metafor's list names (``tau2``, ``lower``,
  ``upper``, ``ni``, ``sd``) or, as in R, a scalar or a positional vector;
  ``transf_ipft_hm`` also accepts the sample sizes directly (R errors);
- R's NA and NaN are both NaN.
"""

import collections.abc
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import scipy.integrate
import scipy.special

import metafopy.core.optim
import metafopy.core.zeroin
import metafopy.errors

__all__ = [
    'transf_abt',
    'transf_ahw',
    'transf_arcsin',
    'transf_clestod',
    'transf_dtobesd',
    'transf_dtocles',
    'transf_dtocles_int',
    'transf_dtocliffd',
    'transf_dtolnor_logis',
    'transf_dtolnor_norm',
    'transf_dtomd',
    'transf_dtoovl',
    'transf_dtorbis',
    'transf_dtorpb',
    'transf_dtou1',
    'transf_dtou2',
    'transf_dtou3',
    'transf_exp_int',
    'transf_exp_mode',
    'transf_iabt',
    'transf_iabt_int',
    'transf_iabt_mode',
    'transf_iahw',
    'transf_iahw_int',
    'transf_iahw_mode',
    'transf_iarcsin',
    'transf_iarcsin_int',
    'transf_iarcsin_mode',
    'transf_iirft',
    'transf_ilogit',
    'transf_ilogit_int',
    'transf_ilogit_mode',
    'transf_ipft',
    'transf_ipft_hm',
    'transf_iprobit',
    'transf_iprobit_int',
    'transf_iprobit_mode',
    'transf_irft',
    'transf_isqrt',
    'transf_lnortod_logis',
    'transf_lnortod_norm',
    'transf_lnortord',
    'transf_lnortorr',
    'transf_lnortortet_digby',
    'transf_lnortortet_pearson',
    'transf_logit',
    'transf_pft',
    'transf_probit',
    'transf_r2toz',
    'transf_rpbtod',
    'transf_rpbtorbis',
    'transf_rtod',
    'transf_rtorpb',
    'transf_rtoz',
    'transf_ztor',
    'transf_ztor2',
    'transf_ztor_int',
    'transf_ztor_mode',
]

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Targs = numpy.typing.ArrayLike | collections.abc.Mapping[str, typing.Any] | None

EPSILON = sys.float_info.epsilon
DBL_MAX = sys.float_info.max
TOL = EPSILON**0.25  # R's integrate() rel.tol and abs.tol
M_1_SQRT_2PI = 0.398942280401432677939946059934
HALF_PI = math.pi / 2
# x beyond which R's dnorm() returns 0: sqrt(-2*M_LN2*(DBL_MIN_EXP + 1 - DBL_MANT_DIG))
DNORM_UFLOW = math.sqrt(-2 * math.log(2) * (-1021 + 1 - 53))


# --------------------------------------------------------------------------
# helpers: R's distribution functions and metafor's argument checks


def arr(x: numpy.typing.ArrayLike) -> FloatArray:
    return numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64))


def dnorm(x: numpy.typing.ArrayLike, mean: numpy.typing.ArrayLike = 0.0, sd: numpy.typing.ArrayLike = 1.0) -> FloatArray:
    """R's ``dnorm()`` (``dnorm4`` in ``nmath/dnorm.c``), including its accurate tail branch."""
    xb, mu, sigma = numpy.broadcast_arrays(numpy.asarray(x, numpy.float64), numpy.asarray(mean, numpy.float64), numpy.asarray(sd, numpy.float64))
    with numpy.errstate(all='ignore'):
        z = numpy.abs((xb - mu) / sigma)
        x1 = numpy.ldexp(numpy.rint(numpy.ldexp(z, 16)), -16)
        x2 = z - x1
        body = numpy.where(
            z < 5,
            M_1_SQRT_2PI * numpy.exp(-0.5 * z * z) / sigma,
            numpy.where(z > DNORM_UFLOW, 0.0, M_1_SQRT_2PI / sigma * (numpy.exp(-0.5 * x1 * x1) * numpy.exp((-0.5 * x2 - x1) * x2))),
        )
        out: FloatArray = numpy.select(
            [
                numpy.isnan(xb) | numpy.isnan(mu) | numpy.isnan(sigma),
                sigma < 0,
                ~numpy.isfinite(sigma),
                ~numpy.isfinite(xb) & (mu == xb),
                sigma == 0,
                ~numpy.isfinite(z) | (z >= 2 * math.sqrt(DBL_MAX)),
            ],
            [xb + mu + sigma, numpy.nan, 0.0, numpy.nan, numpy.where(xb == mu, numpy.inf, 0.0), 0.0],
            body,
        )
    return out


def d1(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    return float(dnorm(x, mean, sd))


def pnorm(x: numpy.typing.ArrayLike) -> FloatArray:
    return numpy.asarray(scipy.special.ndtr(numpy.asarray(x, numpy.float64)), numpy.float64)


def p1(x: float) -> float:
    return float(scipy.special.ndtr(x))


def qnorm(p: numpy.typing.ArrayLike) -> FloatArray:
    return numpy.asarray(scipy.special.ndtri(numpy.asarray(p, numpy.float64)), numpy.float64)


def plogis(x: numpy.typing.ArrayLike) -> FloatArray:
    with numpy.errstate(over='ignore'):
        return numpy.asarray(1 / (1 + numpy.exp(-numpy.asarray(x, numpy.float64))), numpy.float64)


def qlogis(p: numpy.typing.ArrayLike) -> FloatArray:
    pa = numpy.asarray(p, numpy.float64)
    with numpy.errstate(all='ignore'):
        out = numpy.log(pa / (1 - pa))
    return numpy.asarray(numpy.where((pa < 0) | (pa > 1), numpy.nan, out), numpy.float64)


def chktargsint(targs: Targs) -> dict[str, float]:
    """metafor's ``.chktargsint``: ``tau2`` (required, >= 0) and optional ``lower``/``upper`` bounds.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(targs, collections.abc.Mapping):
        if len(targs) > 3:
            raise metafopy.errors.InputError("Length of the 'targs' argument must be <= 3.")
        items = {k: v for k, v in targs.items() if v is not None}
    elif targs is None:
        items = {}
    else:
        flat = numpy.atleast_1d(numpy.asarray(targs, dtype=numpy.float64)).ravel()
        if flat.size > 3:
            raise metafopy.errors.InputError("Length of the 'targs' argument must be <= 3.")
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        items = dict(zip(('tau2', 'lower', 'upper'), flat, strict=False))
    if any(numpy.size(v) > 1 for v in items.values()):
        raise metafopy.errors.InputError("Elements of the 'targs' arguments must be scalars.")
    out: dict[str, float] = {
        k: float(numpy.asarray(items[k], dtype=numpy.float64).reshape(-1)[0])
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        for k in ('tau2', 'lower', 'upper')
        if k in items and numpy.size(items[k]) == 1
    }
    if 'tau2' not in out:
        raise metafopy.errors.InputError("Must specify a 'tau2' value via the 'targs' argument.")
    if not out['tau2'] >= 0:
        raise metafopy.errors.InputError("Value of 'tau2' must be >= 0 to use an integral transformation.")
    return out


def mode_tau2(targs: Targs) -> float:
    """The ``tau2`` checks shared by the ``transf.*.mode`` functions; values below machine epsilon become 0.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if targs is None or (isinstance(targs, collections.abc.Mapping) and targs.get('tau2') is None):
        raise metafopy.errors.InputError("Must specify a 'tau2' value via the 'targs' argument.")
    tau2 = numpy.atleast_1d(numpy.asarray(targs['tau2'] if isinstance(targs, collections.abc.Mapping) else targs, dtype=numpy.float64)).ravel()
    if tau2.size > 1:
        raise metafopy.errors.InputError("Value of 'tau2' argument must be a scalar.")
    t2 = float(tau2[0])
    if not t2 >= 0:
        raise metafopy.errors.InputError("Value of 'tau2' must be >= 0 to use this transformation function.")
    return 0.0 if t2 < EPSILON else t2


class NonFiniteError(Exception):
    pass


def integrate_function(f: collections.abc.Callable[[float], float], lower: float, upper: float) -> float:
    """``integrate(f, lower, upper)$value`` as metafor calls it, with NaN where R raises an error."""
    if math.isnan(lower) or math.isnan(upper):
        return math.nan

    def g(z: float) -> float:
        try:
            val = f(z)
        except ArithmeticError, ValueError:  # x/0 and domain errors, which are Inf/NaN in R
            raise NonFiniteError from None
        if not math.isfinite(val):  # R: "non-finite function value"
            raise NonFiniteError
        return val

    # R uses dqags for finite limits, else dqagi on [bound, Inf), (-Inf, bound] or the whole line
    if math.isfinite(lower) and math.isfinite(upper):
        a, b = lower, upper
    elif math.isfinite(lower):
        a, b = lower, math.inf
    elif math.isfinite(upper):
        a, b = -math.inf, upper
    else:
        a, b = -math.inf, math.inf
    try:
        res: tuple[typing.Any, ...] = scipy.integrate.quad(g, a, b, epsabs=TOL, epsrel=TOL, limit=100, full_output=1)
    except NonFiniteError:
        return math.nan
    if len(res) > 3:  # ier > 0: R stops ("roundoff error", "maximum number of subdivisions", ...)
        return math.nan
    return float(res[0])


def mapint(toint: collections.abc.Callable[[float, float], float], xi: FloatArray, lower: numpy.typing.ArrayLike, upper: numpy.typing.ArrayLike) -> FloatArray:
    """``mapply(xi, FUN=cfunc, lower=..., upper=...)``: integrate ``toint(z, xi)`` for each ``xi``."""
    xb, lb, ub = numpy.broadcast_arrays(xi, numpy.asarray(lower, numpy.float64), numpy.asarray(upper, numpy.float64))
    out = numpy.empty(xb.shape, dtype=numpy.float64)
    for idx in numpy.ndindex(xb.shape):
        x = float(xb[idx])

        def f(z: float, x: float = x) -> float:
            return toint(z, x)

        out[idx] = integrate_function(f, float(lb[idx]), float(ub[idx]))
    return out


def mapmode(dfun: collections.abc.Callable[[float, float], float], xi: FloatArray, lower: float, upper: float) -> FloatArray:
    """``sapply(xi, optimize(dfun, maximum=TRUE, lower, upper, mu=x)$maximum)``."""
    out = numpy.empty(xi.shape, dtype=numpy.float64)
    for idx in numpy.ndindex(xi.shape):
        mu = float(xi[idx])

        def negf(x: float, mu: float = mu) -> float:
            try:
                return -dfun(x, mu)
            except ArithmeticError, ValueError:  # Inf/NaN in R, which optimize() replaces by DBL_MAX
                return math.nan

        out[idx] = metafopy.core.optim.fmin(negf, lower, upper)
    return out


def lengths(xi: FloatArray, n1i: numpy.typing.ArrayLike, n2i: numpy.typing.ArrayLike, msg: str) -> tuple[FloatArray, FloatArray]:
    n1, n2 = arr(n1i), arr(n2i)
    if n1.size != n2.size:
        raise metafopy.errors.InputError("Length of 'n1i' does not match the length of 'n2i'.")
    if n1.size != xi.size:
        raise metafopy.errors.InputError(msg)
    return n1, n2


def expand_p(xi: FloatArray, p: numpy.typing.ArrayLike, name: str) -> FloatArray:
    pa = arr(p)
    if pa.size == 1:
        pa = numpy.full(xi.shape, pa.reshape(-1)[0])
    if pa.size != xi.size:
        raise metafopy.errors.InputError(f"Length of 'xi' does not match the length of '{name}'.")
    return pa


# --------------------------------------------------------------------------
# correlations


def transf_rtoz(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Fisher's r-to-z transformation, ``atanh(xi)``, with ``xi`` truncated to [-1, 1]."""
    with numpy.errstate(divide='ignore'):
        return numpy.arctanh(numpy.clip(arr(xi), -1, 1))


def transf_ztor(xi: numpy.typing.ArrayLike) -> FloatArray:
    """The z-to-r transformation, ``tanh(xi)``."""
    return numpy.tanh(arr(xi))


def transf_ztor_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for z-to-r: E[tanh(Z)] for Z ~ N(xi, tau2), ``targs = {'tau2', 'lower', 'upper'}``."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return numpy.tanh(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return mapint(lambda z, m: math.tanh(z) * d1(z, m, tau), x, lower, upper)


def transf_ztor_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of an atanh-normal variable (``targs`` = ``tau2`` or ``{'tau2': ...}``)."""
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return numpy.tanh(x)
    tau = math.sqrt(tau2)
    return mapmode(lambda r, mu: d1(math.atanh(r), mu, tau) / (1 - r**2), x, -0.9999, 0.9999)


def transf_r2toz(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Variance-stabilising transformation for R-squared, ``atanh(sqrt(xi))``, with ``xi`` truncated to [0, 1]."""
    with numpy.errstate(divide='ignore'):
        return numpy.arctanh(numpy.sqrt(numpy.clip(arr(xi), 0, 1)))


def transf_ztor2(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_r2toz`, ``tanh(xi)^2``."""
    return numpy.tanh(arr(xi)) ** 2


# --------------------------------------------------------------------------
# exponential, logit, arcsine and probit


def transf_exp_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for ``exp``: the log-normal mean ``exp(xi + tau2/2)``."""
    t = chktargsint(targs)
    return numpy.exp(arr(xi) + t['tau2'] / 2)


def transf_exp_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a log-normal variable, ``exp(xi - tau2)``."""
    return numpy.exp(arr(xi) - mode_tau2(targs))


def transf_logit(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Logit transformation, ``qlogis(xi)``."""
    return qlogis(arr(xi))


def transf_ilogit(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse logit transformation, ``plogis(xi)``."""
    return plogis(arr(xi))


def transf_ilogit_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse logit (mean of a logit-normal variable)."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return plogis(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return mapint(lambda z, m: float(plogis(z)) * d1(z, m, tau), x, lower, upper)


XS = numpy.concatenate(([0.0], numpy.arange(1, 10**5 - 1) * (1 / (10**5 - 1)), [1.0]))  # seq(0, 1, length=10^5)


def transf_ilogit_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a logit-normal variable; NaN when the density is bimodal (as in metafor).

    Raises:
        metafopy.errors.InputError: the density cannot be evaluated (an NA value).
    """
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return plogis(x)
    tau = math.sqrt(tau2)
    t2 = tau * tau
    with numpy.errstate(invalid='ignore'):
        lead = t2 * (2 * XS - 1)
        q_xs = qlogis(XS)
    out = numpy.empty(x.shape, dtype=numpy.float64)
    for idx in numpy.ndindex(x.shape):
        mu = float(x[idx])
        with numpy.errstate(invalid='ignore'):
            ys = lead + mu - q_xs
            nmodes = numpy.unique(numpy.sign(numpy.diff(ys))).size
        if nmodes != 1:
            out[idx] = numpy.nan
            continue

        def modefun(p: float, mu: float = mu) -> float:
            val = t2 * (2 * p - 1) + mu - float(qlogis(p))
            if math.isnan(val):  # R's uniroot() stops on NA
                raise metafopy.errors.InputError('NA value')
            return max(-DBL_MAX, min(DBL_MAX, val))  # and replaces +-Inf by the largest doubles

        try:
            out[idx] = metafopy.core.zeroin.uniroot(modefun, 0.0, 1.0).root
        except ValueError:
            out[idx] = numpy.nan
    return out


def transf_arcsin(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Arcsine square root transformation, ``asin(sqrt(xi))``."""
    with numpy.errstate(invalid='ignore'):
        return numpy.arcsin(numpy.sqrt(arr(xi)))


def transf_iarcsin(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse arcsine square root transformation, ``sin(xi)^2``, giving 0 below 0 and 1 above pi/2."""
    x = arr(xi)
    with numpy.errstate(invalid='ignore'):
        zi = numpy.sin(x) ** 2
    zi[x < 0] = 0
    zi[x > HALF_PI] = 1
    return zi


def transf_iarcsin_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse arcsine, with the normal truncated to [0, pi/2].

    The default bounds are ``max(0, xi - 10*tau)`` and ``min(pi/2, xi + 10*tau)`` taken over all ``xi``, as in metafor.
    """
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iarcsin(x)
    tau = math.sqrt(t['tau2'])
    lower = t.get('lower', float(numpy.max(numpy.append(0.0, x - 10 * tau))))
    upper = t.get('upper', float(numpy.min(numpy.append(HALF_PI, x + 10 * tau))))

    def toint(z: float, m: float) -> float:
        return float(transf_iarcsin(z)[0]) * d1(z, m, tau) / (p1((HALF_PI - m) / tau) - p1(-m / tau))

    return mapint(toint, x, lower, upper)


def transf_iarcsin_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of an arcsine-square-root-normal variable."""
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return transf_iarcsin(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return d1(math.asin(math.sqrt(p)), mu, tau) / (2 * math.sqrt(p * (1 - p)) * (p1((HALF_PI - mu) / tau) - p1(-mu / tau)))

    return mapmode(dfun, x, 0.0, 1.0)


def transf_probit(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Probit transformation, ``qnorm(xi)``."""
    return qnorm(arr(xi))


def transf_iprobit(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse probit transformation, ``pnorm(xi)``."""
    return pnorm(arr(xi))


def transf_iprobit_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse probit."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return pnorm(x)
    tau = math.sqrt(t['tau2'])
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return mapint(lambda z, m: p1(z) * d1(z, m, tau), x, lower, upper)


def transf_iprobit_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a probit-normal variable."""
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return pnorm(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        q = float(qnorm(p))
        return d1(q, mu, tau) / d1(q)

    return mapmode(dfun, x, 0.0, 1.0)


# --------------------------------------------------------------------------
# Freeman-Tukey and square root


def recycle_pair(x: FloatArray, n: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``x`` and ``n`` recycled to the length of the longer as R does, with its warning when the longer is not a multiple of the shorter."""
    size = max(x.size, n.size)
    if x.size == 0 or n.size == 0:
        return x[:0], n[:0]
    if size % x.size != 0 or size % n.size != 0:
        warnings.warn('longer object length is not a multiple of shorter object length', stacklevel=4)
    shape = x.shape if x.size == size else n.shape
    return numpy.resize(x, size).reshape(shape), numpy.resize(n, size).reshape(shape)


def transf_pft(xi: numpy.typing.ArrayLike, ni: numpy.typing.ArrayLike) -> FloatArray:
    """Freeman-Tukey double arcsine transformation of proportions ``xi`` with sample sizes ``ni``."""
    xi_, n = recycle_pair(arr(xi), arr(ni))
    x = xi_ * n
    with numpy.errstate(invalid='ignore'):
        return numpy.asarray(1 / 2 * (numpy.arcsin(numpy.sqrt(x / (n + 1))) + numpy.arcsin(numpy.sqrt((x + 1) / (n + 1)))), numpy.float64)


def ipft(x: FloatArray, n: FloatArray) -> FloatArray:
    with numpy.errstate(all='ignore'):
        s = numpy.sin(2 * x)
        zi = 1 / 2 * (1 - numpy.sign(numpy.cos(2 * x)) * numpy.sqrt(1 - (s + (s - 1 / s) / n) ** 2))
        zi = numpy.where(x > transf_pft(1, n), 1.0, zi)
        zi = numpy.where(x < transf_pft(0, n), 0.0, zi)
    return numpy.atleast_1d(numpy.asarray(zi, numpy.float64))


def transf_ipft(xi: numpy.typing.ArrayLike, ni: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of the Freeman-Tukey transformation (Miller, 1978), 0 and 1 outside the attainable range."""
    return ipft(*recycle_pair(arr(xi), arr(ni)))


def transf_ipft_hm(xi: numpy.typing.ArrayLike, targs: Targs) -> FloatArray:
    """Inverse Freeman-Tukey transformation using the harmonic mean of ``targs['ni']``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if targs is None or (isinstance(targs, collections.abc.Mapping) and targs.get('ni') is None):
        raise metafopy.errors.InputError("Must specify the sample sizes via the 'targs' argument.")
    ni = arr(targs['ni'] if isinstance(targs, collections.abc.Mapping) else targs)
    nhm = 1 / numpy.nanmean(1 / ni)
    return ipft(arr(xi), numpy.asarray(nhm, numpy.float64))


def transf_isqrt(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of the square root transformation (``xi^2``, 0 for negative ``xi``)."""
    x = arr(xi)
    zi = x * x
    zi[x < 0] = 0
    return zi


def transf_irft(xi: numpy.typing.ArrayLike, ti: numpy.typing.ArrayLike) -> FloatArray:
    """Freeman-Tukey transformation of incidence rates ``xi`` with person-times ``ti``."""
    with numpy.errstate(invalid='ignore'):
        return numpy.atleast_1d(numpy.asarray(1 / 2 * (numpy.sqrt(arr(xi)) + numpy.sqrt(arr(xi) + 1 / numpy.asarray(ti, numpy.float64))), numpy.float64))


def transf_iirft(xi: numpy.typing.ArrayLike, ti: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_irft`, 0 below the attainable range."""
    x = arr(xi)
    t = numpy.asarray(ti, numpy.float64)
    with numpy.errstate(all='ignore'):
        zi = (1 / t - 8 * x**2 + 16 * t * x**4) / (16 * x**2 * t)
        zi = numpy.where(x < transf_irft(0, t), 0.0, zi)
        zi = numpy.where(zi <= EPSILON, 0.0, zi)  # metafor: avoid finite-precision errors in back-transformed zeros
    return numpy.atleast_1d(numpy.asarray(zi, numpy.float64))


# --------------------------------------------------------------------------
# coefficient alpha


def transf_ahw(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Hakstian-Whalen transformation of alpha, ``1 - (1-xi)^(1/3)``."""
    base = 1 - arr(xi)
    with numpy.errstate(invalid='ignore'):
        return numpy.where(base < 0, numpy.nan, 1 - base ** (1 / 3))  # R's (-Inf)^(1/3) is NaN, C's pow() gives Inf


def transf_iahw(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_ahw`, ``1 - (1-xi)^3``, truncated to [0, 1]."""
    x = arr(xi)
    zi = 1 - (1 - x) ** 3
    zi[x > 1] = 1
    zi[x < 0] = 0
    return zi


def transf_iahw_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_iahw` (normal truncated to [0, 1]; bounds over all ``xi``)."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iahw(x)
    tau = math.sqrt(tau2)
    lower = t.get('lower', float(numpy.max(numpy.append(0.0, x - 10 * tau))))
    upper = t.get('upper', float(numpy.min(numpy.append(1.0, x + 10 * tau))))

    def toint(z: float, m: float) -> float:
        return float(transf_iahw(z)[0]) * d1(z, m, tau) / (p1((1 - m) / tau) - p1(-m / tau))

    return mapint(toint, x, lower, upper)


def transf_iahw_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a :func:`transf_ahw`-normal variable."""
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return transf_iahw(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return d1(1 - math.pow(1 - p, 1 / 3), mu, tau) / (3 * math.pow(1 - p, 2 / 3) * (p1((1 - mu) / tau) - p1(-mu / tau)))

    return mapmode(dfun, x, 0.0, 1.0)


def transf_abt(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Bonett transformation of alpha, ``-log(1-xi)``."""
    with numpy.errstate(divide='ignore', invalid='ignore'):
        return -numpy.log(1 - arr(xi))


def transf_iabt(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_abt`, ``1 - exp(-xi)``, 0 for negative ``xi``."""
    x = arr(xi)
    with numpy.errstate(over='ignore'):
        zi = 1 - numpy.exp(-x)
    zi[x < 0] = 0
    return zi


def transf_iabt_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_iabt` (normal truncated to [0, Inf); lower bound over all ``xi``)."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iabt(x)
    tau = math.sqrt(tau2)
    lower = t.get('lower', float(numpy.max(numpy.append(0.0, x - 10 * tau))))
    upper = t.get('upper', x + 10 * tau)

    def toint(z: float, m: float) -> float:
        return float(transf_iabt(z)[0]) * d1(z, m, tau) / p1(m / tau)

    return mapint(toint, x, lower, upper)


def transf_iabt_mode(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a :func:`transf_abt`-normal variable."""
    tau2 = mode_tau2(targs)
    x = arr(xi)
    if tau2 == 0:
        return transf_iabt(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return d1(-math.log(1 - p), mu, tau) / ((1 - p) * p1(mu / tau))

    return mapmode(dfun, x, 0.0, 1.0)


# --------------------------------------------------------------------------
# standardised mean differences


def transf_dtou1(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to Cohen's U1 (proportion of non-overlap)."""
    u2i = pnorm(numpy.abs(arr(xi)) / 2)
    return (2 * u2i - 1) / u2i


def transf_dtou2(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to Cohen's U2, ``pnorm(xi/2)``."""
    return pnorm(arr(xi) / 2)


def transf_dtou3(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to Cohen's U3, ``pnorm(xi)``."""
    return pnorm(arr(xi))


def transf_dtoovl(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to the overlapping coefficient, ``2*pnorm(-|xi|/2)``."""
    return 2 * pnorm(-numpy.abs(arr(xi)) / 2)


def transf_dtocles(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to the common language effect size, ``pnorm(xi/sqrt(2))``."""
    return pnorm(arr(xi) / math.sqrt(2))


def transf_clestod(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_dtocles`, ``qnorm(xi)*sqrt(2)``."""
    return qnorm(arr(xi)) * math.sqrt(2)


def transf_dtocles_int(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_dtocles`."""
    t = chktargsint(targs)
    x = arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return transf_dtocles(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return mapint(lambda z, m: p1(z / math.sqrt(2)) * d1(z, m, tau), x, lower, upper)


def transf_dtocliffd(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to Cliff's delta, ``2*pnorm(xi/sqrt(2)) - 1``."""
    return 2 * pnorm(arr(xi) / math.sqrt(2)) - 1


def transf_dtobesd(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to the binomial effect size display (proportion above the median in the first group)."""
    x = arr(xi)
    with numpy.errstate(invalid='ignore'):
        rpbi = x / numpy.sqrt(x**2 + 4)
    return 0.50 + rpbi / 2


def transf_dtomd(xi: numpy.typing.ArrayLike, targs: Targs = None) -> FloatArray:
    """SMD to mean difference given a known standard deviation (``targs`` = ``sd`` or ``{'sd': ...}``).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if targs is None or (isinstance(targs, collections.abc.Mapping) and targs.get('sd') is None):
        raise metafopy.errors.InputError("Must specify a standard deviation value via the 'targs' argument.")
    sd = numpy.atleast_1d(numpy.asarray(targs['sd'] if isinstance(targs, collections.abc.Mapping) else targs, dtype=numpy.float64)).ravel()
    if sd.size != 1:
        raise metafopy.errors.InputError("Specify a single standard deviation value via the 'targs' argument.")
    return arr(xi) * float(sd[0])


def transf_dtorpb(xi: numpy.typing.ArrayLike, n1i: numpy.typing.ArrayLike | None = None, n2i: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """SMD to point-biserial correlation; exact when ``n1i``/``n2i`` are given, else assumes equal groups."""
    x = arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
    else:
        n1, n2 = lengths(x, n1i, n2i, "Length of 'n1i' and 'n2i' does not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with numpy.errstate(invalid='ignore'):
        return numpy.asarray(x / numpy.sqrt(x**2 + hi), numpy.float64)


def transf_dtorbis(xi: numpy.typing.ArrayLike, n1i: numpy.typing.ArrayLike | None = None, n2i: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """SMD to biserial correlation (via the point-biserial correlation)."""
    x = arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
        n1, n2 = numpy.ones(1), numpy.ones(1)
    else:
        n1, n2 = lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with numpy.errstate(all='ignore'):
        rpbi = x / numpy.sqrt(x**2 + hi)
        p = n1 / (n1 + n2)
        return numpy.asarray(numpy.sqrt(p * (1 - p)) / dnorm(qnorm(p)) * rpbi, numpy.float64)


def pi_check(x: FloatArray, pi: numpy.typing.ArrayLike | None) -> FloatArray:
    if pi is None:
        return numpy.full(1, 0.5)
    p = expand_p(x, pi, 'pi')
    if numpy.any((p < 0) | (p > 1)):
        raise metafopy.errors.InputError("One or more 'pi' values are < 0 or > 1.")
    return p


def transf_rpbtorbis(xi: numpy.typing.ArrayLike, pi: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """Point-biserial to biserial correlation; ``pi`` is the proportion in the first group (default 0.5)."""
    x = arr(xi)
    p = pi_check(x, pi)
    with numpy.errstate(all='ignore'):
        return numpy.asarray(numpy.sqrt(p * (1 - p)) / dnorm(qnorm(p)) * x, numpy.float64)


def transf_rtorpb(xi: numpy.typing.ArrayLike, pi: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """Pearson correlation to the point-biserial correlation after dichotomising at proportion ``pi``."""
    x = arr(xi)
    p = pi_check(x, pi)
    with numpy.errstate(all='ignore'):
        return numpy.asarray(x * dnorm(qnorm(p)) / numpy.sqrt(p * (1 - p)), numpy.float64)


def transf_rtod(xi: numpy.typing.ArrayLike, n1i: numpy.typing.ArrayLike | None = None, n2i: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """Pearson correlation to SMD for a dichotomised variable (inverse of :func:`transf_dtorbis`).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
        n1, n2 = numpy.ones(1), numpy.ones(1)
    else:
        n1, n2 = lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    if numpy.any(n1 < 0) or numpy.any(n2 < 0):
        raise metafopy.errors.InputError("One or more values specified via the 'n1i' or 'n2i' arguments are negative.")
    with numpy.errstate(all='ignore'):
        p = n1 / (n1 + n2)
        rpbi = x * dnorm(qnorm(p)) / numpy.sqrt(p * (1 - p))
        return numpy.asarray(numpy.sqrt(hi) * rpbi / numpy.sqrt(1 - rpbi**2), numpy.float64)


def transf_rpbtod(xi: numpy.typing.ArrayLike, n1i: numpy.typing.ArrayLike | None = None, n2i: numpy.typing.ArrayLike | None = None) -> FloatArray:
    """Point-biserial correlation to SMD (inverse of :func:`transf_dtorpb`)."""
    x = arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
    else:
        n1, n2 = lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with numpy.errstate(all='ignore'):
        return numpy.asarray(numpy.sqrt(hi) * x / numpy.sqrt(1 - x**2), numpy.float64)


# --------------------------------------------------------------------------
# log odds ratios


def pc_check(x: FloatArray, pc: numpy.typing.ArrayLike) -> FloatArray:
    p = expand_p(x, pc, 'pc')
    if numpy.any(p < 0) or numpy.any(p > 1):
        raise metafopy.errors.InputError("The control group risk 'pc' must be between 0 and 1.")
    return p


def transf_lnortord(xi: numpy.typing.ArrayLike, pc: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to risk difference for control group risk ``pc``."""
    x = arr(xi)
    p = pc_check(x, pc)
    with numpy.errstate(all='ignore'):
        e = numpy.exp(x)
        return numpy.asarray(e * p / (1 - p + p * e) - p, numpy.float64)


def transf_lnortorr(xi: numpy.typing.ArrayLike, pc: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to risk ratio (not log risk ratio) for control group risk ``pc``."""
    x = arr(xi)
    p = pc_check(x, pc)
    with numpy.errstate(all='ignore'):
        e = numpy.exp(x)
        return numpy.asarray(e / (p * (e - 1) + 1), numpy.float64)


def transf_lnortod_norm(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to SMD assuming normal distributions (Cox & Snell), ``xi/1.65``."""
    return arr(xi) / 1.65


def transf_lnortod_logis(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to SMD assuming logistic distributions (Chinn), ``sqrt(3)/pi*xi``."""
    return math.sqrt(3) / math.pi * arr(xi)


def transf_dtolnor_norm(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to log odds ratio assuming normal distributions, ``xi*1.65``."""
    return arr(xi) * 1.65


def transf_dtolnor_logis(xi: numpy.typing.ArrayLike) -> FloatArray:
    """SMD to log odds ratio assuming logistic distributions, ``xi/sqrt(3)*pi``."""
    return arr(xi) / math.sqrt(3) * math.pi


def transf_lnortortet_pearson(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to tetrachoric correlation (Pearson, 1900), ``cos(pi/(1+sqrt(exp(xi))))``."""
    with numpy.errstate(over='ignore'):
        return numpy.cos(math.pi / (1 + numpy.sqrt(numpy.exp(arr(xi)))))


def transf_lnortortet_digby(xi: numpy.typing.ArrayLike) -> FloatArray:
    """Log odds ratio to tetrachoric correlation (Digby, 1983), ``(OR^(3/4) - 1)/(OR^(3/4) + 1)``."""
    with numpy.errstate(over='ignore', invalid='ignore'):
        e = numpy.exp(arr(xi)) ** (3 / 4)
        return numpy.asarray((e - 1) / (e + 1), numpy.float64)


# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
TAU2_DEFAULT = frozenset(
    {
        'transf_exp_int',
        'transf_ilogit_int',
        'transf_iprobit_int',
        'transf_ztor_int',
        'transf_iarcsin_int',
        'transf_iahw_int',
        'transf_iabt_int',
        'transf_dtocles_int',
        'transf_exp_mode',
        'transf_ilogit_mode',
        'transf_iprobit_mode',
        'transf_ztor_mode',
        'transf_iarcsin_mode',
        'transf_iahw_mode',
        'transf_iabt_mode',
    }
)


def is_tau2_transf(transf: collections.abc.Callable[..., typing.Any]) -> bool:
    """Whether ``transf`` is one of the integral/mode transformations that take ``tau2``."""
    name = getattr(transf, '__name__', None)
    return name in TAU2_DEFAULT and transf is globals().get(name)


def apply_transf(
    values: numpy.typing.ArrayLike, transf: collections.abc.Callable[..., typing.Any], targs: typing.Any = None, tau2: float | None = None
) -> FloatArray:
    """Apply ``transf`` to each value separately, as metafor's ``sapply(x, transf, targs)``.

    For the integral and mode transformations, ``targs`` defaults to
    ``{'tau2': tau2}`` when a model's ``tau2`` is given, as in
    ``predict.rma``/``confint.rma.uni``.
    """
    if targs is None and tau2 is not None and is_tau2_transf(transf):
        targs = {'tau2': tau2}
    vals = numpy.atleast_1d(numpy.asarray(values, dtype=numpy.float64))
    out = [transf(v) if targs is None else transf(v, targs) for v in vals]
    return numpy.array([float(numpy.asarray(o, dtype=numpy.float64).reshape(-1)[0]) for o in out], dtype=numpy.float64).reshape(vals.shape)
