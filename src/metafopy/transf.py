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

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special

from metafopy.core.optim import fmin
from metafopy.core.zeroin import uniroot

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

type FloatArray = NDArray[np.float64]
type Targs = ArrayLike | Mapping[str, Any] | None

EPSILON = sys.float_info.epsilon
DBL_MAX = sys.float_info.max
_TOL = EPSILON**0.25  # R's integrate() rel.tol and abs.tol
_M_1_SQRT_2PI = 0.398942280401432677939946059934
_HALF_PI = math.pi / 2
# x beyond which R's dnorm() returns 0: sqrt(-2*M_LN2*(DBL_MIN_EXP + 1 - DBL_MANT_DIG))
_DNORM_UFLOW = math.sqrt(-2 * math.log(2) * (-1021 + 1 - 53))


# --------------------------------------------------------------------------
# helpers: R's distribution functions and metafor's argument checks


def _arr(x: ArrayLike) -> FloatArray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def _dnorm(x: ArrayLike, mean: ArrayLike = 0.0, sd: ArrayLike = 1.0) -> FloatArray:
    """R's ``dnorm()`` (``dnorm4`` in ``nmath/dnorm.c``), including its accurate tail branch."""
    xb, mu, sigma = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(mean, np.float64), np.asarray(sd, np.float64))
    with np.errstate(all='ignore'):
        z = np.abs((xb - mu) / sigma)
        x1 = np.ldexp(np.rint(np.ldexp(z, 16)), -16)
        x2 = z - x1
        body = np.where(
            z < 5,
            _M_1_SQRT_2PI * np.exp(-0.5 * z * z) / sigma,
            np.where(z > _DNORM_UFLOW, 0.0, _M_1_SQRT_2PI / sigma * (np.exp(-0.5 * x1 * x1) * np.exp((-0.5 * x2 - x1) * x2))),
        )
        out: FloatArray = np.select(
            [
                np.isnan(xb) | np.isnan(mu) | np.isnan(sigma),
                sigma < 0,
                ~np.isfinite(sigma),
                ~np.isfinite(xb) & (mu == xb),
                sigma == 0,
                ~np.isfinite(z) | (z >= 2 * math.sqrt(DBL_MAX)),
            ],
            [xb + mu + sigma, np.nan, 0.0, np.nan, np.where(xb == mu, np.inf, 0.0), 0.0],
            body,
        )
    return out


def _d1(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    return float(_dnorm(x, mean, sd))


def _pnorm(x: ArrayLike) -> FloatArray:
    return np.asarray(special.ndtr(np.asarray(x, np.float64)), np.float64)


def _p1(x: float) -> float:
    return float(special.ndtr(x))


def _qnorm(p: ArrayLike) -> FloatArray:
    return np.asarray(special.ndtri(np.asarray(p, np.float64)), np.float64)


def _plogis(x: ArrayLike) -> FloatArray:
    with np.errstate(over='ignore'):
        return np.asarray(1 / (1 + np.exp(-np.asarray(x, np.float64))), np.float64)


def _qlogis(p: ArrayLike) -> FloatArray:
    pa = np.asarray(p, np.float64)
    with np.errstate(all='ignore'):
        out = np.log(pa / (1 - pa))
    return np.asarray(np.where((pa < 0) | (pa > 1), np.nan, out), np.float64)


def _chktargsint(targs: Targs) -> dict[str, float]:
    """metafor's ``.chktargsint``: ``tau2`` (required, >= 0) and optional ``lower``/``upper`` bounds."""
    if isinstance(targs, Mapping):
        if len(targs) > 3:
            raise ValueError("Length of the 'targs' argument must be <= 3.")
        items = {k: v for k, v in targs.items() if v is not None}
    elif targs is None:
        items = {}
    else:
        flat = np.atleast_1d(np.asarray(targs, dtype=np.float64)).ravel()
        if flat.size > 3:
            raise ValueError("Length of the 'targs' argument must be <= 3.")
        items = dict(zip(('tau2', 'lower', 'upper'), flat, strict=False))
    if any(np.size(v) > 1 for v in items.values()):
        raise ValueError("Elements of the 'targs' arguments must be scalars.")
    out: dict[str, float] = {
        k: float(np.asarray(items[k], dtype=np.float64).reshape(-1)[0]) for k in ('tau2', 'lower', 'upper') if k in items and np.size(items[k]) == 1
    }
    if 'tau2' not in out:
        raise ValueError("Must specify a 'tau2' value via the 'targs' argument.")
    if not out['tau2'] >= 0:
        raise ValueError("Value of 'tau2' must be >= 0 to use an integral transformation.")
    return out


def _mode_tau2(targs: Targs) -> float:
    """The ``tau2`` checks shared by the ``transf.*.mode`` functions; values below machine epsilon become 0."""
    if targs is None or (isinstance(targs, Mapping) and targs.get('tau2') is None):
        raise ValueError("Must specify a 'tau2' value via the 'targs' argument.")
    tau2 = np.atleast_1d(np.asarray(targs['tau2'] if isinstance(targs, Mapping) else targs, dtype=np.float64)).ravel()
    if tau2.size > 1:
        raise ValueError("Value of 'tau2' argument must be a scalar.")
    t2 = float(tau2[0])
    if not t2 >= 0:
        raise ValueError("Value of 'tau2' must be >= 0 to use this transformation function.")
    return 0.0 if t2 < EPSILON else t2


class _NonFiniteError(Exception):
    pass


def _integrate(f: Callable[[float], float], lower: float, upper: float) -> float:
    """``integrate(f, lower, upper)$value`` as metafor calls it, with NaN where R raises an error."""
    if math.isnan(lower) or math.isnan(upper):
        return math.nan

    def g(z: float) -> float:
        try:
            val = f(z)
        except ArithmeticError, ValueError:  # x/0 and domain errors, which are Inf/NaN in R
            raise _NonFiniteError from None
        if not math.isfinite(val):  # R: "non-finite function value"
            raise _NonFiniteError
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
        res: tuple[Any, ...] = integrate.quad(g, a, b, epsabs=_TOL, epsrel=_TOL, limit=100, full_output=1)
    except _NonFiniteError:
        return math.nan
    if len(res) > 3:  # ier > 0: R stops ("roundoff error", "maximum number of subdivisions", ...)
        return math.nan
    return float(res[0])


def _mapint(toint: Callable[[float, float], float], xi: FloatArray, lower: ArrayLike, upper: ArrayLike) -> FloatArray:
    """``mapply(xi, FUN=cfunc, lower=..., upper=...)``: integrate ``toint(z, xi)`` for each ``xi``."""
    xb, lb, ub = np.broadcast_arrays(xi, np.asarray(lower, np.float64), np.asarray(upper, np.float64))
    out = np.empty(xb.shape, dtype=np.float64)
    for idx in np.ndindex(xb.shape):
        x = float(xb[idx])

        def f(z: float, x: float = x) -> float:
            return toint(z, x)

        out[idx] = _integrate(f, float(lb[idx]), float(ub[idx]))
    return out


def _mapmode(dfun: Callable[[float, float], float], xi: FloatArray, lower: float, upper: float) -> FloatArray:
    """``sapply(xi, optimize(dfun, maximum=TRUE, lower, upper, mu=x)$maximum)``."""
    out = np.empty(xi.shape, dtype=np.float64)
    for idx in np.ndindex(xi.shape):
        mu = float(xi[idx])

        def negf(x: float, mu: float = mu) -> float:
            try:
                return -dfun(x, mu)
            except ArithmeticError, ValueError:  # Inf/NaN in R, which optimize() replaces by DBL_MAX
                return math.nan

        out[idx] = fmin(negf, lower, upper)
    return out


def _lengths(xi: FloatArray, n1i: ArrayLike, n2i: ArrayLike, msg: str) -> tuple[FloatArray, FloatArray]:
    n1, n2 = _arr(n1i), _arr(n2i)
    if n1.size != n2.size:
        raise ValueError("Length of 'n1i' does not match the length of 'n2i'.")
    if n1.size != xi.size:
        raise ValueError(msg)
    return n1, n2


def _expand_p(xi: FloatArray, p: ArrayLike, name: str) -> FloatArray:
    pa = _arr(p)
    if pa.size == 1:
        pa = np.full(xi.shape, pa.reshape(-1)[0])
    if pa.size != xi.size:
        raise ValueError(f"Length of 'xi' does not match the length of '{name}'.")
    return pa


# --------------------------------------------------------------------------
# correlations


def transf_rtoz(xi: ArrayLike) -> FloatArray:
    """Fisher's r-to-z transformation, ``atanh(xi)``, with ``xi`` truncated to [-1, 1]."""
    with np.errstate(divide='ignore'):
        return np.arctanh(np.clip(_arr(xi), -1, 1))


def transf_ztor(xi: ArrayLike) -> FloatArray:
    """The z-to-r transformation, ``tanh(xi)``."""
    return np.tanh(_arr(xi))


def transf_ztor_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for z-to-r: E[tanh(Z)] for Z ~ N(xi, tau2), ``targs = {'tau2', 'lower', 'upper'}``."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return np.tanh(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return _mapint(lambda z, m: math.tanh(z) * _d1(z, m, tau), x, lower, upper)


def transf_ztor_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of an atanh-normal variable (``targs`` = ``tau2`` or ``{'tau2': ...}``)."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return np.tanh(x)
    tau = math.sqrt(tau2)
    return _mapmode(lambda r, mu: _d1(math.atanh(r), mu, tau) / (1 - r**2), x, -0.9999, 0.9999)


def transf_r2toz(xi: ArrayLike) -> FloatArray:
    """Variance-stabilising transformation for R-squared, ``atanh(sqrt(xi))``, with ``xi`` truncated to [0, 1]."""
    with np.errstate(divide='ignore'):
        return np.arctanh(np.sqrt(np.clip(_arr(xi), 0, 1)))


def transf_ztor2(xi: ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_r2toz`, ``tanh(xi)^2``."""
    return np.tanh(_arr(xi)) ** 2


# --------------------------------------------------------------------------
# exponential, logit, arcsine and probit


def transf_exp_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for ``exp``: the log-normal mean ``exp(xi + tau2/2)``."""
    t = _chktargsint(targs)
    return np.exp(_arr(xi) + t['tau2'] / 2)


def transf_exp_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a log-normal variable, ``exp(xi - tau2)``."""
    return np.exp(_arr(xi) - _mode_tau2(targs))


def transf_logit(xi: ArrayLike) -> FloatArray:
    """Logit transformation, ``qlogis(xi)``."""
    return _qlogis(_arr(xi))


def transf_ilogit(xi: ArrayLike) -> FloatArray:
    """Inverse logit transformation, ``plogis(xi)``."""
    return _plogis(_arr(xi))


def transf_ilogit_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse logit (mean of a logit-normal variable)."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return _plogis(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return _mapint(lambda z, m: float(_plogis(z)) * _d1(z, m, tau), x, lower, upper)


_XS = np.concatenate(([0.0], np.arange(1, 10**5 - 1) * (1 / (10**5 - 1)), [1.0]))  # seq(0, 1, length=10^5)


def transf_ilogit_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a logit-normal variable; NaN when the density is bimodal (as in metafor)."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return _plogis(x)
    tau = math.sqrt(tau2)
    t2 = tau * tau
    with np.errstate(invalid='ignore'):
        lead = t2 * (2 * _XS - 1)
        q_xs = _qlogis(_XS)
    out = np.empty(x.shape, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        mu = float(x[idx])
        with np.errstate(invalid='ignore'):
            ys = lead + mu - q_xs
            nmodes = np.unique(np.sign(np.diff(ys))).size
        if nmodes != 1:
            out[idx] = np.nan
            continue

        def modefun(p: float, mu: float = mu) -> float:
            val = t2 * (2 * p - 1) + mu - float(_qlogis(p))
            if math.isnan(val):  # R's uniroot() stops on NA
                raise ValueError('NA value')
            return max(-DBL_MAX, min(DBL_MAX, val))  # and replaces +-Inf by the largest doubles

        try:
            out[idx] = uniroot(modefun, 0.0, 1.0).root
        except ValueError:
            out[idx] = np.nan
    return out


def transf_arcsin(xi: ArrayLike) -> FloatArray:
    """Arcsine square root transformation, ``asin(sqrt(xi))``."""
    with np.errstate(invalid='ignore'):
        return np.arcsin(np.sqrt(_arr(xi)))


def transf_iarcsin(xi: ArrayLike) -> FloatArray:
    """Inverse arcsine square root transformation, ``sin(xi)^2``, giving 0 below 0 and 1 above pi/2."""
    x = _arr(xi)
    with np.errstate(invalid='ignore'):
        zi = np.sin(x) ** 2
    zi[x < 0] = 0
    zi[x > _HALF_PI] = 1
    return zi


def transf_iarcsin_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse arcsine, with the normal truncated to [0, pi/2].

    The default bounds are ``max(0, xi - 10*tau)`` and ``min(pi/2, xi + 10*tau)`` taken over all ``xi``, as in metafor.
    """
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iarcsin(x)
    tau = math.sqrt(t['tau2'])
    lower = t.get('lower', float(np.max(np.append(0.0, x - 10 * tau))))
    upper = t.get('upper', float(np.min(np.append(_HALF_PI, x + 10 * tau))))

    def toint(z: float, m: float) -> float:
        return float(transf_iarcsin(z)[0]) * _d1(z, m, tau) / (_p1((_HALF_PI - m) / tau) - _p1(-m / tau))

    return _mapint(toint, x, lower, upper)


def transf_iarcsin_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of an arcsine-square-root-normal variable."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return transf_iarcsin(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return _d1(math.asin(math.sqrt(p)), mu, tau) / (2 * math.sqrt(p * (1 - p)) * (_p1((_HALF_PI - mu) / tau) - _p1(-mu / tau)))

    return _mapmode(dfun, x, 0.0, 1.0)


def transf_probit(xi: ArrayLike) -> FloatArray:
    """Probit transformation, ``qnorm(xi)``."""
    return _qnorm(_arr(xi))


def transf_iprobit(xi: ArrayLike) -> FloatArray:
    """Inverse probit transformation, ``pnorm(xi)``."""
    return _pnorm(_arr(xi))


def transf_iprobit_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for the inverse probit."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return _pnorm(x)
    tau = math.sqrt(t['tau2'])
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return _mapint(lambda z, m: _p1(z) * _d1(z, m, tau), x, lower, upper)


def transf_iprobit_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a probit-normal variable."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return _pnorm(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        q = float(_qnorm(p))
        return _d1(q, mu, tau) / _d1(q)

    return _mapmode(dfun, x, 0.0, 1.0)


# --------------------------------------------------------------------------
# Freeman-Tukey and square root


def transf_pft(xi: ArrayLike, ni: ArrayLike) -> FloatArray:
    """Freeman-Tukey double arcsine transformation of proportions ``xi`` with sample sizes ``ni``."""
    n = np.asarray(ni, np.float64)
    x = _arr(xi) * n
    with np.errstate(invalid='ignore'):
        return np.asarray(1 / 2 * (np.arcsin(np.sqrt(x / (n + 1))) + np.arcsin(np.sqrt((x + 1) / (n + 1)))), np.float64)


def _ipft(x: FloatArray, n: FloatArray) -> FloatArray:
    with np.errstate(all='ignore'):
        s = np.sin(2 * x)
        zi = 1 / 2 * (1 - np.sign(np.cos(2 * x)) * np.sqrt(1 - (s + (s - 1 / s) / n) ** 2))
        zi = np.where(x > transf_pft(1, n), 1.0, zi)
        zi = np.where(x < transf_pft(0, n), 0.0, zi)
    return np.atleast_1d(np.asarray(zi, np.float64))


def transf_ipft(xi: ArrayLike, ni: ArrayLike) -> FloatArray:
    """Inverse of the Freeman-Tukey transformation (Miller, 1978), 0 and 1 outside the attainable range."""
    return _ipft(_arr(xi), np.asarray(ni, np.float64))


def transf_ipft_hm(xi: ArrayLike, targs: Targs) -> FloatArray:
    """Inverse Freeman-Tukey transformation using the harmonic mean of ``targs['ni']``."""
    if targs is None or (isinstance(targs, Mapping) and targs.get('ni') is None):
        raise ValueError("Must specify the sample sizes via the 'targs' argument.")
    ni = _arr(targs['ni'] if isinstance(targs, Mapping) else targs)
    nhm = 1 / np.nanmean(1 / ni)
    return _ipft(_arr(xi), np.asarray(nhm, np.float64))


def transf_isqrt(xi: ArrayLike) -> FloatArray:
    """Inverse of the square root transformation (``xi^2``, 0 for negative ``xi``)."""
    x = _arr(xi)
    zi = x * x
    zi[x < 0] = 0
    return zi


def transf_irft(xi: ArrayLike, ti: ArrayLike) -> FloatArray:
    """Freeman-Tukey transformation of incidence rates ``xi`` with person-times ``ti``."""
    with np.errstate(invalid='ignore'):
        return np.atleast_1d(np.asarray(1 / 2 * (np.sqrt(_arr(xi)) + np.sqrt(_arr(xi) + 1 / np.asarray(ti, np.float64))), np.float64))


def transf_iirft(xi: ArrayLike, ti: ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_irft`, 0 below the attainable range."""
    x = _arr(xi)
    t = np.asarray(ti, np.float64)
    with np.errstate(all='ignore'):
        zi = (1 / t - 8 * x**2 + 16 * t * x**4) / (16 * x**2 * t)
        zi = np.where(x < transf_irft(0, t), 0.0, zi)
        zi = np.where(zi <= EPSILON, 0.0, zi)  # metafor: avoid finite-precision errors in back-transformed zeros
    return np.atleast_1d(np.asarray(zi, np.float64))


# --------------------------------------------------------------------------
# coefficient alpha


def transf_ahw(xi: ArrayLike) -> FloatArray:
    """Hakstian-Whalen transformation of alpha, ``1 - (1-xi)^(1/3)``."""
    base = 1 - _arr(xi)
    with np.errstate(invalid='ignore'):
        return np.where(base < 0, np.nan, 1 - base ** (1 / 3))  # R's (-Inf)^(1/3) is NaN, C's pow() gives Inf


def transf_iahw(xi: ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_ahw`, ``1 - (1-xi)^3``, truncated to [0, 1]."""
    x = _arr(xi)
    zi = 1 - (1 - x) ** 3
    zi[x > 1] = 1
    zi[x < 0] = 0
    return zi


def transf_iahw_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_iahw` (normal truncated to [0, 1]; bounds over all ``xi``)."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iahw(x)
    tau = math.sqrt(tau2)
    lower = t.get('lower', float(np.max(np.append(0.0, x - 10 * tau))))
    upper = t.get('upper', float(np.min(np.append(1.0, x + 10 * tau))))

    def toint(z: float, m: float) -> float:
        return float(transf_iahw(z)[0]) * _d1(z, m, tau) / (_p1((1 - m) / tau) - _p1(-m / tau))

    return _mapint(toint, x, lower, upper)


def transf_iahw_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a :func:`transf_ahw`-normal variable."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return transf_iahw(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return _d1(1 - math.pow(1 - p, 1 / 3), mu, tau) / (3 * math.pow(1 - p, 2 / 3) * (_p1((1 - mu) / tau) - _p1(-mu / tau)))

    return _mapmode(dfun, x, 0.0, 1.0)


def transf_abt(xi: ArrayLike) -> FloatArray:
    """Bonett transformation of alpha, ``-log(1-xi)``."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.log(1 - _arr(xi))


def transf_iabt(xi: ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_abt`, ``1 - exp(-xi)``, 0 for negative ``xi``."""
    x = _arr(xi)
    with np.errstate(over='ignore'):
        zi = 1 - np.exp(-x)
    zi[x < 0] = 0
    return zi


def transf_iabt_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_iabt` (normal truncated to [0, Inf); lower bound over all ``xi``)."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    if tau2 == 0:
        return transf_iabt(x)
    tau = math.sqrt(tau2)
    lower = t.get('lower', float(np.max(np.append(0.0, x - 10 * tau))))
    upper = t.get('upper', x + 10 * tau)

    def toint(z: float, m: float) -> float:
        return float(transf_iabt(z)[0]) * _d1(z, m, tau) / _p1(m / tau)

    return _mapint(toint, x, lower, upper)


def transf_iabt_mode(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Mode of a :func:`transf_abt`-normal variable."""
    tau2 = _mode_tau2(targs)
    x = _arr(xi)
    if tau2 == 0:
        return transf_iabt(x)
    tau = math.sqrt(tau2)

    def dfun(p: float, mu: float) -> float:
        return _d1(-math.log(1 - p), mu, tau) / ((1 - p) * _p1(mu / tau))

    return _mapmode(dfun, x, 0.0, 1.0)


# --------------------------------------------------------------------------
# standardised mean differences


def transf_dtou1(xi: ArrayLike) -> FloatArray:
    """SMD to Cohen's U1 (proportion of non-overlap)."""
    u2i = _pnorm(np.abs(_arr(xi)) / 2)
    return (2 * u2i - 1) / u2i


def transf_dtou2(xi: ArrayLike) -> FloatArray:
    """SMD to Cohen's U2, ``pnorm(xi/2)``."""
    return _pnorm(_arr(xi) / 2)


def transf_dtou3(xi: ArrayLike) -> FloatArray:
    """SMD to Cohen's U3, ``pnorm(xi)``."""
    return _pnorm(_arr(xi))


def transf_dtoovl(xi: ArrayLike) -> FloatArray:
    """SMD to the overlapping coefficient, ``2*pnorm(-|xi|/2)``."""
    return 2 * _pnorm(-np.abs(_arr(xi)) / 2)


def transf_dtocles(xi: ArrayLike) -> FloatArray:
    """SMD to the common language effect size, ``pnorm(xi/sqrt(2))``."""
    return _pnorm(_arr(xi) / math.sqrt(2))


def transf_clestod(xi: ArrayLike) -> FloatArray:
    """Inverse of :func:`transf_dtocles`, ``qnorm(xi)*sqrt(2)``."""
    return _qnorm(_arr(xi)) * math.sqrt(2)


def transf_dtocles_int(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """Integral transformation for :func:`transf_dtocles`."""
    t = _chktargsint(targs)
    x = _arr(xi)
    tau2 = t['tau2'] if t['tau2'] >= EPSILON else 0.0
    tau = math.sqrt(tau2)
    if tau2 == 0:
        return transf_dtocles(x)
    lower = t.get('lower', x - 10 * tau)
    upper = t.get('upper', x + 10 * tau)
    return _mapint(lambda z, m: _p1(z / math.sqrt(2)) * _d1(z, m, tau), x, lower, upper)


def transf_dtocliffd(xi: ArrayLike) -> FloatArray:
    """SMD to Cliff's delta, ``2*pnorm(xi/sqrt(2)) - 1``."""
    return 2 * _pnorm(_arr(xi) / math.sqrt(2)) - 1


def transf_dtobesd(xi: ArrayLike) -> FloatArray:
    """SMD to the binomial effect size display (proportion above the median in the first group)."""
    x = _arr(xi)
    with np.errstate(invalid='ignore'):
        rpbi = x / np.sqrt(x**2 + 4)
    return 0.50 + rpbi / 2


def transf_dtomd(xi: ArrayLike, targs: Targs = None) -> FloatArray:
    """SMD to mean difference given a known standard deviation (``targs`` = ``sd`` or ``{'sd': ...}``)."""
    if targs is None or (isinstance(targs, Mapping) and targs.get('sd') is None):
        raise ValueError("Must specify a standard deviation value via the 'targs' argument.")
    sd = np.atleast_1d(np.asarray(targs['sd'] if isinstance(targs, Mapping) else targs, dtype=np.float64)).ravel()
    if sd.size != 1:
        raise ValueError("Specify a single standard deviation value via the 'targs' argument.")
    return _arr(xi) * float(sd[0])


def transf_dtorpb(xi: ArrayLike, n1i: ArrayLike | None = None, n2i: ArrayLike | None = None) -> FloatArray:
    """SMD to point-biserial correlation; exact when ``n1i``/``n2i`` are given, else assumes equal groups."""
    x = _arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
    else:
        n1, n2 = _lengths(x, n1i, n2i, "Length of 'n1i' and 'n2i' does not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with np.errstate(invalid='ignore'):
        return np.asarray(x / np.sqrt(x**2 + hi), np.float64)


def transf_dtorbis(xi: ArrayLike, n1i: ArrayLike | None = None, n2i: ArrayLike | None = None) -> FloatArray:
    """SMD to biserial correlation (via the point-biserial correlation)."""
    x = _arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
        n1, n2 = np.ones(1), np.ones(1)
    else:
        n1, n2 = _lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with np.errstate(all='ignore'):
        rpbi = x / np.sqrt(x**2 + hi)
        p = n1 / (n1 + n2)
        return np.asarray(np.sqrt(p * (1 - p)) / _dnorm(_qnorm(p)) * rpbi, np.float64)


def _pi_check(x: FloatArray, pi: ArrayLike | None) -> FloatArray:
    if pi is None:
        return np.full(1, 0.5)
    p = _expand_p(x, pi, 'pi')
    if np.any((p < 0) | (p > 1)):
        raise ValueError("One or more 'pi' values are < 0 or > 1.")
    return p


def transf_rpbtorbis(xi: ArrayLike, pi: ArrayLike | None = None) -> FloatArray:
    """Point-biserial to biserial correlation; ``pi`` is the proportion in the first group (default 0.5)."""
    x = _arr(xi)
    p = _pi_check(x, pi)
    with np.errstate(all='ignore'):
        return np.asarray(np.sqrt(p * (1 - p)) / _dnorm(_qnorm(p)) * x, np.float64)


def transf_rtorpb(xi: ArrayLike, pi: ArrayLike | None = None) -> FloatArray:
    """Pearson correlation to the point-biserial correlation after dichotomising at proportion ``pi``."""
    x = _arr(xi)
    p = _pi_check(x, pi)
    with np.errstate(all='ignore'):
        return np.asarray(x * _dnorm(_qnorm(p)) / np.sqrt(p * (1 - p)), np.float64)


def transf_rtod(xi: ArrayLike, n1i: ArrayLike | None = None, n2i: ArrayLike | None = None) -> FloatArray:
    """Pearson correlation to SMD for a dichotomised variable (inverse of :func:`transf_dtorbis`)."""
    x = _arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
        n1, n2 = np.ones(1), np.ones(1)
    else:
        n1, n2 = _lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    if np.any(n1 < 0) or np.any(n2 < 0):
        raise ValueError("One or more values specified via the 'n1i' or 'n2i' arguments are negative.")
    with np.errstate(all='ignore'):
        p = n1 / (n1 + n2)
        rpbi = x * _dnorm(_qnorm(p)) / np.sqrt(p * (1 - p))
        return np.asarray(np.sqrt(hi) * rpbi / np.sqrt(1 - rpbi**2), np.float64)


def transf_rpbtod(xi: ArrayLike, n1i: ArrayLike | None = None, n2i: ArrayLike | None = None) -> FloatArray:
    """Point-biserial correlation to SMD (inverse of :func:`transf_dtorpb`)."""
    x = _arr(xi)
    if n1i is None or n2i is None:
        hi: FloatArray | float = 4.0
    else:
        n1, n2 = _lengths(x, n1i, n2i, "Lengths of 'n1i' and 'n2i' do not match the length of 'xi'.")
        mi = n1 + n2 - 2
        hi = mi / n1 + mi / n2
    with np.errstate(all='ignore'):
        return np.asarray(np.sqrt(hi) * x / np.sqrt(1 - x**2), np.float64)


# --------------------------------------------------------------------------
# log odds ratios


def _pc_check(x: FloatArray, pc: ArrayLike) -> FloatArray:
    p = _expand_p(x, pc, 'pc')
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("The control group risk 'pc' must be between 0 and 1.")
    return p


def transf_lnortord(xi: ArrayLike, pc: ArrayLike) -> FloatArray:
    """Log odds ratio to risk difference for control group risk ``pc``."""
    x = _arr(xi)
    p = _pc_check(x, pc)
    with np.errstate(all='ignore'):
        e = np.exp(x)
        return np.asarray(e * p / (1 - p + p * e) - p, np.float64)


def transf_lnortorr(xi: ArrayLike, pc: ArrayLike) -> FloatArray:
    """Log odds ratio to risk ratio (not log risk ratio) for control group risk ``pc``."""
    x = _arr(xi)
    p = _pc_check(x, pc)
    with np.errstate(all='ignore'):
        e = np.exp(x)
        return np.asarray(e / (p * (e - 1) + 1), np.float64)


def transf_lnortod_norm(xi: ArrayLike) -> FloatArray:
    """Log odds ratio to SMD assuming normal distributions (Cox & Snell), ``xi/1.65``."""
    return _arr(xi) / 1.65


def transf_lnortod_logis(xi: ArrayLike) -> FloatArray:
    """Log odds ratio to SMD assuming logistic distributions (Chinn), ``sqrt(3)/pi*xi``."""
    return math.sqrt(3) / math.pi * _arr(xi)


def transf_dtolnor_norm(xi: ArrayLike) -> FloatArray:
    """SMD to log odds ratio assuming normal distributions, ``xi*1.65``."""
    return _arr(xi) * 1.65


def transf_dtolnor_logis(xi: ArrayLike) -> FloatArray:
    """SMD to log odds ratio assuming logistic distributions, ``xi/sqrt(3)*pi``."""
    return _arr(xi) / math.sqrt(3) * math.pi


def transf_lnortortet_pearson(xi: ArrayLike) -> FloatArray:
    """Log odds ratio to tetrachoric correlation (Pearson, 1900), ``cos(pi/(1+sqrt(exp(xi))))``."""
    with np.errstate(over='ignore'):
        return np.cos(math.pi / (1 + np.sqrt(np.exp(_arr(xi)))))


def transf_lnortortet_digby(xi: ArrayLike) -> FloatArray:
    """Log odds ratio to tetrachoric correlation (Digby, 1983), ``(OR^(3/4) - 1)/(OR^(3/4) + 1)``."""
    with np.errstate(over='ignore', invalid='ignore'):
        e = np.exp(_arr(xi)) ** (3 / 4)
        return np.asarray((e - 1) / (e + 1), np.float64)
