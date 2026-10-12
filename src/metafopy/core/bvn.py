"""Bivariate normal probabilities (Genz's ``BVNU``), as used by ``mvtnorm::pmvnorm`` in two dimensions.

Port of ``MVBVU``/``MVBVN`` from mvtnorm's ``src/mvt.f`` (GPL-2), which is
Drezner & Wesolowsky (1990), doi:10.1080/00949659008811236, with Genz's
double-precision modifications; see Genz (2004), doi:10.1023/B:STCO.0000035304.20635.31.
Accurate to about 1e-15, so results agree with R's to rounding error.
"""

from __future__ import annotations

import math

from scipy.special import ndtr

__all__ = ['bvn_lower', 'bvn_rect', 'bvnu']

_TWOPI = 6.283185307179586

# Gauss-Legendre points and weights for n = 6, 12 and 20 (half of each symmetric rule)
_W = (
    (0.1713244923791705, 0.3607615730481384, 0.4679139345726904),
    (0.4717533638651177e-01, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659, 0.2334925365383547, 0.2491470458134029),
    (
        0.1761400713915212e-01,
        0.4060142980038694e-01,
        0.6267204833410906e-01,
        0.8327674157670475e-01,
        0.1019301198172404,
        0.1181945319615184,
        0.1316886384491766,
        0.1420961093183821,
        0.1491729864726037,
        0.1527533871307259,
    ),
)
_X = (
    (-0.9324695142031522, -0.6612093864662647, -0.2386191860831970),
    (-0.9815606342467191, -0.9041172563704750, -0.7699026741943050, -0.5873179542866171, -0.3678314989981802, -0.1252334085114692),
    (
        -0.9931285991850949,
        -0.9639719272779138,
        -0.9122344282513259,
        -0.8391169718222188,
        -0.7463319064601508,
        -0.6360536807265150,
        -0.5108670019508271,
        -0.3737060887154196,
        -0.2277858511416451,
        -0.7652652113349733e-01,
    ),
)


def _phi(x: float) -> float:
    return float(ndtr(x))


def bvnu(sh: float, sk: float, r: float) -> float:
    """``P(X > sh, Y > sk)`` for a standard bivariate normal with correlation ``r``."""
    if abs(r) < 0.3:
        ng = 0
    elif abs(r) < 0.75:
        ng = 1
    else:
        ng = 2
    w = _W[ng]
    x = _X[ng]
    h = sh
    k = sk
    hk = h * k
    bvn = 0.0
    if abs(r) < 0.925:
        hs = (h * h + k * k) / 2
        asr = math.asin(r)
        for wi, xi in zip(w, x, strict=True):
            sn = math.sin(asr * (xi + 1) / 2)
            bvn += wi * math.exp((sn * hk - hs) / (1 - sn * sn))
            sn = math.sin(asr * (-xi + 1) / 2)
            bvn += wi * math.exp((sn * hk - hs) / (1 - sn * sn))
        bvn = bvn * asr / (2 * _TWOPI) + _phi(-h) * _phi(-k)
    else:
        if r < 0:
            k = -k
            hk = -hk
        if abs(r) < 1:
            as_ = (1 - r) * (1 + r)
            a = math.sqrt(as_)
            bs = (h - k) ** 2
            c = (4 - hk) / 8
            d = (12 - hk) / 16
            bvn = a * math.exp(-(bs / as_ + hk) / 2) * (1 - c * (bs - as_) * (1 - d * bs / 5) / 3 + c * d * as_ * as_ / 5)
            if hk > -160:
                b = math.sqrt(bs)
                bvn = bvn - math.exp(-hk / 2) * math.sqrt(_TWOPI) * _phi(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
            a = a / 2
            for wi, xi in zip(w, x, strict=True):
                xs = (a * (xi + 1)) ** 2
                rs = math.sqrt(1 - xs)
                bvn = bvn + a * wi * (math.exp(-bs / (2 * xs) - hk / (1 + rs)) / rs - math.exp(-(bs / xs + hk) / 2) * (1 + c * xs * (1 + d * xs)))
                xs = as_ * (-xi + 1) ** 2 / 4
                rs = math.sqrt(1 - xs)
                bvn = bvn + a * wi * math.exp(-(bs / xs + hk) / 2) * (math.exp(-hk * xs / (2 * (1 + rs) ** 2)) / rs - (1 + c * xs * (1 + d * xs)))
            bvn = -bvn / _TWOPI
        if r > 0:
            bvn = bvn + _phi(-max(h, k))
        else:
            bvn = -bvn
            if k > h:
                if h < 0:
                    bvn = bvn + _phi(k) - _phi(h)
                else:
                    bvn = bvn + _phi(-h) - _phi(-k)
    return bvn


def bvn_rect(lower: tuple[float, float], upper: tuple[float, float], r: float) -> float:
    """``P(lower < (X, Y) <= upper)``; infinite limits are allowed (``MVBVN``)."""
    inf = []
    for lo, up in zip(lower, upper, strict=True):
        if math.isinf(lo) and lo < 0 and math.isinf(up) and up > 0:
            inf.append(-1)
        elif math.isinf(lo) and lo < 0:
            inf.append(0)
        elif math.isinf(up) and up > 0:
            inf.append(1)
        else:
            inf.append(2)
    (l1, l2), (u1, u2) = lower, upper
    match inf:
        case [2, 2]:
            return bvnu(l1, l2, r) - bvnu(u1, l2, r) - bvnu(l1, u2, r) + bvnu(u1, u2, r)
        case [2, 1]:
            return bvnu(l1, l2, r) - bvnu(u1, l2, r)
        case [1, 2]:
            return bvnu(l1, l2, r) - bvnu(l1, u2, r)
        case [2, 0]:
            return bvnu(-u1, -u2, r) - bvnu(-l1, -u2, r)
        case [0, 2]:
            return bvnu(-u1, -u2, r) - bvnu(-u1, -l2, r)
        case [1, 0]:
            return bvnu(l1, -u2, -r)
        case [0, 1]:
            return bvnu(-u1, l2, -r)
        case [1, 1]:
            return bvnu(l1, l2, r)
        case [0, 0]:
            return bvnu(-u1, -u2, r)
        case _:
            # at least one dimension is unbounded on both sides
            if inf[0] == -1 and inf[1] == -1:
                return 1.0
            lo, up = (l2, u2) if inf[0] == -1 else (l1, u1)
            return _phi(up) - _phi(lo)


def bvn_lower(h: float, k: float, r: float) -> float:
    """``P(X <= h, Y <= k)``."""
    return bvnu(-h, -k, r)
