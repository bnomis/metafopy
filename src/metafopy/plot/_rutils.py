"""Small ports of the R functions the plotting code relies on (``pretty``, ``axTicks``, ``formatC``, ``density``, ``supsmu``, ...)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from metafopy.plot._setlab_table import SETLAB

type FloatArray = NDArray[np.float64]

_DBL_EPS = float(np.finfo(float).eps)
_DBL_MIN = float(np.finfo(float).tiny)


# -- axes ------------------------------------------------------------------------------------------------------------


def get_ax(ax: Axes | None) -> Axes:
    """The axes to draw on: ``ax`` or the current axes (creating a figure if there is none)."""
    return plt.gca() if ax is None else ax


def level_to_alpha(level: float) -> float:
    """metafor's ``.level()``: 95 (or 0.95) -> 0.05."""
    if level > 100 or level < 0:
        raise ValueError("Argument 'level' must be between 0 and 100.")
    if level == 0:
        return 1.0
    if level >= 1:
        return (100 - level) / 100
    return 1 - level if level > 0.5 else level


def coladj(light: float) -> str:
    """metafor's ``.coladj()`` for a white background: a grey of level ``255 + round(light * 255)``."""
    v = int(min(255, max(0, 255 + round(light * 255))))
    return f'#{v:02X}{v:02X}{v:02X}'


# -- pretty / axTicks ------------------------------------------------------------------------------------------------


def r_pretty(
    lo: float,
    up: float,
    ndiv: int,
    min_n: int,
    shrink_sml: float,
    high_u_fact: tuple[float, float, float],
    eps_correction: int,
) -> tuple[float, float, int, float]:
    """R's internal ``R_pretty()`` (src/appl/pretty.c). Returns ``(ns, nu, ndiv, unit)``: the pretty bounds are ``ns * unit`` and ``nu * unit``."""
    rounding_eps = 1e-10
    h, h5, f_min = high_u_fact
    dx = up - lo
    if dx == 0 and up == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(up))
        u = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        u *= max(1, ndiv) * _DBL_EPS
        i_small = dx < cell * u * 3
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if ndiv > 1:
            cell /= ndiv
    subsmall = f_min * _DBL_MIN
    if cell < subsmall:
        cell = subsmall
    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    ns_ = 2 * base
    if ns_ - cell < h * (cell - unit):
        unit = ns_
        ns_ = 5 * base
        if ns_ - cell < h5 * (cell - unit):
            unit = ns_
            ns_ = 10 * base
            if ns_ - cell < h * (cell - unit):
                unit = ns_
    ns = math.floor(lo / unit + rounding_eps)
    nu = math.ceil(up / unit - rounding_eps)
    if eps_correction and (eps_correction > 1 or not i_small):
        lo = lo * (1 - _DBL_EPS) if lo != 0.0 else -_DBL_MIN
        up = up * (1 + _DBL_EPS) if up != 0.0 else _DBL_MIN
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2
        ndiv = min_n
    else:
        ndiv = k
    return float(ns), float(nu), ndiv, unit


def pretty(x: ArrayLike, n: int = 5, min_n: int | None = None) -> FloatArray:
    """R's ``pretty(x, n)`` (``pretty.default`` with the default biases)."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    mn = n // 3 if min_n is None else min_n
    ns, nu, ndiv, unit = r_pretty(float(arr.min()), float(arr.max()), n, mn, 0.75, (1.5, 2.75, 2.0**-20), 0)
    z_l, z_u = ns * unit, nu * unit
    res = np.linspace(z_l, z_u, ndiv + 1) if ndiv > 0 else np.array([z_l])
    delta = abs(z_u - z_l) / max(ndiv, 1)
    res[np.abs(res) < 1e-14 * delta] = 0.0
    return np.asarray(res, dtype=np.float64)


def ax_ticks(lo: float, hi: float, nint: int = 5) -> FloatArray:
    """The tick locations R's ``axTicks()`` gives for a linear axis whose user coordinates are ``[lo, hi]`` (``GEPretty``)."""
    ns_f, nu_f, ndiv, unit = r_pretty(lo, hi, nint, 1, 0.25, (0.8, 1.7, 1.125), 2)
    ns, nu = int(ns_f), int(nu_f)
    rounding_eps = 1e-10
    if nu >= ns + 1:
        mod = 0
        if ns * unit < lo - rounding_eps * unit:
            ns += 1
            mod += 1
        if nu > ns + 1 and nu * unit > hi + rounding_eps * unit:
            nu -= 1
            mod += 1
        if mod:
            ndiv = nu - ns
    return np.linspace(ns * unit, nu * unit, ndiv + 1)


def usr_range(lo: float, hi: float) -> tuple[float, float]:
    """The user coordinates R uses for data range ``[lo, hi]`` with the default ``xaxs = "r"`` (4% extension)."""
    d = hi - lo
    if d == 0:
        d = abs(lo) * 0.4 if lo != 0 else 0.4
        return lo - d * 0.1, hi + d * 0.1
    return lo - 0.04 * d, hi + 0.04 * d


# -- formatting -------------------------------------------------------------------------------------------------------


def fmtx1(x: float, digits: int, drop0ifint: bool = False) -> str:
    """metafor's ``fmtx()`` for one number: ``formatC(x, format = "f", digits = digits)`` (optionally dropping trailing zeros)."""
    if x is None or math.isnan(x):
        return 'NA'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    out = f'{x:.{digits}f}'
    if drop0ifint and '.' in out:
        out = out.rstrip('0').rstrip('.')
    return out


def fmtx(x: ArrayLike, digits: int | Sequence[int], drop0ifint: bool = False) -> list[str]:
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return [fmtx1(float(v), digits if isinstance(digits, int) else digits[0], drop0ifint) for v in arr]


def rround(x: float, digits: int = 0) -> float:
    """R (>= 4.0.0) ``round()`` of a scalar."""
    if math.isnan(x) or math.isinf(x):
        return x
    return float(round(x, digits))


# -- labels -----------------------------------------------------------------------------------------------------------


def r_func_name(fn: Callable[..., Any] | bool | None) -> str:
    """The name of the R function a transformation corresponds to (``'FALSE'`` for no transformation)."""
    if fn is None or fn is False:
        return 'FALSE'
    name = getattr(fn, '__name__', '')
    if name.startswith('transf_'):
        return name.replace('_', '.')
    return str(name) if name in ('exp', 'tanh', 'sqrt') else 'other'


def setlab(measure: str | None, transf: Callable[..., Any] | bool | None, atransf: Callable[..., Any] | bool | None, gentype: int, short: bool = False) -> str:
    """metafor's internal ``.setlab()``: axis label for a measure and transformation (``gentype`` 1: outcome, 2: overall, 3: estimate)."""
    base = {1: 'Observed Outcome', 2: 'Overall Estimate', 3: 'Estimate'}[gentype]
    if measure is None or measure not in SETLAB:
        return base
    ent = SETLAB[measure][f'{gentype}{"s" if short else "l"}']
    tn, an = r_func_name(transf), r_func_name(atransf)
    if tn == 'FALSE' and an == 'FALSE':
        return str(ent['none'])
    lab = str(ent['t'] if tn != 'FALSE' else ent['a'])
    # the order of the checks in .setlab: atransf matches first, then transf (which wins)
    if an != 'FALSE':
        lab = str(ent['ex'].get(f'a:{an}', ent['a']))
    if tn != 'FALSE':
        lab = str(ent['ex'].get(f't:{tn}', ent['t'])) if f't:{tn}' in ent['ex'] else (lab if an != 'FALSE' and f'a:{an}' in ent['ex'] else str(ent['t']))
    return lab


# -- distributions and smoothers ---------------------------------------------------------------------------------------


def ppoints(n: int) -> FloatArray:
    a = 3 / 8 if n <= 10 else 0.5
    return np.asarray((np.arange(1, n + 1) - a) / (n + 1 - 2 * a), dtype=np.float64)


def qqnorm_xy(y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """R's ``qqnorm(y, plot.it = FALSE)``: theoretical quantiles (in the order of ``y``) and ``y``."""
    from scipy import stats

    n = y.shape[0]
    order = np.argsort(y, kind='stable')
    rank = np.empty(n, dtype=np.intp)
    rank[order] = np.arange(n)
    x = stats.norm.ppf(ppoints(n))[rank]
    return np.asarray(x, dtype=np.float64), y


def hist_breaks_sturges(x: FloatArray) -> FloatArray:
    """The ``breaks`` of ``hist(x)`` (``"Sturges"``): ``pretty(range(x), ceiling(log2(n)) + 1)``."""
    x = x[np.isfinite(x)]
    nclass = math.ceil(math.log2(x.shape[0]) + 1)
    return pretty(np.array([x.min(), x.max()]), nclass)


def hist_density(x: FloatArray, breaks: FloatArray) -> FloatArray:
    """Densities of ``hist(x, breaks, plot = FALSE)`` (right-closed bins, the first one including its left edge)."""
    x = x[np.isfinite(x)]
    counts = np.zeros(breaks.shape[0] - 1)
    idx = np.searchsorted(breaks, x, side='left') - 1
    idx[x == breaks[0]] = 0
    for i in idx:
        if 0 <= i < counts.shape[0]:
            counts[i] += 1
    return np.asarray(counts / (x.shape[0] * np.diff(breaks)), dtype=np.float64)


def density_r(x: FloatArray, adjust: float = 1.0, n: int = 512) -> tuple[FloatArray, FloatArray]:
    """R's ``density(x, adjust = adjust)`` (gaussian kernel, ``bw.nrd0``, linear binning + FFT) as ``(x, y)``."""
    x = np.asarray(x[np.isfinite(x)], dtype=np.float64)
    nx = x.shape[0]
    hi = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    lo = min(hi, iqr / 1.34) if iqr > 0 or hi > 0 else 1.0
    if lo == 0:
        lo = hi if hi != 0 else (abs(x[0]) if x[0] != 0 else 1.0)
    bw = 0.9 * lo * nx ** (-0.2) * adjust
    cut = 3
    frm, to = float(x.min() - cut * bw), float(x.max() + cut * bw)
    n_pts = max(n, 512)
    if n_pts > 512:
        n_pts = 2 ** math.ceil(math.log2(n_pts))
    lo_b = frm - 4 * bw
    up_b = to + 4 * bw
    # BinDist
    ny = n_pts
    y = np.zeros(2 * n_pts)
    xdelta = (up_b - lo_b) / (n_pts - 1)
    w = 1.0 / nx
    xpos = (x - lo_b) / xdelta
    ix = np.floor(xpos).astype(np.intp)
    fx = xpos - ix
    for i in range(nx):
        ii = int(ix[i])
        if 0 <= ii <= ny - 2:
            y[ii] += w * (1 - fx[i])
            y[ii + 1] += w * fx[i]
        elif ii == -1:
            y[0] += w * fx[i]
        elif ii == ny - 1:
            y[ii] += w * (1 - fx[i])
    kords = np.linspace(0, 2 * (up_b - lo_b), 2 * n_pts)
    kords[n_pts + 1 :] = -kords[n_pts - 1 : 0 : -1]
    kords = np.exp(-0.5 * (kords / bw) ** 2) / (bw * math.sqrt(2 * math.pi))
    kords_fft = np.fft.fft(kords)
    y_fft = np.fft.fft(y)
    kords_c = np.real(np.fft.ifft(np.conj(kords_fft) * y_fft))
    dens = np.maximum(0.0, kords_c[:n_pts])
    xords = np.linspace(lo_b, up_b, n_pts)
    xout = np.linspace(frm, to, n)
    return xout, np.interp(xout, xords, dens)


def _smooth(x: FloatArray, y: FloatArray, w: FloatArray, span: float, iper: int, vsmlsq: float) -> tuple[FloatArray, FloatArray]:
    """Running-lines smoother of Friedman's super smoother (Fortran ``smooth``); returns the smooth and the cross-validated residuals."""
    n = x.shape[0]
    smo = np.zeros(n)
    acvr = np.zeros(n)
    xm = ym = var = cvar = fbw = 0.0
    jper = abs(iper)
    ibw = int(0.5 * span * n + 0.5)
    if ibw < 2:
        ibw = 2
    it = 2 * ibw + 1
    if it > n:
        it = n
    for i in range(1, it + 1):
        j = i
        if jper == 2:
            j = i - ibw - 1
        if j >= 1:
            xti = x[j - 1]
        else:
            j = n + j
            xti = x[j - 1] - 1.0
        wt = w[j - 1]
        fbo = fbw
        fbw = fbw + wt
        if fbw > 0:
            xm = (fbo * xm + wt * xti) / fbw
            ym = (fbo * ym + wt * y[j - 1]) / fbw
        tmp = 0.0
        if fbo > 0:
            tmp = fbw * wt * (xti - xm) / fbo
        var += tmp * (xti - xm)
        cvar += tmp * (y[j - 1] - ym)
    for j in range(1, n + 1):
        out = j - ibw - 1
        inn = j + ibw
        if not (jper != 2 and (out < 1 or inn > n)):
            if out < 1:
                out = n + out
                xto = x[out - 1] - 1.0
                xti = x[inn - 1]
            elif inn > n:
                inn = inn - n
                xti = x[inn - 1] + 1.0
                xto = x[out - 1]
            else:
                xto = x[out - 1]
                xti = x[inn - 1]
            wt = w[out - 1]
            fbo = fbw
            fbw = fbw - wt
            tmp = 0.0
            if fbw > 0:
                tmp = fbo * wt * (xto - xm) / fbw
            var -= tmp * (xto - xm)
            cvar -= tmp * (y[out - 1] - ym)
            if fbw > 0:
                xm = (fbo * xm - wt * xto) / fbw
                ym = (fbo * ym - wt * y[out - 1]) / fbw
            wt = w[inn - 1]
            fbo = fbw
            fbw = fbw + wt
            if fbw > 0:
                xm = (fbo * xm + wt * xti) / fbw
                ym = (fbo * ym + wt * y[inn - 1]) / fbw
            tmp = 0.0
            if fbo > 0:
                tmp = fbw * wt * (xti - xm) / fbo
            var += tmp * (xti - xm)
            cvar += tmp * (y[inn - 1] - ym)
        a = 0.0
        if var > vsmlsq:
            a = cvar / var
        smo[j - 1] = a * (x[j - 1] - xm) + ym
        if iper > 0:
            h = 0.0
            if fbw > 0:
                h = 1.0 / fbw
            if var > vsmlsq:
                h += (x[j - 1] - xm) ** 2 / var
            acvr[j - 1] = 0.0
            a = 1.0 - w[j - 1] * h
            if a > 0:
                acvr[j - 1] = abs(y[j - 1] - smo[j - 1]) / a
            elif j > 1:
                acvr[j - 1] = acvr[j - 2]
    j = 1
    while j <= n:
        j0 = j
        sy = smo[j - 1] * w[j - 1]
        fbw = w[j - 1]
        while j < n and x[j] <= x[j - 1]:
            j += 1
            sy += w[j - 1] * smo[j - 1]
            fbw += w[j - 1]
        if j > j0:
            a = sy / fbw if fbw > 0 else 0.0
            smo[j0 - 1 : j] = a
        j += 1
    return smo, acvr


def supsmu(x: ArrayLike, y: ArrayLike, bass: float = 0.0) -> tuple[FloatArray, FloatArray]:
    """R's ``supsmu(x, y, bass = bass)`` (Friedman's super smoother with cross-validated span); returns ``(x, y)`` for the unique ``x``."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    ord_ = np.lexsort((ya, xa))
    xo, yo = xa[ord_], ya[ord_]
    n = xo.shape[0]
    w = np.ones(n)
    spans = (0.05, 0.2, 0.5)
    big, sml = 1e20, 1e-7
    if not xo[-1] > xo[0]:
        smo = np.full(n, float(np.sum(w * yo) / np.sum(w)))
    else:
        i = n // 4
        j = 3 * i
        i = max(i, 1)
        scale = xo[j - 1] - xo[i - 1]
        while scale <= 0:
            if j < n:
                j += 1
            if i > 1:
                i -= 1
            scale = xo[j - 1] - xo[i - 1]
        vsmlsq = (1e-3 * scale) ** 2
        jper = 1
        sc = np.zeros((n, 7))
        for k in range(3):
            sc[:, 2 * k], sc[:, 6] = _smooth(xo, yo, w, spans[k], jper, vsmlsq)
            sc[:, 2 * k + 1], _ = _smooth(xo, sc[:, 6].copy(), w, spans[1], -jper, vsmlsq)
        for jj in range(n):
            resmin = big
            for k in range(3):
                if sc[jj, 2 * k + 1] < resmin:
                    resmin = sc[jj, 2 * k + 1]
                    sc[jj, 6] = spans[k]
            if 0 < bass <= 10 and resmin < sc[jj, 5] and resmin > 0:
                sc[jj, 6] += (spans[2] - sc[jj, 6]) * max(sml, resmin / sc[jj, 5]) ** (10.0 - bass)
        sc[:, 1], _ = _smooth(xo, sc[:, 6].copy(), w, spans[1], -jper, vsmlsq)
        for jj in range(n):
            if sc[jj, 1] <= spans[0]:
                sc[jj, 1] = spans[0]
            if sc[jj, 1] >= spans[2]:
                sc[jj, 1] = spans[2]
            f = sc[jj, 1] - spans[1]
            if f >= 0:
                f = f / (spans[2] - spans[1])
                sc[jj, 3] = (1.0 - f) * sc[jj, 2] + f * sc[jj, 4]
            else:
                f = -f / (spans[1] - spans[0])
                sc[jj, 3] = (1.0 - f) * sc[jj, 2] + f * sc[jj, 0]
        smo, _ = _smooth(xo, sc[:, 3].copy(), w, spans[0], -jper, vsmlsq)
    dup = np.concatenate([[False], xo[1:] == xo[:-1]])
    return xo[~dup], smo[~dup]


def rformat(values: ArrayLike, digits: int = 7) -> list[str]:
    """``format(x, trim = TRUE, digits)``: a common number of decimals, enough to show every value to ``digits`` significant digits."""
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    rgt = 0
    for v in arr:
        if not math.isfinite(v) or v == 0:
            continue
        sig = f'{abs(v):.{digits - 1}e}'
        mant, exp_s = sig.split('e')
        nsig = len(mant.replace('.', '').rstrip('0')) or 1
        rgt = max(rgt, nsig - int(exp_s) - 1)
    return [('NA' if math.isnan(v) else f'{v:.{rgt}f}') for v in arr]


def log_ticks(lo: float, hi: float) -> FloatArray:
    """``axTicks()`` of a logarithmic axis whose data range is ``[lo, hi]`` (user coordinates ``log10`` of it, extended by 4%)."""
    ulo, uhi = usr_range(math.log10(lo), math.log10(hi))
    dl, dh = 10.0**ulo, 10.0**uhi
    p1, p2 = math.ceil(math.log10(dl)), math.floor(math.log10(dh))
    if p2 <= p1 and dh / dl > 10.0:
        p1, p2 = math.ceil(math.log10(dl) - 0.5), math.floor(math.log10(dh) + 0.5)
    if p2 <= p1:
        ns_f, nu_f, ndiv, unit = r_pretty(dl, dh, 5, 1, 0.25, (0.8, 1.7, 1.125), 2)
        ns, nu = int(ns_f), int(nu_f)
        if nu >= ns + 1:
            mod = 0
            if ns * unit < dl - 1e-10 * unit:
                ns += 1
                mod += 1
            if nu > ns + 1 and nu * unit > dh + 1e-10 * unit:
                nu -= 1
                mod += 1
            if mod:
                ndiv = nu - ns
        at = np.linspace(ns * unit, nu * unit, ndiv + 1)
    else:
        mult = (1.0, 2.0, 5.0) if p2 - p1 <= 2 else ((1.0, 5.0) if p2 - p1 <= 3 else (1.0,))
        at = np.array([m * 10.0**e for e in range(p1 - 1, p2 + 1) for m in mult])
        at = at[(at >= dl * (1 - 1e-10)) & (at <= dh * (1 + 1e-10))]
    return np.asarray(at, dtype=np.float64)
