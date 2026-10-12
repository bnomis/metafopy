"""Effect sizes and sampling variances (metafor's ``escalc``).

A port of ``escalc()`` from metafor 5.2-1 (``R/escalc.r``). Every measure,
``vtype`` option, zero-cell adjustment (``add``/``to``/``drop00``) and the
``addyi``/``addvi``/``onlyo1`` switches follow the R code path, so values agree
with metafor to floating-point rounding. Formulas and their sources are
documented in metafor's help page, ``?escalc``
(https://wviechtb.github.io/metafor/reference/escalc.html), and in
Viechtbauer (2010), doi:10.18637/jss.v036.i03.

Differences from R that follow Python conventions:

- arguments are given by keyword (or positionally in metafor's order) and may
  be array-likes, scalars, or column names in ``data``;
- numeric ``subset``/``include`` values are 0-based row positions;
- the result is a :class:`pandas.DataFrame`; metafor's attributes (``measure``,
  ``slab``, ``ni``, ``digits``, ``yi.names``, ``vi.names``) live in
  ``DataFrame.attrs`` under the same names with ``.`` replaced by ``_``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special, stats

from metafopy._data import make_unique
from metafopy.effects._na import NA as _NA
from metafopy.effects._na import is_na as _is_na
from metafopy.effects._rtet import rtet

__all__ = ['MEASURES', 'cmicalc', 'escalc']

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type Arg = ArrayLike | str | None
type To = Literal['all', 'only0', 'if0all', 'none']

MEASURES: tuple[str, ...] = (
    # 2x2 table measures
    'RR', 'OR', 'PETO', 'RD', 'AS', 'PHI', 'ZPHI', 'YUQ', 'YUY', 'RTET', 'ZTET',
    # 2x2 table transformations to SMDs
    'PBIT', 'OR2D', 'OR2DN', 'OR2DL',
    # 2x2 table measures for matched pairs / pre-post data
    'MPRD', 'MPRR', 'MPOR', 'MPORC', 'MPPETO', 'MPORM',
    # two-group person-time data
    'IRR', 'IRD', 'IRSD',
    # two-group mean/SD measures
    'MD', 'POMPMD', 'SMD', 'SMDH', 'SMD1', 'SMD1H', 'ROM',
    'VR', 'CVR',
    'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL',
    # correlations
    'COR', 'UCOR', 'ZCOR',
    'PCOR', 'ZPCOR', 'SPCOR', 'ZSPCOR',
    'R2', 'ZR2', 'R2F', 'ZR2F',
    # single proportions
    'PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT',
    # single-group person-time data
    'IR', 'IRLN', 'IRS', 'IRFT',
    # single-group means, SDs, CVs
    'MN', 'POMPMN', 'SMN', 'MNLN', 'SDLN', 'CVLN',
    # dependent-samples measures
    'MC', 'POMPMC', 'SMCC', 'SMCR', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC',
    # Cronbach's alpha
    'ARAW', 'AHW', 'ABT',
    # relative excess heterozygosity, common language effect size / AUC
    'REH', 'CLES', 'CLESN', 'AUC', 'AUCN',
    # generic outcomes
    'GEN',
)  # fmt: skip

_VTYPES = ('UB', 'LS', 'LS2', 'LS3', 'HO', 'ST', 'CS', 'AV', 'AV2', 'AVHO', 'H0', 'H0a', 'H0b', 'MAX')

_TWO_BY_TWO = (
    'RR',
    'OR',
    'RD',
    'AS',
    'PETO',
    'PHI',
    'ZPHI',
    'YUQ',
    'YUY',
    'RTET',
    'ZTET',
    'PBIT',
    'OR2D',
    'OR2DN',
    'OR2DL',
    'MPRD',
    'MPRR',
    'MPOR',
    'MPORC',
    'MPPETO',
    'MPORM',
)
_MEANS_2G = ('MD', 'POMPMD', 'SMD', 'SMDH', 'SMD1', 'SMD1H', 'ROM', 'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL', 'VR', 'CVR')
_CHANGE = ('MC', 'POMPMC', 'SMCC', 'SMCR', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC')

_INCOMPLETE = 'Cannot compute outcomes. Check that all of the required information is specified\n  via the appropriate arguments'


# --------------------------------------------------------------------------
# helpers mirroring metafor's hidden utility functions


def cmicalc(mi: ArrayLike, method: bool | str = 'exact') -> FloatArray:
    """Small-sample bias correction factor ``c(m)`` for SMDs (Hedges, 1981, doi:10.3102/10769986006002107).

    ``method`` is ``"exact"`` (gamma-function form, computed on the log scale),
    ``"approx"`` (``1 - 3/(4m - 1)``), or a logical (``True`` = exact, ``False`` = none).
    """
    m = np.asarray(mi, dtype=np.float64)
    if isinstance(method, bool):
        kind = 'exact' if method else 'none'
    else:
        opts = [o for o in ('exact', 'approx') if o.startswith(method)] if method else []
        if not opts:
            raise ValueError("Correction method must be either 'exact' or 'approx'.")
        kind = opts[0]
    with np.errstate(all='ignore'):
        if kind == 'exact':
            out = np.where(m <= 1, _NA, np.exp(special.gammaln(m / 2) - np.log(np.sqrt(m / 2)) - special.gammaln((m - 1) / 2)))
        elif kind == 'approx':
            out = np.where(m <= 1, _NA, 1 - 3 / (4 * m - 1))
        else:
            out = np.ones_like(m)
    # comparisons with NaN give False above, so NaN inputs propagate through the formula
    return np.asarray(out, dtype=np.float64)


def _fcalc(a: float, b: float, g: FloatArray, x: FloatArray) -> FloatArray:
    """Gauss hypergeometric 2F1(a, b; g; x); NaN unless ``g > a + b`` (metafor's ``.Fcalc``)."""
    g, x = np.broadcast_arrays(np.asarray(g, dtype=np.float64), np.asarray(x, dtype=np.float64))
    out = np.full(g.shape, _NA)
    ok = ~np.isnan(g) & ~np.isnan(x) & (g > a + b)
    out[ok] = special.hyp2f1(a, b, g[ok], x[ok])
    # SciPy's hyp2f1 can fail (NaN) for large g with x near 1, where GSL's does not;
    # fall back to Euler's integral representation (DLMF 15.6.1), valid for g > b > 0.
    bad = ok & ~np.isfinite(out)
    for i in np.flatnonzero(bad):
        out[i] = _hyp2f1_euler(a, b, float(g[i]), float(x[i]))
    return out


def _hyp2f1_euler(a: float, b: float, c: float, x: float) -> float:
    val, _ = integrate.quad(lambda t: (1 - x * t) ** (-a), 0, 1, weight='alg', wvar=(b - 1, c - b - 1), epsabs=0, epsrel=1.2e-14, limit=200)
    return math.exp(special.gammaln(c) - special.gammaln(b) - special.gammaln(c - b)) * val


def _wmean(x: FloatArray, w: FloatArray) -> float:
    i = ~(np.isnan(x) | np.isnan(w))
    with np.errstate(all='ignore'):
        return float(np.sum(x[i] * w[i]) / np.sum(w[i]))


def _convp2t(pval: FloatArray | None, df: FloatArray) -> FloatArray | None:
    if pval is None:
        return None
    df = np.where(df < 1, _NA, df)
    pval = np.where(np.abs(pval) > 1, _NA, pval)
    with np.errstate(all='ignore'):
        return np.asarray(np.sign(pval) * stats.t.isf(np.abs(pval) / 2, df), dtype=np.float64)


def _convp2f(pval: FloatArray | None, df1: FloatArray, df2: FloatArray) -> FloatArray | None:
    if pval is None:
        return None
    df1 = np.where(df1 < 1, _NA, df1)
    df2 = np.where(df2 < 1, _NA, df2)
    pval = np.where((pval < 0) | (pval > 1), _NA, pval)
    with np.errstate(all='ignore'):
        return np.asarray(stats.f.isf(pval, df1, df2), dtype=np.float64)


def _replmiss(x: FloatArray | None, y: FloatArray | None) -> FloatArray | None:
    """Replace NaNs in ``x`` by ``y`` (metafor's ``replmiss``); ``None`` means R's ``NULL``."""
    if y is None or y.size == 0:
        return x
    if x is None or x.size == 0:
        x = np.full(y.shape[0], _NA)
    if y.shape[0] == 1:
        y = np.repeat(y, x.shape[0])
    if x.shape[0] != y.shape[0]:
        raise ValueError("Length of 'x' and 'y' are not the same.")
    out = x.copy()
    na = np.isnan(out)
    out[na] = y[na]
    return out


def _rtoz(x: FloatArray) -> FloatArray:
    x = np.clip(x, -1, 1)
    with np.errstate(all='ignore'):
        return np.asarray(np.arctanh(x), dtype=np.float64)


def _expand(x: FloatArray | None, k: int) -> FloatArray | None:
    if x is not None and x.shape[0] == 1:
        return np.repeat(x, k)
    return x


def _equal_length(*xs: FloatArray | None) -> bool:
    ks = {x.shape[0] for x in xs if x is not None and x.shape[0] > 0}
    return len(ks) <= 1


def _all_specified(*xs: FloatArray | None) -> bool:
    return all(x is not None and x.shape[0] > 0 for x in xs)


def _anyna(cond: FloatArray | BoolArray, *operands: FloatArray | None) -> bool:
    """``any(cond, na.rm=TRUE)``: ignore positions where an operand is NaN."""
    mask = np.asarray(cond, dtype=np.bool_)
    for op in operands:
        if op is not None:
            mask = mask & ~np.isnan(op)
    return bool(mask.any())


def _vtypes(vtype: str | Sequence[str], k: int, allowed: Sequence[str], msg: str) -> NDArray[np.str_]:
    vt = np.array([vtype] if isinstance(vtype, str) else list(vtype), dtype=np.str_)
    if vt.shape[0] == 1:
        vt = np.repeat(vt, k)
    if not np.isin(vt, allowed).all():
        raise ValueError(f"For this outcome measure, 'vtype' must be {msg}.")
    return vt


def _subset(x: FloatArray | None, subset: BoolArray | None) -> FloatArray | None:
    if x is None or subset is None:
        return x
    if x.shape[0] != subset.shape[0]:
        raise ValueError(f'Element is not of length {subset.shape[0]}.')
    return x[subset]


def _chksubset(x: ArrayLike | str | None, k: int, data: pd.DataFrame | None, name: str, stoponk0: bool = True) -> BoolArray | None:
    if x is None:
        return None
    if isinstance(x, str):
        if data is None:
            raise ValueError(f'a string {name} needs data to evaluate against')
        values = np.asarray(data.eval(x))
    else:
        values = np.asarray(x)
    if values.ndim == 0:
        values = values[None]
    if values.shape[0] == 0:
        raise ValueError(f"Argument '{name}' is of length 0.")
    if values.dtype == np.bool_ or values.dtype == object:
        if values.shape[0] != k:
            raise ValueError(f"Length of the '{name}' argument ({values.shape[0]}) is not of length k = {k}.")
        mask = np.array([bool(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else False for v in values], dtype=np.bool_)
    else:
        idx = values.astype(np.float64)
        idx = idx[~np.isnan(idx)].round().astype(np.intp)
        if np.any(idx >= k) or np.any(idx < -k):
            raise ValueError(f"Argument '{name}' includes values outside 0..{k - 1}.")
        mask = np.zeros(k, dtype=np.bool_)
        mask[idx] = True
    if stoponk0 and not mask.any():
        raise ValueError('Stopped because k = 0 after subsetting.')
    return mask


def _chkyisize(yi: FloatArray, measure: str, cutoff: float | None) -> None:
    addtxt = None
    if cutoff is None:
        cut = math.nan
        if measure in ('SMD', 'SMDH', 'SMD1', 'SMD1H'):
            cut = 2.0
            addtxt = 'Maybe SEs were used in place of SDs to compute them?'
    else:
        cut = abs(float(cutoff))
    if not math.isnan(cut) and _anyna(np.abs(yi) > cut, yi):
        txt = f'One or more {measure} values are unusually large (i.e., abs(yi) > {cut:g}).'
        if addtxt:
            txt += f'\n{addtxt}'
        warnings.warn(txt, stacklevel=3)


# --------------------------------------------------------------------------


class _Inputs:
    """Resolves escalc arguments (arrays, scalars or column names) the way ``.getx`` does."""

    def __init__(self, data: pd.DataFrame | None, values: dict[str, Arg]) -> None:
        self.data = data
        self.values = values

    def get(self, name: str) -> FloatArray | None:
        value = self.values.get(name)
        if value is None:
            return None
        if isinstance(value, str):
            if self.data is None:
                raise ValueError(f'{name}={value!r} names a column, but no data was given')
            if value not in self.data.columns:
                raise KeyError(f'column {value!r} (for {name}) not found in data')
            value = self.data[value].to_numpy()
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise TypeError(f"The object/variable specified for the '{name}' argument is not numeric.") from err
        if arr.ndim == 0:
            arr = arr[None]
        if arr.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional, got shape {arr.shape}')
        # missing values supplied by the user are R's NA
        return np.where(np.isnan(arr), _NA, arr)


def escalc(
    measure: str | None = None,
    ai: Arg = None,
    bi: Arg = None,
    ci: Arg = None,
    di: Arg = None,
    n1i: Arg = None,
    n2i: Arg = None,
    x1i: Arg = None,
    x2i: Arg = None,
    t1i: Arg = None,
    t2i: Arg = None,
    m1i: Arg = None,
    m2i: Arg = None,
    sd1i: Arg = None,
    sd2i: Arg = None,
    xi: Arg = None,
    mi: Arg = None,
    ri: Arg = None,
    ti: Arg = None,
    fi: Arg = None,
    pi: Arg = None,
    sdi: Arg = None,
    r2i: Arg = None,
    mini: Arg = None,
    maxi: Arg = None,
    ni: Arg = None,
    yi: Arg = None,
    vi: Arg = None,
    sei: Arg = None,
    *,
    data: pd.DataFrame | Mapping[str, ArrayLike] | None = None,
    slab: ArrayLike | str | None = None,
    flip: ArrayLike | str | None = None,
    subset: ArrayLike | str | None = None,
    include: ArrayLike | str | None = None,
    add: float | None = None,
    to: To = 'only0',
    drop00: bool = False,
    vtype: str | Sequence[str] = 'LS',
    correct: bool | str = True,
    cutoff: float | None = None,
    var_names: Sequence[str] = ('yi', 'vi'),
    add_measure: bool = False,
    append: bool = True,
    replace: bool = True,
    digits: int | Sequence[int] | None = None,
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> pd.DataFrame:
    """Calculate effect sizes ``yi`` and sampling variances ``vi`` (metafor's ``escalc``).

    See https://wviechtb.github.io/metafor/reference/escalc.html for the
    measures, their required arguments and the ``vtype`` options. ``add``
    defaults to 1/2, except for measures where metafor uses 0 (AS, PHI, ZPHI,
    RTET, ZTET, IRSD, PAS, PFT, IRS, IRFT). ``onlyo1``, ``addyi`` and
    ``addvi`` correspond to metafor's ``...`` options of the same names.

    Returns a :class:`pandas.DataFrame` with columns ``var_names`` (appended
    to ``data`` when given and ``append=True``); ``attrs['measure']``,
    ``attrs['slab']`` and ``attrs['ni']`` carry metafor's attributes.
    """
    if measure is None and yi is None:
        raise ValueError("Must specify an effect size or outcome measure via the 'measure' argument.")
    if yi is not None and measure is None:
        measure = 'GEN'
    if not isinstance(measure, str):
        raise TypeError("The 'measure' argument must be a character string.")
    if measure in ('HR', 'HD'):
        raise NotImplementedError(f'measure={measure!r} is reserved in metafor but not implemented there either')
    if measure not in MEASURES:
        raise ValueError("Unknown 'measure' specified.")
    if to not in ('all', 'only0', 'if0all', 'none'):
        raise ValueError("Unknown 'to' argument specified.")
    if not isinstance(drop00, bool):
        raise TypeError("Unknown 'drop00' argument specified.")
    vtype_list = [vtype] if isinstance(vtype, str) else list(vtype)
    if any(v not in _VTYPES for v in vtype_list if v is not None):
        raise ValueError("Unknown 'vtype' argument specified.")

    var_names = list(var_names)
    if add_measure:
        if len(var_names) == 2:
            var_names.append('measure')
        if len(var_names) != 3:
            raise ValueError("Argument 'var_names' must be of length 2 or 3.")
    else:
        if len(var_names) == 3:
            var_names = var_names[:2]
        if len(var_names) != 2:
            raise ValueError("Argument 'var_names' must be of length 2.")

    if add is None:
        add = 0.0 if measure in ('AS', 'PHI', 'ZPHI', 'RTET', 'ZTET', 'IRSD', 'PAS', 'PFT', 'IRS', 'IRFT') else 0.5

    if measure in ('ROM', 'VR', 'CVR', 'SDLN', 'CVLN', 'ROMC', 'CVRC') and not isinstance(correct, bool):
        raise TypeError("Argument 'correct' must be a logical for this measure.")

    has_data = data is not None
    if data is not None and not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    assert data is None or isinstance(data, pd.DataFrame)

    args: dict[str, Arg] = dict(
        ai=ai, bi=bi, ci=ci, di=di, n1i=n1i, n2i=n2i, x1i=x1i, x2i=x2i, t1i=t1i, t2i=t2i, m1i=m1i, m2i=m2i, sd1i=sd1i, sd2i=sd2i,
        xi=xi, mi=mi, ri=ri, ti=ti, fi=fi, pi=pi, sdi=sdi, r2i=r2i, mini=mini, maxi=maxi, ni=ni, yi=yi, vi=vi, sei=sei,
    )  # fmt: skip
    inp = _Inputs(data, args)

    with np.errstate(all='ignore'):
        if yi is None:
            out = _compute(measure, inp, data, subset, add, to, drop00, vtype, correct, onlyo1, addyi, addvi)
        else:
            out = _generic(inp, data, subset, slab)
    yv, vv, ni_u, k_all, sub = out.yi, out.vi, out.ni_u, out.k_all, out.subset
    if out.slab is not None and slab is None:
        slab = out.slab

    yv = np.asarray(yv, dtype=np.float64).ravel().copy()
    vv = np.asarray(vv, dtype=np.float64).ravel().copy()
    k = yv.shape[0]

    is_inf = np.isinf(yv) | np.isinf(vv)
    if is_inf.any():
        warnings.warn("Some 'yi' and/or 'vi' values equal to +-Inf. Recoded to NAs.", stacklevel=2)
        yv[is_inf] = _NA
        vv[is_inf] = _NA
    is_nan = (np.isnan(yv) & ~_is_na(yv)) | (np.isnan(vv) & ~_is_na(vv))
    yv[is_nan] = _NA
    vv[is_nan] = _NA

    _chkyisize(yv, measure, cutoff)

    with np.errstate(all='ignore'):
        vv[vv < 0] = _NA

    labels: list[str] | None = None
    if slab is not None:
        if isinstance(slab, str):
            if data is None:
                raise ValueError(f'slab={slab!r} names a column, but no data was given')
            raw = data[slab].tolist()
        else:
            raw = np.asarray(slab, dtype=object).ravel().tolist()
        if len(raw) != k_all:
            raise ValueError(f"Length of the 'slab' argument ({len(raw)}) does not correspond to the size of the dataset ({k_all}).")
        if sub is not None:
            raw = [s for s, keep in zip(raw, sub, strict=True) if keep]
        if any(s is None or (isinstance(s, float) and math.isnan(s)) for s in raw):
            raise ValueError('NAs in study labels.')
        labels = [str(s) for s in raw]
        if len(set(labels)) != len(labels):
            labels = list(_make_unique_dot1(labels))

    inc = _chksubset(include, k_all, data, 'include', stoponk0=False)
    if inc is None:
        inc = np.ones(k_all, dtype=np.bool_)
    if sub is None:
        sub = np.ones(k_all, dtype=np.bool_)
    inc = inc[sub]

    flip_arr = _flip(flip, k_all, data)
    flip_arr = flip_arr[sub]
    with np.errstate(all='ignore'):
        yv[inc] = flip_arr[inc] * yv[inc]
        vv[inc] = flip_arr[inc] ** 2 * vv[inc]
    # R's NA is a signalling NaN; hand back ordinary quiet NaNs
    yv[np.isnan(yv)] = math.nan
    vv[np.isnan(vv)] = math.nan

    if has_data and append:
        assert data is not None
        dat = data.loc[sub].reset_index(drop=True).copy() if not sub.all() else data.reset_index(drop=True).copy()
        cols: list[tuple[str, FloatArray | None]] = [(var_names[0], yv), (var_names[1], vv)]
        for name, values in cols:
            assert values is not None
            if replace or name not in dat.columns:
                repl = np.ones(k, dtype=np.bool_)
            else:
                repl = dat[name].isna().to_numpy()
            if name not in dat.columns:
                dat[name] = _NA
            col = dat[name].to_numpy(dtype=np.float64, copy=True)
            col[inc & repl] = values[inc & repl]
            dat[name] = col
        if add_measure:
            mname = var_names[2]
            if replace or mname not in dat.columns:
                mrepl = np.ones(k, dtype=np.bool_)
            else:
                mvals = dat[mname]
                mrepl = (mvals.isna() | (mvals == '')).to_numpy()
            mcol = dat[mname].astype(object).to_numpy(copy=True) if mname in dat.columns else np.full(k, None, dtype=object)
            mcol[~np.isnan(yv) & inc & mrepl] = measure
            dat[mname] = mcol
    else:
        dat = pd.DataFrame({var_names[0]: np.where(inc, yv, _NA), var_names[1]: np.where(inc, vv, _NA)})
        if add_measure:
            mcol = np.full(k, None, dtype=object)
            mcol[~np.isnan(yv) & inc] = measure
            dat[var_names[2]] = mcol

    if add_measure:
        dat[var_names[2]] = dat[var_names[2]].where(dat[var_names[2]].notna(), '')

    prev_attrs = dict(data.attrs) if has_data and data is not None else {}
    dat.attrs = {}
    dat.attrs['measure'] = measure
    dat.attrs['slab'] = labels
    dat.attrs['ni'] = None if ni_u is None else np.where(inc, ni_u, _NA)
    dat.attrs['digits'] = digits
    yi_names = [var_names[0], *[n for n in prev_attrs.get('yi_names', []) if n != var_names[0]]]
    vi_names = [var_names[1], *[n for n in prev_attrs.get('vi_names', []) if n != var_names[1]]]
    dat.attrs['yi_names'] = [n for n in yi_names if n in dat.columns]
    dat.attrs['vi_names'] = [n for n in vi_names if n in dat.columns]
    return dat


def _make_unique_dot1(labels: Sequence[str]) -> tuple[str, ...]:
    """metafor's ``.make.unique``: every repeated label gets ``.1``, ``.2``, ... (including the first)."""
    counts: dict[str, int] = {}
    for s in labels:
        counts[s] = counts.get(s, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for s in labels:
        if counts[s] == 1:
            out.append(s)
        else:
            seen[s] = seen.get(s, 0) + 1
            out.append(f'{s}.{seen[s]}')
    return make_unique(out)


def _flip(flip: ArrayLike | str | None, k_all: int, data: pd.DataFrame | None) -> FloatArray:
    if flip is None:
        return np.ones(k_all)
    if isinstance(flip, str):
        if data is None:
            raise ValueError(f'flip={flip!r} names a column, but no data was given')
        values = data[flip].to_numpy()
    else:
        values = np.asarray(flip)
    if values.ndim == 0:
        values = values[None]
    if values.dtype == np.bool_ or values.dtype == object:
        if values.shape[0] == 1:
            values = np.repeat(values, k_all)
        out = np.where([v is True or v is np.True_ for v in values], -1.0, 1.0)
    else:
        out = values.astype(np.float64)
    if out.shape[0] != k_all:
        raise ValueError(f"Length of the 'flip' argument ({out.shape[0]}) does not correspond to the size of the dataset ({k_all}).")
    return np.asarray(out, dtype=np.float64)


class _Out:
    __slots__ = ('k_all', 'ni_u', 'slab', 'subset', 'vi', 'yi')

    def __init__(self, yi: FloatArray, vi: FloatArray, ni_u: FloatArray | None, k_all: int, subset: BoolArray | None, slab: ArrayLike | None = None) -> None:
        self.yi = yi
        self.vi = vi
        self.ni_u = ni_u
        self.k_all = k_all
        self.subset = subset
        self.slab = slab


def _generic(inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, slab: ArrayLike | str | None) -> _Out:
    yi = inp.get('yi')
    assert yi is not None
    vi = inp.get('vi')
    sei = inp.get('sei')
    ni = inp.get('ni')
    if vi is None:
        if sei is None:
            raise ValueError("Must specify the 'vi' or 'sei' argument.")
        vi = sei**2
    if not _equal_length(yi, vi, ni):
        raise ValueError('Supplied data vectors are not all of the same length.')
    k_all = yi.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    return _Out(_req(_subset(yi, sub)), _req(_subset(vi, sub)), _subset(ni, sub), k_all, sub)


def _req(x: FloatArray | None) -> FloatArray:
    assert x is not None
    return x


def _compute(
    measure: str,
    inp: _Inputs,
    data: pd.DataFrame | None,
    subset: ArrayLike | str | None,
    add: float,
    to: To,
    drop00: bool,
    vtype: str | Sequence[str],
    correct: bool | str,
    onlyo1: bool,
    addyi: bool,
    addvi: bool,
) -> _Out:
    if measure in _TWO_BY_TWO:
        return _two_by_two(measure, inp, data, subset, add, to, drop00, vtype, onlyo1, addyi, addvi)
    if measure in ('IRR', 'IRD', 'IRSD'):
        return _two_group_rates(measure, inp, data, subset, add, to, drop00, addyi, addvi)
    if measure in _MEANS_2G:
        return _two_group_means(measure, inp, data, subset, vtype, correct)
    if measure in ('CLES', 'AUC'):
        return _cles(inp, data, subset, vtype)
    if measure in ('CLESN', 'AUCN'):
        return _clesn(inp, data, subset, vtype)
    if measure in ('COR', 'UCOR', 'ZCOR'):
        return _cor(measure, inp, data, subset, vtype)
    if measure in ('PCOR', 'ZPCOR', 'SPCOR', 'ZSPCOR'):
        return _pcor(measure, inp, data, subset, vtype)
    if measure in ('R2', 'ZR2', 'R2F', 'ZR2F'):
        return _r2(measure, inp, data, subset, vtype)
    if measure in ('PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT'):
        return _proportions(measure, inp, data, subset, add, to, vtype, addyi, addvi)
    if measure in ('IR', 'IRLN', 'IRS', 'IRFT'):
        return _rates(measure, inp, data, subset, add, to, addyi, addvi)
    if measure in ('MN', 'POMPMN', 'SMN', 'MNLN', 'SDLN', 'CVLN'):
        return _means(measure, inp, data, subset, vtype, correct)
    if measure in _CHANGE:
        return _change(measure, inp, data, subset, vtype, correct)
    if measure in ('ARAW', 'AHW', 'ABT'):
        return _alpha(measure, inp, data, subset)
    if measure == 'REH':
        return _reh(inp, data, subset)
    raise ValueError("Unknown 'measure' specified.")  # pragma: no cover


def _two_by_two(
    measure: str,
    inp: _Inputs,
    data: pd.DataFrame | None,
    subset: ArrayLike | str | None,
    add: float,
    to: To,
    drop00: bool,
    vtype: str | Sequence[str],
    onlyo1: bool,
    addyi: bool,
    addvi: bool,
) -> _Out:
    ai = inp.get('ai')
    bi = inp.get('bi')
    ci = inp.get('ci')
    di = inp.get('di')
    n1i = inp.get('n1i')
    n2i = inp.get('n2i')
    ri = inp.get('ri')
    pi = inp.get('pi')

    if not _equal_length(ai, bi, ci, di, n1i, n2i, ri, pi):
        raise ValueError('Supplied data vectors are not all of the same length.')

    if n1i is not None and ai is not None and bi is not None and _anyna(n1i != ai + bi, n1i, ai, bi):
        raise ValueError("One or more 'n1i' values are not equal to 'ai + bi'.")
    if n2i is not None and ci is not None and di is not None and _anyna(n2i != ci + di, n2i, ci, di):
        raise ValueError("One or more 'n2i' values are not equal to 'ci + di'.")

    if n1i is not None and ai is not None:
        bi = _replmiss(bi, n1i - ai)
    if n2i is not None and ci is not None:
        di = _replmiss(di, n2i - ci)

    if not _all_specified(ai, bi, ci, di):
        raise ValueError(f'{_INCOMPLETE} (i.e., ai, bi, ci, di or ai, n1i, ci, n2i).')
    assert ai is not None
    assert bi is not None
    assert ci is not None
    assert di is not None

    if measure == 'MPORM' and not (_all_specified(ri) or _all_specified(pi)):
        raise ValueError("Need to specify also argument 'ri' (and/or 'pi') for this measure.")

    k_all = ai.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    if sub is not None:
        ai, bi, ci, di = _req(_subset(ai, sub)), _req(_subset(bi, sub)), _req(_subset(ci, sub)), _req(_subset(di, sub))
        ri = _subset(ri, sub)
        pi = _subset(pi, sub)

    n1i = ai + bi
    n2i = ci + di

    if _anyna(np.concatenate([ai > n1i, ci > n2i]), np.concatenate([ai, ci]), np.concatenate([n1i, n2i])):
        raise ValueError('One or more event counts are larger than the corresponding group sizes.')
    allc = np.concatenate([ai, bi, ci, di])
    if _anyna(allc < 0, allc):
        raise ValueError('One or more counts are negative.')
    alln = np.concatenate([n1i, n2i])
    if _anyna(alln < 0, alln):
        raise ValueError('One or more group sizes are negative.')
    if measure == 'MPORM' and ri is not None and _anyna(np.abs(ri) > 1, ri):
        raise ValueError('One or more correlations are > 1 or < -1.')
    if measure == 'MPORM' and pi is not None and _anyna((pi < 0) | (pi > 1), pi):
        raise ValueError('One or more proportions are > 1 or < 0.')

    ni_u = ai + bi + ci + di
    if measure == 'MPORM':
        ni_u = np.round(ni_u / 2)

    k = ai.shape[0]

    ai = ai.copy()
    bi = bi.copy()
    ci = ci.copy()
    di = di.copy()

    if drop00:
        id00 = ((ai == 0) & (ci == 0)) | ((bi == 0) & (di == 0))
        ai[id00] = _NA
        bi[id00] = _NA
        ci[id00] = _NA
        di[id00] = _NA

    ai_u, bi_u, ci_u, di_u = ai.copy(), bi.copy(), ci.copy(), di.copy()
    n1i_u = ai + bi
    n2i_u = ci + di

    if to == 'all':
        ai = ai + add
        ci = ci + add
        if not onlyo1:
            bi = bi + add
            di = di + add

    if to in ('only0', 'if0all'):
        id0 = (ai == 0) | (ci == 0) | (bi == 0) | (di == 0)

    if to == 'only0':
        ai[id0] += add
        ci[id0] += add
        if not onlyo1:
            bi[id0] += add
            di[id0] += add

    if to == 'if0all' and id0.any():
        ai = ai + add
        ci = ci + add
        if not onlyo1:
            bi = bi + add
            di = di + add

    n1i = ai + bi
    n2i = ci + di
    ni = n1i + n2i
    if measure == 'MPORM':
        ni = np.round(ni / 2)

    p1i_u = ai_u / n1i_u
    p2i_u = ci_u / n2i_u
    p1i = ai / n1i
    p2i = ci / n2i

    if addvi:
        mnwp1i = _wmean(p1i, n1i)
        mnwp2i = _wmean(p2i, n2i)
        ppi = (ai + ci) / (n1i + n2i)
    else:
        mnwp1i_u = _wmean(p1i_u, n1i_u)
        mnwp2i_u = _wmean(p2i_u, n2i_u)
        ppi_u = (ai_u + ci_u) / (n1i_u + n2i_u)

    yi = np.full(k, _NA)
    vi = np.full(k, _NA)

    if measure == 'RR':
        yi = np.log(p1i) - np.log(p2i) if addyi else np.log(p1i_u) - np.log(p2i_u)
        vt = _vtypes(vtype, k, ('LS', 'AV', 'H0'), "either 'LS' or 'AV'")
        if addvi:
            ls = 1 / ai - 1 / n1i + 1 / ci - 1 / n2i
            av = (1 - mnwp1i) / (mnwp1i * n1i) + (1 - mnwp2i) / (mnwp2i * n2i)
            h0 = (1 - ppi) / ppi * (1 / n1i + 1 / n2i)
        else:
            ls = 1 / ai_u - 1 / n1i_u + 1 / ci_u - 1 / n2i_u
            av = (1 - mnwp1i_u) / (mnwp1i_u * n1i_u) + (1 - mnwp2i_u) / (mnwp2i_u * n2i_u)
            h0 = (1 - ppi_u) / ppi_u * (1 / n1i_u + 1 / n2i_u)
        vi = _pick(vt, {'LS': ls, 'AV': av, 'H0': h0})

    if measure in ('OR', 'OR2D', 'OR2DN', 'OR2DL', 'MPORM'):
        yi = np.log(p1i / (1 - p1i)) - np.log(p2i / (1 - p2i)) if addyi else np.log(p1i_u / (1 - p1i_u)) - np.log(p2i_u / (1 - p2i_u))
        vt = _vtypes(vtype, k, ('LS', 'AV', 'H0'), "either 'LS' or 'AV'")
        if addvi:
            ls = 1 / ai + 1 / bi + 1 / ci + 1 / di
            av = 1 / (mnwp1i * (1 - mnwp1i) * n1i) + 1 / (mnwp2i * (1 - mnwp2i) * n2i)
            h0 = 1 / (ppi * (1 - ppi)) * (1 / n1i + 1 / n2i)
        else:
            ls = 1 / ai_u + 1 / bi_u + 1 / ci_u + 1 / di_u
            # metafor uses the adjusted group sizes here (R/escalc.r, OR with vtype="AV" and addvi=FALSE)
            av = 1 / (mnwp1i_u * (1 - mnwp1i_u) * n1i) + 1 / (mnwp2i_u * (1 - mnwp2i_u) * n2i)
            h0 = 1 / (ppi_u * (1 - ppi_u)) * (1 / n1i_u + 1 / n2i_u)
        vi = _pick(vt, {'LS': ls, 'AV': av, 'H0': h0})

    if measure == 'RD':
        yi = p1i - p2i if addyi else p1i_u - p2i_u
        vt = _vtypes(vtype, k, ('UB', 'LS', 'AV', 'H0'), "either 'UB', 'LS', or 'AV'")
        if addvi:
            ub = p1i * (1 - p1i) / (n1i - 1) + p2i * (1 - p2i) / (n2i - 1)
            ls = p1i * (1 - p1i) / n1i + p2i * (1 - p2i) / n2i
            av = mnwp1i * (1 - mnwp1i) / n1i + mnwp2i * (1 - mnwp2i) / n2i
            h0 = (1 / n1i + 1 / n2i) * ppi * (1 - ppi)
        else:
            ub = p1i_u * (1 - p1i_u) / (n1i_u - 1) + p2i_u * (1 - p2i_u) / (n2i_u - 1)
            ls = p1i_u * (1 - p1i_u) / n1i_u + p2i_u * (1 - p2i_u) / n2i_u
            av = mnwp1i_u * (1 - mnwp1i_u) / n1i_u + mnwp2i_u * (1 - mnwp2i_u) / n2i_u
            h0 = (1 / n1i_u + 1 / n2i_u) * ppi_u * (1 - ppi_u)
        vi = _pick(vt, {'UB': ub, 'LS': ls, 'AV': av, 'H0': h0})

    if measure == 'PETO':
        xt = ai + ci
        yt = bi + di
        Ei = xt * n1i / ni
        Vi = xt * yt * (n1i / ni) * (n2i / ni) / (ni - 1)
        yi = (ai - Ei) / Vi
        vi = 1 / Vi

    if measure == 'AS':
        yi = np.arcsin(np.sqrt(p1i)) - np.arcsin(np.sqrt(p2i))
        vi = 1 / (4 * n1i) + 1 / (4 * n2i)

    if measure in ('PHI', 'ZPHI'):
        yi = (ai * di - bi * ci) / np.sqrt((ai + bi) * (ci + di) * (ai + ci) * (bi + di))
        vt = _vtypes(vtype, k, ('ST', 'CS', 'LS'), "either 'ST', 'CS', or 'LS'")
        q1i = 1 - p1i
        q2i = 1 - p2i
        pi1_ = (ai + bi) / ni
        pi2_ = (ci + di) / ni
        pi_1 = (ai + ci) / ni
        pi_2 = (bi + di) / ni
        n1, n2, p1, p2 = n1i, n2i, p1i, p2i
        st = ((n1 + n2) ** 2 * (4 * n1**3 * p1**2 * p2 * q1i**2 * q2i
                                + 4 * n2**3 * p1 * p2**2 * q1i * q2i**2
                                + n1 * n2**2 * p2 * q2i * (p2 * q1i + p1 * q2i) * (p2 * q1i + p1 * (4 * q1i + q2i))
                                + n1**2 * n2 * p1 * q1i * (p2 * q1i + p1 * q2i) * (p1 * q2i + p2 * (q1i + 4 * q2i)))) / (
            4 * (ai + ci) ** 3 * (bi + di) ** 3
        )  # fmt: skip
        cs = (
            1
            / ni
            * (
                1
                - yi**2
                + yi * (1 + 1 / 2 * yi**2) * (pi1_ - pi2_) * (pi_1 - pi_2) / np.sqrt(pi1_ * pi2_ * pi_1 * pi_2)
                - 3 / 4 * yi**2 * ((pi1_ - pi2_) ** 2 / (pi1_ * pi2_) + (pi_1 - pi_2) ** 2 / (pi_1 * pi_2))
            )
        )
        vi = _pick(vt, {'ST': st, 'CS': cs, 'LS': cs})

    if measure == 'YUQ':
        yi = (ai / bi) / (ci / di)
        yi = (yi - 1) / (yi + 1)
        vi = 1 / 4 * (1 - yi**2) ** 2 * (1 / ai + 1 / bi + 1 / ci + 1 / di)

    if measure == 'YUY':
        yi = (ai / bi) / (ci / di)
        yi = (np.sqrt(yi) - 1) / (np.sqrt(yi) + 1)
        vi = 1 / 16 * (1 - yi**2) ** 2 * (1 / ai + 1 / bi + 1 / ci + 1 / di)

    if measure in ('RTET', 'ZTET'):
        yi = np.full(k, _NA)
        vi = np.full(k, _NA)
        for i in range(k):
            if np.isnan(ai[i]) or np.isnan(bi[i]) or np.isnan(ci[i]) or np.isnan(di[i]):
                continue
            yi[i], vi[i] = rtet(float(ai[i]), float(bi[i]), float(ci[i]), float(di[i]))

    if measure in ('ZPHI', 'ZTET'):
        vi = vi / (1 - np.where(yi**2 > 1, 1, yi**2)) ** 2
        yi = _rtoz(yi)

    if measure == 'PBIT':
        z1i = special.ndtri(p1i)
        z2i = special.ndtri(p2i)
        yi = z1i - z2i
        vi = 2 * math.pi * p1i * (1 - p1i) * np.exp(z1i**2) / n1i + 2 * math.pi * p2i * (1 - p2i) * np.exp(z2i**2) / n2i

    if measure in ('OR2D', 'OR2DL'):
        yi = math.sqrt(3) / math.pi * yi
        vi = 3 / math.pi**2 * vi

    if measure == 'OR2DN':
        yi = yi / 1.65
        vi = vi / 1.65**2

    if measure in ('MPRD', 'MPRR', 'MPOR'):
        pi12 = bi / ni
        pi21 = ci / ni
        pi1_ = (ai + bi) / ni
        pi_1 = (ai + ci) / ni
        if measure == 'MPRD':
            yi = pi1_ - pi_1
            vi = pi12 * (1 - pi12) / ni + 2 * pi12 * pi21 / ni + pi21 * (1 - pi21) / ni
        if measure == 'MPRR':
            yi = np.log(pi1_) - np.log(pi_1)
            vi = (pi12 + pi21) / (ni * pi1_ * pi_1)
        if measure == 'MPOR':
            yi = np.log(pi1_ / (1 - pi1_)) - np.log(pi_1 / (1 - pi_1))
            vi = (pi12 * (1 - pi12) + pi21 * (1 - pi21) + 2 * pi12 * pi21) / (ni * pi1_ * (1 - pi1_) * pi_1 * (1 - pi_1))

    if measure == 'MPORM':
        if pi is not None:
            ai_p = pi * (ai_u + bi_u)
            bi_p = ai_u - ai_p
            ci_p = ci_u - ai_p
            di_p = bi_u - ci_u + ai_p
            ri_p = (ai_p * di_p - bi_p * ci_p) / np.sqrt((ai_p + bi_p) * (ci_p + di_p) * (ai_p + ci_p) * (bi_p + di_p))
            ri_p = np.where((ri_p < -1) | (ri_p > 1), _NA, ri_p)
            ri = _replmiss(ri, ri_p)
        assert ri is not None
        if addvi:
            si = (ri * np.sqrt(ai * bi * ci * di) + (ai * bi)) / ni
            deltai = ni**2 * (ni * si - ai * bi) / (ai * bi * ci * di)
            vi = vi - 2 * deltai / ni
        else:
            si_u = (ri * np.sqrt(ai_u * bi_u * ci_u * di_u) + (ai_u * bi_u)) / ni_u
            # metafor uses the adjusted bi in this term (R/escalc.r, MPORM with addvi=FALSE)
            deltai_u = ni_u**2 * (ni_u * si_u - ai_u * bi) / (ai_u * bi_u * ci_u * di_u)
            vi = vi - 2 * deltai_u / ni_u

    if measure == 'MPORC':
        yi = np.log(bi) - np.log(ci)
        vi = 1 / bi + 1 / ci

    if measure == 'MPPETO':
        Ei = (bi + ci) / 2
        Vi = (bi + ci) / 4
        yi = (bi - Ei) / Vi
        vi = 1 / Vi

    return _Out(yi, vi, ni_u, k_all, sub)


def _pick(vt: NDArray[np.str_], options: dict[str, FloatArray | float]) -> FloatArray:
    out = np.full(vt.shape[0], _NA)
    for key, values in options.items():
        sel = vt == key
        if sel.any():
            out[sel] = np.broadcast_to(np.asarray(values, dtype=np.float64), vt.shape)[sel]
    return out


def _two_group_rates(
    measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, add: float, to: To, drop00: bool, addyi: bool, addvi: bool
) -> _Out:
    x1i, x2i, t1i, t2i = inp.get('x1i'), inp.get('x2i'), inp.get('t1i'), inp.get('t2i')
    if not _all_specified(x1i, x2i, t1i, t2i):
        raise ValueError(f'{_INCOMPLETE} (i.e., x1i, x2i, t1i, t2i).')
    if not _equal_length(x1i, x2i, t1i, t2i):
        raise ValueError('Supplied data vectors are not all of the same length.')
    assert x1i is not None
    assert x2i is not None
    assert t1i is not None
    assert t2i is not None
    k_all = x1i.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    x1i, x2i, t1i, t2i = (_req(_subset(v, sub)) for v in (x1i, x2i, t1i, t2i))

    xs = np.concatenate([x1i, x2i])
    if _anyna(xs < 0, xs):
        raise ValueError('One or more counts are negative.')
    ts = np.concatenate([t1i, t2i])
    if _anyna(ts <= 0, ts):
        raise ValueError('One or more person-times are <= 0.')

    ni_u = t1i + t2i
    x1i = x1i.copy()
    x2i = x2i.copy()

    if drop00:
        id00 = (x1i == 0) & (x2i == 0)
        x1i[id00] = _NA
        x2i[id00] = _NA

    x1i_u = x1i.copy()
    x2i_u = x2i.copy()

    if to == 'all':
        x1i = x1i + add
        x2i = x2i + add
    if to in ('only0', 'if0all'):
        id0 = (x1i == 0) | (x2i == 0)
    if to == 'only0':
        x1i[id0] += add
        x2i[id0] += add
    if to == 'if0all' and id0.any():
        x1i = x1i + add
        x2i = x2i + add

    ir1i_u = x1i_u / t1i
    ir2i_u = x2i_u / t2i
    ir1i = x1i / t1i
    ir2i = x2i / t2i

    if measure == 'IRR':
        yi = np.log(ir1i) - np.log(ir2i) if addyi else np.log(ir1i_u) - np.log(ir2i_u)
        vi = 1 / x1i + 1 / x2i if addvi else 1 / x1i_u + 1 / x2i_u
    elif measure == 'IRD':
        yi = ir1i - ir2i if addyi else ir1i_u - ir2i_u
        vi = ir1i / t1i + ir2i / t2i if addvi else ir1i_u / t1i + ir2i_u / t2i
    else:
        yi = np.sqrt(ir1i) - np.sqrt(ir2i) if addyi else np.sqrt(ir1i_u) - np.sqrt(ir2i_u)
        vi = 1 / (4 * t1i) + 1 / (4 * t2i)
    return _Out(yi, vi, ni_u, k_all, sub)


def _two_group_means(
    measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str], correct: bool | str
) -> _Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    n1i, n2i = inp.get('n1i'), inp.get('n2i')
    di, ti, pi, ri = inp.get('di'), inp.get('ti'), inp.get('pi'), inp.get('ri')
    mini, maxi = inp.get('mini'), inp.get('maxi')

    if measure in ('SMD', 'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL'):
        if not _equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ri):
            raise ValueError('Supplied data vectors are not all of the same length.')
        if n1i is not None and n2i is not None:
            ti = _replmiss(ti, _convp2t(pi, n1i + n2i - 2))
            if ti is not None:
                di = _replmiss(di, ti * np.sqrt(1 / n1i + 1 / n2i))
            mi_ = n1i + n2i - 2
            hi = mi_ / n1i + mi_ / n2i
            if ri is not None:
                di = _replmiss(di, np.sqrt(hi) * ri / np.sqrt(1 - ri**2))
        if di is not None:
            has = ~np.isnan(di)
            m1i = _fill(m1i, has, di, di.shape[0])
            m2i = _fill(m2i, has, 0.0, di.shape[0])
            sd1i = _fill(sd1i, has, 1.0, di.shape[0])
            sd2i = _fill(sd2i, has, 1.0, di.shape[0])
        if not _all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i (and di, ti, pi)).')

    if measure in ('MD', 'SMDH', 'SMD1H', 'ROM', 'CVR'):
        if not _all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i).')
        if not _equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise ValueError('Supplied data vectors are not all of the same length.')

    if measure == 'POMPMD':
        if not _all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi).')
        if not _equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi):
            raise ValueError('Supplied data vectors are not all of the same length.')
        assert m1i is not None
        assert m2i is not None
        assert sd1i is not None
        assert sd2i is not None
        assert mini is not None
        assert maxi is not None
        m1i = 100 * (m1i - mini) / (maxi - mini)
        m2i = 100 * (m2i - mini) / (maxi - mini)
        sd1i = 100 * sd1i / (maxi - mini)
        sd2i = 100 * sd2i / (maxi - mini)

    if measure == 'VR':
        if not _all_specified(sd1i, sd2i, n1i, n2i):
            raise ValueError(f'{_INCOMPLETE} (i.e., sd1i, sd2i, n1i, n2i).')
        if not _equal_length(sd1i, sd2i, n1i, n2i):
            raise ValueError('Supplied data vectors are not all of the same length.')

    if measure == 'SMD1':
        if not _all_specified(m1i, m2i, sd2i, n1i, n2i):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd2i, n1i, n2i).')
        if not _equal_length(m1i, m2i, sd2i, n1i, n2i):
            raise ValueError('Supplied data vectors are not all of the same length.')

    assert n1i is not None
    assert n2i is not None
    k_all = n1i.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    m1i, m2i, sd1i, sd2i = (_subset(v, sub) for v in (m1i, m2i, sd1i, sd2i))
    n1i, n2i = _req(_subset(n1i, sub)), _req(_subset(n2i, sub))

    sds = np.concatenate([v for v in (sd1i, sd2i) if v is not None]) if (sd1i is not None or sd2i is not None) else np.empty(0)
    if _anyna(sds < 0, sds):
        raise ValueError('One or more standard deviations are negative.')
    ns = np.concatenate([n1i, n2i])
    if _anyna(ns <= 0, ns):
        raise ValueError('One or more group sizes are <= 0.')

    ni_u = n1i + n2i
    k = n1i.shape[0]
    ni = ni_u

    nan = np.full(k, _NA)
    m1 = nan if m1i is None else m1i
    m2 = nan if m2i is None else m2i
    sd1 = nan if sd1i is None else sd1i
    sd2 = nan if sd2i is None else sd2i

    if measure in ('SMD1', 'SMD1H'):
        mi = n2i - 1
        sdpi = sd2
        npi = n2i
    else:
        mi = ni - 2
        sdpi = np.sqrt(((n1i - 1) * sd1**2 + (n2i - 1) * sd2**2) / mi)
        npi = ni

    d = (m1 - m2) / sdpi

    yi = nan
    vi = nan

    if measure in ('MD', 'POMPMD'):
        yi = m1 - m2
        vt = _vtypes(vtype, k, ('LS', 'UB', 'HO'), "either 'LS', 'UB', or 'HO'")
        ub = sd1**2 / n1i + sd2**2 / n2i
        vi = _pick(vt, {'UB': ub, 'LS': ub, 'HO': sdpi**2 * (1 / n1i + 1 / n2i)})

    if measure in ('SMD', 'SMD1'):
        cmi = cmicalc(mi, method=correct)
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2', 'LS3', 'UB', 'AV', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        mnwyi = _wmean(yi, ni)
        vi = _pick(
            vt,
            {
                'LS': 1 / n1i + 1 / n2i + yi**2 / (2 * npi),
                'LS2': cmi**2 * (1 / n1i + 1 / n2i + d**2 / (2 * npi)),
                'LS3': 1 / n1i + 1 / n2i + yi**2 / (2 * (npi - 3.94)),
                'UB': 1 / n1i + 1 / n2i + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'AV': 1 / n1i + 1 / n2i + mnwyi**2 / (2 * npi),
                'H0': np.where(mi > 2, cmi**2 * (1 / n1i + 1 / n2i) * mi / (mi - 2), _NA),
            },
        )

    if measure == 'SMDH':
        cmi = cmicalc(mi, method=correct)
        sdpi = np.sqrt((sd1**2 + sd2**2) / 2)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2', 'LS3'), "either 'LS' or 'LS2'")
        vi = _pick(
            vt,
            {
                'LS': yi**2 * (sd1**4 / (n1i - 1) + sd2**4 / (n2i - 1)) / (8 * sdpi**4) + (sd1**2 / (n1i - 1) + sd2**2 / (n2i - 1)) / sdpi**2,
                'LS2': sd1**2 / (n1i * sdpi**2) + sd2**2 / (n2i * sdpi**2) + yi**2 / (8 * sdpi**4) * (sd1**4 / n1i + sd2**4 / n2i),
                'LS3': sd1**2 / (n1i * sdpi**2) + sd2**2 / (n2i * sdpi**2) + yi**2 / (8 * sdpi**4) * (sd1**4 / (n1i - 1) + sd2**4 / (n2i - 1)),
            },
        )

    if measure == 'SMD1H':
        cmi = cmicalc(mi, method=correct)
        yi = cmi * d
        vi = (sd1**2 / sd2**2) / (n1i - 1) + 1 / (n2i - 1) + yi**2 / (2 * (n2i - 1))

    if measure == 'ROM':
        if correct:
            yi = np.log(m1 / m2) + 1 / 2 * (sd1**2 / (n1i * m1**2) - sd2**2 / (n2i * m2**2))
        else:
            yi = np.log(m1 / m2)
        vt = _vtypes(vtype, k, ('LS', 'HO', 'AV', 'AVHO', 'LS2'), "either 'LS', 'LS2', 'HO', 'AV', or 'AVHO'")
        cv1 = sd1 / m1
        cv2 = sd2 / m2
        mn1wcvi = _wmean(cv1, n1i)
        mn2wcvi = _wmean(cv2, n2i)
        not_na = ~(np.isnan(n1i) | np.isnan(n2i) | np.isnan(cv1) | np.isnan(cv2))
        mnwcvi = (np.sum(n1i[not_na] * cv1[not_na]) + np.sum(n2i[not_na] * cv2[not_na])) / np.sum((n1i + n2i)[not_na])
        vi = _pick(
            vt,
            {
                'LS': sd1**2 / (n1i * m1**2) + sd2**2 / (n2i * m2**2),
                'HO': sdpi**2 / (n1i * m1**2) + sdpi**2 / (n2i * m2**2),
                'AV': mn1wcvi**2 / n1i + mn2wcvi**2 / n2i,
                'AVHO': mnwcvi**2 * (1 / n1i + 1 / n2i),
                'LS2': sd1**2 / (n1i * m1**2) + sd2**2 / (n2i * m2**2) + sd1**4 / (2 * n1i**2 * m1**4) + sd2**4 / (2 * n2i**2 * m2**4),
            },
        )

    if measure in ('RPB', 'ZPB'):
        hi = mi / n1i + mi / n2i
        yi = d / np.sqrt(d**2 + hi)
        vt = _vtypes(vtype, k, ('ST', 'CS', 'LS'), "either 'ST', 'CS', or 'LS'")
        st = hi**2 / (hi + d**2) ** 3 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))
        cs = (1 - yi**2) ** 2 * (ni * yi**2 / (4 * n1i * n2i) + (2 - 3 * yi**2) / (2 * ni))
        vi = _pick(vt, {'ST': st, 'LS': st, 'CS': cs})

    if measure in ('RBIS', 'ZBIS'):
        hi = mi / n1i + mi / n2i
        yi = d / np.sqrt(d**2 + hi)
        p1i = n1i / ni
        p2i = n2i / ni
        zi = stats.norm.isf(p1i)
        fzi = stats.norm.pdf(zi)
        yi = np.sqrt(p1i * p2i) / fzi * yi
        yi_t = np.where(np.abs(yi) > 1, np.sign(yi), yi)
        vi = 1 / (ni - 1) * (p1i * p2i / fzi**2 - (3 / 2 + (1 - p1i * zi / fzi) * (1 + p2i * zi / fzi)) * yi_t**2 + yi_t**4)

    if measure in ('ZPB', 'ZBIS'):
        vi = vi / (1 - np.where(yi**2 > 1, 1, yi**2)) ** 2
        yi = _rtoz(yi)

    if measure in ('D2OR', 'D2ORL'):
        yi = math.pi / math.sqrt(3) * d
        vi = math.pi**2 / 3 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))

    if measure == 'D2ORN':
        yi = 1.65 * d
        vi = 1.65**2 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))

    if measure == 'VR':
        if correct:
            yi = np.log(sd1 / sd2) + 1 / (2 * (n1i - 1)) - 1 / (2 * (n2i - 1))
        else:
            yi = np.log(sd1 / sd2)
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(vt, {'LS': 1 / (2 * (n1i - 1)) + 1 / (2 * (n2i - 1)), 'LS2': 1 / 2 * (n1i / (n1i - 1) ** 2 + n2i / (n2i - 1) ** 2)})

    if measure == 'CVR':
        if correct:
            yi = np.log(sd1 / m1) - np.log(sd2 / m2) + 1 / (2 * (n1i - 1)) - 1 / (2 * (n2i - 1)) + 1 / 2 * (sd2**2 / (n2i * m2**2) - sd1**2 / (n1i * m1**2))
        else:
            yi = np.log(sd1 / m1) - np.log(sd2 / m2)
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(
            vt,
            {
                'LS': 1 / (2 * (n1i - 1)) + sd1**2 / (n1i * m1**2) + 1 / (2 * (n2i - 1)) + sd2**2 / (n2i * m2**2),
                'LS2': sd1**2 / (n1i * m1**2)
                + sd2**2 / (n2i * m2**2)
                + sd1**4 / (2 * n1i**2 * m1**4)
                + sd2**4 / (2 * n2i**2 * m2**4)
                + 1 / 2 * (n1i / (n1i - 1) ** 2 + n2i / (n2i - 1) ** 2),
            },
        )

    return _Out(yi, vi, ni_u, k_all, sub)


def _fill(x: FloatArray | None, where: BoolArray, value: FloatArray | float, k: int) -> FloatArray:
    out = np.full(k, _NA) if x is None else x.copy()
    if isinstance(value, np.ndarray):
        out[where] = value[where]
    else:
        out[where] = value
    return out


def _cles(inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str]) -> _Out:
    ai, n1i, n2i, mi = inp.get('ai'), inp.get('n1i'), inp.get('n2i'), inp.get('mi')
    if not _all_specified(ai, n1i, n2i):
        raise ValueError(f'{_INCOMPLETE} (i.e., ai, n1i, n2i).')
    if not _equal_length(ai, n1i, n2i, mi):
        raise ValueError('Supplied data vectors are not all of the same length.')
    assert ai is not None
    assert n1i is not None
    assert n2i is not None
    mi = np.zeros(ai.shape[0]) if mi is None else np.where(np.isnan(mi), 0.0, mi)
    k_all = ai.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    ai, n1i, n2i, mi = (_req(_subset(v, sub)) for v in (ai, n1i, n2i, mi))

    ns = np.concatenate([n1i, n2i])
    if _anyna(ns <= 0, ns):
        raise ValueError('One or more group sizes are <= 0.')
    if _anyna((ai < 0) | (ai > 1), ai):
        raise ValueError('One or more AUC values are < 0 or > 1.')
    if _anyna((mi < 0) | (mi > 1), mi):
        raise ValueError("One or more 'mi' values are < 0 or > 1.")

    ni = n1i + n2i
    k = ai.shape[0]
    yi = ai
    vt = _vtypes(vtype, k, ('LS', 'LS2', 'H0', 'MAX'), "'LS' or 'LS2'")
    navgi = (n1i + n2i) / 2
    q0 = ai * (1 - ai)
    q1 = ai / (2 - ai)
    q2 = 2 * ai**2 / (1 + ai)
    vi = _pick(
        vt,
        {
            'LS': q0 / ((n1i - 1) * (n2i - 1)) * (2 * navgi - 1 - (3 * navgi - 3) / ((2 - ai) * (1 + ai))),
            'LS2': (q0 - mi / 4 + (n1i - 1) * (q1 - ai**2) + (n2i - 1) * (q2 - ai**2)) / ((n1i - 1) * (n2i - 1)),
            'H0': (n1i + n2i + 1) / (12 * n1i * n2i),
            'MAX': q0 / (np.minimum(n1i, n2i) - 1),
        },
    )
    return _Out(yi, vi, ni, k_all, sub)


def _clesn(inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str]) -> _Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    n1i, n2i, di, ti, pi, ai = inp.get('n1i'), inp.get('n2i'), inp.get('di'), inp.get('ti'), inp.get('pi'), inp.get('ai')
    if not _equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ai):
        raise ValueError('Supplied data vectors are not all of the same length.')
    if not _all_specified(n1i, n2i):
        raise ValueError(f'{_INCOMPLETE}.')
    assert n1i is not None
    assert n2i is not None
    k_all = max(v.shape[0] for v in (m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ai) if v is not None)
    vt_list = [vtype] if isinstance(vtype, str) else list(vtype)
    vt = np.array(vt_list * k_all if len(vt_list) == 1 else vt_list, dtype=object)

    if sd1i is None or sd2i is None:
        sd1i = np.full(k_all, _NA)
        sd2i = np.full(k_all, _NA)

    ti = _replmiss(ti, _convp2t(pi, n1i + n2i - 2))
    if ti is not None:
        di = _replmiss(di, ti * np.sqrt(1 / n1i + 1 / n2i))
    if di is not None:
        vt[~np.isnan(di)] = 'HO'

    ho = vt == 'HO'
    sdpi = np.where(ho, np.sqrt(((n1i - 1) * sd1i**2 + (n2i - 1) * sd2i**2) / (n1i + n2i - 2)), np.sqrt((sd1i**2 + sd2i**2) / 2))
    if m1i is not None and m2i is not None:
        di = _replmiss(di, (m1i - m2i) / sdpi)
    if di is not None:
        ai = _replmiss(ai, stats.norm.cdf(di / math.sqrt(2)))
    if ai is not None:
        di = _replmiss(di, special.ndtri(ai) * math.sqrt(2))
    if ai is None or di is None:
        raise ValueError(f'{_INCOMPLETE}.')

    k_all = ai.shape[0]
    sdsmiss = np.isnan(sd1i) | np.isnan(sd2i)
    sd1i = np.where(sdsmiss, 1.0, sd1i)
    sd2i = np.where(sdsmiss, 1.0, sd2i)
    vt[sdsmiss] = 'HO'

    sub = _chksubset(subset, k_all, data, 'subset')
    if sub is not None:
        vt = vt[sub]
    ai, di, sd1i, sd2i, n1i, n2i = (_req(_subset(v, sub)) for v in (ai, di, sd1i, sd2i, n1i, n2i))

    sds = np.concatenate([sd1i, sd2i])
    if _anyna(sds < 0, sds):
        raise ValueError('One or more standard deviations are negative.')
    ns = np.concatenate([n1i, n2i])
    if _anyna(ns <= 0, ns):
        raise ValueError('One or more group sizes are <= 0.')
    if _anyna((ai < 0) | (ai > 1), ai):
        raise ValueError('One or more AUC values are < 0 or > 1.')

    ni_u = n1i + n2i
    k = ai.shape[0]
    yi = ai
    vts = _vtypes([str(v) for v in vt], k, ('LS', 'LS2', 'LS3', 'HO', 'H0'), "'LS' or 'HO'")
    vri = sd1i**2 / (sd1i**2 + sd2i**2)
    e = np.exp(-(di**2) / 2)
    vi = _pick(
        vts,
        {
            'LS': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 1) + di**2 * (1 - vri) ** 2 / (n2i - 1) + 4 * vri / (n1i - 1) + 4 * (1 - vri) / (n2i - 1)),
            'LS2': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 0) + di**2 * (1 - vri) ** 2 / (n2i - 0) + 4 * vri / (n1i - 0) + 4 * (1 - vri) / (n2i - 0)),
            'LS3': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 1) + di**2 * (1 - vri) ** 2 / (n2i - 1) + 4 * vri / (n1i - 0) + 4 * (1 - vri) / (n2i - 0)),
            'HO': e / (4 * math.pi) * (1 / n1i + 1 / n2i + di**2 / (2 * (n1i + n2i))),
            'H0': 1 / (8 * math.pi) * (4 * vri / (n1i - 1) + 4 * (1 - vri) / (n2i - 1)),
        },
    )
    return _Out(yi, vi, ni_u, k_all, sub)


def _cor(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str]) -> _Out:
    ri, ni, ti, pi = inp.get('ri'), inp.get('ni'), inp.get('ti'), inp.get('pi')
    if not _equal_length(ri, ni, ti, pi):
        raise ValueError('Supplied data vectors are not all of the same length.')
    if ni is not None:
        ti = _replmiss(ti, _convp2t(pi, ni - 2))
        if ti is not None:
            ri = _replmiss(ri, ti / np.sqrt(ti**2 + ni - 2))
    if not _all_specified(ri, ni):
        raise ValueError(f'{_INCOMPLETE} (i.e., ri, ni (and ti, pi)).')
    assert ri is not None
    assert ni is not None
    k_all = ri.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    ri, ni = _req(_subset(ri, sub)), _req(_subset(ni, sub))

    if _anyna(np.abs(ri) > 1, ri):
        raise ValueError('One or more correlations are > 1 or < -1.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')
    vtl = [vtype] if isinstance(vtype, str) else list(vtype)
    if measure != 'UCOR' and 'UB' in vtl:
        raise ValueError("Use of vtype='UB' only permitted when measure='UCOR'.")
    if measure == 'UCOR' and _anyna(ni <= 4, ni):
        warnings.warn('Cannot compute the bias-corrected correlation coefficient when ni <= 4.', stacklevel=4)
    if measure == 'ZCOR' and _anyna(ni <= 3, ni):
        warnings.warn('Cannot estimate the sampling variance when ni <= 3.', stacklevel=4)

    k = ri.shape[0]
    yi = ri
    if measure == 'UCOR':
        yi = ri * _fcalc(1 / 2, 1 / 2, (ni - 2) / 2, 1 - ri**2)

    if measure in ('COR', 'UCOR'):
        allowed = ('LS', 'UB', 'AV', 'H0', 'H0a', 'H0b') if measure == 'COR' else ('LS', 'UB', 'AV')
        msg = "either 'LS', 'UB', 'AV', or 'H0'" if measure == 'COR' else "either 'LS', 'UB', or 'AV'"
        vt = _vtypes(vtype, k, allowed, msg)
        mnwyi = _wmean(yi, ni)
        vi = _pick(
            vt,
            {
                'LS': (1 - yi**2) ** 2 / (ni - 1),
                'UB': yi**2 - (1 - (ni - 3) / (ni - 2) * (1 - ri**2) * _fcalc(1, 1, ni / 2, 1 - ri**2)),
                'AV': (1 - mnwyi**2) ** 2 / (ni - 1),
                'H0': (1 - yi**2) / (ni - 2),
                'H0a': (1 - yi**2) / (ni - 2),
                'H0b': 1 / ni,
            },
        )
    else:
        yi = _rtoz(ri)
        vi = 1 / (ni - 3)

    vi = np.where(ni <= 3, _NA, vi)
    return _Out(yi, vi, ni, k_all, sub)


def _pcor(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str]) -> _Out:
    ri, ti, mi, ni, pi, r2i = inp.get('ri'), inp.get('ti'), inp.get('mi'), inp.get('ni'), inp.get('pi'), inp.get('r2i')
    if not _equal_length(ri, ti, mi, ni, pi, r2i):
        raise ValueError('Supplied data vectors are not all of the same length.')
    if ni is not None and mi is not None:
        ti = _replmiss(ti, _convp2t(pi, ni - mi - 1))
        if ti is not None:
            if measure in ('PCOR', 'ZPCOR'):
                ri = _replmiss(ri, ti / np.sqrt(ti**2 + ni - mi - 1))
            elif r2i is not None:
                ri = _replmiss(ri, ti * np.sqrt(1 - r2i) / np.sqrt(ni - mi - 1))
    if measure in ('PCOR', 'ZPCOR') and not _all_specified(ri, mi, ni):
        raise ValueError(f'{_INCOMPLETE} (i.e., ri, ti, mi, ni (and pi)).')
    if measure in ('SPCOR', 'ZSPCOR') and not _all_specified(ri, mi, ni, r2i):
        raise ValueError(f'{_INCOMPLETE} (i.e., ri, ti, mi, ni, r2i (and pi)).')
    assert ri is not None
    assert mi is not None
    assert ni is not None
    k_all = ri.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    ri, mi, ni = _req(_subset(ri, sub)), _req(_subset(mi, sub)), _req(_subset(ni, sub))
    r2i = _subset(r2i, sub)

    if _anyna(np.abs(ri) > 1, ri):
        raise ValueError('One or more (semi-)partial correlations are > 1 or < -1.')
    if measure in ('SPCOR', 'ZSPCOR') and r2i is not None and _anyna((r2i > 1) | (r2i < 0), r2i):
        raise ValueError('One or more R^2 values are > 1 or < 0.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')
    if _anyna(mi <= 0, mi):
        raise ValueError('One or more mi values are <= 0.')
    if _anyna(ni - mi - 1 <= 0, ni, mi):
        raise ValueError('One or more dfs are <= 0.')

    k = ri.shape[0]
    if measure == 'PCOR':
        yi = ri
        vt = _vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        mnwyi = _wmean(yi, ni)
        vi = _pick(vt, {'LS': (1 - yi**2) ** 2 / (ni - mi), 'AV': (1 - mnwyi**2) ** 2 / (ni - mi)})
    elif measure == 'ZPCOR':
        yi = _rtoz(ri)
        vi = 1 / (ni - mi - 2)
    else:
        assert r2i is not None
        yi = ri
        vt = _vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        mnwyi = _wmean(yi, ni)
        vi = _pick(
            vt,
            {
                'LS': (r2i**2 - 2 * r2i + (r2i - yi**2) + 1 - (r2i - yi**2) ** 2) / ni,
                'AV': (r2i**2 - 2 * r2i + (r2i - mnwyi**2) + 1 - (r2i - mnwyi**2) ** 2) / ni,
            },
        )
        if measure == 'ZSPCOR':
            vi = vi / (1 - np.where(yi**2 > 1, 1, yi**2)) ** 2
            yi = _rtoz(yi)
    return _Out(yi, vi, ni, k_all, sub)


def _r2(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str]) -> _Out:
    r2i, mi, ni, fi, pi = inp.get('r2i'), inp.get('mi'), inp.get('ni'), inp.get('fi'), inp.get('pi')
    if not _equal_length(r2i, mi, ni):
        raise ValueError('Supplied data vectors are not all of the same length.')
    if mi is not None and ni is not None:
        fi = _replmiss(fi, _convp2f(pi, mi, ni - mi - 1))
        if fi is not None:
            r2i = _replmiss(r2i, mi * fi / (mi * fi + (ni - mi - 1)))
    if not _all_specified(r2i, mi, ni):
        raise ValueError(f'{_INCOMPLETE} (i.e., r2i, mi, ni (and fi, pi)).')
    assert r2i is not None
    assert mi is not None
    assert ni is not None
    k_all = r2i.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    r2i, mi, ni = _req(_subset(r2i, sub)), _req(_subset(mi, sub)), _req(_subset(ni, sub))

    if _anyna((r2i > 1) | (r2i < 0), r2i):
        raise ValueError('One or more R^2 values are > 1 or < 0.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')
    if _anyna(mi <= 0, mi):
        raise ValueError('One or more mi values are <= 0.')
    if _anyna(ni - mi - 1 <= 0, ni, mi):
        raise ValueError('One or more dfs are <= 0.')

    k = r2i.shape[0]
    if measure == 'R2':
        yi = r2i
        vt = _vtypes(vtype, k, ('LS', 'AV', 'LS2', 'AV2'), "either 'LS' or 'AV'")
        mnwyi = _wmean(yi, ni)
        vi = _pick(
            vt,
            {
                'LS': 4 * yi * (1 - yi) ** 2 / ni,
                'AV': 4 * mnwyi * (1 - mnwyi) ** 2 / ni,
                'LS2': 4 * yi * (1 - yi) ** 2 * (ni - mi - 1) ** 2 / ((ni**2 - 1) * (ni + 3)),
                'AV2': 4 * mnwyi * (1 - mnwyi) ** 2 * (ni - mi - 1) ** 2 / ((ni**2 - 1) * (ni + 3)),
            },
        )
    elif measure == 'ZR2':
        vtl = [vtype] if isinstance(vtype, str) else list(vtype)
        if not all(v == 'LS' for v in vtl):
            raise ValueError("For this outcome measure, 'vtype' must be 'LS'.")
        yi = _rtoz(np.sqrt(r2i))
        vi = 1 / ni
    else:
        yi = r2i
        _vtypes(vtype, k, ('LS',), "'LS'")
        vi = np.full(k, _NA)
        R2s = np.linspace(0.0001, 0.9999, 10**5)
        dx = R2s[1] - R2s[0]
        for i in range(k):
            if np.isnan(r2i[i]) or np.isnan(mi[i]) or np.isnan(ni[i]):
                continue
            Fval = (ni[i] - mi[i] - 1) / mi[i] * r2i[i] / (1 - r2i[i])
            ncps = (mi[i] + ni[i] - mi[i]) * R2s / (1 - R2s)
            den = stats.ncf.pdf(Fval, mi[i], ni[i] - mi[i] - 1, ncps)
            den = den / (np.sum(np.diff(R2s) * (den[1:] + den[:-1])) / 2)
            vi[i] = np.sum(dx * R2s**2 * den) - np.sum(dx * R2s * den) ** 2
        if measure == 'ZR2F':
            vi = vi / (1 - yi**2) ** 2
            yi = _rtoz(np.sqrt(r2i))
    return _Out(yi, vi, ni, k_all, sub)


def _proportions(
    measure: str,
    inp: _Inputs,
    data: pd.DataFrame | None,
    subset: ArrayLike | str | None,
    add: float,
    to: To,
    vtype: str | Sequence[str],
    addyi: bool,
    addvi: bool,
) -> _Out:
    xi, mi, ni = inp.get('xi'), inp.get('mi'), inp.get('ni')
    if not _equal_length(xi, mi, ni):
        raise ValueError('Supplied data vectors are not all of the same length.')
    if ni is not None and xi is not None and mi is not None and _anyna(ni != xi + mi, ni, xi, mi):
        raise ValueError("One or more 'ni' values are not equal to 'xi + mi'.")
    if ni is not None and xi is not None:
        mi = _replmiss(mi, ni - xi)
    if not _all_specified(xi, mi):
        raise ValueError(f'{_INCOMPLETE} (i.e., xi, mi or xi, ni).')
    assert xi is not None
    assert mi is not None
    k_all = xi.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    xi, mi = _req(_subset(xi, sub)).copy(), _req(_subset(mi, sub)).copy()

    ni = xi + mi
    if _anyna(xi > ni, xi, ni):
        raise ValueError('One or more event counts are larger than the corresponding group sizes.')
    c = np.concatenate([xi, mi])
    if _anyna(c < 0, c):
        raise ValueError('One or more counts are negative.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more group sizes are <= 0.')

    ni_u = ni
    k = xi.shape[0]
    xi_u = xi.copy()
    mi_u = mi.copy()

    if to == 'all':
        xi = xi + add
        mi = mi + add
    if to in ('only0', 'if0all'):
        id0 = (xi == 0) | (mi == 0)
    if to == 'only0':
        xi[id0] += add
        mi[id0] += add
    if to == 'if0all' and id0.any():
        xi = xi + add
        mi = mi + add

    ni = xi + mi
    pri_u = xi_u / ni_u
    pri = xi / ni

    if measure in ('PR', 'PLN', 'PLO'):
        if addvi:
            mnwpri = _wmean(pri, ni)
        else:
            mnwpri_u = _wmean(pri_u, ni_u)

    if measure == 'PR':
        yi = pri if addyi else pri_u
        vt = _vtypes(vtype, k, ('LS', 'UB', 'AV'), "either 'LS', 'UB', or 'AV'")
        if addvi:
            vi = _pick(vt, {'LS': pri * (1 - pri) / ni, 'UB': pri * (1 - pri) / (ni - 1), 'AV': mnwpri * (1 - mnwpri) / ni})
        else:
            vi = _pick(vt, {'LS': pri_u * (1 - pri_u) / ni_u, 'UB': pri_u * (1 - pri_u) / (ni_u - 1), 'AV': mnwpri_u * (1 - mnwpri_u) / ni_u})
    elif measure == 'PLN':
        yi = np.log(pri) if addyi else np.log(pri_u)
        vt = _vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        if addvi:
            vi = _pick(vt, {'LS': 1 / xi - 1 / ni, 'AV': 1 / (mnwpri * ni) - 1 / ni})
        else:
            vi = _pick(vt, {'LS': 1 / xi_u - 1 / ni_u, 'AV': 1 / (mnwpri_u * ni_u) - 1 / ni_u})
    elif measure == 'PLO':
        yi = np.log(pri / (1 - pri)) if addyi else np.log(pri_u / (1 - pri_u))
        vt = _vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        if addvi:
            vi = _pick(vt, {'LS': 1 / xi + 1 / mi, 'AV': 1 / (mnwpri * ni) + 1 / ((1 - mnwpri) * ni)})
        else:
            vi = _pick(vt, {'LS': 1 / xi_u + 1 / mi_u, 'AV': 1 / (mnwpri_u * ni_u) + 1 / ((1 - mnwpri_u) * ni_u)})
    elif measure == 'PRZ':
        yi = special.ndtri(pri)
        vi = 2 * math.pi * pri * (1 - pri) * np.exp(yi**2) / ni
    elif measure == 'PAS':
        yi = np.arcsin(np.sqrt(pri))
        vi = 1 / (4 * ni)
    else:
        yi = 1 / 2 * (np.arcsin(np.sqrt(xi / (ni + 1))) + np.arcsin(np.sqrt((xi + 1) / (ni + 1))))
        vi = 1 / (4 * ni + 2)
    return _Out(yi, vi, ni_u, k_all, sub)


def _rates(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, add: float, to: To, addyi: bool, addvi: bool) -> _Out:
    xi, ti = inp.get('xi'), inp.get('ti')
    if not _all_specified(xi, ti):
        raise ValueError(f'{_INCOMPLETE} (i.e., xi, ti).')
    if not _equal_length(xi, ti):
        raise ValueError('Supplied data vectors are not all of the same length.')
    assert xi is not None
    assert ti is not None
    k_all = xi.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    xi, ti = _req(_subset(xi, sub)).copy(), _req(_subset(ti, sub))
    if _anyna(xi < 0, xi):
        raise ValueError('One or more counts are negative.')
    if _anyna(ti <= 0, ti):
        raise ValueError('One or more person-times are <= 0.')

    ni_u = ti
    xi_u = xi.copy()
    if to == 'all':
        xi = xi + add
    if to in ('only0', 'if0all'):
        id0 = xi == 0
    if to == 'only0':
        xi[id0] += add
    if to == 'if0all' and id0.any():
        xi = xi + add

    iri_u = xi_u / ti
    iri = xi / ti
    if measure == 'IR':
        yi = iri if addyi else iri_u
        vi = iri / ti if addvi else iri_u / ti
    elif measure == 'IRLN':
        yi = np.log(iri) if addyi else np.log(iri_u)
        vi = 1 / xi if addvi else 1 / xi_u
    elif measure == 'IRS':
        yi = np.sqrt(iri) if addyi else np.sqrt(iri_u)
        vi = 1 / (4 * ti)
    else:
        yi = 1 / 2 * (np.sqrt(iri) + np.sqrt(iri + 1 / ti))
        vi = 1 / (4 * ti)
    return _Out(yi, vi, ni_u, k_all, sub)


def _means(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str], correct: bool | str) -> _Out:
    mi, sdi, ni, mini, maxi = inp.get('mi'), inp.get('sdi'), inp.get('ni'), inp.get('mini'), inp.get('maxi')
    if measure in ('MN', 'SMN', 'MNLN', 'CVLN'):
        if not _all_specified(mi, sdi, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., mi, sdi, ni).')
        if not _equal_length(mi, sdi, ni):
            raise ValueError('Supplied data vectors are not all of the same length.')
    if measure == 'POMPMN':
        if not _all_specified(mi, sdi, ni, mini, maxi):
            raise ValueError(f'{_INCOMPLETE} (i.e., mi, sdi, ni, mini, maxi).')
        if not _equal_length(mi, sdi, ni, mini, maxi):
            raise ValueError('Supplied data vectors are not all of the same length.')
        assert mi is not None
        assert sdi is not None
        assert mini is not None
        assert maxi is not None
        mi = 100 * (mi - mini) / (maxi - mini)
        sdi = 100 * sdi / (maxi - mini)
    if measure == 'SDLN':
        if not _all_specified(sdi, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., sdi, ni).')
        if not _equal_length(sdi, ni):
            raise ValueError('Supplied data vectors are not all of the same length.')
    assert sdi is not None
    assert ni is not None
    k_all = ni.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    mi = _subset(mi, sub)
    sdi, ni = _req(_subset(sdi, sub)), _req(_subset(ni, sub))

    if _anyna(sdi < 0, sdi):
        raise ValueError('One or more standard deviations are negative.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')
    if measure in ('MNLN', 'CVLN') and mi is not None and _anyna(mi < 0, mi):
        raise ValueError('One or more means are negative.')

    k = ni.shape[0]
    m = np.full(k, _NA) if mi is None else mi
    if measure in ('MN', 'POMPMN'):
        yi = m
        sdpi = math.sqrt(_wmean(sdi**2, ni - 1))
        vt = _vtypes(vtype, k, ('LS', 'HO'), "either 'LS' or 'HO'")
        vi = _pick(vt, {'LS': sdi**2 / ni, 'HO': sdpi**2 / ni})
    elif measure == 'SMN':
        cmi = cmicalc(ni - 1, method=correct)
        yi = cmi * m / sdi
        vi = 1 / ni + yi**2 / (2 * ni)
    elif measure == 'MNLN':
        yi = np.log(m)
        vi = sdi**2 / (ni * m**2)
    elif measure == 'SDLN':
        yi = np.log(sdi) + 1 / (2 * (ni - 1)) if correct else np.log(sdi)
        vi = 1 / (2 * (ni - 1))
    else:
        yi = np.log(sdi / m) + 1 / (2 * (ni - 1)) if correct else np.log(sdi / m)
        vi = 1 / (2 * (ni - 1)) + sdi**2 / (ni * m**2)
    return _Out(yi, vi, ni, k_all, sub)


def _change(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None, vtype: str | Sequence[str], correct: bool | str) -> _Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    ri, ni, di, ti, pi = inp.get('ri'), inp.get('ni'), inp.get('di'), inp.get('ti'), inp.get('pi')
    mini, maxi = inp.get('mini'), inp.get('maxi')

    kmax = max((v.shape[0] for v in (m1i, m2i, sd1i, sd2i, ni, di, ti, pi) if v is not None), default=0)
    ri = _expand(ri, kmax)

    if measure in ('MC', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'CVRC'):
        if not _all_specified(m1i, m2i, sd1i, sd2i, ri, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, ri).')
        if not _equal_length(m1i, m2i, sd1i, sd2i, ri, ni):
            raise ValueError('Supplied data vectors are not all of the same length.')

    if measure == 'POMPMC':
        if not _all_specified(m1i, m2i, sd1i, sd2i, ni, ri, mini, maxi):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, mini, maxi).')
        if not _equal_length(m1i, m2i, sd1i, sd2i, ni, ri, mini, maxi):
            raise ValueError('Supplied data vectors are not all of the same length.')
        assert m1i is not None
        assert m2i is not None
        assert sd1i is not None
        assert sd2i is not None
        assert mini is not None
        assert maxi is not None
        m1i = 100 * (m1i - mini) / (maxi - mini)
        m2i = 100 * (m2i - mini) / (maxi - mini)
        sd1i = 100 * sd1i / (maxi - mini)
        sd2i = 100 * sd2i / (maxi - mini)

    if measure == 'SMCC':
        if not _equal_length(m1i, m2i, sd1i, sd2i, ri, ni, di, ti, pi):
            raise ValueError('Supplied data vectors are not all of the same length.')
        if ni is not None:
            ti = _replmiss(ti, _convp2t(pi, ni - 1))
            if ti is not None:
                di = _replmiss(di, ti * np.sqrt(1 / ni))
        if di is not None:
            has = ~np.isnan(di)
            n = di.shape[0]
            m1i = _fill(m1i, has, di, n)
            m2i = _fill(m2i, has, 0.0, n)
            sd1i = _fill(sd1i, has, 1.0, n)
            sd2i = _fill(sd2i, has, 1.0, n)
            ri = _fill(ri, has, 0.5, n)
        if not _all_specified(m1i, m2i, sd1i, sd2i, ri, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, ri (and di, ti, pi)).')

    if measure == 'SMCR':
        if not _all_specified(m1i, m2i, sd1i, ri, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., m1i, m2i, sd1i, ni, ri).')
        if not _equal_length(m1i, m2i, sd1i, ri, ni):
            raise ValueError('Supplied data vectors are not all of the same length.')

    if measure == 'VRC':
        if not _all_specified(sd1i, sd2i, ri, ni):
            raise ValueError(f'{_INCOMPLETE} (i.e., sd1i, sd2i, ni, ri).')
        if not _equal_length(sd1i, sd2i, ri, ni):
            raise ValueError('Supplied data vectors are not all of the same length.')

    assert ni is not None
    assert ri is not None
    k_all = ni.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    m1i, m2i, sd1i, sd2i = (_subset(v, sub) for v in (m1i, m2i, sd1i, sd2i))
    ni, ri = _req(_subset(ni, sub)), _req(_subset(ri, sub))

    if measure in ('MC', 'POMPMC', 'SMCC', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC'):
        sds = np.concatenate([v for v in (sd1i, sd2i) if v is not None]) if (sd1i is not None or sd2i is not None) else np.empty(0)
        if _anyna(sds < 0, sds):
            raise ValueError('One or more standard deviations are negative.')
    if measure == 'SMCR' and sd1i is not None and _anyna(sd1i < 0, sd1i):
        raise ValueError('One or more standard deviations are negative.')
    if _anyna(np.abs(ri) > 1, ri):
        raise ValueError('One or more correlations are > 1 or < -1.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')

    k = ni.shape[0]
    mi = ni - 1
    nan = np.full(k, _NA)
    m1 = nan if m1i is None else m1i
    m2 = nan if m2i is None else m2i
    sd1 = nan if sd1i is None else sd1i
    sd2 = nan if sd2i is None else sd2i

    sddiffi = np.sqrt(sd1**2 + sd2**2 - 2 * ri * sd1 * sd2)
    sdpi = np.sqrt((sd1**2 + sd2**2) / 2)

    yi = nan
    vi = nan

    if measure in ('MC', 'POMPMC'):
        yi = m1 - m2
        vi = sddiffi**2 / ni

    if measure == 'SMCC':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sddiffi
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2', 'UB', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        vi = _pick(
            vt,
            {
                'LS': 1 / ni + yi**2 / (2 * ni),
                'LS2': cmi**2 * (1 / ni + d**2 / (2 * ni)),
                'UB': 1 / ni + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'H0': np.where(mi > 2, cmi**2 / ni * mi / (mi - 2), _NA),
            },
        )

    if measure == 'SMCR':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sd1
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2', 'UB', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        rui = ri * _fcalc(1 / 2, 1 / 2, (ni - 2) / 2, 1 - ri**2)
        vi = _pick(
            vt,
            {
                'LS': 2 * (1 - ri) / ni + yi**2 / (2 * ni),
                'LS2': cmi**2 * (2 * (1 - ri) / ni + d**2 / (2 * ni)),
                'UB': 2 * (1 - rui) / ni + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'H0': np.where(mi > 2, cmi**2 * 2 * (1 - rui) / ni * mi / (mi - 2), _NA),
            },
        )

    if measure == 'SMCRH':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sd1
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(vt, {'LS': sddiffi**2 / (sd1**2 * (ni - 1)) + yi**2 / (2 * (ni - 1)), 'LS2': sddiffi**2 / (sd1**2 * ni) + yi**2 / (2 * ni)})

    if measure == 'SMCRP':
        mi = 2 * (ni - 1) / (1 + ri**2)
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS',), "'LS'")
        vi = _pick(vt, {'LS': 2 * (1 - ri) / ni + yi**2 * (1 + ri**2) / (4 * ni)})

    if measure == 'SMCRPH':
        mi = 2 * (ni - 1) / (1 + ri**2)
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "'LS' or 'LS2'")
        num = sd1**4 + sd2**4 + 2 * ri**2 * sd1**2 * sd2**2
        vi = _pick(
            vt,
            {
                'LS': sddiffi**2 / (sdpi**2 * (ni - 1)) + yi**2 * num / (8 * sdpi**4 * (ni - 1)),
                'LS2': sddiffi**2 / (sdpi**2 * ni) + yi**2 * num / (8 * sdpi**4 * ni),
            },
        )

    if measure in ('CLESCN', 'AUCCN'):
        d = (m1 - m2) / sdpi
        yi = stats.norm.cdf(d / math.sqrt(2))
        vi = (
            np.exp(-(d**2) / 2)
            / (4 * math.pi)
            * (sddiffi**2 / (sdpi**2 * (ni - 1)) + d**2 * (sd1**4 + sd2**4 + 2 * ri**2 * sd1**2 * sd2**2) / (8 * sdpi**4 * (ni - 1)))
        )

    ls_romc = sd1**2 / (ni * m1**2) + sd2**2 / (ni * m2**2) - 2 * ri * sd1 * sd2 / (m1 * m2 * ni)
    ls2_romc = ls_romc + sd1**4 / (2 * ni**2 * m1**4) + sd2**4 / (2 * ni**2 * m2**4) - ri**2 * sd1**2 * sd2**2 / (ni**2 * m1**2 * m2**2)

    if measure == 'ROMC':
        if correct:
            yi = np.log(m1 / m2) + 1 / 2 * (sd1**2 / (ni * m1**2) - sd2**2 / (ni * m2**2))
        else:
            yi = np.log(m1 / m2)
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(vt, {'LS': ls_romc, 'LS2': ls2_romc})

    if measure == 'VRC':
        yi = np.log(sd1 / sd2)
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(vt, {'LS': (1 - ri**2) / (ni - 1), 'LS2': (1 - ri**2) / (ni - 1) + (1 - ri**4) / (ni - 1) ** 2})

    if measure == 'CVRC':
        if correct:
            yi = np.log(sd1 / m1) - np.log(sd2 / m2) + 1 / 2 * (sd2**2 / (ni * m2**2) - sd1**2 / (ni * m1**2))
        else:
            yi = np.log(sd1 / m1) - np.log(sd2 / m2)
        vt = _vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = _pick(vt, {'LS': ls_romc + (1 - ri**2) / (ni - 1), 'LS2': ls2_romc + (1 - ri**2) / (ni - 1) + (1 - ri**4) / (ni - 1) ** 2})

    return _Out(yi, vi, ni, k_all, sub)


def _alpha(measure: str, inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None) -> _Out:
    ai, mi, ni = inp.get('ai'), inp.get('mi'), inp.get('ni')
    if not _all_specified(ai, mi, ni):
        raise ValueError(f'{_INCOMPLETE} (i.e., ai, mi, ni).')
    if not _equal_length(ai, mi, ni):
        raise ValueError('Supplied data vectors are not all of the same length.')
    assert ai is not None
    assert mi is not None
    assert ni is not None
    k_all = ai.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    ai, mi, ni = (_req(_subset(v, sub)) for v in (ai, mi, ni))
    if _anyna(ai > 1, ai):
        raise ValueError('One or more alpha values are > 1.')
    if _anyna(mi < 2, mi):
        raise ValueError('One or more mi values are < 2.')
    if _anyna(ni <= 0, ni):
        raise ValueError('One or more sample sizes are <= 0.')
    if measure == 'ARAW':
        yi = ai
        vi = 2 * mi * (1 - ai) ** 2 / ((mi - 1) * (ni - 2))
    elif measure == 'AHW':
        yi = 1 - (1 - ai) ** (1 / 3)
        vi = 18 * mi * (ni - 1) * (1 - ai) ** (2 / 3) / ((mi - 1) * (9 * ni - 11) ** 2)
    else:
        yi = -np.log(1 - ai)
        vi = 2 * mi / ((mi - 1) * (ni - 2))
    return _Out(yi, vi, ni, k_all, sub)


def _reh(inp: _Inputs, data: pd.DataFrame | None, subset: ArrayLike | str | None) -> _Out:
    ai, bi, ci = inp.get('ai'), inp.get('bi'), inp.get('ci')
    if not _all_specified(ai, bi, ci):
        raise ValueError(f'{_INCOMPLETE} (i.e., ai, bi, ci).')
    if not _equal_length(ai, bi, ci):
        raise ValueError('Supplied data vectors are not all of the same length.')
    assert ai is not None
    assert bi is not None
    assert ci is not None
    k_all = ai.shape[0]
    sub = _chksubset(subset, k_all, data, 'subset')
    ai, bi, ci = (_req(_subset(v, sub)) for v in (ai, bi, ci))
    if _anyna(ai < 0, ai) or _anyna(bi < 0, bi) or _anyna(ci < 0, ci):
        raise ValueError('One or more group sizes are negative.')
    ni = ai + bi + ci
    p0i = ai / ni
    p1i = bi / ni
    p2i = ci / ni
    yi = np.log(p1i) - np.log(2 * np.sqrt(p0i * p2i))
    vi = ((1 - p1i) / (4 * p0i * p2i) + 1 / p1i) / ni
    return _Out(yi, vi, ni, k_all, sub)
