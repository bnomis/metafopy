"""Standard errors, test statistics and confidence intervals for escalc output (metafor's ``summary.escalc``).

A port of ``summary.escalc()`` from metafor 5.2-1 (``R/summary.escalc.r``);
see https://wviechtb.github.io/metafor/reference/print.escalc.html. It works on
the :class:`pandas.DataFrame` returned by :func:`metafopy.escalc`, whose
metafor attributes live in ``DataFrame.attrs`` (``yi_names``, ``vi_names``,
``measure``, ``ni``, ...).

Differences from R that follow Python conventions:

- ``out_names`` defaults to ``('sei', 'zi', 'pval', 'ci_lb', 'ci_ub')`` and
  any strings are accepted (R makes them syntactically valid names);
- like R's ``mapply()``, ``transf`` is called once per value; keyword
  arguments in ``**kwargs`` that have one entry per row are passed
  elementwise, everything else (scalars, mappings such as ``targs``) as is;
- the result keeps the input's row index.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

__all__ = ['summary_escalc']

type FloatArray = NDArray[np.float64]

_ATTR_KEYS = ('sei_names', 'zi_names', 'pval_names', 'ci_lb_names', 'ci_ub_names')


def _level(level: float) -> float:
    """metafor's ``.level``: a confidence level in percent (or as a proportion) to alpha."""
    if level > 100 or level < 0:
        raise ValueError("Argument 'level' must be between 0 and 100.")
    if level == 0:
        return 1.0
    if level >= 1:
        return (100 - level) / 100
    return 1 - level if level > 0.5 else level


def _names(value: Any) -> list[str]:
    """An attribute holding one name or several as a list (``None`` for none)."""
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def _union(first: str, rest: Any) -> list[str]:
    """R's ``union(first, rest)`` for character vectors."""
    out = [first]
    for name in _names(rest):
        if name not in out:
            out.append(name)
    return out


def _mapply(transf: Callable[..., Any], values: FloatArray, kwargs: Mapping[str, Any]) -> FloatArray:
    """``mapply(transf, values, ...)``: one call per value, with per-row keyword arguments taken elementwise."""
    n = values.size
    per_row: dict[str, Any] = {}
    shared: dict[str, Any] = {}
    for key, val in kwargs.items():
        if val is None or isinstance(val, (str, Mapping)) or np.ndim(val) == 0:
            shared[key] = val
        else:
            arr = np.asarray(val)
            if arr.size == n:
                per_row[key] = arr.reshape(-1)
            elif arr.size == 1:
                shared[key] = arr.reshape(-1)[0]
            else:
                raise ValueError(f"Argument '{key}' must have length 1 or one value per row.")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        res = np.asarray(transf(values[i], **shared, **{k: v[i] for k, v in per_row.items()}), dtype=np.float64).reshape(-1)
        if res.size != 1:
            raise ValueError("The 'transf' function must return a single value for each value it is given.")
        out[i] = res[0]
    return out


def summary_escalc(
    object: pd.DataFrame,
    out_names: Sequence[str] = ('sei', 'zi', 'pval', 'ci_lb', 'ci_ub'),
    var_names: Sequence[str] | None = None,
    H0: ArrayLike = 0,
    append: bool = True,
    replace: bool = True,
    level: float = 95,
    olim: Sequence[float] | None = None,
    digits: int | Sequence[int] | None = None,
    transf: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Add standard errors, z-values, p-values and confidence intervals to escalc output.

    ``sei = sqrt(vi)``, ``zi = (yi - H0)/sei``, two-sided ``pval`` and the
    Wald-type interval ``yi -/+ z_crit * sei`` at the given ``level``. With
    ``transf``, ``yi`` and the interval bounds are transformed (the interval
    is re-ordered when ``transf`` is decreasing) and ``vi``, ``sei``, ``zi``
    and ``pval`` are dropped; ``kwargs`` are passed on to ``transf``.
    ``olim`` clips ``yi`` and the bounds to an interval (``zi`` and ``pval``
    are based on the unclipped values).
    """
    alpha = _level(level)
    crit = float(stats.norm.isf(alpha / 2))
    if len(out_names) != 5:
        raise ValueError("Argument 'out_names' must be of length 5.")
    out_names = list(out_names)

    attrs = object.attrs
    if var_names is None:
        if attrs.get('yi_names'):
            yi_name = _names(attrs['yi_names'])[0]
        elif 'yi' in object.columns:
            yi_name = 'yi'
        else:
            raise ValueError("Cannot determine the name of the 'yi' variable.")
        if attrs.get('vi_names'):
            vi_name = _names(attrs['vi_names'])[0]
        elif 'vi' in object.columns:
            vi_name = 'vi'
        else:
            raise ValueError("Cannot determine the name of the 'vi' variable.")
    else:
        if len(var_names) != 2:
            raise ValueError("Argument 'var_names' must be of length 2.")
        yi_name, vi_name = var_names[0], var_names[1]
    for name in (yi_name, vi_name):
        if name not in object.columns:
            raise ValueError(f"Cannot find variable '{name}' in the data frame.")

    yi = object[yi_name].to_numpy(dtype=np.float64, copy=True)
    vi_in = object[vi_name].to_numpy(dtype=np.float64, copy=True)
    vi: FloatArray | None = vi_in
    k = yi.size
    h0 = np.asarray(H0, dtype=np.float64).reshape(-1)
    if h0.size == 1:
        h0 = np.full(k, h0[0])

    with np.errstate(invalid='ignore', divide='ignore'):
        se = np.sqrt(vi_in)
        z = (yi - h0) / se
    sei: FloatArray | None = se
    zi: FloatArray | None = z
    pval: FloatArray | None = 2 * special.ndtr(-np.abs(z))
    if transf is not None:
        ci_lb = _mapply(transf, yi - crit * se, kwargs)
        ci_ub = _mapply(transf, yi + crit * se, kwargs)
        yi = _mapply(transf, yi, kwargs)
        vi = sei = zi = pval = None
    else:
        ci_lb = yi - crit * se
        ci_ub = yi + crit * se

    # make sure the bounds are in increasing order (pairs with a NaN are left alone)
    flip = ci_lb > ci_ub
    ci_lb, ci_ub = np.where(flip, ci_ub, ci_lb), np.where(flip, ci_lb, ci_ub)

    if olim is not None:
        if len(olim) != 2:
            raise ValueError("Argument 'olim' must be of length 2.")
        lo, hi = sorted(float(v) for v in olim)
        yi, ci_lb, ci_ub = (np.where(v < lo, lo, np.where(v > hi, hi, v)) for v in (yi, ci_lb, ci_ub))

    x = object.copy()
    x[yi_name] = yi
    if vi is None:
        x = x.drop(columns=vi_name)

    values = (sei, zi, pval, ci_lb, ci_ub)
    if append:
        dat = x
        for name, val in zip(out_names, values, strict=True):
            if val is None:
                # R assigns NULL: the column is removed (replace=TRUE) or left as it is
                if replace and name in dat.columns:
                    dat = dat.drop(columns=name)
            elif replace or name not in dat.columns:
                dat[name] = val
            else:
                col = dat[name].to_numpy(dtype=np.float64, copy=True)
                miss = np.isnan(col)
                col[miss] = val[miss]
                dat[name] = col
    elif transf is not None:
        dat = pd.DataFrame({yi_name: yi, out_names[3]: ci_lb, out_names[4]: ci_ub}, index=object.index)
    else:
        dat = pd.DataFrame(dict(zip([yi_name, vi_name, *out_names], [yi, vi, *values], strict=True)), index=object.index)

    # the escalc attributes stay with an untransformed yi (in R they are attributes of the yi vector)
    new_attrs: dict[str, Any] = {}
    if transf is None:
        new_attrs.update({key: attrs[key] for key in ('measure', 'slab', 'ni') if key in attrs})
    new_attrs['digits'] = digits if digits is not None else attrs.get('digits')
    new_attrs['yi_names'] = _union(yi_name, attrs.get('yi_names'))
    new_attrs['vi_names'] = _union(vi_name, attrs.get('vi_names'))
    for key, name in zip(_ATTR_KEYS, out_names, strict=True):
        new_attrs[key] = _union(name, attrs.get(key))
    dat.attrs = {}
    dat.attrs.update(new_attrs)
    return dat
