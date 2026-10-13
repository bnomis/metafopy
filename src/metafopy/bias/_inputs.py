"""Input handling shared by the bias tests: an ``rma`` fit or a vector of estimates with sampling variances (or standard errors)."""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from metafopy._print import fmt_num, fmt_pval
from metafopy.models.rma_uni import RMAResult

type FloatArray = NDArray[np.float64]


def as_vector(value: Any, name: str, data: pd.DataFrame | None = None) -> FloatArray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if data is None or value not in data.columns:
            raise ValueError(f'{name}={value!r} does not name a column of data')
        value = data[value].to_numpy()
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return arr


def vector_inputs(
    x: ArrayLike | str,
    vi: ArrayLike | str | None,
    sei: ArrayLike | str | None,
    *,
    ni: ArrayLike | str | None = None,
    subset: ArrayLike | None = None,
    data: pd.DataFrame | None = None,
    what: str = 'the test',
    vi_required: bool = True,
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    """``(yi, vi, ni)`` of a vector-of-estimates call, with the subset applied and rows with NAs omitted (with a warning, as metafor does)."""
    yi = as_vector(x, 'x', data)
    if yi is None:
        raise ValueError("Argument 'x' must be a vector with estimates or an 'rma' model object.")
    v = as_vector(vi, 'vi', data)
    s = as_vector(sei, 'sei', data)
    n = as_vector(ni, 'ni', data)
    if v is None and s is not None:
        v = s**2
    if v is None and vi_required:
        raise ValueError("Must specify the 'vi' or 'sei' argument.")
    if v is not None and v.shape != yi.shape:
        raise ValueError("Length of 'yi' and 'vi' (or 'sei') are not the same.")
    if n is not None and n.shape != yi.shape:
        raise ValueError("Length of 'yi' and 'ni' are not the same.")
    if subset is not None:
        sub = np.asarray(subset)
        sel = sub if sub.dtype == np.bool_ else sub.astype(int)
        yi = yi[sel]
        v = None if v is None else v[sel]
        n = None if n is None else n[sel]
    bad = np.isnan(yi)
    if v is not None:
        bad |= np.isnan(v)
    if n is not None:
        bad |= np.isnan(n)
    if bad.any():
        warnings.warn(f'{int(bad.sum())} {"studies" if bad.sum() > 1 else "study"} with NAs omitted from {what}.', stacklevel=3)
        yi = yi[~bad]
        v = None if v is None else v[~bad]
        n = None if n is None else n[~bad]
    return yi, v, n


def data_of(res: RMAResult) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    return res.yi, res.vi, res.data.ni


def fx(x: float, digits: int, flag: str = '') -> str:
    """R's ``fmtx(x, digits, flag=)``: fixed notation, with ``flag=' '`` leaving a space before non-negative numbers."""
    s = fmt_num(x, digits)
    return ' ' + s if flag == ' ' and not s.startswith('-') else s


def fp(p: float, digits: int, *, equal: bool = False, sep: bool = False, pname: str = '') -> str:
    """R's ``fmtp(p, digits, equal=, sep=)``."""
    if np.isnan(p):
        return f'{pname}{"=" if equal else ""}{" " if sep else ""}NA'
    s = fmt_pval(min(max(p, 0.0), 1.0), digits)
    if s.startswith('<'):
        return f'{pname}<{" " if sep else ""}{s[1:]}'
    return f'{pname}{"=" if equal else ""}{" " if sep else ""}{s}'


def sign_flag(a: float, b: float) -> str:
    return ' ' if np.sign(a) != np.sign(b) else ''


def print_vector(names: Sequence[str], values: Sequence[str]) -> str:
    """R's ``.print.vector`` followed by ``.print.table``: a named row of strings, each right-aligned, two spaces apart, trailing space."""
    widths = [max(len(n), len(v)) for n, v in zip(names, values, strict=True)]
    row_n = '  '.join(n.rjust(w) for n, w in zip(names, widths, strict=True))
    row_v = '  '.join(v.rjust(w) for v, w in zip(values, widths, strict=True))
    return f'{row_n} \n{row_v} '
