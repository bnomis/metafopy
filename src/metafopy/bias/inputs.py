"""Input handling shared by the bias tests: an ``rma`` fit or a vector of estimates with sampling variances (or standard errors)."""

import collections.abc
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.metadata
import metafopy.methods.common
import metafopy.models.rma_uni
import metafopy.printing

type FloatArray = numpy.typing.NDArray[numpy.float64]


def as_vector(value: typing.Any, name: str, data: pandas.DataFrame | None = None) -> FloatArray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if data is None or value not in data.columns:
            raise metafopy.errors.InputError(f'{name}={value!r} does not name a column of data')
        value = data[value].to_numpy()
    arr = numpy.atleast_1d(numpy.asarray(value, dtype=numpy.float64))
    if arr.ndim != 1:
        raise metafopy.errors.InputError(f'{name} must be one-dimensional')
    return arr


def vector_inputs(
    x: numpy.typing.ArrayLike | str,
    vi: numpy.typing.ArrayLike | str | None,
    sei: numpy.typing.ArrayLike | str | None,
    *,
    ni: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    data: pandas.DataFrame | None = None,
    what: str = 'the test',
    vi_required: bool = True,
    attr_ni: bool = False,
    expand_vi: bool = False,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
    """``(yi, vi, ni)`` of a vector-of-estimates call, with the subset applied and rows with NAs omitted (with a warning, as metafor does).

    With ``attr_ni`` the sample sizes stored with ``x`` by ``escalc`` are used when ``ni`` is not given (as ``regtest`` does). With ``expand_vi`` a single
    variance is repeated for every estimate (metafor's ``.expand1()``, which ``fsn`` applies). ``na_action`` stands for
    R's ``options(na.action=)``: ``'omit'``, ``'exclude'`` and ``'pass'`` omit the rows with a warning, ``'fail'`` stops.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or rows are missing and ``na_action='fail'``.
    """
    metafopy.methods.common.check_na_action(na_action)
    yi = as_vector(x, 'x', data)
    if yi is None:
        raise metafopy.errors.InputError("Argument 'x' must be a vector with estimates or an 'rma' model object.")
    v = as_vector(vi, 'vi', data)
    s = as_vector(sei, 'sei', data)
    n = as_vector(ni, 'ni', data)
    if v is None and s is not None:
        v = s**2
    if v is None and vi_required:
        raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
    if expand_vi and v is not None and v.shape[0] == 1:
        v = numpy.full(yi.shape, v[0])
    if v is not None and v.shape != yi.shape:
        raise metafopy.errors.InputError("Length of 'yi' and 'vi' (or 'sei') are not the same.")
    if n is not None and n.shape != yi.shape:
        raise metafopy.errors.InputError("Length of 'yi' and 'ni' are not the same.")
    if n is None and attr_ni:
        stored = metafopy.metadata.escalc_ni(x, data, yi.shape[0])
        n = None if stored is None else stored
    if subset is not None:
        sub = numpy.asarray(subset)
        sel = sub if sub.dtype == numpy.bool_ else sub.astype(int)
        yi = yi[sel]
        v = None if v is None else v[sel]
        n = None if n is None else n[sel]
    bad = numpy.isnan(yi)
    if v is not None:
        bad |= numpy.isnan(v)
    if n is not None:
        bad |= numpy.isnan(n)
    if bad.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('Missing values in data.')
        warnings.warn(f'{int(bad.sum())} {"studies" if bad.sum() > 1 else "study"} with NAs omitted from {what}.', stacklevel=3)
        yi = yi[~bad]
        v = None if v is None else v[~bad]
        n = None if n is None else n[~bad]
    return yi, v, n


def data_of(res: metafopy.models.rma_uni.RMAResult) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    return res.yi, res.vi, res.data.ni


def fx(x: float, digits: int, flag: str = '') -> str:
    """R's ``fmtx(x, digits, flag=)``: fixed notation, with ``flag=' '`` leaving a space before non-negative numbers."""
    s = metafopy.printing.fmt_num(x, digits)
    return ' ' + s if flag == ' ' and not s.startswith('-') else s


def fp(p: float, digits: int, *, equal: bool = False, sep: bool = False, pname: str = '') -> str:
    """R's ``fmtp(p, digits, equal=, sep=)``."""
    if numpy.isnan(p):
        return f'{pname}{"=" if equal else ""}{" " if sep else ""}NA'
    s = metafopy.printing.fmt_pval(min(max(p, 0.0), 1.0), digits)
    if s.startswith('<'):
        return f'{pname}<{" " if sep else ""}{s[1:]}'
    return f'{pname}{"=" if equal else ""}{" " if sep else ""}{s}'


def sign_flag(a: float, b: float) -> str:
    return ' ' if numpy.sign(a) != numpy.sign(b) else ''


def print_vector(names: collections.abc.Sequence[str], values: collections.abc.Sequence[str]) -> str:
    """R's ``.print.vector`` followed by ``.print.table``: a named row of strings, each right-aligned, two spaces apart, trailing space."""
    widths = [max(len(n), len(v)) for n, v in zip(names, values, strict=True)]
    row_n = '  '.join(n.rjust(w) for n, w in zip(names, widths, strict=True))
    row_v = '  '.join(v.rjust(w) for v, w in zip(values, widths, strict=True))
    return f'{row_n} \n{row_v} '
