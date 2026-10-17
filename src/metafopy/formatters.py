"""Formatter functions ``fmtx``, ``fmtp``, ``fmtp2`` and ``fmtt``, ports of metafor's ``R/formatters.r``.

R returns character vectors; these functions return a ``str`` for a scalar input
and a ``list[str]`` for a sequence (``fmtx`` returns a ``pandas.DataFrame`` of
strings for a data frame and a ``numpy`` string array for a 2-d array). The
``quote`` argument only changes how R prints the result, so it is accepted and
has no effect.

R source: https://github.com/wviechtb/metafor/blob/master/R/formatters.r;
reference page: https://wviechtb.github.io/metafor/reference/formatters.html.
"""

import collections.abc
import math
import re
import typing

import numpy
import pandas

import metafopy.errors

__all__ = ['fmtp', 'fmtp2', 'fmtt', 'fmtx', 'format_c', 'r_as_character']

type Number = float | int | None
type Numbers = Number | collections.abc.Sequence[Number] | numpy.ndarray[typing.Any, typing.Any]


def is_na(x: Number) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def as_list(x: Numbers) -> tuple[list[Number], bool]:
    """The values of ``x`` and whether ``x`` was a scalar."""
    if x is None or isinstance(x, int | float):
        return [x], True
    arr = numpy.asarray(x, dtype=object)
    if arr.ndim == 0:
        return [arr.item()], True
    return [None if v is None else float(v) for v in arr.ravel().tolist()], False


def drop0trailing(s: str) -> str:
    """R's ``prettyNum(drop0trailing=TRUE)`` for fixed notation: drop trailing zeros and a bare decimal point."""
    s = re.sub(r'\.0+$', '', s)
    return re.sub(r'(\.[0-9]*[1-9])0+$', r'\1', s)


def format_c(x: Number, digits: int, flag: str = '', width: int | None = None, drop0: bool = False) -> str:
    """R's ``formatC(x, format="f", digits=digits, flag=flag, width=width, drop0trailing=drop0)`` for one value."""
    if width is None:
        width = digits + 1
    elif width == 0:
        width = digits
    if is_na(x) or (isinstance(x, float) and math.isinf(x)):
        text = 'NA' if x is None else ('NaN' if math.isnan(typing.cast(float, x)) else ('Inf' if typing.cast(float, x) > 0 else '-Inf'))
        return text.ljust(width) if '-' in flag else text.rjust(width)
    out = f'%{flag}{width}.{digits}f' % float(typing.cast(float, x))
    return drop0trailing(out) if drop0 else out


def fmtx(
    x: Numbers | pandas.DataFrame,
    digits: int | collections.abc.Sequence[int] = 4,
    flag: str = '',
    quote: bool = False,
    *,
    addwidth: int | None = None,
    drop0ifint: bool = False,
    add0: bool = True,
    thresh: float | None = None,
    postfix: str = '',
) -> typing.Any:
    """Format numbers to ``digits`` decimals without scientific notation, as metafor's ``fmtx()``.

    ``addwidth`` pads to a width of ``digits + addwidth``; ``drop0ifint`` drops
    trailing zeros (R does so only when ``digits`` is an R integer, which has no
    Python counterpart); ``add0=False`` drops the 0 before the decimal point;
    ``thresh`` (scalar ``x`` only) uses 0 decimals when ``|x| <= thresh``;
    ``postfix`` is appended to every value.

    Raises:
        metafopy.errors.InputError: ``thresh`` is given for a non-scalar ``x``.
    """
    two_d = isinstance(x, pandas.DataFrame) or (isinstance(x, numpy.ndarray) and x.ndim == 2)
    if two_d:
        frame = x if isinstance(x, pandas.DataFrame) else pandas.DataFrame(numpy.asarray(x))
        ncol = frame.shape[1]
        dig = [int(digits)] * ncol if isinstance(digits, int) else [int(d) for d in digits]
        if len(dig) == 1:
            dig = dig * ncol
        out = pandas.DataFrame(
            {
                col: fmtx(frame.iloc[:, j].to_numpy(dtype=float), dig[j], flag, addwidth=addwidth, drop0ifint=drop0ifint, add0=add0, postfix=postfix)
                for j, col in enumerate(frame.columns)
            },
            index=frame.index,
        )
        return out if isinstance(x, pandas.DataFrame) else out.to_numpy(dtype=str)
    values, scalar = as_list(typing.cast(Numbers, x))
    d = int(digits) if isinstance(digits, int) else int(next(iter(digits)))
    if thresh is not None:
        if not scalar:
            raise metafopy.errors.InputError("Can only use 'thresh' when 'x' is a scalar.")
        v = values[0]
        if not is_na(v) and abs(typing.cast(float, v)) <= thresh:
            d = 0
    width = None if addwidth is None else d + addwidth
    res: list[str] = []
    for v in values:
        if is_na(v):
            res.append(f'NA{postfix}')
            continue
        s = format_c(v, d, flag, width, drop0ifint)
        if not add0:
            s = s.replace('0.', '.')
        res.append(f'{s}{postfix}')
    return res[0] if scalar else res


def fmtp(p: Numbers, digits: int = 4, pname: str = '', equal: bool = False, sep: bool = False, add0: bool = False, quote: bool = False) -> str | list[str]:
    """Format p-values, printing those below ``10**-digits`` as ``<.0001``, as metafor's ``fmtp()``."""
    values, scalar = as_list(p)
    digits = max(digits, 1)
    cutoff = '.' + '0' * (digits - 1) + '1'
    ncutoff = float(cutoff)
    eq = '=' if equal else ''
    if sep:
        if pname != '':
            pname = f'{pname} '
        sp = ' '
    else:
        sp = ''
    res: list[str] = []
    for v in values:
        if is_na(v):
            res.append(f'{pname}{eq}{sp}NA')
            continue
        q = min(max(float(typing.cast(float, v)), 0.0), 1.0)
        if q >= ncutoff:
            res.append(f'{pname}{eq}{sp}{format_c(q, digits)}')
        else:
            res.append(f'{pname}<{sp}{"0" if add0 else ""}{cutoff}')
    return res[0] if scalar else res


def fmtp2(
    p: Numbers,
    cutoff: collections.abc.Sequence[float] = (0.001, 0.06),
    pname: str = 'p',
    sep: bool = True,
    add0: bool = False,
    quote: bool = False,
) -> str | list[str]:
    """Format p-values in APA style with two cutoffs, as metafor's ``fmtp2()``.

    Raises:
        metafopy.errors.InputError: ``cutoff`` is not of length 2 or its lower value is not positive.
    """
    values, scalar = as_list(p)
    if len(cutoff) != 2:
        raise metafopy.errors.InputError("Argument 'cutoff' must be of length 2.")
    lo, hi = sorted(float(c) for c in cutoff)
    if lo == 0:
        raise metafopy.errors.InputError("The lower 'cutoff' value must be > 0.")
    digits1 = len(format_c(lo, 10, drop0=True)) - 2
    digits2 = len(format_c(hi, 10, drop0=True)) - 2
    if sep:
        if pname != '':
            pname = f'{pname} '
        sp = ' '
    else:
        sp = ''
    res: list[str] = []
    for v in values:
        if is_na(v):
            s = f'{pname}={sp}NA'
        else:
            q = min(max(float(typing.cast(float, v)), 0.0), 1.0)
            if q < lo:
                s = f'{pname}<{sp}{format_c(lo, digits1)}'
            elif q < hi:
                s = f'{pname}={sp}{format_c(q, digits1)}'
            else:
                s = f'{pname}={sp}{format_c(q, digits2)}'
        if not add0:
            s = s.replace('0.', '.')
        res.append(s)
    return res[0] if scalar else res


def r_as_character(x: float) -> str:
    """R's ``as.character()`` of a double (15 significant digits, fixed unless scientific notation is narrower)."""
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    if x == 0:
        return '0'
    mant, exp_s = f'{abs(x):.14e}'.split('e')
    digits = mant.replace('.', '').rstrip('0')
    nsig = max(len(digits), 1)
    e = int(exp_s)
    sign = '-' if x < 0 else ''
    fixed = f'{sign}{abs(x):.{max(0, nsig - 1 - e)}f}'
    smant = digits[0] + ('.' + digits[1:] if nsig > 1 else '')
    sci = f'{sign}{smant}e{"-" if e < 0 else "+"}{abs(e):02d}'
    return fixed if len(fixed) <= len(sci) else sci


def fmtt(
    val: float,
    tname: str,
    df: float | None = None,
    df1: float | None = None,
    df2: float | None = None,
    pval: float | None = None,
    digits: int | collections.abc.Sequence[int] | collections.abc.Mapping[str, int] = 4,
    pname: str = 'p-val',
    format: int = 1,
    sep: bool = True,
    quote: bool = False,
    call: bool = False,
    *,
    flag: str = '',
) -> str:
    """Format a test statistic with its degrees of freedom and p-value, as metafor's ``fmtt()``.

    ``digits`` is a scalar, a pair (test, p-value) or a mapping with ``test``
    and ``pval``. ``call=True`` (an R plotmath call) has no Python counterpart.

    Raises:
        metafopy.errors.InputError: ``format`` is not 1 or 2, ``pval`` is missing, ``digits`` lacks an element, or ``call=True``.
    """
    if format not in (1, 2):
        raise metafopy.errors.InputError("Argument 'format' can only be equal to 1 or 2.")
    if numpy.size(val) != 1:
        raise metafopy.errors.InputError("Argument 'val' must be a scalar.")
    if pval is None:
        raise metafopy.errors.InputError("Must specify the 'pval' argument.")
    if call:
        raise metafopy.errors.InputError("'call=True' returns an R plotmath call, which has no Python counterpart.")
    if isinstance(digits, collections.abc.Mapping):
        if 'test' not in digits or 'pval' not in digits:
            raise metafopy.errors.InputError("Argument 'digits' must have a 'test' and a 'pval' element.")
        dt, dp = int(digits['test']), int(digits['pval'])
    elif isinstance(digits, int):
        dt = dp = digits
    else:
        dl = list(digits)
        if len(dl) == 1:
            dt = dp = int(dl[0])
        elif len(dl) == 2:
            dt, dp = int(dl[0]), int(dl[1])
        else:
            raise metafopy.errors.InputError("Argument 'digits' must have a 'test' and a 'pval' element.")
    sp = ' ' if sep else ''
    stat = fmtx(val, dt, flag=flag)
    pv = fmtp(pval, dp, equal=True, sep=sep)
    tail = f'{sp}={sp}{stat}'
    pstr = f', {pname}{sp}{pv}'
    if format == 1:
        if df is None:
            if df1 is not None and df2 is not None:
                return f'{tname}(df1{sp}={sp}{r_as_character(df1)},{sp}df2{sp}={sp}{r_as_character(round(df2, 2))}){tail}{pstr}'
            return f'{tname}{tail}{pstr}'
        return f'{tname}(df{sp}={sp}{r_as_character(df)}){tail}{pstr}'
    if df is None:
        if df1 is not None and df2 is not None:
            return f'{tname}{tail}, df1{sp}={sp}{r_as_character(df1)}, df2{sp}={sp}{r_as_character(round(df2, 2))}{pstr}'
        return f'{tname}{tail}{pstr}'
    return f'{tname}{tail}, df{sp}={sp}{r_as_character(df)}{pstr}'
