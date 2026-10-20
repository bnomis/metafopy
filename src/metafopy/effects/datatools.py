"""Small data-preparation helpers: metafor's ``vec2mat``, ``dfround``, ``replmiss`` and ``contrmat``.

R sources: https://github.com/wviechtb/metafor/blob/master/R/vec2mat.r, ``R/dfround.r``,
``R/replmiss.r`` and ``R/contrmat.r``. Documentation:
https://wviechtb.github.io/metafor/reference/vec2mat.html,
https://wviechtb.github.io/metafor/reference/dfround.html,
https://wviechtb.github.io/metafor/reference/replmiss.html and
https://wviechtb.github.io/metafor/reference/contrmat.html.

The module also holds the R string helpers these functions (and :mod:`metafopy.effects.reshape`) rely on:
``charmatch()``, ``make.names()`` and metafor's ``.shorten()``.
"""

import collections.abc
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.labelled
import metafopy.effects.measures
import metafopy.errors
import metafopy.metadata

__all__ = ['contrmat', 'dfround', 'replmiss', 'vec2mat']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Column = str | int

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
R_RESERVED = frozenset(
    (
        'if',
        'else',
        'repeat',
        'while',
        'function',
        'for',
        'next',
        'break',
        'TRUE',
        'FALSE',
        'NULL',
        'Inf',
        'NaN',
        'NA',
        'NA_integer_',
        'NA_real_',
        'NA_character_',
        'in',
    )
)


def charmatch(x: str, table: collections.abc.Sequence[str]) -> int | None:
    """R's ``charmatch()``: the position of the exact match, else of the unique partial match; ``None`` if there is none or it is ambiguous."""
    if x in table:
        return list(table).index(x)
    hits = [i for i, name in enumerate(table) if name.startswith(x)]
    return hits[0] if len(hits) == 1 else None


def is_missing(value: typing.Any) -> bool:
    """R's ``is.na()`` for a single value."""
    if value is None or value is pandas.NA or value is pandas.NaT:
        return True
    return isinstance(value, float | numpy.floating) and math.isnan(value)


def make_names(names: collections.abc.Sequence[str], unique: bool = False) -> list[str]:
    """R's ``make.names()``: syntactically valid names (invalid characters become ``.``, an invalid start gets an ``X``)."""
    out: list[str] = []
    for name in names:
        s = ''.join(ch if ch.isalnum() or ch in '._' else '.' for ch in name)
        if not s or not (s[0].isalpha() or (s[0] == '.' and not (len(s) > 1 and s[1].isdigit()))):
            s = f'X{s}'
        if s in R_RESERVED:
            s = f'{s}.'
        out.append(s)
    return list(metafopy.metadata.make_unique(out)) if unique else out


def shorten_names(x: collections.abc.Sequence[str | None], minlen: int | None = None) -> list[str | None]:
    """metafor's ``.shorten()``: cut every string to the shortest common length at which the distinct values stay distinct."""
    values = [v for v in x if v is not None]
    uniq = set(values)
    maxlen = max((len(v) for v in uniq), default=0)
    length = maxlen
    for n in range(1, maxlen + 1):
        if len({v[:n] for v in uniq}) == len(uniq):
            length = n
            break
    if minlen is not None and length < minlen:
        length = min(minlen, maxlen)
    return [None if v is None else v[:length] for v in x]


def resolve_column(data: pandas.DataFrame, spec: Column, name: str) -> int:
    """The position of the variable given by name (``charmatch``) or by 0-based position.

    Raises:
        metafopy.errors.InputError: the variable cannot be found or uniquely identified.
    """
    columns = [str(c) for c in data.columns]
    if isinstance(spec, str):
        pos = charmatch(spec, columns)
        if pos is None:
            raise metafopy.errors.InputError(f"Could not find or uniquely identify variable specified via the '{name}' argument.")
        return pos
    if isinstance(spec, bool) or not isinstance(spec, int | numpy.integer):
        raise metafopy.errors.InputError(f"Argument '{name}' must either be a character string or a number.")
    pos = int(spec)
    if pos < 0 or pos >= len(columns):
        raise metafopy.errors.InputError(f"Specified position of '{name}' variable does not exist in the data frame.")
    return pos


def levels_of(values: pandas.Series) -> list[typing.Any]:
    """``levels(factor(x))`` as values: the categories of a categorical, else the sorted distinct non-missing values."""
    if isinstance(values.dtype, pandas.CategoricalDtype):
        return list(values.cat.categories)
    return sorted({v for v in values.tolist() if not is_missing(v)})


def most_common(values: collections.abc.Sequence[typing.Any], levels: collections.abc.Sequence[typing.Any]) -> typing.Any:
    """``names(sort(table(x), decreasing=TRUE)[1])``: the most frequent level, the first one in level order among ties."""
    counts = {lvl: 0 for lvl in levels}
    for v in values:
        if v in counts:
            counts[v] += 1
    best = None
    best_count = -1
    for lvl in levels:
        if counts[lvl] > best_count:
            best, best_count = lvl, counts[lvl]
    return best


def vec2mat(
    x: numpy.typing.ArrayLike, diag: bool = False, corr: bool | None = None, dimnames: collections.abc.Sequence[str] | None = None
) -> FloatArray | pandas.DataFrame:
    """Fill a symmetric matrix from the elements of its lower triangle (taken column by column, as in R).

    With ``diag=True`` the vector includes the diagonal; ``corr`` (default ``not diag``) sets the diagonal
    to 1. With ``dimnames`` the matrix is returned as a :class:`pandas.DataFrame` with those row and
    column names.

    Raises:
        metafopy.errors.InputError: the length of ``x`` does not correspond to a square matrix.
    """
    if corr is None:
        corr = not diag
    values = numpy.asarray(x, dtype=numpy.float64).reshape(-1)
    p = values.shape[0]
    dims = math.sqrt(2 * p + 1 / 4) + (-1 / 2 if diag else 1 / 2)
    if abs(dims - round(dims)) >= math.sqrt(numpy.finfo(float).eps):
        raise metafopy.errors.InputError("Length of 'x' does not correspond to a square matrix.")
    n = round(dims)
    if dimnames is not None and len(dimnames) != n:
        raise metafopy.errors.InputError(f"Length of 'dimnames' ({len(dimnames)}) does not correspond to the dimensions of the matrix ({n}).")
    R = numpy.full((n, n), numpy.nan)
    # R's lower.tri() runs down the columns
    pos = [(i, j) for j in range(n) for i in range(n) if i > j or (diag and i == j)]
    for value, (i, j) in zip(values, pos, strict=True):
        R[i, j] = R[j, i] = value
    if corr:
        numpy.fill_diagonal(R, 1.0)
    if dimnames is not None:
        return pandas.DataFrame(R, index=list(dimnames), columns=list(dimnames))
    return R


def r_round(x: float, digits: int) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    return round(x, int(digits))


def dfround(
    x: pandas.DataFrame | numpy.typing.ArrayLike,
    digits: int | float | collections.abc.Sequence[int | float | None] = 0,
    drop0: bool | collections.abc.Sequence[bool] = True,
) -> pandas.DataFrame:
    """Round the numeric columns of a data frame (a 2-d array is turned into one first).

    ``digits`` and ``drop0`` have one value for all columns or one per column; a numeric column whose ``digits`` is missing (``None`` or NaN)
    becomes all missing, as ``round(x, NA)`` does in R (give any value for the columns that are not numeric). Columns with ``drop0=False``
    become strings formatted with exactly ``digits`` decimals (R's ``formatC(format="f")``); non-numeric
    columns are left alone.

    Raises:
        metafopy.errors.InputError: ``digits`` or ``drop0`` does not have one value per column.
        metafopy.errors.ArgumentError: ``x`` is not a data frame or matrix.
    """
    if isinstance(x, pandas.DataFrame):
        out = x.copy()
    else:
        arr = numpy.asarray(x)
        if arr.ndim != 2:
            raise metafopy.errors.ArgumentError('Argument \'x\' must be an object of class "data.frame".')
        out = pandas.DataFrame(arr)
    p = out.shape[1]
    dig: list[int | None] = (
        [int(digits)] * p
        if isinstance(digits, int | numpy.integer)
        else [None if d is None or math.isnan(d) else int(d) for d in ([digits] if isinstance(digits, float) else digits)]
    )
    drop = [bool(drop0)] * p if isinstance(drop0, bool | numpy.bool_) else [bool(d) for d in drop0]
    if len(dig) == 1:
        dig = dig * p
    if len(drop) == 1:
        drop = drop * p
    if len(dig) != p:
        raise metafopy.errors.InputError(f"Number of columns in 'x' ({p}) does not match the length of 'digits' ({len(dig)}).")
    if len(drop) != p:
        raise metafopy.errors.InputError(f"Number of columns in 'x' ({p}) does not match the length of 'drop0' ({len(drop)}).")
    for j in range(p):
        col = out.iloc[:, j]
        if not pandas.api.types.is_numeric_dtype(col) or pandas.api.types.is_bool_dtype(col):
            continue
        digits_j = dig[j]
        vals = col.to_numpy(dtype=numpy.float64)
        if digits_j is None:
            if not drop[j]:
                raise metafopy.errors.InputError('missing value where TRUE/FALSE needed')
            out[out.columns[j]] = numpy.full(len(vals), numpy.nan)
            continue
        if drop[j]:
            out[out.columns[j]] = [r_round(float(v), digits_j) for v in vals]
        else:
            out[out.columns[j]] = ['NA' if math.isnan(v) else f'{v:.{digits_j}f}' for v in vals]
    return metafopy.core.labelled.escalc_frame(out) if isinstance(x, metafopy.core.labelled.EscalcFrame) else out


def column_or_values(value: typing.Any, data: pandas.DataFrame | None, name: str) -> FloatArray | None:
    """A numeric argument given as values or as the name of a column of ``data``.

    Raises:
        metafopy.errors.InputError: a column name is given without ``data``.
        metafopy.errors.NotFoundError: the column does not exist.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if data is None:
            raise metafopy.errors.InputError(f'{name}={value!r} names a column, but no data was given')
        if value not in data.columns:
            raise metafopy.errors.NotFoundError(f'column {value!r} (for {name}) not found in data')
        value = data[value].to_numpy()
    return numpy.atleast_1d(numpy.asarray(value, dtype=numpy.float64)).reshape(-1)


def replmiss(x: numpy.typing.ArrayLike | str | None, y: numpy.typing.ArrayLike | str | None, data: pandas.DataFrame | None = None) -> FloatArray | None:
    """Replace the missing values of ``x`` by the corresponding values of ``y`` (or by ``y`` when it is a single value).

    ``x`` and ``y`` may be names of columns of ``data``. An empty or ``None`` ``x`` counts as all
    missing; an empty or ``None`` ``y`` returns ``x`` unchanged.

    Raises:
        metafopy.errors.InputError: ``x`` and ``y`` have different lengths.
    """
    xv = column_or_values(x, data, 'x')
    yv = column_or_values(y, data, 'y')
    out = metafopy.effects.measures.replmiss(xv, yv)
    if out is None:
        return None
    out = out.copy()
    out[numpy.isnan(out)] = math.nan
    return out


def contrmat(
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any],
    grp1: Column,
    grp2: Column,
    last: typing.Any = None,
    shorten: bool = False,
    minlen: int = 2,
    check: bool = True,
    append: bool = True,
) -> pandas.DataFrame:
    """The contrast matrix of a network meta-analysis: ``+1`` for the ``grp1`` and ``-1`` for the ``grp2`` group of each row.

    ``grp1`` and ``grp2`` name (or give the 0-based position of) the two group variables. The columns are
    the groups in sorted order, with ``last`` (default: the most common ``grp2`` group; a missing value such
    as ``float('nan')`` keeps the sorted order) moved to the end. ``shorten`` cuts the group names to the
    shortest distinct length (at least ``minlen``), ``check`` makes them syntactically valid R names and
    ``append`` adds the matrix to ``data``.

    Raises:
        metafopy.errors.InputError: a variable or group cannot be found, or a group variable has missing values.
    """
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)
    g1 = data.iloc[:, resolve_column(data, grp1, 'grp1')]
    if any(is_missing(v) for v in g1):
        raise metafopy.errors.InputError("Variable specified via 'grp1' argument should not contain missing values.")
    g2 = data.iloc[:, resolve_column(data, grp2, 'grp2')]
    if any(is_missing(v) for v in g2):
        raise metafopy.errors.InputError("Variable specified via 'grp2' argument should not contain missing values.")
    both_cat = isinstance(g1.dtype, pandas.CategoricalDtype) and isinstance(g2.dtype, pandas.CategoricalDtype)
    if both_cat and [str(c) for c in g1.cat.categories] == [str(c) for c in g2.cat.categories]:
        lvls = [str(c) for c in g1.cat.categories]
    else:
        lvls = sorted({str(v) for v in levels_of(g1)} | {str(v) for v in levels_of(g2)})
    g1s = [str(v) for v in g1]
    g2s = [str(v) for v in g2]
    if last is None:
        last = most_common(g2s, sorted(set(g2s)))
    if is_missing(last):
        last = lvls[-1]
    pos = charmatch(str(last), lvls)
    if pos is None:
        raise metafopy.errors.InputError("Could not find or uniquely identify group specified via the 'last' argument.")
    lvls = [*lvls[:pos], *lvls[pos + 1 :], lvls[pos]]
    index = {lvl: j for j, lvl in enumerate(lvls)}
    X = numpy.zeros((data.shape[0], len(lvls)))
    for i, (a, b) in enumerate(zip(g1s, g2s, strict=True)):
        X[i, index[a]] += 1.0
        X[i, index[b]] -= 1.0
    names = [str(v) for v in shorten_names(lvls, minlen=minlen)] if shorten else lvls
    names = make_names(names, unique=True) if check else names
    out = pandas.DataFrame(X, columns=names, index=data.index)
    if append:
        return pandas.concat([data, out], axis=1)
    return out
