"""Grouping variables for the ``random`` argument of :func:`metafopy.rma_mv`.

R turns every variable of a random-effects formula into a factor whose levels
are the sorted unique values (``factor(x)``); an existing factor keeps its level
order and drops unused levels. ``~ 1 | a/b`` expands to the interactions ``a``,
``a/b`` (``interaction(..., drop=TRUE, lex.order=TRUE, sep="/")``). The same
rules are reproduced here so that the order of variance components and of the
random-effects levels matches metafor.
"""

import dataclasses
import re

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.errors

__all__ = ['Factor', 'evaluate_variable', 'indicator', 'interaction', 'split_top_level', 'to_categorical', 'to_factor']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]

NUMERIC = re.compile(r'^(?:as\.numeric|I)\((.*)\)$')
WRAPPERS = re.compile(r'^(?:as\.)?(?:factor|character|numeric|C)\((.*)\)$')
CALL = re.compile(r'^(interaction|paste0?)\((.*)\)$')


@dataclasses.dataclass(frozen=True, slots=True)
class Factor:
    """Integer codes into ``levels``; ``-1`` marks a missing value."""

    codes: IntArray
    levels: tuple[str, ...]

    @property
    def nlevels(self) -> int:
        return len(self.levels)

    @property
    def has_na(self) -> bool:
        return bool((self.codes < 0).any())

    def take(self, idx: numpy.typing.ArrayLike) -> Factor:
        """Subset the rows without changing the levels."""
        return Factor(self.codes[numpy.asarray(idx)], self.levels)

    def drop_unused(self) -> Factor:
        """Re-evaluate the levels present, keeping their order (R's ``factor(f)``)."""
        used = numpy.unique(self.codes[self.codes >= 0])
        remap = numpy.full(self.nlevels + 1, -1, dtype=numpy.intp)
        remap[used] = numpy.arange(used.shape[0])
        codes = numpy.where(self.codes >= 0, remap[numpy.where(self.codes >= 0, self.codes, self.nlevels)], -1)
        return Factor(codes.astype(numpy.intp), tuple(self.levels[i] for i in used))

    def labels(self) -> numpy.typing.NDArray[numpy.object_]:
        lev = numpy.array([*self.levels, ''], dtype=object)
        return lev[numpy.where(self.codes >= 0, self.codes, self.nlevels)]


def format_number(x: float) -> str:
    """``as.character()`` of a number: up to 15 significant digits."""
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f'{x:.15g}'


def to_factor(values: pandas.Series | numpy.typing.ArrayLike) -> Factor:
    """R's ``factor(x)``: sorted unique values become the levels (an existing categorical keeps its order)."""
    s = values if isinstance(values, pandas.Series) else pandas.Series(numpy.asarray(values))
    missing = s.isna().to_numpy()
    if isinstance(s.dtype, pandas.CategoricalDtype):
        cats = [str(c) for c in s.cat.categories]
        codes = numpy.asarray(s.cat.codes, dtype=numpy.intp)
        return Factor(codes, tuple(cats)).drop_unused()
    present = s[~missing]
    if pandas.api.types.is_bool_dtype(s):
        uniq = sorted({bool(v) for v in present.tolist()})
        labels = [str(v).upper() for v in uniq]
        lookup: dict[object, int] = {v: i for i, v in enumerate(uniq)}
    elif pandas.api.types.is_numeric_dtype(s):
        uniq_n = sorted({float(v) for v in present.tolist()})
        labels = [format_number(v) for v in uniq_n]
        lookup = {v: i for i, v in enumerate(uniq_n)}
    else:
        uniq_s = sorted({str(v) for v in present.tolist()})
        labels = uniq_s
        lookup = {v: i for i, v in enumerate(uniq_s)}
    codes = numpy.full(len(s), -1, dtype=numpy.intp)
    keep = numpy.flatnonzero(~missing)
    if pandas.api.types.is_bool_dtype(s):
        codes[keep] = [lookup[bool(v)] for v in present.tolist()]
    elif pandas.api.types.is_numeric_dtype(s):
        codes[keep] = [lookup[float(v)] for v in present.tolist()]
    else:
        codes[keep] = [lookup[str(v)] for v in present.tolist()]
    return Factor(codes, tuple(labels))


def interaction(factors: list[Factor], sep: str = '.', lex_order: bool = False) -> Factor:
    """R's ``interaction(..., drop=TRUE)``; with ``lex_order`` the first factor varies slowest."""
    if len(factors) == 1:
        return factors[0].drop_unused()
    n = factors[0].codes.shape[0]
    missing = numpy.zeros(n, dtype=numpy.bool_)
    for f in factors:
        missing |= f.codes < 0
    combo = numpy.zeros(n, dtype=numpy.int64)
    sizes = [max(f.nlevels, 1) for f in factors]
    if lex_order:
        for f, size in zip(factors, sizes, strict=True):
            combo = combo * size + numpy.maximum(f.codes, 0)
    else:
        mult = 1
        for f, size in zip(factors, sizes, strict=True):
            combo = combo + mult * numpy.maximum(f.codes, 0)
            mult *= size
    uniq = numpy.unique(combo[~missing])
    codes = numpy.full(n, -1, dtype=numpy.intp)
    pos = {int(c): i for i, c in enumerate(uniq)}
    for i in numpy.flatnonzero(~missing):
        codes[i] = pos[int(combo[i])]
    labels: list[str] = []
    for c in uniq:
        parts: list[str] = []
        rest = int(c)
        if lex_order:
            for f, size in zip(reversed(factors), reversed(sizes), strict=True):
                parts.append(f.levels[rest % size])
                rest //= size
            parts.reverse()
        else:
            for f, size in zip(factors, sizes, strict=True):
                parts.append(f.levels[rest % size])
                rest //= size
        labels.append(sep.join(parts))
    return Factor(codes, tuple(labels))


def split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` outside of parentheses, brackets and quotes."""
    parts: list[str] = []
    depth = 0
    quote = ''
    cur: list[str] = []
    for ch in text:
        if quote:
            quote = '' if ch == quote else quote
        elif ch in ('"', "'"):
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == sep and depth == 0 and not quote:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur).strip())
    return parts


def evaluate_variable(expr: str, data: pandas.DataFrame) -> pandas.Series:
    """Evaluate one variable of a random-effects formula against ``data``.

    Supports column names, ``factor(x)``/``as.factor(x)``/``C(x)``, ``interaction(a, b)``
    (levels joined with ``.``) and ``paste(a, b)``/``paste0(a, b)`` (joined with a space/nothing).

    Raises:
        metafopy.errors.NotFoundError: a variable is not in ``data``.
    """
    name = expr.strip()
    if name in data.columns:
        return data[name].reset_index(drop=True)
    numeric = NUMERIC.match(name)
    if numeric is not None:
        return evaluate_variable(numeric.group(1), data).astype(float)
    wrapped = WRAPPERS.match(name)
    if wrapped is not None:
        inner = evaluate_variable(wrapped.group(1), data)
        if name.startswith(('factor(', 'as.factor(', 'C(')):
            return to_categorical(to_factor(inner))
        if name.startswith(('as.character(',)):
            return inner.astype(str)
        return inner
    call = CALL.match(name)
    if call is not None:
        args = [a for a in split_top_level(call.group(2), ',') if not a.replace(' ', '').startswith('sep=')]
        parts = [evaluate_variable(a, data) for a in args]
        if call.group(1) == 'interaction':
            fac = interaction([to_factor(p) for p in parts], sep='.')
            return to_categorical(fac)
        sep = ' ' if call.group(1) == 'paste' else ''
        joined = parts[0].astype(str)
        for p in parts[1:]:
            joined = joined + sep + p.astype(str)
        return joined
    try:
        out = data.eval(name)
    except (NameError, SyntaxError, KeyError, ValueError, TypeError, pandas.errors.UndefinedVariableError) as exc:
        raise metafopy.errors.NotFoundError(f'cannot evaluate {name!r} (used in a random-effects formula) in the data') from exc
    return pandas.Series(numpy.asarray(out)).reset_index(drop=True)


def to_categorical(fac: Factor) -> pandas.Series:
    """The factor as a categorical Series (missing values stay missing)."""
    return pandas.Series(pandas.Categorical.from_codes(fac.codes, categories=pandas.Index(list(fac.levels))))


def indicator(fac: Factor, sparse: bool) -> FloatArray | scipy.sparse.csc_array:
    """The model matrix ``~ 0 + f``: one 0/1 column per level (a single column of ones for a one-level factor)."""
    n = fac.codes.shape[0]
    if fac.nlevels <= 1:
        ones = numpy.ones((n, 1))
        return scipy.sparse.csc_array(ones) if sparse else ones
    rows = numpy.arange(n)
    mat = scipy.sparse.csc_array((numpy.ones(n), (rows, fac.codes)), shape=(n, fac.nlevels))
    return mat if sparse else mat.toarray()
