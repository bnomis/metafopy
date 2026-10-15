"""Nested factors of reference grids (emmeans' ``.find_nests``, ``.parse_nest``, ``.find.nonempty.nests`` and ``.std.order``).

A factor B is nested in a factor A when each level of B occurs with only one level of A (or, in the model, B has no main effect
and enters through ``A:B``). The reference grid then has combinations that cannot occur; they are hidden from summaries and left
out of the averages. R sources: https://github.com/rvlenth/emmeans/blob/master/R/ref-grid.R and
https://github.com/rvlenth/emmeans/blob/master/R/nested.R; the vignette at
https://rvlenth.github.io/emmeans/articles/sophisticated.html#nesting describes the structures. The coding of the terms follows
Chambers and Hastie (1992), *Statistical Models in S*, Chapter 2.
"""

import collections.abc
import dataclasses
import itertools
import re

import formulaic
import numpy
import numpy.typing
import pandas

import metafopy.errors

__all__ = ['TermInfo', 'find_nests', 'nonempty_display', 'parse_nest', 'std_order', 'term_info', 'union']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Nesting = dict[str, tuple[str, ...]]
type NestSpec = str | collections.abc.Mapping[str, str | collections.abc.Sequence[str]]

NAME = re.compile(r'[A-Za-z_.][\w.]*')


@dataclasses.dataclass(frozen=True, slots=True)
class TermInfo:
    """The ``factors`` attribute of R's ``terms``: for each term the variables in it, coded 1 (contrasts) or 2 (full dummy coding).

    ``main`` holds the variables that are terms of their own.
    """

    codes: tuple[dict[str, int], ...]
    main: frozenset[str]


def union(*lists: collections.abc.Iterable[str]) -> list[str]:
    """R's ``union``: the distinct elements of all the lists, in order of appearance."""
    return list(dict.fromkeys(itertools.chain.from_iterable(lists)))


def factor_id(expr: str, variables: collections.abc.Sequence[str]) -> str:
    """The variable a factor of a formula stands for (``C(x)`` is ``x``), else the expression itself."""
    text = expr.strip()
    for var in variables:
        if text == var or re.fullmatch(rf'C\(\s*{re.escape(var)}\s*(,.*)?\)', text):
            return var
    return text


def term_info(rhs: str, variables: collections.abc.Sequence[str]) -> TermInfo:
    """Code the variables in each term of the formula ``rhs`` as R's ``terms`` does.

    A variable in a term is coded 1 when the term without it is in the model (the empty term needs the intercept) and 2 otherwise.
    Transformed variables such as ``I(x ** 2)`` take part in the margins but are not reported.
    """
    terms = list(formulaic.Formula(rhs))
    intercept = any(str(t) == '1' for t in terms)
    sets = [tuple(factor_id(f.expr, variables) for f in t.factors) for t in terms if str(t) != '1']
    present = {frozenset(s) for s in sets}
    codes: list[dict[str, int]] = []
    main: set[str] = set()
    for ids in sets:
        entry: dict[str, int] = {}
        for var in ids:
            rest = frozenset(ids) - {var}
            coded = 1 if (rest in present or (not rest and intercept)) else 2
            if var in variables:
                entry[var] = coded
        codes.append(entry)
        if len(ids) == 1 and ids[0] in variables:
            main.add(ids[0])
    return TermInfo(tuple(codes), frozenset(main))


def strip_supersets(sets: list[list[str]]) -> list[str]:
    """Drop the sets that contain a smaller one and return the union of what is left (emmeans' ``.strip.supersets``)."""
    if len(sets) > 1 and len({len(s) for s in sets}) > 1:
        ordered = sorted(sets, key=len)
        for i in range(len(ordered) - 1, 0, -1):
            if any(set(ordered[j]) <= set(ordered[i]) for j in range(i)):
                del ordered[i]
        sets = ordered
    return union(*sets)


def find_nests(
    frame: pandas.DataFrame, wgt: FloatArray, levels: collections.abc.Mapping[str, collections.abc.Sequence[object]], terms: TermInfo | None
) -> Nesting:
    """The nesting structure of a reference grid, from the combinations that occur in the data and from the terms of the model.

    ``frame`` has the level of each variable in each row of the grid, ``wgt`` the number of studies behind each row and ``levels``
    the levels of each variable.
    """
    result: dict[str, list[str]] = {}
    names = [v for v, lev in levels.items() if len(lev) > 1]
    if len(names) < 2:
        return {}
    seen = frame.loc[wgt > 0, names]
    for nm in names:
        excluded = [lnm for lnm, parents in result.items() if nm in parents]
        others = [n for n in names if n != nm and n not in excluded]
        hits = [n for n in others if max(seen.loc[seen[nm] == lev, n].nunique() for lev in levels[nm]) == 1]
        if hits:
            result[nm] = hits
    if terms is not None and terms.codes:
        rows = [v for v in names if any(v in c for c in terms.codes)]
        for nm in [v for v in rows if v not in terms.main]:
            chosen = [c for c in terms.codes if c.get(nm) == 1]
            if not chosen:
                chosen = [{**c, nm: 1} for c in terms.codes if c.get(nm) == 2]
            parents = strip_supersets([[v for v in rows if c.get(v) == 2] for c in chosen])
            if parents:
                result[nm] = union(result.get(nm, []), parents)
    for nm in list(result):
        result[nm] = union([x for p in result[nm] if p in result for x in result[p]], result[nm])
    for nm in list(result):
        if nm in result[nm]:
            del result[nm]
    return {k: tuple(v) for k, v in result.items()}


def parse_nest(spec: NestSpec) -> Nesting:
    """A nesting structure from ``'B %in% A, C %in% (A*B)'`` or a mapping from names to the names they are nested in.

    Raises:
        metafopy.errors.InputError: the text cannot be read.
    """
    if isinstance(spec, collections.abc.Mapping):
        return {k: (v,) if isinstance(v, str) else tuple(v) for k, v in spec.items()}
    result: Nesting = {}
    for part in (p.strip() for p in spec.split(',')):
        if not part:
            continue
        pieces = re.split(r'\s+%in%\s+', part)
        if len(pieces) != 2 or len(NAME.findall(pieces[0])) != 1:
            raise metafopy.errors.InputError(f"Cannot read the nesting {part!r}; use 'B %in% A'.")
        result[NAME.findall(pieces[0])[0]] = tuple(dict.fromkeys(NAME.findall(pieces[1])))
    return result


def nonempty_display(
    frame: pandas.DataFrame,
    wgt: FloatArray,
    levels: collections.abc.Mapping[str, collections.abc.Sequence[object]],
    nesting: collections.abc.Mapping[str, collections.abc.Sequence[str]],
    xfacs: collections.abc.Sequence[str] | None = None,
) -> tuple[bool, ...]:
    """Which rows of the grid to show: those whose level of each nested factor and its groups occurs in the data."""
    keep = numpy.ones(len(frame), dtype=bool)
    for x in union(nesting, itertools.chain.from_iterable(nesting.values())) if xfacs is None else xfacs:
        facs = union([x], nesting.get(x, ()))
        labels = {f: frame[f].astype(str).to_numpy() for f in facs}
        for combo in itertools.product(*[[str(lev) for lev in levels[f]] for f in facs]):
            hit = numpy.ones(len(frame), dtype=bool)
            for f, lev in zip(facs, combo, strict=True):
                hit &= labels[f] == lev
            if hit.any() and bool(numpy.all(wgt[hit] == 0)):
                keep[hit] = False
    return tuple(bool(k) for k in keep)


def std_order(frame: pandas.DataFrame, levels: collections.abc.Mapping[str, collections.abc.Sequence[object]]) -> numpy.ndarray:
    """The row order of a standard grid: the first variable varies fastest and the levels follow their order."""
    keys: list[numpy.ndarray] = []
    for var, lev in levels.items():
        position = {str(label): i for i, label in enumerate(lev)}
        keys.append(numpy.array([position.get(str(x), len(position)) for x in frame[var]]))
    return numpy.lexsort(keys) if keys else numpy.arange(len(frame))
