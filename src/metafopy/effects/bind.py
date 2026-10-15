"""Combine ``escalc`` data frames and keep their attributes (metafor's ``cbind.escalc`` and ``rbind.escalc``).

The attributes of ``escalc`` output live in ``DataFrame.attrs`` (``yi_names``, ``vi_names``, ``sei_names``, ``zi_names``, ``pval_names``,
``ci_lb_names``, ``ci_ub_names``, ``digits``, ``measure``, ``slab`` and ``ni``; see :func:`metafopy.escalc`). ``pandas.concat`` keeps none of
them for a frame built from several, so :func:`cbind_escalc` and :func:`rbind_escalc` rebuild them the way metafor does.

R sources: https://github.com/wviechtb/metafor/blob/master/R/cbind.escalc.r and https://github.com/wviechtb/metafor/blob/master/R/rbind.escalc.r;
reference page: https://wviechtb.github.io/metafor/reference/print.escalc.html.

Differences from R: R keeps ``measure``, ``slab`` and ``ni`` as attributes of each ``yi`` column, so a combined frame may hold several measures.
A frame has only one such set here: :func:`cbind_escalc` takes them from the first argument that has them (the first argument of
:func:`rbind_escalc` supplies ``measure``, as in R).
"""

import typing

import numpy
import pandas

import metafopy.effects.measures
import metafopy.errors

__all__ = ['cbind_escalc', 'rbind_escalc']

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
NAME_KEYS = ('yi_names', 'vi_names', 'sei_names', 'zi_names', 'pval_names', 'ci_lb_names', 'ci_ub_names')
YI_KEYS = ('measure', 'slab', 'ni')


def as_names(value: typing.Any) -> list[str]:
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def make_unique_row_names(labels: list[typing.Any]) -> list[typing.Any]:
    """R's ``make.unique(labels, sep = "")`` for the row names of ``rbind()``."""
    seen = {str(x) for x in labels}
    counts: dict[str, int] = {}
    used: set[str] = set()
    out: list[typing.Any] = []
    for x in labels:
        key = str(x)
        if key not in used:
            used.add(key)
            out.append(x)
            continue
        n = counts.get(key, 0)
        while True:
            n += 1
            candidate = f'{key}{n}'
            if candidate not in seen and candidate not in used:
                break
        counts[key] = n
        used.add(candidate)
        out.append(candidate)
    return out


def bare(x: pandas.DataFrame) -> pandas.DataFrame:
    """A copy of ``x`` without attributes (``pandas.concat`` compares them with ``==``, which fails for arrays)."""
    out = x.copy()
    out.attrs = {}
    return out


def cbind_escalc(*args: pandas.DataFrame | pandas.Series, **columns: typing.Any) -> pandas.DataFrame:
    """Combine data frames (and further columns) side by side and keep the ``escalc`` attributes, as metafor's ``cbind()`` of ``escalc`` frames.

    ``args`` are data frames (usually from :func:`metafopy.escalc` or :func:`metafopy.summary_escalc`) or named Series; ``columns`` add vectors
    under the names of the keywords. The result has the columns of all arguments in order (names are not made unique, like
    ``data.frame(..., check.names = FALSE)``). The names of the effect sizes, sampling variances, standard errors, test statistics, p-values and
    confidence interval bounds of all arguments are collected (those of later arguments first, as in R), ``digits`` is that of the last argument
    and ``measure``, ``slab`` and ``ni`` come from the first argument that has them.

    Raises:
        metafopy.errors.InputError: no argument is given, or the arguments have different numbers of rows.
        metafopy.errors.ArgumentError: an argument is neither a data frame nor a named Series.
    """
    parts: list[pandas.DataFrame | pandas.Series] = [*args]
    for name, values in columns.items():
        parts.append(pandas.Series(numpy.asarray(values), name=name))
    if not parts:
        raise metafopy.errors.InputError('At least one argument is needed.')
    for part in parts:
        if isinstance(part, pandas.Series):
            if part.name is None:
                raise metafopy.errors.ArgumentError('A Series must be named to be combined as a column; give vectors as keyword arguments.')
        elif not isinstance(part, pandas.DataFrame):
            raise metafopy.errors.ArgumentError(f'Expected a data frame or a Series, got {type(part).__name__}.')
    sizes = [int(p.shape[0]) for p in parts]
    if len(set(sizes)) > 1:
        raise metafopy.errors.InputError(f'arguments imply differing number of rows: {", ".join(str(n) for n in dict.fromkeys(sizes))}')
    index = parts[0].index
    dat = pandas.concat([(bare(p) if isinstance(p, pandas.DataFrame) else p).reset_index(drop=True) for p in parts], axis=1)
    dat.index = index

    attrs: dict[str, typing.Any] = {}
    for key in NAME_KEYS:
        collected: list[str] = []
        for part in parts:
            if isinstance(part, pandas.DataFrame):
                # R's `[.escalc` drops the names of columns that are not in the frame, so a subset of columns only contributes its own
                collected = [name for name in as_names(part.attrs.get(key)) if name in part.columns] + collected
        names = unique(collected)
        if names:
            attrs[key] = names
    last = parts[-1]
    if isinstance(last, pandas.DataFrame) and last.attrs.get('digits') is not None:
        attrs['digits'] = last.attrs['digits']
    for part in parts:
        if isinstance(part, pandas.DataFrame) and 'measure' in part.attrs:
            attrs.update({key: part.attrs[key] for key in YI_KEYS if key in part.attrs})
            break
    dat.attrs = {}
    dat.attrs.update(attrs)
    return dat


def rbind_escalc(*args: pandas.DataFrame) -> pandas.DataFrame:
    """Stack data frames on top of each other and keep the ``escalc`` attributes, as metafor's ``rbind()`` of ``escalc`` frames.

    The columns are matched by name and the attributes of the first argument are kept (``measure``, ``digits``, the names of the effect size
    columns, ...). ``ni`` is the concatenation of the sample sizes of all arguments, or missing if one of them has none, and ``slab`` the
    concatenation of the study labels made unique as metafor's ``.make.unique`` does (every repeated label gets ``.1``, ``.2``, ...), or missing
    if one argument has none.

    Raises:
        metafopy.errors.InputError: no argument is given, or the numbers or names of the columns differ.
        metafopy.errors.ArgumentError: an argument is not a data frame.
    """
    if not args:
        raise metafopy.errors.InputError('At least one argument is needed.')
    for part in args:
        if not isinstance(part, pandas.DataFrame):
            raise metafopy.errors.ArgumentError(f'Expected a data frame, got {type(part).__name__}.')
    first = args[0]
    columns = list(first.columns)
    for part in args[1:]:
        if part.shape[1] != len(columns):
            raise metafopy.errors.InputError('numbers of columns of arguments do not match')
        if set(part.columns) != set(columns) or len(set(columns)) != len(columns):
            raise metafopy.errors.InputError('names do not match previous names')
    aligned = [bare(first)] + [bare(part[columns]) for part in args[1:]]
    automatic = all(isinstance(p.index, pandas.RangeIndex) and p.index.start == 0 and p.index.step == 1 for p in args)
    dat = pandas.concat(aligned, axis=0, ignore_index=automatic)
    if not automatic:
        dat.index = pandas.Index(make_unique_row_names(list(dat.index)))

    attrs = dict(first.attrs)
    sizes = [p.attrs.get('ni') for p in args]
    attrs['ni'] = None if any(v is None for v in sizes) else numpy.concatenate([numpy.asarray(v, dtype=numpy.float64).reshape(-1) for v in sizes])
    labels: list[str] = []
    complete = True
    for p in args:
        stored = p.attrs.get('slab')
        if stored is None:
            complete = False
            break
        labels.extend(str(s) for s in stored)
    if complete:
        attrs['slab'] = list(metafopy.effects.measures.make_unique_dot1(labels))
    else:
        attrs.pop('slab', None)
    dat.attrs = {}
    dat.attrs.update(attrs)
    return dat
