"""The runtime of translated R scripts: the small part of R's base and stats that metafor's examples and tests use, with R's semantics
(1-based indexing, recycling, ``$`` on data frames), and the ``rcall`` dispatcher that sends every other call to metafopy.

A function that is not here and not in metafopy raises ``UnsupportedError``: that is a gap of the harness, counted as coverage and never reported as a
difference between the packages.
"""

import contextlib
import functools
import inspect
import io
import math
import typing
import warnings

import numpy
import pandas

import metafopy
import metafopy.core.rng
import metafopy.datasets
import metafopy.errors
from audit import pycall
from audit.valuetree import py_name

Value = typing.Any


class UnsupportedError(Exception):
    """The script uses something the harness cannot run in Python (not a difference between R and metafopy)."""


class MissingRFunctionError(Exception):
    """metafor has the function and metafopy has no callable for it."""


class Empty:
    """An empty argument, as in ``x[i, ]``."""


EMPTY = Empty()

METAFOR_FUNCTIONS: set[str] = set()


class Env:
    """The variables of a script. Lookups see the data scopes first (``with(data, ...)`` and the data of a model call), then the variables."""

    def __init__(self) -> None:
        self.vars: dict[str, Value] = {}
        self.scopes: list[typing.Any] = []

    def __getitem__(self, name: str) -> Value:
        for scope in reversed(self.scopes):
            if isinstance(scope, pandas.DataFrame) and name in scope.columns:
                return scope[name]
            if isinstance(scope, dict) and name in scope:
                return scope[name]
        if name in self.vars:
            return self.vars[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name.startswith('dat.') or name in metafopy.datasets.list_datasets():
            try:
                return metafopy.datasets.load(name).copy()
            except metafopy.errors.NotFoundError as e:
                raise UnsupportedError(f'dataset {name} is not bundled with metafopy') from e
        if name in BASE or name in METAFOR_FUNCTIONS:
            return lambda *a, **k: rcall(self, name, *a, **k)
        raise UnsupportedError(f'object {name} not found')

    def set(self, name: str, value: Value) -> Value:
        self.vars[name] = value
        return value


CONSTANTS: dict[str, Value] = {
    'T': True,
    'F': False,
    'pi': math.pi,
    'letters': numpy.array([chr(97 + i) for i in range(26)]),
    'LETTERS': numpy.array([chr(65 + i) for i in range(26)]),
}


def is_missing(x: Value) -> bool:
    return isinstance(x, Empty)


def truth(x: Value) -> bool:
    arr = numpy.asarray(x)
    if arr.size == 0:
        raise UnsupportedError('argument is of length zero')
    return bool(arr.ravel()[0])


def block(*values: Value) -> Value:
    return values[-1] if values else None


# -- vectors ------------------------------------------------------------------------------------------------------------------------------------


def vector(values: list[Value]) -> Value:
    flat: list[Value] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        elif isinstance(v, (numpy.ndarray, pandas.Series)):
            flat.extend(numpy.asarray(v).ravel().tolist())
        elif isinstance(v, dict):
            flat.extend(v.values())
        elif v is None:
            continue
        else:
            flat.append(v)
    if any(isinstance(v, str) for v in flat):
        return numpy.array([v if isinstance(v, str) else format_number(v) for v in flat], dtype=object)
    if flat and all(isinstance(v, (bool, numpy.bool_)) for v in flat):
        return numpy.array(flat, dtype=bool)
    return numpy.array(flat, dtype=float)


def format_number(v: Value) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def r_c(*args: Value, **named: Value) -> Value:
    if named:
        items = [(None, a) for a in args] + list(named.items())
        values = [v for _, v in items]
        out = vector(values)
        names = []
        for k, v in items:
            n = len(numpy.atleast_1d(numpy.asarray(v)))
            names.extend([k or ''] * n)
        return pandas.Series(out, index=names)
    if any(isinstance(a, dict) for a in args):
        merged: dict[str, Value] = {}
        for a in args:
            merged.update(a if isinstance(a, dict) else {str(i): v for i, v in enumerate(a)})
        return merged
    return vector(list(args))


def r_list(*args: Value, **named: Value) -> Value:
    if named and not args:
        return dict(named)
    if named:
        return {**{str(i + 1): a for i, a in enumerate(args)}, **named}
    return list(args)


def r_rep(x: Value, times: Value = 1, each: Value = 1, length_out: Value = None, **kw: Value) -> Value:
    length_out = kw.get('length.out', length_out)
    base = numpy.asarray(x)
    if isinstance(each, (int, float)) and each != 1:
        base = numpy.repeat(base, int(each))
    if isinstance(times, (list, numpy.ndarray)) and len(numpy.atleast_1d(times)) > 1:
        out = numpy.repeat(base, numpy.asarray(times).astype(int))
    else:
        out = numpy.tile(base, int(numpy.asarray(times).ravel()[0]))
    if length_out is not None:
        out = numpy.resize(out, int(length_out))
    return out


def r_seq(*args: Value, **kw: Value) -> Value:
    by = kw.get('by')
    length_out = kw.get('length.out', kw.get('length_out'))
    if 'along.with' in kw:
        return numpy.arange(1, len(kw['along.with']) + 1)
    if len(args) == 1:
        a = numpy.asarray(args[0])
        if a.size == 1:
            return numpy.arange(1, int(a.ravel()[0]) + 1)
        return numpy.arange(1, a.size + 1)
    start = float(args[0]) if args else float(kw.get('from', 1))
    stop = float(args[1]) if len(args) > 1 else kw.get('to')
    if length_out is not None:
        if stop is None:
            return start + (float(by) if by is not None else 1.0) * numpy.arange(int(length_out))
        return numpy.linspace(start, float(stop), int(length_out))
    end = float(typing.cast(float, stop))
    step = float(by) if by is not None else (1.0 if end >= start else -1.0)
    n = math.floor((end - start) / step + 1e-10) + 1
    return start + step * numpy.arange(n)


def r_colon(a: Value, b: Value) -> Value:
    lo, hi = float(numpy.asarray(a).ravel()[0]), float(numpy.asarray(b).ravel()[0])
    n = math.floor(abs(hi - lo) + 1e-10) + 1
    return lo + numpy.arange(n) * (1 if hi >= lo else -1)


def as_array(x: Value) -> Value:
    if isinstance(x, (pandas.Series, pandas.DataFrame)):
        return x.to_numpy()
    return numpy.asarray(x)


def r_length(x: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        return x.shape[1]
    if isinstance(x, (dict, list)):
        return len(x)
    return int(numpy.asarray(x).size)


def r_round(x: Value, digits: Value = 0) -> Value:
    return numpy.round(as_array(x).astype(float) if not isinstance(x, (int, float)) else x, int(digits))


def r_sum(*xs: Value, **kw: Value) -> Value:
    narm = kw.get('na.rm', False)
    total = 0.0
    for x in xs:
        a = as_array(x).astype(float)
        total += float(numpy.nansum(a)) if narm else float(a.sum())
    return total


def r_mean(x: Value, **kw: Value) -> Value:
    a = as_array(x).astype(float)
    return float(numpy.nanmean(a)) if kw.get('na.rm') else float(a.mean())


def r_sd(x: Value, **kw: Value) -> Value:
    a = as_array(x).astype(float)
    return float(numpy.nanstd(a, ddof=1)) if kw.get('na.rm') else float(a.std(ddof=1))


def r_var(x: Value, y: Value = None, **kw: Value) -> Value:
    a = as_array(x).astype(float)
    if a.ndim == 2 or y is not None:
        return numpy.cov(a, rowvar=False) if y is None else numpy.cov(a, as_array(y).astype(float))[0, 1]
    return float(a.var(ddof=1))


def r_max(*xs: Value, **kw: Value) -> Value:
    a = numpy.concatenate([as_array(x).astype(float).ravel() for x in xs])
    return float(numpy.nanmax(a)) if kw.get('na.rm') else float(a.max())


def r_min(*xs: Value, **kw: Value) -> Value:
    a = numpy.concatenate([as_array(x).astype(float).ravel() for x in xs])
    return float(numpy.nanmin(a)) if kw.get('na.rm') else float(a.min())


def r_ifelse(test: Value, yes: Value, no: Value) -> Value:
    return numpy.where(as_array(test).astype(bool), as_array(yes), as_array(no))


def r_isna(x: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        return x.isna().to_numpy()
    return pandas.isna(numpy.asarray(x, dtype=object) if isinstance(x, list) else x)


def r_which(x: Value) -> Value:
    return numpy.flatnonzero(as_array(x).astype(bool)) + 1


def r_paste(*xs: Value, sep: Value = ' ', collapse: Value = None, **kw: Value) -> Value:
    parts = [numpy.atleast_1d(numpy.asarray(x, dtype=object)) for x in xs]
    n = max((len(p) for p in parts), default=0)
    rows = [str(sep).join(format_cell(p[i % len(p)]) for p in parts) for i in range(n)]
    if collapse is not None:
        return str(collapse).join(rows)
    return numpy.array(rows, dtype=object)


def format_cell(v: Value) -> str:
    if isinstance(v, (float, numpy.floating)):
        return format_number(v)
    return str(v)


def r_paste0(*xs: Value, **kw: Value) -> Value:
    return r_paste(*xs, sep='', **kw)


def r_names(x: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        return numpy.array(list(x.columns), dtype=object)
    if isinstance(x, pandas.Series):
        return numpy.array(list(x.index), dtype=object)
    if isinstance(x, dict):
        return numpy.array(list(x.keys()), dtype=object)
    names = getattr(x, 'coef_names', None)
    return None if names is None else numpy.array(list(names), dtype=object)


def r_matrix(data: Value = 0, nrow: Value = None, ncol: Value = None, byrow: Value = False, **kw: Value) -> Value:
    a = numpy.asarray(data, dtype=float).ravel()
    byrow = kw.get('byrow', byrow)
    if nrow is None and ncol is None:
        nrow, ncol = a.size, 1
    elif nrow is None:
        nrow = math.ceil(a.size / int(ncol))
    elif ncol is None:
        ncol = math.ceil(a.size / int(nrow))
    nrow, ncol = int(nrow), int(ncol)
    a = numpy.resize(a, nrow * ncol)
    return a.reshape((nrow, ncol)) if byrow else a.reshape((ncol, nrow)).T


def r_diag(x: Value, **kw: Value) -> Value:
    a = numpy.asarray(x, dtype=float)
    if a.ndim == 2:
        return numpy.diag(a)
    if a.size == 1:
        return numpy.eye(int(a.ravel()[0]))
    return numpy.diag(a)


def r_cbind(*xs: Value, **named: Value) -> Value:
    cols: dict[str, Value] = {}
    frame = any(isinstance(x, pandas.DataFrame) for x in list(xs) + list(named.values()))
    for x in xs:
        if isinstance(x, pandas.DataFrame):
            for c in x.columns:
                cols[str(c)] = x[c].to_numpy()
        else:
            a = numpy.asarray(x)
            if a.ndim == 2:
                for j in range(a.shape[1]):
                    cols[f'V{len(cols) + 1}'] = a[:, j]
            else:
                cols[f'V{len(cols) + 1}'] = a
    for k, v in named.items():
        cols[k] = numpy.asarray(v)
    if frame:
        return pandas.DataFrame(cols)
    return numpy.column_stack(list(cols.values()))


def r_rbind(*xs: Value, **named: Value) -> Value:
    parts = list(xs) + list(named.values())
    if any(isinstance(p, pandas.DataFrame) for p in parts):
        return pandas.concat([p if isinstance(p, pandas.DataFrame) else pandas.DataFrame([numpy.asarray(p)]) for p in parts], ignore_index=True)
    return numpy.vstack([numpy.atleast_2d(numpy.asarray(p, dtype=float)) for p in parts])


def r_dataframe(*xs: Value, **named: Value) -> Value:
    kw = {k: v for k, v in named.items() if k not in ('stringsAsFactors', 'check.names')}
    cols: dict[str, Value] = {}
    for x in xs:
        if isinstance(x, pandas.DataFrame):
            for c in x.columns:
                cols[str(c)] = x[c].to_numpy()
        else:
            cols[f'V{len(cols) + 1}'] = numpy.asarray(x)
    for k, v in kw.items():
        cols[k] = as_array(v)
    n = max((len(numpy.atleast_1d(v)) for v in cols.values()), default=0)
    return pandas.DataFrame({k: numpy.resize(numpy.atleast_1d(v), n) for k, v in cols.items()})


def r_dim(x: Value) -> Value:
    return numpy.array(x.shape if isinstance(x, (pandas.DataFrame, numpy.ndarray)) else (len(x),))


def r_t(x: Value) -> Value:
    return numpy.asarray(x, dtype=float).T if numpy.ndim(x) == 2 else numpy.asarray(x, dtype=float).reshape(1, -1)


def r_solve(a: Value, b: Value = None) -> Value:
    m = numpy.asarray(a, dtype=float)
    return numpy.linalg.inv(m) if b is None else numpy.linalg.solve(m, numpy.asarray(b, dtype=float))


def r_matmul(a: Value, b: Value) -> Value:
    return numpy.asarray(a, dtype=float) @ numpy.asarray(b, dtype=float)


def r_sapply(x: Value, fun: typing.Callable[..., Value], *args: Value, **kw: Value) -> Value:
    items = (
        list(x.values())
        if isinstance(x, dict)
        else (list(x.columns) if False else list(numpy.atleast_1d(x) if not isinstance(x, (list, pandas.DataFrame)) else x))
    )
    if isinstance(x, pandas.DataFrame):
        items = [x[c] for c in x.columns]
    out = [fun(item, *args) for item in items]
    try:
        return numpy.array(out, dtype=float)
    except TypeError, ValueError:
        return out


def r_lapply(x: Value, fun: typing.Callable[..., Value], *args: Value, **kw: Value) -> Value:
    if isinstance(x, dict):
        return {k: fun(v, *args) for k, v in x.items()}
    if isinstance(x, pandas.DataFrame):
        return {c: fun(x[c], *args) for c in x.columns}
    return [fun(item, *args) for item in numpy.atleast_1d(x).tolist()]


def r_split(x: Value, f: Value) -> Value:
    keys = numpy.asarray(f)
    levels = sorted(set(keys.tolist()), key=lambda v: (str(type(v)), v))
    out: dict[str, Value] = {}
    for level in levels:
        mask = keys == level
        label = format_cell(level)
        out[label] = x[mask] if isinstance(x, pandas.DataFrame) else numpy.asarray(x)[mask]
    return out


def r_unique(x: Value) -> Value:
    return pandas.unique(numpy.asarray(x).ravel())


def r_order(x: Value, **kw: Value) -> Value:
    a = numpy.asarray(x)
    order = numpy.argsort(a, kind='stable')
    return (order[::-1] if kw.get('decreasing') else order) + 1


def r_sort(x: Value, **kw: Value) -> Value:
    a = numpy.sort(numpy.asarray(x))
    return a[::-1] if kw.get('decreasing') else a


def r_rev(x: Value) -> Value:
    return numpy.asarray(x)[::-1]


def r_cumsum(x: Value) -> Value:
    return numpy.cumsum(as_array(x).astype(float))


def r_as_numeric(x: Value) -> Value:
    a = as_array(x)
    return a.astype(float) if a.dtype.kind != 'U' else numpy.array([float(v) for v in a])


def r_as_vector(x: Value, mode: Value = 'any') -> Value:
    return as_array(x).ravel(order='F') if isinstance(x, (numpy.ndarray, pandas.DataFrame)) else x


def r_as_matrix(x: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        return x.to_numpy()
    a = numpy.asarray(x)
    return a.reshape(-1, 1) if a.ndim < 2 else a


def r_as_dataframe(x: Value, **kw: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        return x
    if hasattr(x, 'to_frame'):
        return x.to_frame()
    return pandas.DataFrame(x)


def r_factor(x: Value, levels: Value = None, labels: Value = None, **kw: Value) -> Value:
    return pandas.Categorical(numpy.asarray(x), categories=None if levels is None else list(numpy.asarray(levels)))


def r_head(x: Value, n: Value = 6, **kw: Value) -> Value:
    n = int(n)
    return x.head(n) if isinstance(x, pandas.DataFrame) else numpy.asarray(x)[:n]


def r_tail(x: Value, n: Value = 6, **kw: Value) -> Value:
    n = int(n)
    return x.tail(n) if isinstance(x, pandas.DataFrame) else numpy.asarray(x)[-n:]


def r_nrow(x: Value) -> Value:
    return x.shape[0] if hasattr(x, 'shape') else len(x)


def r_ncol(x: Value) -> Value:
    return x.shape[1] if hasattr(x, 'shape') and len(x.shape) > 1 else None


def r_any(*xs: Value, **kw: Value) -> Value:
    return bool(any(numpy.asarray(x).astype(bool).any() for x in xs))


def r_all(*xs: Value, **kw: Value) -> Value:
    return bool(all(numpy.asarray(x).astype(bool).all() for x in xs))


def r_in(a: Value, b: Value) -> Value:
    return numpy.isin(numpy.asarray(a), numpy.asarray(b))


def r_identical(a: Value, b: Value) -> Value:
    try:
        return bool(numpy.array_equal(numpy.asarray(a), numpy.asarray(b)))
    except Exception:
        return a == b


def r_set_seed(seed: Value, **kw: Value) -> Value:
    metafopy.set_seed(int(seed))
    return None


def r_rnorm(n: Value, mean: Value = 0.0, sd: Value = 1.0) -> Value:
    z = metafopy.core.rng.get_rng().rnorm(int(n))
    return numpy.asarray(mean) + numpy.asarray(sd) * numpy.asarray(z)


def r_runif(n: Value, min: Value = 0.0, max: Value = 1.0) -> Value:
    u = metafopy.core.rng.get_rng().runif(int(n))
    return float(min) + (float(max) - float(min)) * numpy.asarray(u)


def r_capture_output(thunk: typing.Callable[[], Value], **kw: Value) -> Value:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = thunk()
        if value is not None and not isinstance(value, bool):
            pass
    return numpy.array(buf.getvalue().splitlines(), dtype=object)


def r_print(x: Value, *args: Value, **kw: Value) -> Value:
    print(x)
    return x


def r_noop(*args: Value, **kw: Value) -> Value:
    return None


def r_invisible(x: Value = None) -> Value:
    return x


def r_structure(x: Value, **kw: Value) -> Value:
    return x


def r_suppress(thunk: typing.Callable[[], Value]) -> Value:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return thunk()


def r_expect_error(thunk: typing.Callable[[], Value], *args: Value, **kw: Value) -> Value:
    try:
        thunk()
    except UnsupportedError, MissingRFunctionError:
        raise
    except Exception:
        return True
    return False


def r_try(thunk: typing.Callable[[], Value], *args: Value, **kw: Value) -> Value:
    try:
        return thunk()
    except UnsupportedError, MissingRFunctionError:
        raise
    except Exception as e:
        return f'Error: {e}'


def r_first(object: Value = None, *args: Value, **kw: Value) -> Value:
    return object


def r_with(data: Value, thunk: typing.Callable[[], Value], env: Env) -> Value:
    env.scopes.append(data)
    try:
        return thunk()
    finally:
        env.scopes.pop()


def r_cov2cor(x: Value) -> Value:
    a = numpy.asarray(x, dtype=float)
    d = numpy.sqrt(numpy.diag(a))
    return a / numpy.outer(d, d)


def r_log(x: Value, base: Value = None) -> Value:
    return numpy.log(as_array(x).astype(float)) if base is None else numpy.log(as_array(x).astype(float)) / math.log(float(base))


def r_attr(x: Value, which: Value, **kw: Value) -> Value:
    attrs = getattr(x, 'attrs', None)
    return attrs.get(str(which)) if isinstance(attrs, dict) else None


def r_levels(x: Value) -> Value:
    return numpy.array(list(pandas.Categorical(x).categories), dtype=object)


def r_nlevels(x: Value) -> Value:
    return len(pandas.Categorical(x).categories)


def r_is_null(x: Value) -> Value:
    return x is None


def r_stop(*args: Value, **kw: Value) -> Value:
    raise ValueError(''.join(str(a) for a in args))


def r_stopifnot(*args: Value, **kw: Value) -> Value:
    for a in args:
        if not numpy.all(numpy.asarray(a)):
            raise ValueError('stopifnot failed')
    return None


def r_nchar(x: Value) -> Value:
    return numpy.array([len(str(v)) for v in numpy.atleast_1d(x)])


def r_table(*xs: Value, **kw: Value) -> Value:
    values, counts = numpy.unique(numpy.asarray(xs[0]), return_counts=True)
    return pandas.Series(counts, index=[str(v) for v in values])


def r_cor(x: Value, y: Value = None, **kw: Value) -> Value:
    a = as_array(x).astype(float)
    if y is None:
        return numpy.corrcoef(a, rowvar=False)
    return float(numpy.corrcoef(a, as_array(y).astype(float))[0, 1])


def r_range(*xs: Value, **kw: Value) -> Value:
    a = numpy.concatenate([as_array(x).astype(float).ravel() for x in xs])
    return numpy.array([numpy.nanmin(a), numpy.nanmax(a)])


def r_unlist(x: Value, **kw: Value) -> Value:
    if isinstance(x, dict):
        return vector(list(x.values()))
    return vector(list(x))


def r_class(x: Value) -> Value:
    return numpy.array([type(x).__name__], dtype=object)


def r_inherits(x: Value, what: Value) -> Value:
    return type(x).__name__ in set(numpy.atleast_1d(what).tolist())


def r_exp(x: Value) -> Value:
    return numpy.exp(as_array(x).astype(float))


def r_sqrt(x: Value) -> Value:
    return numpy.sqrt(as_array(x).astype(float))


def r_abs(x: Value) -> Value:
    return numpy.abs(as_array(x).astype(float))


# The settings a script makes with ``options()`` that a Python argument stands for; ``run_script`` clears them for each script.
OPTIONS: dict[str, Value] = {}


def r_options(*args: Value, **kw: Value) -> Value:
    """``options(na.action = "na.pass")`` is kept and handed to the functions that take ``na_action`` (see ``rcall``); other settings are ignored."""
    for key, value in kw.items():
        if key == 'na.action' or key == 'na_action':
            OPTIONS['na.action'] = value
    return None


def r_apply(x: Value, margin: Value, fun: typing.Callable[..., Value], *args: Value) -> Value:
    a = as_array(x)
    axis = int(margin) - 1
    return numpy.array([fun(numpy.take(a, i, axis=1 - axis), *args) for i in range(a.shape[1 - axis])])


def r_sample(x: Value, size: Value = None, **kw: Value) -> Value:
    raise UnsupportedError("sample() draws from R's sampler")


def r_mapply(fun: typing.Callable[..., Value], *xs: Value, **kw: Value) -> Value:
    return numpy.array([fun(*items) for items in zip(*[numpy.atleast_1d(x) for x in xs], strict=False)])


def r_toupper(x: Value) -> Value:
    return numpy.array([str(v).upper() for v in numpy.atleast_1d(x)], dtype=object)


EXPECTATIONS = [
    'expect_condition',
    'expect_equal',
    'expect_equivalent',
    'expect_false',
    'expect_gt',
    'expect_gte',
    'expect_identical',
    'expect_length',
    'expect_lt',
    'expect_lte',
    'expect_match',
    'expect_message',
    'expect_named',
    'expect_no_error',
    'expect_no_warning',
    'expect_null',
    'expect_output',
    'expect_s3_class',
    'expect_setequal',
    'expect_silent',
    'expect_true',
    'expect_type',
    'expect_vector',
    'expect_warning',
]

BASE: dict[str, typing.Callable[..., Value]] = {
    '.': r_noop,
    'abs': r_abs,
    'all': r_all,
    'any': r_any,
    'apply': r_apply,
    'as.data.frame': r_as_dataframe,
    'as.matrix': r_as_matrix,
    'as.numeric': r_as_numeric,
    'as.vector': r_as_vector,
    'attr': r_attr,
    'c': r_c,
    'cat': r_noop,
    'cbind': r_cbind,
    'class': r_class,
    'context': r_noop,
    'cor': r_cor,
    'cov2cor': r_cov2cor,
    'cumsum': r_cumsum,
    'data.frame': r_dataframe,
    'dev.off': r_noop,
    'diag': r_diag,
    'dim': r_dim,
    'exp': r_exp,
    'factor': r_factor,
    'head': r_head,
    'identical': r_identical,
    'ifelse': r_ifelse,
    'inherits': r_inherits,
    'invisible': r_invisible,
    'is.na': r_isna,
    'is.null': r_is_null,
    'lapply': r_lapply,
    'length': r_length,
    'levels': r_levels,
    'library': r_noop,
    'list': r_list,
    'log': r_log,
    'mapply': r_mapply,
    'matrix': r_matrix,
    'max': r_max,
    'mean': r_mean,
    'message': r_noop,
    'min': r_min,
    'names': r_names,
    'nchar': r_nchar,
    'ncol': r_ncol,
    'nlevels': r_nlevels,
    'nrow': r_nrow,
    'options': r_options,
    'order': r_order,
    'par': r_noop,
    'paste': r_paste,
    'paste0': r_paste0,
    'png': r_noop,
    'print': r_print,
    'range': r_range,
    'rbind': r_rbind,
    'rep': r_rep,
    'require': r_noop,
    'rev': r_rev,
    'rm': r_noop,
    'rnorm': r_rnorm,
    'round': r_round,
    'runif': r_runif,
    'sapply': r_sapply,
    'sd': r_sd,
    'seq': r_seq,
    'set.seed': r_set_seed,
    'skip_on_cran': r_noop,
    'solve': r_solve,
    'sort': r_sort,
    'split': r_split,
    'sqrt': r_sqrt,
    'stop': r_stop,
    'stopifnot': r_stopifnot,
    'suppressMessages': r_suppress,
    'suppressWarnings': r_suppress,
    'try': r_try,
    'expect_error': r_expect_error,
    'capture.output': r_capture_output,
    'structure': r_structure,
    'sum': r_sum,
    't': r_t,
    'table': r_table,
    'tail': r_tail,
    'toupper': r_toupper,
    'unique': r_unique,
    'unlist': r_unlist,
    'var': r_var,
    'vapply': r_sapply,
    'which': r_which,
}


# -- indexing and operators ---------------------------------------------------------------------------------------------------------------------


def to_positions(index: Value, size: int) -> Value:
    """An R subscript as 0-based positions (or a boolean mask)."""
    if isinstance(index, Empty):
        return slice(None)
    arr = numpy.asarray(index)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in 'US' or arr.dtype == object:
        return arr
    pos = arr.astype(int)
    if (pos < 0).all() and pos.size:
        keep = numpy.ones(size, dtype=bool)
        keep[(-pos) - 1] = False
        return keep
    return pos - 1


def r_index(x: Value, *idx: Value, **kw: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        if len(idx) == 1:
            sel = idx[0]
            if isinstance(sel, Empty):
                return x
            arr = numpy.asarray(sel)
            if arr.dtype.kind in 'US' or arr.dtype == object:
                return x[list(arr)]
            return x.iloc[:, to_positions(sel, x.shape[1])]
        rows, cols = idx
        out: typing.Any = x
        if not isinstance(rows, Empty):
            out = (
                out.loc[numpy.asarray(rows)]
                if (numpy.asarray(rows).dtype == bool and False)
                else out.iloc[to_positions(rows, x.shape[0])]
                if numpy.asarray(rows).dtype != bool
                else out[numpy.asarray(rows)]
            )
        if not isinstance(cols, Empty):
            arr = numpy.asarray(cols)
            if arr.dtype.kind in 'US' or arr.dtype == object:
                out = out[list(arr)]
            else:
                out = out.iloc[:, to_positions(cols, x.shape[1])]
        if kw.get('drop', True) and out.shape[1] == 1:
            return out.iloc[:, 0]
        return out
    if isinstance(x, pandas.Series):
        sel = idx[0]
        arr = numpy.asarray(sel)
        if arr.dtype.kind in 'US' or arr.dtype == object:
            return x[list(arr)]
        return x.iloc[to_positions(sel, len(x))]
    if isinstance(x, dict):
        key = idx[0]
        return {k: x[k] for k in numpy.atleast_1d(key).tolist()}
    a = numpy.asarray(x)
    if len(idx) == 1:
        return a.ravel()[to_positions(idx[0], a.size)] if a.ndim < 2 else a.ravel(order='F')[to_positions(idx[0], a.size)]
    a2 = a if a.ndim == 2 else a.reshape(-1, 1)
    rows = to_positions(idx[0], a2.shape[0])
    cols = to_positions(idx[1], a2.shape[1])
    out = a2[rows][:, cols] if not isinstance(rows, slice) or not isinstance(cols, slice) else a2
    if kw.get('drop', True) and 1 in out.shape:
        return out.ravel()
    return out


def r_index2(x: Value, i: Value, *rest: Value) -> Value:
    if isinstance(x, dict):
        if isinstance(i, str):
            return x[i]
        return list(x.values())[int(numpy.asarray(i).ravel()[0]) - 1]
    if isinstance(x, pandas.DataFrame):
        if rest:
            return x.iat[int(numpy.asarray(i).ravel()[0]) - 1, int(numpy.asarray(rest[0]).ravel()[0]) - 1]
        return x[i] if isinstance(i, str) else x.iloc[:, int(numpy.asarray(i).ravel()[0]) - 1]
    if isinstance(x, list):
        return x[int(numpy.asarray(i).ravel()[0]) - 1]
    if isinstance(i, str):
        found, value = lookup_name(x, i)
        if found:
            return value
        raise UnsupportedError(f'no element {i}')
    return numpy.asarray(x).ravel()[int(numpy.asarray(i).ravel()[0]) - 1]


def lookup_name(x: Value, name: str) -> tuple[bool, Value]:
    if isinstance(x, pandas.Series) and name in x.index:
        return True, x[name]
    for spelling in (name, py_name(name)):
        if isinstance(x, dict) and spelling in x:
            return True, x[spelling]
        if isinstance(x, pandas.DataFrame) and spelling in x.columns:
            return True, x[spelling]
        if not isinstance(x, (dict, pandas.DataFrame)):
            try:
                return True, getattr(x, spelling)
            except AttributeError:
                continue
            except Exception as e:
                raise e
    return False, None


def r_dollar(x: Value, name: str) -> Value:
    found, value = lookup_name(x, name)
    if found:
        return value
    if isinstance(x, (dict, pandas.DataFrame)):
        return None
    raise AttributeError(f'{type(x).__name__} has no field {name}')


def r_assign_index(x: Value, value: Value, *idx: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        out = x.copy()
        if len(idx) == 2:
            rows, cols = idx
            r = to_positions(rows, out.shape[0])
            c = to_positions(cols, out.shape[1])
            cname = list(numpy.asarray(cols)) if numpy.asarray(cols).dtype.kind in 'US' or numpy.asarray(cols).dtype == object else None
            if cname is not None:
                out.loc[out.index[r], cname] = value
            else:
                out.iloc[r, c] = value
            return out
        return out
    a = numpy.array(x, dtype=float if not isinstance(x, (list, dict)) else object, copy=True)
    pos = to_positions(idx[0], a.size)
    a[pos] = value
    return a


def r_assign_dollar(x: Value, name: str, value: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        out = x.copy()
        out.attrs = dict(x.attrs)
        out[name] = value
        return out
    if isinstance(x, dict):
        out2 = dict(x)
        out2[name] = value
        return out2
    setattr(x, py_name(name), value)
    return x


def r_assign_names(x: Value, value: Value) -> Value:
    if isinstance(x, pandas.DataFrame):
        out = x.copy()
        out.columns = [str(v) for v in numpy.atleast_1d(value)]
        return out
    if isinstance(x, pandas.Series):
        out_s = x.copy()
        out_s.index = [str(v) for v in numpy.atleast_1d(value)]
        return out_s
    if isinstance(x, dict):
        return dict(zip([str(v) for v in numpy.atleast_1d(value)], x.values(), strict=False))
    return x


def r_assign_dimnames(x: Value, which: str, value: Value) -> Value:
    return x


OPERATORS: dict[str, typing.Callable[..., Value]] = {
    '+': lambda a, b=None: numpy.add(a, b) if b is not None else numpy.asarray(a),
    '-': lambda a, b=None: numpy.subtract(a, b) if b is not None else numpy.negative(a),
    '*': numpy.multiply,
    '/': numpy.true_divide,
    '^': numpy.power,
    '%%': numpy.mod,
    '%/%': numpy.floor_divide,
    '==': numpy.equal,
    '!=': numpy.not_equal,
    '<': numpy.less,
    '>': numpy.greater,
    '<=': numpy.less_equal,
    '>=': numpy.greater_equal,
    '&': numpy.logical_and,
    '|': numpy.logical_or,
    '!': numpy.logical_not,
    ':': r_colon,
    '%in%': r_in,
    '%*%': r_matmul,
}


def rfor(env: Env, var: str, seq: Value, body: typing.Callable[[], Value]) -> Value:
    for item in numpy.atleast_1d(seq).tolist() if not isinstance(seq, list) else seq:
        env.set(var, item)
        body()
    return None


# -- calls --------------------------------------------------------------------------------------------------------------------------------------

INDEX_ARGUMENTS = frozenset({'btt', 'att', 'subset', 'which'})


def shift_arguments(kwargs: dict[str, Value]) -> dict[str, Value]:
    out: dict[str, Value] = {}
    for k, v in kwargs.items():
        if k in INDEX_ARGUMENTS and v is not None and not isinstance(v, (str, bool)):
            arr = numpy.asarray(v)
            if arr.dtype.kind in 'if' and arr.dtype != bool:
                data = kwargs.get('data')
                if arr.size and (arr < 0).all() and isinstance(data, pandas.DataFrame):
                    # R's negative indices leave rows out (subset = -2); the Python argument takes the rows to keep as a mask
                    keep = numpy.ones(len(data), dtype=bool)
                    keep[(-numpy.atleast_1d(arr)).astype(int) - 1] = False
                    v = keep.tolist()
                else:
                    v = (arr.astype(int) - 1).tolist() if arr.ndim else int(arr) - 1
        out[k] = v
    return out


BASE.update({name: r_first for name in EXPECTATIONS})


def run_settings(env: Env) -> None:
    env.vars['.tol'] = {
        k: 0.01 for k in ('est', 'coef', 'pred', 'se', 'test', 'pval', 'ci', 'var', 'cor', 'cov', 'sevar', 'fit', 'r2', 'het', 'inf', 'den', 'misc')
    }
    env.vars['.tol']['count'] = 0.0
    env.vars['.sparse'] = False
    env.vars['.vistest'] = lambda *a, **k: True


def rcall(env: Env, name: str, *args: Value, **kwargs: Value) -> Value:
    if name == 'source':
        run_settings(env)
        return None
    if name in OPERATORS:
        return OPERATORS[name](*args)
    if name in BASE:
        return BASE[name](*args, **kwargs)
    first = args[0] if args else None
    kwargs = shift_arguments(kwargs)
    try:
        fn = pycall.resolve_function(name, first)
    except pycall.MissingFunctionError as e:
        if name in METAFOR_FUNCTIONS:
            raise MissingRFunctionError(name) from e
        raise UnsupportedError(f'R function {name} is not available in the harness') from e
    mapped = {pycall.arg_name(k): v for k, v in kwargs.items()}
    pycall.check_arguments(fn, list(mapped))
    if 'na.action' in OPTIONS and 'na_action' not in mapped and accepts(fn, 'na_action'):
        mapped['na_action'] = pycall.option_value('na.action', OPTIONS['na.action'])
    elif 'na.action' in OPTIONS and 'na_action' not in mapped and accepts_any_keyword(fn) and name in NA_ACTION_GENERICS:
        # a generic such as blup() hands its keywords to the method for the class of the model
        with contextlib.suppress(TypeError):
            return call_with(fn, args, {**mapped, 'na_action': pycall.option_value('na.action', OPTIONS['na.action'])})
    return call_with(fn, args, mapped)


def call_with(fn: typing.Callable[..., Value], args: tuple[Value, ...], mapped: dict[str, Value]) -> Value:
    if isinstance(fn, functools.partial):
        return fn(*args[1:], **mapped)
    return fn(*args, **mapped)


# R functions that read ``getOption("na.action")`` and whose Python generic passes keywords on to the method (from metafor's sources)
NA_ACTION_GENERICS = frozenset(
    {
        'blup',
        'cooks.distance',
        'dfbetas',
        'fitted',
        'hatvalues',
        'influence',
        'leave1out',
        'ranef',
        'residuals',
        'resid',
        'rstandard',
        'rstudent',
        'weights',
        'predict',
        'simulate',
        'vcov',
    }
)


def accepts_any_keyword(fn: typing.Callable[..., Value]) -> bool:
    try:
        params = inspect.signature(fn.func if isinstance(fn, functools.partial) else fn).parameters
    except TypeError, ValueError:
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def accepts(fn: typing.Callable[..., Value], name: str) -> bool:
    try:
        return name in inspect.signature(fn.func if isinstance(fn, functools.partial) else fn).parameters
    except TypeError, ValueError:
        return False


def r_symbol(env: Env, name: str) -> Value:
    """A bare name in an argument of a model function: a column of the data (passed as its name), else the variable, function or constant of that name.

    ``rma(yi, vi, data = dat, ...)`` takes ``yi`` as a column, ``conv.delta(..., transf = transf.logit)`` takes ``transf.logit`` as the function.
    """
    for scope in reversed(env.scopes):
        if isinstance(scope, pandas.DataFrame) and name in scope.columns:
            return name
        if isinstance(scope, dict) and name in scope:
            return name
    try:
        return env[name]
    except UnsupportedError:
        return name


def rnse(env: Env, name: str, data: typing.Callable[[], Value], args: list[typing.Callable[[], Value]], kwargs: dict[str, typing.Callable[[], Value]]) -> Value:
    """A call whose arguments are evaluated with the columns of ``data`` in scope, as metafor's model functions do."""
    frame = data()
    env.scopes.append(frame)
    try:
        values = [a() for a in args]
        named = {k: v() for k, v in kwargs.items()}
    finally:
        env.scopes.pop()
    named['data'] = frame
    return rcall(env, name, *values, **named)


def pairs(items: list[tuple[str, Value]]) -> dict[str, Value]:
    return dict(items)


__all__ = ['BASE', 'EMPTY', 'Env', 'MissingRFunctionError', 'UnsupportedError', 'rcall']
