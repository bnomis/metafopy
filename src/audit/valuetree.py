"""Canonical value trees: the language-neutral form in which R and Python results are compared.

The R side builds the trees in ``r/common.R``; ``to_node`` here builds the same shape from a Python value, guided by the R tree so that only
what R returned is extracted (a Python result object has many more attributes than an R list has names).
"""

import dataclasses
import math
import typing

import numpy
import pandas

import audit.config
import metafopy.effects.reshape
import metafopy.metadata
import metafopy.methods.accessors

Node = dict[str, typing.Any]
Labels = list[str] | tuple[list[str], list[str]] | None

MAX_DEPTH = 6

# results that wrap the array R returns: the class name and the attribute that holds the array (llplot() returns the likelihoods, to.table() the tables)
ARRAY_RESULTS = {'LlplotResult': 'lls', 'TableArray': 'values'}


SCALE_FIELDS = frozenset({'alpha', 'se.alpha', 'zval.alpha', 'pval.alpha', 'ci.lb.alpha', 'ci.ub.alpha', 'va', 'H', 'Z', 'Z.f'})


STRUCTURE_FIELDS = ('G', 'H', 'hessian', 'vvc', 'V', 'M', 'W', 'V.f', 'W.f')


def py_name(name: str) -> str:
    """The Python spelling of an R name: dots become underscores."""
    return name.replace('.', '_')


def is_finite_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def num_cell(x: object) -> float | str | None:
    value = float(typing.cast(float, x))
    if math.isnan(value):
        return None
    if math.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return value


def label_fields(guide: Node, labels: Labels, shape: list[int] | None, length: int) -> Node:
    """The names of a vector or the row and column names of a matrix, when Python has labels of the length R's names have."""
    out: Node = {}
    if not labels:
        return out
    if shape and len(shape) == 2:
        rows, cols = labels if isinstance(labels, tuple) else (labels, labels)
        if guide.get('rn') and rows and shape[0] == len(rows):
            out['rn'] = list(rows)
        if guide.get('cn') and cols and shape[1] == len(cols):
            out['cn'] = list(cols)
    elif isinstance(labels, list) and length == len(labels):
        for key in ('n', 'rn'):
            if guide.get(key):
                out[key] = list(labels)
    return out


def vector_node(values: typing.Any, guide: Node, labels: Labels = None) -> Node:
    kind = guide['k']
    arr = numpy.asarray(values)
    dims = guide.get('d')
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if dims and len(dims) >= 2 and arr.ndim == len(dims):
        flat = arr.ravel(order='F')
        shape = list(arr.shape)
    elif dims and len(dims) == 3 and arr.ndim == 3:
        flat = arr.ravel(order='F')
        shape = None
    else:
        flat = arr.ravel()
        shape = None
    cells: list[typing.Any]
    if kind != 'str' and arr.dtype.kind in 'US':
        try:
            arr = arr.astype(float)
        except ValueError:
            return {'k': 'type', 'got': 'str'}
        flat = arr.ravel(order='F') if flat.ndim and shape else arr.ravel()
    if kind == 'str':
        cells = [None if (c is None or (isinstance(c, float) and math.isnan(c))) else str(c) for c in flat.tolist()]
    elif kind == 'lgl':
        cells = [None if (isinstance(c, float) and math.isnan(c)) or c is None else bool(c) for c in flat.tolist()]
    else:
        cells = []
        for c in flat.tolist():
            if c is None:
                cells.append(None)
            elif isinstance(c, str):
                return {'k': 'type', 'got': 'str'}
            else:
                cells.append(num_cell(c))
    return {'k': kind, 'v': cells, 'd': shape, **label_fields(guide, labels, shape, len(cells))}


def lookup(value: typing.Any, name: str) -> tuple[bool, typing.Any]:
    """Find the field of ``value`` that corresponds to the R name ``name``."""
    if name.startswith('@'):
        attrs = getattr(value, 'attrs', None)
        if isinstance(attrs, dict) and name[1:] in attrs:
            return True, attrs[name[1:]]
        return False, None
    spellings = [name, py_name(name), py_name(name) + '_']
    if name == '' and isinstance(value, (tuple, list)):
        return True, value  # a data frame column R leaves unnamed (the `hyp` of an anova)
    inputs = getattr(value, 'data', None)
    if isinstance(inputs, metafopy.metadata.MetaData) and name in ('data', 'slab'):
        # R's `data` is the data frame the model was given and its `slab` has one label for every row of it; in metafopy `data` holds the model
        # inputs of the studies used (`MetaData`) and `slab` the labels of those, so the frame and all labels are looked up in the inputs
        if name == 'data':
            return True, inputs.data
        if inputs.labels_all is not None:
            rows = inputs.keep if inputs.na_rows is None else inputs.keep | inputs.na_rows
            return True, [label for label, used in zip(inputs.labels_all, rows, strict=True) if used]
    if name == 'vcov' and callable(getattr(value, 'vcov', None)) and hasattr(value, 'vb'):
        # R keeps the matrix in `vcov`; in metafopy `vcov()` is the accessor and the matrix is `vb`
        return True, value.vb
    slab_f = getattr(value, 'slab_f', None)
    if name == 'slab' and slab_f is not None and len(slab_f) != len(getattr(value, 'slab', slab_f)):
        # rma.mv keeps the labels of all rows of the subset in `slab` in R; metafopy has the labels of the studies used in `slab` and all of them in `slab_f`
        return True, list(slab_f)
    if isinstance(value, pandas.DataFrame):
        for spelling in spellings:
            if spelling in value.columns:
                # R's `$` takes the first of several columns with one name
                return True, numpy.asarray(value.iloc[:, list(value.columns).index(spelling)])
            if spelling in value.attrs:
                return True, value.attrs[spelling]
        return False, None
    if isinstance(value, dict):
        for spelling in spellings:
            if spelling in value:
                return True, value[spelling]
        return False, None
    if isinstance(value, tuple) and not hasattr(value, '_fields'):
        # a list R names by its entries (the ``vif`` of ``vif()``): metafopy keeps a tuple of entries that carry their own name
        for item in value:
            if name in (getattr(item, 'coefname', None), getattr(item, 'coefs', None)):
                return True, item
    for spelling in spellings:
        try:
            return True, getattr(value, spelling)
        except AttributeError:
            continue
        except Exception:  # a property that raises is a field R has and this object cannot give
            return False, None
    return False, None


def named_vector(value: typing.Any, guide: Node, path: str) -> typing.Any:
    """A mapping stands for the named vector with its entries (names R's vector lacks are NA); a number for the ``digits`` of a model.

    R keeps ``digits`` as a named vector of nine numbers (``est``, ``se``, ``test``, ...) and returns summaries such as ``precis`` as named
    vectors; metafopy keeps one number for ``digits``, which stands for the vector in which every element is that number, and mappings.
    """
    if isinstance(value, bool) or value is None:
        return value
    if path.endswith('$digits') and isinstance(value, (int, float)):
        return [value] * len(guide['v'])
    if isinstance(value, metafopy.methods.accessors.NobsAll):
        return list(dataclasses.asdict(value).values())
    if isinstance(value, metafopy.effects.reshape.TableArray):
        return value.values
    if isinstance(value, dict) and guide.get('n') and set(value) <= set(guide['n']):
        return [value.get(name) for name in guide['n']]
    return value


def to_node(value: typing.Any, guide: Node, path: str, missing: list[str], depth: int = 0, labels: Labels = None) -> Node:
    """The tree of ``value`` in the shape of the R tree ``guide``; fields R has and ``value`` lacks are appended to ``missing``."""
    kind = guide['k']
    if kind in ('opaque', 'null'):
        return {'k': kind}
    if depth > MAX_DEPTH:
        return {'k': 'opaque'}
    if kind in ('num', 'str', 'lgl'):
        if type(value).__name__ in ARRAY_RESULTS:
            value = getattr(value, ARRAY_RESULTS[type(value).__name__])
        if value is None:
            return {'k': kind, 'v': [None] * len(guide['v']), 'd': None}
        value = named_vector(value, guide, path)
        if isinstance(value, dict):
            return {'k': 'type', 'got': type(value).__name__}
        if isinstance(value, pandas.Series):
            labels = [str(i) for i in value.index]
        if isinstance(value, pandas.DataFrame):
            labels = ([str(i) for i in value.index], [str(c) for c in value.columns])
        if isinstance(value, (pandas.DataFrame, pandas.Series)):
            value = value.to_numpy()
        try:
            return vector_node(value, guide, None if kind == 'str' else labels)
        except TypeError, ValueError:
            return {'k': 'type', 'got': type(value).__name__}
    if kind == 'df':
        cols = []
        for col in guide['cols']:
            name = col['name']
            found, sub = lookup(value, name)
            if not found:
                missing.append(f'{path}${name}')
                cols.append({'name': name, 'node': {'k': 'missing'}})
                continue
            cols.append({'name': name, 'node': to_node(sub, col['node'], f'{path}${name}', missing, depth + 1)})
        return {'k': 'df', 'cols': cols}
    if kind == 'list':
        items = []
        coef_names = getattr(value, 'coef_names', None)
        labels = [str(n) for n in coef_names] if coef_names is not None and not isinstance(coef_names, str) else None
        alpha_names = getattr(value, 'alpha_names', None)
        alpha_labels = [str(n) for n in alpha_names] if alpha_names is not None and not isinstance(alpha_names, str) else None
        for i, item in enumerate(guide['items']):
            name = item.get('name')
            if name is None:
                if isinstance(value, (list, tuple)) and i < len(value):
                    items.append({'name': None, 'node': to_node(value[i], item['node'], f'{path}[{i}]', missing, depth + 1)})
                else:
                    missing.append(f'{path}[{i}]')
                    items.append({'name': None, 'node': {'k': 'missing'}})
                continue
            found, sub = lookup(value, name)
            if not found:
                missing.append(f'{path}${name}')
                items.append({'name': name, 'node': {'k': 'missing'}})
                continue
            # the scale part of a location-scale model is labelled with the names of the scale coefficients
            item_labels = alpha_labels if alpha_labels is not None and name in SCALE_FIELDS else labels
            if name in STRUCTURE_FIELDS and (name != 'H' or hasattr(value, 'H_dimnames')):
                # the matrices of the random effects are not indexed by the coefficients: G and H by the levels of the inner factor, the rest unnamed
                dim = getattr(value, f'{name}_dimnames', None)
                item_labels = [str(n) for n in dim] if dim else None
            items.append({'name': name, 'node': to_node(sub, item['node'], f'{path}${name}', missing, depth + 1, item_labels)})
        return {'k': 'list', 'items': items}
    return {'k': 'missing'}


@dataclasses.dataclass(frozen=True)
class Diff:
    path: str
    kind: str
    detail: str


def straight(text: str) -> str:
    """The text with curly single quotes replaced by straight ones."""
    return text.replace('\u2018', "'").replace('\u2019', "'")


def cell_equal(a: typing.Any, b: typing.Any) -> tuple[str, float]:
    """Compare two cells: ('same'|'drift'|'diff', relative difference)."""
    if a is None and b is None:
        return 'same', 0.0
    if a is None or b is None:
        return 'diff', math.inf
    if isinstance(a, str) and isinstance(b, str):
        # printed lines (capture.output(print(x))) end in padding that R's print methods add and the tests of metafor never compare
        # and R prints the quotes of the significance codes as curly quotes in a UTF-8 locale
        return ('same', 0.0) if straight(a).rstrip() == straight(b).rstrip() else ('diff', math.inf)
    if isinstance(a, bool) or isinstance(b, bool):
        return ('same', 0.0) if bool(a) == bool(b) else ('diff', math.inf)
    if isinstance(a, str) or isinstance(b, str):
        return 'diff', math.inf
    delta = abs(a - b)
    scale = max(abs(a), abs(b))
    if delta <= audit.config.ATOL + audit.config.RTOL * scale:
        return 'same', 0.0
    rel = delta / scale if scale > 0 else math.inf
    return ('drift' if rel <= audit.config.DRIFT_RTOL else 'diff'), rel


def compare(r: Node, py: Node, path: str, out: list[Diff]) -> None:
    rk, pk = r['k'], py['k']
    if rk in ('opaque', 'null'):
        return
    if pk == 'missing':
        return
    if pk == 'type':
        out.append(Diff(path, 'type-diff', f'R {rk}, Python {py["got"]}'))
        return
    if pk == 'null':
        out.append(Diff(path, 'type-diff', f'R {rk}, Python None'))
        return
    if rk in ('num', 'str', 'lgl'):
        compare_vector(r, py, path, out)
    elif rk == 'df':
        for rc, pc in zip(r['cols'], py['cols'], strict=True):
            compare(rc['node'], pc['node'], f'{path}${rc["name"]}', out)
    elif rk == 'list':
        for ri, pi in zip(r['items'], py['items'], strict=True):
            label = ri.get('name')
            compare(ri['node'], pi['node'], f'{path}${label}' if label is not None else f'{path}[]', out)


def index_shifted(rv: list[typing.Any], pv: list[typing.Any]) -> bool:
    """R counts from 1, metafopy from 0: integer vectors that differ by exactly one throughout are the same indices."""
    if not rv or not all(is_finite_number(a) and is_finite_number(b) for a, b in zip(rv, pv, strict=True)):
        return False
    return all(float(a).is_integer() and float(b).is_integer() and a - b == 1 for a, b in zip(rv, pv, strict=True))


def counting(names: list[typing.Any]) -> bool:
    """Names that are only the positions, which R gives to rows without labels."""
    return names == [str(i + 1) for i in range(len(names))]


def compare_vector(r: Node, py: Node, path: str, out: list[Diff]) -> None:
    rv, pv = r['v'], py['v']
    if len(rv) != len(pv) and r['k'] == 'str' and py['k'] == 'str':
        # printed lines: R puts blank lines around and inside its tables, metafopy does not always, and the tests of metafor never compare them
        rv, pv = [x for x in rv if x is None or str(x).strip()], [x for x in pv if x is None or str(x).strip()]
        r, py = {**r, 'v': rv}, {**py, 'v': pv}
    if len(rv) != len(pv):
        out.append(Diff(path, 'shape-diff', f'length R {len(rv)}, Python {len(pv)}'))
        return
    if r.get('d') and py.get('d') and r['d'] != py['d']:
        out.append(Diff(path, 'shape-diff', f'dim R {r["d"]}, Python {py["d"]}'))
        return
    for key, what in (('n', 'names'), ('rn', 'row names'), ('cn', 'column names')):
        if r.get(key) and py.get(key) and r[key] != py[key]:
            out.append(Diff(path, 'label-diff', f'{what} R {r[key][:4]}, Python {py[key][:4]}'))
        elif key != 'n' and r.get(key) and not py.get(key) and r.get('d') and len(r['d']) == 2 and r['d'][0] > 1 and r['d'][1] > 1 and not counting(r[key]):
            out.append(Diff(path, 'label-diff', f'{what} R {r[key][:4]}, Python has none'))
    if index_shifted(rv, pv):
        return
    worst = 0.0
    worst_at = -1
    verdict = 'same'
    for i, (a, b) in enumerate(zip(rv, pv, strict=True)):
        state, rel = cell_equal(a, b)
        if state == 'diff':
            verdict = 'diff'
            if rel >= worst:
                worst, worst_at = rel, i
        elif state == 'drift' and verdict != 'diff':
            verdict = 'drift'
            if rel >= worst:
                worst, worst_at = rel, i
    if verdict == 'same':
        return
    detail = f'element {worst_at}: R {rv[worst_at]}, Python {pv[worst_at]}'
    if math.isfinite(worst):
        detail += f' (relative {worst:.2g})'
    out.append(Diff(path, 'value-diff' if verdict == 'diff' else 'value-drift', detail))
