"""Formula handling for ``mods=``, ``scale=`` and ``random=`` arguments.

Fixed-effects formulas (``mods``, ``scale``) are expanded with formulaic, and
the resulting column names are rewritten to match R's ``model.matrix`` naming
so that coefficient labels line up with metafor (``intrcpt``, ``factorlevel``).

Random-effects formulas for ``rma.mv`` (``~ 1 | study``, ``~ 1 | a/b``,
``~ inner | outer``) are parsed here into :class:`RandomTerm` objects; the
variance structures they imply are built by the model code.
"""

import ast
import collections.abc
import dataclasses
import inspect
import math
import re
import typing

import formulaic.model_matrix
import formulaic.transforms
import numpy
import numpy.typing
import pandas

import metafopy.errors

__all__ = ['RandomTerm', 'design_matrix', 'formula_context', 'parse_random', 'r_column_name', 'r_factor', 'r_logical_formula', 'r_relevel', 'r_scale']

KEYWORD = re.compile(r'(?<=[(,] )(\w+)=(?!=)')
RAW_TRUE = re.compile(r'raw\s*=\s*(True|TRUE|T)\b')
POLY_TERM = re.compile(r'^(poly\(.*\))\[(\d+)\]$')
LEVEL = re.compile(r'\[(?:T\.)?(.*?)\]')
CWRAP = re.compile(r'^C\((.*?)(?:,.*)?\)$')

type FloatArray = numpy.typing.NDArray[numpy.float64]


def r_column_name(name: str) -> str:
    """Translate a formulaic column name into R ``model.matrix`` style.

    ``Intercept`` becomes metafor's ``intrcpt``; treatment-coded factor columns
    such as ``alloc[T.random]`` or ``C(alloc)[T.random]`` become ``allocrandom``.
    Interaction terms are translated part by part.
    """
    if name == 'Intercept':
        return 'intrcpt'
    name = KEYWORD.sub(r'\1 = ', name.replace(' ** ', '^').replace(' / ', '/'))
    name = re.sub(r"'([^']*)'", r'"\1"', name) if '(' in name else name
    poly = POLY_TERM.match(name)
    if poly is not None:
        # formulaic counts raw polynomial terms from 0 and orthogonal ones from 1; R counts both from 1
        return f'{poly.group(1)}{int(poly.group(2)) + (1 if RAW_TRUE.search(poly.group(1)) else 0)}'
    parts: list[str] = []
    for part in name.split(':'):
        level = LEVEL.search(part)
        if level is not None:
            var = part[: level.start()]
            m = CWRAP.match(var)
            if m is not None:
                var = m.group(1)
            part = f'{var}{level.group(1)}'
        parts.append(part)
    return ':'.join(parts)


def r_number(value: float) -> str:
    """R's ``as.character`` of a number: ``8`` for 8.0, 15 significant digits otherwise."""
    return str(int(value)) if math.isfinite(value) and value == int(value) and abs(value) < 1e15 else f'{value:.15g}'


def r_factor(
    x: typing.Any, levels: collections.abc.Sequence[typing.Any] | None = None, labels: collections.abc.Sequence[typing.Any] | None = None
) -> pandas.Categorical:
    """R's ``factor(x, levels, labels)``: sorted unique non-missing values are the levels unless ``levels`` is given.

    See https://stat.ethz.ch/R-manual/R-devel/library/base/html/factor.html.

    Raises:
        metafopy.errors.InputError: ``labels`` does not match the levels.
    """
    values = pandas.Series(x)
    cat = pandas.Categorical(values, categories=None if levels is None else list(levels))
    if pandas.api.types.is_float_dtype(cat.categories.dtype):
        cat = cat.rename_categories(dict(zip(cat.categories, [r_number(v) for v in cat.categories], strict=True)))
    if labels is not None:
        if len(labels) != len(cat.categories):
            raise metafopy.errors.InputError("invalid 'labels': the length must equal the number of levels")
        cat = cat.rename_categories(dict(zip(cat.categories, labels, strict=True)))
    return cat


def r_relevel(x: typing.Any, ref: typing.Any) -> pandas.Categorical:
    """R's ``relevel(x, ref)``: move the level ``ref`` (a level or its 1-based position) to the front of the levels of the factor ``x``.

    See https://stat.ethz.ch/R-manual/R-devel/library/stats/html/relevel.html.

    Raises:
        metafopy.errors.InputError: ``ref`` is not a level of ``x``.
    """
    cat = (
        x
        if isinstance(x, pandas.Categorical)
        else pandas.Categorical(x)
        if isinstance(x, pandas.Series) and isinstance(x.dtype, pandas.CategoricalDtype)
        else None
    )
    if cat is None:
        raise metafopy.errors.InputError("'relevel' only for (unordered) factors")
    levels = list(cat.categories)
    if ref in levels:
        pos = levels.index(ref)
    elif isinstance(ref, int) and not isinstance(ref, bool) and 1 <= ref <= len(levels):
        pos = ref - 1
    else:
        raise metafopy.errors.InputError(f"'ref' must be an existing level, got {ref!r}")
    return typing.cast(pandas.Categorical, cat.reorder_categories([levels[pos], *levels[:pos], *levels[pos + 1 :]]))


def r_scale(x: typing.Any, center: bool | float = True, scale: bool | float = True) -> FloatArray:
    """R's ``scale(x, center, scale)`` for a vector: centre and scale ignoring missing values.

    ``center=True`` subtracts the mean; ``scale=True`` divides by the root mean square of the centred values with ``n - 1`` degrees of freedom (the
    standard deviation when centred). Numeric values are used as the centre and the scale. See
    https://stat.ethz.ch/R-manual/R-devel/library/base/html/scale.html.
    """
    arr = numpy.asarray(x, dtype=numpy.float64).copy()
    ok = ~numpy.isnan(arr)
    if not isinstance(center, bool):
        arr = arr - float(center)
    elif center and ok.any():
        arr = arr - arr[ok].mean()
    if not isinstance(scale, bool):
        arr = arr / float(scale)
    elif scale and ok.any():
        n = int(ok.sum())
        with numpy.errstate(invalid='ignore', divide='ignore'):
            arr = arr / numpy.sqrt(float((arr[ok] ** 2).sum()) / (n - 1))
    return arr


def r_power(rhs: str) -> str:
    """Read ``^`` inside a function call of a formula as the power it is in R (``I(x^2)``); outside a call it stays formula algebra."""
    out: list[str] = []
    calls: list[bool] = []
    for i, ch in enumerate(rhs):
        if ch == '(':
            calls.append((i > 0 and (rhs[i - 1].isalnum() or rhs[i - 1] in '_.')) or any(calls))
        elif ch == ')' and calls:
            calls.pop()
        out.append('**' if ch == '^' and any(calls) else ch)
    return ''.join(out)


R_TOKEN = re.compile(r"\s+|'[^']*'|\"[^\"]*\"|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+|[A-Za-z_.][\w.]*|\|\||&&|[=!<>]=|.")
R_LOGICAL = {'!': 'not', '&': 'and', '&&': 'and', '|': 'or', '||': 'or'}
R_SPACED = frozenset({'!=', '&', '&&', '*', '+', '<', '<=', '==', '>', '>=', '|', '||'})
LOGICAL_PLACEHOLDER = 'metafopy_logical_'
LOGICAL_DTYPES = frozenset({'bool', 'boolean'})


def r_tokens(text: str) -> list[str]:
    return [t for t in R_TOKEN.findall(text) if not t.isspace()]


def r_deparse(tokens: list[str]) -> str:
    """Lay out R code as R's ``deparse`` does, which gives ``model.matrix`` its column names: ``1|ablat`` is ``1 | ablat``, ``x^2`` stays."""
    out = ''
    for i, tok in enumerate(tokens):
        binary = i > 0 and (tokens[i - 1][-1].isalnum() or tokens[i - 1][-1] in '_.)\'"')
        if tok in R_SPACED or (tok == '-' and binary):
            out += f' {tok} '
        elif tok == ',':
            out += ', '
        else:
            out += tok
    return out


def r_logical(x: typing.Any) -> typing.Any:
    """R's ``as.logical`` of a number: non-zero is TRUE, NA stays NA (a pandas nullable boolean, whose ``|``, ``&`` and ``~`` are R's three-valued logic)."""
    if isinstance(x, (bool, int, float)):
        return pandas.NA if isinstance(x, float) and math.isnan(x) else bool(x)
    values = pandas.Series(x)
    if str(values.dtype) not in LOGICAL_DTYPES:
        values = values.astype('Float64') != 0
    return values.astype('boolean')


def r_or(x: typing.Any, y: typing.Any) -> typing.Any:
    return r_logical(x) | r_logical(y)


def r_and(x: typing.Any, y: typing.Any) -> typing.Any:
    return r_logical(x) & r_logical(y)


def r_not(x: typing.Any) -> typing.Any:
    value = r_logical(x)
    return (not value) if isinstance(value, bool) else (value if value is pandas.NA else ~value)


class RLogicalCalls(ast.NodeTransformer):
    """Turn Python's ``or``, ``and`` and ``not`` (R's ``|``, ``&`` and ``!``, which have the same precedence) into element-wise calls."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        values: list[ast.expr] = [self.visit(v) for v in node.values]
        name = 'r_or' if isinstance(node.op, ast.Or) else 'r_and'
        out = values[0]
        for value in values[1:]:
            out = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[out, value], keywords=[])
        return out

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name(id='r_not', ctx=ast.Load()), args=[self.visit(node.operand)], keywords=[])
        node.operand = self.visit(node.operand)
        return node


def r_identity(x: typing.Any) -> typing.Any:
    """R's ``I()`` in ``model.matrix``: a logical value becomes a factor with levels FALSE and TRUE, anything else is left as it is."""
    if isinstance(x, (pandas.Series, pandas.api.extensions.ExtensionArray, numpy.ndarray)) and str(x.dtype) in LOGICAL_DTYPES:
        values = pandas.Series(x).astype('boolean')
        return pandas.Categorical(values.map({True: 'TRUE', False: 'FALSE'}), categories=['FALSE', 'TRUE'])
    return x


def call_spans(rhs: str) -> list[tuple[int, int]]:
    """The outermost function calls of a formula, as ``(start, end)`` with ``rhs[start:end]`` the name and its parenthesised arguments."""
    spans: list[tuple[int, int]] = []
    depth, start = 0, -1
    for i, ch in enumerate(rhs):
        if ch == '(':
            if depth == 0 and i > 0 and (rhs[i - 1].isalnum() or rhs[i - 1] in '_.'):
                start = i
                while start > 0 and (rhs[start - 1].isalnum() or rhs[start - 1] in '_.'):
                    start -= 1
            depth += 1
        elif ch == ')' and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    return spans


def r_logical_calls(rhs: str) -> list[tuple[int, int, str, str]]:
    """The outermost calls of a formula that use R's ``|``, ``&`` or ``!``, which Python reads as bitwise operators.

    Each is ``(start, end, python, r_text)``: the call as Python code that calls
    ``r_or``, ``r_and`` and ``r_not`` element-wise, and as R lays it out.
    """
    calls: list[tuple[int, int, str, str]] = []
    for start, end in call_spans(rhs):
        tokens = r_tokens(rhs[start:end])
        if any(t in R_LOGICAL for t in tokens):
            python = ' '.join(R_LOGICAL.get(t, '**' if t == '^' else t) for t in tokens)
            calls.append((start, end, ast.unparse(RLogicalCalls().visit(ast.parse(python, mode='eval'))), r_deparse(tokens)))
    return calls


def r_logical_formula(rhs: str) -> str:
    """The formula with each call that uses R's logical operators in the Python formulaic evaluates (see :func:`r_logical_calls`)."""
    for start, end, python, _ in reversed(r_logical_calls(rhs)):
        rhs = rhs[:start] + python + rhs[end:]
    return rhs


def r_logical_terms(rhs: str, data: pandas.DataFrame, context: dict[str, typing.Any]) -> tuple[str, pandas.DataFrame, dict[str, str]]:
    """Evaluate each call that uses R's logical operators as R would, missing values included (``1 | NA`` is TRUE, ``!NA`` is NA).

    Each such call becomes a placeholder column of ``data``, read with pandas'
    nullable types, whose ``|``, ``&`` and ``~`` are R's three-valued logic;
    the returned mapping gives each placeholder the call's R text, for the
    column names.
    """
    calls = r_logical_calls(rhs)
    if not calls:
        return rhs, data, {}
    namespace = {**formulaic.transforms.TRANSFORMS, **context}
    namespace |= {str(c): data[c].convert_dtypes(convert_string=False, convert_integer=False) for c in data.columns}
    data = data.copy()
    names: dict[str, str] = {}
    for i, (start, end, python, r_text) in reversed(list(enumerate(calls))):
        placeholder = f'{LOGICAL_PLACEHOLDER}{i}'
        data[placeholder] = eval(python, namespace)  # the formula is the caller's own R code, which R evaluates too
        names[placeholder] = r_text
        rhs = rhs[:start] + placeholder + rhs[end:]
    return rhs, data, names


R_FORMULA_FUNCTIONS: dict[str, typing.Any] = {
    'FALSE': False,
    'I': r_identity,
    'TRUE': True,
    'abs': numpy.abs,
    'factor': r_factor,
    'mean': lambda x: float(numpy.mean(numpy.asarray(x, dtype=numpy.float64))),
    'median': lambda x: float(numpy.median(numpy.asarray(x, dtype=numpy.float64))),
    'r_and': r_and,
    'r_not': r_not,
    'r_or': r_or,
    'relevel': r_relevel,
    'scale': r_scale,
    'sqrt': numpy.sqrt,
}


def caller_namespace() -> dict[str, typing.Any]:
    """The variables of the code that called into metafopy (the environment R evaluates a formula in), without metafopy's own frames."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            module = str(frame.f_globals.get('__name__', ''))
            if module != 'metafopy' and not module.startswith('metafopy.') and module != 'functools':
                return {**frame.f_globals, **frame.f_locals}
            frame = frame.f_back
    finally:
        del frame
    return {}


def formula_context() -> dict[str, typing.Any]:
    """The names a formula can use: R's functions (``factor``, ``sqrt``, ...) and the variables of the code that called into metafopy."""
    context: dict[str, typing.Any] = dict(R_FORMULA_FUNCTIONS)
    for name, value in caller_namespace().items():
        if name not in context or callable(value):
            context[name] = value
    return context


def design_matrix(formula: str, data: pandas.DataFrame) -> tuple[numpy.typing.NDArray[numpy.float64], tuple[str, ...], numpy.typing.NDArray[numpy.bool_]]:
    """Build a model matrix from a one-sided formula.

    Returns the matrix (one row per row of ``data``), R-style column names, and
    a boolean mask of rows that contained missing values in any variable used.
    Rows with missing values are kept (filled with NaN) so the caller can apply
    metafor's ``na.action`` rules jointly with ``yi`` and ``vi``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    rhs = formula.strip()
    if '~' in rhs:
        lhs, _, rhs = rhs.partition('~')
        if lhs.strip():
            raise metafopy.errors.InputError(f'expected a one-sided formula, got {formula!r}')
    if type(data) is not pandas.DataFrame:
        data = pandas.DataFrame(data)  # formulaic evaluates the variables of a data frame subclass (the result of escalc) differently
    context = formula_context()
    rhs, data, logical = r_logical_terms(rhs, data, context)
    mm = formulaic.model_matrix(r_power(rhs), data, na_action='ignore', context=context)
    X = numpy.array(mm.to_numpy(dtype=float), dtype=numpy.float64)
    if X.ndim == 1:
        X = X[:, None]
    names = tuple(r_column_name(str(c)) for c in mm.columns)
    # formulaic encodes a missing factor level as all-zero dummies rather than
    # NaN, so detect missingness from the variables the formula actually uses
    spec = mm.model_spec
    required = spec.required_variables if spec is not None else set(data.columns)
    used = [v for v in required if v in data.columns]
    missing = numpy.isnan(X).any(axis=1) | data[used].isna().any(axis=1).to_numpy()
    # R's model matrix of a missing value is missing in the columns that variable makes (the intercept stays 1)
    for var in used:
        gone = data[var].isna().to_numpy()
        if gone.any():
            cols = [j for j, name in enumerate(names) if re.search(rf'(?<![\w.]){re.escape(var)}(?![\w.])|^{re.escape(var)}', name)]
            if cols:
                X[numpy.ix_(gone, cols)] = numpy.nan
    for placeholder, text in logical.items():
        names = tuple(name.replace(placeholder, text) for name in names)
    return X, names, missing


@dataclasses.dataclass(frozen=True, slots=True)
class RandomTerm:
    """One ``random=`` term of an ``rma.mv`` model.

    ``~ 1 | study`` has ``inner=None`` and ``outer=("study",)``;
    ``~ 1 | district/school`` has ``outer=("district", "school")`` and
    ``nested=True``; ``~ outcome | study`` has ``inner="outcome"``. ``double_bar``
    marks ``~ x1 + x2 || study``, which asks for ``struct="GDIAG"`` instead of ``"GEN"``.
    """

    inner: str | None
    outer: tuple[str, ...]
    nested: bool
    double_bar: bool = False

    def __str__(self) -> str:
        lhs = '1' if self.inner is None else self.inner
        sep = '/' if self.nested else ''
        return f'~ {lhs} | {sep.join(self.outer)}'


def parse_random(formula: str) -> RandomTerm:
    """Parse a single ``rma.mv`` random-effects formula.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    text = formula.strip()
    if not text.startswith('~'):
        raise metafopy.errors.InputError(f"random effects formula must start with '~': {formula!r}")
    body = text[1:]
    for token, hint in (
        ('$', "cannot use '$' notation (use the 'data' argument instead)"),
        (':', "cannot use ':' notation (use interaction() instead)"),
        ('%in%', "cannot use '%in%' notation (use interaction() instead)"),
    ):
        if token in body:
            raise metafopy.errors.InputError(f'random effects formula {formula!r}: {hint}')
    double_bar = '||' in body
    if double_bar:
        body = body.replace('||', '|')
    if body.count('|') != 1:
        raise metafopy.errors.InputError(f"random effects formula needs exactly one '|': {formula!r}")
    lhs, rhs = (s.strip() for s in body.split('|'))
    if not lhs or not rhs:
        raise metafopy.errors.InputError(f'malformed random effects formula: {formula!r}')
    outer = tuple(s.strip() for s in rhs.split('/'))
    if any(not s for s in outer):
        raise metafopy.errors.InputError(f'malformed nesting in random effects formula: {formula!r}')
    inner = None if lhs == '1' else lhs
    if inner is not None and len(outer) > 1:
        raise metafopy.errors.InputError(f"'~ inner | outer' terms cannot also be nested: {formula!r}")
    return RandomTerm(inner=inner, outer=outer, nested=len(outer) > 1, double_bar=double_bar)
