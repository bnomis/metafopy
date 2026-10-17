"""The Python side of a case: resolve the values and the function of a call spec, then make the call.

A call spec is ``{'fn': name, 'target': value, 'args': {...}}`` in the R spelling (``rma.uni``, ``transf.exp``, ``ci.lb``): dots become underscores
and the function is looked up on ``metafopy``, then ``metafopy.transf``, then as a method of the target. See ``r/common.R`` for the value tags.
"""

import dataclasses
import functools
import inspect
import tomllib
import typing

import numpy
import pandas

import audit.config
import metafopy
import metafopy.datasets
import metafopy.transf
from audit.valuetree import py_name

Env = dict[str, typing.Any]


class MissingFunctionError(Exception):
    """metafopy has no callable for the R function."""


class MissingArgumentError(Exception):
    """The Python callable has no parameter for an R argument."""


@functools.cache
def aliases() -> dict[str, str]:
    doc = tomllib.loads((audit.config.DATA_DIR / 'domains.toml').read_text())
    out: dict[str, str] = doc.get('alias', {})
    return out


def arg_name(name: str) -> str:
    return aliases().get(name, py_name(name))


def shift_index(value: typing.Any) -> typing.Any:
    """R indices are 1-based; metafopy's are 0-based."""
    if isinstance(value, list):
        return [shift_index(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) - 1
    return value


def resolve_value(x: typing.Any, env: Env) -> typing.Any:
    if isinstance(x, dict) and len(x) == 1:
        ((tag, val),) = x.items()
        if tag.startswith('$'):
            match tag:
                case '$ref':
                    return env[val]
                case '$data':
                    return metafopy.datasets.load(val).copy()
                case '$sym':
                    return val
                case '$formula':
                    return val
                case '$fn':
                    return resolve_function(val, None)
                case '$col':
                    return env[val[0]][val[1]]
                case '$na':
                    return numpy.nan
                case '$inf':
                    return numpy.inf if val > 0 else -numpy.inf
                case '$idx':
                    return shift_index(val)
                case '$lambda':
                    return eval(f'lambda {val["args"]}: {val["body"]}', {'numpy': numpy, 'log': numpy.log, 'exp': numpy.exp, 'sqrt': numpy.sqrt})
                case '$matrix':
                    if isinstance(val, dict):
                        return pandas.DataFrame(numpy.array(val['data'], dtype=float), index=val.get('rn'), columns=val.get('cn'))
                    return numpy.array(val, dtype=float)
                case '$named':
                    return pandas.Series(val)
                case '$call':
                    return run_call(val, env)
                case _:
                    raise ValueError(f'unknown value tag {tag}')
    if isinstance(x, dict):
        return {k: resolve_value(v, env) for k, v in x.items()}
    if isinstance(x, list):
        resolved = [resolve_value(v, env) for v in x]
        if resolved and all(isinstance(v, (int, float, bool)) or v is None for v in resolved):
            return [numpy.nan if v is None else v for v in resolved]
        return resolved
    return x


def candidates(name: str) -> list[str]:
    base = py_name(name)
    return [base, f'{base}_']


def getitem(x: typing.Any, i: typing.Any = None, j: typing.Any = None) -> typing.Any:
    """R's ``x[i]`` and ``x[i, j]``: a single subscript picks the columns of a data frame and the elements (down the columns) of a matrix."""
    if j is not None:
        return x[i, j]
    if isinstance(x, pandas.DataFrame):
        return x.iloc[:, i]
    if isinstance(x, pandas.Series):
        return x.iloc[i]
    if isinstance(x, numpy.ndarray) and x.ndim == 2:
        return x.ravel(order='F')[i]
    return x[i]


def setitem(x: typing.Any, name: str, value: typing.Any) -> typing.Any:
    """``x$name <- value``: the copy of a data frame or list, or of a result (a frozen dataclass, which has no element ``name`` unless it is a field)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        if name not in {f.name for f in dataclasses.fields(x)}:
            raise AttributeError(f'{type(x).__name__} has no field {name}')
        return dataclasses.replace(x, **{name: value})
    out = x.copy()
    out[name] = value
    return out


UNDOCUMENTED_PRINT_ARGUMENTS = frozenset({'legend', 'num'})


def print_call(x: typing.Any, **kwargs: typing.Any) -> list[str]:
    """``print(x, ...)`` as text: metafopy results print through ``str()``, which takes no arguments; the arguments of R's ``print`` go to ``summary()``."""
    if kwargs:
        summary: typing.Any = getattr(x, 'summary', None)
        accepted = inspect.signature(summary).parameters if callable(summary) else {}
        # ``legend`` and ``num`` are read from ``...`` by print.rma.uni and are not documented; R's other print methods ignore them
        kwargs = {k: v for k, v in kwargs.items() if k in accepted or k not in UNDOCUMENTED_PRINT_ARGUMENTS}
        if not kwargs:
            return str(x).split('\n')
        if 'digits' in kwargs and 'digits' not in accepted and dataclasses.is_dataclass(x) and 'digits' in {f.name for f in dataclasses.fields(x)}:
            # print(x, digits = ) of a result that prints through str(): the digits of the result are the ones it prints with
            rest = {k: v for k, v in kwargs.items() if k != 'digits'}
            if not rest:
                return str(dataclasses.replace(typing.cast(typing.Any, x), digits=int(kwargs['digits']))).split('\n')
        missing = sorted(k for k in kwargs if k not in accepted)
        if missing:
            raise MissingArgumentError(f'print() of {type(x).__name__} takes no argument {missing}; R was given {sorted(kwargs)}')
        # R's print() has no fit statistics block; the summary() that takes its arguments shows it unless told not to
        shown = {'showfit': False, **kwargs} if 'showfit' in accepted else kwargs
        return str(summary(**shown)).split('\n')
    return str(x).split('\n')


class SummaryText(str):
    """``summary()`` of a fit is its printed text in metafopy; the object keeps the fit for ``coef()``, which R applies to the summary."""

    fit: typing.Any


def summary_for(x: typing.Any) -> typing.Callable[..., typing.Any]:
    """R's ``summary`` of ``x``: ``summary_escalc`` for a data frame of effect sizes, the ``summary`` method of the result otherwise."""
    if isinstance(x, pandas.DataFrame):
        return functools.partial(metafopy.summary_escalc, x)
    method = getattr(x, 'summary', None)
    if method is None:
        raise MissingFunctionError('summary')

    @functools.wraps(method)
    def call(*args: typing.Any, **kwargs: typing.Any) -> SummaryText:
        text = SummaryText(method(*args, **kwargs))
        text.fit = x
        return text

    return functools.partial(call)


def coef_for(x: typing.Any) -> typing.Callable[..., typing.Any]:
    """R's ``coef`` of ``x``: ``coef_summary`` of the fit for a summary, the ``coef`` method of the result otherwise."""
    if isinstance(x, SummaryText):
        return functools.partial(metafopy.coef_summary, x.fit)
    attr = getattr(x, 'coef', None)
    if attr is None:
        raise MissingFunctionError('coef')
    return functools.partial(attr if callable(attr) else lambda: attr)


def frame_of(x: typing.Any) -> typing.Any:
    """The data frame of a result: R's lists of results (``list.rma``, ``anova.rma``, ...) are data frames in Python through ``to_frame()``."""
    to_frame = getattr(x, 'to_frame', None)
    return to_frame() if callable(to_frame) else x


def as_data_frame(x: typing.Any) -> typing.Any:
    return frame_of(x)


def as_matrix(x: typing.Any) -> typing.Any:
    return frame_of(x)


def head(x: typing.Any, n: int = 6) -> typing.Any:
    """``head.list.rma`` keeps the other fields of the result, so a result that can select rows is cut with ``x[:n]``."""
    if hasattr(type(x), '__getitem__') and hasattr(x, 'to_frame'):
        return x[:n]
    return frame_of(x).head(n)


def tail(x: typing.Any, n: int = 6) -> typing.Any:
    if hasattr(type(x), '__getitem__') and hasattr(x, 'to_frame'):
        return x[-n:]
    return frame_of(x).tail(n)


HELPERS: dict[str, typing.Callable[..., typing.Any]] = {
    'identity': lambda x: x,
    '[': getitem,
    '$<-': setitem,
    'print': print_call,
    'as.data.frame': as_data_frame,
    'as.matrix': as_matrix,
    'head': head,
    'tail': tail,
}

# generics whose Python callable depends on the object they are applied to
TARGET_FUNCTIONS: dict[str, typing.Callable[[typing.Any], typing.Callable[..., typing.Any]]] = {'summary': summary_for, 'coef': coef_for}

# R functions that are operators or base methods of data frames and matrices; Python uses pandas and numpy for them, so there are no formals to compare
NATIVE = frozenset({'[', '$<-', 'as.data.frame', 'as.matrix', 'head', 'tail'})


# R's plot() is one generic; metafopy has one function per class of the object that is drawn
PLOT_FUNCTIONS = {
    'CumulResult': 'plot_cumul',
    'GoshResult': 'plot_gosh',
    'InfluenceResult': 'plot_infl',
    'LSPermutestResult': 'plot_permutest',
    'PermutestResult': 'plot_permutest',
    'SelmodelResult': 'plot_selmodel',
    'VIFResult': 'plot_vif',
}


def resolve_function(name: str, target: typing.Any) -> typing.Callable[..., typing.Any]:
    """The callable for an R function name; a method found on the target comes back already bound to it (call it without the target)."""
    if name in HELPERS:
        return HELPERS[name]
    if target is not None and name in TARGET_FUNCTIONS:
        return TARGET_FUNCTIONS[name](target)
    if name == 'plot' and target is not None:
        shown = target[0] if isinstance(target, list) and target else target
        kind = type(shown).__name__
        name = 'plot_profile' if kind.endswith('ProfileResult') else PLOT_FUNCTIONS.get(kind, 'plot_rma')
    for spelling in candidates(name):
        for module in (metafopy, metafopy.transf):
            fn = getattr(module, spelling, None)
            if callable(fn):
                return typing.cast(typing.Callable[..., typing.Any], fn)
    if isinstance(target, pandas.DataFrame):
        for spelling in candidates(name):
            fn = getattr(metafopy, f'{spelling}_escalc', None)
            if callable(fn):
                return functools.partial(fn, target)
    if target is not None:
        for spelling in candidates(name):
            fn = getattr(target, spelling, None)
            if callable(fn):
                return functools.partial(fn)
    fn = getattr(numpy, py_name(name), None)
    if callable(fn) and target is None:
        return typing.cast(typing.Callable[..., typing.Any], fn)
    raise MissingFunctionError(name)


def check_arguments(fn: typing.Callable[..., typing.Any], names: list[str]) -> None:
    try:
        sig = inspect.signature(fn.func if isinstance(fn, functools.partial) else fn)
    except TypeError, ValueError:
        return
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    unknown = [n for n in names if n not in params]
    if unknown:
        raise MissingArgumentError(', '.join(unknown))


def match_positional(
    fn: typing.Callable[..., typing.Any], args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    """R matches the named arguments first and gives the positional ones to the formals left over, in order: ``blsplit(V, cluster = x, cov2cor)``
    passes ``cov2cor`` as ``fun``. Python fills the positions first, so a positional argument that would land on a named one is named."""
    try:
        params = inspect.signature(fn).parameters
    except TypeError, ValueError:
        return args, kwargs
    positional = [n for n, p in params.items() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not any(n in kwargs for n in positional[: len(args)]) or any(p.kind is p.VAR_POSITIONAL for p in params.values()):
        return args, kwargs
    free = [n for n in positional if n not in kwargs]
    if len(args) > len(free):
        return args, kwargs
    return (), {**dict(zip(free, args, strict=False)), **kwargs}


def integral_arguments(fn: typing.Callable[..., typing.Any], kwargs: dict[str, typing.Any], dispatch: typing.Any = None) -> dict[str, typing.Any]:
    """R has one number type: ``iter = 50`` is a double that is whole. The arguments metafopy types as an ``int`` (not as a ``float``) get an ``int``."""
    if dispatch is not None and hasattr(fn, 'dispatch'):
        fn = fn.dispatch(type(dispatch))
    try:
        params = inspect.signature(fn.func if isinstance(fn, functools.partial) else fn).parameters
    except TypeError, ValueError:
        return kwargs
    out = dict(kwargs)
    for name, value in kwargs.items():
        param = params.get(name)
        if param is None or not isinstance(value, float) or not value.is_integer():
            continue
        text = str(param.annotation)
        if 'int' in text and 'float' not in text:
            out[name] = int(value)
    return out


def option_value(name: str, value: typing.Any) -> typing.Any:
    """R settings are strings such as ``na.omit``; the Python argument that replaces them takes ``omit``."""
    if name == 'na.action' and isinstance(value, str):
        return value.removeprefix('na.')
    return value


def run_call(spec: dict[str, typing.Any], env: Env) -> typing.Any:
    if spec.get('op') == 'setna':
        frame = metafopy.datasets.load(spec['from']).copy()
        for row, column in spec['cells']:
            frame.loc[frame.index[row - 1], column] = numpy.nan
        return frame
    for earlier in spec.get('pre', []):
        run_call(earlier, env)
    target = resolve_value(spec['target'], env) if spec.get('target') is not None else None
    fn = resolve_function(spec['fn'], target)
    args = {arg_name(k): resolve_value(v, env) for k, v in (spec.get('args') or {}).items()}
    args.update({arg_name(k): option_value(k, resolve_value(v, env)) for k, v in (spec.get('options') or {}).items()})
    check_arguments(fn, list(args))
    if target is not None and not isinstance(fn, functools.partial):
        return fn(target, **args)
    return fn(**args)
