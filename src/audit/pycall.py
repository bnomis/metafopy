"""The Python side of a case: resolve the values and the function of a call spec, then make the call.

A call spec is ``{'fn': name, 'target': value, 'args': {...}}`` in the R spelling (``rma.uni``, ``transf.exp``, ``ci.lb``): dots become underscores
and the function is looked up on ``metafopy``, then ``metafopy.transf``, then as a method of the target. See ``r/common.R`` for the value tags.
"""

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
    return x[i] if j is None else x[i, j]


def setitem(x: typing.Any, name: str, value: typing.Any) -> typing.Any:
    out = x.copy()
    out[name] = value
    return out


def print_call(x: typing.Any, **kwargs: typing.Any) -> list[str]:
    """``print(x, ...)`` as text: metafopy results print through ``str()``, which takes no arguments; the arguments of R's ``print`` go to ``summary()``."""
    if kwargs:
        summary: typing.Any = getattr(x, 'summary', None)
        accepted = inspect.signature(summary).parameters if callable(summary) else {}
        missing = sorted(k for k in kwargs if k not in accepted)
        if missing:
            raise MissingArgumentError(f'print() of {type(x).__name__} takes no argument {missing}; R was given {sorted(kwargs)}')
        return str(summary(**kwargs)).split('\n')
    return str(x).split('\n')


HELPERS: dict[str, typing.Callable[..., typing.Any]] = {'identity': lambda x: x, '[': getitem, '$<-': setitem, 'print': print_call}


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
    if name == 'plot' and target is not None:
        shown = target[0] if isinstance(target, list) and target else target
        kind = type(shown).__name__
        name = 'plot_profile' if kind.endswith('ProfileResult') else PLOT_FUNCTIONS.get(kind, 'plot_rma')
    for spelling in candidates(name):
        for module in (metafopy, metafopy.transf):
            fn = getattr(module, spelling, None)
            if callable(fn):
                return typing.cast(typing.Callable[..., typing.Any], fn)
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
