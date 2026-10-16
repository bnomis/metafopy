"""The API step: the formals of every R function and S3 method against the Python callable for the same fit class.

The comparison is per class (the union over classes hid gaps in earlier audits): for a generic and a fitted object, R dispatches to one method
and its formals, plus the arguments it reads from ``...``, are compared with the parameters of what metafopy dispatches to for that object.
"""

import functools
import inspect
import re
import typing

import metafopy
import metafopy.transf
from audit import domains, pycall, universe, valuetree

Json = dict[str, typing.Any]
Finding = tuple[str, str, str, str]  # id, kind, path, detail


def parameters(fn: typing.Callable[..., typing.Any], drop_first: bool) -> tuple[set[str], bool]:
    """Parameter names of a callable and whether it takes ``**kwargs``."""
    try:
        sig = inspect.signature(fn)
    except TypeError, ValueError:
        return set(), True
    params = list(sig.parameters.values())
    if drop_first and params:
        params = params[1:]
    names = {p.name for p in params if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    return names, any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def callables_for(name: str, obj: typing.Any) -> list[tuple[typing.Callable[..., typing.Any], bool]]:
    """Every Python callable that handles R's ``name`` for ``obj``: (callable, takes the object as its first parameter)."""
    found: list[tuple[typing.Callable[..., typing.Any], bool]] = []
    for spelling in pycall.candidates(name):
        method = getattr(obj, spelling, None) if obj is not None else None
        if callable(method):
            found.append((method, False))
        for module in (metafopy, metafopy.transf):
            fn = getattr(module, spelling, None)
            if not callable(fn):
                continue
            dispatch = getattr(fn, 'dispatch', None)
            if dispatch is not None and obj is not None:
                impl = dispatch(type(obj))
                found.append((impl, True))
            else:
                found.append((fn, obj is not None))
    return found


def source_mentions(fns: list[typing.Callable[..., typing.Any]], name: str) -> bool:
    """Whether the source of any of the callables names ``name``: a ``**kwargs`` that never looks at an argument silently ignores it."""
    for fn in fns:
        try:
            text = inspect.getsource(fn.func if isinstance(fn, functools.partial) else fn)
        except TypeError, OSError:
            return True
        if re.search(rf'\b{re.escape(name)}\b', text):
            return True
    return False


def comparable_default(text: str | None) -> typing.Any:
    if text is None:
        return None
    text = text.strip()
    table = {'TRUE': True, 'FALSE': False, 'NULL': None}
    if text in table:
        return table[text]
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    try:
        return float(text)
    except ValueError:
        return ('r-expression', text)


def python_defaults(fn: typing.Callable[..., typing.Any]) -> dict[str, typing.Any]:
    try:
        sig = inspect.signature(fn.func if isinstance(fn, functools.partial) else fn)
    except TypeError, ValueError:
        return {}
    return {p.name: p.default for p in sig.parameters.values() if p.default is not inspect.Parameter.empty}


def compare_function(
    case_id: str,
    entry: Json,
    obj: typing.Any,
    name: str,
    aliases: dict[str, str],
    skip_first: bool,
) -> list[Finding]:
    out: list[Finding] = []
    candidates = callables_for(name, obj)
    if not candidates:
        return [(case_id, 'missing-function', '', f'no Python callable for {name}')]
    have: set[str] = set()
    any_kwargs = False
    defaults: dict[str, typing.Any] = {}
    for fn, takes_obj in candidates:
        names, kwargs = parameters(fn, takes_obj)
        have |= names
        any_kwargs = any_kwargs or kwargs
        for k, v in python_defaults(fn).items():
            defaults.setdefault(k, v)
    r_args = [f for f in entry['formals'] if f['name'] != '...']
    if skip_first and r_args:
        r_args = r_args[1:]
    r_args += [{'name': d, 'default': None} for d in entry.get('dots', [])]
    for formal in r_args:
        arg: str = formal['name']
        py = aliases.get(arg, valuetree.py_name(arg))
        if py not in have and arg not in have:
            if not any_kwargs:
                out.append((f'{case_id}/{arg}', 'missing-arg', arg, f'{name}() has no parameter for {arg}'))
            elif not source_mentions([fn for fn, _ in candidates], py):
                out.append((f'{case_id}/{arg}', 'missing-arg', arg, f'{name}() takes **kwargs that never read {arg}'))
            continue
        r_default = comparable_default(formal.get('default'))
        if py in defaults and isinstance(r_default, (bool, float, str)):
            p_default = defaults[py]
            if isinstance(p_default, (bool, str)) or p_default is None or isinstance(p_default, (int, float)):
                if isinstance(r_default, float) and isinstance(p_default, (int, float)) and not isinstance(p_default, bool):
                    same = float(p_default) == r_default
                else:
                    same = p_default == r_default
                if not same:
                    out.append((f'{case_id}/{arg}', 'default-diff', arg, f'R default {formal.get("default")}, Python {p_default!r}'))
    return out


def run(u: universe.Universe, env: pycall.Env, doms: domains.Domains, derived: frozenset[str] = frozenset()) -> list[Finding]:
    findings: list[Finding] = []
    aliases = dict(doms.args_alias)
    for fixture, obj in env.items():
        classes = u.classes_of(fixture)
        if not classes:
            continue
        for generic in u.generics():
            method = u.method_for(generic, classes)
            if method is None or generic == 'print' or (fixture in derived and method['class'].startswith('rma')):
                continue
            findings += compare_function(f'api/{generic}/{fixture}', method, obj, generic, aliases, True)
    return findings


def run_functions(u: universe.Universe, names: list[str], doms: domains.Domains) -> list[Finding]:
    """Formals of the functions that take no model: R's against the Python function of the same name."""
    findings: list[Finding] = []
    aliases = dict(doms.args_alias)
    for name in names:
        findings += compare_function(f'api/{name}', u.functions[name], None, name, aliases, False)
    return findings
