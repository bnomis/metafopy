"""Translate the parse trees of R scripts (``r/runscripts.R`` writes them) into Python expressions over ``audit.rlib``.

The translation is mechanical and total or explicit: a construct it cannot express raises ``UntranslatableError`` with the reason, and the unit is
counted as not covered. Nothing is dropped silently, and nothing is translated by hand, so two runs of the audit translate alike.
"""

import dataclasses
import re
import typing

Node = dict[str, typing.Any]

LAZY = frozenset({'capture.output', 'expect_error', 'suppressMessages', 'suppressWarnings', 'try'})
UNARY_OR_BINARY = frozenset({'+', '-'})
BINARY = frozenset({'*', '/', '^', '%%', '%/%', '==', '!=', '<', '>', '<=', '>=', '&', '|', ':', '%in%', '%*%'})
ASSIGN = frozenset({'<-', '=', '<<-'})
# bare names that are functions passed as arguments (`transf = log`), not columns of the data
FUNCTION_SYMBOLS = frozenset({'abs', 'exp', 'identity', 'log', 'log10', 'log2', 'sqrt'})
CONSTANT_SYMBOLS = {
    'T': 'True',
    'F': 'False',
    'NA_real_': 'numpy.nan',
    'NA_integer_': 'numpy.nan',
    'NA_character_': 'None',
    'pi': 'math.pi',
}
PY_KEYWORDS = frozenset(
    {
        'False',
        'None',
        'True',
        'and',
        'as',
        'assert',
        'async',
        'await',
        'break',
        'class',
        'continue',
        'def',
        'del',
        'elif',
        'else',
        'except',
        'finally',
        'for',
        'from',
        'global',
        'if',
        'import',
        'in',
        'is',
        'lambda',
        'nonlocal',
        'not',
        'or',
        'pass',
        'raise',
        'return',
        'try',
        'while',
        'with',
        'yield',
    }
)


class UntranslatableError(Exception):
    """A construct of R the translation does not cover."""


@dataclasses.dataclass
class Context:
    assigned: set[str]
    nse: set[str]
    params: set[str] = dataclasses.field(default_factory=set)
    in_function: bool = False

    def child(self, params: set[str]) -> Context:
        return Context(self.assigned, self.nse, self.params | params, True)


def local_name(name: str) -> str:
    return 'v_' + re.sub(r'\W', '_', name)


def unparse(n: Node) -> str:
    """The R text of a parse tree (for formulas)."""
    t = n['t']
    if t == 'sym':
        return str(n['v'])
    if t == 'num':
        v = n['v']
        return str(int(v)) if float(v).is_integer() else repr(v)
    if t == 'int':
        return f'{n["v"]}L'
    if t == 'str':
        return '"' + str(n['v']).replace('"', '\\"') + '"'
    if t == 'lgl':
        return 'TRUE' if n['v'] else 'FALSE'
    if t == 'null':
        return 'NULL'
    if t == 'na':
        return 'NA'
    if t == 'inf':
        return 'Inf' if n['v'] > 0 else '-Inf'
    if t == 'empty':
        return ''
    if t == 'call':
        f = n['f']
        args = n['args']
        if f['t'] == 'sym':
            name = f['v']
            if name == '(' and len(args) == 1:
                return f'({unparse(args[0]["v"])})'
            if name in UNARY_OR_BINARY | BINARY | {'~', '|', '&', '$'} and len(args) == 2:
                sep = '' if name in {':', '^', '$'} else ' '
                return f'{unparse(args[0]["v"])}{sep}{name}{sep}{unparse(args[1]["v"])}'
            if name in {'-', '+', '~', '!'} and len(args) == 1:
                return f'{name}{unparse(args[0]["v"])}'
            if name in {'[', '[['} and args:
                close = ']' if name == '[' else ']]'
                return f'{unparse(args[0]["v"])}{name}{", ".join(unparse(a["v"]) for a in args[1:])}{close}'
            rendered = ', '.join((f'{a["name"]} = ' if a.get('name') else '') + unparse(a['v']) for a in args)
            return f'{name}({rendered})'
    raise UntranslatableError(f'cannot write {t} as R text')


def tr_sym(name: str, c: Context) -> str:
    if name in CONSTANT_SYMBOLS:
        return CONSTANT_SYMBOLS[name]
    if name in c.params:
        return local_name(name)
    return f'E[{name!r}]'


def tr(n: Node, c: Context) -> str:
    t = n['t']
    if t == 'num':
        return repr(float(n['v']))
    if t == 'int':
        return repr(int(n['v']))
    if t == 'str':
        return repr(n['v'])
    if t == 'lgl':
        return 'True' if n['v'] else 'False'
    if t == 'null':
        return 'None'
    if t == 'na':
        return 'numpy.nan'
    if t == 'inf':
        return 'numpy.inf' if n['v'] > 0 else '(-numpy.inf)'
    if t == 'sym':
        return tr_sym(n['v'], c)
    if t == 'empty':
        return 'EMPTY'
    if t == 'call':
        return tr_call(n, c)
    raise UntranslatableError(f'{t} node')


def split_args(args: list[Node]) -> tuple[list[Node], list[tuple[str, Node]]]:
    positional = [a['v'] for a in args if not a.get('name')]
    named = [(a['name'], a['v']) for a in args if a.get('name')]
    return positional, named


def args_code(args: list[Node], c: Context) -> str:
    parts = [tr(a['v'], c) for a in args if not a.get('name')]
    named = [f'{a["name"]!r}: {tr(a["v"], c)}' for a in args if a.get('name')]
    if named:
        parts.append('**{' + ', '.join(named) + '}')
    return ', '.join(parts)


def callee_name(f: Node) -> str | None:
    if f['t'] == 'sym':
        return str(f['v'])
    if f['t'] == 'call' and f['f']['t'] == 'sym' and f['f']['v'] in {'::', ':::'} and len(f['args']) == 2:
        return str(f['args'][1]['v']['v'])
    return None


def replacement(head: str, current: str, args: list[Node], value_code: str, c: Context) -> str:
    """The code of the new value of ``current`` after ``head(current, args...) <- value``."""
    if head in {'[', '[['}:
        idx = ', '.join(tr(a['v'], c) for a in args)
        return f'rassign_index({current}, {value_code}, {idx})'
    if head == '$':
        return f'rassign_dollar({current}, {args[0]["v"]["v"]!r}, {value_code})'
    if head in {'names', 'colnames'}:
        return f'rassign_names({current}, {value_code})'
    raise UntranslatableError(f'replacement function {head}<-')


def tr_assign(lhs: Node, rhs_code: str, c: Context) -> str:
    """``x <- v``, and ``f(g(x, a), b) <- v``, which R evaluates as ``x <- `g<-`(x, a, `f<-`(g(x, a), b, v))``."""
    plain = lhs['t'] == 'sym' or lhs['t'] == 'str'
    value_code = rhs_code if plain else 'RHS_VALUE'
    while not (lhs['t'] == 'sym' or lhs['t'] == 'str'):
        if not (lhs['t'] == 'call' and lhs['f']['t'] == 'sym'):
            raise UntranslatableError('assignment target')
        args = lhs['args']
        target = args[0]['v']
        value_code = replacement(lhs['f']['v'], tr(target, c), args[1:], value_code, c)
        lhs = target
    name = lhs['v']
    if c.in_function:
        c.params.add(name)
        return f'({local_name(name)} := {value_code.replace("RHS_VALUE", rhs_code)})'
    c.assigned.add(name)
    if plain:
        return f'E.set({name!r}, {value_code})'
    # the value of ``x$a <- v`` is ``v``, not the changed ``x``
    return f'E.replace({name!r}, lambda RHS_VALUE: {value_code}, {rhs_code})'


def walk(n: Node) -> typing.Iterator[Node]:
    """Every node of a parse tree, except the plain names on the left of an assignment (``x`` in ``x <- v``, which are written and not read)."""
    yield n
    if n.get('t') == 'call':
        yield from walk(n['f'])
        plain = n['f'].get('t') == 'sym' and n['f']['v'] in ASSIGN and len(n['args']) == 2 and n['args'][0]['v'].get('t') in {'sym', 'str'}
        for i, a in enumerate(n['args']):
            if not (plain and i == 0):
                yield from walk(a['v'])


def symbols(ast: Node) -> set[str]:
    """The names a unit reads."""
    return {str(n['v']) for n in walk(ast) if n.get('t') == 'sym'}


def assignment_targets(ast: Node) -> set[str]:
    """The variables a unit assigns to: the base name of the left side of every ``<-``, ``=`` and ``<<-`` in it."""
    out: set[str] = set()
    for n in walk(ast):
        if n.get('t') == 'call' and n['f'].get('t') == 'sym' and n['f']['v'] in ASSIGN and len(n['args']) == 2:
            lhs = n['args'][0]['v']
            while lhs.get('t') == 'call' and lhs['args']:
                lhs = lhs['args'][0]['v']
            if lhs.get('t') in {'sym', 'str'}:
                out.add(str(lhs['v']))
    return out


def tr_function(n: Node, c: Context) -> str:
    formals, body = n['args'][0], n['args'][1]['v']
    params: list[str] = []
    names: set[str] = set()
    for item in formals['v'].get('args', []) if formals['v']['t'] == 'pairlist' else []:
        name = item['name']
        if name == '...':
            raise UntranslatableError('function with ... argument')
        names.add(name)
        params.append(local_name(name) + ('' if item['v']['t'] == 'empty' else f'={tr(item["v"], c)}'))
    inner = c.child(names)
    return f'(lambda {", ".join(params)}: {tr(body, inner)})' if params else f'(lambda: {tr(body, inner)})'


def tr_call(n: Node, c: Context) -> str:
    f, args = n['f'], n['args']
    name = callee_name(f)
    if name is None:
        raise UntranslatableError('call of a computed function')
    if name in ASSIGN and len(args) == 2:
        return tr_assign(args[0]['v'], tr(args[1]['v'], c), c)
    if name == '{':
        return 'block(' + ', '.join(tr(a['v'], c) for a in args) + ')' if args else 'None'
    if name == '(':
        return '(' + tr(args[0]['v'], c) + ')'
    if name == 'if':
        cond = f'truth({tr(args[0]["v"], c)})'
        yes = tr(args[1]['v'], c)
        no = tr(args[2]['v'], c) if len(args) > 2 else 'None'
        return f'({yes} if {cond} else {no})'
    if name == 'for':
        var = args[0]['v']['v']
        c.assigned.add(var)
        return f'rfor(E, {var!r}, {tr(args[1]["v"], c)}, lambda: {tr(args[2]["v"], c)})'
    if name in {'while', 'repeat', 'break', 'next', 'return', 'quote', 'bquote', 'substitute', 'missing', 'on.exit', 'tryCatch', 'Recall', 'eval', 'parse'}:
        raise UntranslatableError(f'{name}()')
    if name == 'function':
        return tr_function(n, c)
    if name == '~':
        return repr(unparse(n))
    if name == '[':
        return f'rindex({args_code(args, c)})'
    if name == '[[':
        return f'rindex2({args_code(args, c)})'
    if name == '$':
        key = args[1]['v']
        field = key['v'] if key['t'] in {'sym', 'str'} else None
        if field is None:
            raise UntranslatableError('computed field name')
        return f'rdollar({tr(args[0]["v"], c)}, {field!r})'
    if name == '@':
        raise UntranslatableError('slot access')
    if name in {'&&', '||'}:
        op = 'and' if name == '&&' else 'or'
        return f'(truth({tr(args[0]["v"], c)}) {op} truth({tr(args[1]["v"], c)}))'
    if name in UNARY_OR_BINARY | BINARY or (name == '!' and len(args) == 1):
        return f'OP[{name!r}](' + ', '.join(tr(a['v'], c) for a in args) + ')'
    if name == 'with' and len(args) == 2:
        return f'rwith({tr(args[0]["v"], c)}, lambda: {tr(args[1]["v"], c)}, E)'
    if name in LAZY and args:
        first = f'lambda: {tr(args[0]["v"], c)}'
        rest = args_code(args[1:], c)
        return f'rcall(E, {name!r}, {first}' + (f', {rest}' if rest else '') + ')'
    if name in {'anova', 'predict'} and args and not args[0].get('name'):
        args = [with_model(a, args[0]['v']) for a in args]
    if name in c.params:
        return f'{local_name(name)}({args_code(args, c)})'
    if name in c.nse and any(a.get('name') == 'data' for a in args):
        return tr_nse(name, args, c)
    return f'rcall(E, {name!r}' + (f', {args_code(args, c)}' if args else '') + ')'


def with_model(arg: Node, model: Node) -> Node:
    """`anova(res, X = pairmat(btt = 1:3))`: metafor's pairmat() finds the model of the call it is an argument of, so it is passed the model."""
    value = arg['v']
    is_pairmat = value['t'] == 'call' and value['f'].get('v') == 'pairmat'
    if arg.get('name') and is_pairmat and not any(a.get('name') in {None, 'x'} for a in value['args']):
        return {**arg, 'v': {**value, 'args': [{'v': model}, *value['args']]}}
    return arg


def tr_nse(name: str, args: list[Node], c: Context) -> str:
    """A model-function call: bare names that are not variables of the script are column names; other arguments see the data's columns."""
    data = next(a['v'] for a in args if a.get('name') == 'data')

    def thunk(node: Node) -> str:
        if node['t'] == 'sym' and node['v'] not in c.assigned and node['v'] not in c.params and node['v'] not in CONSTANT_SYMBOLS:
            return f'lambda: rsym(E, {node["v"]!r})'
        if node['t'] == 'call' and node['f'].get('v') == '~':
            return repr(unparse(node))
        return f'lambda: {tr(node, c)}'

    def deferred(node: Node) -> str:
        code = thunk(node)
        return code if code.startswith('lambda') else f'(lambda: {code})'

    def column_names(node: Node) -> str | None:
        """metafor takes `rvars = c(v1i, v2i)` as the names of columns of the data, not as the values of the variables."""
        if node['t'] == 'call' and node['f'].get('v') == 'c' and node['args'] and all(a['v']['t'] == 'sym' and not a.get('name') for a in node['args']):
            return repr([a['v']['v'] for a in node['args']])
        return None

    positional = [deferred(a['v']) for a in args if not a.get('name')]
    named = [
        f'{a["name"]!r}: ' + (f'(lambda: {names})' if a['name'] == 'rvars' and (names := column_names(a['v'])) is not None else deferred(a['v']))
        for a in args
        if a.get('name') and a['name'] != 'data'
    ]
    return f'rnse(E, {name!r}, lambda: {tr(data, c)}, [{", ".join(positional)}], {{{", ".join(named)}}})'


def translate_unit(ast: Node, assigned: set[str], nse: set[str]) -> str:
    return tr(ast, Context(assigned, nse))
