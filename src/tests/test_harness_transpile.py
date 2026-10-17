"""The translation of R scripts into Python (``src/audit/transpile.py``): assignments into nested expressions and the variables a unit changes."""

import inspect
import typing

import numpy
import pandas
import pytest

import audit.rlib
import audit.scripts
import audit.transpile

Node = dict[str, typing.Any]


def sym(name: str) -> Node:
    return {'t': 'sym', 'v': name}


def num(value: float) -> Node:
    return {'t': 'num', 'v': value}


def call(name: str, *args: Node) -> Node:
    return {'t': 'call', 'f': sym(name), 'args': [{'v': a} for a in args]}


def run(ast: Node, env: audit.rlib.Env) -> typing.Any:
    code = audit.transpile.translate_unit(ast, set(), set())
    return eval(code, audit.scripts.namespace(env))


def test_assignment_into_an_element_of_a_column_changes_the_column() -> None:
    env = audit.rlib.Env()
    env.set('dat', pandas.DataFrame({'yi': [1.0, 2.0, 3.0], 'vi': [1.0, 1.0, 1.0]}))
    # dat$yi[2] <- NA
    run(call('<-', call('[', call('$', sym('dat'), sym('yi')), num(2)), {'t': 'na'}), env)
    frame = env['dat']
    assert numpy.isnan(frame['yi'].to_numpy()[1])
    assert frame['yi'].to_numpy()[[0, 2]].tolist() == [1.0, 3.0]
    assert frame['vi'].to_list() == [1.0, 1.0, 1.0]


def test_bare_name_in_a_model_call_is_a_column_or_else_the_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.rlib, 'METAFOR_FUNCTIONS', {'transf.logit'})
    env = audit.rlib.Env()
    frame = pandas.DataFrame({'yi': [1.0, 2.0]})
    env.scopes.append(frame)
    assert audit.rlib.r_symbol(env, 'yi') == 'yi'
    assert callable(audit.rlib.r_symbol(env, 'transf.logit'))
    assert audit.rlib.r_symbol(env, 'no_such_name') == 'no_such_name'


def test_assignment_targets_and_reads() -> None:
    ast = call('<-', call('[', call('$', sym('dat'), sym('yi')), num(1)), {'t': 'na'})
    assert audit.transpile.assignment_targets(ast) == {'dat'}
    assert audit.transpile.symbols(ast) >= {'dat'}
    plain = call('<-', sym('res'), call('rma', sym('yi'), sym('vi')))
    assert audit.transpile.assignment_targets(plain) == {'res'}
    assert 'res' not in audit.transpile.symbols(plain)


def test_value_of_a_replacement_is_the_assigned_value() -> None:
    env = audit.rlib.Env()
    env.set('dat', pandas.DataFrame({'yi': [1.0, 2.0, 3.0]}))
    # dat$yi[2] <- 7 evaluates to 7, as in R (the data frame is not the value)
    assert run(call('<-', call('[', call('$', sym('dat'), sym('yi')), num(2)), num(7)), env) == 7
    assert env['dat']['yi'].to_numpy()[1] == 7


def test_transformation_as_a_value_has_the_signature_of_the_function() -> None:
    fn = audit.rlib.function_value(audit.rlib.Env(), 'transf.dtorpb')
    assert 'n1i' in inspect.signature(fn).parameters
