"""The error handling of the audit harness (``src/audit/``): what a failing case, an R error or a null table reports."""

import typing

import pytest

import audit.domains
import audit.pycall
import audit.pyrun
import audit.rlib
import audit.universe
import audit.valuetree


def test_a_null_function_entry_in_the_domains_is_an_empty_one() -> None:
    domains = audit.domains.Domains(
        args={'level': [90]},
        skip={'x': 'why'},
        per_function={'fn': None},  # type: ignore[dict-item]
        args_alias={},
        graphics=[],
        options={},
        options_skip={},
        companions={},
    )
    assert domains.values('fn', 'level') == [90]
    assert domains.skipped('fn', 'x') == 'why'


def test_null_tables_in_the_recipes_are_empty_ones() -> None:
    recipes = audit.domains.Recipes({'generics': {'g': None}})
    assert recipes.generic_calls('g') == [{}]
    assert recipes.pre('g') == []
    recipes = audit.domains.Recipes({'generics': {'g': {'by_fit': {'f': None}, 'base': None}}})
    assert recipes.base_args('g', 'f') == {}


def test_a_fixture_without_a_class_list_has_no_classes() -> None:
    universe = audit.universe.Universe({'fixtures': {'fx': None}, 'functions': {}, 'methods': []})
    assert universe.classes_of('fx') == []


def test_a_metafor_function_without_a_python_callable_is_a_missing_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.rlib, 'METAFOR_FUNCTIONS', {'nosuchfunction'})
    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.rlib.rcall(audit.rlib.Env(), 'nosuchfunction')


def test_an_r_function_outside_metafor_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit.rlib, 'METAFOR_FUNCTIONS', set())
    with pytest.raises(audit.rlib.UnsupportedError):
        audit.rlib.rcall(audit.rlib.Env(), 'nosuchfunction')


def test_a_missing_function_in_python_is_reported_as_missing_function() -> None:
    py = {'status': 'error', 'error_class': 'MissingFunctionError', 'error': 'nosuchfunction'}
    findings = audit.pyrun.classify({'status': 'ok'}, py, None)
    assert [f['kind'] for f in findings] == ['missing-function']


def test_identical_compares_values_numpy_cannot() -> None:
    assert audit.rlib.r_identical([1, 2], [1, 2]) is True
    assert audit.rlib.r_identical('a', 'b') is False


def test_try_and_expect_error_catch_the_errors_of_their_thunk() -> None:
    def fails() -> typing.Any:
        raise ValueError('bad')

    assert audit.rlib.r_try(fails) == 'Error: bad'
    assert audit.rlib.r_expect_error(fails) is True


def test_try_and_expect_error_let_a_missing_function_through() -> None:
    def missing() -> typing.Any:
        raise audit.pycall.MissingFunctionError('f')

    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.rlib.r_expect_error(missing)
    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.rlib.r_try(missing)


def test_a_failing_fixture_is_recorded_with_its_type_and_message() -> None:
    env, errors = audit.pyrun.build_env([{'name': 'fx', 'fn': 'nosuchfunction'}])
    assert env == {}
    assert errors['fx'].startswith('MissingFunctionError')


def test_a_raising_property_is_a_field_the_object_cannot_give() -> None:
    class Holder:
        @property
        def broken(self) -> int:
            raise ValueError('no')

    assert audit.valuetree.lookup(Holder(), 'broken') == (False, None)


def test_an_unexpected_error_in_a_property_is_not_hidden() -> None:
    class Holder:
        @property
        def broken(self) -> int:
            raise SystemError('harness bug')

    with pytest.raises(SystemError):
        audit.valuetree.lookup(Holder(), 'broken')
