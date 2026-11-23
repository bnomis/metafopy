"""The snippet helpers of auditsnippets.py: loading under the pinned versions, running a snippet in Python, and comparing it with R's value."""

import json
import pathlib

import auditsnippets
import numpy
import pytest


def test_snippet_loads_with_tree_and_value() -> None:
    snippet = auditsnippets.load_snippet('rep_each')
    assert snippet.r_text == 'rep(c(1, 2, 3), each = 2)'
    assert snippet.units[0]['t'] == 'call'
    assert snippet.value['k'] == 'num'
    assert snippet.value['v'] == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert snippet.deparsed is None


def test_formula_snippet_records_the_deparsed_text() -> None:
    assert auditsnippets.load_snippet('formula_two_sides').deparsed == 'yi ~ ablat + year'


def test_error_snippet_records_the_message() -> None:
    assert auditsnippets.load_snippet('stopifnot_false').error == 'c(TRUE, FALSE) are not all TRUE'


def test_load_fails_naming_both_versions_when_the_fixture_differs(tmp_path: pathlib.Path) -> None:
    document = json.loads(auditsnippets.SNIPPETS_PATH.read_text(encoding='utf-8'))
    document['meta']['metafor'] = '0.0-1'
    changed = tmp_path / 'snippets.json'
    changed.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match=r'metafor 0\.0-1.*pins R 4\.6\.2 and metafor 5\.2-1'):
        auditsnippets.load_snippet('rep_each', changed)


def test_load_fails_when_the_fixture_is_from_another_r(tmp_path: pathlib.Path) -> None:
    document = json.loads(auditsnippets.SNIPPETS_PATH.read_text(encoding='utf-8'))
    document['meta']['R_minor'] = '5.0'
    changed = tmp_path / 'snippets.json'
    changed.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match=r'R 4\.5\.0'):
        auditsnippets.load_snippet('rep_each', changed)


def test_assert_same_as_r_passes_on_a_match() -> None:
    snippet = auditsnippets.load_snippet('operator_arithmetic')
    auditsnippets.assert_same_as_r(snippet, auditsnippets.run_in_python(snippet))


def test_run_in_python_runs_every_unit_in_one_environment() -> None:
    snippet = auditsnippets.load_snippet('assign_plain')
    assert auditsnippets.run_in_python(snippet) == 4


def test_assert_same_as_r_fails_on_a_changed_cell() -> None:
    snippet = auditsnippets.load_snippet('rep_each')
    with pytest.raises(AssertionError, match='differs from R'):
        auditsnippets.assert_same_as_r(snippet, numpy.array([1.0, 1.0, 2.0, 9.0, 3.0, 3.0]))


def test_assert_same_as_r_fails_on_a_drift_inside_the_comparator_tolerance() -> None:
    snippet = auditsnippets.load_snippet('rep_each')
    with pytest.raises(AssertionError):
        auditsnippets.assert_same_as_r(snippet, numpy.array([1.0, 1.0, 2.0, 2.0 + 1e-9, 3.0, 3.0]))


def test_assert_same_as_r_refuses_a_snippet_on_which_r_raises() -> None:
    snippet = auditsnippets.load_snippet('stopifnot_false')
    with pytest.raises(AssertionError, match='R raises'):
        auditsnippets.assert_same_as_r(snippet, None)
