"""The comparison rules of the audit harness (``src/audit``): value trees, printed output, tolerances and the recall matcher."""

import pathlib

import audit.printdiff
import audit.recall
import audit.report
import audit.steps
import audit.universe
import audit.valuetree as valuetree


def vector(values: list[object], kind: str = 'num') -> valuetree.Node:
    return {'k': kind, 'v': values, 'd': None}


def diffs(r: valuetree.Node, py: valuetree.Node) -> list[valuetree.Diff]:
    out: list[valuetree.Diff] = []
    valuetree.compare(r, py, '', out)
    return out


def test_equal_vectors_have_no_difference() -> None:
    assert diffs(vector([1.0, 2.0]), vector([1.0, 2.0 + 1e-12])) == []


def test_value_difference_beyond_drift_is_listed() -> None:
    (d,) = diffs(vector([1.0, 2.0]), vector([1.0, 2.001]))
    assert d.kind == 'value-diff'
    assert 'element 1' in d.detail


def test_small_difference_is_drift_not_a_finding() -> None:
    (d,) = diffs(vector([1.0]), vector([1.0 + 5e-6]))
    assert d.kind == 'value-drift'


def test_length_difference_is_a_shape_difference() -> None:
    (d,) = diffs(vector([1.0, 2.0, 3.0]), vector([1.0, 2.0]))
    assert d.kind == 'shape-diff'


def test_one_based_indices_equal_zero_based_ones() -> None:
    assert diffs(vector([1, 2, 3]), vector([0, 1, 2])) == []


def test_not_every_difference_of_one_is_an_index() -> None:
    assert [d.kind for d in diffs(vector([1.5, 2.5]), vector([0.5, 1.5]))] == ['value-diff']


def test_missing_values_match_each_other_only() -> None:
    assert diffs(vector([None, 1.0]), vector([None, 1.0])) == []
    assert [d.kind for d in diffs(vector([None]), vector([1.0]))] == ['value-diff']


def test_missing_field_is_reported_by_to_node() -> None:
    guide: valuetree.Node = {'k': 'list', 'items': [{'name': 'a', 'node': vector([1.0])}, {'name': 'b', 'node': vector([2.0])}], 'cls': ['list']}
    missing: list[str] = []
    valuetree.to_node({'a': 1.0}, guide, '', missing)
    assert missing == ['$b']


def test_python_names_find_r_names() -> None:
    assert valuetree.py_name('ci.lb') == 'ci_lb'
    found, value = valuetree.lookup({'ci_lb': 3}, 'ci.lb')
    assert found
    assert value == 3


def test_printed_output_tolerates_the_last_digit() -> None:
    assert audit.printdiff.first_difference('pred  0.1234  0.5678\n', 'pred 0.1235 0.5678\n') is None
    assert audit.printdiff.first_difference('pred  0.1234\n', 'pred 0.1334\n') is not None


def test_printed_output_notices_extra_columns() -> None:
    assert audit.printdiff.first_difference('pred se\n', 'pred\n') is not None


def test_recall_item_matches_by_globs() -> None:
    item = audit.recall.Item('x', 4, 'text', [{'id': 'options/vcov/*/type=*', 'kind': 'crash'}], [], '', '')
    assert item.matches({'id': 'options/vcov/uni_reml/type="obs"', 'kind': 'crash', 'path': '', 'detail': ''})
    assert not item.matches({'id': 'options/vcov/uni_reml/type="obs"', 'kind': 'value-diff', 'path': '', 'detail': ''})
    assert not item.matches({'id': 'options/predict/uni_reml/type="obs"', 'kind': 'crash', 'path': '', 'detail': ''})


def test_recall_hints_select_script_topics_and_cases() -> None:
    items = [
        audit.recall.Item('a', 5, '', [{'id': 'scripts/rma.mh/*'}], [], '', ''),
        audit.recall.Item('b', 5, '', [{'id': 'options/vcov/*/type=*'}], [], '', ''),
        audit.recall.Item('c', 5, '', [{'id': 'matrix/print/mv_ks'}], ['matrix/print/'], '', ''),
    ]
    assert audit.recall.hints(items) == ['matrix/print/', 'options/vcov/*/type=*', 'rma.mh']
    assert audit.recall.steps_for(items) == ['matrix', 'options', 'scripts']


def test_different_names_are_a_label_difference() -> None:
    r = {**vector([1.0, 2.0]), 'n': ['intrcpt', 'I(1/ablat)']}
    py = {**vector([1.0, 2.0]), 'n': ['intrcpt', 'I(1 / ablat)']}
    (d,) = diffs(r, py)
    assert d.kind == 'label-diff'


def test_matrix_without_dimnames_where_r_has_them_is_a_label_difference() -> None:
    r = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2], 'rn': ['a', 'b'], 'cn': ['a', 'b']}
    py = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2]}
    assert {d.kind for d in diffs(r, py)} == {'label-diff'}


def test_positions_as_row_names_are_not_labels() -> None:
    r = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2], 'rn': ['1', '2'], 'cn': None}
    py = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2]}
    assert diffs(r, py) == []


def test_printed_zero_must_have_the_same_decimals() -> None:
    assert audit.printdiff.first_difference('tau^2: 0 (SE = 0.0061)', 'tau^2: 0.0000 (SE = 0.0061)') is not None
    assert audit.printdiff.first_difference('tau^2: 0.1446', 'tau^2: 0.1447') is None


def test_every_new_finding_is_rendered_in_the_report(tmp_path: pathlib.Path) -> None:
    findings = [audit.steps.Finding('matrix', 'matrix/x/y', kind, 'p', 'd') for kind in ('default-diff', 'value-diff', 'a-kind-nobody-listed')]
    ctx = audit.steps.Context(
        work=tmp_path, jobs=1, timeout=0.0, quick=False, only=None, no_ledger=True, universe=audit.universe.Universe.load(audit.universe.snapshot_path())
    )
    outcome = audit.steps.Outcome()
    outcome.findings.extend(findings)
    text = audit.report.write(ctx, {'matrix': outcome}, tmp_path).read_text()
    assert 'Different defaults (1)' in text
    assert 'Other (1)' in text
    assert 'Wrong values (1)' in text


def test_digits_number_stands_for_the_vector_of_nine() -> None:
    guide: valuetree.Node = {'k': 'list', 'items': [{'name': 'digits', 'node': vector([4.0] * 9)}], 'cls': ['list']}
    missing: list[str] = []
    py = valuetree.to_node({'digits': 4}, guide, '', missing)
    out: list[valuetree.Diff] = []
    valuetree.compare(guide, py, '', out)
    assert out == []
    other = valuetree.to_node({'digits': 2}, guide, '', missing)
    valuetree.compare(guide, other, '', out)
    assert [d.kind for d in out] == ['value-diff']


def test_printed_lines_ignore_curly_quotes_and_trailing_spaces() -> None:
    assert valuetree.cell_equal("Signif. codes:  0 '***' 0.001  ", 'Signif. codes:  0 \u2018***\u2019 0.001')[0] == 'same'
    assert valuetree.cell_equal("a 'b'", "a 'c'")[0] == 'diff'


def test_printed_lines_ignore_blank_lines() -> None:
    r = vector(['', 'QM = 1', ''], 'str')
    py = vector(['QM = 1'], 'str')
    assert diffs(r, py) == []
    assert [d.kind for d in diffs(vector(['a', 'b'], 'str'), vector(['a'], 'str'))] == ['shape-diff']
