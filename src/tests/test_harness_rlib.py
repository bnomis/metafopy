"""Direct tests of the R-library shims in audit.rlib (group A of the audit-harness spec), checked against the R values recorded in
reference/audit/snippets.json.

A strict xfail names a case where the shim disagrees with R (the item "R-library shims disagree with R" in TODO.md); the passing tests of the
same candidates cover the branches that agree.
"""

import contextlib
import io

import auditsnippets
import numpy
import pandas
import pytest

import audit.rlib


def same_as_r(snippet_id: str) -> None:
    snippet = auditsnippets.load_snippet(snippet_id)
    auditsnippets.assert_same_as_r(snippet, auditsnippets.run_in_python(snippet))


MATCHING = [
    'as_dataframe_list',
    'as_matrix_data_frame',
    'as_matrix_vector',
    'assign_index',
    'attr_missing',
    'cbind_matrix_and_vector',
    'cbind_vectors',
    'class_data_frame',
    'cor_matrix',
    'cor_vectors',
    'dataframe_columns',
    'dataframe_recycled',
    'diag_of_matrix',
    'diag_of_vector',
    'diag_size',
    'equal_number_and_text',
    'equal_numbers',
    'head_data_frame',
    'head_default',
    'head_negative',
    'index_data_frame_rows',
    'index_logical',
    'index_negative',
    'index_two_brackets',
    'inherits_data_frame',
    'names_assigned',
    'names_assigned_data_frame',
]


@pytest.mark.parametrize('snippet_id', MATCHING)
def test_shim_agrees_with_r(snippet_id: str) -> None:
    """rlib.py group A (T1): the snippet's value in Python equals R's recorded value, through the shim the snippet reaches."""
    same_as_r(snippet_id)


DISAGREE = {
    'as_dataframe_matrix': 'r_as_dataframe: a matrix gets no V1, V2 column names',
    'as_numeric_text': 'r_as_numeric: a single string is a 0-d array and fails',
    'cbind_named': 'r_cbind: named arguments give no column names',
    'class_matrix': 'r_class: a matrix is one class name, R says "matrix" and "array"',
    'class_numeric': 'r_class: a number is "float", R says "numeric"',
    'inherits_number': 'r_inherits: a number does not inherit "numeric"',
}


@pytest.mark.parametrize('snippet_id', sorted(DISAGREE))
def test_shim_disagrees_with_r(snippet_id: str, request: pytest.FixtureRequest) -> None:
    """rlib.py group A (T1): cases where the shim differs from R, named in DISAGREE."""
    request.applymarker(pytest.mark.xfail(strict=True, reason='R-library shims disagree with R', raises=(AssertionError, TypeError)))
    same_as_r(snippet_id)


def test_named_array_wrap_keeps_names_and_drops_them_for_a_scalar() -> None:
    """rlib.py Named.__array_wrap__: a numpy operation keeps the names of a matrix result; a scalar result is plain."""
    named = audit.rlib.Named(numpy.arange(4.0).reshape(2, 2), ['row_a', 'row_b'], ['col_a', 'col_b'])
    doubled = named * 2
    assert isinstance(doubled, audit.rlib.Named)
    assert doubled.rownames == ('row_a', 'row_b')
    assert doubled.colnames == ('col_a', 'col_b')
    total = numpy.sum(named)
    assert not isinstance(total, audit.rlib.Named)
    assert total == 6.0


def test_format_cell_whole_float_decimal_float_and_text() -> None:
    """rlib.py format_cell: a whole float prints without a decimal point, a float with decimals as R does, other cells as str."""
    assert audit.rlib.format_cell(2.0) == '2'
    assert audit.rlib.format_cell(numpy.float64(1.5)) == '1.5'
    assert audit.rlib.format_cell('a') == 'a'
    assert audit.rlib.format_cell(3) == '3'


def test_capture_output_returns_the_printed_lines() -> None:
    """rlib.py r_capture_output: the lines the thunk prints, as an object vector (hand-built: R prints numbers in its own format)."""
    assert audit.rlib.r_capture_output(lambda: print('a\nb')).tolist() == ['a', 'b']
    assert audit.rlib.r_capture_output(lambda: None).tolist() == []
    with contextlib.redirect_stdout(io.StringIO()) as sink:
        audit.rlib.r_capture_output(lambda: print('x'))
    assert sink.getvalue() == ''


def test_index2_missing_name_raises() -> None:
    """rlib.py r_index2: [[ with a name the dict lacks raises."""
    with pytest.raises(KeyError):
        audit.rlib.r_index2({'a': 1}, 'z')


def test_index_series_and_dict() -> None:
    """rlib.py r_index: [ on a Series by position and on a dict by name (hand-built: no R snippet makes either)."""
    series = pandas.Series([5.0, 6.0, 7.0])
    assert numpy.asarray(audit.rlib.r_index(series, numpy.array([2]))).tolist() == [6.0]
    assert audit.rlib.r_index({'a': 1, 'b': 2}, 'b') == {'b': 2}
