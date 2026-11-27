"""Direct tests of the R-library shims in audit.rlib (group A of the audit-harness spec), checked against the R values recorded in
reference/audit/snippets.json.

Every shim case agrees with R except `list_mixed`, which the shim refuses (decisions D1 and D2 of
`agents/reviews/r-library-shims-follow-r-spec.md`) and which a test checks is refused.
"""

import contextlib
import functools
import io
import typing

import auditsnippets
import matplotlib.pyplot
import numpy
import pandas
import pytest

import audit.errors
import audit.pycall
import audit.rlib


def same_as_r(snippet_id: str) -> None:
    snippet = auditsnippets.load_snippet(snippet_id)
    auditsnippets.assert_same_as_r(snippet, auditsnippets.run_in_python(snippet))


MATCHING = [
    'as_dataframe_list',
    'as_dataframe_matrix',
    'as_matrix_data_frame',
    'as_matrix_vector',
    'as_numeric_factor',
    'as_numeric_logical',
    'as_numeric_text',
    'as_numeric_texts',
    'as_numeric_vector',
    'assign_index',
    'attr_missing',
    'cbind_matrix_and_vector',
    'cbind_named',
    'cbind_named_and_unnamed',
    'cbind_vectors',
    'class_character',
    'class_data_frame',
    'class_matrix',
    'class_null',
    'class_numeric',
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
    'head_list',
    'head_negative',
    'index_data_frame_rows',
    'index_logical',
    'index_negative',
    'index_two_brackets',
    'inherits_data_frame',
    'inherits_fit',
    'inherits_number',
    'isna_data_frame',
    'isna_list',
    'isna_matrix',
    'isna_vector',
    'lapply_data_frame',
    'lapply_named_list',
    'lapply_vector',
    'length_data_frame',
    'length_list',
    'length_matrix',
    'length_null',
    'length_vector',
    'list_named',
    'list_unnamed',
    'matrix_both_dims',
    'matrix_by_column',
    'matrix_by_row',
    'matrix_byrow_ncol',
    'matrix_dimnames',
    'matrix_ncol_only',
    'matrix_neither_dim',
    'matrix_recycled',
    'matrix_rownames_only',
    'names_assigned',
    'names_assigned_data_frame',
    'names_data_frame',
    'names_list',
    'names_unnamed',
    'paste_collapse',
    'paste_default',
    'paste_numbers',
    'paste_recycled',
    'paste_sep',
    'rbind_data_frames',
    'rbind_matrix_and_vector',
    'rbind_named',
    'rbind_vectors',
    'recycled_matrix_vector',
    'recycled_operands',
    'rep_each',
    'rep_length_out',
    'rep_times_scalar',
    'rep_times_vector',
    'seq_along_with',
    'seq_by',
    'seq_by_descending',
    'seq_from_to',
    'seq_length_out',
    'seq_length_out_only',
    'seq_length_out_with_to',
    'seq_one_argument',
    'seq_to_by_length_out',
    'split_data_frame',
    'split_matrix',
    'split_vector',
    'stopifnot_true',
    'tail_data_frame',
    'tail_default',
    'tail_list',
    'tail_negative',
    'var_matrix',
    'var_two_vectors',
    'var_vector',
]


@pytest.mark.parametrize('snippet_id', MATCHING)
def test_shim_agrees_with_r(snippet_id: str) -> None:
    """rlib.py group A (T1): the snippet's value in Python equals R's recorded value, through the shim the snippet reaches."""
    same_as_r(snippet_id)


REFUSED = ['list_mixed']


@pytest.mark.parametrize('snippet_id', REFUSED)
def test_shim_refuses_what_r_does_in_an_order_it_cannot_keep(snippet_id: str) -> None:
    """rlib.py r_list: a call mixing named and unnamed arguments raises UnsupportedError (the transpiler passes keywords after positionals)."""
    snippet = auditsnippets.load_snippet(snippet_id)
    with pytest.raises(audit.rlib.UnsupportedError):
        auditsnippets.run_in_python(snippet)


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


def test_stopifnot_raises_on_a_false_condition_and_returns_none_on_true() -> None:
    """rlib.py r_stopifnot: a true condition passes, a false one raises HarnessError (R raises too; the snippet has no value to compare)."""
    assert audit.rlib.r_stopifnot(True, numpy.array([True, True])) is None
    with pytest.raises(audit.errors.HarnessError):
        audit.rlib.r_stopifnot(numpy.array([True, False]))


def test_truth_of_length_zero_raises_unsupported() -> None:
    """rlib.py truth: an if condition of length zero raises UnsupportedError (R: "argument is of length zero"); other conditions use the first element."""
    with pytest.raises(audit.rlib.UnsupportedError):
        audit.rlib.truth(numpy.zeros(0, dtype=bool))
    assert audit.rlib.truth(numpy.array([True, False])) is True
    assert audit.rlib.truth(0) is False


def test_recycled_turns_a_series_of_another_length_into_an_array() -> None:
    """rlib.py recycled: a Series against an operand of another length becomes an array, so numpy recycles; of equal length it stays a Series."""
    series = pandas.Series([1.0, 2.0, 3.0, 4.0])
    a, b = audit.rlib.recycled(series, numpy.array([10.0, 20.0]))
    assert isinstance(a, numpy.ndarray)
    assert b.tolist() == [10.0, 20.0]
    c, d = audit.rlib.recycled(numpy.array([10.0, 20.0]), series)
    assert isinstance(d, numpy.ndarray)
    assert c.tolist() == [10.0, 20.0]
    kept, _ = audit.rlib.recycled(series, numpy.arange(4.0))
    assert isinstance(kept, pandas.Series)
    scalar, _ = audit.rlib.recycled(series, 2.0)
    assert isinstance(scalar, pandas.Series)


class Fit:
    coef_names = ('intrcpt', 'ablat')


def test_names_of_series_dict_fit_and_a_value_without_names() -> None:
    """rlib.py r_names: names of a Series are its index, of a dict its keys, of a fit its coef_names; a bare value has none (hand-built)."""
    assert audit.rlib.r_names(pandas.Series([1, 2], index=['p', 'q'])).tolist() == ['p', 'q']
    assert audit.rlib.r_names({'a': 1, 'b': 2}).tolist() == ['a', 'b']
    assert audit.rlib.r_names(Fit()).tolist() == ['intrcpt', 'ablat']
    assert audit.rlib.r_names(3.0) is None


def test_par_old_values_and_no_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """rlib.py r_par: par(name = value) returns the old values and records the new; par() with nothing returns None; par("usr") is the axes' limits."""
    state: dict[str, object] = {}
    monkeypatch.setattr(audit.rlib, 'PAR_STATE', state)
    assert audit.rlib.r_par() is None
    assert audit.rlib.r_par(mar=3) == {'mar': None}
    assert audit.rlib.r_par({'mar': 4}) == {'mar': 3}
    assert state == {'mar': 4}
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot([0, 1], [2, 3])
    usr = audit.rlib.r_par('usr')
    matplotlib.pyplot.close('all')
    assert usr.shape == (4,)
    assert usr[0] < 0 < 1 < usr[1]
    assert usr[2] < 2 < 3 < usr[3]


def test_shift_index_counts_from_zero_and_keeps_names() -> None:
    """rlib.py shift_index: a scalar, a vector and a list shift from 1 to 0; a name, a bool and None stay."""
    assert audit.rlib.shift_index(3) == 2
    assert audit.rlib.shift_index(numpy.array([1.0, 4.0])) == [0, 3]
    assert audit.rlib.shift_index([2, [3, 'a']]) == [1, [2, 'a']]
    assert audit.rlib.shift_index('tau2') == 'tau2'
    assert audit.rlib.shift_index(True) is True
    assert audit.rlib.shift_index(None) is None


def plain_na_action(x: object, na_action: str = 'omit') -> str:
    return na_action


@functools.singledispatch
def dispatched(x: object) -> None:
    return None


@dispatched.register
def dispatched_int(x: int, na_action: str = 'omit') -> None:
    return None


def test_accepts_reads_the_signature_and_the_dispatched_implementation() -> None:
    """rlib.py accepts: a plain function, a partial, a singledispatch function (decided by the type of the first argument) and a non-callable."""
    assert audit.rlib.accepts(plain_na_action, 'na_action')
    assert not audit.rlib.accepts(plain_na_action, 'nothere')
    assert audit.rlib.accepts(functools.partial(plain_na_action, 1), 'na_action')
    assert audit.rlib.accepts(dispatched, 'na_action', 1)
    assert not audit.rlib.accepts(dispatched, 'na_action', 'text')
    assert not audit.rlib.accepts(typing.cast('typing.Any', 3), 'na_action')


def test_rcall_passes_the_na_action_setting_only_to_functions_that_take_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """rcall reads ``options(na.action=...)`` (SETTINGS) and passes it as ``na_action`` when the function accepts it."""
    monkeypatch.setattr(audit.rlib, 'SETTINGS', {'na.action': 'na.exclude'})
    monkeypatch.setattr(audit.pycall, 'resolve_function', lambda name, first: plain_na_action)
    monkeypatch.setattr(audit.pycall, 'check_arguments', lambda f, names: None)
    assert audit.rlib.rcall(typing.cast('typing.Any', {}), 'somefn', 1) == 'exclude'
