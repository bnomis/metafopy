"""The R generics the audit runs natively in Python (``src/audit/pycall.py``)."""

import typing

import numpy
import pandas
import pytest

import audit.pycall
import metafopy
import metafopy.core.labelled
import metafopy.datasets


def test_setitem_keeps_an_escalc_frame() -> None:
    """R's ``x$zz <- 1`` keeps the class ``escalc``, as ``x['zz'] = 1`` does in Python; the copy the audit makes must too (audit cases
    ``matrix/$<-/es_*``, where the copy printed as a plain pandas frame)."""
    dat = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    out = audit.pycall.setitem(dat, 'zz', 1)
    assert isinstance(out, metafopy.core.labelled.EscalcFrame)
    assert out.attrs == dat.attrs
    assert 'zz' not in dat.columns
    assert list(out['zz']) == [1] * len(dat)


def test_data_fixture_keeps_an_escalc_dataset() -> None:
    """``{"$data": "dat.konstantopoulos2011"}`` is metadat's ``escalc`` object in R; the copy the audit loads must print as one (audit cases
    ``matrix/print/ks_dat`` and ``matrix/$<-/ks_dat``, which numbered the rows from 0)."""
    out = audit.pycall.resolve_value({'$data': 'dat.konstantopoulos2011'}, {})
    assert isinstance(out, metafopy.core.labelled.EscalcFrame)


def bcg_frame() -> pandas.DataFrame:
    return metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def small_fit() -> typing.Any:
    dat = bcg_frame()
    return metafopy.rma_uni(yi=dat['yi'], vi=dat['vi'])


def test_getitem_single_subscript_picks_columns_of_a_frame_and_elements_of_a_matrix() -> None:
    """``getitem``: ``x[i]`` is the columns of a data frame, the elements down the columns of a matrix, an element of a series or list;
    ``x[i, j]`` indexes both."""
    frame = pandas.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert list(audit.pycall.getitem(frame, 1)) == [3, 4]
    matrix = numpy.array([[1, 2], [3, 4]])
    assert audit.pycall.getitem(matrix, 1) == 3
    assert audit.pycall.getitem(pandas.Series([5, 6]), 1) == 6
    assert audit.pycall.getitem([7, 8], 0) == 7
    assert audit.pycall.getitem(matrix, 1, 0) == 3


def test_head_and_tail_cut_a_frame() -> None:
    """``head`` and ``tail`` of a plain data frame cut it with pandas' ``head`` and ``tail``."""
    frame = pandas.DataFrame({'a': range(10)})
    assert list(audit.pycall.head(frame, 2)['a']) == [0, 1]
    assert list(audit.pycall.tail(frame, 2)['a']) == [8, 9]
    assert len(audit.pycall.head(frame)) == 6
    assert len(audit.pycall.tail(frame)) == 6


class Selectable:
    """A result that selects rows with a slice and converts to a frame, as ``list.rma`` results do."""

    def __getitem__(self, i: slice) -> str:
        return f'rows {i.start}:{i.stop}'

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'a': [1]})


def test_head_and_tail_of_a_selectable_result_slice_it() -> None:
    """``head``/``tail`` of a result with ``__getitem__`` and ``to_frame`` keep its other fields, so they slice instead of converting."""
    assert audit.pycall.head(Selectable(), 3) == 'rows None:3'
    assert audit.pycall.tail(Selectable(), 3) == 'rows -3:None'


def test_print_call_without_arguments_is_str() -> None:
    """``print_call`` of a result with no arguments is its ``str()`` split into lines."""
    fit = small_fit()
    assert audit.pycall.print_call(fit) == str(fit).split('\n')


def test_print_call_ignores_undocumented_arguments_and_rejects_unknown_ones() -> None:
    """``legend`` and ``num`` are read from ``...`` by R and dropped; any other argument that ``summary()`` does not take is an error."""
    fit = small_fit()
    assert audit.pycall.print_call(fit, legend=True) == audit.pycall.print_call(fit)
    with pytest.raises(audit.pycall.MissingArgumentError):
        audit.pycall.print_call(fit, nonsense=1)


def test_summary_for_a_frame_is_summary_escalc_and_for_a_fit_its_text() -> None:
    """``summary_for`` of a frame is ``summary_escalc`` on it; of a fit the summary text that keeps the fit; of an object without one an error."""
    assert audit.pycall.summary_for(bcg_frame()).func is metafopy.summary_escalc  # type: ignore[attr-defined]
    fit = small_fit()
    text = audit.pycall.summary_for(fit)()
    assert isinstance(text, str)
    assert getattr(text, 'fit') is fit  # noqa: B009
    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.pycall.summary_for(3)


def test_coef_for_a_summary_text_uses_the_fit_and_otherwise_the_attribute() -> None:
    """``coef_for``: the summary text's fit gives ``coef_summary``; a result gives its ``coef``; an object without one is an error."""
    fit = small_fit()
    assert audit.pycall.coef_for(audit.pycall.summary_for(fit)())() is not None
    assert audit.pycall.coef_for(fit)() is not None
    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.pycall.coef_for(3)


def test_check_arguments_names_the_unknown_ones() -> None:
    """``check_arguments``: a parameter the callable lacks is a ``MissingArgumentError``; ``**kwargs`` takes anything."""

    def fn(a: int, b: int = 0) -> int:
        return a + b

    def anything(**kwargs: int) -> int:
        return len(kwargs)

    audit.pycall.check_arguments(fn, ['b'])
    audit.pycall.check_arguments(anything, ['zzz'])
    with pytest.raises(audit.pycall.MissingArgumentError, match='c, d'):
        audit.pycall.check_arguments(fn, ['a', 'c', 'd'])


def test_match_positional_names_a_position_that_lands_on_a_named_argument() -> None:
    """R gives the positional arguments to the formals the named ones leave: ``f(1, x = 2)`` for ``f(x, y)`` makes ``y = 1``."""

    def fn(x: int, y: int) -> int:
        return x - y

    assert audit.pycall.match_positional(fn, (1,), {'x': 2}) == ((), {'y': 1, 'x': 2})
    assert audit.pycall.match_positional(fn, (1, 2), {}) == ((1, 2), {})
    assert audit.pycall.match_positional(fn, (1, 2, 3), {'x': 2}) == ((1, 2, 3), {'x': 2})


def test_integral_arguments_turns_whole_floats_into_ints_for_int_parameters() -> None:
    """R has one number type: a whole float for a parameter typed ``int`` becomes an ``int``; a ``float`` parameter or a fraction stays."""

    def fn(count: int, level: float, other: int | float, frac: int) -> None:
        return None

    out = audit.pycall.integral_arguments(fn, {'count': 50.0, 'level': 95.0, 'other': 3.0, 'frac': 2.5})
    assert out == {'count': 50, 'level': 95.0, 'other': 3.0, 'frac': 2.5}
    assert isinstance(out['count'], int)


def test_option_value_strips_na_from_the_na_action() -> None:
    """``option_value``: R's ``na.omit`` is Python's ``omit``; any other option is unchanged."""
    assert audit.pycall.option_value('na.action', 'na.omit') == 'omit'
    assert audit.pycall.option_value('na.action', 3) == 3
    assert audit.pycall.option_value('digits', 'na.omit') == 'na.omit'


def test_run_call_makes_a_call_from_a_spec() -> None:
    """``run_call`` resolves the target, the function and the arguments of a call spec (here ``rma.uni`` then ``coef`` of the fit)."""
    spec = {'fn': 'rma.uni', 'args': {'vi': {'$col': ['dat', 'vi']}, 'yi': {'$col': ['dat', 'yi']}}}
    fit = audit.pycall.run_call(spec, {'dat': bcg_frame()})
    assert fit.k == 13
    assert audit.pycall.run_call({'fn': 'coef', 'target': {'$ref': 'fit'}}, {'fit': fit}) is not None
    with pytest.raises(audit.pycall.MissingFunctionError):
        audit.pycall.run_call({'fn': 'no.such.function.at.all'}, {})
