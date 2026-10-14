"""vec2mat(), dfround() and replmiss() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R)."""

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF = refutil.load_phase7('misc_helpers')


def r_matrix(rows: list[list[float | None]]) -> numpy.typing.NDArray[numpy.float64]:
    return numpy.array([[numpy.nan if v is None else v for v in row] for row in rows], dtype=float)


@pytest.mark.parametrize(
    ('key', 'x', 'kwargs'),
    [
        ('v1', range(1, 7), {'corr': False}),
        ('v2', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], {'corr': True}),
        ('v3', range(1, 11), {'diag': True}),
        ('v5', range(1, 7), {}),
    ],
)
def test_vec2mat(key: str, x: range | list[float], kwargs: dict[str, bool]) -> None:
    numpy.testing.assert_array_equal(metafopy.vec2mat(list(x), **kwargs), r_matrix(REF[key]))


def test_vec2mat_dimnames() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    out = metafopy.vec2mat(list(range(1, 7)), corr=False, dimnames=['A', 'B', 'C', 'D'])
    assert isinstance(out, pandas.DataFrame)
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    assert list(out.index) == list(out.columns) == ['A', 'B', 'C', 'D']
    numpy.testing.assert_array_equal(out.to_numpy(), r_matrix(REF['v4']))


def test_vec2mat_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='square matrix'):
        metafopy.vec2mat([1, 2, 3, 4])
    with pytest.raises(metafopy.errors.InputError, match='dimnames'):
        # auditme: accept[literal-duplication] each test case spells out its own R arguments
        metafopy.vec2mat([1, 2, 3], dimnames=['a', 'b'])


@pytest.fixture(scope='module')
def cs() -> pandas.DataFrame:
    out = refutil.r_frame(REF['cs'])
    out.index = pandas.Index(REF['cs_rownames'])
    return out


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    ('key', 'kwargs'),
    [
        ('dfround', {'digits': [2, 3, 2, 3, 2, 2]}),
        ('dfround0', {}),
        ('dfround_drop0', {'digits': 3, 'drop0': [True, False, True, False, True, False]}),
    ],
)
def test_dfround(cs: pandas.DataFrame, key: str, kwargs: dict[str, object]) -> None:
    out = metafopy.dfround(cs, **kwargs)  # type: ignore[arg-type]
    refutil.assert_frame(out, REF[key], key, rtol=0)
    assert list(out.index) == REF['cs_rownames']


def test_dfround_mixed() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = pandas.DataFrame({'a': ['x', 'y'], 'b': [1.23456, -2.5], 'c': [numpy.nan, 0.0049]})
    out = metafopy.dfround(dat, digits=[0, 1, 2], drop0=False)
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    assert out['a'].tolist() == ['x', 'y']
    assert out['b'].tolist() == REF['dfround_mixed']['columns'][1]
    # R's format() pads NA to the width of the other values
    assert [v.strip() for v in out['c']] == [v.strip() for v in REF['dfround_mixed']['columns'][2]]


def test_dfround_matrix_and_errors(cs: pandas.DataFrame) -> None:
    out = metafopy.dfround(cs.to_numpy(), digits=2)
    numpy.testing.assert_allclose(out.iloc[:, 0], [-3.55, -0.03, 0.0])
    with pytest.raises(metafopy.errors.InputError, match="'digits'"):
        metafopy.dfround(cs, digits=[1, 2])
    with pytest.raises(metafopy.errors.InputError, match="'drop0'"):
        metafopy.dfround(cs, drop0=[True, False])
    with pytest.raises(metafopy.errors.ArgumentError, match=r'data\.frame'):
        metafopy.dfround([1.0, 2.0])


def test_replmiss() -> None:
    x = [4, 2, 7, numpy.nan, 1, numpy.nan, 5]
    numpy.testing.assert_array_equal(metafopy.replmiss(x, 0), REF['repl1'])  # type: ignore[arg-type]
    numpy.testing.assert_array_equal(metafopy.replmiss(x, [2, 3, 6, 5, 8, 1, 2]), REF['repl2'])  # type: ignore[arg-type]
    dat = pandas.DataFrame({'x': [numpy.nan, 1], 'y': [5, 6]})
    numpy.testing.assert_array_equal(metafopy.replmiss('x', 'y', data=dat), REF['repl3'])  # type: ignore[arg-type]
    numpy.testing.assert_array_equal(metafopy.replmiss(None, [1, 2]), REF['repl4'])  # type: ignore[arg-type]


def test_replmiss_errors() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.replmiss([1.0, numpy.nan], [1.0, 2.0, 3.0])
    with pytest.raises(metafopy.errors.InputError, match='no data'):
        metafopy.replmiss('x', 0)
    with pytest.raises(metafopy.errors.NotFoundError):
        metafopy.replmiss('z', 0, data=pandas.DataFrame({'x': [1.0]}))
