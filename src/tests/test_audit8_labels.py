"""Audit 8, round 4: row and column names and printouts that differ from R metafor 5.2-1 (checked against R)."""

import typing

import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.printing


@pytest.fixture(autouse=True)
def console_80() -> None:
    metafopy.printing.SETTINGS['width'] = 80


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def test_hatvalues_matrix_is_named_by_study(dat: pandas.DataFrame) -> None:
    res: typing.Any = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    hat = res.hatvalues(type='matrix')
    assert list(hat.rownames)[:3] == ['1', '2', '3']
    assert list(hat.colnames)[:3] == ['1', '2', '3']


def test_predict_addx_names(dat: pandas.DataFrame) -> None:
    res: typing.Any = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    pred: typing.Any = res.predict(newmods=[[20]], addx=True)
    assert tuple(pred.X.colnames) == ('intrcpt', 'ablat')


def test_moderator_test_only_with_several_coefficients(dat: pandas.DataFrame) -> None:
    # R prints "Test of Moderators" only when the model has more than one coefficient
    single: typing.Any = metafopy.rma('yi', 'vi', mods='~ ablat - 1', data=dat)
    assert 'Test of Moderators' not in str(single)
    several: typing.Any = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    assert 'Test of Moderators' in str(several)


def test_poly_raw_true_coefficient_names(dat: pandas.DataFrame) -> None:
    res: typing.Any = metafopy.rma('yi', 'vi', mods='~ poly(ablat, 2, raw=TRUE)', data=dat)
    assert list(res.coef_names) == ['intrcpt', 'poly(ablat, 2, raw = TRUE)1', 'poly(ablat, 2, raw = TRUE)2']
