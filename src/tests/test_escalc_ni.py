"""Sample sizes stored by escalc() are picked up by rma() as in metafor (attr(yi, "ni")), checked against R metafor 5.2-1 (tests/r/generate_audit3.R)."""

import typing

import matplotlib
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

matplotlib.use('Agg')

REF: dict[str, typing.Any] = refutil.load_phase12('audit3')['ni']


@pytest.fixture(scope='module')
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture helpers
def escalc_frame() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def plain_fit(escalc_frame: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', data=escalc_frame)


def test_ni_is_stored(escalc_frame: pandas.DataFrame, plain_fit: metafopy.RMAResult) -> None:
    refutil.close(plain_fit.data.ni, REF['ni'], 'ni')
    sub = metafopy.rma('yi', 'vi', data=escalc_frame, subset='ablat > 20')
    refutil.close(sub.data.ni, REF['ni_sub'], 'ni subset')
    mods = metafopy.rma('yi', 'vi', mods='~ ablat', data=escalc_frame)
    refutil.close(mods.data.ni, REF['ni_mods'], 'ni mods')


def test_explicit_ni_wins(escalc_frame: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=escalc_frame, ni=numpy.arange(1, 14))
    refutil.close(res.data.ni, list(range(1, 14)), 'ni')


def test_no_ni_without_escalc() -> None:
    res = metafopy.rma([0.1, 0.3, 0.2], [0.1, 0.1, 0.2])
    assert res.data.ni is None
    with pytest.raises(metafopy.errors.InputError, match='No sample size'):
        metafopy.regtest(res, predictor='ni')


def test_series_with_attrs(escalc_frame: pandas.DataFrame) -> None:
    res = metafopy.rma(escalc_frame['yi'], escalc_frame['vi'])
    refutil.close(res.data.ni, REF['ni'], 'ni')


@pytest.mark.parametrize('case', REF['regtest'], ids=lambda c: c['predictor'])
def test_regtest_predictor(plain_fit: metafopy.RMAResult, case: dict[str, typing.Any]) -> None:
    out = metafopy.regtest(plain_fit, predictor=case['predictor'])
    refutil.close(out.zval, case['zval'], 'zval')
    refutil.close(out.pval, case['pval'], 'pval')
    refutil.close(out.dfs, case['dfs'], 'dfs')


@pytest.mark.parametrize('case', REF['funnel'], ids=lambda c: c['yaxis'])
def test_funnel_yaxis(plain_fit: metafopy.RMAResult, case: dict[str, typing.Any]) -> None:
    out = metafopy.funnel(plain_fit, yaxis=case['yaxis'])
    refutil.close(out.x, case['x'], 'x')
    refutil.close(out.y, case['y'], 'y')


@pytest.mark.parametrize('case', REF['selmodel'], ids=lambda c: c['prec'])
def test_selmodel_prec(plain_fit: metafopy.RMAResult, case: dict[str, typing.Any]) -> None:
    out = typing.cast(typing.Any, metafopy.selmodel(plain_fit, type='halfnorm', prec=case['prec']))
    refutil.close(out.beta, case['beta'], 'beta', rtol=1e-6)
    refutil.close(out.tau2, case['tau2'], 'tau2', rtol=1e-5)
    refutil.close(out.delta, case['delta'], 'delta', rtol=1e-5)
    refutil.close(out.fit_stats_ml.ll, case['ll'], 'll', rtol=1e-8)
