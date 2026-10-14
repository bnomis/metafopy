"""predict() arguments hetvar, prob, addx, vcov and adjust against R (src/tests/r/generate_predict_args.R)."""

import functools
import json
import pathlib
import typing

import numpy
import pandas
import pytest
import refutil
import scipy.linalg

import metafopy
import metafopy.errors

REF = pathlib.Path(__file__).parent / 'reference' / 'phase7' / 'predict_args.json'
# rma.mv and rma.glmm estimates come from an optimiser, so they agree with R to its convergence tolerance only
OPTIM = 1e-5
# the rma.glmm variance-covariance matrix comes from a numerical Hessian in R (lme4), which agrees to about 1e-4
GLMM = 1e-4


@functools.cache
def ref() -> dict[str, typing.Any]:
    data: dict[str, typing.Any] = json.loads(REF.read_text(encoding='utf-8'))
    return data


def check(got: metafopy.PredictResult, key: str, rtol: float = refutil.RTOL) -> None:
    exp = ref()[key]
    for field in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        if exp[field] is None:
            assert numpy.all(numpy.isnan(getattr(got, field))), field
        else:
            refutil.close(getattr(got, field), exp[field], f'{key}.{field}', rtol=rtol)
    for field in ('prob', 'X', 'vcov'):
        value = getattr(got, field)
        if exp[field] is None:
            assert value is None, field
        else:
            assert value is not None, field
            numpy.testing.assert_allclose(value, numpy.array(exp[field], dtype=float), rtol=rtol, atol=1e-10, err_msg=f'{key}.{field}')


@pytest.fixture(scope='module')
def bcg() -> pandas.DataFrame:
    return refutil.bcg()


def test_rma_uni_options(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=bcg)
    check(metafopy.predict(res, [10, 30, 50], hetvar=0.1, prob='>-0.5', addx=True, vcov=True), 'uni_newmods')
    check(metafopy.predict(res, [10, 30, 50], adjust=True, tau2_levels=0), 'uni_adjust')
    check(metafopy.predict(res, prob='<-0.5', addx=True, vcov=True), 'uni_fitted')
    check(metafopy.predict(metafopy.rma_uni('yi', 'vi', data=bcg), prob='<0', hetvar=[0.2]), 'uni_int')
    check(metafopy.predict(metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=bcg, test='knha'), [10, 50], prob='>0', vcov=True), 'uni_knha')
    check(metafopy.predict(metafopy.rma_uni('yi', 'vi', data=bcg, method='FE'), prob=0, vcov=True), 'uni_fe')
    check(metafopy.predict(res, [10, 50], transf=numpy.exp, vcov=True, addx=True, prob='>0'), 'uni_transf')
    with pytest.raises(metafopy.errors.InputError, match='prob'):
        metafopy.predict(res, [10], prob='>0', adjust=True)
    with pytest.raises(metafopy.errors.InputError, match='hetvar'):
        metafopy.predict(res, [10, 20], hetvar=[0.1, 0.2, 0.3])


def test_rma_mv() -> None:
    dat = metafopy.datasets.load('konstantopoulos2011')
    res = metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=dat)
    check(metafopy.predict(res, [-10, 0, 10], prob='>0', addx=True, vcov=True), 'mv_newmods', OPTIM)
    check(metafopy.predict(res, [-10, 0, 10], hetvar=0.05, adjust=True), 'mv_hetvar', OPTIM)


def test_rma_mv_levels() -> None:
    dat = metafopy.datasets.load('berkey1998')
    V = scipy.linalg.block_diag(*[g[['v1i', 'v2i']].to_numpy(dtype=float) for _, g in dat.groupby('trial', sort=True)])
    res = metafopy.rma_mv('yi', V, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', data=dat)
    check(metafopy.predict(res, numpy.eye(2), tau2_levels=['AL', 'PD'], prob='<0.2', vcov=True), 'un_levels', OPTIM)
    with pytest.warns(UserWarning, match='hetvar'):
        got = metafopy.predict(res, numpy.eye(2), tau2_levels=['AL', 'PD'], hetvar=[0.01, 0.02])
    check(got, 'un_hetvar', OPTIM)


def test_rma_glmm() -> None:
    dat = metafopy.datasets.load('bcg')
    res = metafopy.rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    check(metafopy.predict(res, prob='<0', hetvar=0.1, vcov=True, addx=True), 'glmm', GLMM)
