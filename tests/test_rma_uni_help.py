"""The rma.uni help-page examples and the Phase 1 additions, against metafor (tests/r/generate_rma_uni_help.R)."""

from __future__ import annotations

import json
import math
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from metafopy import datasets, escalc, rma
from metafopy.methods.anova import WaldResult, anova
from metafopy.methods.confint import confint
from metafopy.methods.pairmat import pairmat
from metafopy.methods.predict import predict
from metafopy.models.rma_uni import RMAResult
from metafopy.transf import transf_exp_int, transf_ilogit_int

FIXTURE = Path(__file__).parent / 'reference' / 'extras' / 'rma_uni_help.json'
RTOL = 1e-6


@cache
def _ref() -> dict[str, Any]:
    with FIXTURE.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _arr(v: Any) -> np.ndarray:
    return np.array([math.nan if x is None else x for x in np.atleast_1d(v)], dtype=float)


def _close(got: Any, want: Any, rtol: float = RTOL) -> None:
    np.testing.assert_allclose(_arr(got), _arr(want), rtol=rtol, atol=1e-10, equal_nan=True)


def _check(res: RMAResult, key: str, *, names: bool = True) -> None:
    ref = _ref()[key]
    assert res.k == ref['k']
    for attr in ('beta', 'se', 'zval', 'pval', 'ci_lb', 'ci_ub'):
        _close(getattr(res, attr), ref[attr])
    for attr in ('tau2', 'QE', 'QEp', 'QM', 'QMp', 'I2', 'H2'):
        _close(getattr(res, attr), ref[attr])
    if ref['se_tau2'] is not None:
        _close(res.se_tau2, ref['se_tau2'])
    if ref['R2'] is not None:
        assert res.R2 is not None
        _close(res.R2, ref['R2'])
    _close(res.logLik(), ref['ll'])
    if names:
        assert list(res.coef_names) == list(np.atleast_1d(ref['coef_names']))


@pytest.fixture(scope='module')
def dat() -> pd.DataFrame:
    return escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def re(dat: pd.DataFrame) -> RMAResult:
    return rma('yi', 'vi', data=dat)


def test_random_effects(dat: pd.DataFrame, re: RMAResult) -> None:
    _check(rma('yi', 'vi', data=dat, method='REML'), 're_reml')
    _check(rma('yi', sei=np.sqrt(dat['vi']), data=dat), 're_sei')
    _check(re, 're_reml')


def test_measure_passthrough(dat: pd.DataFrame) -> None:
    res = rma(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    _check(res, 'measure_rr')
    assert res.measure == 'RR'


def test_mods_matrix_and_formulas(dat: pd.DataFrame) -> None:
    _check(rma('yi', 'vi', mods=dat[['ablat', 'year']], data=dat), 'mods_cbind')
    _check(rma('yi', 'vi', mods='~ ablat + year', data=dat), 'mods_formula')
    _check(rma('yi ~ ablat + year', 'vi', data=dat), 'yi_formula')


def test_allocation_factor(dat: pd.DataFrame) -> None:
    dummies = pd.DataFrame(
        {
            'alloc.random': (dat['alloc'] == 'random').astype(float),
            'alloc.systematic': (dat['alloc'] == 'systematic').astype(float),
            'year': dat['year'],
            'ablat': dat['ablat'],
        }
    )
    _check(rma('yi', 'vi', mods=dummies, data=dat, btt=[1, 2]), 'alloc_dummies')
    _check(rma('yi', 'vi', mods='~ C(alloc) + year + ablat', data=dat, btt=[1, 2]), 'alloc_factor', names=False)
    _check(rma('yi', 'vi', mods='~ alloc + year + ablat', data=dat, btt=[1, 2]), 'alloc_char')


def test_anova_pairmat(dat: pd.DataFrame) -> None:
    res = rma('yi', 'vi', mods='~ alloc + year + ablat', data=dat, btt=[1, 2])
    ref = _ref()['anova_pairmat']
    pm = pairmat(res, btt='alloc')
    np.testing.assert_array_equal(pm.to_numpy(), np.atleast_2d(ref['X']))
    assert list(pm.index) == list(np.atleast_1d(ref['rownames']))
    a = anova(res, X=pm)
    assert isinstance(a, WaldResult)
    for attr in ('Xb', 'se', 'zval', 'pval', 'QM', 'QMp'):
        _close(getattr(a, attr), ref[attr])
    pm2 = pairmat(res, btt=[1, 2], btt2=[1, 2])
    np.testing.assert_array_equal(pm2.to_numpy(), np.array(_ref()['pairmat_btt2']['X']))
    assert list(pm2.index) == _ref()['pairmat_btt2']['rownames']


def test_pairmat_needs_several_coefficients(dat: pd.DataFrame, re: RMAResult) -> None:
    with pytest.raises(ValueError, match='intercept-only'):
        pairmat(re, btt=[0])
    res = rma('yi', 'vi', mods='~ ablat', data=dat)
    with pytest.raises(ValueError, match='multiple coefficients'):
        pairmat(res, btt=[1])


def test_subgroups_and_q_decomposition(dat: pd.DataFrame) -> None:
    subs = {lvl: rma('yi', 'vi', data=dat, subset=f'alloc == "{lvl}"') for lvl in ('alternate', 'random', 'systematic')}
    for lvl, res in subs.items():
        _check(res, f'sub_{lvl}')
    _check(rma('yi', 'vi', mods='~ 0 + C(alloc)', data=dat), 'alloc_noint', names=False)
    full = rma('yi', 'vi', mods='~ C(alloc)', data=dat)
    _check(full, 'alloc_factor_only', names=False)
    np.testing.assert_allclose(full.QE, sum(r.QE for r in subs.values()), rtol=1e-12)


def test_fixed_effects_q_partition(dat: pd.DataFrame) -> None:
    total = rma('yi', 'vi', data=dat, method='FE')
    _check(total, 'fe_total')
    part = rma('yi', 'vi', mods='~ ablat + year', data=dat, method='FE')
    _check(part, 'fe_mods')
    np.testing.assert_allclose(part.QE + part.QM, total.QE, rtol=1e-10)


def test_bangertdrowns() -> None:
    _check(rma('yi', 'vi', data=datasets.load('dat.bangertdrowns2004')), 'bangert_re')


def test_co_vc_estimators(dat: pd.DataFrame) -> None:
    _check(rma('yi', 'vi', data=dat, method='CO'), 'co')
    _check(rma('yi', 'vi', mods='~ ablat', data=dat, method='VC'), 'vc_mods')


def test_genqm(dat: pd.DataFrame) -> None:
    n = datasets.load('dat.bcg')[['tpos', 'tneg', 'cpos', 'cneg']].sum(axis=1).to_numpy(dtype=float)
    res = rma('yi', 'vi', weights=n, data=dat, method='GENQM')
    _check(res, 'genqm')
    _check(rma('yi', 'vi', weights=n, mods='~ ablat', data=dat, method='GENQM'), 'genqm_mods')
    _ci(confint(res), 'genqm_ci')
    _ci(confint(rma('yi', 'vi', weights=n, data=dat, method='GENQ')), 'genq_ci')
    with pytest.raises(ValueError, match='GENQ'):
        confint(rma('yi', 'vi', data=dat), type='GENQ')


def _ci(ci: Any, key: str, rtol: float = RTOL) -> None:
    ref = _ref()[key]
    tab = ci.random
    _close(tab['estimate'], ref['estimate'], rtol)
    _close(tab['ci.lb'], ref['ci_lb'], rtol)
    _close(tab['ci.ub'], ref['ci_ub'], rtol)


def test_confint_types(dat: pd.DataFrame, re: RMAResult) -> None:
    _ci(confint(re), 'qp_ci')
    _ci(confint(re, type='PL'), 'pl_ci')
    _ci(confint(re, type='HT'), 'ht_ci')
    _ci(confint(rma('yi', 'vi', mods='~ ablat', data=dat, method='ML'), type='PL'), 'pl_ci_ml')


def test_confint_transf(re: RMAResult) -> None:
    ci = confint(re, fixed=True, random=False, transf=np.exp)
    ref = _ref()['confint_transf']
    assert ci.fixed is not None
    _close(ci.fixed['estimate'], ref['estimate'])
    _close(ci.fixed['ci.lb'], ref['ci_lb'])
    _close(ci.fixed['ci.ub'], ref['ci_ub'])


def test_measure_passthrough_other_measures() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _check(
            rma(measure='SMD', m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i', data=datasets.load('dat.normand1999')),
            'measure_smd',
        )
    bcg = datasets.load('dat.bcg')
    _check(
        rma(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', mods='~ ablat', data=bcg, subset='alloc != "random"'),
        'measure_or_subset',
    )
    _check(rma(measure='PLO', xi='ci', ni='n1i', data=datasets.load('dat.nielweise2007'), add=0.25, to='all'), 'measure_pr_add')


def test_measure_required_without_yi() -> None:
    with pytest.raises(ValueError, match='measure'):
        rma(ai=[1, 2], bi=[3, 4], ci=[5, 6], di=[7, 8])


def test_hksj_alias(dat: pd.DataFrame) -> None:
    res = rma('yi', 'vi', data=dat, test='hksj')
    assert res.test == 'knha'
    _check(res, 'test_hksj')


def test_integral_transf_defaults_to_model_tau2(re: RMAResult) -> None:
    pr = predict(re, transf=transf_exp_int)
    ref = _ref()['predict_exp_int']
    for attr in ('pred', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        _close(getattr(pr, attr), ref[attr])
    ci = confint(re, fixed=True, random=False, transf=transf_ilogit_int)
    ref = _ref()['confint_ilogit_int']
    assert ci.fixed is not None
    _close(ci.fixed['estimate'], ref['estimate'])
    _close(ci.fixed['ci.lb'], ref['ci_lb'])
    _close(ci.fixed['ci.ub'], ref['ci_ub'])
