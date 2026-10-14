"""The rma.uni help-page examples and further variations, against metafor (src/tests/r/generate_rma_uni_help.R)."""

import functools
import json
import math
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.methods.anova
import metafopy.methods.confint
import metafopy.methods.pairmat
import metafopy.methods.predict
import metafopy.models.rma_uni
import metafopy.transf

FIXTURE = pathlib.Path(__file__).parent / 'reference' / 'extras' / 'rma_uni_help.json'
RTOL = 1e-6


@functools.cache
def case_ref() -> dict[str, typing.Any]:
    with FIXTURE.open() as fh:
        data: dict[str, typing.Any] = json.load(fh)
    return data


def arr(v: typing.Any) -> numpy.ndarray:
    return numpy.array([math.nan if x is None else x for x in numpy.atleast_1d(v)], dtype=float)


def close(got: typing.Any, want: typing.Any, rtol: float = RTOL) -> None:
    numpy.testing.assert_allclose(arr(got), arr(want), rtol=rtol, atol=1e-10, equal_nan=True)


def check(res: metafopy.models.rma_uni.RMAResult, key: str, *, names: bool = True) -> None:
    ref = case_ref()[key]
    assert res.k == ref['k']
    for attr in ('beta', 'se', 'zval', 'pval', 'ci_lb', 'ci_ub'):
        close(getattr(res, attr), ref[attr])
    for attr in ('tau2', 'QE', 'QEp', 'QM', 'QMp', 'I2', 'H2'):
        close(getattr(res, attr), ref[attr])
    if ref['se_tau2'] is not None:
        close(res.se_tau2, ref['se_tau2'])
    if ref['R2'] is not None:
        assert res.R2 is not None
        close(res.R2, ref['R2'])
    close(res.logLik(), ref['ll'])
    if names:
        assert list(res.coef_names) == list(numpy.atleast_1d(ref['coef_names']))


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def re(dat: pandas.DataFrame) -> metafopy.models.rma_uni.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat)


def test_random_effects(dat: pandas.DataFrame, re: metafopy.models.rma_uni.RMAResult) -> None:
    check(metafopy.rma('yi', 'vi', data=dat, method='REML'), 're_reml')
    check(metafopy.rma('yi', sei=numpy.sqrt(dat['vi']), data=dat), 're_sei')
    check(re, 're_reml')


def test_measure_passthrough(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    check(res, 'measure_rr')
    assert res.measure == 'RR'


def test_mods_matrix_and_formulas(dat: pandas.DataFrame) -> None:
    # auditme: accept[literal-duplication] test cases spell out their own R arguments
    check(metafopy.rma('yi', 'vi', mods=dat[['ablat', 'year']], data=dat), 'mods_cbind')
    check(metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat), 'mods_formula')
    check(metafopy.rma('yi ~ ablat + year', 'vi', data=dat), 'yi_formula')


def test_allocation_factor(dat: pandas.DataFrame) -> None:
    dummies = pandas.DataFrame(
        {
            'alloc.random': (dat['alloc'] == 'random').astype(float),
            'alloc.systematic': (dat['alloc'] == 'systematic').astype(float),
            'year': dat['year'],
            'ablat': dat['ablat'],
        }
    )
    check(metafopy.rma('yi', 'vi', mods=dummies, data=dat, btt=[1, 2]), 'alloc_dummies')
    check(metafopy.rma('yi', 'vi', mods='~ C(alloc) + year + ablat', data=dat, btt=[1, 2]), 'alloc_factor', names=False)
    check(metafopy.rma('yi', 'vi', mods='~ alloc + year + ablat', data=dat, btt=[1, 2]), 'alloc_char')


def test_anova_pairmat(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ alloc + year + ablat', data=dat, btt=[1, 2])
    ref = case_ref()['anova_pairmat']
    pm = metafopy.methods.pairmat.pairmat(res, btt='alloc')
    numpy.testing.assert_array_equal(pm.to_numpy(), numpy.atleast_2d(ref['X']))
    assert list(pm.index) == list(numpy.atleast_1d(ref['rownames']))
    a = metafopy.methods.anova.anova(res, X=pm)
    assert isinstance(a, metafopy.methods.anova.WaldResult)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    for attr in ('Xb', 'se', 'zval', 'pval', 'QM', 'QMp'):
        close(getattr(a, attr), ref[attr])
    pm2 = metafopy.methods.pairmat.pairmat(res, btt=[1, 2], btt2=[1, 2])
    numpy.testing.assert_array_equal(pm2.to_numpy(), numpy.array(case_ref()['pairmat_btt2']['X']))
    assert list(pm2.index) == case_ref()['pairmat_btt2']['rownames']


def test_pairmat_needs_several_coefficients(dat: pandas.DataFrame, re: metafopy.models.rma_uni.RMAResult) -> None:
    with pytest.raises(ValueError, match='intercept-only'):
        metafopy.methods.pairmat.pairmat(re, btt=[0])
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    with pytest.raises(ValueError, match='multiple coefficients'):
        metafopy.methods.pairmat.pairmat(res, btt=[1])


def test_subgroups_and_q_decomposition(dat: pandas.DataFrame) -> None:
    subs = {lvl: metafopy.rma('yi', 'vi', data=dat, subset=f'alloc == "{lvl}"') for lvl in ('alternate', 'random', 'systematic')}
    for lvl, res in subs.items():
        check(res, f'sub_{lvl}')
    check(metafopy.rma('yi', 'vi', mods='~ 0 + C(alloc)', data=dat), 'alloc_noint', names=False)
    full = metafopy.rma('yi', 'vi', mods='~ C(alloc)', data=dat)
    check(full, 'alloc_factor_only', names=False)
    numpy.testing.assert_allclose(full.QE, sum(r.QE for r in subs.values()), rtol=1e-12)


def test_fixed_effects_q_partition(dat: pandas.DataFrame) -> None:
    total = metafopy.rma('yi', 'vi', data=dat, method='FE')
    check(total, 'fe_total')
    part = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat, method='FE')
    check(part, 'fe_mods')
    numpy.testing.assert_allclose(part.QE + part.QM, total.QE, rtol=1e-10)


def test_bangertdrowns() -> None:
    check(metafopy.rma('yi', 'vi', data=metafopy.datasets.load('dat.bangertdrowns2004')), 'bangert_re')


def test_co_vc_estimators(dat: pandas.DataFrame) -> None:
    check(metafopy.rma('yi', 'vi', data=dat, method='CO'), 'co')
    check(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, method='VC'), 'vc_mods')


def test_genqm(dat: pandas.DataFrame) -> None:
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    n = metafopy.datasets.load('dat.bcg')[['tpos', 'tneg', 'cpos', 'cneg']].sum(axis=1).to_numpy(dtype=float)
    res = metafopy.rma('yi', 'vi', weights=n, data=dat, method='GENQM')
    check(res, 'genqm')
    check(metafopy.rma('yi', 'vi', weights=n, mods='~ ablat', data=dat, method='GENQM'), 'genqm_mods')
    case_ci(metafopy.methods.confint.confint(res), 'genqm_ci')
    case_ci(metafopy.methods.confint.confint(metafopy.rma('yi', 'vi', weights=n, data=dat, method='GENQ')), 'genq_ci')
    with pytest.raises(ValueError, match='GENQ'):
        metafopy.methods.confint.confint(metafopy.rma('yi', 'vi', data=dat), type='GENQ')


def case_ci(ci: typing.Any, key: str, rtol: float = RTOL) -> None:
    ref = case_ref()[key]
    tab = ci.random
    close(tab['estimate'], ref['estimate'], rtol)
    close(tab['ci.lb'], ref['ci_lb'], rtol)
    close(tab['ci.ub'], ref['ci_ub'], rtol)


def test_confint_types(dat: pandas.DataFrame, re: metafopy.models.rma_uni.RMAResult) -> None:
    case_ci(metafopy.methods.confint.confint(re), 'qp_ci')
    case_ci(metafopy.methods.confint.confint(re, type='PL'), 'pl_ci')
    case_ci(metafopy.methods.confint.confint(re, type='HT'), 'ht_ci')
    case_ci(metafopy.methods.confint.confint(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, method='ML'), type='PL'), 'pl_ci_ml')


def test_confint_transf(re: metafopy.models.rma_uni.RMAResult) -> None:
    ci = metafopy.methods.confint.confint(re, fixed=True, random=False, transf=numpy.exp)
    ref = case_ref()['confint_transf']
    assert ci.fixed is not None
    close(ci.fixed['estimate'], ref['estimate'])
    close(ci.fixed['ci.lb'], ref['ci_lb'])
    close(ci.fixed['ci.ub'], ref['ci_ub'])


def test_measure_passthrough_other_measures() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        check(
            metafopy.rma(measure='SMD', m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i', data=metafopy.datasets.load('dat.normand1999')),
            'measure_smd',
        )
    bcg = metafopy.datasets.load('dat.bcg')
    check(
        metafopy.rma(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', mods='~ ablat', data=bcg, subset='alloc != "random"'),
        'measure_or_subset',
    )
    check(metafopy.rma(measure='PLO', xi='ci', ni='n1i', data=metafopy.datasets.load('dat.nielweise2007'), add=0.25, to='all'), 'measure_pr_add')


def test_measure_required_without_yi() -> None:
    with pytest.raises(ValueError, match='measure'):
        metafopy.rma(ai=[1, 2], bi=[3, 4], ci=[5, 6], di=[7, 8])


def test_hksj_alias(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat, test='hksj')
    assert res.test == 'knha'
    check(res, 'test_hksj')


def test_integral_transf_defaults_to_model_tau2(re: metafopy.models.rma_uni.RMAResult) -> None:
    pr = metafopy.methods.predict.predict(re, transf=metafopy.transf.transf_exp_int)
    ref = case_ref()['predict_exp_int']
    for attr in ('pred', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        close(getattr(pr, attr), ref[attr])
    ci = metafopy.methods.confint.confint(re, fixed=True, random=False, transf=metafopy.transf.transf_ilogit_int)
    ref = case_ref()['confint_ilogit_int']
    assert ci.fixed is not None
    close(ci.fixed['estimate'], ref['estimate'])
    close(ci.fixed['ci.lb'], ref['ci_lb'])
    close(ci.fixed['ci.ub'], ref['ci_ub'])
