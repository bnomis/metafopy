"""rma_mv (fits, structures, tests, printing, weights, hat values) checked against R metafor 5.2-1."""

import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase4' / 'rma_mv.json').read_text(encoding='utf-8'))
RTOL = 1e-5
VC_RTOL = 1e-4
VC_ATOL = 1e-6

type Fit = typing.Callable[[], metafopy.RMAMVResult]


def frame(name: str) -> pandas.DataFrame:
    return pandas.DataFrame(REF[name])


def matrix(name: str) -> numpy.typing.NDArray[numpy.float64]:
    return numpy.array(REF[name], dtype=float)


def kon() -> pandas.DataFrame:
    return frame('kon_data')


def ass() -> pandas.DataFrame:
    return frame('ass_data')


def ber() -> pandas.DataFrame:
    return frame('ber_data')


def sim() -> pandas.DataFrame:
    return frame('sim_data')


def sim_na() -> pandas.DataFrame:
    d = sim()
    d.loc[2, 'yi'] = numpy.nan
    d.loc[9, 'x1'] = numpy.nan
    return d


def species_matrix() -> pandas.DataFrame:
    names = REF['sim_R_names']
    return pandas.DataFrame(matrix('sim_R'), index=names, columns=names)


def berkey(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    kwargs.setdefault('mods', '~ 0 + outcome')
    return metafopy.rma_mv('yi', matrix('ber_V'), data=ber(), **kwargs)


def simulated(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    data = kwargs.pop('data', None)
    return metafopy.rma_mv('yi', matrix('sim_V'), data=sim() if data is None else data, **kwargs)


def konst(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    kwargs.setdefault('random', '~ 1 | district/school')
    return metafopy.rma_mv('yi', 'vi', data=kon(), **kwargs)


YEAR = '~ I(year - 1990)'
NESTED = '~ 1 | district/school'

CASES: dict[str, Fit] = {
    'kon_nested': lambda: konst(),
    'kon_nested_ml': lambda: konst(method='ML'),
    'kon_list': lambda: konst(random=['~ 1 | district', '~ 1 | study']),
    'kon_mods': lambda: konst(mods=YEAR),
    'kon_t': lambda: konst(mods=YEAR, test='t'),
    'kon_contain': lambda: konst(mods=YEAR, test='t', dfs='contain'),
    'kon_dfs_num': lambda: konst(dfs=20),
    'kon_knha': lambda: konst(mods=YEAR, test='knha'),
    'kon_adhoc': lambda: konst(test='adhoc'),
    'kon_fix_sigma2': lambda: konst(sigma2=[0.05, numpy.nan]),
    'kon_fix_both': lambda: konst(sigma2=[0.05, 0.02]),
    'kon_fix_all': lambda: konst(sigma2=0.03),
    'kon_sparse': lambda: konst(sparse=True),
    'kon_fe': lambda: metafopy.rma_mv('yi', 'vi', data=kon(), method='FE'),
    'kon_ee_random_ignored': lambda: metafopy.rma_mv('yi', 'vi', random='~ 1 | district', data=kon(), method='EE'),
    'kon_subset': lambda: konst(subset='district != 11'),
    'kon_level90': lambda: konst(level=90),
    'kon_btt': lambda: konst(mods='~ I(year - 1990) + I((year - 1990)**2)', btt=2),
    'kon_formula': lambda: metafopy.rma_mv('yi ~ I(year - 1990)', 'vi', random=NESTED, data=kon()),
    'kon_interaction': lambda: konst(random='~ 1 | interaction(district, school)'),
    'kon_weights': lambda: konst(W=1 / kon()['vi'].to_numpy()),
    'kon_spexp': lambda: konst(random='~ year | district', struct='SPEXP'),
    'ass_cs': lambda: metafopy.rma_mv('yi', 'vi', random=['~ 1 | study', '~ 1 | esid'], data=ass()),
    'ass_cs_ml': lambda: metafopy.rma_mv('yi', 'vi', random='~ 1 | study/esid', data=ass(), method='ML'),
    'ass_mods': lambda: metafopy.rma_mv('yi', 'vi', mods='~ C(deltype) + year', random='~ 1 | study/esid', data=ass()),
    'ass_mods_t': lambda: metafopy.rma_mv('yi', 'vi', mods='~ C(deltype) + year', random='~ 1 | study/esid', data=ass(), test='t', dfs='contain'),
    'ass_vcalc': lambda: metafopy.rma_mv('yi', matrix('ass_V'), random='~ 1 | study/esid', data=ass()),
    'ass_vcalc_sparse': lambda: metafopy.rma_mv('yi', matrix('ass_V'), random='~ 1 | study/esid', data=ass(), sparse=True),
    'ber_un': lambda: berkey(random='~ outcome | trial', struct='UN'),
    'ber_un_ml': lambda: berkey(random='~ outcome | trial', struct='UN', method='ML'),
    'ber_un_nochol': lambda: berkey(random='~ outcome | trial', struct='UN', control={'cholesky': False}),
    'ber_un_t': lambda: berkey(random='~ outcome | trial', struct='UN', test='t'),
    'ber_un_mods': lambda: berkey(mods='~ 0 + outcome + I(year - 1985):outcome', random='~ outcome | trial', struct='UN'),
    'ber_cs': lambda: berkey(random='~ outcome | trial', struct='CS'),
    'ber_hcs': lambda: berkey(random='~ outcome | trial', struct='HCS'),
    'ber_ar': lambda: berkey(random='~ outcome | trial', struct='AR'),
    'ber_har': lambda: berkey(random='~ outcome | trial', struct='HAR'),
    'ber_diag': lambda: berkey(random='~ outcome | trial', struct='DIAG'),
    'ber_id': lambda: berkey(random='~ outcome | trial', struct='ID'),
    'ber_un_fixrho': lambda: berkey(random='~ outcome | trial', struct='UN', rho=0.5),
    'ber_un_fixtau2': lambda: berkey(random='~ outcome | trial', struct='UN', tau2=[numpy.nan, 0.01]),
    'ber_hcs_fixrho': lambda: berkey(random='~ outcome | trial', struct='HCS', rho=0),
    'ber_cs_sparse': lambda: berkey(random='~ outcome | trial', struct='CS', sparse=True),
    'ber_un_sparse': lambda: berkey(random='~ outcome | trial', struct='UN', sparse=True),
    'ber_gen': lambda: berkey(random='~ 0 + outcome | trial', struct='GEN'),
    'ber_gdiag': lambda: berkey(random='~ 0 + outcome || trial', struct='GEN'),
    'ber_car': lambda: berkey(random='~ as.numeric(outcome == "AL") | trial', struct='CAR'),
    'sim_cs': lambda: simulated(random='~ outcome | study', struct='CS'),
    'sim_un': lambda: simulated(random='~ outcome | study', struct='UN'),
    'sim_hcs': lambda: simulated(random='~ outcome | study', struct='HCS'),
    'sim_car': lambda: simulated(random='~ time | study', struct='CAR'),
    'sim_spgau': lambda: simulated(random='~ time | study', struct='SPGAU'),
    'sim_splin': lambda: simulated(random='~ time | study', struct='SPLIN'),
    'sim_gen': lambda: simulated(mods='~ x1', random='~ x1 + x2 | study', struct='GEN'),
    'sim_gdiag': lambda: simulated(mods='~ x1', random='~ x1 + x2 || study', struct='GEN'),
    'sim_s_and_g': lambda: simulated(random=['~ 1 | id', '~ outcome | study'], struct='CS'),
    'sim_two_inner_outer': lambda: simulated(random=['~ outcome | study', '~ outcome | group'], struct=['CS', 'HCS']),
    'sim_R_known': lambda: simulated(random=['~ 1 | species', '~ 1 | study'], R={'species': species_matrix()}),
    'sim_R_cov0': lambda: simulated(random=['~ 1 | species', '~ 1 | study'], R={'species': species_matrix()}, Rscale='cor0'),
    'sim_R_none': lambda: simulated(random=['~ 1 | species', '~ 1 | study'], R={'species': species_matrix()}, Rscale='none'),
    'sim_R_sparse': lambda: simulated(random=['~ 1 | species', '~ 1 | study'], R={'species': species_matrix()}, sparse=True),
    'sim_na': lambda: simulated(mods='~ x1', random='~ 1 | study', data=sim_na()),
    'sim_na_un': lambda: simulated(random='~ outcome | study', struct='UN', data=sim_na()),
    'sim_subset': lambda: simulated(mods='~ x1 + x2', random='~ 1 | study', subset='x2 > 0.3'),
}


def run(name: str) -> metafopy.RMAMVResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CASES[name]()


def vector(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = 1e-9) -> None:
    if expected is None:
        assert actual is None or numpy.size(actual) == 0 or bool(numpy.all(numpy.isnan(numpy.asarray(actual, dtype=float)))), what
        return
    refutil.close(numpy.asarray(actual, dtype=float).reshape(-1), numpy.asarray(expected, dtype=float).reshape(-1).tolist(), what, rtol=rtol, atol=atol)


def components(res: metafopy.RMAMVResult, ref: dict[str, typing.Any], rtol: float = VC_RTOL) -> None:
    for key in ('sigma2', 'tau2', 'rho', 'gamma2', 'phi'):
        present = {'sigma2': ref['withS'], 'tau2': ref['withG'], 'rho': ref['withG'], 'gamma2': ref['withH'], 'phi': ref['withH']}[key]
        actual = getattr(res, key)
        if present:
            vector(actual, ref[key], key, rtol=rtol, atol=VC_ATOL)
        else:
            assert actual is None, key


# Flat likelihoods: the log-likelihood agrees with R to ~1e-9 but the maximiser lands a hair away along a ridge.
FLAT = {'kon_spexp': 2e-3, 'sim_s_and_g': 3e-2}
PRINT_UNSTABLE = frozenset(FLAT) | {'sim_R_known', 'sim_R_cov0', 'sim_R_none', 'sim_R_sparse', 'sim_gdiag'}
NAMES_DIFFER = frozenset({'ass_mods', 'ass_mods_t', 'ber_un_mods', 'kon_btt'})


@pytest.mark.parametrize('name', sorted(CASES))
def test_fit(name: str) -> None:
    ref = REF[name]
    assert 'error' not in ref, ref
    res = run(name)
    tol = FLAT.get(name, VC_RTOL)
    components(res, ref, tol)
    for key, attr in (('beta', 'beta'), ('se', 'se'), ('zval', 'zval'), ('pval', 'pval'), ('ci_lb', 'ci_lb'), ('ci_ub', 'ci_ub')):
        vector(getattr(res, attr), ref[key], f'{name} {key}', rtol=tol, atol=1e-7)
    vector(res.vb, numpy.array(ref['vb']).reshape(-1), f'{name} vb', rtol=tol, atol=1e-8)
    vector(res.QE, ref['QE'], 'QE', rtol=1e-6)
    vector(res.QEp, ref['QEp'], 'QEp', rtol=1e-4, atol=1e-12)
    assert res.QEdf == ref['QEdf']
    vector(res.QM, ref['QM'], 'QM', rtol=tol)
    vector(res.QMp, ref['QMp'], 'QMp', rtol=tol, atol=1e-10)
    assert res.k == ref['k']
    assert res.p == ref['p']
    assert res.m == ref['m']
    assert res.parms == ref['parms']
    assert res.test == ref['test']
    assert res.btt == tuple(int(b) - 1 for b in refutil.as_list(ref['btt']))
    vector(res.ddf, ref['ddf'], 'ddf')
    vector(res.s2w, ref['s2w'], 's2w', rtol=tol)
    vector(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', rtol=1e-5, atol=1e-6)
    vector(res.fit_stats_reml.as_tuple(), ref['fit_reml'], 'fit REML', rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', sorted(CASES))
def test_structure_info(name: str) -> None:
    ref = REF[name]
    res = run(name)
    assert res.with_s == ref['withS']
    assert res.with_g == ref['withG']
    assert res.with_h == ref['withH']
    if res.with_s:
        assert list(res.s_nlevels) == [int(v) for v in refutil.as_list(ref['s_nlevels'])]
        assert list(res.s_names) == refutil.as_list(ref['s_names'])
    if res.with_g and res.group_g is not None:
        assert list(res.g_names) == refutil.as_list(ref['g_names'])
        assert list(res.group_g.nlevels_f) == [int(v) for v in refutil.as_list(ref['g_nlevels_f'])]
        assert list(res.group_g.nlevels) == [int(v) for v in refutil.as_list(ref['g_nlevels'])]
        assert list(res.group_g.levels_f[0]) == refutil.as_list(ref['g_levels_f'])
        vector(res.group_g.levels_k, ref['g_levels_k'], 'g_levels_k')
        if ref['g_levels_comb_k'] is not None:
            vector(res.group_g.levels_comb_k, ref['g_levels_comb_k'], 'g_levels_comb_k')
        vector(res.G, numpy.array(ref['G']).reshape(-1), 'G', rtol=FLAT.get(name, VC_RTOL), atol=VC_ATOL)
    if res.with_h and res.group_h is not None:
        assert list(res.h_names) == refutil.as_list(ref['h_names'])
        vector(res.group_h.levels_k, ref['h_levels_k'], 'h_levels_k')
        vector(res.H, numpy.array(ref['H']).reshape(-1), 'H', rtol=FLAT.get(name, VC_RTOL), atol=VC_ATOL)
    if name not in NAMES_DIFFER:
        assert list(res.coef_names) == refutil.as_list(ref['coef_names'])
    for key, got in (('sigma2_fix', 'sigma2'), ('tau2_fix', 'tau2'), ('rho_fix', 'rho'), ('gamma2_fix', 'gamma2'), ('phi_fix', 'phi')):
        expected = ref[key]
        flags = res.vc_fix[got]
        if flags is None:
            assert expected is None or expected == [], key
        else:
            assert [bool(b) for b in flags] == [bool(v) for v in refutil.as_list(expected)], key


@pytest.mark.parametrize('name', sorted(CASES))
def test_derived(name: str) -> None:
    ref = REF[name]
    res = run(name)
    vector(res.hatvalues(), ref['hat'], 'hat', rtol=1e-5, atol=1e-6)
    vector(res.weights_(), ref['weights'], 'weights', rtol=1e-4, atol=1e-6)
    vector(res.residuals(), ref['residuals'], 'residuals', rtol=1e-5, atol=1e-7)
    vector(res.fitted(), ref['fitted'], 'fitted', rtol=1e-5, atol=1e-7)
    if res.int_only:
        vector(res.weights_('rowsum'), ref['weights_rowsum'], 'rowsum', rtol=1e-3, atol=1e-5)
    m = res.M.toarray() if hasattr(res.M, 'toarray') else res.M
    vector(numpy.diag(m), numpy.diag(numpy.array(ref['M'])), 'diag(M)', rtol=VC_RTOL, atol=1e-8)


@pytest.mark.parametrize('name', sorted(CASES))
def test_print(name: str) -> None:
    if name in PRINT_UNSTABLE:
        pytest.skip('printed digits sit on a flat likelihood / boundary where the optimum differs from R in the 4th decimal')
    if name in NAMES_DIFFER:
        pytest.skip('coefficient labels follow the formula engine, not R')
    ref = REF[name]
    res = run(name)
    assert refutil.r_print(res.summary()).strip() == refutil.r_print(ref['print']).strip()
    assert refutil.r_print(res.summary(showfit=True)).strip() == refutil.r_print(ref['print_showfit']).strip()


def test_dense_equals_sparse() -> None:
    dense = run('kon_nested')
    sparse = run('kon_sparse')
    vector(dense.sigma2, sparse.sigma2, 'sigma2', rtol=1e-5)
    vector(dense.beta, sparse.beta, 'beta', rtol=1e-6)


def test_single_term_equals_rma_uni() -> None:
    data = kon()
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | study', data=data)
    uni = metafopy.rma_uni('yi', 'vi', data=data, method='REML')
    vector(mv.sigma2, [uni.tau2], 'sigma2', rtol=1e-4)
    vector(mv.beta, uni.beta, 'beta', rtol=1e-5)
    vector(mv.se, uni.se, 'se', rtol=1e-5)
    vector(mv.logLik(), uni.logLik(), 'logLik', rtol=1e-6)


def test_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='Unknown'):
        metafopy.rma_mv('yi', 'vi', data=kon(), method='XX')
    with pytest.raises(metafopy.errors.InputError, match='Must specify'):
        metafopy.rma_mv('yi', None, data=kon())
    d = kon()
    d.loc[0, 'district'] = numpy.nan
    with pytest.raises(metafopy.errors.InputError, match='NAs'):
        metafopy.rma_mv('yi', 'vi', random='~ 1 | district', data=d)
    with pytest.raises(metafopy.errors.InputError, match='dimension names'):
        simulated(random='~ 1 | species', R={'species': matrix('sim_R')})
    with pytest.raises(metafopy.errors.InputError, match='non-negative'):
        konst(sigma2=[-1.0, 0.1])
    with pytest.raises(metafopy.errors.InputError, match='does not match'):
        konst(sigma2=[0.1, 0.1, 0.1])
    with pytest.raises(metafopy.errors.InputError, match="cannot use ':'"):
        konst(random='~ 1 | district:school')


def kon_predict_model(test: str = 'z') -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=kon(), test=test)


@pytest.mark.parametrize(('key', 'test', 'newmods'), [('predict_kon', 'z', [-5, 0, 5]), ('predict_kon_t', 't', [-5, 0, 5]), ('predict_kon_fitted', 'z', None)])
def test_predict(key: str, test: str, newmods: list[int] | None) -> None:
    ref = REF[key]
    pr = kon_predict_model(test).predict(newmods)
    for field in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        if field in ref:
            vector(getattr(pr, field), ref[field], f'{key} {field}', rtol=1e-5, atol=1e-8)


def test_anova_lrt() -> None:
    ref = REF['anova_lrt']
    m0 = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon(), method='ML')
    m1 = metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=kon(), method='ML')
    out = m0.anova(m1)
    assert isinstance(out, metafopy.methods.anova.MVLRTResult)
    vector(out.LRT, ref['LRT'], 'LRT', rtol=1e-5)
    vector(out.pval, ref['pval'], 'pval', rtol=1e-4)
    vector(out.fit_stats_full, ref['fit_stats_f'], 'fit_stats_f', rtol=1e-5)
    vector(out.fit_stats_reduced, ref['fit_stats_r'], 'fit_stats_r', rtol=1e-5)
    assert (out.df_full, out.df_reduced) == tuple(ref['parms'])
    assert refutil.r_print(str(out)).strip() == refutil.r_print(ref['print']).strip()
    assert metafopy.methods.anova.anova(m1, m0).LRT == out.LRT


def test_anova_wald() -> None:
    ref = REF['anova_wald']
    out = kon_predict_model().anova(btt=[1])
    vector(out.QM, ref['QM'], 'QM', rtol=1e-5)
    vector(out.QMp, ref['QMp'], 'QMp', rtol=1e-5)
    assert refutil.r_print(str(out)).strip() == refutil.r_print(ref['print']).strip()


def test_fitstats() -> None:
    vector(kon_predict_model().fitstats(), REF['fitstats'], 'fitstats', rtol=1e-5)


def ci_model(name: str) -> metafopy.RMAMVResult:
    if name == 'kon':
        return metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon())
    if name == 'ass':
        return metafopy.rma_mv('yi', 'vi', random=['~ 1 | study', '~ 1 | esid'], data=ass())
    return berkey(mods='~ 0 + outcome', random='~ outcome | trial', struct=name.removeprefix('ber_'))


CI_CASES = {
    'ci_kon_sigma2_1': ('kon', {'sigma2': 0}),
    'ci_kon_sigma2_2': ('kon', {'sigma2': 1}),
    'ci_kon_level90': ('kon', {'sigma2': 0, 'level': 90}),
    'ci_ber_tau2_1': ('ber_UN', {'tau2': 0}),
    'ci_ber_rho': ('ber_UN', {'rho': 0}),
    'ci_ber_cs_tau2': ('ber_CS', {'tau2': 0}),
    'ci_ber_cs_rho': ('ber_CS', {'rho': 0}),
    'ci_ass_sigma2_1': ('ass', {'sigma2': 0}),
    'ci_ass_sigma2_2': ('ass', {'sigma2': 1}),
}


@pytest.mark.parametrize('key', sorted(CI_CASES))
def test_confint_random(key: str) -> None:
    name, kwargs = CI_CASES[key]
    out = ci_model(name).confint(**kwargs)
    assert out.random is not None
    vector(out.random.to_numpy(), numpy.array(REF[key]['random']).reshape(-1), key, rtol=2e-4, atol=1e-5)


def test_confint_fixed() -> None:
    out = ci_model('kon').confint(fixed=True)
    assert out.fixed is not None
    vector(out.fixed.to_numpy(), numpy.array(REF['ci_kon_fixed']['fixed']).reshape(-1), 'fixed', rtol=1e-5)


def test_confint_all_components() -> None:
    out = ci_model('kon').confint()
    assert isinstance(out, list)
    assert len(out) == 2
