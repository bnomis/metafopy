"""influence(), rstandard(), rstudent(), leave1out() and cumul() checked against R metafor (tests/r/generate_diagnostics.R)."""

import typing

import numpy
import pandas
import pytest
import refutil
import scipy.stats

import metafopy
import metafopy.methods.influence
import metafopy.transf

REF = refutil.load_phase2('diagnostics')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def dna() -> pandas.DataFrame:
    return refutil.bcg_na()


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
INFL_CASES: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'infl_re': ({}, {}),
    'infl_mods': ({'mods': '~ ablat + year'}, {}),
    'infl_mods_btt': ({'mods': '~ ablat + year'}, {'btt': [1, 2]}),
    'infl_ee': ({'method': 'EE'}, {}),
    'infl_dl': ({'method': 'DL'}, {}),
    'infl_ml_mods': ({'mods': '~ ablat', 'method': 'ML'}, {}),
    'infl_knha': ({'mods': '~ ablat', 'test': 'knha'}, {}),
    'infl_weights': ({'mods': '~ ablat', 'weights': 'n'}, {}),
    'infl_unweighted': ({'mods': '~ ablat', 'weighted': False}, {}),
    'infl_slab': ({'slab': 'label'}, {}),
    'infl_fixtau2': ({'tau2': 0.2}, {}),
}


@pytest.mark.parametrize('key', list(INFL_CASES))
def test_influence_matches_metafor(dat: pandas.DataFrame, key: str) -> None:
    fit_kw, infl_kw = INFL_CASES[key]
    res = metafopy.rma('yi', 'vi', data=dat, **fit_kw)
    out = metafopy.influence(res, **infl_kw)
    ref = REF[key]
    for attr, name in (
        ('rstudent', 'rstudent'),
        ('dffits', 'dffits'),
        ('cook_d', 'cook.d'),
        ('cov_r', 'cov.r'),
        ('tau2_del', 'tau2.del'),
        ('QE_del', 'QE.del'),
        ('hat', 'hat'),
        ('weight', 'weight'),
    ):
        refutil.close(getattr(out, attr), ref[name], f'{key}: {name}')
    for j, name in enumerate(res.coef_names):
        refutil.close(out.dfbs[:, j], ref['dfbs'][name], f'{key}: dfbs {name}')
    assert list(out.inf) == ['*' if v == '*' else '' for v in refutil.as_list(ref['inf'])]
    assert out.m == ref['m']
    assert out.is_infl.tolist() == [bool(v) for v in refutil.as_list(ref['is.infl'])]
    assert refutil.same_print(str(out), ref['print'])
    assert refutil.same_print(out.summary(infonly=True), ref['print_infonly'])


def test_influence_na_handling(dat: pandas.DataFrame, dna: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dna)
    out = metafopy.influence(res)
    ref = REF['infl_na_omit']
    refutil.close(out.rstudent, ref['rstudent'])
    refutil.close(out.cook_d, ref['cook.d'])
    out_x = metafopy.influence(res, na_action='exclude')
    ref_x = REF['infl_na_exclude']
    assert len(out_x.rstudent) == 13
    refutil.close(out_x.rstudent, ref_x['rstudent'])
    refutil.close(out_x.hat, ref_x['hat'])
    refutil.close(out_x.dfbs[:, 1], ref_x['dfbs']['ablat'])
    assert out_x.is_infl.tolist() == [bool(v) for v in refutil.as_list(ref_x['is.infl'])]


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
RESID_CASES: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'rstandard_marginal': ({}, {}),
    'rstandard_conditional': ({}, {'type': 'conditional'}),
    'rstandard_mods': ({'mods': '~ ablat + year'}, {}),
    'rstandard_mods_cond': ({'mods': '~ ablat + year'}, {'type': 'conditional'}),
    'rstandard_ee': ({'method': 'EE'}, {}),
    'rstandard_weights': ({'mods': '~ ablat', 'weights': 'n'}, {}),
    'rstandard_unweighted': ({'mods': '~ ablat', 'weighted': False}, {}),
}


@pytest.mark.parametrize('key', list(RESID_CASES))
def test_rstandard_matches_metafor(dat: pandas.DataFrame, key: str) -> None:
    fit_kw, kw = RESID_CASES[key]
    out = metafopy.rstandard(metafopy.rma('yi', 'vi', data=dat, **fit_kw), **kw)
    ref = REF[key]
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    for attr in ('resid', 'se', 'z'):
        refutil.close(getattr(out, attr), ref[attr], f'{key}: {attr}')


def test_rstandard_conditional_rejects_nonstandard_weights(dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='non-standard weights'):
        metafopy.rstandard(metafopy.rma('yi', 'vi', weights='n', data=dat), type='conditional')


def test_rstandard_rstudent_print(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat)
    assert refutil.r_print(str(metafopy.rstandard(res))) == refutil.r_print(REF['rstandard_print'])
    assert refutil.r_print(str(metafopy.rstudent(res))) == refutil.r_print(REF['rstudent_print'])


def test_rstudent_cooks_dfbetas(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat)
    mres = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    for attr in ('resid', 'se', 'z'):
        refutil.close(getattr(metafopy.rstudent(res), attr), REF['rstudent_re'][attr])
        refutil.close(getattr(metafopy.rstudent(mres), attr), REF['rstudent_mods'][attr])
    cd = metafopy.cooks_distance(res)
    refutil.close(cd.to_numpy(), REF['cooks_re']['cook'])
    assert list(cd.index) == [str(i) for i in range(1, 14)]
    refutil.close(metafopy.cooks_distance(mres).to_numpy(), REF['cooks_mods']['cook'])
    dfb = metafopy.dfbetas(mres)
    for name in ('intrcpt', 'ablat', 'year'):
        refutil.close(dfb[name].to_numpy(), REF['dfbetas_mods'][name])


def test_hatvalues(dat: pandas.DataFrame) -> None:
    mres = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)
    refutil.close(metafopy.methods.influence.hatvalues(mres), REF['hat_matrix']['diag'])
    full = metafopy.methods.influence.hatvalues(mres, type='matrix')
    for i, row in enumerate(REF['hat_matrix']['full']):
        refutil.close(full[i], row)


def test_residuals_with_missing(dna: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dna)
    for fn, key in ((metafopy.rstandard, 'rstandard'), (metafopy.rstudent, 'rstudent')):
        out = fn(res, na_action='exclude')
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        for attr in ('resid', 'se', 'z'):
            refutil.close(getattr(out, attr), REF[f'{key}_na_exclude'][attr], f'{key} {attr}')
    out = metafopy.rstandard(res)
    refutil.close(out.z, REF['rstandard_na_omit']['z'])


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
L1O_CASES: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'l1o_re': ({}, {}),
    'l1o_exp': ({}, {'transf': numpy.exp}),
    'l1o_ee': ({'method': 'EE'}, {}),
    'l1o_knha': ({'test': 'knha'}, {}),
    'l1o_dl': ({'method': 'DL'}, {}),
    'l1o_ml': ({'method': 'ML'}, {}),
    'l1o_exp_int': ({}, {'transf': metafopy.transf.transf_exp_int}),
    'l1o_cluster': ({}, {'cluster': 'alloc'}),
    'l1o_weights': ({'weights': 'n'}, {}),
    'l1o_level90': ({'level': 90}, {}),
}

COLUMNS = (
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    ('estimate', 'estimate'),
    ('se', 'se'),
    ('pval', 'pval'),
    ('ci_lb', 'ci.lb'),
    ('ci_ub', 'ci.ub'),
    ('Q', 'Q'),
    ('Qp', 'Qp'),
    ('I2', 'I2'),
    ('H2', 'H2'),
)


def check_table(out: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    for attr, name in COLUMNS:
        refutil.close(getattr(out, attr), ref[name], f'{what}: {name}')
    refutil.close(out.stat, ref['tval' if 'tval' in ref else 'zval'], f'{what}: stat')
    if 'tau2' in ref:
        refutil.close(out.tau2, ref['tau2'], f'{what}: tau2')
    else:
        assert out.tau2 is None
    assert [str(s) for s in out.slab] == [str(s) for s in refutil.as_list(ref['slab'])]


@pytest.mark.parametrize('key', list(L1O_CASES))
def test_leave1out_matches_metafor(dat: pandas.DataFrame, key: str) -> None:
    fit_kw, kw = L1O_CASES[key]
    res = metafopy.rma('yi', 'vi', data=dat, **fit_kw)
    out = metafopy.leave1out(res, **kw)
    ref = REF[key]
    check_table(out, ref, key)
    assert refutil.same_print(str(out), ref['print'])


def test_leave1out_missing_values(dna: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dna)
    check_table(metafopy.leave1out(res), REF['l1o_na_omit'], 'omit')
    check_table(metafopy.leave1out(res, na_action='exclude'), REF['l1o_na_exclude'], 'exclude')


def test_leave1out_requires_intercept_only(dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='without moderators'):
        metafopy.leave1out(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat))


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
CUMUL_CASES: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'cumul_re': ({}, {}),
    'cumul_year': ({}, {'order': 'year'}),
    'cumul_year_dec': ({}, {'order': 'year', 'decreasing': True}),
    'cumul_collapse': ({}, {'order': 'alloc', 'collapse': True}),
    'cumul_collapse_year': ({}, {'order': 'year', 'collapse': True}),
    'cumul_exp': ({}, {'order': 'year', 'transf': numpy.exp}),
    'cumul_ee': ({'method': 'EE'}, {'order': 'year'}),
    'cumul_knha': ({'test': 'knha'}, {'order': 'year'}),
    'cumul_dl': ({'method': 'DL'}, {'order': 'ablat'}),
}


@pytest.mark.parametrize('key', list(CUMUL_CASES))
def test_cumul_matches_metafor(dat: pandas.DataFrame, key: str) -> None:
    fit_kw, kw = CUMUL_CASES[key]
    res = metafopy.rma('yi', 'vi', data=dat, **fit_kw)
    out = metafopy.cumul(res, **kw)
    ref = REF[key]
    check_table(out, ref, key)
    assert out.k is not None
    refutil.close(out.k, ref['k'])
    assert refutil.same_print(str(out), ref['print'])


# -- permutest ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    d8, d9 = dat.iloc[:8], dat.iloc[:9]
    return {
        'r8': metafopy.rma('yi', 'vi', data=d8),
        'r8_knha': metafopy.rma('yi', 'vi', data=d8, test='knha'),
        'r8_ee': metafopy.rma('yi', 'vi', data=d8, method='EE'),
        'r8_tau2': metafopy.rma('yi', 'vi', data=d8, tau2=0.1),
        'r9m': metafopy.rma('yi', 'vi', mods='~ alloc', data=d9),
        'r9m_knha': metafopy.rma('yi', 'vi', mods='~ alloc', data=d9, test='knha'),
        'r13': metafopy.rma('yi', 'vi', data=dat),
        'r13m': metafopy.rma('yi', 'vi', mods='~ ablat', data=dat),
        'r13mm': metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat),
        'r13m_knha': metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, test='knha'),
        'r13_dl': metafopy.rma('yi', 'vi', data=dat, method='DL'),
    }


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PERM_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'perm_exact_int': ('r8', {'exact': True}),
    'perm_exact_int_knha': ('r8_knha', {'exact': True}),
    'perm_exact_int_ee': ('r8_ee', {'exact': True}),
    'perm_exact_int_coef': ('r8', {'exact': True, 'control': {'stat': 'coef'}}),
    'perm_exact_int_px2': ('r8', {'exact': True, 'control': {'p2defn': 'px2'}}),
    'perm_exact_int_less': ('r8', {'exact': True, 'control': {'alternative': 'less'}}),
    'perm_exact_int_greater': ('r8', {'exact': True, 'control': {'alternative': 'greater'}}),
    'perm_exact_mods': ('r9m', {'exact': True}),
    'perm_exact_mods_knha': ('r9m_knha', {'exact': True}),
    'perm_exact_auto': ('r8', {'iter': 1000}),
    'perm_exact_permci': ('r8', {'exact': True, 'permci': True}),
    'perm_exact_permci_mods': ('r9m', {'exact': True, 'permci': [1]}),
    'perm_approx_int': ('r13', {'iter': 200, 'seed': 1234}),
    'perm_approx_mods': ('r13m', {'iter': 200, 'seed': 99}),
    'perm_approx_mods_multi': ('r13mm', {'iter': 150, 'seed': 7}),
    'perm_approx_btt': ('r13mm', {'iter': 150, 'seed': 7, 'btt': [1]}),
    'perm_approx_knha': ('r13m_knha', {'iter': 100, 'seed': 5}),
    'perm_approx_dl': ('r13_dl', {'iter': 100, 'seed': 3}),
    'perm_tau2_fixed': ('r8_tau2', {'exact': True}),
}


@pytest.mark.parametrize('key', list(PERM_CASES))
def test_permutest_matches_metafor(fits: dict[str, typing.Any], key: str) -> None:
    fit, kw = PERM_CASES[key]
    out = metafopy.permutest(fits[fit], **kw)
    assert isinstance(out, metafopy.PermutestResult)
    ref = REF[key]
    refutil.close(out.pval, ref['pval'], f'{key}: pval')
    refutil.close(out.QMp, ref['QMp'], f'{key}: QMp')
    refutil.close(out.QM_perm, ref['QM.perm'], f'{key}: QM.perm', rtol=1e-7)
    for name in out.coef_names:
        refutil.close(out.zval_perm[name], ref['zval.perm'][name], f'{key}: zval.perm {name}', rtol=1e-7)
        refutil.close(out.beta_perm[name], ref['beta.perm'][name], f'{key}: beta.perm {name}', rtol=1e-7)
    refutil.close(out.ci_lb, ref['ci.lb'], f'{key}: ci.lb', rtol=1e-6)
    refutil.close(out.ci_ub, ref['ci.ub'], f'{key}: ci.ub', rtol=1e-6)
    assert out.exact_iter == ref['exact.iter']
    assert out.permci == ref['permci']
    assert out.alternative == ref['alternative']
    assert refutil.same_print(str(out), ref['print']), str(out)


def test_permutest_exact_iterations(dat: pandas.DataFrame, fits: dict[str, typing.Any]) -> None:
    ref = REF['perm_exact_iter_i']
    assert metafopy.permutest(fits['r13'], exact='i') == ref['int']
    assert metafopy.permutest(fits['r13m'], exact='i') == ref['mods']
    assert metafopy.permutest(fits['r9m'], exact='i') == ref['mods9']


# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
@pytest.mark.parametrize(('key', 'fit', 'kw'), [('perm_approx_global', 'r13m', {}), ('perm_approx_global2', 'r13', {})])
def test_permutest_global_seed(fits: dict[str, typing.Any], key: str, fit: str, kw: dict[str, typing.Any]) -> None:
    # R: set.seed(42); permutest(r13m, iter=100); permutest(r13, iter=100) -- the second call continues the stream
    metafopy.set_seed(42)
    first = metafopy.permutest(fits['r13m'], iter=100)
    second = metafopy.permutest(fits['r13'], iter=100)
    out = first if key == 'perm_approx_global' else second
    assert isinstance(out, metafopy.PermutestResult)
    refutil.close(out.pval, REF[key]['pval'])
    refutil.close(out.QM_perm, REF[key]['QM.perm'], rtol=1e-7)


def test_permutest_permci_global_seed(fits: dict[str, typing.Any]) -> None:
    metafopy.set_seed(11)
    out = metafopy.permutest(fits['r13m'], iter=100, permci=True)
    assert isinstance(out, metafopy.PermutestResult)
    ref = REF['perm_approx_permci']
    refutil.close(out.pval, ref['pval'])
    refutil.close(out.ci_lb, ref['ci.lb'], rtol=1e-6)
    refutil.close(out.ci_ub, ref['ci.ub'], rtol=1e-6)


# -- profile ------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PROF_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'prof_reml': ('r13', {}),
    'prof_dl': ('r13_dl', {}),
    'prof_mods': ('r13m', {}),
    'prof_steps': ('r13', {'steps': 10}),
    'prof_xlim': ('r13', {'xlim': (0.05, 1.2), 'steps': 12}),
    'prof_ylim': ('r13', {'ylim': (-30, -10)}),
    'prof_seq': ('r13', {'steps': [0.01, 0.1, 0.2, 0.5]}),
    'prof_sqrt': ('r13', {'sqrt': True}),
    'prof_exp': ('r13', {'exp': True}),
    'prof_knha': ('r13m_knha', {}),
}


@pytest.mark.parametrize('key', list(PROF_CASES))
def test_profile_matches_metafor(fits: dict[str, typing.Any], key: str) -> None:
    fit, kw = PROF_CASES[key]
    out = metafopy.profile(fits[fit], **kw)
    ref = REF[key]
    refutil.close(out.tau2, ref['grid'], f'{key}: grid')
    refutil.close(out.ll, ref['ll'], f'{key}: ll', rtol=1e-7)
    for name in out.beta.columns:
        refutil.close(out.beta[name], ref['beta'][name], f'{key}: beta {name}', rtol=1e-7)
        refutil.close(out.ci_lb[name], ref['ci.lb'][name], f'{key}: ci.lb {name}', rtol=1e-7)
        refutil.close(out.ci_ub[name], ref['ci.ub'][name], f'{key}: ci.ub {name}', rtol=1e-7)
    refutil.close(out.xlim, ref['xlim'], f'{key}: xlim', rtol=1e-6)
    refutil.close(out.ylim, ref['ylim'], f'{key}: ylim', rtol=1e-6)
    refutil.close(out.vc, ref['vc'])
    refutil.close(out.maxll, ref['maxll'])
    refutil.close(out.I2, ref['I2'], rtol=1e-7)
    refutil.close(out.H2, ref['H2'], rtol=1e-7)


def test_profile_ml(dat: pandas.DataFrame) -> None:
    out = metafopy.profile(metafopy.rma('yi', 'vi', data=dat, method='ML'))
    refutil.close(out.ll, REF['prof_ml']['ll'], rtol=1e-7)
    refutil.close(out.xlim, REF['prof_ml']['xlim'], rtol=1e-6)


def test_profile_rejects_equal_effects(dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='Cannot profile'):
        metafopy.profile(metafopy.rma('yi', 'vi', data=dat, method='EE'))


# -- gosh ---------------------------------------------------------------------------


def check_gosh(out: typing.Any, ref: dict[str, typing.Any], what: str, *, same_subsets: bool = True) -> None:
    assert out.k == ref['k']
    ref_incl = numpy.array(ref['incl'], dtype=bool)
    if same_subsets:
        numpy.testing.assert_array_equal(out.incl, ref_incl, err_msg=f'{what}: incl')
    assert out.fit.tolist() == [bool(v) for v in refutil.as_list(ref['fit'])]
    for col in out.res.columns:
        rname = 'estimate' if col == 'estimate' else col
        refutil.close(out.res[col], ref['res'][rname], f'{what}: {col}', rtol=1e-7)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
GOSH_EXACT: dict[str, tuple[str, str]] = {
    'gosh_int_reml': ('r8', 'gosh_int_reml'),
    'gosh_int_ee': ('r8_ee', 'gosh_int_ee'),
    'gosh_int_dl': ('r8_dl', 'gosh_int_dl'),
    'gosh_int_ml': ('r8_ml', 'gosh_int_ml'),
    'gosh_mods': ('r9_ablat', 'gosh_mods'),
    'gosh_mods_knha': ('r9_ablat_knha', 'gosh_mods_knha'),
    'gosh_weights': ('r8_w', 'gosh_weights'),
}


@pytest.fixture(scope='module')
def gosh_fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    d8, d9 = dat.iloc[:8], dat.iloc[:9]
    return {
        'r8': metafopy.rma('yi', 'vi', data=d8),
        'r8_ee': metafopy.rma('yi', 'vi', data=d8, method='EE'),
        'r8_dl': metafopy.rma('yi', 'vi', data=d8, method='DL'),
        'r8_ml': metafopy.rma('yi', 'vi', data=d8, method='ML'),
        'r8_w': metafopy.rma('yi', 'vi', data=d8, weights='n'),
        'r9_ablat': metafopy.rma('yi', 'vi', mods='~ ablat', data=d9),
        'r9_ablat_knha': metafopy.rma('yi', 'vi', mods='~ ablat', data=d9, test='knha'),
    }


@pytest.mark.parametrize('key', list(GOSH_EXACT))
def test_gosh_exact_matches_metafor(gosh_fits: dict[str, typing.Any], key: str) -> None:
    fit, ref_key = GOSH_EXACT[key]
    out = metafopy.gosh(gosh_fits[fit])
    check_gosh(out, REF[ref_key], key)
    assert refutil.same_print(str(out), REF[ref_key]['print']), str(out)


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    ('key', 'fit', 'kw'),
    [
        ('gosh_random_int', 'r13', {'subsets': 300, 'seed': 123}),
        ('gosh_random_mods', 'r13m', {'subsets': 200, 'seed': 7}),
    ],
)
def test_gosh_random_subsets(fits: dict[str, typing.Any], monkeypatch: pytest.MonkeyPatch, key: str, fit: str, kw: dict[str, typing.Any]) -> None:
    # Subset sizes and study indices are drawn with R's generator. SciPy's binomial pmf differs from R's dbinom in the
    # last bit, which reorders tied sizes inside R's sampler, so feed R's probabilities (as 17 digit strings) to compare
    # the rest of the pipeline.
    ref = REF[key]
    monkeypatch.setattr(scipy.stats.binom, 'pmf', lambda *a, **k: numpy.array([float(v) for v in ref['dbinom']]))
    check_gosh(metafopy.gosh(fits[fit], **kw), ref, key)


def test_gosh_random_subsets_global_seed(fits: dict[str, typing.Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ref = REF['gosh_random_global']
    monkeypatch.setattr(scipy.stats.binom, 'pmf', lambda *a, **k: numpy.array([float(v) for v in ref['dbinom']]))
    metafopy.set_seed(99)
    check_gosh(metafopy.gosh(fits['r13'], subsets=100), ref, 'global seed')


def test_gosh_random_subsets_distribution(fits: dict[str, typing.Any]) -> None:
    out = metafopy.gosh(fits['r13'], subsets=300, seed=123)
    assert out.incl.shape == (300, 13)
    assert (out.res['k'].to_numpy() == out.incl.sum(axis=1)).all()
    assert (numpy.diff(out.res['k'].to_numpy()) >= 0).all()
