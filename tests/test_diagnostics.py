"""influence(), rstandard(), rstudent(), leave1out() and cumul() checked against R metafor (tests/r/generate_diagnostics.R)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from metafopy import PermutestResult, cooks_distance, cumul, dfbetas, gosh, influence, leave1out, permutest, profile, rma, rstandard, rstudent, set_seed
from metafopy.methods.influence import hatvalues
from metafopy.transf import transf_exp_int
from refutil import as_list, bcg, bcg_na, close, load_phase2, r_print, same_print

REF = load_phase2('diagnostics')


@pytest.fixture(scope='module')
def dat() -> pd.DataFrame:
    return bcg()


@pytest.fixture(scope='module')
def dna() -> pd.DataFrame:
    return bcg_na()


INFL_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
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
def test_influence_matches_metafor(dat: pd.DataFrame, key: str) -> None:
    fit_kw, infl_kw = INFL_CASES[key]
    res = rma('yi', 'vi', data=dat, **fit_kw)
    out = influence(res, **infl_kw)
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
        close(getattr(out, attr), ref[name], f'{key}: {name}')
    for j, name in enumerate(res.coef_names):
        close(out.dfbs[:, j], ref['dfbs'][name], f'{key}: dfbs {name}')
    assert list(out.inf) == ['*' if v == '*' else '' for v in as_list(ref['inf'])]
    assert out.m == ref['m']
    assert out.is_infl.tolist() == [bool(v) for v in as_list(ref['is.infl'])]
    assert same_print(str(out), ref['print'])
    assert same_print(out.summary(infonly=True), ref['print_infonly'])


def test_influence_na_handling(dat: pd.DataFrame, dna: pd.DataFrame) -> None:
    res = rma('yi', 'vi', mods='~ ablat', data=dna)
    out = influence(res)
    ref = REF['infl_na_omit']
    close(out.rstudent, ref['rstudent'])
    close(out.cook_d, ref['cook.d'])
    out_x = influence(res, na_action='exclude')
    ref_x = REF['infl_na_exclude']
    assert len(out_x.rstudent) == 13
    close(out_x.rstudent, ref_x['rstudent'])
    close(out_x.hat, ref_x['hat'])
    close(out_x.dfbs[:, 1], ref_x['dfbs']['ablat'])
    assert out_x.is_infl.tolist() == [bool(v) for v in as_list(ref_x['is.infl'])]


RESID_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'rstandard_marginal': ({}, {}),
    'rstandard_conditional': ({}, {'type': 'conditional'}),
    'rstandard_mods': ({'mods': '~ ablat + year'}, {}),
    'rstandard_mods_cond': ({'mods': '~ ablat + year'}, {'type': 'conditional'}),
    'rstandard_ee': ({'method': 'EE'}, {}),
    'rstandard_weights': ({'mods': '~ ablat', 'weights': 'n'}, {}),
    'rstandard_unweighted': ({'mods': '~ ablat', 'weighted': False}, {}),
}


@pytest.mark.parametrize('key', list(RESID_CASES))
def test_rstandard_matches_metafor(dat: pd.DataFrame, key: str) -> None:
    fit_kw, kw = RESID_CASES[key]
    out = rstandard(rma('yi', 'vi', data=dat, **fit_kw), **kw)
    ref = REF[key]
    for attr in ('resid', 'se', 'z'):
        close(getattr(out, attr), ref[attr], f'{key}: {attr}')


def test_rstandard_conditional_rejects_nonstandard_weights(dat: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='non-standard weights'):
        rstandard(rma('yi', 'vi', weights='n', data=dat), type='conditional')


def test_rstandard_rstudent_print(dat: pd.DataFrame) -> None:
    res = rma('yi', 'vi', data=dat)
    assert r_print(str(rstandard(res))) == r_print(REF['rstandard_print'])
    assert r_print(str(rstudent(res))) == r_print(REF['rstudent_print'])


def test_rstudent_cooks_dfbetas(dat: pd.DataFrame) -> None:
    res = rma('yi', 'vi', data=dat)
    mres = rma('yi', 'vi', mods='~ ablat + year', data=dat)
    for attr in ('resid', 'se', 'z'):
        close(getattr(rstudent(res), attr), REF['rstudent_re'][attr])
        close(getattr(rstudent(mres), attr), REF['rstudent_mods'][attr])
    cd = cooks_distance(res)
    close(cd.to_numpy(), REF['cooks_re']['cook'])
    assert list(cd.index) == [str(i) for i in range(1, 14)]
    close(cooks_distance(mres).to_numpy(), REF['cooks_mods']['cook'])
    dfb = dfbetas(mres)
    for name in ('intrcpt', 'ablat', 'year'):
        close(dfb[name].to_numpy(), REF['dfbetas_mods'][name])


def test_hatvalues(dat: pd.DataFrame) -> None:
    mres = rma('yi', 'vi', mods='~ ablat + year', data=dat)
    close(hatvalues(mres), REF['hat_matrix']['diag'])
    full = hatvalues(mres, type='matrix')
    for i, row in enumerate(REF['hat_matrix']['full']):
        close(full[i], row)


def test_residuals_with_missing(dna: pd.DataFrame) -> None:
    res = rma('yi', 'vi', mods='~ ablat', data=dna)
    for fn, key in ((rstandard, 'rstandard'), (rstudent, 'rstudent')):
        out = fn(res, na_action='exclude')
        for attr in ('resid', 'se', 'z'):
            close(getattr(out, attr), REF[f'{key}_na_exclude'][attr], f'{key} {attr}')
    out = rstandard(res)
    close(out.z, REF['rstandard_na_omit']['z'])


L1O_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'l1o_re': ({}, {}),
    'l1o_exp': ({}, {'transf': np.exp}),
    'l1o_ee': ({'method': 'EE'}, {}),
    'l1o_knha': ({'test': 'knha'}, {}),
    'l1o_dl': ({'method': 'DL'}, {}),
    'l1o_ml': ({'method': 'ML'}, {}),
    'l1o_exp_int': ({}, {'transf': transf_exp_int}),
    'l1o_cluster': ({}, {'cluster': 'alloc'}),
    'l1o_weights': ({'weights': 'n'}, {}),
    'l1o_level90': ({'level': 90}, {}),
}

_COLUMNS = (
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


def _check_table(out: Any, ref: dict[str, Any], what: str) -> None:
    for attr, name in _COLUMNS:
        close(getattr(out, attr), ref[name], f'{what}: {name}')
    close(out.stat, ref['tval' if 'tval' in ref else 'zval'], f'{what}: stat')
    if 'tau2' in ref:
        close(out.tau2, ref['tau2'], f'{what}: tau2')
    else:
        assert out.tau2 is None
    assert [str(s) for s in out.slab] == [str(s) for s in as_list(ref['slab'])]


@pytest.mark.parametrize('key', list(L1O_CASES))
def test_leave1out_matches_metafor(dat: pd.DataFrame, key: str) -> None:
    fit_kw, kw = L1O_CASES[key]
    res = rma('yi', 'vi', data=dat, **fit_kw)
    out = leave1out(res, **kw)
    ref = REF[key]
    _check_table(out, ref, key)
    assert same_print(str(out), ref['print'])


def test_leave1out_missing_values(dna: pd.DataFrame) -> None:
    res = rma('yi', 'vi', data=dna)
    _check_table(leave1out(res), REF['l1o_na_omit'], 'omit')
    _check_table(leave1out(res, na_action='exclude'), REF['l1o_na_exclude'], 'exclude')


def test_leave1out_requires_intercept_only(dat: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='without moderators'):
        leave1out(rma('yi', 'vi', mods='~ ablat', data=dat))


CUMUL_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'cumul_re': ({}, {}),
    'cumul_year': ({}, {'order': 'year'}),
    'cumul_year_dec': ({}, {'order': 'year', 'decreasing': True}),
    'cumul_collapse': ({}, {'order': 'alloc', 'collapse': True}),
    'cumul_collapse_year': ({}, {'order': 'year', 'collapse': True}),
    'cumul_exp': ({}, {'order': 'year', 'transf': np.exp}),
    'cumul_ee': ({'method': 'EE'}, {'order': 'year'}),
    'cumul_knha': ({'test': 'knha'}, {'order': 'year'}),
    'cumul_dl': ({'method': 'DL'}, {'order': 'ablat'}),
}


@pytest.mark.parametrize('key', list(CUMUL_CASES))
def test_cumul_matches_metafor(dat: pd.DataFrame, key: str) -> None:
    fit_kw, kw = CUMUL_CASES[key]
    res = rma('yi', 'vi', data=dat, **fit_kw)
    out = cumul(res, **kw)
    ref = REF[key]
    _check_table(out, ref, key)
    assert out.k is not None
    close(out.k, ref['k'])
    assert same_print(str(out), ref['print'])


# -- permutest ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def fits(dat: pd.DataFrame) -> dict[str, Any]:
    d8, d9 = dat.iloc[:8], dat.iloc[:9]
    return {
        'r8': rma('yi', 'vi', data=d8),
        'r8_knha': rma('yi', 'vi', data=d8, test='knha'),
        'r8_ee': rma('yi', 'vi', data=d8, method='EE'),
        'r8_tau2': rma('yi', 'vi', data=d8, tau2=0.1),
        'r9m': rma('yi', 'vi', mods='~ alloc', data=d9),
        'r9m_knha': rma('yi', 'vi', mods='~ alloc', data=d9, test='knha'),
        'r13': rma('yi', 'vi', data=dat),
        'r13m': rma('yi', 'vi', mods='~ ablat', data=dat),
        'r13mm': rma('yi', 'vi', mods='~ ablat + year', data=dat),
        'r13m_knha': rma('yi', 'vi', mods='~ ablat', data=dat, test='knha'),
        'r13_dl': rma('yi', 'vi', data=dat, method='DL'),
    }


PERM_CASES: dict[str, tuple[str, dict[str, Any]]] = {
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
def test_permutest_matches_metafor(fits: dict[str, Any], key: str) -> None:
    fit, kw = PERM_CASES[key]
    out = permutest(fits[fit], **kw)
    assert isinstance(out, PermutestResult)
    ref = REF[key]
    close(out.pval, ref['pval'], f'{key}: pval')
    close(out.QMp, ref['QMp'], f'{key}: QMp')
    close(out.QM_perm, ref['QM.perm'], f'{key}: QM.perm', rtol=1e-7)
    for name in out.coef_names:
        close(out.zval_perm[name], ref['zval.perm'][name], f'{key}: zval.perm {name}', rtol=1e-7)
        close(out.beta_perm[name], ref['beta.perm'][name], f'{key}: beta.perm {name}', rtol=1e-7)
    close(out.ci_lb, ref['ci.lb'], f'{key}: ci.lb', rtol=1e-6)
    close(out.ci_ub, ref['ci.ub'], f'{key}: ci.ub', rtol=1e-6)
    assert out.exact_iter == ref['exact.iter']
    assert out.permci == ref['permci']
    assert out.alternative == ref['alternative']
    assert same_print(str(out), ref['print']), str(out)


def test_permutest_exact_iterations(dat: pd.DataFrame, fits: dict[str, Any]) -> None:
    ref = REF['perm_exact_iter_i']
    assert permutest(fits['r13'], exact='i') == ref['int']
    assert permutest(fits['r13m'], exact='i') == ref['mods']
    assert permutest(fits['r9m'], exact='i') == ref['mods9']


@pytest.mark.parametrize(('key', 'fit', 'kw'), [('perm_approx_global', 'r13m', {}), ('perm_approx_global2', 'r13', {})])
def test_permutest_global_seed(fits: dict[str, Any], key: str, fit: str, kw: dict[str, Any]) -> None:
    # R: set.seed(42); permutest(r13m, iter=100); permutest(r13, iter=100) -- the second call continues the stream
    set_seed(42)
    first = permutest(fits['r13m'], iter=100)
    second = permutest(fits['r13'], iter=100)
    out = first if key == 'perm_approx_global' else second
    assert isinstance(out, PermutestResult)
    close(out.pval, REF[key]['pval'])
    close(out.QM_perm, REF[key]['QM.perm'], rtol=1e-7)


def test_permutest_permci_global_seed(fits: dict[str, Any]) -> None:
    set_seed(11)
    out = permutest(fits['r13m'], iter=100, permci=True)
    assert isinstance(out, PermutestResult)
    ref = REF['perm_approx_permci']
    close(out.pval, ref['pval'])
    close(out.ci_lb, ref['ci.lb'], rtol=1e-6)
    close(out.ci_ub, ref['ci.ub'], rtol=1e-6)


# -- profile ------------------------------------------------------------------------

PROF_CASES: dict[str, tuple[str, dict[str, Any]]] = {
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
def test_profile_matches_metafor(fits: dict[str, Any], key: str) -> None:
    fit, kw = PROF_CASES[key]
    out = profile(fits[fit], **kw)
    ref = REF[key]
    close(out.tau2, ref['grid'], f'{key}: grid')
    close(out.ll, ref['ll'], f'{key}: ll', rtol=1e-7)
    for name in out.beta.columns:
        close(out.beta[name], ref['beta'][name], f'{key}: beta {name}', rtol=1e-7)
        close(out.ci_lb[name], ref['ci.lb'][name], f'{key}: ci.lb {name}', rtol=1e-7)
        close(out.ci_ub[name], ref['ci.ub'][name], f'{key}: ci.ub {name}', rtol=1e-7)
    close(out.xlim, ref['xlim'], f'{key}: xlim', rtol=1e-6)
    close(out.ylim, ref['ylim'], f'{key}: ylim', rtol=1e-6)
    close(out.vc, ref['vc'])
    close(out.maxll, ref['maxll'])
    close(out.I2, ref['I2'], rtol=1e-7)
    close(out.H2, ref['H2'], rtol=1e-7)


def test_profile_ml(dat: pd.DataFrame) -> None:
    out = profile(rma('yi', 'vi', data=dat, method='ML'))
    close(out.ll, REF['prof_ml']['ll'], rtol=1e-7)
    close(out.xlim, REF['prof_ml']['xlim'], rtol=1e-6)


def test_profile_rejects_equal_effects(dat: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='Cannot profile'):
        profile(rma('yi', 'vi', data=dat, method='EE'))


# -- gosh ---------------------------------------------------------------------------


def _check_gosh(out: Any, ref: dict[str, Any], what: str, *, same_subsets: bool = True) -> None:
    assert out.k == ref['k']
    ref_incl = np.array(ref['incl'], dtype=bool)
    if same_subsets:
        np.testing.assert_array_equal(out.incl, ref_incl, err_msg=f'{what}: incl')
    assert out.fit.tolist() == [bool(v) for v in as_list(ref['fit'])]
    for col in out.res.columns:
        rname = 'estimate' if col == 'estimate' else col
        close(out.res[col], ref['res'][rname], f'{what}: {col}', rtol=1e-7)


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
def gosh_fits(dat: pd.DataFrame) -> dict[str, Any]:
    d8, d9 = dat.iloc[:8], dat.iloc[:9]
    return {
        'r8': rma('yi', 'vi', data=d8),
        'r8_ee': rma('yi', 'vi', data=d8, method='EE'),
        'r8_dl': rma('yi', 'vi', data=d8, method='DL'),
        'r8_ml': rma('yi', 'vi', data=d8, method='ML'),
        'r8_w': rma('yi', 'vi', data=d8, weights='n'),
        'r9_ablat': rma('yi', 'vi', mods='~ ablat', data=d9),
        'r9_ablat_knha': rma('yi', 'vi', mods='~ ablat', data=d9, test='knha'),
    }


@pytest.mark.parametrize('key', list(GOSH_EXACT))
def test_gosh_exact_matches_metafor(gosh_fits: dict[str, Any], key: str) -> None:
    fit, ref_key = GOSH_EXACT[key]
    out = gosh(gosh_fits[fit])
    _check_gosh(out, REF[ref_key], key)
    assert same_print(str(out), REF[ref_key]['print']), str(out)


@pytest.mark.parametrize(
    ('key', 'fit', 'kw'),
    [
        ('gosh_random_int', 'r13', {'subsets': 300, 'seed': 123}),
        ('gosh_random_mods', 'r13m', {'subsets': 200, 'seed': 7}),
    ],
)
def test_gosh_random_subsets(fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch, key: str, fit: str, kw: dict[str, Any]) -> None:
    # Subset sizes and study indices are drawn with R's generator. SciPy's binomial pmf differs from R's dbinom in the
    # last bit, which reorders tied sizes inside R's sampler, so feed R's probabilities (as 17 digit strings) to compare
    # the rest of the pipeline.
    ref = REF[key]
    monkeypatch.setattr(binom, 'pmf', lambda *a, **k: np.array([float(v) for v in ref['dbinom']]))
    _check_gosh(gosh(fits[fit], **kw), ref, key)


def test_gosh_random_subsets_global_seed(fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ref = REF['gosh_random_global']
    monkeypatch.setattr(binom, 'pmf', lambda *a, **k: np.array([float(v) for v in ref['dbinom']]))
    set_seed(99)
    _check_gosh(gosh(fits['r13'], subsets=100), ref, 'global seed')


def test_gosh_random_subsets_distribution(fits: dict[str, Any]) -> None:
    out = gosh(fits['r13'], subsets=300, seed=123)
    assert out.incl.shape == (300, 13)
    assert (out.res['k'].to_numpy() == out.incl.sum(axis=1)).all()
    assert (np.diff(out.res['k'].to_numpy()) >= 0).all()
