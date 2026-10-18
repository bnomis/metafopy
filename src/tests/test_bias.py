"""regtest(), ranktest(), trimfill(), fsn(), hc() and tes() checked against R metafor (tests/r/generate_bias.R)."""

import collections.abc
import math
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.bias.fsn
import metafopy.bias.hc
import metafopy.bias.ranktest
import metafopy.bias.regtest
import metafopy.bias.tes
import metafopy.bias.trimfill
import metafopy.core.rng
import metafopy.datasets
import metafopy.transf

REF = refutil.load_phase2('bias')


def same_text(actual: str, expected: str) -> None:
    assert refutil.r_print(actual).strip() == refutil.r_print(expected).strip()


def hackshaw() -> pandas.DataFrame:
    return metafopy.datasets.load('hackshaw1998')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def hack() -> pandas.DataFrame:
    return hackshaw()


@pytest.fixture(scope='module')
def sim() -> pandas.DataFrame:
    return pandas.DataFrame({'yi': REF['sim']['yi'], 'vi': REF['sim']['vi']})


@pytest.fixture(scope='module')
def models(dat: pandas.DataFrame, hack: pandas.DataFrame, sim: pandas.DataFrame) -> dict[str, typing.Any]:
    dna = dat.copy()
    dna.loc[2, 'yi'] = numpy.nan
    return {
        'res': metafopy.rma(dat.yi, dat.vi),
        'mods': metafopy.rma(dat.yi, dat.vi, mods=dat.ablat),
        'knha': metafopy.rma(dat.yi, dat.vi, test='knha'),
        'ee': metafopy.rma(dat.yi, dat.vi, method='EE'),
        'ml': metafopy.rma(dat.yi, dat.vi, method='ML'),
        'dl': metafopy.rma(dat.yi, dat.vi, method='DL'),
        'hack': metafopy.rma(hack.yi, hack.vi),
        'hack_ee': metafopy.rma(hack.yi, hack.vi, method='EE'),
        'hack_dl': metafopy.rma(hack.yi, hack.vi, method='DL'),
        'hack_knha': metafopy.rma(hack.yi, hack.vi, test='knha'),
        'sim': metafopy.rma(sim.yi, sim.vi),
        'sim_ee': metafopy.rma(sim.yi, sim.vi),
        'na': metafopy.rma(dna.yi, dna.vi),
        'lvl90': metafopy.rma(dat.yi, dat.vi, level=90),
    }


# -- regtest -------------------------------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
REG_CASES: dict[str, collections.abc.Callable[[dict[str, typing.Any], pandas.DataFrame, pandas.DataFrame], typing.Any]] = {
    'reg_sei': lambda m, d, h: metafopy.bias.regtest(m['res']),
    'reg_sei_fit': lambda m, d, h: metafopy.bias.regtest(m['res'], ret_fit=True),
    'reg_vi': lambda m, d, h: metafopy.bias.regtest(m['res'], predictor='vi'),
    'reg_lm': lambda m, d, h: metafopy.bias.regtest(m['res'], model='lm'),
    'reg_lm_vi': lambda m, d, h: metafopy.bias.regtest(m['res'], model='lm', predictor='vi'),
    'reg_knha': lambda m, d, h: metafopy.bias.regtest(m['knha']),
    'reg_ninv': lambda m, d, h: metafopy.bias.regtest(m['res'], predictor='ninv', ni=d.n),
    'reg_ni': lambda m, d, h: metafopy.bias.regtest(m['res'], predictor='ni', ni=d.n),
    'reg_sqrtni': lambda m, d, h: metafopy.bias.regtest(m['res'], predictor='sqrtni', ni=d.n),
    'reg_sqrtninv': lambda m, d, h: metafopy.bias.regtest(m['res'], predictor='sqrtninv', ni=d.n),
    'reg_mods': lambda m, d, h: metafopy.bias.regtest(m['mods']),
    'reg_ee': lambda m, d, h: metafopy.bias.regtest(m['ee']),
    'reg_ml': lambda m, d, h: metafopy.bias.regtest(m['ml']),
    'reg_level': lambda m, d, h: metafopy.bias.regtest(m['res'], level=90),
    'reg_hack': lambda m, d, h: metafopy.bias.regtest(m['hack']),
    'reg_hack_lm': lambda m, d, h: metafopy.bias.regtest(m['hack'], model='lm'),
    'reg_vec': lambda m, d, h: metafopy.bias.regtest(d.yi, d.vi),
    'reg_vec_sei': lambda m, d, h: metafopy.bias.regtest(d.yi, sei=numpy.sqrt(d.vi), model='lm'),
    'reg_vec_ni': lambda m, d, h: metafopy.bias.regtest(d.yi, d.vi, ni=d.n, predictor='ninv', method='DL'),
    'reg_vec_na': lambda m, d, h: metafopy.bias.regtest(d.yi.where(d.index != 3), d.vi),
    'reg_na_model': lambda m, d, h: metafopy.bias.regtest(m['na']),
}


@pytest.mark.parametrize('key', REG_CASES)
def test_regtest(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    ref = REF[key]
    if key == 'reg_vec_na':
        with pytest.warns(UserWarning, match='NAs omitted'):
            out = REG_CASES[key](models, dat, hack)
    else:
        out = REG_CASES[key](models, dat, hack)
    refutil.close(out.zval, ref['zval'], f'{key}: zval')
    refutil.close(out.pval, ref['pval'], f'{key}: pval')
    if ref['ddf'] is None:
        assert out.ddf is None
    else:
        assert out.ddf == ref['ddf']
    if ref['est'] is None:
        assert out.est is None
    else:
        refutil.close(out.est, ref['est'], f'{key}: est')
        refutil.close(out.ci_lb, ref['ci.lb'], f'{key}: ci.lb')
        refutil.close(out.ci_ub, ref['ci.ub'], f'{key}: ci.ub')
    same_text(str(out), ref['print'])


def test_regtest_errors(models: dict[str, typing.Any], dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='sample size'):
        metafopy.bias.regtest(models['res'], predictor='ninv')
    with pytest.raises(ValueError, match='vi'):
        metafopy.bias.regtest(dat.yi)
    with pytest.raises(ValueError, match='model'):
        metafopy.bias.regtest(models['res'], model='glm')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='predictor'):
        metafopy.bias.regtest(models['res'], predictor='foo')  # type: ignore[arg-type]


# -- ranktest ------------------------------------------------------------------------------------------------------------------------------------

TIES = ([0.1, 0.4, 0.4, 0.8, 0.2, 0.9, 0.5, 0.3], [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5])

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
RANK_CASES: dict[str, collections.abc.Callable[[dict[str, typing.Any], pandas.DataFrame], typing.Any]] = {
    'rank_re': lambda m, d: metafopy.bias.ranktest(m['res']),
    'rank_hack': lambda m, d: metafopy.bias.ranktest(m['hack']),
    'rank_sim': lambda m, d: metafopy.bias.ranktest(m['sim']),
    'rank_vec': lambda m, d: metafopy.bias.ranktest(d.yi, d.vi),
    'rank_noexact': lambda m, d: metafopy.bias.ranktest(m['res'], exact=False),
    'rank_ties': lambda m, d: metafopy.bias.ranktest(TIES[0], TIES[1]),
}


@pytest.mark.parametrize('key', RANK_CASES)
def test_ranktest(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame) -> None:
    ref = REF[key]
    out = RANK_CASES[key](models, dat)
    refutil.close(out.tau, ref['tau'], f'{key}: tau')
    refutil.close(out.pval, ref['pval'], f'{key}: pval')
    same_text(str(out), ref['print'])


def test_ranktest_needs_intercept_only_model(models: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError, match='moderators'):
        metafopy.bias.ranktest(models['mods'])


# -- trimfill ------------------------------------------------------------------------------------------------------------------------------------


def tf(name: str, **kw: typing.Any) -> collections.abc.Callable[[dict[str, typing.Any], pandas.DataFrame], typing.Any]:
    return lambda m, d: metafopy.bias.trimfill(m[name], **kw)


TF_CASES: dict[str, collections.abc.Callable[[dict[str, typing.Any], pandas.DataFrame], typing.Any]] = {
    **{f'tf_{name}_{est}': tf(name if name != 're' else 'res', estimator=est) for name in ('re', 'hack', 'sim') for est in ('L0', 'R0', 'Q0')},
    'tf_left': lambda m, d: metafopy.bias.trimfill(m['hack'], side='left'),
    'tf_right': lambda m, d: metafopy.bias.trimfill(m['hack'], side='right'),
    'tf_right_sim': lambda m, d: metafopy.bias.trimfill(m['sim'], side='right', estimator='R0'),
    'tf_ee': lambda m, d: metafopy.bias.trimfill(m['hack_ee']),
    'tf_dl': lambda m, d: metafopy.bias.trimfill(m['hack_dl']),
    'tf_knha': lambda m, d: metafopy.bias.trimfill(m['hack_knha']),
    'tf_ilim': lambda m, d: metafopy.bias.trimfill(m['hack'], ilim=(-1, 3)),
    'tf_slab': lambda m, d: metafopy.bias.trimfill(metafopy.rma(d.yi, d.vi, slab=(d.author + ' ' + d.year.astype(str)).tolist()), side='right'),
    'tf_level': lambda m, d: metafopy.bias.trimfill(m['hack'], level=90),
}
TF_CASES['tf_weights'] = lambda m, d: metafopy.bias.trimfill(metafopy.rma(hackshaw().yi, hackshaw().vi, weights=1 / numpy.sqrt(hackshaw().vi)))


@pytest.mark.parametrize('key', TF_CASES)
def test_trimfill(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame) -> None:
    ref = REF[key]
    out = TF_CASES[key](models, dat)
    assert out.k0 == ref['k0']
    assert out.side == ref['side']
    refutil.close(out.se_k0, ref['se.k0'], f'{key}: se.k0')
    refutil.close(out.p_k0, ref['p.k0'], f'{key}: p.k0')
    assert [int(f) for f in out.fill] == refutil.as_list(ref['fill'])
    assert out.k == ref['k']
    refutil.close(out.beta, ref['beta'], f'{key}: beta')
    refutil.close(out.se, ref['se'], f'{key}: se')
    refutil.close(out.tau2, ref['tau2'], f'{key}: tau2')
    refutil.close(out.yi, ref['yi'], f'{key}: yi')
    refutil.close(out.vi, ref['vi'], f'{key}: vi')
    refutil.close(out.QE, ref['QE'], f'{key}: QE')
    if out.k0 > 0:
        assert list(out.slab) == refutil.as_list(ref['slab'])
    same_text(str(out), ref['print'])


def test_trimfill_needs_intercept_only_model(models: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError, match='moderators'):
        metafopy.bias.trimfill(models['mods'])


# -- fsn -----------------------------------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FSN_CASES: dict[str, collections.abc.Callable[..., typing.Any]] = {
    'fsn_rosenthal': lambda m, d, h: metafopy.bias.fsn(d.yi, d.vi),
    'fsn_rosenthal_hack': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi),
    'fsn_fisher': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, pool='fisher'),
    'fsn_rosenthal_alpha': lambda m, d, h: metafopy.bias.fsn(d.yi, d.vi, alpha=0.01),
    'fsn_binomial': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, type='Binomial'),
    'fsn_orwin': lambda m, d, h: metafopy.bias.fsn(d.yi, d.vi, type='Orwin'),
    'fsn_orwin_target': lambda m, d, h: metafopy.bias.fsn(d.yi, d.vi, type='Orwin', target=-0.1),
    'fsn_orwin_unw': lambda m, d, h: metafopy.bias.fsn(d.yi, type='Orwin', weighted=False),
    'fsn_orwin_unw_target': lambda m, d, h: metafopy.bias.fsn(d.yi, type='Orwin', weighted=False, target=-0.2),
    'fsn_rosenberg': lambda m, d, h: metafopy.bias.fsn(d.yi, d.vi, type='Rosenberg'),
    'fsn_general': lambda m, d, h: metafopy.bias.fsn(m['res']),
    'fsn_general_hack': lambda m, d, h: metafopy.bias.fsn(m['hack']),
    'fsn_general_ee': lambda m, d, h: metafopy.bias.fsn(m['ee']),
    'fsn_general_dl': lambda m, d, h: metafopy.bias.fsn(m['hack_dl']),
    'fsn_general_target': lambda m, d, h: metafopy.bias.fsn(m['res'], target=-0.3),
    'fsn_general_target_pos': lambda m, d, h: metafopy.bias.fsn(m['sim'], target=0.1),
    'fsn_general_vec': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, type='General'),
    'fsn_general_vec_method': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, type='General', method='DL'),
    'fsn_general_knha': lambda m, d, h: metafopy.bias.fsn(m['hack_knha']),
    'fsn_general_weighted': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, type='General', weighted=False),
    'fsn_general_tau2': lambda m, d, h: metafopy.bias.fsn(h.yi, h.vi, type='General', tau2=0.05),
    'fsn_general_mumiss': lambda m, d, h: metafopy.bias.fsn(m['hack'], mumiss=0.1),
    'fsn_general_interval': lambda m, d, h: metafopy.bias.fsn(m['hack'], interval=(0, 2000)),
    'fsn_general_exact': lambda m, d, h: metafopy.bias.fsn(m['res'], exact=True, seed=42),
    'fsn_general_exact_target': lambda m, d, h: metafopy.bias.fsn(m['res'], exact=True, target=-0.3, seed=42),
    'fsn_general_exact_hack': lambda m, d, h: metafopy.bias.fsn(m['hack'], exact=True, seed=7),
    'fsn_general_maxint': lambda m, d, h: metafopy.bias.fsn(m['hack'], target=0.001, maxint=1000),
}


@pytest.mark.parametrize('key', FSN_CASES)
def test_fsn(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    ref = REF[key]
    if 'error' in ref:
        with pytest.raises(ValueError, match='fail-safe N'):
            FSN_CASES[key](models, dat, hack)
        return
    out = FSN_CASES[key](models, dat, hack)
    assert out.type == ref['type']
    refutil.close(out.fsnum, ref['fsnum'], f'{key}: fsnum', rtol=0, atol=0)
    refutil.close(out.est, ref['est'], f'{key}: est')
    refutil.close(out.tau2, ref['tau2'], f'{key}: tau2')
    refutil.close(out.pval, ref['pval'], f'{key}: pval')
    refutil.close(out.target, ref['target'], f'{key}: target')
    refutil.close(out.est_fsn, ref['est.fsn'], f'{key}: est.fsn', rtol=1e-6)
    refutil.close(out.tau2_fsn, ref['tau2.fsn'], f'{key}: tau2.fsn', rtol=1e-6)
    refutil.close(out.pval_fsn, ref['pval.fsn'], f'{key}: pval.fsn', rtol=1e-6)
    assert out.ub_sign == ref['ub.sign']
    same_text(str(out), ref['print'])


def test_fsn_errors(models: dict[str, typing.Any], dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='moderators'):
        metafopy.bias.fsn(models['mods'])
    with pytest.raises(ValueError, match='vi'):
        metafopy.bias.fsn(dat.yi)
    with pytest.raises(ValueError, match='type'):
        metafopy.bias.fsn(dat.yi, dat.vi, type='Nonsense')  # type: ignore[arg-type]


def test_fsn_exact_uses_global_seed(models: dict[str, typing.Any]) -> None:
    metafopy.core.rng.set_seed(42)
    a = metafopy.bias.fsn(models['res'], exact=True)
    b = metafopy.bias.fsn(models['res'], exact=True, seed=42)
    assert a.fsnum == b.fsnum
    assert a.est_fsn == b.est_fsn


# -- hc ------------------------------------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
HC_CASES: dict[str, collections.abc.Callable[[dict[str, typing.Any]], typing.Any]] = {
    'hc_re': lambda m: metafopy.bias.hc(m['res']),
    'hc_dl': lambda m: metafopy.bias.hc(m['dl']),
    'hc_hack': lambda m: metafopy.bias.hc(m['hack']),
    'hc_ml': lambda m: metafopy.bias.hc(m['ml']),
    'hc_level': lambda m: metafopy.bias.hc(m['lvl90']),
    'hc_exp': lambda m: metafopy.bias.hc(m['res'], transf=math.exp),
    'hc_ee': lambda m: metafopy.bias.hc(m['ee']),
    'hc_sim': lambda m: metafopy.bias.hc(m['sim']),
    'hc_intexp': lambda m: metafopy.bias.hc(m['res'], transf=metafopy.transf.exp_int),
}


@pytest.mark.parametrize('key', HC_CASES)
def test_hc(key: str, models: dict[str, typing.Any]) -> None:
    ref = REF[key]
    out = HC_CASES[key](models)
    for name in ('beta', 'se', 'ci.lb', 'ci.ub', 'beta.rma', 'se.rma', 'ci.lb.rma', 'ci.ub.rma', 'tau2', 'tau2.rma'):
        refutil.close(getattr(out, name.replace('.', '_')), ref[name], f'{key}: {name}', rtol=1e-6)
    same_text(str(out), ref['print'])


def test_hc_needs_intercept_only_model(models: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError, match='moderators'):
        metafopy.bias.hc(models['mods'])


# -- tes -----------------------------------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
TES_CASES: dict[str, collections.abc.Callable[[dict[str, typing.Any], pandas.DataFrame, pandas.DataFrame], typing.Any]] = {
    'tes_re': lambda m, d, h: metafopy.bias.tes(m['res']),
    'tes_re_chi2': lambda m, d, h: metafopy.bias.tes(m['res'], test='chi2'),
    'tes_re_binom': lambda m, d, h: metafopy.bias.tes(m['res'], test='binom'),
    'tes_ee': lambda m, d, h: metafopy.bias.tes(m['ee']),
    'tes_hack': lambda m, d, h: metafopy.bias.tes(m['hack']),
    'tes_sim': lambda m, d, h: metafopy.bias.tes(m['sim']),
    'tes_mods': lambda m, d, h: metafopy.bias.tes(m['mods']),
    'tes_vec': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi),
    'tes_vec_chi2': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, test='chi2'),
    'tes_vec_binom': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, test='binom'),
    'tes_vec_correct': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, test='chi2', correct=True),
    'tes_theta': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, theta=-0.5),
    'tes_theta_vec': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, theta=numpy.linspace(-1, 0.1, 13)),
    'tes_tau2': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, theta=-0.5, tau2=0.3),
    'tes_tau2_chi2': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, theta=-0.5, tau2=0.3, test='chi2'),
    'tes_greater': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, alternative='greater', theta=0.2),
    'tes_less': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, alternative='less'),
    'tes_less_tes': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, tes_alternative='less'),
    'tes_two': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, tes_alternative='two.sided'),
    'tes_two_chi2': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, tes_alternative='two.sided', test='chi2'),
    'tes_alpha': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, alpha=0.1, tes_alpha=0.05),
    'tes_h0': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, H0=-0.3),
    'tes_small': lambda m, d, h: metafopy.bias.tes(d.yi[:5], d.vi[:5]),
    'tes_nolim': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, find_lim=False),
    'tes_sei': lambda m, d, h: metafopy.bias.tes(d.yi, sei=numpy.sqrt(d.vi)),
    'tes_big_chi2': lambda m, d, h: metafopy.bias.tes(h.yi, h.vi),
    'tes_two_lim': lambda m, d, h: metafopy.bias.tes(d.yi, d.vi, tes_alternative='two.sided', theta=-0.7),
}


@pytest.mark.parametrize('key', TES_CASES)
def test_tes(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    ref = REF[key]
    out = TES_CASES[key](models, dat, hack)
    assert out.k == ref['k']
    assert ref['O'] == out.O
    assert out.test == ref['test']
    refutil.close(out.E, ref['E'], f'{key}: E', rtol=1e-6)
    refutil.close(out.OEratio, ref['OEratio'], f'{key}: OEratio', rtol=1e-6)
    refutil.close(out.X2, ref['X2'], f'{key}: X2', rtol=1e-6)
    refutil.close(out.pval, ref['pval'], f'{key}: pval', rtol=1e-6)
    refutil.close(out.power, ref['power'], f'{key}: power', rtol=1e-6)
    assert [int(s) for s in out.sig] == refutil.as_list(ref['sig'])
    refutil.close(out.theta, ref['theta'], f'{key}: theta')
    if ref['theta.lim'] == 'none':
        assert out.theta_lim is None
    else:
        refutil.close(out.theta_lim, ref['theta.lim'], f'{key}: theta.lim', rtol=1e-5)
    same_text(str(out), ref['print'])


def test_tes_errors(dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='vi'):
        metafopy.bias.tes(dat.yi)
    with pytest.raises(ValueError, match='alpha'):
        metafopy.bias.tes(dat.yi, dat.vi, alpha=2)
