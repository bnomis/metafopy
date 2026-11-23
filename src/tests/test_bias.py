"""regtest(), ranktest(), trimfill(), fsn(), hc() and tes() checked against R metafor (tests/r/generate_bias.R)."""

import collections.abc
import importlib
import itertools
import math
import typing

import hypothesis
import hypothesis.strategies
import numpy
import pandas
import pytest
import refutil
import rstrategies

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
AUDIT3 = refutil.load_phase12('audit3')


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def same_text(actual: str, expected: str) -> None:
    assert refutil.r_print(actual).strip() == refutil.r_print(expected).strip()


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
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


# print(regtest(rma(yi, vi, data = dat)), digits = 2) in R 4.6 with metafor 5.2-1 (dat: escalc("RR") of dat.bcg), run on 2026-11-23
REG_DIGITS_2 = (
    '\nRegression Test for Funnel Plot Asymmetry\n\nModel:     mixed-effects meta-regression model\nPredictor: standard error\n\n'
    'Test for Funnel Plot Asymmetry: z = -0.80, p = 0.42\nLimit Estimate (as sei -> 0):   b = -0.51 (CI: -1.12, 0.10)\n'
)


def test_regtest_summary(models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    """Direct test of src/metafopy/bias/regtest.py::RegtestResult.summary: R's print() with the stored digits and with digits = 2."""
    out = REG_CASES['reg_sei'](models, dat, hack)
    same_text(out.summary(), REF['reg_sei']['print'])
    same_text(out.summary(digits=2), REG_DIGITS_2)


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


def test_trimfill_summary(models: dict[str, typing.Any], dat: pandas.DataFrame) -> None:
    """Direct test of src/metafopy/bias/trimfill.py::TrimfillResult.summary: R's summary() (with the fit statistics, showfit=True) and print()
    (showfit=False) of trimfill(rma(yi, vi, data = dat)), from tests/r/generate_audit3.R and tests/r/generate_bias.R."""
    out = TF_CASES['tf_re_L0'](models, dat)
    shown = out.summary(showfit=True)
    assert 'logLik' in shown
    assert refutil.same_print(shown, AUDIT3['summary']['trimfill']), shown
    same_text(out.summary(showfit=False), REF['tf_re_L0']['print'])


def test_trimfill_needs_intercept_only_model(models: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError, match='moderators'):
        metafopy.bias.trimfill(models['mods'])


# -- fsn -----------------------------------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FSN_CASES: dict[str, collections.abc.Callable[..., typing.Any]] = {
    'fsn_rosenthal': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, d.vi, **kw),
    'fsn_rosenthal_hack': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, **kw),
    'fsn_fisher': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, pool='fisher', **kw),
    'fsn_rosenthal_alpha': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, d.vi, alpha=0.01, **kw),
    'fsn_binomial': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, type='Binomial', **kw),
    'fsn_orwin': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, d.vi, type='Orwin', **kw),
    'fsn_orwin_target': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, d.vi, type='Orwin', target=-0.1, **kw),
    'fsn_orwin_unw': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, type='Orwin', weighted=False, **kw),
    'fsn_orwin_unw_target': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, type='Orwin', weighted=False, target=-0.2, **kw),
    'fsn_rosenberg': lambda m, d, h, **kw: metafopy.bias.fsn(d.yi, d.vi, type='Rosenberg', **kw),
    'fsn_general': lambda m, d, h, **kw: metafopy.bias.fsn(m['res'], **kw),
    'fsn_general_hack': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack'], **kw),
    'fsn_general_ee': lambda m, d, h, **kw: metafopy.bias.fsn(m['ee'], **kw),
    'fsn_general_dl': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack_dl'], **kw),
    'fsn_general_target': lambda m, d, h, **kw: metafopy.bias.fsn(m['res'], target=-0.3, **kw),
    'fsn_general_target_pos': lambda m, d, h, **kw: metafopy.bias.fsn(m['sim'], target=0.1, **kw),
    'fsn_general_vec': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, type='General', **kw),
    'fsn_general_vec_method': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, type='General', method='DL', **kw),
    'fsn_general_knha': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack_knha'], **kw),
    'fsn_general_weighted': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, type='General', weighted=False, **kw),
    'fsn_general_tau2': lambda m, d, h, **kw: metafopy.bias.fsn(h.yi, h.vi, type='General', tau2=0.05, **kw),
    'fsn_general_mumiss': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack'], mumiss=0.1, **kw),
    'fsn_general_interval': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack'], interval=(0, 2000), **kw),
    'fsn_general_exact': lambda m, d, h, **kw: metafopy.bias.fsn(m['res'], exact=True, seed=42, **kw),
    'fsn_general_exact_target': lambda m, d, h, **kw: metafopy.bias.fsn(m['res'], exact=True, target=-0.3, seed=42, **kw),
    'fsn_general_exact_hack': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack'], exact=True, seed=7, **kw),
    'fsn_general_maxint': lambda m, d, h, **kw: metafopy.bias.fsn(m['hack'], target=0.001, maxint=1000, **kw),
}


# Keys whose fail-safe N is one below R's (a step function of N that R's root search rounds up), with metafopy's N.
FSN_DIFFERS: dict[str, int] = {
    'fsn_general': 25,
    'fsn_general_dl': 120,
    'fsn_general_exact': 26,
    'fsn_general_knha': 114,
    'fsn_general_target_pos': 203,
    'fsn_general_vec_method': 120,
    'fsn_general_weighted': 302,
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
    refutil.close(out.est, ref['est'], f'{key}: est')
    refutil.close(out.tau2, ref['tau2'], f'{key}: tau2')
    refutil.close(out.pval, ref['pval'], f'{key}: pval')
    refutil.close(out.target, ref['target'], f'{key}: target')
    assert out.ub_sign == ref['ub.sign']
    if key in FSN_DIFFERS:
        # metafopy returns the least integer N whose criterion holds, R's uniroot one more: docs/differences.md, "fsn(): Fisher pooling and the
        # General type", and the correctness cases bias.fsn.egger2001.rosenthal.fisher and bias.fsn.egger2001.general.target.
        assert out.fsnum == FSN_DIFFERS[key]
        assert ref['fsnum'] != out.fsnum
        below = FSN_CASES[key](models, dat, hack, maxint=out.fsnum - 1)
        assert below.ub_sign == '>'
        if math.isnan(out.target):
            assert out.pval_fsn >= out.alpha > below.pval_fsn
        else:
            assert abs(out.est_fsn) <= abs(out.target) < abs(below.est_fsn)
        return
    refutil.close(out.fsnum, ref['fsnum'], f'{key}: fsnum', rtol=0, atol=0)
    refutil.close(out.est_fsn, ref['est.fsn'], f'{key}: est.fsn', rtol=1e-6)
    refutil.close(out.tau2_fsn, ref['tau2.fsn'], f'{key}: tau2.fsn', rtol=1e-6)
    refutil.close(out.pval_fsn, ref['pval.fsn'], f'{key}: pval.fsn', rtol=1e-6)
    same_text(str(out), ref['print'])


# print(fsn(yi, vi, data = dat), digits = 2) in R 4.6 with metafor 5.2-1 (dat: escalc("RR") of dat.bcg), run on 2026-11-23
FSN_DIGITS_2 = (
    '\nFail-safe N Calculation Using the Rosenthal Approach\n\nObserved Significance Level: <.01\nTarget Significance Level:   0.05\n\nFail-safe N: 598\n'
)


def test_fsn_summary(models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    """Direct test of src/metafopy/bias/fsn.py::FsnResult.summary: R's print() with the stored digits and with digits = 2."""
    out = FSN_CASES['fsn_rosenthal'](models, dat, hack)
    same_text(out.summary(), REF['fsn_rosenthal']['print'])
    same_text(out.summary(digits=2), FSN_DIGITS_2)


def test_fsn_regression_integer_search() -> None:
    data = metafopy.datasets.load('egger2001').drop(index=[7, 15])
    esc = metafopy.escalc(measure='RR', ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=data, add=0.5, to='all')
    yi, vi = numpy.asarray(esc['yi'], dtype=float), numpy.asarray(esc['vi'], dtype=float)
    assert metafopy.bias.fsn(yi, vi, pool='fisher').fsnum == 48
    target = math.log(0.95)
    default = metafopy.bias.fsn(yi, vi, type='General', method='REML', exact=True, target=target, seed=1)
    assert default.fsnum == 116
    assert metafopy.bias.fsn(yi, vi, type='General', method='REML', exact=True, target=target, seed=1, tol=1e-10).fsnum == 116
    other = metafopy.bias.fsn(yi, vi, type='General', method='REML', exact=True, target=target, seed=2)
    for name in ('fsnum', 'est_fsn', 'tau2_fsn', 'pval_fsn'):
        refutil.close(getattr(other, name), getattr(default, name), f'seed: {name}', rtol=1e-9)


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


# print(hc(rma(yi, vi, data = dat)), digits = 2) in R 4.6 with metafor 5.2-1 (dat: escalc("RR") of dat.bcg), run on 2026-11-23
HC_DIGITS_2 = '\n    method tau2 estimate   se ci.lb ci.ub \nrma   REML 0.31    -0.71 0.18 -1.07 -0.36 \nhc      DL 0.31    -0.43 0.28 -1.16  0.29 \n'


def test_hc_summary(models: dict[str, typing.Any]) -> None:
    """Direct test of src/metafopy/bias/hc.py::HcResult.summary: R's print() with the stored digits and with digits = 2."""
    out = HC_CASES['hc_re'](models)
    same_text(out.summary(), REF['hc_re']['print'])
    same_text(out.summary(digits=2), HC_DIGITS_2)


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


@pytest.mark.parametrize('key', ['tes_re', 'tes_theta_vec', 'tes_two_lim'])
def test_tes_summary(key: str, models: dict[str, typing.Any], dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    """Direct test of src/metafopy/bias/tes.py::TesResult.summary: R's print() of the result."""
    same_text(TES_CASES[key](models, dat, hack).summary(), REF[key]['print'])


def test_tes_errors(dat: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='vi'):
        metafopy.bias.tes(dat.yi)
    with pytest.raises(ValueError, match='alpha'):
        metafopy.bias.tes(dat.yi, dat.vi, alpha=2)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e9, 1e9))
def test_rnd_fsn_property_whole_number_just_above_or_at_x(x: float) -> None:
    """Candidate ``src/metafopy/bias/fsn.py::rnd_fsn``: a whole number with ``x - sqrt(EPS) <= result < x + 1``."""
    result = importlib.import_module('metafopy.bias.fsn').rnd_fsn(x)
    assert result == math.floor(result)
    assert x - importlib.import_module('metafopy.bias.fsn').EPS ** 0.5 <= result < x + 1


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=hypothesis.strategies.sampled_from([math.inf, -math.inf, math.nan]))
def test_rnd_fsn_property_non_finite_unchanged(x: float) -> None:
    """Candidate ``src/metafopy/bias/fsn.py::rnd_fsn``: a non-finite ``x`` is returned unchanged."""
    result = importlib.import_module('metafopy.bias.fsn').rnd_fsn(x)
    assert (math.isnan(result) and math.isnan(x)) or result == x


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=1, max_value=7), data=hypothesis.strategies.data())
def test_pkendall_property_matches_brute_force_and_is_a_distribution(n: int, data: hypothesis.strategies.DataObject) -> None:
    """Candidate ``src/metafopy/bias/ranktest.py::pkendall``: the permutation count, non-decreasing, 1 at the maximum and symmetric."""
    top = n * (n - 1) // 2
    q = data.draw(hypothesis.strategies.integers(min_value=-1, max_value=top + 1))
    inversions = [sum(a > b for i, a in enumerate(p) for b in p[i + 1 :]) for p in itertools.permutations(range(n))]
    value = importlib.import_module('metafopy.bias.ranktest').pkendall(q, n)
    assert value == pytest.approx(sum(v <= q for v in inversions) / len(inversions), abs=1e-12)
    assert importlib.import_module('metafopy.bias.ranktest').pkendall(q + 1, n) >= value
    assert importlib.import_module('metafopy.bias.ranktest').pkendall(top, n) == pytest.approx(1.0, abs=1e-12)
    if 0 <= q < top:
        assert value + importlib.import_module('metafopy.bias.ranktest').pkendall(top - q - 1, n) == pytest.approx(1.0, abs=1e-12)
