"""update() of trim-and-fill results, gosh() with a seed and the standard errors of location-scale models, checked against R metafor 5.2-1
(tests/r/generate_audit3.R)."""

import typing

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets

matplotlib.use('Agg')

REF: dict[str, typing.Any] = refutil.load_phase12('audit3')


@pytest.fixture(scope='module')
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture helpers
def escalc_frame() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def hack() -> pandas.DataFrame:
    return metafopy.datasets.load('dat.hackshaw1998')


def fit(data: pandas.DataFrame, **kwargs: typing.Any) -> typing.Any:
    return metafopy.rma('yi', 'vi', data=data, **kwargs)


def test_update_trimfill(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['update_trimfill']
    tf = metafopy.trimfill(fit(escalc_frame))
    assert tf.k0 == ref['k0']
    tf_nofill = metafopy.trimfill(fit(escalc_frame, subset='ablat > 40'))
    cases = {
        'ml': metafopy.update(tf, method='ML'),
        'same': metafopy.update(tf),
        'knha': metafopy.update(tf, test='knha'),
        'nofill': metafopy.update(tf_nofill, method='ML'),
    }
    for key, out in cases.items():
        want = ref[key]
        assert type(out) is metafopy.RMAResult, key
        refutil.close(out.beta, want['beta'], f'{key} beta', rtol=1e-6)
        refutil.close(out.se, want['se'], f'{key} se', rtol=1e-6)
        refutil.close(out.tau2, want['tau2'], f'{key} tau2', rtol=1e-5)
        assert out.k == want['k']
        assert out.method == want['method']
        assert refutil.same_print(str(out), want['print']), key


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
GOSH_CASES = {
    'bcg13': ('escalc_frame', {}, {'subsets': 60, 'seed': 1}),
    'bcg13_seed2': ('escalc_frame', {}, {'subsets': 40, 'seed': 2}),
    'bcg13_mods': ('escalc_frame', {'mods': '~ ablat'}, {'subsets': 50, 'seed': 3}),
    'bcg13_ml': ('escalc_frame', {'method': 'ML'}, {'subsets': 30, 'seed': 4}),
    'hackshaw': ('hack', {}, {'subsets': 80, 'seed': 5}),
    'hackshaw_mods': ('hack', {'mods': '~ year'}, {'subsets': 80, 'seed': 6}),
}


@pytest.mark.parametrize('key', sorted(GOSH_CASES))
def test_gosh_seeded(escalc_frame: pandas.DataFrame, hack: pandas.DataFrame, key: str) -> None:
    """The subsets of a seeded gosh() are those of R, which orders the tied probabilities of the subset sizes by their last bit."""
    data, fit_kw, kw = GOSH_CASES[key]
    out = metafopy.gosh(fit(escalc_frame if data == 'escalc_frame' else hack, **fit_kw), **kw)
    ref = REF['gosh'][key]
    numpy.testing.assert_array_equal(out.incl.astype(int), numpy.array(ref['incl']), err_msg=key)
    for column, want in ref['res'].items():
        refutil.close(out.res['estimate' if column == 'estimate' else column], want, f'{key} {column}', rtol=1e-6)
    assert refutil.same_print(str(out), ref['print']), key


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
LS_CASES = {
    'bcg_ablat': ('escalc_frame', {'scale': '~ ablat'}),
    'bcg_ablat_loc': ('escalc_frame', {'mods': '~ ablat', 'scale': '~ ablat'}),
    'bcg_year': ('escalc_frame', {'scale': '~ I(year - 1950)'}),
    'hack_year': ('hack', {'scale': '~ year'}),
    'bcg_ml': ('escalc_frame', {'scale': '~ ablat', 'method': 'ML'}),
}


@pytest.mark.parametrize('key', sorted(LS_CASES))
def test_ls_standard_errors(escalc_frame: pandas.DataFrame, hack: pandas.DataFrame, key: str) -> None:
    """The Hessian of the location-scale likelihood is numerical in R as well, so R's own standard errors carry noise of about 1e-5."""
    data, kw = LS_CASES[key]
    res = fit(escalc_frame if data == 'escalc_frame' else hack, **kw)
    ref = REF['ls_se'][key]
    refutil.close(res.alpha, ref['alpha'], f'{key} alpha', rtol=1e-4)
    refutil.close(res.se_alpha, ref['se_alpha'], f'{key} se_alpha', rtol=1e-4)
    refutil.close(res.beta, ref['beta'], f'{key} beta', rtol=1e-5)
    refutil.close(res.se, ref['se'], f'{key} se', rtol=1e-4)


def test_ls_hessian_r_controls_precision(escalc_frame: pandas.DataFrame) -> None:
    """With the Richardson depth of metafor's numDeriv call (r = 8) the standard errors are noisy, as they are in R."""
    res = fit(escalc_frame, scale='~ ablat', control={'hessian_r': 8})
    refutil.close(res.se_alpha, REF['ls_se']['bcg_ablat_r8']['se_alpha'], 'se_alpha r=8', rtol=1e-2)


def test_confint_print(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['small']
    r0, r1 = fit(escalc_frame), fit(escalc_frame, mods='~ ablat')
    assert refutil.r_print(str(metafopy.confint(r0))).strip() == refutil.r_print(ref['confint_random']).strip()
    assert refutil.r_print(str(metafopy.confint(r1, fixed=True))).strip() == refutil.r_print(ref['confint_fixed']).strip()
    assert refutil.r_print(str(metafopy.confint(r1, fixed=True, digits=2))).strip() == refutil.r_print(ref['confint_digits']).strip()


def test_confint_control(escalc_frame: pandas.DataFrame) -> None:
    r0 = fit(escalc_frame)
    out = metafopy.confint(r0, control={'tau2.max': 50, 'tol': 1e-8})
    refutil.close(out.random.to_numpy().reshape(-1), numpy.asarray(REF['small']['confint_control']).reshape(-1).tolist(), 'control', rtol=1e-4)
    keyword = metafopy.confint(r0, tau2_max=50, tol=1e-8)
    refutil.close(out.random.to_numpy().reshape(-1), keyword.random.to_numpy().reshape(-1).tolist(), 'keywords', rtol=1e-12)
    with pytest.raises(metafopy.errors.InputError, match='control'):
        metafopy.confint(r0, control={'bogus': 1})


def test_anova_lrt_print(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['small']
    ml0, ml1 = fit(escalc_frame, method='ML'), fit(escalc_frame, mods='~ ablat', method='ML')
    assert refutil.r_print(str(metafopy.anova(ml0, ml1))).strip() == refutil.r_print(ref['anova_lrt_ml']).strip()
    with pytest.warns(UserWarning, match='REML comparisons'):
        reml = metafopy.anova(fit(escalc_frame), fit(escalc_frame, mods='~ ablat'))
    assert refutil.r_print(str(reml)).strip() == refutil.r_print(ref['anova_lrt']).strip()


def test_result_fields(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['small']['fields']
    r0, r1 = fit(escalc_frame), fit(escalc_frame, mods='~ ablat')
    assert (r0.k_eff, r1.p_eff, r0.k_all) == (ref['k_eff'], ref['p_eff'], ref['k_all'])
    refutil.close(r0.ni, ref['ni'], 'ni')
    assert r0.formula_yi is None
    assert r1.formula_mods is not None
    assert r1.formula_mods.replace(' ', '') == ref['formula_mods']
    assert metafopy.rma('yi', 'vi', scale='~ ablat', data=escalc_frame).k_all == ref['k_all']
    assert typing.cast(typing.Any, metafopy.selmodel(r0, type='halfnorm')).k_eff == ref['k_eff']


def test_robust_meat_and_predict_aliases(escalc_frame: pandas.DataFrame) -> None:
    kon = metafopy.datasets.load('konstantopoulos2011')
    uni1 = metafopy.rma('yi', 'vi', mods='~ I(year - 1990)', data=kon)
    rb = metafopy.robust(uni1, 'district')
    assert rb.robust is not None
    assert rb.robust.meat is not None
    refutil.close(rb.robust.meat.reshape(-1), numpy.asarray(REF['small']['meat']).reshape(-1).tolist(), 'meat', rtol=1e-8)
    assert metafopy.robust(uni1, 'district', club_sandwich=True).robust.meat is None
    pr = fit(escalc_frame, mods='~ ablat').predict(newmods=[30])
    refutil.close(pr.cr_lb, REF['small']['cr']['lb'], 'cr_lb')
    refutil.close(pr.cr_ub, REF['small']['cr']['ub'], 'cr_ub')


def test_addpoly_explicit_rows_of_prediction(escalc_frame: pandas.DataFrame) -> None:
    """``rows=-1`` is an explicit row now, not the default that moves to ``row``."""
    pr = fit(escalc_frame).predict()
    fig, ax = matplotlib.pyplot.subplots()
    metafopy.forest(fit(escalc_frame), ax=ax)
    metafopy.addpoly(pr, ax=ax)
    first = ax.texts[-1].get_position()[1]
    metafopy.addpoly(pr, rows=-1, ax=ax)
    assert ax.texts[-1].get_position()[1] == -1 != first
    matplotlib.pyplot.close(fig)
