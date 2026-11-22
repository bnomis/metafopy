"""update() of trim-and-fill results, gosh() with a seed and the standard errors of location-scale models, checked against R metafor 5.2-1
(tests/r/generate_audit3.R)."""

import math
import typing

import hypothesis
import hypothesis.strategies
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil
import rstrategies
import scipy.stats

import metafopy
import metafopy.core.dbinom
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


@refutil.linux_only
@pytest.mark.parametrize('key', sorted(GOSH_CASES))
def test_gosh_seeded(monkeypatch: pytest.MonkeyPatch, escalc_frame: pandas.DataFrame, hack: pandas.DataFrame, key: str) -> None:
    """The subsets of a seeded gosh() are those of R, which orders the tied probabilities of the subset sizes by their last bit."""
    monkeypatch.setattr(metafopy.core.dbinom, 'FUSED', False)
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


fused_modes = hypothesis.strategies.booleans()


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(1e-3, 1e3), ratio=hypothesis.strategies.floats(min_value=0.25, max_value=4.0), fused=fused_modes)
def test_bd0_property_non_negative_and_matches_naive_away_from_np(x: float, ratio: float, fused: bool) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::bd0``: non-negative, and equal to ``x log(x/np) + np - x`` when x is not close to np."""
    np = x / ratio
    value = metafopy.core.dbinom.bd0(x, np, fused)
    assert value >= -4 * math.ulp(x)
    if abs(ratio - 1.0) > 0.2:
        naive = x * math.log(x / np) + np - x
        assert value == pytest.approx(naive, rel=1e-8, abs=1e-12)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(1e-3, 1e6), fused=fused_modes)
def test_bd0_property_zero_at_x_equal_np(x: float, fused: bool) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::bd0``: ``bd0(x, x)`` is 0."""
    assert metafopy.core.dbinom.bd0(x, x, fused) == 0.0


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    n=hypothesis.strategies.integers(min_value=0, max_value=200),
    p=hypothesis.strategies.one_of(
        hypothesis.strategies.sampled_from([0.0, 1.0]),
        hypothesis.strategies.floats(min_value=1e-100, max_value=1.0, exclude_max=True),
    ),
    fused=fused_modes,
    data=hypothesis.strategies.data(),
)
def test_dbinom_raw_property_matches_scipy_and_is_a_probability(n: int, p: float, fused: bool, data: hypothesis.strategies.DataObject) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::dbinom_raw``: in [0, 1] and equal to the scipy binomial pmf.

    p starts at 1e-100: scipy's pmf overflows in ``ibeta_derivative`` for subnormal p, so the oracle gives no value there.
    """
    x = data.draw(hypothesis.strategies.integers(min_value=0, max_value=n))
    value = metafopy.core.dbinom.dbinom_raw(x, n, p, 1.0 - p, fused)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(float(scipy.stats.binom.pmf(x, n, p)), rel=1e-8, abs=1e-300)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=0, max_value=100), p=rstrategies.probabilities(open_interval=False), fused=fused_modes)
def test_dbinom_raw_property_pmf_sums_to_one(n: int, p: float, fused: bool) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::dbinom_raw``: the pmf sums to 1 over x = 0..n."""
    total = math.fsum(metafopy.core.dbinom.dbinom_raw(x, n, p, 1.0 - p, fused) for x in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-12)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-0.9, 10.0), y=hypothesis.strategies.integers(min_value=0, max_value=60), fused=fused_modes)
def test_pow1p_property_matches_direct_power(x: float, y: int, fused: bool) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::pow1p``: ``(1 + x) ** y`` for a non-negative integer ``y``."""
    assert metafopy.core.dbinom.pow1p(x, float(y), fused) == pytest.approx((1.0 + x) ** y, rel=1e-8, abs=1e-300)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=1, max_value=1000))
def test_stirlerr_property_matches_lgamma_definition_and_is_positive(n: int) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::stirlerr``: ``lgamma(n + 1) - log(sqrt(2 pi n) (n / e) ** n)``, and positive."""
    value = metafopy.core.dbinom.stirlerr(n)
    assert value > 0.0
    definition = math.lgamma(n + 1) - (0.5 * math.log(2 * math.pi * n) + n * (math.log(n) - 1.0))
    assert value == pytest.approx(definition, abs=1e-10)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=1, max_value=10**8))
def test_stirlerr_property_decreasing_in_n(n: int) -> None:
    """Candidate ``src/metafopy/core/dbinom.py::stirlerr``: the error of Stirling's formula does not increase with n.

    n starts at 1: ``stirlerr(0)`` is 0 by convention (R's ``sferr_halves[0]``), below ``stirlerr(1)``.
    """
    here = metafopy.core.dbinom.stirlerr(n)
    assert metafopy.core.dbinom.stirlerr(n + 1) <= here + 4 * math.ulp(here)
