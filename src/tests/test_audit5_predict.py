"""predict() with named newmods and an evaluated prob cutoff, the checks of anova() for likelihood-ratio tests, accessors (coef, vcov, dfs, rma_mv group
fields) and the undocumented arguments of anova, influence, profile and hettest that metafor reads through ``...`` (audit of round five).

The 0.2016 of ``prob='>log(0.8)'`` is the value of the audit (metafor 5.2-1, bcg with the random-effects model of log risk ratios); the other tests check
the behaviour and the agreement between spellings of the same call.
"""

import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.permutest
import metafopy.methods.predict

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def two_mods(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma(yi='yi', vi='vi', mods='~ ablat + year', data=dat)


@pytest.fixture(scope='module')
def re_model(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma(yi='yi', vi='vi', data=dat)


def pred_of(res: metafopy.RMAResult, newmods: typing.Any) -> numpy.ndarray:
    return metafopy.predict(res, newmods=newmods).pred


# -- predict(newmods=) with names -------------------------------------------------------------------------------------------------------------


def test_newmods_by_position(two_mods: metafopy.RMAResult) -> None:
    by_position = pred_of(two_mods, [30, 1960])
    assert by_position.shape == (1,)


@pytest.mark.parametrize(
    'newmods',
    [
        pandas.Series({'year': 1960, 'ablat': 30}),
        {'year': 1960, 'ablat': 30},
        pandas.DataFrame({'year': [1960], 'ablat': [30]}),
        pandas.Series({'yr': 1960, 'abl': 30}),
        {'ye': 1960, 'ab': 30},
    ],
)
def test_newmods_matched_by_name(two_mods: metafopy.RMAResult, newmods: typing.Any) -> None:
    numpy.testing.assert_allclose(pred_of(two_mods, newmods), pred_of(two_mods, [30, 1960]), rtol=1e-12)


def test_newmods_frame_with_several_rows(two_mods: metafopy.RMAResult) -> None:
    frame = pandas.DataFrame({'year': [1950, 1960, 1970], 'ablat': [10, 30, 50]})
    expected = pred_of(two_mods, numpy.column_stack([frame['ablat'], frame['year']]))
    numpy.testing.assert_allclose(pred_of(two_mods, frame), expected, rtol=1e-12)


def test_newmods_unnamed_series_is_by_position(two_mods: metafopy.RMAResult) -> None:
    numpy.testing.assert_allclose(pred_of(two_mods, pandas.Series([30, 1960])), pred_of(two_mods, [30, 1960]), rtol=1e-12)


def test_newmods_unknown_name(two_mods: metafopy.RMAResult) -> None:
    with pytest.raises(metafopy.errors.InputError, match="Could not find variable 'latitude' in the model"):
        pred_of(two_mods, {'latitude': 30, 'year': 1960})


def test_newmods_same_variable_twice(two_mods: metafopy.RMAResult) -> None:
    with pytest.raises(metafopy.errors.InputError, match='Found multiple matches for the same variable name'):
        pred_of(two_mods, pandas.DataFrame({'ab': [30], 'abl': [31]}))


def test_newmods_missing_variable(two_mods: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    three = metafopy.rma(yi='yi', vi='vi', mods='~ ablat + year + tpos', data=dat)
    with pytest.raises(metafopy.errors.InputError, match='does not specify values for this variable: tpos'):
        pred_of(three, pandas.Series({'year': 1960, 'ablat': 30}))


def test_newmods_mv(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mv('yi', 'vi', mods='~ ablat + year', random='~ 1 | trial', data=dat)
    named = metafopy.predict(res, newmods={'year': 1960, 'ablat': 30}).pred
    by_position = metafopy.predict(res, newmods=[30, 1960]).pred
    numpy.testing.assert_allclose(named, by_position, rtol=1e-12)


# -- predict(prob=) ---------------------------------------------------------------------------------------------------------------------------


def test_prob_cutoff_is_evaluated(re_model: metafopy.RMAResult) -> None:
    out = metafopy.predict(re_model, prob='>log(0.8)')
    assert out.prob is not None
    assert out.prob[0] == pytest.approx(0.2016, abs=5e-5)
    plain = metafopy.predict(re_model, prob=f'>{float(numpy.log(0.8))!r}')
    assert plain.prob is not None
    numpy.testing.assert_allclose(out.prob, plain.prob, rtol=1e-12)


@pytest.mark.parametrize(
    ('text', 'value'),
    [('0.2', 0.2), ('-0.5', -0.5), ('log(0.8)', numpy.log(0.8)), ('exp(1) - 1', numpy.e - 1), ('-0.5^2', -0.25), ('2*(1+3)/4', 2.0), ('qlogis(0.5)', 0.0)],
)
def test_evaluate_cutoff(text: str, value: float) -> None:
    assert metafopy.methods.predict.evaluate_cutoff(text) == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('text', ['abc', '__import__("os")', 'log(0.8, 2)', '1 +', 'open("x")', '(1).real'])
def test_evaluate_cutoff_rejects(text: str) -> None:
    assert metafopy.methods.predict.evaluate_cutoff(text) is None


def test_prob_unparsable_cutoff_drops_prob(re_model: metafopy.RMAResult) -> None:
    assert metafopy.predict(re_model, prob='>foo').prob is None


# -- predict(addx=True) -----------------------------------------------------------------------------------------------------------------------


def test_addx_in_frame_and_print(two_mods: metafopy.RMAResult) -> None:
    out = metafopy.predict(two_mods, addx=True)
    frame = out.to_frame()
    assert list(frame.columns[-3:]) == ['X.intrcpt', 'X.ablat', 'X.year']
    numpy.testing.assert_allclose(frame[['X.intrcpt', 'X.ablat', 'X.year']].to_numpy(), two_mods.X_f if hasattr(two_mods, 'X_f') else two_mods.X)
    assert 'X.ablat' in str(out)
    plain = metafopy.predict(two_mods).to_frame()
    assert not any(c.startswith('X.') for c in plain.columns)


def test_addx_with_prob_keeps_prob_before_x(two_mods: metafopy.RMAResult) -> None:
    frame = metafopy.predict(two_mods, addx=True, prob='>0', newmods=[[10, 1950], [30, 1960]]).to_frame()
    assert list(frame.columns)[-4:] == ['prob', 'X.intrcpt', 'X.ablat', 'X.year']
    assert frame.shape[0] == 2


# -- anova(full, reduced) ---------------------------------------------------------------------------------------------------------------------


def fit(dat: pandas.DataFrame, mods: str | None, method: str) -> metafopy.RMAResult:
    return metafopy.rma(yi='yi', vi='vi', mods=mods, method=method, data=dat)


def test_lrt_ml_against_reml_stops(dat: pandas.DataFrame) -> None:
    with pytest.raises(metafopy.errors.InputError, match='Mismatch between the use of ML and REML estimation'):
        metafopy.anova(fit(dat, '~ ablat', 'ML'), fit(dat, None, 'REML'))
    with pytest.raises(metafopy.errors.InputError, match='Mismatch between the use of REML and ML estimation'):
        metafopy.anova(fit(dat, '~ ablat', 'REML'), fit(dat, None, 'ML'))


def test_lrt_fixed_full_random_reduced_stops(dat: pandas.DataFrame) -> None:
    with pytest.raises(metafopy.errors.InputError, match='Full model uses a fixed- and reduced model uses a random/mixed-effects model'):
        metafopy.anova(fit(dat, '~ ablat + year', 'FE'), fit(dat, None, 'REML'))


def test_lrt_fixed_reduced_random_full_is_fine(dat: pandas.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = metafopy.anova(fit(dat, '~ ablat', 'ML'), fit(dat, None, 'FE'))
    assert out.pval >= 0


@pytest.mark.parametrize('method', ['DL', 'EB', 'HE', 'SJ', 'PM'])
def test_lrt_warns_for_other_estimators(dat: pandas.DataFrame, method: str) -> None:
    with pytest.warns(UserWarning, match='LRTs should be based on ML/REML estimation'):
        metafopy.anova(fit(dat, '~ ablat', method), fit(dat, None, method))


def test_lrt_ml_does_not_warn_about_estimators(dat: pandas.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metafopy.anova(fit(dat, '~ ablat', 'ML'), fit(dat, None, 'ML'))


def test_lrt_reml_with_different_fixed_effects_warns(dat: pandas.DataFrame) -> None:
    with pytest.warns(UserWarning, match='REML comparisons not meaningful'):
        metafopy.anova(fit(dat, '~ ablat', 'REML'), fit(dat, None, 'REML'))


def test_lrt_not_nested_warns(dat: pandas.DataFrame) -> None:
    other = dat.copy()
    other['noise'] = numpy.sin(numpy.arange(len(dat)) * 1.7) + numpy.arange(len(dat)) % 3
    full = metafopy.rma(yi='yi', vi='vi', mods='~ ablat + year', method='ML', data=other)
    reduced = metafopy.rma(yi='yi', vi='vi', mods='~ noise', method='ML', data=other)
    with pytest.warns(UserWarning, match='appear not to be nested'):
        metafopy.anova(full, reduced)


def test_lrt_nested_models_do_not_warn(dat: pandas.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = metafopy.anova(fit(dat, '~ ablat + year', 'ML'), fit(dat, '~ ablat', 'ML'))
    assert out.pval >= 0


def test_lrt_different_sampling_variances_stop(dat: pandas.DataFrame) -> None:
    other = dat.copy()
    other['vi'] = other['vi'] * 1.5
    with pytest.raises(metafopy.errors.InputError, match='sampling variances are not equal'):
        metafopy.anova(fit(dat, '~ ablat', 'ML'), fit(other, None, 'ML'))


def test_lrt_mv_checks(dat: pandas.DataFrame) -> None:
    full = metafopy.rma_mv('yi', 'vi', mods='~ ablat', random='~ 1 | trial', method='ML', data=dat)
    reduced = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', method='REML', data=dat)
    with pytest.raises(metafopy.errors.InputError, match='Mismatch between the use of ML and REML'):
        metafopy.anova(full, reduced)


# -- accessors --------------------------------------------------------------------------------------------------------------------------------


def test_selmodel_coef_gives_beta_and_delta(re_model: metafopy.RMAResult) -> None:
    sel = metafopy.selmodel(re_model, 'halfnorm')
    assert isinstance(sel, metafopy.SelmodelResult)
    both = sel.coef()
    assert set(both) == {'beta', 'delta'}
    assert list(both['beta'].index) == list(sel.coef_names)
    assert list(both['delta'].index) == ['delta']
    numpy.testing.assert_allclose(both['beta'].to_numpy(), sel.coef('beta').to_numpy())
    numpy.testing.assert_allclose(both['delta'].to_numpy(), sel.coef('delta').to_numpy())
    assert sel.coef('all').shape[0] == sel.p + sel.deltas


def test_permutest_coef(re_model: metafopy.RMAResult) -> None:
    perm = metafopy.permutest(re_model, iter=20, seed=1)
    assert isinstance(perm, metafopy.methods.permutest.PermutestResult)
    coef = perm.coef()
    assert list(coef.index) == list(re_model.coef_names)
    numpy.testing.assert_allclose(coef.to_numpy(), re_model.beta)


def test_top_level_vcov_of_deltamethod_and_matreg(re_model: metafopy.RMAResult) -> None:
    delta = metafopy.deltamethod(re_model, fun=lambda b: {'e': numpy.exp(b[0])})
    pandas.testing.assert_frame_equal(typing.cast(pandas.DataFrame, metafopy.vcov(delta)), delta.vcov())
    cov = pandas.DataFrame([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]], index=['y', 'a', 'b'], columns=['y', 'a', 'b'])
    mat = metafopy.matreg('y', ['a', 'b'], R=cov, n=100)
    pandas.testing.assert_frame_equal(typing.cast(pandas.DataFrame, metafopy.vcov(mat)), mat.vcov())


def test_robust_dfs(re_model: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    assert re_model.dfs is None
    cluster = numpy.arange(len(dat)) // 2
    rob = metafopy.robust(re_model, cluster=cluster)
    assert rob.dfs == pytest.approx(float(len(set(cluster)) - 1))
    assert rob.ddf == len(set(cluster)) - 1
    cs = metafopy.robust(re_model, cluster=cluster, club_sandwich=True)
    assert isinstance(cs.dfs, tuple)
    assert len(cs.dfs) == cs.p
    assert cs.robust is not None
    numpy.testing.assert_allclose(cs.dfs, cs.robust.dfs)


def test_mv_group_fields(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mv('yi', 'vi', random='~ alloc | trial', struct='UN', data=dat)
    assert res.group_g is not None
    assert res.g_nlevels == res.group_g.nlevels
    assert res.g_nlevels_f == res.group_g.nlevels_f
    assert res.g_levels_f == res.group_g.levels_f
    assert res.g_levels_k is not None
    assert res.g_levels_k.shape == (3,)
    assert res.g_levels_comb_k is not None
    assert res.h_nlevels is None
    assert res.h_levels_f is None
    assert res.h_levels_k is None
    assert res.G_dimnames == ('alternate', 'random', 'systematic')
    frame = res.G_frame
    assert frame is not None
    assert list(frame.index) == list(frame.columns) == ['alternate', 'random', 'systematic']
    assert res.G is not None
    numpy.testing.assert_allclose(frame.to_numpy(), res.G)
    assert res.H_frame is None


def test_mv_gen_dimnames(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mv('yi', 'vi', mods='~ ablat', random='~ ablat | trial', struct='GEN', data=dat)
    assert res.G_dimnames == tuple(res.g_names[:-1])
    assert res.G_frame is not None


def test_mv_without_group_terms(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat)
    assert res.g_nlevels is None
    assert res.G_dimnames == ()
    assert res.G_frame is None


# -- arguments read through ... ---------------------------------------------------------------------------------------------------------------


def test_influence_measure(re_model: metafopy.RMAResult) -> None:
    full = metafopy.influence(re_model)
    cooks = metafopy.influence(re_model, measure='cooks.distance')
    assert isinstance(cooks, pandas.Series)
    numpy.testing.assert_allclose(cooks.to_numpy(), full.cook_d)
    assert list(cooks.index) == list(full.slab)
    dfbs = metafopy.influence(re_model, measure='dfbetas')
    assert isinstance(dfbs, pandas.DataFrame)
    numpy.testing.assert_allclose(dfbs.to_numpy(), full.dfbs_array)
    rst = metafopy.influence(re_model, measure='rstudent')
    expected = metafopy.rstudent(re_model)
    numpy.testing.assert_allclose(rst.resid, expected.resid)
    numpy.testing.assert_allclose(rst.z, expected.z)
    assert isinstance(metafopy.influence(re_model, measure='all'), metafopy.InfluenceResult)


def test_influence_measure_robust(re_model: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    rob = metafopy.robust(re_model, cluster=numpy.arange(len(dat)) // 2)
    assert isinstance(metafopy.influence(rob, measure='cooks.distance'), pandas.Series)
    with pytest.raises(metafopy.errors.MetaforError):
        metafopy.influence(rob, measure='dfbetas')


def test_profile_pred_and_blup(two_mods: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    prof = metafopy.profile(two_mods, steps=4, xlim=(0.0, 0.3), pred=True, blup=True, newmods=[[10, 1950], [30, 1960]])
    assert prof.pred is not None
    assert prof.pred.shape == (2, 4)
    for frame in (prof.pred, prof.pred_ci_lb, prof.pred_ci_ub, prof.pred_pi_lb, prof.pred_pi_ub):
        assert frame is not None
        assert frame.shape == (2, 4)
    assert prof.blup is not None
    assert prof.blup.shape == (len(dat), 4)
    assert list(prof.blup.index) == list(two_mods.slab)
    # the second grid column agrees with a refit at that tau^2
    assert prof.pred_pi_ub is not None
    grid = prof.tau2
    refit = metafopy.rma(yi='yi', vi='vi', mods='~ ablat + year', tau2=float(grid[1]), data=dat)
    expected = metafopy.predict(refit, newmods=[[10, 1950], [30, 1960]])
    numpy.testing.assert_allclose(prof.pred[1].to_numpy(), expected.pred, rtol=1e-8)
    numpy.testing.assert_allclose(prof.pred_pi_ub[1].to_numpy(), expected.pi_ub, rtol=1e-8)
    numpy.testing.assert_allclose(prof.blup[1].to_numpy(), metafopy.blup(refit).pred, rtol=1e-8)
    plain = metafopy.profile(two_mods, steps=4, xlim=(0.0, 0.3))
    assert plain.pred is None
    assert plain.blup is None


def test_hettest_model_argument(re_model: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    from_model = metafopy.hettest(re_model, test='score', boot=False)
    given = metafopy.hettest(dat['yi'].to_numpy(), vi=dat['vi'].to_numpy(), model=re_model, test='score', boot=False)
    numpy.testing.assert_allclose(given.statistic, from_model.statistic, rtol=1e-12)
    with pytest.raises(metafopy.errors.InputError, match="Argument 'model'"):
        metafopy.hettest(dat['yi'].to_numpy()[:5], vi=dat['vi'].to_numpy()[:5], model=re_model, test='score', boot=False)


def test_anova_abbrev_is_accepted(two_mods: metafopy.RMAResult) -> None:
    plain = metafopy.anova(two_mods, X=[[1, 30, 1960]])
    abbreviated = metafopy.anova(two_mods, X=[[1, 30, 1960]], abbrev=True)
    assert type(plain) is type(abbreviated)
    numpy.testing.assert_allclose(abbreviated.zval, plain.zval)
    assert metafopy.methods.anova.abbreviate_hyp(plain, False) is plain


def test_abbreviate_hyp_removes_factor_and_parentheses() -> None:
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Hyp:
        hyp: tuple[str, ...]

    out = metafopy.methods.anova.abbreviate_hyp([Hyp(('factor(alloc)random - factor(alloc)systematic = 0',))], True)
    assert out[0].hyp == ('allocrandom - allocsystematic = 0',)
