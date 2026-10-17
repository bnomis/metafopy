"""Regression tests for the audit-8 findings on arguments metafopy refused or accepted differently from R metafor 5.2-1 (values cross-checked in R)."""

import math
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import audit.rlib
import metafopy
import metafopy.core.zeroin
import metafopy.methods.emm_links
import metafopy.methods.permutest


def fit() -> metafopy.models.rma_uni.RMAResult:
    return metafopy.rma_uni('yi', 'vi', data=refutil.bcg())


# -- control: entries that only other functions of metafor read are ignored, as R ignores them --------------
def test_rma_uni_ignores_control_entries_of_other_functions() -> None:
    ref = fit()
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg(), control={'vc.min': 0, 'nearpd': True, 'extendInt': 'yes'})
    assert res.tau2 == ref.tau2


def test_rma_uni_still_refuses_control_entries_no_function_reads() -> None:
    with pytest.raises(metafopy.errors.ArgumentError, match='unknownentry'):
        metafopy.rma_uni('yi', 'vi', data=refutil.bcg(), control={'unknownentry': 1})


def test_permutest_ignores_control_entries_of_other_functions() -> None:
    res = fit()
    ref = metafopy.permutest(res, exact=False, iter=100, seed=1)
    other = metafopy.permutest(res, exact=False, iter=100, seed=1, control={'threshold': 1e-6, 'stepadj': 0.5})
    assert isinstance(ref, metafopy.methods.permutest.PermutestResult)
    assert isinstance(other, metafopy.methods.permutest.PermutestResult)
    assert other.pval == pytest.approx(ref.pval)
    with pytest.raises(metafopy.errors.ArgumentError, match='unknownentry'):
        metafopy.permutest(res, control={'unknownentry': 1})


def test_confint_ignores_control_entries_of_other_functions() -> None:
    res = fit()
    ref = metafopy.confint(res)
    other = metafopy.confint(res, control={'threshold': 1e-6, 'optimizer': 'BFGS', 'extendInt': 'yes'})
    assert other.random is not None
    assert ref.random is not None
    numpy.testing.assert_allclose(other.random.to_numpy(), ref.random.to_numpy())
    with pytest.raises(metafopy.errors.InputError, match='unknownentry'):
        metafopy.confint(res, control={'unknownentry': 1})


def test_hettest_ignores_control_entries_of_other_functions() -> None:
    res = fit()
    ref = metafopy.hettest(res, test='lrt', boot=False)
    other = metafopy.hettest(res, test='lrt', boot=False, control={'stepadj': 0.5, 'optimizer': 'BFGS'})
    assert other.statistic == pytest.approx(ref.statistic)
    with pytest.raises(metafopy.errors.InputError, match='unknownentry'):
        metafopy.hettest(res, control={'unknownentry': 1})


def test_rma_mv_ignores_control_entries_of_other_functions() -> None:
    dat = refutil.bcg()
    ref = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat)
    other = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat, control={'threshold': 1})
    assert other.sigma2 is not None
    assert ref.sigma2 is not None
    numpy.testing.assert_allclose(other.sigma2, ref.sigma2)


# -- permutest(control=list(extendInt=)) -------------------------------------------------------------------
def test_permutest_extend_int_finds_the_bound_a_narrow_search_misses() -> None:
    """R 4.3.3 with metafor 5.2-1: bounds (-1.620814355, NA) with extendInt 'no' and (-1.620814355, -0.2176118037) with 'yes'."""
    dat = refutil.bcg().iloc[[0, 1, 2, 3, 4, 6]].reset_index(drop=True)
    res = metafopy.rma_uni('yi', 'vi', data=dat)
    narrow = {'distfac': 0.2, 'cialt': 'one.sided'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        no = metafopy.permutest(res, exact=True, permci=True, control={**narrow, 'extendInt': 'no'})
        yes = metafopy.permutest(res, exact=True, permci=True, control={**narrow, 'extendInt': 'yes'})
    assert isinstance(no, metafopy.methods.permutest.PermutestResult)
    assert isinstance(yes, metafopy.methods.permutest.PermutestResult)
    assert no.ci_lb[0] == pytest.approx(-1.620814355)
    assert math.isnan(no.ci_ub[0])
    assert yes.ci_lb[0] == pytest.approx(-1.620814355)
    assert yes.ci_ub[0] == pytest.approx(-0.2176118037)


# -- confint(xlim=) with negative marginal variances -------------------------------------------------------
def test_confint_search_interval_with_negative_marginal_variances_gives_na_and_warnings() -> None:
    """R: ``confint(res, xlim=c(-3, 3))`` gives NA bounds with the warnings 'Error in iterative search for the lower/upper bound.'"""
    res = fit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = metafopy.confint(res, xlim=(-3, 3))
    assert out.random is not None
    assert out.random.iloc[0, 0] == pytest.approx(0.3132433, rel=1e-6)
    assert numpy.isnan(out.random.iloc[:, 1:].to_numpy()).all()
    assert [str(w.message) for w in caught] == ['Error in iterative search for the lower bound.', 'Error in iterative search for the upper bound.']


# -- uniroot() warns when it does not converge -------------------------------------------------------------
def test_uniroot_warns_when_not_converged() -> None:
    with pytest.warns(UserWarning, match='_NOT_ converged in 5 iterations'):
        out = metafopy.core.zeroin.uniroot(lambda x: x**3 - 2, 0.0, 10.0, maxiter=5)
    assert not out.converged
    assert out.iterations == 5


def test_confint_warns_when_the_root_search_does_not_converge() -> None:
    with pytest.warns(UserWarning, match='_NOT_ converged in 5 iterations'):
        metafopy.confint(fit(), control={'maxiter': 5})


# -- emmprep(tran=) accepts every transformation emmeans knows ---------------------------------------------
@pytest.mark.parametrize(
    ('tran', 'expected'),
    [
        # R: summary(emmeans(emmprep(res, tran=), ~1), type='response') on the bcg random-effects model (log risk ratios)
        ('atanh', (-0.613511085703, 0.112112508902, -0.788289551395, -0.347121375479)),
        ('cauchit', (0.302516479029, 0.0378842100879, 0.2397011058, 0.389396019999)),
        ('cloglog', (0.387018753677, 0.0539355123004, 0.291123738735, 0.501507175458)),
        ('log1p', (-0.510579063212, 0.0879888457012, -0.6559257066, -0.303833916218)),
        ('asinh.sqrt', (0.603581380362, 0.353743518722, 0.137000937423, 1.64131299789)),
        ('log.o.r.', (0.489420936788, 0.0879888457012, 0.3440742934, 0.696166083782)),
        # an unknown name is the identity link carrying that name
        ('holm', (-0.714532348365, 0.17978153178, -1.06689767574, -0.36216702099)),
    ],
)
def test_emmprep_transformations_of_emmeans(tran: str, expected: tuple[float, float, float, float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        summary = metafopy.emmeans(metafopy.emmprep(fit(), tran=tran), specs='1').summary(type='response')
    row = summary.frame.iloc[0]
    numpy.testing.assert_allclose([row['response'], row['SE'], row['asymp.LCL'], row['asymp.UCL']], expected, rtol=1e-8)


def test_emmprep_exp_back_transforms_with_the_log() -> None:
    """R: ``tran='exp'`` gives log(eta) as the response, which is NaN for the negative estimate, with SE 0.251607267595."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        summary = metafopy.emmeans(metafopy.emmprep(fit(), tran='exp'), specs='1').summary(type='response')
    row = summary.frame.iloc[0]
    assert math.isnan(row['response'])
    assert row['SE'] == pytest.approx(0.251607267595, rel=1e-8)


def test_emmprep_plus_sqrt_is_not_supported() -> None:
    with pytest.raises(metafopy.errors.NotSupportedError, match='sqrt'):
        metafopy.methods.emm_links.make_link('+.sqrt')


# -- warnings R gives --------------------------------------------------------------------------------------
def test_rma_mv_warns_that_struct_is_disregarded_without_inner_outer() -> None:
    dat = refutil.bcg()
    with pytest.warns(UserWarning, match="'struct' argument is disregaded"):
        metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', struct='AR', data=dat)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat)


def test_fsn_warns_that_vi_sei_and_subset_are_ignored_for_a_model() -> None:
    res = fit()
    with pytest.warns(UserWarning, match="Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object"):
        metafopy.fsn(res, vi=0.1)


def test_rma_without_intercept_and_moderators_coerces_an_intercept() -> None:
    with pytest.warns(UserWarning, match='Coerced an intercept into the model'):
        res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg(), intercept=False)
    assert res.p == 1


def test_rma_uni_checks_the_measure() -> None:
    with pytest.raises(metafopy.errors.InputError, match="Unknown 'measure' specified"):
        metafopy.rma_uni('yi', 'vi', data=refutil.bcg(), measure='ICC')
    assert metafopy.rma_uni('yi', 'vi', data=refutil.bcg(), measure='HR').measure == 'HR'


def large_smd() -> pandas.DataFrame:
    return metafopy.escalc(measure='SMD', m1i=[64.0, 66.0], sd1i=[17.0, 20.0], n1i=[75, 18], m2i=[119.0, 137.0], sd2i=[29.0, 48.0], n2i=[71, 18])


def test_unusually_large_smd_warns_once_per_yi_vector() -> None:
    # .chkyisize keeps the signatures of the last 100 yi vectors it warned about
    metafopy.effects.measures.YICHECKS.clear()
    with pytest.warns(UserWarning, match='unusually large'):
        large_smd()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        large_smd()


# -- refusals R makes ------------------------------------------------------------------------------------------
def test_fsn_pool_is_matched_partially_and_checked() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'arg' should be one of"):
        metafopy.fsn([0.1, 0.2, 0.3], [0.01, 0.02, 0.03], pool=typing.cast(typing.Literal['stouffer', 'fisher'], 'x'))


def test_fmtt_requires_a_scalar() -> None:
    with pytest.raises(metafopy.errors.InputError, match="Argument 'val' must be a scalar"):
        metafopy.fmtt(typing.cast(float, [1.0, 2.0]), 'z')


def test_harness_factor_matches_values_to_levels_as_character() -> None:
    # R: factor(factor(c(1, 2)), levels = c(2, 1), labels = c('con', 'exp')) keeps both values
    out = audit.rlib.r_factor(pandas.Categorical(['1', '2', '1']), levels=[2, 1], labels=['con', 'exp'])
    assert list(out) == ['exp', 'con', 'exp']
    assert list(out.categories) == ['con', 'exp']


def test_emmeans_type_sets_the_default_of_the_summary() -> None:
    """R: ``emmeans(emmprep(res), specs = '1', type = 'response')`` is 0.4894209 (0.3440743, 0.6961661) on the bcg risk ratios."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row = metafopy.emmeans(metafopy.emmprep(fit(), tran='log'), specs='1', type='response').summary().frame.iloc[0]
    numpy.testing.assert_allclose([row['response'], row['asymp.LCL'], row['asymp.UCL']], [0.4894209, 0.3440743, 0.6961661], rtol=1e-6)
    with pytest.raises(metafopy.errors.InputError, match="'type' must be"):
        metafopy.emmeans(metafopy.emmprep(fit()), specs='1', type='x')


def test_regtest_lm_of_an_rma_peto_fit() -> None:
    """R: ``regtest(rma.peto(...bcg), model = 'lm')`` has t = -1.2564743, p = 0.2349720, b = -0.2308426 (CI -0.7683200, 0.3066348), 11 df."""
    peto = metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    out = metafopy.regtest(peto, model='lm')
    assert out.zval == pytest.approx(-1.2564743440, rel=1e-8)
    assert out.pval == pytest.approx(0.2349719521, rel=1e-8)
    assert out.est == pytest.approx(-0.2308426232, rel=1e-8)
    with pytest.raises(metafopy.errors.InputError, match="'tau2' must be a scalar"):
        metafopy.regtest(peto)


def test_forest_takes_rows_for_the_studies_with_missing_values() -> None:
    """R: ``rows`` has one entry per outcome of the data, those left out for missing values included, whose rows are closed up."""
    import matplotlib

    matplotlib.use('Agg')
    dat = refutil.bcg().copy()
    dat.loc[[1, 4], 'yi'] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', data=dat)
        out = metafopy.forest(res, rows=list(range(1, 14)))
    assert list(out.rows) == [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


def test_pairmat_columns_are_unnamed_so_predict_takes_them_by_position() -> None:
    """R: ``predict(rma(yi, vi, mods = ~ablat + year), newmods = pairmat(rma(..., mods = ~factor(alloc)), btt = 2:3))`` runs, by position."""
    dat = refutil.bcg()
    by_alloc = metafopy.rma_uni('yi', 'vi', mods='~ C(alloc)', data=dat)
    pm = metafopy.pairmat(by_alloc, btt=[1, 2])
    assert [str(c) for c in pm.columns] == ['0', '1', '2']
    other = metafopy.rma_uni('yi', 'vi', mods='~ ablat + year', data=dat)
    assert metafopy.predict(other, newmods=pm).pred.shape == (1,)
