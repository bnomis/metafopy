"""Input and data handling of the round-five audit: escalc attributes (``slab``, ``ni``), warnings for omitted studies, scalar recycling, missing values
in the scale model, functions in formulas, ``emmprep(data=)``, ``regtest`` and ``fsn`` column arguments, ``labbe`` of fits from counts, ``dfround``
and ``rma_mh`` without events.

The numbers come from the audit (checked in R with metafor 5.2-1): ``rma(scale=~0+C(meta))`` of bangertdrowns2004 has k = 46 and alpha -4.3350, and
a second ``escalc(include=)`` of the bcg data leaves ``ni`` at 262, 609, 451, 26465.
"""

import typing
import warnings

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.formula

matplotlib.use('Agg')

any_rma: typing.Callable[..., typing.Any] = metafopy.rma
any_escalc: typing.Callable[..., typing.Any] = metafopy.escalc


def bcg_raw() -> pandas.DataFrame:
    dat = metafopy.datasets.load('dat.bcg').copy()
    dat['label'] = [f'{a} {y}' for a, y in zip(dat['author'], dat['year'], strict=True)]
    return dat


def bcg() -> pandas.DataFrame:
    dat = bcg_raw()
    out: pandas.DataFrame = any_escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, slab=dat['label'])
    return out


# -- escalc attributes ---------------------------------------------------------------------------------------------------------------------------


def test_rma_uses_the_slab_of_an_escalc_frame() -> None:
    dat = bcg()
    res = any_rma('yi', 'vi', data=dat)
    assert list(res.slab) == list(dat.attrs['slab'])
    assert res.slab[0] == 'Aronson 1948'
    assert list(any_rma(dat).slab) == list(dat.attrs['slab'])


def test_slab_argument_beats_the_escalc_slab() -> None:
    dat = bcg()
    assert list(any_rma('yi', 'vi', data=dat, slab=[f's{i}' for i in range(13)]).slab)[:2] == ['s0', 's1']


def test_rma_uses_the_slab_of_a_row_subset() -> None:
    dat = bcg()
    sub = dat.iloc[3:8]
    assert list(any_rma('yi', 'vi', data=sub).slab) == list(dat.attrs['slab'][3:8])
    assert list(any_rma('yi', 'vi', data=dat[dat['ablat'] > 40]).slab) == [s for s, a in zip(dat.attrs['slab'], dat['ablat'], strict=True) if a > 40]


def test_rma_mv_uses_the_slab_of_an_escalc_frame() -> None:
    dat = bcg()
    dat['study'] = range(13)
    res = metafopy.rma_mv('yi', 'vi', random='~ 1 | study', data=dat)
    assert list(res.slab) == list(dat.attrs['slab'])


def test_escalc_include_keeps_ni_outside_the_included_rows() -> None:
    raw = bcg_raw()
    first = any_escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=raw, include=[False, False, *[True] * 11])
    numpy.testing.assert_array_equal(first.attrs['ni'][:3], [numpy.nan, numpy.nan, 451])
    second = any_escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=first, include=[True, True, *[False] * 11])
    numpy.testing.assert_array_equal(second.attrs['ni'][:4], [262, 609, 451, 26465])
    assert not numpy.isnan(second.attrs['ni']).any()


def test_escalc_keeps_the_slab_when_run_again_without_one() -> None:
    dat = bcg()
    again = any_escalc('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    assert list(again.attrs['slab']) == list(dat.attrs['slab'])


def test_rbind_escalc_of_row_subsets_has_one_ni_per_row() -> None:
    dat = bcg()
    out = metafopy.rbind_escalc(dat.iloc[3:8], dat[dat['ablat'] > 40])
    assert len(out.attrs['ni']) == out.shape[0]
    ni = numpy.asarray(dat.attrs['ni'])
    expected = [*ni[3:8], *ni[(dat['ablat'] > 40).to_numpy()]]
    numpy.testing.assert_array_equal(out.attrs['ni'], expected)
    assert len(out.attrs['slab']) == out.shape[0]


def test_summary_escalc_of_a_column_subset() -> None:
    dat = bcg()
    out = metafopy.summary_escalc(dat[['ablat', 'yi', 'vi']])
    assert {'sei', 'zi', 'pval', 'ci.lb', 'ci.ub'} <= set(out.columns)
    with pytest.raises(metafopy.errors.InputError):
        metafopy.summary_escalc(dat[['ablat', 'yi']])


def test_summary_escalc_picks_another_yi_when_the_first_is_dropped() -> None:
    dat = bcg()
    dat = any_escalc('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, var_names=['yi2', 'vi2'])
    assert dat.attrs['yi_names'] == ['yi2', 'yi']
    out = metafopy.summary_escalc(dat.drop(columns=['yi2', 'vi2']))
    assert 'sei' in out.columns


# -- warnings and recycling ----------------------------------------------------------------------------------------------------------------------


def test_rma_warns_about_studies_omitted_for_missing_values() -> None:
    dat = bcg()
    dat.loc[2, 'yi'] = numpy.nan
    dat.loc[4, 'ablat'] = numpy.nan
    with pytest.warns(UserWarning, match=r'^1 study with NAs omitted from model fitting\.$'):
        any_rma('yi', 'vi', data=dat)
    with pytest.warns(UserWarning, match=r'^2 studies with NAs omitted from model fitting\.$'):
        any_rma('yi', 'vi', mods='~ ablat', data=dat)


def test_rma_does_not_warn_without_missing_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        any_rma('yi', 'vi', data=bcg())


def test_scalar_vi_and_weights_are_recycled() -> None:
    y = numpy.array([0.1, 0.3, 0.2, 0.5])
    numpy.testing.assert_allclose(any_rma(y, 0.04).beta, any_rma(y, numpy.full(4, 0.04)).beta)
    full = any_rma(y, numpy.array([0.02, 0.04, 0.05, 0.08]), weights=numpy.ones(4))
    numpy.testing.assert_allclose(any_rma(y, numpy.array([0.02, 0.04, 0.05, 0.08]), weights=1).beta, full.beta)
    numpy.testing.assert_allclose(any_rma(y, 0.04, weights=1).vi, numpy.full(4, 0.04))


def test_vectors_of_the_wrong_length_are_still_refused() -> None:
    with pytest.raises(metafopy.errors.InputError):
        any_rma(numpy.array([0.1, 0.3, 0.2]), numpy.array([0.04, 0.05]))


# -- location-scale models -----------------------------------------------------------------------------------------------------------------------


def test_scale_factor_with_missing_values_drops_those_studies() -> None:
    dat = metafopy.datasets.load('dat.bangertdrowns2004')
    assert int(dat['meta'].isna().sum()) == 2
    with pytest.warns(UserWarning, match=r'2 studies with NAs omitted from model fitting\.'):
        res = any_rma('yi', 'vi', data=dat, scale='~ 0 + C(meta)')
    assert res.k == 46
    assert res.alpha[0] == pytest.approx(-4.3350, abs=2e-4)


def test_scale_factor_with_missing_values_and_identity_link() -> None:
    dat = metafopy.datasets.load('dat.bangertdrowns2004')
    with pytest.warns(UserWarning, match=r'2 studies with NAs omitted from model fitting\.'):
        res = any_rma('yi', 'vi', data=dat, scale='~ 0 + C(meta)', link='identity')
    assert res.k == 46
    log = any_rma('yi', 'vi', data=dat, scale='~ 0 + C(meta)')
    numpy.testing.assert_allclose(res.alpha, numpy.exp(log.alpha), rtol=1e-3)


# -- functions in formulas -----------------------------------------------------------------------------------------------------------------------


def alloc_frame() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def test_factor_in_mods_equals_the_plain_factor() -> None:
    dat = alloc_frame()
    plain = any_rma('yi', 'vi', mods='~ alloc', data=dat)
    wrapped = any_rma('yi', 'vi', mods='~ factor(alloc)', data=dat)
    numpy.testing.assert_allclose(wrapped.beta, plain.beta)
    assert list(wrapped.coef_names) == ['intrcpt', 'factor(alloc)random', 'factor(alloc)systematic']


def test_relevel_changes_the_reference_level() -> None:
    dat = alloc_frame()
    res = any_rma('yi', 'vi', mods='~ relevel(factor(alloc), ref="random")', data=dat)
    base = any_rma('yi', 'vi', mods='~ 0 + alloc', data=dat)
    names = list(res.coef_names)
    assert names[0] == 'intrcpt'
    assert len(names) == 3
    numpy.testing.assert_allclose(res.beta[0], base.beta[list(base.coef_names).index('allocrandom')])
    assert res.ci_lb.shape == (3,)


def test_scale_skips_missing_values() -> None:
    dat = alloc_frame()
    dat.loc[3, 'ablat'] = numpy.nan
    with pytest.warns(UserWarning, match='1 study with NAs omitted'):
        res = any_rma('yi', 'vi', mods='~ scale(ablat)', data=dat)
    assert res.k == 12
    assert res.X[:, 1].mean() == pytest.approx(0, abs=1e-12)
    assert res.X[:, 1].std(ddof=1) == pytest.approx(1)


def test_r_scale_matches_r() -> None:
    x = numpy.array([1.0, 2.0, numpy.nan, 6.0])
    out = metafopy.formula.r_scale(x)
    numpy.testing.assert_allclose(out[[0, 1, 3]], [-2 / numpy.sqrt(7), -1 / numpy.sqrt(7), 3 / numpy.sqrt(7)])
    assert numpy.isnan(out[2])
    numpy.testing.assert_allclose(metafopy.formula.r_scale(x, center=False)[[0, 1, 3]], numpy.array([1.0, 2.0, 6.0]) / numpy.sqrt(41 / 2))
    numpy.testing.assert_allclose(metafopy.formula.r_scale(x, scale=False)[[0, 1, 3]], [-2.0, -1.0, 3.0])


def test_user_function_in_a_formula() -> None:
    dat = alloc_frame()

    def twice(x: pandas.Series) -> pandas.Series:
        return 2 * x

    res = any_rma('yi', 'vi', mods='~ twice(ablat)', data=dat)
    ref = any_rma('yi', 'vi', mods=numpy.asarray(2 * dat['ablat']), data=dat)
    numpy.testing.assert_allclose(res.beta, ref.beta)
    assert 'twice(ablat)' in res.coef_names


def test_a_variable_named_like_a_builtin_does_not_hide_the_builtin() -> None:
    dat = alloc_frame()
    scale = 'not a function'  # noqa: F841
    res = any_rma('yi', 'vi', mods='~ scale(ablat)', data=dat)
    assert res.k == 13


def test_relevel_rejects_an_unknown_level() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.formula.r_relevel(pandas.Categorical(['a', 'b']), ref='z')


# -- emmprep -------------------------------------------------------------------------------------------------------------------------------------


def test_emmprep_uses_the_data_as_given() -> None:
    dat = alloc_frame()
    dna = dat.copy()
    dna.loc[2, 'ablat'] = numpy.nan
    res = any_rma('yi', 'vi', mods='~ alloc + ablat', data=dna, subset=numpy.arange(13) > 0)
    fitted = metafopy.emmeans(metafopy.emmprep(res), 'alloc').summary()
    given = metafopy.emmeans(metafopy.emmprep(res, data=dat), 'alloc').summary()
    assert list(given.to_frame()['alloc']) == ['alternate', 'random', 'systematic']
    assert not numpy.allclose(given.to_frame()['emmean'], fitted.to_frame()['emmean'])
    expected_mean = dat['ablat'].mean()
    manual = res.coef()
    random_mean = manual['intrcpt'] + manual['allocrandom'] + manual['ablat'] * expected_mean
    assert given.to_frame().set_index('alloc').loc['random', 'emmean'] == pytest.approx(random_mean)


# -- regtest and fsn -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('predictor', ['ni', 'ninv', 'sqrtni', 'sqrtninv'])
def test_regtest_finds_ni_of_an_escalc_frame(predictor: typing.Any) -> None:
    dat = bcg()
    ref = metafopy.regtest(dat['yi'].to_numpy(), dat['vi'].to_numpy(), ni=dat.attrs['ni'], predictor=predictor)
    out = metafopy.regtest('yi', 'vi', data=dat, predictor=predictor)
    assert out.pval == pytest.approx(ref.pval, rel=1e-12)
    assert out.zval == pytest.approx(ref.zval, rel=1e-12)


def test_regtest_of_a_model_resolves_the_ni_column() -> None:
    dat = bcg()
    dat['n'] = dat.attrs['ni']
    res = any_rma('yi', 'vi', data=dat)
    named = metafopy.regtest(res, predictor='ni', ni='n', data=dat)
    arr = metafopy.regtest(res, predictor='ni', ni=dat['n'].to_numpy())
    assert named.pval == pytest.approx(arr.pval, rel=1e-12)


def test_fsn_orwin_without_vi_resolves_the_column() -> None:
    dat = bcg()
    named = metafopy.fsn('yi', data=dat, type='Orwin')
    arr = metafopy.fsn(dat['yi'].to_numpy(), type='Orwin')
    assert named.fsnum == arr.fsnum == 13


def test_fsn_general_with_a_numpy_target() -> None:
    dat = bcg()
    numpy_target = metafopy.fsn('yi', 'vi', data=dat, type='General', target=numpy.log(0.95))
    float_target = metafopy.fsn('yi', 'vi', data=dat, type='General', target=float(numpy.log(0.95)))
    assert numpy_target.fsnum == float_target.fsnum == 138


# -- dfround -------------------------------------------------------------------------------------------------------------------------------------


def test_dfround_makes_numeric_columns_with_missing_digits_missing() -> None:
    dat = pandas.DataFrame({'a': [1.234, 2.345], 'b': [1.234, 2.345], 'c': ['x', 'y']})
    out = metafopy.dfround(dat, [1, numpy.nan, None])
    assert out['a'].tolist() == [1.2, 2.3]
    assert out['b'].isna().all()  # R: round(x, NA) is NA (dfround(dat, c(1, NA, NA)))
    assert out['c'].tolist() == ['x', 'y']
    assert metafopy.dfround(dat, numpy.nan)['a'].isna().all()


# -- rma_mh without events -----------------------------------------------------------------------------------------------------------------------


def test_rma_mh_rd_without_events() -> None:
    with pytest.warns(UserWarning, match='yi/vi values are NA'):
        res = metafopy.rma_mh(ai=[0, 0, 0], bi=[10, 12, 9], ci=[0, 0, 0], di=[11, 10, 12], measure='RD')
    assert res.beta[0] == 0
    assert res.se[0] == 0
    assert numpy.isnan(res.zval[0])
    assert numpy.isnan(res.pval[0])
    assert 'NA' in str(res)


def test_rma_mh_rr_and_or_without_events_are_missing() -> None:
    for measure in ('RR', 'OR'):
        with pytest.warns(UserWarning, match='yi/vi values are NA'):
            res = metafopy.rma_mh(ai=[0, 0, 0], bi=[10, 12, 9], ci=[0, 0, 0], di=[11, 10, 12], measure=measure)
        assert numpy.isnan(res.beta[0])
        assert numpy.isnan(res.pval[0])


def test_rma_mh_when_one_outcome_never_occurs_does_not_fail() -> None:
    for measure in ('RD', 'RR'):
        with pytest.warns(UserWarning, match='yi/vi values are NA'):
            res = metafopy.rma_mh(ai=[5, 6, 7], bi=[0, 0, 0], ci=[4, 3, 2], di=[0, 0, 0], measure=measure)
        assert res.k == 3


# -- labbe ---------------------------------------------------------------------------------------------------------------------------------------


def test_labbe_of_a_fit_from_counts() -> None:
    dat = metafopy.datasets.load('dat.bcg')
    res = any_rma(measure='RR', ai=dat['tpos'], bi=dat['tneg'], ci=dat['cpos'], di=dat['cneg'])
    out = metafopy.labbe(res)
    named = metafopy.labbe(any_rma(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat))
    mh = metafopy.labbe(metafopy.rma_mh(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat))
    assert out.x.shape == (13,)
    numpy.testing.assert_allclose(out.x, named.x)
    numpy.testing.assert_allclose(out.y, mh.y)
    matplotlib.pyplot.close('all')


def test_labbe_of_a_fit_from_counts_with_subset_and_slab() -> None:
    dat = metafopy.datasets.load('dat.bcg')
    res = any_rma(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, subset=numpy.arange(13) > 2)
    out = metafopy.labbe(res)
    assert out.x.shape == (10,)
    assert out.slab[:2] == ('4', '5')
    matplotlib.pyplot.close('all')


def test_labbe_of_a_fit_from_yi_and_vi_still_needs_the_counts() -> None:
    dat = metafopy.datasets.load('dat.bcg')
    res = any_rma('yi', 'vi', data=metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat))
    with pytest.raises(metafopy.errors.InputError):
        metafopy.labbe(res)
    matplotlib.pyplot.close('all')
