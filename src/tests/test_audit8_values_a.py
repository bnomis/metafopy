"""Values that audit 8 found different from R metafor 5.2-1 (checked against R 4.3.3, metafor 5.2-1)."""

import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets


def test_escalc_dataset_carries_its_measure_into_rma() -> None:
    # R: rma(yi, vi, data = dat.konstantopoulos2011)$measure is "SMD" because the data set is an escalc object
    frame = metafopy.datasets.load('dat.konstantopoulos2011')
    assert frame.attrs['measure'] == 'SMD'
    assert metafopy.rma(yi='yi', vi='vi', data=frame).measure == 'SMD'


def test_dfs_equals_ddf_for_knha() -> None:
    # R: res$dfs == res$ddf for test = "knha" (11 for 13 studies and 2 coefficients)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=data, test='knha')
        pred = metafopy.predict(res)
    assert res.dfs == pytest.approx(11.0)
    assert res.ddf == 11
    # R labels the predictions 1..k when the fit has no study labels
    assert pred.slab == tuple(str(i) for i in range(1, 14))


def test_k_all_counts_the_rows_before_subsetting_for_counts() -> None:
    # R: rma(measure = "OR", ai, bi, ci, di, data = dat.bcg, subset = 1:8)$k.all is 13 (the 8 studies are $k)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame, subset=list(range(8)))
    assert (res.k, res.k_all) == (8, 13)


def test_simulate_pads_to_the_subset_unless_na_omit() -> None:
    # tests/testthat/test_misc_handling_nas.r: length(simulate(res, seed = 1234)[[1]]) is 5 under na.omit and 7 under na.pass and na.exclude
    frame = pandas.DataFrame({'yi': [numpy.nan, 1, 3, 2, 5, 4, 6], 'vi': [1, numpy.nan, 1, 1, 1, 1, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(yi='yi', vi='vi', data=frame)
    assert len(metafopy.simulate(res, seed=1234)) == 5
    for action in ('pass', 'exclude'):
        sim = metafopy.simulate(res, seed=1234, na_action=action)
        assert len(sim) == 7
        assert sim['sim_1'].isna().to_list() == [True, True, False, False, False, False, False]


def refit_fixtures() -> list[tuple[pandas.DataFrame, str | None]]:
    plain = metafopy.datasets.load('dat.bcg')
    missing = plain.copy()
    missing.loc[2, 'tpos'] = numpy.nan
    missing.loc[6, 'cpos'] = numpy.nan
    missing.loc[4, 'ablat'] = numpy.nan
    return [(plain, None), (missing, None), (missing, '~ ablat')]


# R 4.3.3, metafor 5.2-1: regtest(r)$fit and selmodel(r, type = "beta") on rma(yi, vi, [mods = ~ ablat,] data = escalc("RR", ...)) give
# measure "RR", slab 1..k, k = k.f = k.all and slab.null TRUE exactly when studies with missing values were dropped
EXPECTED_REFIT = [(13, False), (11, True), (10, True)]


@pytest.mark.parametrize('which', [0, 1, 2])
def test_regtest_and_selmodel_refit_the_studies_used(which: int) -> None:
    frame, mods = refit_fixtures()[which]
    k, slab_null = EXPECTED_REFIT[which]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', mods=mods, data=data)
        views: list[typing.Any] = [metafopy.regtest(res).fit, metafopy.selmodel(res, type='beta')]
    for view in views:
        assert view is not None
        assert view.measure == 'RR'
        assert (view.k, view.k_f, view.k_all) == (k, k, k)
        assert view.slab[:3] == ('1', '2', '3')
        assert view.slab_null is slab_null
        assert len(view.not_na) == k
        assert view.ids.tolist() == list(range(k))
        assert all(value is None for value in view.outdat_f.values())


@pytest.mark.parametrize(('measure', 'add'), [('ZTET', 0.0), ('PFT', 0.0), ('IRSD', 0.0), ('RR', 0.5), ('PLO', 0.5)])
def test_default_add_is_zero_for_measures_that_need_none(measure: str, add: float) -> None:
    # R 4.3.3, metafor 5.2-1: rma(yi, vi, measure = "ZTET")$add is 0, and 1/2 for RR and PLO
    assert metafopy.models.rma_uni.default_add(measure, None) == add
    assert metafopy.models.rma_uni.default_add(measure, 0.25) == 0.25


def test_predict_of_mantel_haenszel_fit_carries_a_method() -> None:
    # R: predict(rma.mh(ai, bi, ci, di, data = dat.bcg))$method is "FE"
    data = metafopy.datasets.load('dat.bcg')
    fit = metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=data)
    assert metafopy.predict(fit).method == 'FE'


def test_forest_keeps_missing_rows_under_exclude_and_returns_ilab_arguments() -> None:
    # R 4.3.3, metafor 5.2-1, 2 studies with missing values: forest() has 11 rows and ylim c(-2, 14) under na.omit and
    # 13 rows and c(-2, 16) under na.exclude and na.pass; without ilab, ilab.xpos = -5 and ilab.pos = 4 come back as given
    frame = metafopy.datasets.load('dat.bcg').copy()
    frame.loc[2, 'tpos'] = numpy.nan
    frame.loc[6, 'cpos'] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', data=data)
        omit = metafopy.forest(res, na_action='omit')
        exclude = metafopy.forest(res, na_action='exclude')
        keep = metafopy.forest(res, na_action='pass')
        given = metafopy.forest(res, ilab_xpos=-5, ilab_pos=4)
    assert [len(f.rows) for f in (omit, exclude, keep)] == [11, 13, 13]
    assert [f.ylim for f in (omit, exclude, keep)] == [(-2, 14), (-2, 16), (-2, 16)]
    assert given.ilab_xpos is not None
    assert given.ilab_xpos.tolist() == [-5]
    assert given.ilab_pos == [4]


def test_regtest_and_selmodel_refit_keeps_given_labels() -> None:
    # R 4.3.3, metafor 5.2-1: with slab = paste(author, year) and no missing values the refit keeps the labels (and slab.null is FALSE)
    frame = metafopy.datasets.load('dat.bcg')
    labels = [f'{a} {y}' for a, y in zip(frame['author'], frame['year'], strict=True)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi=data['yi'], vi=data['vi'], slab=labels)
        views: list[typing.Any] = [metafopy.regtest(res).fit, metafopy.selmodel(res, type='beta')]
    for view in views:
        assert view.slab[:2] == ('Aronson 1948', 'Ferguson & Simes 1949')
        assert view.slab_null is False


def test_confint_list_of_location_scale_and_selection_fits_has_digits() -> None:
    # R 4.3.3, metafor 5.2-1: confint(rma(yi, vi, mods = ~ ablat, scale = ~ alloc))$digits is 4 for every one of the nine entries
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        ls = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', scale='~ alloc', data=data)
        assert metafopy.confint(ls, digits=2).digits == 2
        assert metafopy.confint(ls).digits == 4


def test_location_scale_predict_labels_follow_the_names_of_newmods() -> None:
    # R 4.3.3, metafor 5.2-1: predict(rma(yi, vi, mods = ~ year, scale = ~ ablat), newmods = c(ablat = 30, year = 1960))$slab is c("ablat", "year")
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', mods='~ year', scale='~ ablat', data=data)
        named = metafopy.predict(res, newmods={'ablat': 30, 'year': 1960})
        plain = metafopy.predict(res, newmods=[10, 60])
    assert named.slab == ('ablat', 'year')
    assert plain.slab == ('1', '2')
