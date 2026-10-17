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


def test_reh_fit_keeps_three_cells_in_outdat_f() -> None:
    # R 4.3.3, metafor 5.2-1: rma(measure = "REH", ai, bi, ci, di, data = dat.bcg)$outdat.f has ai, bi and ci (13 values) and NA for di
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(measure='REH', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
    assert len(res.outdat_f['ci']) == 13  # type: ignore[arg-type]
    assert res.outdat_f['di'] is None


def test_subsetting_a_cumulative_analysis_keeps_its_order() -> None:
    # R 4.3.3, metafor 5.2-1: length(cumul(res)[1:3]$order) is 13 and length(cumul(res)[1:3]$estimate) is 3
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', data=data)
        first = metafopy.cumul(res)[:3]
    assert len(first.estimate) == 3
    assert first.order is not None
    assert len(first.order) == 13


def test_update_of_trimfill_keeps_the_measure() -> None:
    # R 4.3.3, metafor 5.2-1: update(trimfill(rma(yi, vi, data = escalc("RR", ...))))$measure is "RR" (14 studies)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        filled = metafopy.trimfill(metafopy.rma(yi='yi', vi='vi', data=data))
        again = metafopy.update(filled, method='ML')
    assert again.measure == 'RR'
    assert again.k == 14


def test_hettest_of_a_model_does_not_use_control() -> None:
    # R 4.3.3, metafor 5.2-1: hettest(res, control = list(maxiter = 5)) finds the tau2i values and all 1000 bootstrap samples like the default
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', data=data)
        plain = metafopy.hettest(res, boot=20, seed=1)
        limited = metafopy.hettest(res, boot=20, seed=1, control={'maxiter': 5, 'threshold': 1e-3})
    assert limited.tau2i is not None
    assert limited.tau2i.tolist() == plain.tau2i.tolist()  # type: ignore[union-attr]
    assert limited.pval == plain.pval


def test_rma_mv_formula_response_takes_ni_from_escalc() -> None:
    # R 4.3.3, metafor 5.2-1: rma.mv(yi ~ ablat, vi, random = ~ 1 | trial, data = escalc(...))$ni is the sample sizes (34767 for the last trial)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma_mv('yi ~ ablat', 'vi', random='~ 1 | trial', data=data)
    assert res.ni is not None
    assert res.ni[-1] == 34767


def test_rma_mv_p_counts_the_coefficients_estimated() -> None:
    # R 4.3.3, metafor 5.2-1: rma.mv(..., mods = ~ factor(study %% 2), beta = c(0.1, NA))$p is 1, with both coefficients fixed it is 0 and otherwise 2
    frame = metafopy.datasets.load('dat.assink2016').iloc[:30].copy()
    frame['g'] = (frame['study'] % 2).astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        free = metafopy.rma_mv('yi', 'vi', mods='~ g', random='~ 1 | study', data=frame)
        half = metafopy.rma_mv('yi', 'vi', mods='~ g', random='~ 1 | study', data=frame, beta=[0.1, numpy.nan])
        fixed = metafopy.rma_mv('yi', 'vi', mods='~ g', random='~ 1 | study', data=frame, beta=[0.1, 0.2])
    assert (free.p, half.p, fixed.p) == (2, 1, 0)


def test_all_rows_model_matrix_is_missing_where_a_factor_is_missing() -> None:
    # R 4.3.3, metafor 5.2-1: rma(yi, vi, mods = ~ factor(type) + factor(struct), data = escalc("ZCOR", ..., data = dat.mcdaniel1994))$X.f is NA in
    # the columns of a factor at the rows where that factor is missing (positions 273, 283, 285, 433 ... counting from 1), not zero
    frame = metafopy.datasets.load('dat.mcdaniel1994')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('ZCOR', ri='ri', ni='ni', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', mods='~ factor(type) + factor(struct)', data=data)
    assert res.X_f is not None
    assert numpy.flatnonzero(numpy.isnan(res.X_f.T.ravel()))[:6].tolist() == [272, 282, 284, 432, 442, 444]


def test_selmodel_keeps_an_equal_effects_model_as_it_is() -> None:
    # R 4.3.3, metafor 5.2-1: selmodel() refits models fitted by neither ML nor an equal-effects method, and keeps the others: with three studies
    # lost to drop00 an EE model has k = 9 but k.f = k.all = 12 (ids 1 to 12) after selmodel(res, type = "halfnorm", alternative = "less")
    ai = [4, 0, 34, 7, 6, 1, 0, 11, 2, 0, 0, 33]
    n1i = [19, 18, 341, 71, 45, 94, 22, 88, 82, 33, 15, 221]
    ci = [5, 0, 50, 16, 5, 8, 0, 12, 7, 0, 1, 43]
    n2i = [19, 18, 334, 69, 44, 96, 22, 82, 84, 30, 20, 218]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('OR', ai=ai, n1i=n1i, ci=ci, n2i=n2i, drop00=True)
        res = metafopy.rma(yi='yi', vi='vi', data=data, method='EE')
        sel: typing.Any = metafopy.selmodel(res, type='halfnorm', alternative='less')
    assert (sel.k, sel.k_f, sel.k_all) == (9, 12, 12)
    assert sel.ids.tolist() == list(range(12))


@pytest.mark.parametrize('name', ['hc', 'permutest', 'trimfill', 'fsn', 'ranktest', 'regtest', 'tes'])
def test_bias_functions_inherit_the_digits_of_the_model(name: str) -> None:
    # R 4.3.3, metafor 5.2-1: hc(), permutest(), trimfill(), fsn(), ranktest(), regtest() and tes() of rma(..., digits = 2) have digits 2
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
        res = metafopy.rma(yi='yi', vi='vi', data=data, digits=2)
        out = getattr(metafopy, name)(res)
    assert out.digits == 2
