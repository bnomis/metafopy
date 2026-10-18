"""Values that audit 8 found different from R metafor 5.2-1 (checked against R 4.3.3, metafor 5.2-1)."""

import typing
import warnings

import numpy
import pandas
import pytest

import audit.rlib
import audit.valuetree
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


def bcg_frame() -> pandas.DataFrame:
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)


def test_glmm_outdat_f_has_the_sample_sizes() -> None:
    # R 4.3.3, metafor 5.2-1: rma.glmm(measure = "OR", ...)$outdat.f$ni is ai + bi + ci + di (262, 609, ...)
    frame = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res: typing.Any = metafopy.rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame, model='UM.FS')
    ni = res.outdat_f['ni']
    assert ni is not None
    assert ni[:3].tolist() == [262.0, 609.0, 451.0]
    assert ni[-1] == 34767.0


def test_cumul_of_a_mantel_haenszel_fit_has_ids() -> None:
    # R 4.3.3, metafor 5.2-1: cumul(rma.mh(...))$ids is 1:13; with order = ablat the ids follow the order (5 8 11 7 9 12 13 3 10 1 6 4 2)
    frame = metafopy.datasets.load('dat.bcg')
    res = metafopy.rma_mh(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=frame)
    plain: typing.Any = metafopy.cumul(res)
    assert plain.ids.tolist() == list(range(13))
    ordered: typing.Any = metafopy.cumul(res, order=frame['ablat'].to_numpy())
    assert (ordered.ids + 1).tolist() == [5, 8, 11, 7, 9, 12, 13, 3, 10, 1, 6, 4, 2]


def test_predict_ddf_has_one_value_per_prediction() -> None:
    # R 4.3.3, metafor 5.2-1: predict(robust(rma(yi, vi, mods = ~ ablat), cluster = trial))$ddf is 11 for each of the 13 predictions
    data = bcg_frame()
    res = metafopy.robust(metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=data), cluster=data['trial'].to_numpy())
    pred: typing.Any = metafopy.predict(res)
    assert numpy.asarray(pred.ddf).tolist() == [11.0] * 13
    assert numpy.asarray(metafopy.predict(res, newmods=[30, 40]).ddf).tolist() == [11.0, 11.0]


def test_mv_dist_is_kept_in_full() -> None:
    # R 4.3.3, metafor 5.2-1: rma.mv(..., dist = "euclid")$dist is "euclidean"
    prop = typing.cast(property, metafopy.models.mv_result.RMAMVResult.dist)
    holder = typing.cast(typing.Any, type('Holder', (), {'spec': {'dist': 'euclid'}}))()
    assert prop.fget is not None
    assert prop.fget(holder) == ('euclidean', 'euclidean')


def test_deltamethod_inherits_the_digits_of_the_model() -> None:
    # R 4.3.3, metafor 5.2-1: deltamethod(rma(..., digits = 2), fun)$digits has the nine entries 2; for estimates it has est, se, test, pval and ci
    data = bcg_frame()
    res = metafopy.rma(yi='yi', vi='vi', data=data, digits=2)
    out = metafopy.deltamethod(res, fun=numpy.exp)
    assert set(out.digits.values()) == {2}
    assert len(out.digits) == 9
    plain = metafopy.deltamethod(numpy.array([1.0, 2.0]), vcov=numpy.eye(2), fun=lambda x: x[0] * x[1])
    assert list(plain.digits) == ['est', 'se', 'test', 'pval', 'ci']


def test_anova_of_a_selection_model_uses_its_digits() -> None:
    # R 4.3.3, metafor 5.2-1: anova(selmodel(rma(..., digits = 2), type = "stepfun", steps = c(.05, 1)))$digits is 2
    data = bcg_frame()
    res = metafopy.rma(yi='yi', vi='vi', data=data, digits=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sel = metafopy.selmodel(res, type='stepfun', steps=[0.05, 1])
        out: typing.Any = metafopy.anova(sel)
    assert out.digits == 2


def test_add_follows_the_measure_argument_not_the_measure_of_the_data() -> None:
    # R 4.3.3, metafor 5.2-1: rma(yi, vi, data = escalc(measure = "PFT", ...))$add is 0.5; rma(measure = "PFT", ...)$add is 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = metafopy.escalc('PFT', xi=[3, 5, 8], ni=[20, 30, 40])
        assert metafopy.rma(yi='yi', vi='vi', data=data, method='EE').add == 0.5
        assert metafopy.rma(yi='yi', vi='vi', data=data, method='EE', measure='PFT').add == 0.0


def test_regtest_of_estimates_in_an_escalc_frame_keeps_the_measure() -> None:
    # R 4.3.3, metafor 5.2-1: regtest(yi, vi, data = dat)$fit$measure is "RR" for an escalc object
    data = bcg_frame()
    out: typing.Any = metafopy.regtest('yi', vi='vi', data=data)
    assert out.fit.measure == 'RR'


def test_structure_with_a_vector_class_is_not_a_factor() -> None:
    # structure(14.3, dim = c(1L, 1L), class = c('vcovmat', 'matrix')) crashed on the comparison of the class vector with 'factor'
    out = audit.rlib.r_structure(numpy.array([14.3]), dim=numpy.array([1, 1]), **{'class': numpy.array(['vcovmat', 'matrix'], dtype=object)})
    assert out.shape == (1, 1)


def test_regtest_of_a_trimfill_object_numbers_the_studies() -> None:
    # R 4.3.3, metafor 5.2-1: regtest(trimfill(rma(yi, vi, data = dat, slab = author)))$fit$slab is 1:14, as the yi of trimfill() has no slab
    data = bcg_frame()
    filled = metafopy.trimfill(metafopy.rma(yi='yi', vi='vi', data=data))
    out: typing.Any = metafopy.regtest(filled)
    assert list(out.fit.slab) == [str(i) for i in range(1, 15)]


def test_predict_vcov_is_a_list_of_the_predictions_and_the_matrix() -> None:
    # R: predict(res, vcov = TRUE) is list(pred = <predict.rma>, vcov = <matrix>), so $pred$pred are the numbers
    out = metafopy.predict(metafopy.rma(yi='yi', vi='vi', data=bcg_frame()), vcov=True)
    found, inner = audit.valuetree.lookup(out, 'pred')
    assert found
    found, numbers = audit.valuetree.lookup(inner, 'pred')
    assert found
    assert isinstance(numbers, numpy.ndarray)
    assert audit.valuetree.lookup(out, 'vcov')[1] is out.vcov


def test_assigning_c_of_yi_removes_the_escalc_attributes() -> None:
    # R 4.3.3, metafor 5.2-1: after dat$yi <- c(dat$yi), regtest(yi, vi, ni = nitotal, data = dat)$fit$measure is "GEN"
    data = bcg_frame()
    plain = audit.rlib.r_assign_dollar(data, 'yi', audit.rlib.r_c(data['yi']))
    assert 'measure' not in plain.attrs
    assert data.attrs['measure'] == 'RR'
    kept = audit.rlib.r_assign_dollar(data, 'yi', data['yi'].to_numpy() * 2)
    assert kept.attrs['measure'] == 'RR'


def test_cumul_na_exclude_keeps_a_row_for_each_study_left_out() -> None:
    # R 4.3.3, metafor 5.2-1: options(na.action = "na.exclude"); cumul(rma(yi, vi, data = dat)) with studies 3 and 7 missing has 13 rows, NA at 3 and 7
    data = bcg_frame()
    data.loc[[2, 6], ['yi', 'vi']] = numpy.nan
    fit = metafopy.rma(yi='yi', vi='vi', data=data)
    out = metafopy.cumul(fit, na_action='exclude')
    assert out.order is not None
    assert len(out.estimate) == len(out.slab) == len(out.order) == 13
    assert numpy.flatnonzero(numpy.isnan(out.estimate)).tolist() == [2, 6]
    assert numpy.round(out.estimate[[0, 1, 3, 12]], 3).tolist() == [-0.889, -1.325, -1.425, -0.622]
    assert len(metafopy.cumul(fit).estimate) == 11
    year = data['year'].to_numpy()
    ordered = metafopy.cumul(fit, order=year, na_action='exclude')
    assert numpy.flatnonzero(numpy.isnan(ordered.estimate)).tolist() == [3, 8]
    grouped = metafopy.cumul(fit, order=year, collapse=True, na_action='exclude')
    assert len(grouped.estimate) == 12
    assert numpy.flatnonzero(numpy.isnan(grouped.estimate)).tolist() == [3]


def test_print_num_numbers_the_coefficient_rows() -> None:
    # R 4.3.3, metafor 5.2-1: print(rma(yi, vi, mods = ~ ablat, data = dat), num = TRUE) has the rows "1) intrcpt" and "2) ablat"
    fit = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=bcg_frame())
    shown = fit.summary(showfit=False, num=True).split('\n')
    assert any(line.startswith('1) intrcpt') for line in shown)
    assert any(line.startswith('2) ablat') for line in shown)
    assert not any(line.startswith('1) intrcpt') for line in fit.summary(showfit=False).split('\n'))
    permuted: typing.Any = metafopy.permutest(fit, iter=20, seed=1)
    assert any(line.startswith('2) ablat') for line in permuted.summary(num=True).split('\n'))


def test_permutest_print_keeps_the_footnote_without_the_codes() -> None:
    # R 4.3.3, metafor 5.2-1: print(permutest(res), signif.stars = FALSE) ends "---", "1) p-values based on permutation testing" and no codes
    fit = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=bcg_frame())
    permuted: typing.Any = metafopy.permutest(fit, iter=20, seed=1)
    for kwargs in ({'signif_stars': False}, {'signif_legend': False}):
        lines = permuted.summary(**kwargs).split('\n')
        assert lines[-4:] == ['', '---', '1) p-values based on permutation testing', '']
        assert not any(line.startswith('Signif. codes') for line in lines)


def test_sapply_simplifies_to_a_matrix_with_a_column_per_element() -> None:
    # R: t(sapply(list(1, 2, 3, 4, 5), function(x) c(x, 2 * x, 3 * x, 4 * x, 5 * x, 6 * x))) has 5 rows and 6 columns
    out = audit.rlib.r_sapply([1, 2, 3, 4, 5], lambda x: numpy.arange(1, 7) * x)
    assert out.shape == (6, 5)
    assert audit.rlib.r_t(out).shape == (5, 6)
    assert audit.rlib.r_sapply([1, 2, 3], lambda x: x * 2).shape == (3,)


def test_print_arguments_reach_the_summary_and_unlist_reads_a_data_frame(capsys: pytest.CaptureFixture[str]) -> None:
    # R: print(influence(res), infonly = TRUE) prints the influential studies only; unlist() of a data frame is its values
    fit = metafopy.rma(yi='yi', vi='vi', data=bcg_frame(), method='EE')
    inf = metafopy.influence(fit)
    audit.rlib.r_print(inf)
    everyone = capsys.readouterr().out.strip().split('\n')
    audit.rlib.r_print(inf, infonly=True)
    only = capsys.readouterr().out.strip().split('\n')
    assert len(everyone) == 14
    assert len(only) < len(everyone)
    assert audit.rlib.r_unlist(pandas.DataFrame({'a': [1.0], 'b': [2.0]})).tolist() == [1.0, 2.0]


def test_factor_of_numbers_has_labels_without_decimals() -> None:
    # R: levels(factor(c(1, 0, NA, 1))) is "0" "1", not "0.0" "1.0"
    out = audit.rlib.r_factor(numpy.array([1.0, 0.0, numpy.nan, 1.0]))
    assert list(out.categories) == ['0', '1']
    assert out.tolist()[:2] == ['1', '0']


def test_round_of_a_data_frame_and_its_dotted_columns() -> None:
    # R: round(dat2[, c("or", "ci.lb")], digits = 2) is a data frame and dat2[c("yi", "ci.lb")] <- ... replaces the column that metafopy calls ci_lb
    frame = pandas.DataFrame({'or': [1.234, 2.345], 'ci_lb': [0.1234, 0.5678]})
    assert isinstance(audit.rlib.r_round(frame, 2), pandas.DataFrame)
    assigned = audit.rlib.r_assign_index(frame, pandas.DataFrame({'ci.lb': [7.0, 8.0]}), numpy.array(['ci.lb'], dtype=object))
    assert list(assigned.columns) == ['or', 'ci_lb']
    assert assigned['ci_lb'].tolist() == [7.0, 8.0]
    scope = audit.rlib.Env()
    scope.scopes.append(frame)
    assert audit.rlib.r_symbol(scope, 'ci.lb') == 'ci_lb'


def test_club_sandwich_ddf_is_read_from_dfs() -> None:
    # R 4.3.3, metafor 5.2-1: robust(rma(yi, vi, mods = ~ ablat), cluster = trial, clubSandwich = TRUE)$ddf is c(3.705255, 5.212865), as dfs
    data = bcg_frame()
    fit = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', data=data)
    robust = metafopy.robust(fit, cluster=data['trial'], clubSandwich=True)
    found, ddf = audit.valuetree.lookup(robust, 'ddf')
    assert found
    assert numpy.round(ddf, 6).tolist() == [3.705255, 5.212865]


def test_summary_digits_are_those_it_printed_with() -> None:
    # R 4.3.3, metafor 5.2-1: summary(rma(yi, vi, data = dat), digits = 2)$digits is 2 for every element (4 for the fit itself)
    fit = metafopy.rma(yi='yi', vi='vi', data=bcg_frame())
    shown: typing.Any = fit.summary(digits=2)
    assert shown.digits == 2
    plain: typing.Any = fit.summary()
    assert plain.digits == 4
    assert fit.digits == 4


def test_formula_names_outside_the_data_are_found_in_the_script() -> None:
    # R: rma(yi, vi, mods = ~ alloc.random + year, data = dat) finds alloc.random in the calling environment
    data = bcg_frame()
    env = audit.rlib.Env()
    env.vars['alloc.random'] = (data['alloc'] == 'random').to_numpy().astype(float)
    out = audit.rlib.formula_data(env, metafopy.rma, {'mods': '~ alloc.random + year', 'data': data})
    assert 'alloc.random' in out['data'].columns
    assert out['data'].attrs == data.attrs
    assert 'alloc.random' not in data.columns
