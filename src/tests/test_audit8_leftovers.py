"""Audit 8, leftovers of the final run: ``regtest()`` refits, skipped parts of location-scale ``permutest()``, ``sandwiches`` and the names of
``dfbetas()`` of ``rma.mv`` fits. The expected values are those of R metafor 5.2-1 (R 4.3.3)."""

import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def test_regtest_of_trimfill_numbers_the_studies_and_has_slab(dat: pandas.DataFrame) -> None:
    # R: regtest(trimfill(rma(yi, vi, data = dat)))$fit has slab 1..14 and slab.null FALSE, with or without labels given
    labelled = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'), slab=dat['author'])
    for frame in (dat, labelled):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = metafopy.regtest(metafopy.trimfill(metafopy.rma('yi', 'vi', data=frame))).fit
        assert fit is not None
        assert fit.slab_null is False
        assert fit.slab == tuple(str(i + 1) for i in range(14))


def test_regtest_refit_keeps_row_names_of_the_moderator_matrix(dat: pandas.DataFrame) -> None:
    # R: with studies 3 and 5 dropped, rownames(regtest(rma(yi, vi, mods = ~ ablat, data = dat))$fit$X) skips them; an intercept-only model has none
    frame = dat.copy()
    frame.loc[2, 'yi'] = numpy.nan
    frame.loc[4, 'ablat'] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.regtest(metafopy.rma('yi', 'vi', mods='~ ablat', data=frame)).fit
    assert fit is not None
    assert fit.data.rownames[:4] == ('1', '2', '4', '6')
    assert fit.slab_null is True


def test_skipped_parts_of_location_scale_permutest_have_one_missing_value() -> None:
    # R: permutest(rma(yi, vi, mods = ~ ablat, scale = ~ ablat), skip.alpha = TRUE)$alpha.perm is a data frame with the column alpha.perm and one NA
    frame = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=frame)
        no_alpha = typing.cast(typing.Any, metafopy.permutest(res, iter=20, seed=1, skip_alpha=True))
        no_beta = typing.cast(typing.Any, metafopy.permutest(res, iter=20, seed=1, skip_beta=True))
    assert list(no_alpha.alpha_perm.columns) == ['alpha.perm']
    assert list(no_alpha.zval_alpha_perm.columns) == ['zval.alpha.perm']
    assert list(no_beta.beta_perm.columns) == ['beta.perm']
    assert list(no_beta.zval_perm.columns) == ['zval.perm']
    assert no_alpha.alpha_perm.shape == (1, 1)
    assert no_alpha.alpha_perm.isna().all().all()
    tables = no_alpha.coef()
    assert list(tables) == ['beta', 'alpha']
    assert list(tables['beta'].columns[:2]) == ['estimate', 'se']


def test_sandwiches_of_club_sandwich_fit(dat: pandas.DataFrame) -> None:
    # R: robust(rma(yi, vi, data = dat), cluster = trial, clubSandwich = TRUE)$sandwiches
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma('yi', 'vi', data=dat)
        robust = metafopy.robust(res, cluster=dat['trial'], club_sandwich=True)
    assert metafopy.rma('yi', 'vi', data=dat).sandwiches is None
    tables = robust.sandwiches
    assert tables is not None
    assert list(tables) == ['coef_test', 'conf_int', 'Wald_test']
    coef = tables['coef_test']
    assert list(coef.columns) == ['Coef', 'beta', 'SE', 'null_value', 'tstat', 'df_Satt', 'p_Satt']
    numpy.testing.assert_allclose(
        coef.loc[0, ['beta', 'SE', 'tstat', 'df_Satt', 'p_Satt']].to_numpy(dtype=float), [-0.715, 0.18, -3.98, 11.04, 0.00216], atol=1e-2
    )
    numpy.testing.assert_allclose(tables['conf_int'].loc[0, ['CI_L', 'CI_U']].to_numpy(dtype=float), [-1.11, -0.319], atol=1e-2)
    assert numpy.isnan(tables['Wald_test'])


def test_dfbetas_of_mv_fit_has_syntactic_names() -> None:
    # R: names(dfbetas(rma.mv(yi, vi, mods = ~ log(year - 1900 + 1000), ...))) is c("intrcpt", "log.year...1900...1000.")
    data = metafopy.datasets.load('dat.konstantopoulos2011')
    res = metafopy.rma_mv('yi', 'vi', mods='~ log(year - 1900 + 1000)', random='~ 1 | district/school', data=data)
    assert list(metafopy.dfbetas(res).columns) == ['intrcpt', 'log.year...1900...1000.']


def test_wald_table_of_club_sandwich_fit_with_moderators(dat: pandas.DataFrame) -> None:
    # R: robust(rma(yi, vi, mods = ~ ablat, data = dat), cluster = trial, clubSandwich = TRUE)$sandwiches$Wald_test is
    # test HTZ, Fstat 29.4, delta 1, df_num 1, df_denom 5.21, p_val 0.00254
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
        tables = metafopy.robust(res, cluster=dat['trial'], club_sandwich=True).sandwiches
    assert tables is not None
    wald = tables['Wald_test']
    assert list(wald.columns) == ['test', 'Fstat', 'delta', 'df_num', 'df_denom', 'p_val']
    numpy.testing.assert_allclose(wald.loc[0, ['Fstat', 'delta', 'df_denom', 'p_val']].to_numpy(dtype=float), [29.4, 1.0, 5.21, 0.00254], rtol=2e-2)


def test_blsplit_keeps_dimnames_through_a_function() -> None:
    # R: blsplit(V, dat$study, round, 4)$`1` of rcalc(ri ~ var1 + var2 | study) has the dimnames acog.perf, asom.perf, ...
    tmp = typing.cast(typing.Any, metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=metafopy.datasets.load('dat.craft2003')))
    block = metafopy.blsplit(tmp.V, tmp.dat['study'], numpy.round, 4)['1']
    assert block.rownames[:3] == ('acog.perf', 'asom.perf', 'conf.perf')
    assert block.colnames[:3] == ('acog.perf', 'asom.perf', 'conf.perf')


def test_mv_known_matrices_have_an_entry_for_every_factor() -> None:
    # R: names(rma.mv(yi, vi, random = list(~ 1 | district, ~ 1 | school), R = list(school = P), data = dat)$R) is c("district", "school"), the first NULL
    data = metafopy.datasets.load('dat.konstantopoulos2011').assign(school=lambda f: f['school'].astype(str))
    levels = [str(v) for v in data['school'].unique()]
    corr = pandas.DataFrame(numpy.eye(len(levels)), index=levels, columns=levels)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_mv('yi', 'vi', random=['~ 1 | district', '~ 1 | school'], R={'school': corr}, data=data)
    assert list(res.R) == ['district', 'school']
    assert res.R['district'] is None
    assert res.R['school'].shape == (len(levels), len(levels))
