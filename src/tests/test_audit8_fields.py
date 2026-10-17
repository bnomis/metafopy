"""Audit 8: the bookkeeping fields of results that metafor 5.2-1 stores (``ids``, ``*_f``, ``not_na``, ``dfs``, ...) and the printouts that
the audit found different. The expected printouts are those of R metafor 5.2-1 (the ``matrix/print`` cases of the audit)."""

import math

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.methods.accessors
import metafopy.methods.permutest
import metafopy.printing


@pytest.fixture(autouse=True)
def console_80() -> None:
    metafopy.printing.SETTINGS['width'] = 80


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def dat_na(dat: pandas.DataFrame) -> pandas.DataFrame:
    out = dat.copy()
    out.loc[[2, 5], 'yi'] = numpy.nan
    return out


def test_uni_fields_with_missing_values(dat_na: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat_na)
    assert res.k == 11
    # R keeps the studies with missing values in ids, yi.f, vi.f, X.f and tau2.f, and flags the ones used in not.na
    assert len(res.ids) == 13
    assert res.ids[0] == 0
    assert res.ids[-1] == 12
    assert int(res.not_na.sum()) == 11
    assert res.yi_f.shape == (13,)
    assert int(numpy.isnan(res.yi_f).sum()) == 2
    assert res.vi_f.shape == (13,)
    assert res.k_f == 13
    assert res.X_f.shape == (13, 2)
    assert res.slab_null is True
    assert res.subset is None
    assert res.intercept is True
    assert res.model == 'rma.uni'
    assert res.coef_na == (False, False)
    assert set(res.outdat_f) >= {'ai', 'bi', 'ci', 'di'} or all(v is None for v in res.outdat_f.values())
    assert res.add == 0.5
    assert res.to == 'only0'
    assert res.drop00 is False


def test_dfs_equals_ddf_when_not_robust(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, test='knha')
    assert res.dfs == res.ddf == 11


def test_fit_stats_is_the_table_of_r(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat, method='REML')
    table = res.fit_stats
    assert list(table.index) == ['ll', 'dev', 'AIC', 'BIC', 'AICc']
    assert list(table.columns) == ['ML', 'REML']
    assert table.loc['ll', 'REML'] == pytest.approx(res.fit_stats_for(True).ll)


def test_counts_of_tables(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    assert len(res.ids) == 13
    assert res.yi_f.shape == (13,)
    assert res.ni_f.shape == (13,)
    assert res.slab_null is True
    assert set(res.outdat) >= {'ai', 'bi', 'ci', 'di'}
    assert res.btt == (0,)
    assert res.tau2_f == 0.0


def test_glmm_fields(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, measure='OR', model='UM.FS')
    assert res.dfs == res.ddf
    assert res.control is not None
    assert len(res.ids) == 13


def test_mv_fields(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_mv(yi='yi', V='vi', random='~ 1 | trial', data=dat)
    assert res.weighted is True
    assert res.yi_f.shape == (13,)
    assert res.V_f is not None
    assert res.V_f.shape == (13, 13)
    assert res.slab_null is True
    assert res.subset is None
    assert res.withS is True
    assert res.withG is False


def test_mv_without_sigma2_has_one_rfix(dat: pandas.DataFrame) -> None:
    # R stores Rfix = FALSE (a vector of length one) when the model has no sigma^2 terms
    res = metafopy.rma_mv(yi='yi', V='vi', random='~ year + ablat | alloc', struct='GEN', data=dat)
    assert res.Rfix == (False,)


def test_ls_fields(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(yi='yi', vi='vi', mods='~ ablat', scale='~ year', data=dat)
    assert res.model == 'rma.ls'
    assert res.randhet is False
    assert res.R2 is None
    assert math.isnan(res.se_tau2)
    assert res.Z_intercept is True
    assert res.tau2_f.shape == (13,)
    assert res.dfs == res.ddf


def test_lrt_and_wald_fields(dat: pandas.DataFrame) -> None:
    full = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, method='ML')
    reduced = metafopy.rma('yi', 'vi', data=dat, method='ML')
    lrt = metafopy.anova(reduced, full)
    assert lrt.type == 'LRT'
    assert lrt.tau2_f == pytest.approx(full.tau2)
    assert lrt.tau2_r == pytest.approx(reduced.tau2)
    wald = metafopy.anova(full, btt=1)
    assert wald.type == 'Wald.btt'
    assert wald.k == 13
    assert wald.p == 2


def test_influence_keeps_ids_and_not_na(dat_na: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat_na)
    infl = metafopy.influence(res)
    assert len(infl.ids) == 13
    assert int(infl.not_na.sum()) == 11


def test_permutest_aliases(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    perm = metafopy.permutest(res, iter=20, seed=1)
    assert isinstance(perm, metafopy.methods.permutest.PermutestResult)
    assert perm.dfs == perm.ddf
    assert perm.skip_beta is False


def test_tau2_zero_prints_without_decimals(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat, method='REML', tau2=0)
    lines = str(res).splitlines()
    assert 'tau^2 (specified amount of total heterogeneity): 0 (SE = 0.0061)' in lines
    assert any(line.startswith('tau (square root of specified tau^2 value):') and line.endswith(' 0') for line in lines)


def test_fmt_table_wraps_wide_tables_like_r() -> None:
    header = ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub', '']
    rows = [
        ['-40.2316', '71.7476', '-0.5607', '0.5750', '-180.8544', '100.3912', ''],
        ['5.7782', '10.2575', '0.5633', '0.5732', '-14.3262', '25.8826', ''],
    ]
    names = ['intrcpt', 'log(year - 1900 + 1000)']
    out = metafopy.printing.fmt_table(header, rows, row_names=names, width=80).splitlines()
    # the last column goes to a second block with the row names, as in R
    assert out[0].rstrip().endswith('ci.lb')
    assert out[3].strip() == 'ci.ub'
    assert out[4].startswith('intrcpt')
    assert out[4].rstrip().endswith('100.3912')
    assert metafopy.printing.fmt_table(header[:3], [r[:3] for r in rows], row_names=names, width=80).count('\n') == 2


def test_predict_cumul_profile_fields(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    pred = metafopy.predict(res, newmods=[10, 20], transf=numpy.exp)
    assert pred.method == 'REML'
    assert pred.transf is True
    assert pred.pred_type == 'location'
    plain = metafopy.rma('yi', 'vi', data=dat)
    cum = metafopy.cumul(plain)
    assert numpy.array_equal(cum.zval, cum.stat)
    assert cum.test == 'z'
    prof = metafopy.profile(plain, steps=5)
    assert prof.comps == 1


def test_trimfill_marks_slab_and_ids(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat, method='FE')
    tf = metafopy.trimfill(res)
    assert tf.slab_null is False
    assert len(tf.ids) == res.k + tf.k0
    assert list(tf.ids[res.k :]) == list(range(res.k, res.k + tf.k0))


def test_glmm_outdat_f_has_all_names(dat: pandas.DataFrame) -> None:
    res = metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, measure='OR', model='UM.FS')
    assert list(res.outdat_f) == ['ai', 'bi', 'ci', 'di', 'x1i', 'x2i', 't1i', 't2i', 'xi', 'mi', 'ni', 'ti']
    assert res.outdat_f['x1i'] is None


def test_round2_influence_frames_and_cumul_ids(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=dat)
    infl = metafopy.influence(res)
    assert isinstance(infl.inf, pandas.DataFrame)
    assert isinstance(infl.dfbs, pandas.DataFrame)
    assert infl.dfbs.shape == (13, 1)
    assert len(metafopy.cumul(res).ids) == 13


def test_round2_loglik_prints_like_r(dat: pandas.DataFrame) -> None:
    ll = metafopy.rma('yi', 'vi', data=dat, method='ML').logLik()
    assert math.isfinite(float(ll))
    assert str(ll).startswith("'log Lik.'")
    assert str(ll).endswith('(df=2)')


def test_round2_formula_deparse(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat+year', data=dat)
    assert metafopy.methods.accessors.formula_of(res) == '~ablat + year'
