"""emmprep() and the emmeans functions against R's emmeans package (src/tests/r/generate_emmeans.R)."""

import math

import emmcheck
import numpy
import pandas
import pytest

import metafopy
import metafopy.errors


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    bcg = metafopy.datasets.load('bcg')
    out = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    out['lat'] = numpy.where(out['ablat'] >= 33, 'high', 'low')
    return out


@pytest.fixture(scope='module')
def res(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat)


@pytest.fixture(scope='module')
def grid(res: metafopy.RMAResult) -> metafopy.EmmGrid:
    return metafopy.emmprep(res)


@pytest.fixture(scope='module')
def alloc(grid: metafopy.EmmGrid) -> metafopy.EmmGrid:
    return metafopy.emmeans(grid, 'alloc')


def test_reference_grid(grid: metafopy.EmmGrid) -> None:
    assert list(grid.grid.columns) == ['alloc', 'ablat']
    assert grid.grid['alloc'].tolist() == ['alternate', 'random', 'systematic']
    numpy.testing.assert_allclose(grid.grid['ablat'], [33.46153846153846] * 3)
    numpy.testing.assert_allclose(grid.linfct[:, 3], 33.46153846153846)
    assert grid.wgt is not None
    assert grid.wgt.tolist() == [2, 7, 4]
    assert math.isinf(grid.df)
    assert grid.link is not None
    assert grid.link.name == 'log'


def test_marginal_means(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(alloc.summary(infer=True), 'uni_alloc')
    emmcheck.check(alloc.summary(infer=True, type='response'), 'uni_alloc_resp')
    emmcheck.check(alloc.summary(infer=True, level=0.9), 'uni_alloc_level90')
    emmcheck.check(alloc.summary(infer=True, null=-0.5), 'uni_alloc_null')
    emmcheck.check(alloc.summary(infer=True, type='response', bias_adjust=True), 'uni_alloc_bias')


@pytest.mark.parametrize(
    ('adjust', 'key'),
    [
        (None, 'uni_pairs'),
        ('bonferroni', 'uni_pairs_bonferroni'),
        ('holm', 'uni_pairs_holm'),
        ('hochberg', 'uni_pairs_hochberg'),
        ('hommel', 'uni_pairs_hommel'),
        ('BH', 'uni_pairs_BH'),
        ('BY', 'uni_pairs_BY'),
        ('sidak', 'uni_pairs_sidak'),
        ('scheffe', 'uni_pairs_scheffe'),
        ('none', 'uni_pairs_none'),
    ],
)
def test_pairwise_adjustments(alloc: metafopy.EmmGrid, adjust: str | None, key: str) -> None:
    emmcheck.check(metafopy.pairs(alloc).summary(infer=True, adjust=adjust), key)


def test_pairwise_response_and_reverse(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(metafopy.pairs(alloc).summary(infer=True, type='response'), 'uni_pairs_resp')
    emmcheck.check(metafopy.pairs(alloc).summary(infer=True, adjust='bonferroni', type='response'), 'uni_pairs_ci_bonf_resp')
    emmcheck.check(metafopy.pairs(alloc, reverse=True).summary(infer=True), 'uni_revpairs')


def test_mvt_adjustment(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(metafopy.pairs(alloc).summary(infer=True, adjust='mvt'), 'uni_pairs_mvt', emmcheck.MVT)
    emmcheck.check(metafopy.contrast(alloc, 'consec').summary(infer=True), 'uni_consec_mvt', emmcheck.MVT)


def test_contrast_methods(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(metafopy.contrast(alloc, 'trt_vs_ctrl').summary(infer=True), 'uni_trt_ctrl')
    emmcheck.check(metafopy.contrast(alloc, 'trt_vs_ctrl', ref=1).summary(infer=True), 'uni_trt_ctrl_ref2')
    emmcheck.check(metafopy.contrast(alloc, 'trt_vs_ctrlk').summary(infer=True), 'uni_trt_ctrlk')
    emmcheck.check(metafopy.contrast(alloc, 'dunnett').summary(infer=True, type='response'), 'uni_dunnett_resp')
    emmcheck.check(metafopy.contrast(alloc, 'eff').summary(infer=True), 'uni_eff')
    emmcheck.check(metafopy.contrast(alloc, 'del_eff').summary(infer=True), 'uni_del_eff')
    emmcheck.check(metafopy.contrast(alloc, 'consec', adjust='bonferroni').summary(infer=True), 'uni_consec')
    emmcheck.check(metafopy.contrast(alloc, 'poly').summary(infer=True), 'uni_poly')
    emmcheck.check(metafopy.contrast(alloc, 'identity').summary(infer=True), 'uni_identity')
    emmcheck.check(metafopy.contrast(alloc, 'pairwise', exclude=2).summary(infer=True), 'uni_excl')
    custom: dict[str, list[float]] = {'rnd vs others': [-0.5, 1, -0.5], 'alt vs sys': [1, 0, -1]}
    emmcheck.check(metafopy.contrast(alloc, custom).summary(infer=True), 'uni_custom')


def test_covariates_and_at(res: metafopy.RMAResult, grid: metafopy.EmmGrid) -> None:
    emmcheck.check(metafopy.emmeans(grid, 'ablat').summary(infer=True), 'uni_ablat')
    emmcheck.check(metafopy.emmeans(grid, 'ablat', weights='proportional').summary(infer=True), 'uni_ablat_prop')
    emmcheck.check(metafopy.emmeans(grid, '1').summary(infer=True), 'uni_overall')
    emmcheck.check(metafopy.emmeans(grid, '~ 1').summary(infer=True), 'uni_overall')
    at = metafopy.emmprep(res, at={'ablat': [10, 30, 50]})
    by_alloc = metafopy.emmeans(at, 'alloc', by='ablat')
    emmcheck.check(by_alloc.summary(infer=True), 'uni_at_by')
    emmcheck.check(metafopy.pairs(by_alloc).summary(infer=True), 'uni_at_pairs_by')
    emmcheck.check(metafopy.emmeans(at, '~ alloc | ablat').summary(infer=True), 'uni_at_by')
    emmcheck.check(metafopy.emmeans(at, 'ablat').summary(infer=True), 'uni_at_cov')
    emmcheck.check(metafopy.emmeans(at, 'ablat', by='alloc').summary(infer=True), 'uni_at_cov_by')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(at, 'ablat')).summary(infer=True), 'uni_at_pairs_cov')
    levels = metafopy.emmprep(res, at={'alloc': ['random', 'systematic']})
    emmcheck.check(metafopy.emmeans(levels, 'alloc').summary(infer=True), 'uni_at_levels')
    with pytest.raises(metafopy.errors.InputError, match='not in the model'):
        metafopy.emmprep(res, at={'nothing': [1]})
    with pytest.raises(metafopy.errors.InputError, match='No variable named'):
        metafopy.emmeans(grid, 'nothing')


def test_emmeans_accepts_a_fit(res: metafopy.RMAResult) -> None:
    emmcheck.check(metafopy.emmeans(res, 'alloc').summary(infer=True), 'uni_alloc')
    emmcheck.check(metafopy.emmeans(res, 'ablat', at={'ablat': [10, 30, 50]}).summary(infer=True), 'uni_at_cov')


def test_knapp_hartung_and_t(dat: pandas.DataFrame) -> None:
    knha = metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat, test='knha')
    e = metafopy.emmeans(metafopy.emmprep(knha), 'alloc')
    emmcheck.check(e.summary(infer=True), 'knha_alloc')
    emmcheck.check(metafopy.pairs(e).summary(infer=True), 'knha_pairs')
    emmcheck.check(metafopy.pairs(e).summary(infer=True, type='response'), 'knha_pairs_resp')
    emmcheck.check(metafopy.contrast(e, 'trt_vs_ctrl').summary(infer=True), 'knha_trt_ctrl')
    emmcheck.check(e.summary(infer=True, type='response', bias_adjust=True), 'knha_alloc_bias')
    t = metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat, test='t')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(t), 'alloc').summary(infer=True), 't_alloc')


def test_formula_order_and_terms(dat: pandas.DataFrame) -> None:
    res2 = metafopy.rma_uni('yi', 'vi', mods='~ ablat + alloc', data=dat)
    g2 = metafopy.emmprep(res2)
    assert list(g2.grid.columns) == ['ablat', 'alloc']
    emmcheck.check(metafopy.emmeans(g2, 'alloc').summary(infer=True), 'order_alloc')
    emmcheck.check(metafopy.emmeans(g2, 'ablat').summary(infer=True), 'order_ablat')
    quad = metafopy.rma_uni('yi', 'vi', mods='~ ablat + I(ablat**2)', data=dat)
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(quad, at={'ablat': [10, 30, 50]}), 'ablat').summary(infer=True), 'quad_ablat')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(quad), 'ablat').summary(infer=True), 'quad_mean')


@pytest.mark.parametrize('weights', ['equal', 'proportional', 'outer', 'cells', 'flat'])
def test_weights(dat: pandas.DataFrame, weights: str) -> None:
    two = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + lat', data=dat))
    emmcheck.check(metafopy.emmeans(two, 'alloc', weights=weights).summary(infer=True), f'two_alloc_{weights}')


def test_two_factors(dat: pandas.DataFrame) -> None:
    two = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + lat', data=dat))
    emmcheck.check(metafopy.emmeans(two, 'lat').summary(infer=True), 'two_lat')
    both = metafopy.emmeans(two, ['alloc', 'lat'])
    emmcheck.check(both.summary(infer=True), 'two_both')
    emmcheck.check(metafopy.pairs(both).summary(infer=True), 'two_both_pairs')
    emmcheck.check(metafopy.emmeans(two, '~ alloc * lat').summary(infer=True), 'two_both')
    by = metafopy.emmeans(two, 'alloc', by='lat')
    emmcheck.check(by.summary(infer=True), 'two_by')
    emmcheck.check(metafopy.pairs(by).summary(infer=True), 'two_by_pairs')
    emmcheck.check(metafopy.pairs(by).summary(infer=True, adjust='holm'), 'two_by_pairs_holm')
    emmcheck.check(metafopy.pairs(by).summary(infer=True, adjust='none', cross_adjust='bonferroni'), 'two_by_pairs_cross')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(two, 'lat', by='alloc')).summary(infer=True), 'two_lat_by_alloc_pairs')
    emmcheck.check(metafopy.emmeans(two, 'alloc', weights=[1, 3]).summary(infer=True), 'two_numw')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(two, 'alloc', weights='proportional')).summary(infer=True), 'two_pairs_prop')
    emmcheck.check(metafopy.contrast(by, 'trt_vs_ctrl').summary(infer=True), 'two_contrast_by')
    with pytest.raises(metafopy.errors.InputError, match='Nonconforming number of weights'):
        metafopy.emmeans(two, 'alloc', weights=[1, 2, 3])
    with pytest.raises(metafopy.errors.InputError, match="Invalid 'weights'"):
        metafopy.emmeans(two, 'alloc', weights='nonsense')


def test_interactions(dat: pandas.DataFrame) -> None:
    int4 = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc * lat', data=dat))
    emmcheck.check(metafopy.emmeans(int4, 'alloc', by='lat').summary(infer=True), 'int_by')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(int4, 'alloc', by='lat')).summary(infer=True), 'int_by_pairs')
    both = metafopy.emmeans(int4, ['alloc', 'lat'])
    emmcheck.check(metafopy.pairs(both).summary(infer=True), 'int_both_pairs')
    emmcheck.check(metafopy.contrast(both, 'eff').summary(infer=True), 'int_both_eff')
    emmcheck.check(metafopy.emmeans(int4, 'alloc').summary(infer=True), 'int_alloc')
    emmcheck.check(metafopy.emmeans(int4, 'alloc', weights='proportional').summary(infer=True), 'int_alloc_prop')
    emmcheck.check(metafopy.contrast(both, 'pairwise', interaction=True).summary(infer=True), 'int_interaction')
    emmcheck.check(metafopy.contrast(both, interaction=['pairwise', 'pairwise']).summary(infer=True), 'int_interaction')
    slopes = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc * ablat', data=dat), at={'ablat': [15, 40]})
    emmcheck.check(metafopy.emmeans(slopes, 'alloc', by='ablat').summary(infer=True), 'slopes_by')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(slopes, 'alloc', by='ablat')).summary(infer=True), 'slopes_pairs')


def test_regrid(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(metafopy.regrid(alloc).summary(infer=True), 'regrid')
    emmcheck.check(metafopy.pairs(metafopy.regrid(alloc)).summary(infer=True), 'regrid_pairs')
    emmcheck.check(metafopy.regrid(alloc, bias_adjust=True).summary(infer=True), 'regrid_bias')


def test_fit_variants(dat: pandas.DataFrame, res: metafopy.RMAResult) -> None:
    fe = metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=dat, method='FE')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(fe), 'alloc').summary(infer=True, type='response'), 'fe_alloc')
    dl = metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=dat, method='DL', weighted=False)
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(dl), 'alloc').summary(infer=True), 'dl_alloc')
    res0 = metafopy.rma_uni('yi', 'vi', data=dat)
    overall = metafopy.emmeans(metafopy.emmprep(res0), '1')
    emmcheck.check(overall.summary(infer=True, type='response'), 'int_only')
    emmcheck.check(overall.summary(infer=True, type='response', bias_adjust=True), 'int_only_bias')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(res, df=7), 'alloc').summary(infer=True), 'df_override')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(res, sigma=0.5), 'alloc').summary(infer=True, type='response', bias_adjust=True), 'sigma_override')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(res, tran='sqrt'), 'alloc').summary(infer=True, type='response'), 'tran_override')
    robust = metafopy.robust(res, cluster='trial')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(robust), 'alloc').summary(infer=True), 'robust')


def test_transformations_by_measure() -> None:
    bcg = metafopy.datasets.load('bcg')
    n = bcg['tpos'] + bcg['tneg']
    plo = metafopy.escalc(measure='PLO', xi=bcg['tpos'], ni=n)
    plo['alloc'] = bcg['alloc']
    e = metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=plo)), 'alloc')
    emmcheck.check(e.summary(infer=True, type='response'), 'plo')
    emmcheck.check(metafopy.pairs(e).summary(infer=True, type='response'), 'plo_pairs')
    pas = metafopy.escalc(measure='PAS', xi=bcg['tpos'], ni=n)
    pas['alloc'] = bcg['alloc']
    emmcheck.check(
        metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=pas)), 'alloc').summary(infer=True, type='response'), 'pas'
    )
    irs = metafopy.escalc(measure='IRS', xi=bcg['tpos'], ti=n + 10)
    irs['alloc'] = bcg['alloc']
    emmcheck.check(
        metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=irs)), 'alloc').summary(infer=True, type='response'), 'irs'
    )
    bonett = pandas.DataFrame(
        {
            'ni': [103, 64, 118, 401, 215, 51, 175, 311, 492],
            'ai': [0.93, 0.91, 0.94, 0.89, 0.90, 0.89, 0.86, 0.91, 0.90],
            'mi': [20] * 9,
            'caregivers': ['women', 'mixed', 'women', 'mixed', 'mixed', 'mixed', 'mixed', 'mixed', 'women'],
        }
    )
    for measure, key in (('AHW', 'ahw'), ('ABT', 'abt')):
        out = metafopy.escalc(measure=measure, ai=bonett['ai'], mi=bonett['mi'], ni=bonett['ni'])
        out['caregivers'] = bonett['caregivers']
        fit = metafopy.rma_uni('yi', 'vi', mods='~ caregivers', data=out)
        emmcheck.check(metafopy.emmeans(metafopy.emmprep(fit), 'caregivers').summary(infer=True, type='response'), key)
    molloy = metafopy.datasets.load('molloy2014')
    zcor = metafopy.escalc(measure='ZCOR', ri=molloy['ri'], ni=molloy['ni'])
    zcor['design'] = molloy['design']
    ez = metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ design', data=zcor)), 'design')
    emmcheck.check(ez.summary(infer=True, type='response'), 'zcor')
    emmcheck.check(ez.summary(infer=True), 'zcor_link')
    emmcheck.check(metafopy.pairs(ez).summary(infer=True), 'zcor_pairs')
    orr = metafopy.escalc(measure='OR', ai=bcg['tpos'], bi=bcg['tneg'], ci=bcg['cpos'], di=bcg['cneg'])
    orr['alloc'] = bcg['alloc']
    eor = metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=orr)), 'alloc')
    emmcheck.check(metafopy.pairs(eor).summary(infer=True, type='response'), 'or_ratio')


def test_subset_and_missing(dat: pandas.DataFrame) -> None:
    na = dat.copy()
    na.loc[2, 'ablat'] = numpy.nan
    fit = metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=na, subset=na['year'] > 1950)
    e = metafopy.emmeans(metafopy.emmprep(fit), 'alloc')
    emmcheck.check(e.summary(infer=True), 'subset_na')
    emmcheck.check(metafopy.pairs(e).summary(infer=True), 'subset_na_pairs')


def test_rma_mv() -> None:
    konst = metafopy.datasets.load('konstantopoulos2011')
    years = {'year': [1975, 1990, 2005]}
    mv = metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=konst)
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(mv, at=years), 'year').summary(infer=True), 'mv_year', emmcheck.OPTIM)
    mvt = metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=konst, test='t')
    e = metafopy.emmeans(metafopy.emmprep(mvt, at=years), 'year')
    emmcheck.check(e.summary(infer=True), 'mv_year_t', emmcheck.OPTIM)
    emmcheck.check(metafopy.pairs(e).summary(infer=True), 'mv_year_t_pairs', emmcheck.OPTIM)
    berkey = metafopy.datasets.load('berkey1998')
    v = metafopy.bldiag([berkey.loc[berkey['trial'] == t, ['v1i', 'v2i']].to_numpy() for t in sorted(berkey['trial'].unique())])
    un = metafopy.rma_mv('yi', v, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', data=berkey)
    eun = metafopy.emmeans(metafopy.emmprep(un), 'outcome')
    emmcheck.check(eun.summary(infer=True), 'mv_un', emmcheck.OPTIM)
    emmcheck.check(metafopy.pairs(eun).summary(infer=True), 'mv_un_pairs', emmcheck.OPTIM)
    un2 = metafopy.rma_mv('yi', v, mods='~ outcome', random='~ outcome | trial', struct='UN', data=berkey)
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(un2), 'outcome').summary(infer=True), 'mv_un_treat', emmcheck.OPTIM)
    unt = metafopy.rma_mv('yi', v, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', data=berkey, test='t')
    emmcheck.check(metafopy.emmeans(metafopy.emmprep(unt), 'outcome').summary(infer=True), 'mv_un_t', emmcheck.OPTIM)


def test_glmm_and_location_scale(dat: pandas.DataFrame) -> None:
    bcg = metafopy.datasets.load('bcg')
    gl = metafopy.rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, mods='~ alloc')
    egl = metafopy.emmeans(metafopy.emmprep(gl), 'alloc')
    emmcheck.check(egl.summary(infer=True), 'glmm', emmcheck.GLMM, emmcheck.GLMM_ATOL)
    emmcheck.check(egl.summary(infer=True, type='response'), 'glmm_resp', emmcheck.GLMM, emmcheck.GLMM_ATOL)
    # the z ratio of a contrast near zero moves by up to 2e-3 absolute between platforms (the lme4-style fit stops at a slightly different point on Windows)
    emmcheck.check(metafopy.pairs(egl).summary(infer=True), 'glmm_pairs', emmcheck.GLMM, 5 * emmcheck.GLMM_ATOL)
    ls = metafopy.rma_uni('yi', 'vi', mods='~ alloc', scale='~ ablat', data=dat)
    els = metafopy.emmeans(metafopy.emmprep(ls), 'alloc')
    emmcheck.check(els.summary(infer=True), 'ls', emmcheck.OPTIM)
    emmcheck.check(metafopy.pairs(els).summary(infer=True), 'ls_pairs', emmcheck.OPTIM)
    emmcheck.check(els.summary(infer=True, type='response'), 'ls_resp', emmcheck.OPTIM)


def test_binary_and_selection_fits() -> None:
    bcg = metafopy.datasets.load('bcg')
    mh = metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    out = metafopy.emmeans(metafopy.emmprep(mh), '1').summary(infer=True, type='response')
    numpy.testing.assert_allclose(out.frame['response'].iloc[0], math.exp(mh.beta[0]))
    numpy.testing.assert_allclose(out.frame['SE'].iloc[0], math.exp(mh.beta[0]) * mh.se[0])
    peto = metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    numpy.testing.assert_allclose(metafopy.emmeans(metafopy.emmprep(peto), '1').summary(type='response').frame['response'].iloc[0], math.exp(peto.beta[0]))


def test_errors_and_printing(dat: pandas.DataFrame, alloc: metafopy.EmmGrid) -> None:
    # emmeans takes an unknown name for the identity link carrying that name; '+.sqrt' (which needs an offset) is not supported
    with pytest.raises(metafopy.errors.NotSupportedError, match=r'\+\.sqrt'):
        metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=dat), tran='+.sqrt')
    with pytest.raises(metafopy.errors.InputError, match='not found'):
        metafopy.contrast(alloc, 'nonsense')
    with pytest.raises(metafopy.errors.InputError, match="'type' must be"):
        alloc.summary(type='nonsense')
    with pytest.raises(metafopy.errors.InputError, match='not recognized'):
        metafopy.pairs(alloc).summary(adjust='nonsense')
    with pytest.raises(metafopy.errors.InputError, match='Nonconforming'):
        metafopy.contrast(alloc, {'bad': [1, -1]})
    plain = metafopy.rma_uni('yi', 'vi', mods=dat[['ablat']].to_numpy(), data=dat)
    with pytest.raises(metafopy.errors.InputError, match='without a formula'):
        metafopy.emmprep(plain)
    text = str(alloc)
    assert 'emmean' in text
    assert 'Confidence level used: 0.95' in text
    two = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + lat', data=dat))
    assert 'Results are averaged over the levels of: lat' in str(metafopy.emmeans(two, 'alloc'))
    assert metafopy.pairs(alloc).confint().frame.shape[1] == 6
    assert 'p.value' in metafopy.pairs(alloc).test().frame.columns


def test_regrid_of_log_contrast_labels_as_ratios(alloc: metafopy.EmmGrid) -> None:
    pairs = metafopy.pairs(alloc)
    labels = metafopy.regrid(pairs).grid['contrast'].tolist()
    assert labels == metafopy.pairs(alloc).summary(type='response').frame['contrast'].tolist()
    assert all(' / ' in label for label in labels)


def test_r2_to_z_link_name() -> None:
    assert metafopy.methods.emm_links.make_link('r2-to-z').name == 'r2-to-z'
