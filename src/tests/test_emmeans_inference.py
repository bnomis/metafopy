"""One-sided and equivalence tests (side, delta) and matrix weights in emmeans against R's emmeans package (src/tests/r/generate_emmeans.R)."""

import emmcheck
import numpy
import pandas
import pytest

import metafopy
import metafopy.errors
import metafopy.methods.emm_adjust


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return emmcheck.bcg_lat()


@pytest.fixture(scope='module')
def alloc(dat: pandas.DataFrame) -> metafopy.EmmGrid:
    return metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat)), 'alloc')


@pytest.fixture(scope='module')
def knha(dat: pandas.DataFrame) -> metafopy.EmmGrid:
    return metafopy.emmeans(metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat, test='knha')), 'alloc')


@pytest.fixture(scope='module')
def additive(dat: pandas.DataFrame) -> metafopy.EmmGrid:
    return metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + lat', data=dat))


@pytest.fixture(scope='module')
def crossed(dat: pandas.DataFrame) -> metafopy.EmmGrid:
    return metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc * lat', data=dat))


def test_one_sided_means(alloc: metafopy.EmmGrid) -> None:
    emmcheck.check(alloc.summary(infer=True, side='left'), 'side_left')
    emmcheck.check(alloc.summary(infer=True, side='>'), 'side_right')
    emmcheck.check(alloc.summary(infer=True, side='<', null=-0.5), 'side_left_null')
    emmcheck.check(alloc.summary(infer=True, side='>', type='response'), 'side_right_resp')
    emmcheck.check(alloc.summary(infer=True, side='right', level=0.9), 'side_level90')
    emmcheck.check(alloc.summary(infer=True, side='two-sided'), 'side_both')
    assert numpy.isneginf(alloc.summary(infer=True, side='<').frame['asymp.LCL']).all()
    assert numpy.isposinf(alloc.summary(infer=True, side='>').frame['asymp.UCL']).all()
    assert 'P values are left-tailed' in alloc.summary(infer=True, side='<').messages


def test_one_sided_contrasts(alloc: metafopy.EmmGrid, knha: metafopy.EmmGrid, additive: metafopy.EmmGrid) -> None:
    pairs = metafopy.pairs(alloc)
    emmcheck.check(pairs.summary(infer=True, side='<'), 'side_pairs_left')
    emmcheck.check(pairs.summary(infer=True, side='>', adjust='bonferroni'), 'side_pairs_right_bonf')
    emmcheck.check(pairs.summary(infer=True, side='>', adjust='holm'), 'side_pairs_right_holm')
    emmcheck.check(pairs.summary(infer=True, side='left', adjust='sidak'), 'side_pairs_sidak')
    emmcheck.check(pairs.summary(infer=True, side='left', adjust='scheffe'), 'side_pairs_scheffe')
    emmcheck.check(pairs.summary(infer=True, side='right', adjust='none'), 'side_pairs_none')
    emmcheck.check(pairs.summary(infer=True, side='left', type='response'), 'side_pairs_resp')
    emmcheck.check(pairs.summary(infer=True, side='left', adjust='mvt'), 'side_pairs_mvt', emmcheck.MVT)
    emmcheck.check(metafopy.contrast(alloc, 'trt_vs_ctrl').summary(infer=True, side='>'), 'side_dunnett')
    emmcheck.check(metafopy.contrast(alloc, 'eff').summary(infer=True, side='<'), 'side_eff')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(additive, 'alloc', by='lat')).summary(infer=True, side='>'), 'side_by_pairs')
    emmcheck.check(metafopy.pairs(knha).summary(infer=True, side='<'), 'side_knha_pairs')


def test_tukey_becomes_sidak_for_one_sided_inferences(alloc: metafopy.EmmGrid) -> None:
    summary = metafopy.pairs(alloc).summary(infer=True, side='<')
    assert summary.adjust == 'sidak'
    assert any('not appropriate for one-sided inferences' in m for m in summary.messages)


def test_equivalence_and_noninferiority(alloc: metafopy.EmmGrid, knha: metafopy.EmmGrid) -> None:
    pairs = metafopy.pairs(alloc)
    emmcheck.check(pairs.summary(infer=True, delta=0.5), 'delta_equiv')
    emmcheck.check(pairs.summary(infer=True, delta=0.5, adjust='none'), 'delta_equiv_none')
    emmcheck.check(pairs.summary(infer=True, delta=0.8, adjust='bonferroni'), 'delta_equiv_bonf')
    emmcheck.check(pairs.summary(infer=True, delta=0.3, side='<', adjust='none'), 'delta_nonsup')
    emmcheck.check(pairs.summary(infer=True, delta=0.3, side='>', adjust='none'), 'delta_noninf')
    emmcheck.check(pairs.summary(infer=True, delta=0.3, side='noninferiority', adjust='holm'), 'delta_noninf_holm')
    emmcheck.check(pairs.summary(infer=True, delta=0.5, side='equivalence', adjust='none'), 'delta_equiv_word')
    emmcheck.check(alloc.summary(infer=True, delta=0.4, null=-0.8), 'delta_emm')
    emmcheck.check(alloc.summary(infer=True, delta=0.4, null=-0.8, side='>'), 'delta_emm_right')
    emmcheck.check(alloc.summary(infer=True, delta=0.4, type='response'), 'delta_emm_resp')
    emmcheck.check(metafopy.contrast(alloc, 'trt_vs_ctrl').summary(infer=True, delta=0.5), 'delta_trt')
    emmcheck.check(metafopy.pairs(knha).summary(infer=True, delta=0.5), 'delta_knha')
    messages = pairs.summary(infer=True, delta=0.5).messages
    assert 'Statistics are tests of equivalence with a threshold of 0.5' in messages
    assert 'P values are left-tailed' in messages


def test_options_stored_on_the_grid(dat: pandas.DataFrame, alloc: metafopy.EmmGrid) -> None:
    grid = metafopy.emmprep(metafopy.rma_uni('yi', 'vi', mods='~ alloc + ablat', data=dat))
    emmcheck.check(metafopy.pairs(alloc, side='>').summary(infer=True), 'stored_pairs_side')
    emmcheck.check(metafopy.emmeans(grid, 'alloc', delta=0.3, null=-0.8).summary(infer=True), 'stored_emm_delta')
    emmcheck.check(metafopy.contrast(metafopy.emmeans(grid, 'alloc', side='>'), 'pairwise').summary(infer=True), 'stored_contrast_reset')
    emmcheck.check(metafopy.pairs(alloc, side='>').summary(infer=True, side='<'), 'stored_override')
    assert metafopy.emmeans(grid, 'alloc', side='right').side == 1
    assert metafopy.contrast(metafopy.emmeans(grid, 'alloc', side='right'), 'pairwise').side == 0


@pytest.mark.parametrize(
    ('side', 'code'),
    [
        (-1, -1),
        (0, 0),
        (1, 1),
        ('<', -1),
        ('>', 1),
        ('left', -1),
        ('l', -1),
        ('r', 1),
        ('both', 0),
        ('!=', 0),
        ('=', 0),
        ('equivalence', 0),
        ('+1', 1),
        ('nonsup', -1),
        ('nonsense', 0),
        ('no', 0),
    ],
)
def test_parse_side(side: int | str, code: int) -> None:
    assert metafopy.methods.emm_adjust.parse_side(side) == code


def test_matrix_weights(additive: metafopy.EmmGrid, crossed: metafopy.EmmGrid) -> None:
    mw = numpy.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    emmcheck.check(metafopy.emmeans(additive, 'lat', weights=mw).summary(infer=True), 'mw_lat')
    emmcheck.check(metafopy.pairs(metafopy.emmeans(additive, 'lat', weights=mw)).summary(infer=True), 'mw_lat_pairs')
    emmcheck.check(metafopy.emmeans(additive, 'alloc', weights=numpy.array([[1.0, 3.0]])).summary(infer=True), 'mw_alloc_one_row')
    emmcheck.check(metafopy.emmeans(additive, 'alloc', weights=[[1, 3], [3, 1]]).summary(infer=True), 'mw_alloc_cycle')
    emmcheck.check(metafopy.emmeans(additive, '1', weights=numpy.arange(1.0, 7.0).reshape(1, 6)).summary(infer=True), 'mw_overall')
    emmcheck.check(metafopy.emmeans(additive, 'lat', weights=numpy.arange(1.0, 10.0).reshape(3, 3)).summary(infer=True), 'mw_extra_rows')
    emmcheck.check(metafopy.emmeans(additive, 'alloc', by='lat', weights=[[1, 3], [3, 1]]).summary(infer=True), 'mw_by')
    emmcheck.check(metafopy.emmeans(crossed, 'alloc', weights=[[1, 3], [3, 1], [1, 1]]).summary(infer=True), 'mw_int')
    emmcheck.check(metafopy.emmeans(additive, 'lat', weights=mw).summary(infer=True, type='response'), 'mw_resp')


def test_matrix_weights_must_fit(additive: metafopy.EmmGrid) -> None:
    with pytest.raises(metafopy.errors.InputError, match='Nonconforming number of weights -- need 3'):
        metafopy.emmeans(additive, 'lat', weights=numpy.ones((2, 2)))
