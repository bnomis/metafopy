"""Printed layout of frames, dfbetas names and the reference grid, checked against R metafor 5.2-1 and emmeans (the expected texts are R's output)."""

import pandas
import pytest

import metafopy
import metafopy.datasets
from metafopy.core.labelled import RFrame, column


@pytest.fixture(scope='module')
def bcg_mods() -> metafopy.models.rma_uni.RMAResult:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    return metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ ablat')


def test_column_switches_to_scientific_like_r() -> None:
    # print(c(-1.620293e-03, 5.177601e-05)) in R
    assert column([-1.620293e-03, 5.177601e-05]) == ['-1.620293e-03', '5.177601e-05']
    # print(c(100000, 2.5)), print(c(123456, 0.5))
    assert column([100000.0, 2.5]) == ['1.0e+05', '2.5e+00']
    assert column([123456.0, 0.5]) == ['123456.0', '0.5']


def test_dfbetas_prints_with_the_digits_of_the_model(bcg_mods: metafopy.models.rma_uni.RMAResult) -> None:
    out = str(metafopy.dfbetas(bcg_mods)).splitlines()
    # R: print(dfbetas(res)) gives "1   0.0065  0.0224" for the first study
    assert out[1].split() == ['intrcpt', 'ablat']
    assert out[2].split() == ['1', '0.0065', '0.0224']


def test_vcov_prints_wide_matrices_in_blocks(bcg_mods: metafopy.models.rma_uni.RMAResult) -> None:
    out = metafopy.vcov(bcg_mods, type='obs')
    assert isinstance(out, RFrame)
    assert '...' not in str(out)
    # R: print(vcov(res)) has -1.620293e-03 and 5.177601e-05 in the column of ablat
    assert '5.177601e-05' in str(metafopy.vcov(bcg_mods)).split()


def test_selections_of_a_printed_frame_are_plain_frames(bcg_mods: metafopy.models.rma_uni.RMAResult) -> None:
    sel = metafopy.dfbetas(bcg_mods).iloc[:2]
    assert type(sel) is pandas.DataFrame


def test_influence_prints_syntactic_coefficient_names() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    res = metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ factor(alloc)')
    # R: make.names() of the coefficient names, as the data frame of dfbs has
    assert 'factor.alloc.random' in str(metafopy.influence(res))


def test_reference_grid_prints_without_intervals(bcg_mods: metafopy.models.rma_uni.RMAResult) -> None:
    # R: print(emmprep(res)) shows prediction, SE and df only
    header = str(metafopy.emmprep(bcg_mods)).split('\n')
    assert any(line.split()[:4] == ['ablat', 'prediction', 'SE', 'df'] for line in header if line.strip())
    assert 'LCL' not in str(metafopy.emmprep(bcg_mods))


@pytest.fixture(scope='module')
def bcg_alloc() -> metafopy.models.rma_uni.RMAResult:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    return metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ ablat + alloc')


# R: print(emmeans(emmprep(res), "alloc")) and print(summary(contrast(e, "pairwise"))), emmeans 2.0.4 with metafor 5.2-1
R_MEANS = '\n'.join(
    [
        ' alloc      emmean    SE  df asymp.LCL asymp.UCL',
        ' alternate  -0.619 0.294 Inf     -1.19   -0.0442',
        ' random     -0.887 0.191 Inf     -1.26   -0.5123',
        ' systematic -0.561 0.244 Inf     -1.04   -0.0822',
        '',
        'Results are given on the log (not the response) scale. ',  # R's cat() leaves a space after each note
        'Confidence level used: 0.95 ',
    ]
)

R_PAIRS = '\n'.join(
    [
        ' contrast               estimate   SE  df z.ratio p.value',
        ' alternate - random       0.2675 0.35 Inf   0.763  0.7255',
        ' alternate - systematic  -0.0585 0.38 Inf  -0.154  0.9870',
        ' random - systematic     -0.3260 0.31 Inf  -1.050  0.5452',
        '',
        'Results are given on the log (not the response) scale. ',
        'P value adjustment: tukey method for comparing a family of 3 estimates ',
    ]
)


def test_emmeans_summary_prints_as_r_does(bcg_alloc: metafopy.models.rma_uni.RMAResult) -> None:
    means = metafopy.emmeans(metafopy.emmprep(bcg_alloc), 'alloc')
    assert str(means) == R_MEANS
    assert str(metafopy.contrast(means, 'pairwise').summary()) == R_PAIRS


def test_emmeans_by_groups_print_under_their_level(bcg_alloc: metafopy.models.rma_uni.RMAResult) -> None:
    out = str(metafopy.emmeans(metafopy.emmprep(bcg_alloc, at={'ablat': [10, 30]}), 'alloc', by='ablat'))
    assert out.splitlines()[0] == 'ablat = 10:'
    assert 'ablat = 30:' in out


def test_contrast_methods_take_emmeans_spelling_and_confint_works(bcg_alloc: metafopy.models.rma_uni.RMAResult) -> None:
    means = metafopy.emmeans(metafopy.emmprep(bcg_alloc), 'alloc')
    dotted = metafopy.contrast(means, 'trt.vs.ctrl').summary().to_frame()
    underscored = metafopy.contrast(means, 'trt_vs_ctrl').summary().to_frame()
    assert dotted.equals(underscored)
    assert 'asymp.LCL' in metafopy.confint(means).to_frame().columns


def test_predict_names_the_model_matrix_column_of_one_column_matrices() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    res = metafopy.rma(yi='yi', vi='vi', data=dat)
    # R: print(predict(res, addx=TRUE)) has the column "intrcpt"
    assert str(metafopy.predict(res, addx=True)).split('\n')[1].split()[-1] == 'intrcpt'


def test_marginal_vcov_prints_zeros_as_dots(bcg_mods: metafopy.models.rma_uni.RMAResult) -> None:
    # R: print(vcov(res, type = "obs")) has class vcovmat, which shows four decimals and "." for zero
    lines = str(metafopy.vcov(bcg_mods, type='obs')).splitlines()
    assert lines[1].split()[:4] == ['1', '2', '3', '4']
    assert lines[2].split()[:4] == ['1', '0.4019', '.', '.']


def test_intercept_only_reference_grid_has_the_variable_called_1() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    res = metafopy.rma(yi='yi', vi='vi', data=dat)
    # R: print(emmprep(res))
    assert str(metafopy.emmprep(res)) == '\n'.join(
        [' 1 prediction   SE  df', ' 1     -0.715 0.18 Inf', '', 'Results are given on the log (not the response) scale. '],
    )


def test_covariate_with_two_values_keeps_both_in_the_reference_grid() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    dat['flag'] = (dat['year'] > 1950).astype(float)
    res = metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ ablat + flag')
    # emmeans' cov.keep = 2: flag is 0 and 1, ablat is its mean
    grid = metafopy.emmprep(res).summary().to_frame()
    assert grid['flag'].tolist() == [0.0, 1.0]
    assert grid['ablat'].nunique() == 1


def test_scale_prediction_has_no_prediction_interval() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    res = metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ ablat', scale='~ ablat')
    # R: predict(res, newscale = 30) gives pred, se, ci.lb and ci.ub
    assert str(metafopy.predict(res, newscale=[30])).split()[:4] == ['pred', 'se', 'ci.lb', 'ci.ub']
    assert 'pi.lb' not in str(metafopy.predict(res, newscale=[30]))


def test_mantel_haenszel_prediction_shows_the_column_x() -> None:
    dat = metafopy.datasets.load('dat.bcg')
    res = metafopy.rma_mh(ai=dat['tpos'], bi=dat['tneg'], ci=dat['cpos'], di=dat['cneg'])
    # R: print(predict(res, addx = TRUE)) ends in the column X
    assert str(metafopy.predict(res, addx=True)).split('\n')[1].split()[-1] == 'X'


def test_formatters_print_noquote_vectors() -> None:
    assert str(metafopy.fmtx([1.2345, 2.5, -0.001])) == '[1]  1.2345  2.5000 -0.0010'
    assert str(metafopy.fmtp([0.0123, 0.5, 0.00001])) == '[1] 0.0123 0.5000 <.0001'
    assert str(metafopy.fmtp2([0.0123, 0.5, 0.00001])) == '[1] p = .012 p = .50  p < .001'


def test_summary_and_friends_of_escalc_frames_print_as_escalc() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))
    assert isinstance(metafopy.summary_escalc(dat), metafopy.core.labelled.EscalcFrame)
    assert isinstance(metafopy.dfround(dat, 2), metafopy.core.labelled.EscalcFrame)
    wald = metafopy.conv_wald(out=[0.5, 0.7], ci_lb=[0.1, 0.2], ci_ub=[0.9, 1.2], n=[30, 40], append=False)
    # R: rows numbered from 1 and the variances to four decimals
    assert str(wald).splitlines()[2].split() == ['1', '0.5000', '0.0417']
