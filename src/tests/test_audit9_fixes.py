"""Audit 9: ``print(digits=<vector>)`` of predictions, cumulative and leave-one-out results and influence diagnostics, ``print(num=TRUE)`` of
location-scale, ``rma.mv``, ``rma.glmm`` and selection-model fits and of ``vif()``, the trim-and-fill header of ``selmodel()``, the names of ``Z.G1`` of
``struct = 'GDIAG'`` and ``isTRUE`` flags of ``ranef()``. The expected text is what R metafor 5.2-1 (R 4.3.3) prints."""

import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets

DIGITS = {'est': 2, 'se': 3, 'test': 1, 'pval': 3, 'ci': 1, 'var': 4, 'sevar': 4, 'fit': 4, 'het': 1}


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def small(dat: pandas.DataFrame) -> metafopy.models.rma_uni.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat.iloc[:6], method='REML')


def lines(text: object) -> list[str]:
    """The printout without trailing blanks and without the blank lines around it, as a list of lines."""
    return [line.rstrip() for line in str(text).strip('\n').split('\n')]


def test_print_predict_with_named_digits(small: metafopy.models.rma_uni.RMAResult) -> None:
    # R: print(predict(res), digits = c(est=2, se=3, test=1, pval=3, ci=1, var=4, sevar=4, fit=4, het=1))
    assert lines(metafopy.predict(small).with_digits(DIGITS)) == ['  pred    se ci.lb ci.ub pi.lb pi.ub', ' -0.99 0.237  -1.5  -0.5  -2.0   0.0']
    # the standard error is left out after a transformation
    assert lines(metafopy.predict(small, transf=numpy.exp).with_digits(DIGITS)) == [' pred ci.lb ci.ub pi.lb pi.ub', ' 0.37   0.2   0.6   0.1   1.1']


def test_print_predict_with_unnamed_digits_uses_the_first(small: metafopy.models.rma_uni.RMAResult) -> None:
    # R: print(predict(res), digits = c(3, 1)) gives every column three decimals
    assert lines(metafopy.predict(small).with_digits([3, 1])) == ['   pred    se  ci.lb  ci.ub  pi.lb pi.ub', ' -0.988 0.237 -1.452 -0.524 -2.024 0.049']


def test_print_cumul_with_named_digits(dat: pandas.DataFrame, small: metafopy.models.rma_uni.RMAResult) -> None:
    out = lines(metafopy.cumul(small, order=dat['year'].to_numpy()[:6]).with_digits(DIGITS))
    assert out[0:3] == [
        '  k estimate    se zval  pval ci.lb ci.ub    Q    Qp   tau2   I2  H2',
        '1 1    -0.89 0.571 -1.6 0.119  -2.0   0.2  0.0 1.000 0.0000  0.0 1.0',
        '2 2    -1.33 0.349 -3.8 0.000  -2.0  -0.6  0.9 0.334 0.0000  0.0 1.0',
    ]
    assert out[5] == '5 5    -0.84 0.248 -3.4 0.001  -1.3  -0.4 10.2 0.037 0.1756 69.8 3.3'


def test_print_leave1out_with_named_digits(small: metafopy.models.rma_uni.RMAResult) -> None:
    out = lines(metafopy.leave1out(small).with_digits(DIGITS))
    assert out[0:2] == [
        '   estimate    se zval  pval ci.lb ci.ub    Q    Qp   tau2   I2  H2',
        '-1    -1.01 0.269 -3.7 0.000  -1.5  -0.5 28.5 0.000 0.2680 88.4 8.7',
    ]


def test_print_influence_uses_the_est_digits(small: metafopy.models.rma_uni.RMAResult) -> None:
    # R: every column of the influence table is formatted with digits[["est"]]
    out = lines(metafopy.influence(small).with_digits({'est': 1, 'se': 3}))
    assert out[0:2] == [
        '  rstudent dffits cook.d cov.r tau2.del QE.del hat weight dfbs inf',
        '1      0.1    0.1    0.0   1.3      0.3   28.5 0.1   10.2  0.1',
    ]


def test_digits_map_follows_get_digits() -> None:
    assert metafopy.mfopt.digits_map(None, 4) == dict.fromkeys(metafopy.mfopt.DIGIT_KEYS, 4)
    assert metafopy.mfopt.digits_map([3, 1], 4) == dict.fromkeys(metafopy.mfopt.DIGIT_KEYS, 3)
    named = metafopy.mfopt.digits_map({'est': 2, 'ci': 1, 'pv': 5}, 4)
    assert named['est'] == 2
    assert named['ci'] == 1
    assert named['pval'] == 5  # a name that starts one name is that name, as pmatch has it
    assert named['se'] == 4


def test_selmodel_of_a_trimfill_fit_keeps_the_header_unless_it_was_refitted(dat: pandas.DataFrame) -> None:
    # R: selmodel(trimfill(rma(yi, vi, data = dat, method = "FE")), type = "halfnorm") prints the trim-and-fill header with the k0 of the
    # selection model (0: no study was left out by subset), not that of the trim-and-fill fit (4); a REML fit is fitted again by ML and loses it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kept = metafopy.selmodel(metafopy.trimfill(metafopy.rma('yi', 'vi', data=dat, method='FE')), type='halfnorm')
        refitted = metafopy.selmodel(metafopy.trimfill(metafopy.rma('yi', 'vi', data=dat, method='REML')), type='halfnorm')
    assert lines(kept)[0] == 'Estimated number of missing studies on the right side: 0 (SE = 2.3853)'
    assert lines(kept)[2] == 'Fixed-Effects Model (k = 17)'
    assert 'Estimated number of missing studies' not in str(refitted)


def test_gdiag_design_matrix_has_the_inner_variables_as_columns() -> None:
    # R: colnames(rma.mv(yi, vi, mods = ~ year, random = ~ year | district, struct = "GDIAG", data = dat)$Z.G1) is c("intrcpt", "year")
    frame = metafopy.datasets.load('dat.konstantopoulos2011').copy()
    frame['year'] = frame['year'] - frame['year'].mean()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.rma_mv('yi', 'vi', mods='~ year', random='~ year | district', struct='GDIAG', data=frame)
    z = typing.cast(typing.Any, fit.Z_G1)
    assert z is not None
    assert tuple(z.colnames) == ('intrcpt', 'year')
    assert tuple(z.rownames[:3]) == ('1', '2', '3')


def test_ranef_vcov_is_only_true_for_a_logical() -> None:
    # R: ranef(res, vcov = <a model>) is ranef(res), because the flag is read with isTRUE()
    frame = metafopy.datasets.load('dat.konstantopoulos2011')
    fit = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=frame)
    other = metafopy.rma('yi', 'vi', data=frame)
    plain = typing.cast(typing.Any, metafopy.ranef(fit, vcov=other))
    assert list(plain) == ['district', 'district/school']
    assert list(typing.cast(typing.Any, metafopy.ranef(fit, vcov=True))) == ['pred', 'vcov']


def test_num_numbers_the_rows_of_location_scale_tables() -> None:
    # R: print(rma(yi, vi, mods = ~ I(ni/100) + meta, scale = ~ I(ni/100) + imag, data = dat.bangertdrowns2004), num = TRUE)
    frame = metafopy.datasets.load('dat.bangertdrowns2004')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.rma('yi', 'vi', mods='~ I(ni/100) + meta', scale='~ I(ni/100) + imag', data=frame)
    out = lines(fit.summary(num=True))
    assert '1) intrcpt      0.2303  0.0655   3.5166  0.0004   0.1019   0.3586  ***' in out
    assert '3) meta         0.1469  0.0690   2.1305  0.0331   0.0118   0.2820    *' in out
    assert '2) I(ni/100)   -0.9995  0.6087  -1.6421  0.1006  -2.1925   0.1935' in out
    assert '3) imag         2.1258  1.1857   1.7929  0.0730  -0.1981   4.4497   .' in out


def test_num_numbers_the_rows_of_mv_glmm_and_selection_model_tables(dat: pandas.DataFrame) -> None:
    raw = metafopy.datasets.load('dat.bcg')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mv = metafopy.rma_mv('yi', 'vi', mods='~ ablat + alloc', random='~ 1 | trial', data=dat)
        glmm = metafopy.rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=raw, mods='~ ablat + alloc', model='UM.FS')
        sel = typing.cast(typing.Any, metafopy.selmodel(metafopy.rma('yi', 'vi', mods='~ ablat + alloc', data=dat, method='ML'), type='halfnorm'))
    assert '2) ablat             -0.0273  0.0092  -2.9650  0.0030  -0.0453  -0.0092  **' in lines(mv.summary(num=True))
    assert '2) ablat             -0.0345  0.0032  -10.8965  <.0001  -0.0407  -0.0283  ***' in lines(glmm.summary(num=True))
    assert '4) allocsystematic   -0.0926  0.2422  -0.3822  0.7023  -0.5672   0.3821' in lines(sel.summary(num=True))


def test_num_numbers_the_rows_of_the_vif_table(dat: pandas.DataFrame) -> None:
    fit = metafopy.rma('yi', 'vi', mods='~ ablat + alloc', data=dat)
    out = lines(metafopy.vif(fit, table=True).summary(num=True))
    assert out[0] == '                   estimate     se    zval   pval   ci.lb   ci.ub    vif    sif'
    assert out[2] == '2) ablat            -0.0273 0.0092 -2.9650 0.0030 -0.0453 -0.0092 1.0151 1.0075'
