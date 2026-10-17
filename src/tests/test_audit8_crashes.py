"""Regression tests for the crashes of audit 8; expected values come from R metafor 5.2-1 (the R call is quoted with each test)."""

import warnings

import matplotlib
import numpy
import pytest
import refutil

import metafopy
import metafopy.core.parallel
import metafopy.methods.blocks
import metafopy.models.rma_mv
import metafopy.models.sel_result
import metafopy.transf

matplotlib.use('Agg')

DIGITS = [2, numpy.nan, 1]  # R: digits = c(2, NA, 1); an unnamed vector means its first element for every statistic


def konstantopoulos() -> object:
    return metafopy.datasets.load('konstantopoulos2011')


def test_funnel_shade_shorter_than_level() -> None:
    # R: funnel(rma(yi, vi, data = dat), shade = c("white", "gray"), level = c(90, 95, 99), legend = TRUE) draws without error (shade[3] is NA: no fill)
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    metafopy.funnel(res, shade=['white', 'gray'], level=[90, 95, 99], legend=True)
    matplotlib.pyplot.close('all')


def test_addpoly_digits_vector() -> None:
    # R: addpoly(res, digits = c(2, NA, 1)) uses digits[[1]]
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    metafopy.forest(res)
    metafopy.addpoly(res, digits=DIGITS)
    matplotlib.pyplot.close('all')


def test_digits_vector_in_residuals_and_vif() -> None:
    # R: rstandard(res, digits = c(2, NA, 1)) and vif(res, digits = c(2, NA, 1)) print two decimals
    data = refutil.bcg()
    ls = metafopy.rma_uni('yi', 'vi', mods='~ ablat + year', scale='~ ablat', data=data)
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011'))
    for res, fn in ((ls, metafopy.rstandard), (ls, metafopy.vif), (mv, metafopy.rstandard)):
        assert str(fn(res, digits=DIGITS)) == str(fn(res, digits=2))


def test_deltamethod_digits_vector() -> None:
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg())
    out = metafopy.deltamethod(res, fun=lambda b0, b1: b0 + b1, digits=DIGITS)
    assert str(out) == str(metafopy.deltamethod(res, fun=lambda b0, b1: b0 + b1, digits=2))


def test_cl_that_is_not_a_cluster_is_ignored() -> None:
    # R: profile(res, cl = <a vector>) and the other functions with ``cl`` only use a SOCKcluster; any other value leaves parallel = "no"
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    assert metafopy.core.parallel.resolve_parallel('no', 1, [1, 2, 3]) == 'no'  # type: ignore[arg-type]
    plain = metafopy.profile(res, steps=5)
    odd = metafopy.profile(res, steps=5, cl=list(range(13)))
    numpy.testing.assert_allclose(odd.ll, plain.ll)


def test_pft_and_ipft_recycle_like_r() -> None:
    # R: transf.pft(c(0.1, 0.2, 0.3, 0.4), c(100, 200)); transf.ipft(c(0.1, 0.2, 0.3, 0.4), c(100, 200))
    numpy.testing.assert_allclose(metafopy.transf.transf_pft([0.1, 0.2, 0.3, 0.4], [100, 200]), [0.3282089, 0.4655035, 0.5817857, 0.6852256], atol=1e-7)
    numpy.testing.assert_allclose(
        metafopy.transf.transf_ipft([0.1, 0.2, 0.3, 0.4], [100, 200]), [0.005681389, 0.037204624, 0.083269596, 0.149913298], atol=1e-8
    )
    with pytest.warns(UserWarning, match='not a multiple'):
        out = metafopy.transf.transf_pft(numpy.linspace(0.1, 0.5, 5), numpy.linspace(100, 340, 13))
    assert out.shape == (13,)


def test_regplot_scalar_line_arguments() -> None:
    # R: regplot(res, lcol = "blue", lwd = 2, lty = 2) repeats a single value for the four lines (.expand1(x, 4))
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg())
    metafopy.regplot(res, lcol='blue', lwd=2, lty=2)
    axx = matplotlib.pyplot.gca()
    refline = [ln for ln in axx.lines if ln.get_linestyle() == '--']
    assert refline
    assert all(ln.get_color() == refline[0].get_color() for ln in refline)
    matplotlib.pyplot.close('all')


def test_plot_selmodel_scalar_line_arguments() -> None:
    # R: plot(sel, lty = 2, lwd = 2) only needs lty[1] and lwd[1] without a confidence region
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    sel = metafopy.selmodel(res, type='beta')
    assert isinstance(sel, metafopy.models.sel_result.SelmodelResult)
    for lty in ('solid', 'dotted', 2):
        metafopy.plot.plot_selmodel(sel, lty=lty, lwd=2)
    matplotlib.pyplot.close('all')


def test_rma_mv_restart_true_uses_the_last_values() -> None:
    # R: rma.mv(..., restart = TRUE) starts from the last values the optimiser tried in the session
    dat = metafopy.datasets.load('konstantopoulos2011')
    first = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=dat)
    again = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=dat, restart=True)
    assert again.sigma2 is not None
    assert first.sigma2 is not None
    numpy.testing.assert_allclose(again.sigma2, first.sigma2, rtol=1e-4)
    numpy.testing.assert_allclose(again.beta, first.beta, rtol=1e-4)


def test_rma_mv_restart_true_without_a_fit() -> None:
    metafopy.models.rma_mv.LAST_EVALUATED.clear()
    with pytest.raises(metafopy.errors.InputError, match='Restarting'):
        metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011'), restart=True)


def test_bldiag_takes_a_named_list() -> None:
    # R: bldiag(split(...)) takes the named list that split() returns
    blocks = {'a': numpy.ones((2, 2)), 'b': numpy.full((1, 1), 2.0)}
    out = metafopy.methods.blocks.bldiag(blocks)
    numpy.testing.assert_array_equal(out, metafopy.methods.blocks.bldiag(*blocks.values()))
    assert out.shape == (3, 3)


def test_list_rma_subsetting() -> None:
    # R: res <- rma(yi, vi, mods = ~ ablat, data = dat); rstandard(res)[1:3]; predict(res)[2:3]
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg())
    sub = metafopy.rstandard(res)[0:3]
    numpy.testing.assert_allclose(sub.resid, [0.1397, -0.2363, -0.3773], atol=5e-5)
    numpy.testing.assert_allclose(sub.se, [0.6183, 0.4801, 0.6888], atol=5e-5)
    assert sub.slab == ('1', '2', '3')
    assert sub.digits == metafopy.rstandard(res).digits
    pred = metafopy.predict(res)[1:3]
    numpy.testing.assert_allclose(pred.pred, [-1.3491, -0.9708], atol=5e-5)
    numpy.testing.assert_allclose(pred.pi_lb, [-2.0190, -1.5706], atol=5e-5)
    assert pred.slab is None or len(pred.slab) == 2
    mask = numpy.zeros(13, dtype=bool)
    mask[[0, 5]] = True
    assert metafopy.rstandard(res)[mask].resid.shape == (2,)


def test_list_rma_subsetting_of_leave1out_and_cumul() -> None:
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    for out in (metafopy.leave1out(res), metafopy.cumul(res)):
        sub = out[0:2]
        assert sub.estimate.shape == (2,)
        numpy.testing.assert_array_equal(sub.estimate, out.estimate[:2])
        assert len(sub.slab) == 2


def test_no_warning_for_equal_lengths() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metafopy.transf.transf_pft([0.1, 0.2], [100, 200])


def test_pairmat_without_the_model() -> None:
    # R: anova(res, X = pairmat(btt = "alloc")) and predict(res, newmods = pairmat(btt = "alloc")) take the model of the call they are an argument of
    res = metafopy.rma_uni('yi', 'vi', mods='~ alloc', data=refutil.bcg())
    lazy = metafopy.anova(res, X=metafopy.pairmat(btt='alloc'))
    explicit = metafopy.anova(res, X=metafopy.pairmat(res, btt='alloc'))
    assert str(lazy) == str(explicit)
    numpy.testing.assert_allclose(lazy.pval, explicit.pval)
    pred = metafopy.predict(res, newmods=metafopy.pairmat('alloc'))
    numpy.testing.assert_allclose(pred.pred, metafopy.predict(res, newmods=metafopy.pairmat(res, btt='alloc')).pred)
    with pytest.raises(metafopy.errors.InputError, match="'btt'"):
        metafopy.pairmat()


def test_forest_digits_double_vector() -> None:
    # R: forest(cumul(res), atransf = exp, digits = c(2, 3)) where the digits are doubles, not integers
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    metafopy.forest(metafopy.cumul(res, order='year'), atransf=numpy.exp, digits=numpy.array([2.0, 3.0]))
    matplotlib.pyplot.close('all')


def test_btt_double() -> None:
    # R: anova(res, btt = 2) where 2 is a double
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat + year', data=refutil.bcg())
    out = metafopy.anova(res, btt=numpy.float64(1))
    assert out.btt == (1,)


def test_transf_pft_matrix_keeps_shape() -> None:
    # R: transf.pft(matrix(c(.1, .2, .3, .4), 2), 10) keeps the 2 x 2 shape
    grid = numpy.array([[0.1, 0.2], [0.3, 0.4]])
    assert metafopy.transf.transf_pft(grid, 10).shape == (2, 2)
    assert metafopy.transf.transf_pft(0.2, numpy.array([[10, 20], [30, 40]])).shape == (2, 2)
