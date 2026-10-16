"""Round-five audit, plots: the labels of forest() for the back-transformations, ilab_lab strings, addpoly() of predict results with
predstyle 'shade'/'dist', named colours in colour ramps, legend positions, abbreviated predstyle, and the arguments metafor reads through
``...`` (confint(xlim=), forest(addcred=, clim=), addpoly(addcred=), plot_cumul(addgrid=), layout, graphics pass-through arguments).

The labels follow .setlab() of metafor 5.2-1 (https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.r).
"""

import collections.abc
import importlib
import typing
import warnings

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.permutest
import metafopy.methods.vif
import metafopy.plot.rutils
import metafopy.transf


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def res(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat)


@pytest.fixture(scope='module')
def meta(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)


def texts(a: matplotlib.axes.Axes) -> list[str]:
    return [t.get_text() for t in a.texts]


# -- labels of the back-transformations -------------------------------------------------------------------------------

LABELS: list[tuple[str, typing.Any, typing.Any, int, bool, str]] = [
    ('ZCOR', metafopy.transf.transf_ztor, None, 1, False, 'Correlation Coefficient'),
    ('ZCOR', None, metafopy.transf.transf_ztor, 1, False, 'Correlation Coefficient'),
    ('ZCOR', metafopy.transf.transf_ztor_int, None, 1, False, 'Correlation Coefficient'),
    ('ZCOR', numpy.tanh, None, 2, False, 'Correlation Coefficient'),
    ('ZCOR', None, None, 1, False, "Fisher's z Transformed Correlation Coefficient"),
    ('ZCOR', metafopy.transf.transf_ztor, None, 3, True, 'Correlation'),
    ('PLO', metafopy.transf.transf_ilogit, None, 1, False, 'Proportion'),
    ('PLO', None, metafopy.transf.transf_ilogit, 1, False, 'Proportion (logit scale)'),
    ('PLO', metafopy.transf.plogis, None, 1, True, 'Proportion'),
    ('PLO', numpy.exp, None, 1, False, 'Odds'),
    ('PLO', None, numpy.exp, 1, False, 'Odds (log scale)'),
    ('PAS', metafopy.transf.transf_iarcsin, None, 1, False, 'Proportion'),
    ('PAS', None, metafopy.transf.transf_iarcsin, 1, False, 'Proportion (arcsine scale)'),
    ('PRZ', metafopy.transf.transf_iprobit, None, 1, False, 'Proportion'),
    ('PRZ', None, metafopy.transf.pnorm, 1, False, 'Proportion (probit scale)'),
    ('PFT', metafopy.transf.transf_ipft_hm, None, 1, False, 'Proportion'),
    ('ZR2', metafopy.transf.transf_ztor2, None, 1, False, 'Coefficient of Determination'),
    ('AHW', metafopy.transf.transf_iahw, None, 1, False, "Cronbach's alpha"),
    ('ABT', metafopy.transf.transf_iabt, None, 1, True, 'Alpha'),
    ('IRS', metafopy.transf.transf_isqrt, None, 1, False, 'Incidence Rate'),
    ('IRS', None, metafopy.transf.transf_isqrt, 1, False, 'Incidence Rate (square root scale)'),
    ('RR', numpy.exp, None, 1, False, 'Risk Ratio'),
    ('RR', metafopy.transf.transf_ilogit, None, 1, False, 'Transformed Log Risk Ratio'),
    ('RR', lambda v: v, None, 1, False, 'Transformed Log Risk Ratio'),
]


@pytest.mark.parametrize(('measure', 'transf', 'atransf', 'gentype', 'short', 'expected'), LABELS)
def test_setlab_recognises_the_back_transformations(measure: str, transf: typing.Any, atransf: typing.Any, gentype: int, short: bool, expected: str) -> None:
    assert metafopy.plot.rutils.setlab(measure, transf, atransf, gentype, short) == expected


def test_forest_label_of_a_correlation_back_transformation(ax: matplotlib.axes.Axes) -> None:
    mol = metafopy.escalc('ZCOR', ri='ri', ni='ni', data=metafopy.datasets.load('dat.molloy2014'))
    fit = metafopy.rma('yi', 'vi', data=mol)
    metafopy.forest(fit, transf=metafopy.transf.transf_ztor, ax=ax)
    found = texts(ax)
    assert 'Correlation Coefficient' in found
    assert not any(t.startswith('Transformed') for t in found)


# -- ilab_lab ---------------------------------------------------------------------------------------------------------


def test_forest_ilab_lab_single_string(res: metafopy.RMAResult, dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    metafopy.forest(res, ilab=dat[['ablat']], ilab_lab='Latitude', ax=ax)
    found = texts(ax)
    assert 'Latitude' in found
    assert not {'L', 'a', 't'} & set(found)


# -- addpoly of predict results ---------------------------------------------------------------------------------------


@pytest.mark.parametrize('style', ['shade', 'dist', 's', 'd'])
def test_addpoly_predict_result_with_distribution_styles(res: metafopy.RMAResult, style: str, ax: matplotlib.axes.Axes) -> None:
    metafopy.forest(res, addfit=False, ax=ax)
    before = len(ax.patches) + len(ax.collections)
    pr = metafopy.predict(res)
    assert pr.pi_se is not None
    metafopy.addpoly(pr, predstyle=style, ax=ax)
    assert len(ax.patches) + len(ax.collections) > before


def test_addpoly_predict_result_distribution_matches_the_fit(res: metafopy.RMAResult) -> None:
    fig, (a1, a2) = matplotlib.pyplot.subplots(1, 2)
    try:
        metafopy.forest(res, addfit=False, ax=a1)
        metafopy.forest(res, addfit=False, ax=a2)
        metafopy.addpoly(res, predstyle='shade', ax=a1)
        metafopy.addpoly(metafopy.predict(res), predstyle='shade', ax=a2)
        assert len(a1.patches) == len(a2.patches)
        assert len(a1.collections) == len(a2.collections)
    finally:
        matplotlib.pyplot.close(fig)


def test_predstyle_abbreviations() -> None:
    check = importlib.import_module('metafopy.plot.forest').check_pred
    assert [check(s) for s in ('l', 'p', 'b', 's', 'd', 'li', 'poly', 'dist')] == ['line', 'polygon', 'bar', 'shade', 'dist', 'line', 'polygon', 'dist']
    for bad in ('', 'x', 'lines'):
        with pytest.raises(metafopy.errors.InputError, match="'predstyle' should be one of"):
            check(bad)


def test_forest_accepts_an_abbreviated_predstyle(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.forest(res, predstyle='b', ax=ax)
    assert out.ax is ax


# -- colours given by name --------------------------------------------------------------------------------------------


def test_plot_cumul_named_colours(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.plot_cumul(metafopy.cumul(res), cols=['green', 'red'], ax=ax)
    assert out.col[0] == '#00FF00'
    assert out.col[-1] == '#FF0000'


def test_color_ramp_names_and_hex_agree() -> None:
    assert metafopy.plot.rutils.color_ramp('white', '#000000', 3) == ['#FFFFFF', '#7F7F7F', '#000000']
    assert metafopy.plot.rutils.color_ramp('#FFFFFF', '#000000', 3) == ['#FFFFFF', '#7F7F7F', '#000000']


def test_funnel_named_shade(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.funnel(res, shade='white', level=[90, 95, 99], ax=ax)
    assert out.ax is ax


@pytest.mark.parametrize(
    ('legend', 'code'),
    [({'x': 'topleft'}, 2), ('topleft', 2), ({'x': 'bottomright'}, 4), ({'x': 'bottom'}, 8), ({'x': 'right'}, 7), ({'x': 'top'}, 9), ({'x': 'center'}, 10)],
)
def test_funnel_legend_positions(res: metafopy.RMAResult, legend: typing.Any, code: int, ax: matplotlib.axes.Axes) -> None:
    metafopy.funnel(res, legend=legend, ax=ax)
    drawn = ax.get_legend()
    assert drawn is not None
    assert typing.cast(typing.Any, drawn)._loc == code


# -- arguments read through ... ---------------------------------------------------------------------------------------


def test_forest_addcred_and_clim(res: metafopy.RMAResult) -> None:
    fig, axes = matplotlib.pyplot.subplots(2, 2)
    try:
        metafopy.forest(res, addpred=True, olim=(0.1, 3), ax=axes[0][0], transf=numpy.exp)
        metafopy.forest(res, addcred=True, clim=(0.1, 3), ax=axes[0][1], transf=numpy.exp)
        assert texts(axes[0][0]) == texts(axes[0][1])
        metafopy.forest(res, ax=axes[1][0], transf=numpy.exp)
        assert texts(axes[1][0]) != texts(axes[0][1])
        # addcred wins over addpred
        metafopy.forest(res, addcred=False, addpred=True, ax=axes[1][1], transf=numpy.exp)
        assert texts(axes[1][1]) == texts(axes[1][0])
    finally:
        matplotlib.pyplot.close(fig)


def test_addpoly_addcred(res: metafopy.RMAResult) -> None:
    fig, (a1, a2, a3) = matplotlib.pyplot.subplots(1, 3)
    try:
        for a in (a1, a2, a3):
            metafopy.forest(res, addfit=False, ax=a)
        metafopy.addpoly(res, ax=a1)
        metafopy.addpoly(res, addpred=True, ax=a2)
        metafopy.addpoly(res, addcred=True, ax=a3)
        assert len(a2.lines) == len(a3.lines)
        assert len(a1.lines) < len(a3.lines)
    finally:
        matplotlib.pyplot.close(fig)


def test_plot_cumul_addgrid(res: metafopy.RMAResult) -> None:
    cum = metafopy.cumul(res)
    fig, (a1, a2, a3) = matplotlib.pyplot.subplots(1, 3)
    try:
        metafopy.plot_cumul(cum, ax=a1)
        metafopy.plot_cumul(cum, addgrid=False, ax=a2)
        metafopy.plot_cumul(cum, grid=False, addgrid=True, ax=a3)
        assert len(a2.lines) < len(a1.lines)
        assert len(a3.lines) == len(a1.lines)
    finally:
        matplotlib.pyplot.close(fig)


def test_confint_xlim_sets_the_search_interval(res: metafopy.RMAResult) -> None:
    default = res.confint()
    wide = res.confint(xlim=[0, 5])
    ref = res.confint(control={'tau2_min': 0, 'tau2_max': 5})
    assert wide.random is not None
    assert ref.random is not None
    assert default.random is not None
    numpy.testing.assert_array_equal(wide.random.to_numpy(), ref.random.to_numpy())
    numpy.testing.assert_allclose(wide.random.to_numpy(), default.random.to_numpy(), rtol=1e-3)
    one = res.confint(xlim=5)
    assert one.random is not None
    numpy.testing.assert_array_equal(one.random.to_numpy(), ref.random.to_numpy())
    # an upper limit below the estimate cannot contain the bound
    with pytest.raises(metafopy.errors.InputError, match='length 1 or 2'):
        res.confint(xlim=[0, 1, 2])


def test_confint_xlim_of_rma_mv() -> None:
    kon = metafopy.datasets.load('konstantopoulos2011')
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | district', data=kon)
    a = mv.confint(xlim=[0, 1])
    b = mv.confint(control={'vc_min': 0, 'vc_max': 1})
    assert isinstance(a, metafopy.methods.confint.ConfintResult)
    assert isinstance(b, metafopy.methods.confint.ConfintResult)
    assert a.random is not None
    assert b.random is not None
    numpy.testing.assert_array_equal(a.random.to_numpy(), b.random.to_numpy())
    with pytest.raises(metafopy.errors.InputError, match='length 2'):
        mv.confint(xlim=[0, 1, 2])


def test_confint_xlim_of_location_scale_fits(dat: pandas.DataFrame) -> None:
    fit = metafopy.rma('yi', 'vi', scale='~ ablat', data=dat)
    a = fit.confint(alpha=1, xlim=[-3, 3])
    b = fit.confint(alpha=1, control={'vc_min': -3, 'vc_max': 3})
    assert isinstance(a, metafopy.methods.confint.ConfintResult)
    assert isinstance(b, metafopy.methods.confint.ConfintResult)
    assert a.random is not None
    assert b.random is not None
    numpy.testing.assert_array_equal(a.random.to_numpy(), b.random.to_numpy())
    with pytest.raises(metafopy.errors.InputError, match='length 2'):
        fit.confint(alpha=1, xlim=[1])


def test_layout_is_deprecated(meta: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    infl = metafopy.influence(meta)
    with pytest.warns(UserWarning, match="Argument 'layout' has been deprecated."):
        metafopy.plot_infl(infl, plotinf=[1], ax=ax, layout=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metafopy.plot_infl(infl, plotinf=[1], ax=matplotlib.pyplot.subplots()[1])
    matplotlib.pyplot.close('all')


@pytest.fixture(scope='module')
def perm(meta: metafopy.RMAResult) -> metafopy.methods.permutest.PermutestResult:
    result = metafopy.permutest(meta, iter=50, seed=1)
    assert isinstance(result, metafopy.methods.permutest.PermutestResult)
    return result


@pytest.fixture(scope='module')
def vifs(meta: metafopy.RMAResult) -> metafopy.methods.vif.VIFResult:
    return metafopy.vif(meta, btt=1, sim=30, progbar=False, seed=1)


def test_plot_permutest_layout_and_xlim(perm: metafopy.methods.permutest.PermutestResult, ax: matplotlib.axes.Axes) -> None:
    with pytest.warns(UserWarning, match="Argument 'layout' has been deprecated."):
        metafopy.plot_permutest(perm, beta=[], QM=True, ax=ax, layout=2)
    fig, axes = matplotlib.pyplot.subplots(1, 2)
    try:
        out = metafopy.plot_permutest(perm, ax=axes[0], xlim=(-5, 5))
        assert len(out.axes) == 2
        for panel in out.axes:
            lo, hi = panel.get_xlim()
            numpy.testing.assert_allclose([lo, hi], [-5.4, 5.4])
    finally:
        matplotlib.pyplot.close(fig)


def test_plot_vif_layout_and_xlim(vifs: metafopy.methods.vif.VIFResult, ax: matplotlib.axes.Axes) -> None:
    with pytest.warns(UserWarning, match="Argument 'layout' has been deprecated."):
        metafopy.plot_vif(vifs, ax=ax, layout=1)
    fig, axes = matplotlib.pyplot.subplots()
    try:
        out = metafopy.plot_vif(vifs, ax=axes, xlim=(0, 10))
        lo, hi = out.axes[0].get_xlim()
        numpy.testing.assert_allclose([lo, hi], [-0.4, 10.4])
    finally:
        matplotlib.pyplot.close(fig)


# -- graphics arguments passed through ... ------------------------------------------------------------------------------


def visible_spines(a: matplotlib.axes.Axes) -> set[str]:
    return {name for name, spine in a.spines.items() if spine.get_visible()}


@pytest.mark.parametrize(
    ('bty', 'sides'),
    [
        ('o', {'top', 'right', 'left', 'bottom'}),
        ('l', {'left', 'bottom'}),
        ('7', {'top', 'right'}),
        ('c', {'top', 'left', 'bottom'}),
        ('u', {'left', 'bottom', 'right'}),
        (']', {'top', 'right', 'bottom'}),
        ('n', set()),
    ],
)
def test_baujat_bty(res: metafopy.RMAResult, bty: str, sides: set[str], ax: matplotlib.axes.Axes) -> None:
    metafopy.baujat(res, bty=bty, ax=ax)
    assert visible_spines(ax) == sides


def test_baujat_las(res: metafopy.RMAResult) -> None:
    fig, (a1, a2) = matplotlib.pyplot.subplots(1, 2)
    try:
        metafopy.baujat(res, las=1, ax=a1)
        metafopy.baujat(res, las=2, ax=a2)
        assert all(label.get_rotation() == 0 for label in a1.get_yticklabels())
        assert all(label.get_rotation() == 90 for label in a2.get_xticklabels())
        assert all(label.get_rotation() == 0 for label in a2.get_yticklabels())
    finally:
        matplotlib.pyplot.close(fig)
    with pytest.raises(metafopy.errors.InputError, match='las'):
        metafopy.baujat(res, las=4)


def test_labbe_bty(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    fit = metafopy.rma(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'), measure='RR')
    metafopy.labbe(fit, bty='l', ai='tpos', bi='tneg', ci_counts='cpos', di='cneg', ax=ax)
    assert visible_spines(ax) == {'left', 'bottom'}


def test_plot_selmodel_bty_and_col(meta: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    sel = typing.cast(metafopy.SelmodelResult, metafopy.selmodel(meta, type='halfnorm'))
    metafopy.plot_selmodel(sel, bty='l', col='red', ax=ax)
    assert visible_spines(ax) == {'left', 'bottom'}
    assert any(line.get_color() == '#FF0000' or line.get_color() == 'red' for line in ax.lines)


def test_qqnorm_main(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.qqnorm(res, main='Residuals', envelope=False, ax=ax)
    assert ax.get_title() == 'Residuals'


def test_qqnorm_default_title(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.qqnorm(res, envelope=False, ax=ax)
    assert ax.get_title() == 'Normal Q-Q Plot'


def test_funnel_main(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.funnel(res, main='Funnel', ax=ax)
    assert ax.get_title() == 'Funnel'


def test_regplot_xaxt(meta: metafopy.RMAResult) -> None:
    fig, (a1, a2) = matplotlib.pyplot.subplots(1, 2)
    try:
        metafopy.regplot(meta, mod='ablat', ax=a1)
        metafopy.regplot(meta, mod='ablat', xaxt='n', ax=a2)
        assert len(a1.get_xticks()) > 0
        assert len(a2.get_xticks()) == 0
        assert len(a2.get_yticks()) > 0
    finally:
        matplotlib.pyplot.close(fig)


def test_profile_passes_graphics_arguments_to_the_plot(res: metafopy.RMAResult) -> None:
    fig, (a1, a2) = matplotlib.pyplot.subplots(1, 2)
    try:
        metafopy.profile(res, steps=8, plot=True, ax=a1)
        metafopy.profile(res, steps=8, plot=True, ax=a2, cline=True, pch=1, lwd=3, cex=2)
        assert len(a2.lines) > len(a1.lines)
        oline = next(line for line in a2.lines if line.get_gid() == 'oline')
        assert oline.get_linewidth() == 3
        sizes = {line.get_markersize() for line in a2.lines if line.get_gid() == 'point'}
        assert sizes
        assert all(size > 8 for size in sizes)
    finally:
        matplotlib.pyplot.close(fig)


def test_plot_profile_log_axes(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    prof = metafopy.profile(res, xlim=(0.01, 1.0), steps=8, exp=True)
    metafopy.plot_profile(prof, log='xy', ax=ax)
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'
    with pytest.raises(metafopy.errors.InputError, match='positive'):
        metafopy.plot_profile(metafopy.profile(res, xlim=(0.0, 1.0), steps=5), log='x', ax=matplotlib.pyplot.subplots()[1])
    matplotlib.pyplot.close('all')
