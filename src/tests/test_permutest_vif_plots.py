"""plot_permutest() and plot_vif() checked against R metafor (tests/reference/phase2/permutest_vif_plots.json, written by
tests/r/generate_permutest_vif_plots.R).

R records the histograms it draws (the breaks and the densities or counts of the bars and of each shaded tail), the dashed and the
reference lines and the plot region; the permutation distributions and the simulated VIFs behind them are in the same file, so that the
Python plots draw exactly the same data.
"""

import collections.abc
import dataclasses
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.methods.permutest
import metafopy.methods.vif
from metafopy.plot import PlotPermutestResult, PlotVifResult

REF = refutil.load_phase2('permutest_vif_plots')


@pytest.fixture(scope='module')
def fit() -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat + year', data=refutil.bcg())


@pytest.fixture(scope='module')
def perm(fit: metafopy.RMAResult) -> metafopy.methods.permutest.PermutestResult:
    """A permutation test whose distributions are those R drew."""
    ref = REF['permutest']
    names = list(ref['coef_names'])
    result = metafopy.permutest(fit, iter=20, seed=1)
    assert isinstance(result, metafopy.methods.permutest.PermutestResult)
    return dataclasses.replace(
        result,
        zval_perm=pandas.DataFrame({n: ref['zval_perm'][n] for n in names}),
        beta_perm=pandas.DataFrame({n: ref['beta_perm'][n] for n in names}),
        QM_perm=numpy.asarray(ref['QM_perm'], dtype=float),
        zval=numpy.asarray(ref['zval'], dtype=float),
        beta=numpy.asarray(ref['beta'], dtype=float),
        QM=float(ref['QM']),
    )


@pytest.fixture(scope='module')
def vifs(fit: metafopy.RMAResult) -> metafopy.methods.vif.VIFResult:
    ref = REF['vif']
    result = metafopy.vif(fit, btt=1, sim=20, progbar=False, seed=1)
    sim = numpy.column_stack([numpy.asarray(c, dtype=float) for c in ref['sim']])
    entry = dataclasses.replace(result.vif[0], vif=float(refutil.as_list(ref['vifs'])[0]))
    return dataclasses.replace(result, vif=(entry,), sim=sim)


def test_seeded_permutations_are_those_of_r(fit: metafopy.RMAResult) -> None:
    # R: set.seed(1234); permutest(res, iter = 500)
    ref = REF['permutest']
    metafopy.set_seed(1234)
    result = metafopy.permutest(fit, iter=500)
    assert isinstance(result, metafopy.methods.permutest.PermutestResult)
    for name in ref['coef_names']:
        numpy.testing.assert_allclose(result.zval_perm[name], ref['zval_perm'][name], rtol=1e-8, atol=1e-8)


def test_seeded_simulated_vifs_are_those_of_r(fit: metafopy.RMAResult) -> None:
    # R: set.seed(4321); vif(res, btt = 2, sim = TRUE); the model comes from a formula, so the variables of the data are shuffled
    metafopy.set_seed(4321)
    result = metafopy.vif(fit, btt=1, sim=True, progbar=False)
    assert result.sim is not None
    assert result.sim_type == 'data'
    numpy.testing.assert_allclose(result.sim[:, 0], REF['vif']['sim'][0], rtol=1e-6, atol=1e-8)


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def artists(ax: matplotlib.axes.Axes, gid: str) -> list[typing.Any]:
    return [a for a in ax.get_children() if a.get_gid() == gid]


def bars(ax: matplotlib.axes.Axes) -> list[tuple[float, float, float]]:
    out = []
    for r in artists(ax, 'rect'):
        left, bottom, right, top = r.meta['xy']
        assert bottom == 0
        out.append((left, right, top))
    return out


def check_plot(ax: matplotlib.axes.Axes, ref: dict[str, typing.Any], freq: bool = False) -> None:
    events = ref['events']
    drawn = bars(ax)
    expected: list[tuple[float, float, float]] = []
    for ev in events:
        if ev['type'] in ('hist', 'tail'):
            breaks = refutil.as_list(ev['breaks'])
            heights = refutil.as_list(ev['counts'] if freq else ev['density'])
            expected += [(breaks[i], breaks[i + 1], heights[i]) for i in range(len(breaks) - 1)]
    assert len(drawn) == len(expected)
    refutil.close([v for row in drawn for v in row], [v for row in expected for v in row], 'bars')
    verticals = [x for ev in events if ev['type'] == 'abline' for x in refutil.as_list(ev['v'])]
    refutil.close([a.get_xdata()[0] for a in artists(ax, 'vline')], verticals, 'vertical lines')
    drawn_lines = artists(ax, 'line')
    r_lines = [ev for ev in events if ev['type'] == 'lines']
    assert len(drawn_lines) == len(r_lines)
    for line, ev in zip(drawn_lines, r_lines, strict=True):
        y = refutil.as_list(ev['y']) if not isinstance(ev['y'], dict) else None
        total = float(numpy.asarray(line.get_ydata()).sum())
        refutil.close(total, ev['y']['sum'] if y is None else sum(y), 'line values', rtol=1e-6)
        assert line.get_linewidth() == refutil.as_list(ev['lwd'])[0]
    refutil.close(ax.get_xlim(), ref['usr'][:2], 'x limits', rtol=1e-6)
    refutil.close(ax.get_ylim(), ref['usr'][2:], 'y limits', rtol=1e-6)


def test_distributions_are_those_of_r(perm: metafopy.methods.permutest.PermutestResult) -> None:
    assert perm.zval_perm.shape == (500, 3)
    assert perm.zval_perm.columns.tolist() == REF['permutest']['coef_names']


CASES: list[tuple[str, dict[str, typing.Any]]] = [
    ('p_default', {'beta': 'ablat'}),
    ('p_all_curves', {'beta': 'year', 'lwd': (2, 1, 1, 4), 'breaks': 10}),
    ('p_greater', {'beta': 'ablat', 'alternative': 'greater', 'lwd': (2, 1, 1, 4)}),
    ('p_less_trim', {'beta': 'ablat', 'alternative': 'less', 'trim': 0.1, 'adjust': 2, 'lwd': (1, 2, 0, 2)}),
    ('p_px2', {'beta': 'ablat', 'p2defn': 'px2', 'lwd': (2, 1, 1, 0)}),
    ('p_px2_other', {'beta': 'year', 'p2defn': 'px2'}),
    ('p_coef', {'beta': 'ablat', 'stat': 'coef', 'lwd': (2, 1, 1, 0)}),
    ('p_freq', {'beta': 'ablat', 'freq': True}),
]


@pytest.mark.parametrize(('key', 'kwargs'), CASES, ids=[c[0] for c in CASES])
def test_plot_permutest_matches_metafor(
    key: str, kwargs: dict[str, typing.Any], perm: metafopy.methods.permutest.PermutestResult, ax: matplotlib.axes.Axes
) -> None:
    out = metafopy.plot_permutest(perm, ax=ax, **kwargs)
    assert isinstance(out, PlotPermutestResult)
    assert out.axes == (ax,)
    check_plot(ax, REF[key], freq=bool(kwargs.get('freq')))


def test_plot_permutest_omnibus_matches_metafor(perm: metafopy.methods.permutest.PermutestResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.plot_permutest(perm, beta=[], QM=True, lwd=(2, 1, 1, 4), ax=ax)
    check_plot(ax, REF['p_qm'])
    assert ax.get_title() == 'Omnibus Test of Coefficients'


def test_plot_permutest_omnibus_with_one_degree_of_freedom_draws_the_theoretical_density(ax: matplotlib.axes.Axes) -> None:
    # the chi-square density with one degree of freedom is infinite at 0: drawing it used to raise OverflowError (metafor stops with an infinite ylim)
    one = metafopy.rma('yi', 'vi', mods='~ ablat', data=refutil.bcg())
    result = metafopy.permutest(one, iter=50, seed=1)
    assert isinstance(result, metafopy.methods.permutest.PermutestResult)
    metafopy.plot_permutest(result, beta=[], QM=True, lwd=(2, 3, 3, 4), ax=ax)
    assert numpy.isfinite(ax.get_ylim()).all()


def test_plot_permutest_default_draws_one_panel_per_coefficient(perm: metafopy.methods.permutest.PermutestResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.plot_permutest(perm, QM=True, legend=True, lwd=(2, 1, 1, 0), ax=ax)
    assert out.names == ('QM', 'ablat', 'year')
    assert len(out.axes) == 3
    assert [a.get_title() for a in out.axes] == ['Omnibus Test of Coefficients', 'Coefficient: ablat', 'Coefficient: year']
    assert [a.get_legend() is not None for a in out.axes] == [True, False, False]


def test_plot_permutest_rejects_bad_arguments(perm: metafopy.methods.permutest.PermutestResult, ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(metafopy.errors.InputError, match='trim'):
        metafopy.plot_permutest(perm, trim=0.5, ax=ax)
    with pytest.raises(metafopy.errors.InputError, match='at least one'):
        metafopy.plot_permutest(perm, beta=[], ax=ax)
    with pytest.raises(metafopy.errors.InputError, match="'nope'"):
        metafopy.plot_permutest(perm, beta='nope', ax=ax)


VIF_CASES: list[tuple[str, dict[str, typing.Any]]] = [
    ('v_default', {}),
    ('v_density', {'lwd': (2, 2), 'breaks': 8, 'trim': 0.1, 'adjust': 1.5}),
    ('v_freq', {'freq': True}),
]


@pytest.mark.parametrize(('key', 'kwargs'), VIF_CASES, ids=[c[0] for c in VIF_CASES])
def test_plot_vif_matches_metafor(key: str, kwargs: dict[str, typing.Any], vifs: metafopy.methods.vif.VIFResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.plot_vif(vifs, ax=ax, **kwargs)
    assert isinstance(out, PlotVifResult)
    ref = REF[key]
    check_plot(ax, {**ref, 'events': [e for e in ref['events'] if e['type'] != 'segments']}, freq=bool(kwargs.get('freq')))
    segments = [a for a in artists(ax, 'segment')]
    r_segments = [e for e in ref['events'] if e['type'] == 'segments']
    assert len(segments) == len(r_segments)
    for seg, ev in zip(segments, r_segments, strict=True):
        refutil.close(seg.get_xdata()[0], ev['x0'], 'observed VIF')


def test_plot_vif_without_lines(vifs: metafopy.methods.vif.VIFResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.plot_vif(vifs, lwd=(0, 0), ax=ax)
    ref = REF['v_none']
    check_plot(ax, ref)
    assert not artists(ax, 'segment')


def test_plot_vif_needs_simulation(fit: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(metafopy.errors.InputError, match='sim=True'):
        metafopy.plot_vif(metafopy.vif(fit), ax=ax)
