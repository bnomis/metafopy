"""plot_permutest() of location-scale permutation tests (``alpha=``, ``QS=``) against R
(src/tests/reference/phase7/permutest_ls_plots.json, written by src/tests/r/generate_permutest_ls_plots.R).

As in test_permutest_vif_plots.py, the permutation distributions R drew are put into the Python result, so both plot the same data.
"""

import collections.abc
import dataclasses
import json
import pathlib
import typing
import warnings

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil
import test_permutest_vif_plots

import metafopy
import metafopy.methods.ls_permutest

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase7' / 'permutest_ls_plots.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def perm() -> metafopy.methods.ls_permutest.LSPermutestResult:
    ref = REF['permutest']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=refutil.bcg())
        result = metafopy.permutest(fit, iter=20, seed=1)
    assert isinstance(result, metafopy.methods.ls_permutest.LSPermutestResult)

    def frame(key: str, names: list[str]) -> pandas.DataFrame:
        return pandas.DataFrame({n: ref[key][n] for n in names})

    beta_names, alpha_names = list(ref['coef_names']), list(ref['alpha_names'])
    return dataclasses.replace(
        result,
        zval_perm=frame('zval_perm', beta_names),
        beta_perm=frame('beta_perm', beta_names),
        zval_alpha_perm=frame('zval_alpha_perm', alpha_names),
        alpha_perm=frame('alpha_perm', alpha_names),
        QM_perm=numpy.asarray(ref['QM_perm'], dtype=float),
        QS_perm=numpy.asarray(ref['QS_perm'], dtype=float),
        zval=numpy.asarray(ref['zval'], dtype=float),
        beta=numpy.asarray(ref['beta'], dtype=float),
        zval_alpha=numpy.asarray(ref['zval_alpha'], dtype=float),
        alpha=numpy.asarray(ref['alpha'], dtype=float),
        QM=float(ref['QM']),
        QS=float(ref['QS']),
    )


@pytest.fixture
# auditme: accept[paired-sibling-helpers] each plot test module keeps its own figure fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


CASES: list[tuple[str, dict[str, typing.Any]]] = [
    ('p_qs', {'beta': [], 'alpha': [], 'QS': True, 'lwd': (2, 0, 1, 4)}),
    ('p_alpha', {'beta': [], 'alpha': 1, 'lwd': (2, 1, 1, 4)}),
    ('p_alpha_coef', {'beta': [], 'alpha': 'year', 'stat': 'coef'}),
]


# auditme: accept[literal-duplication] the parametrize names repeat as in test_permutest_vif_plots.py
@pytest.mark.parametrize(('key', 'kwargs'), CASES, ids=[c[0] for c in CASES])
def test_plot_matches_metafor(key: str, kwargs: dict[str, typing.Any], perm: metafopy.methods.ls_permutest.LSPermutestResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.plot_permutest(perm, ax=ax, **kwargs)
    test_permutest_vif_plots.check_plot(ax, REF[key])


def test_panels_and_titles(perm: metafopy.methods.ls_permutest.LSPermutestResult) -> None:
    out = metafopy.plot_permutest(perm, QM=True, QS=True)
    assert out.names == ('QM', 'ablat', 'QS', 'year')
    assert [a.get_title() for a in out.axes] == REF['mains_all']
    matplotlib.pyplot.close('all')
    out = metafopy.plot_permutest(perm)
    assert [a.get_title() for a in out.axes] == REF['mains_default']
    matplotlib.pyplot.close('all')
    skipped = dataclasses.replace(perm, skip_alpha=True)
    # auditme: accept[literal-duplication] the coefficient names come from the R fixture
    assert metafopy.plot_permutest(skipped, QS=True).names == ('ablat',)
    matplotlib.pyplot.close('all')
