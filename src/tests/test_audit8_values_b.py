"""Regression tests for the audit-8 fixes of rarely hit wrong values (values taken from R metafor 5.2-1 when the audit was run)."""

import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.errors


def na_data() -> pandas.DataFrame:
    return pandas.DataFrame({'yi': [numpy.nan, 1, 3, 2, 5, 4, 6], 'vi': [1, numpy.nan, 1, 1, 1, 1, 1], 'xi': [0, 1, numpy.nan, 3, 4, 5, 6]})


# simulate(res, nsim=2, seed=1234) in R: options(na.action = "na.omit") gives the last five rows, "na.pass" and "na.exclude" all seven
SIMULATE_NA = numpy.array(
    [
        [2.0914615, 4.800145],
        [4.4386541, 3.091256],
        [5.7146521, 3.135699],
        [0.2911263, 3.107523],
        [4.6785057, 2.592727],
    ]
)


@pytest.mark.parametrize('na_action', ['omit', 'exclude', 'pass', 'fail'])
def test_simulate_na_action(na_action: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', data=na_data())
    sims = metafopy.simulate(res, nsim=2, seed=1234, na_action=na_action)  # type: ignore[arg-type]
    if na_action == 'omit':
        assert sims.shape == (5, 2)
        numpy.testing.assert_allclose(sims.to_numpy(), SIMULATE_NA, rtol=1e-6)
    else:
        assert sims.shape == (7, 2)
        assert list(sims.index) == ['1', '2', '3', '4', '5', '6', '7']
        assert sims.iloc[:2].isna().all().all()
        numpy.testing.assert_allclose(sims.iloc[2:].to_numpy(), SIMULATE_NA, rtol=1e-6)


def test_simulate_na_action_unknown() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', data=na_data())
    with pytest.raises(metafopy.errors.InputError, match=r'na\.action'):
        metafopy.simulate(res, na_action='bogus')  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('name', 'column', 'missing', 'empty', 'na', 'none'),
    [
        ('michael2013', 'Compensation', 0, 0, 0, 1),
        ('bakdash2021', 'DTIC.link', 501, 0, 18, 0),
        ('axfors2021', 'blinding_exact', 0, 0, 0, 16),
        ('stowe2010', 't3', 0, 28, 0, 0),
        ('linde2015', 'treatment3', 0, 58, 0, 0),
    ],
)
def test_dataset_strings_that_look_missing(name: str, column: str, missing: int, empty: int, na: int, none: int) -> None:
    """The strings "", "NA" and "None" of metadat's text columns are values, not missing (counts from R)."""
    col = metafopy.datasets.load(name)[column]
    assert int(col.isna().sum()) == missing
    assert int((col == '').sum()) == empty
    assert int((col == 'NA').sum()) == na
    assert int((col == 'None').sum()) == none


def test_ls_z_f_drops_redundant_scale_predictors() -> None:
    """Z.f has the columns of Z: R drops 'allocsystematic' (all 0 in the first 8 studies) from both."""
    dat = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))
    with pytest.warns(UserWarning, match='Redundant predictors dropped from the scale model'):
        res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ alloc', data=dat, subset=list(range(8)))
    assert res.Z.shape == (8, 2)
    assert res.Z_f is not None
    assert res.alpha_names == ('intrcpt', 'allocrandom')
    numpy.testing.assert_array_equal(res.Z_f, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]])
