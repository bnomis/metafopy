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
