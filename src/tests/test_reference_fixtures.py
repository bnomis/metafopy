"""Sanity checks on the committed R fixtures and the bundled datasets they use."""

import numpy
import pytest

import metafopy
import metafopy.datasets

import reference_data

CASES = reference_data.case_names()


def test_enough_fixtures() -> None:
    assert len({reference_data.load_case(c)['dataset'] for c in CASES}) >= 20


@pytest.mark.parametrize('case', CASES)
def test_fixture_dataset_is_bundled(case: str) -> None:
    fx = reference_data.load_case(case)
    dat = metafopy.datasets.load(fx['dataset'])
    assert len(dat) == len(fx['yi'])
    if fx['escalc'] is None:
        numpy.testing.assert_allclose(dat['yi'], numpy.asarray(fx['yi'], dtype=float), rtol=1e-12)


@pytest.mark.parametrize('case', CASES)
def test_fixture_fits_succeeded(case: str) -> None:
    fx = reference_data.load_case(case)
    errors = [f for f in fx['fits'] if 'error' in f['result']]
    assert not errors


@pytest.mark.parametrize('case', [c for c in CASES if any(f['mods'] for f in reference_data.load_case(c)['fits'])])
def test_formula_columns_match_r(case: str) -> None:
    fx = reference_data.load_case(case)
    fit = next(f for f in fx['fits'] if f['mods'])
    dat = metafopy.datasets.load(fx['dataset'])
    dat['yi'] = numpy.asarray(fx['yi'], dtype=float)
    dat['vi'] = numpy.asarray(fx['vi'], dtype=float)
    md = metafopy.MetaData.from_inputs(data=dat, mods=fit['mods'])
    names = fit['result']['coef_names']
    assert list(md.X_names) == ([names] if isinstance(names, str) else names)
    assert md.k == fit['result']['k']
