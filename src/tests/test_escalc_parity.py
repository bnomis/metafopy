"""escalc() against metafor: every case in src/tests/reference/escalc/escalc.json (see src/tests/r/generate_escalc.R)."""

import functools
import json
import pathlib
import typing
import warnings

import numpy
import pytest

import metafopy

FIXTURE = pathlib.Path(__file__).parent / 'reference' / 'escalc' / 'escalc.json'

# RTET/ZTET go through two optimisers and a numerical Hessian; the ports follow
# R's iterates, but floating-point differences in the bivariate normal CDF can
# shift the last digits.
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
LOOSE = {'RTET': 1e-6, 'ZTET': 1e-6, 'R2F': 1e-7, 'ZR2F': 1e-7}


@functools.cache
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def cases() -> dict[str, dict[str, typing.Any]]:
    with FIXTURE.open() as fh:
        data: dict[str, dict[str, typing.Any]] = json.load(fh)['cases']
    return data


def to_py(value: typing.Any) -> typing.Any:
    if isinstance(value, list):
        if all(isinstance(v, bool) for v in value):
            return numpy.array(value, dtype=numpy.bool_)
        return numpy.array([numpy.nan if v is None else v for v in value], dtype=numpy.float64)
    return value


def case_kwargs(case: dict[str, typing.Any]) -> dict[str, typing.Any]:
    kwargs = {k: to_py(v) for k, v in case['args'].items()}
    opts = dict(case['opts'])
    for name in ('subset', 'include'):
        if name in opts:
            # logical vectors keep their meaning; R's 1-based positions are not used in the fixtures
            opts[name] = numpy.array(opts[name], dtype=numpy.bool_)
    if 'flip' in opts:
        opts['flip'] = to_py(opts['flip'])
    return {**kwargs, **opts}


@pytest.mark.parametrize('name', sorted(cases()))
def test_escalc_matches_metafor(name: str) -> None:
    case = cases()[name]
    kwargs = case_kwargs(case)
    if 'error' in case:
        with pytest.raises((ValueError, TypeError)):
            metafopy.escalc(case['measure'], **kwargs)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = metafopy.escalc(case['measure'], **kwargs)
    rtol = LOOSE.get(case['measure'], 1e-10)
    expected_yi = numpy.array([numpy.nan if v is None else v for v in numpy.atleast_1d(case['yi'])], dtype=float)
    expected_vi = numpy.array([numpy.nan if v is None else v for v in numpy.atleast_1d(case['vi'])], dtype=float)
    numpy.testing.assert_allclose(res['yi'].to_numpy(), expected_yi, rtol=rtol, atol=1e-14, equal_nan=True)
    got_vi = res['vi'].to_numpy()
    if case['measure'] in ('RTET', 'ZTET'):
        # with a zero cell the likelihood is flat in rho and metafor's variance is the
        # inverse of a Hessian entry that is pure rounding noise (e.g. 1.8e-9); skip those rows
        cells = numpy.column_stack([kwargs[c] for c in ('ai', 'bi', 'ci', 'di')])
        keep = ~(cells == 0).any(axis=1)
        got_vi, expected_vi = got_vi[keep], expected_vi[keep]
    numpy.testing.assert_allclose(got_vi, expected_vi, rtol=rtol, atol=1e-14, equal_nan=True)
    if case.get('ni') is not None:
        expected_ni = numpy.array([numpy.nan if v is None else v for v in numpy.atleast_1d(case['ni'])], dtype=float)
        numpy.testing.assert_allclose(res.attrs['ni'], expected_ni, equal_nan=True)
    # metafor warns about unusually large SMDs only once per distinct yi vector
    # and session, so that warning is checked separately (test_escalc.py)
    raw = case.get('warnings', [])
    expected_warnings = [w for w in ([raw] if isinstance(raw, str) else raw) if 'unusually large' not in w]
    got = [str(w.message) for w in caught if 'unusually large' not in str(w.message) and 'tetrachoric' not in str(w.message)]
    assert bool(expected_warnings) == bool(got), (expected_warnings, got)
