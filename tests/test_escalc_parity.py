"""escalc() against metafor: every case in tests/reference/escalc/escalc.json (see tests/r/generate_escalc.R)."""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metafopy import escalc

FIXTURE = Path(__file__).parent / 'reference' / 'escalc' / 'escalc.json'

# RTET/ZTET go through two optimisers and a numerical Hessian; the ports follow
# R's iterates, but floating-point differences in the bivariate normal CDF can
# shift the last digits.
LOOSE = {'RTET': 1e-6, 'ZTET': 1e-6, 'R2F': 1e-7, 'ZR2F': 1e-7}


@cache
def _cases() -> dict[str, dict[str, Any]]:
    with FIXTURE.open() as fh:
        data: dict[str, dict[str, Any]] = json.load(fh)['cases']
    return data


def _to_py(value: Any) -> Any:
    if isinstance(value, list):
        if all(isinstance(v, bool) for v in value):
            return np.array(value, dtype=np.bool_)
        return np.array([np.nan if v is None else v for v in value], dtype=np.float64)
    return value


def _kwargs(case: dict[str, Any]) -> dict[str, Any]:
    kwargs = {k: _to_py(v) for k, v in case['args'].items()}
    opts = dict(case['opts'])
    for name in ('subset', 'include'):
        if name in opts:
            # logical vectors keep their meaning; R's 1-based positions are not used in the fixtures
            opts[name] = np.array(opts[name], dtype=np.bool_)
    if 'flip' in opts:
        opts['flip'] = _to_py(opts['flip'])
    return {**kwargs, **opts}


@pytest.mark.parametrize('name', sorted(_cases()))
def test_escalc_matches_metafor(name: str) -> None:
    case = _cases()[name]
    kwargs = _kwargs(case)
    if 'error' in case:
        with pytest.raises((ValueError, TypeError)):
            escalc(case['measure'], **kwargs)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = escalc(case['measure'], **kwargs)
    rtol = LOOSE.get(case['measure'], 1e-10)
    expected_yi = np.array([np.nan if v is None else v for v in np.atleast_1d(case['yi'])], dtype=float)
    expected_vi = np.array([np.nan if v is None else v for v in np.atleast_1d(case['vi'])], dtype=float)
    np.testing.assert_allclose(res['yi'].to_numpy(), expected_yi, rtol=rtol, atol=1e-14, equal_nan=True)
    got_vi = res['vi'].to_numpy()
    if case['measure'] in ('RTET', 'ZTET'):
        # with a zero cell the likelihood is flat in rho and metafor's variance is the
        # inverse of a Hessian entry that is pure rounding noise (e.g. 1.8e-9); skip those rows
        cells = np.column_stack([kwargs[c] for c in ('ai', 'bi', 'ci', 'di')])
        keep = ~(cells == 0).any(axis=1)
        got_vi, expected_vi = got_vi[keep], expected_vi[keep]
    np.testing.assert_allclose(got_vi, expected_vi, rtol=rtol, atol=1e-14, equal_nan=True)
    if case.get('ni') is not None:
        expected_ni = np.array([np.nan if v is None else v for v in np.atleast_1d(case['ni'])], dtype=float)
        np.testing.assert_allclose(res.attrs['ni'], expected_ni, equal_nan=True)
    # metafor warns about unusually large SMDs only once per distinct yi vector
    # and session, so that warning is checked separately (test_escalc.py)
    raw = case.get('warnings', [])
    expected_warnings = [w for w in ([raw] if isinstance(raw, str) else raw) if 'unusually large' not in w]
    got = [str(w.message) for w in caught if 'unusually large' not in str(w.message) and 'tetrachoric' not in str(w.message)]
    assert bool(expected_warnings) == bool(got), (expected_warnings, got)
