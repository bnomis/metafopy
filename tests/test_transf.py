"""transf_* against metafor: every case in tests/reference/extras/transf.json (see tests/r/generate_transf.R)."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import metafopy.transf as tm

FIXTURE = Path(__file__).parent / 'reference' / 'extras' / 'transf.json'

# integral (integrate()) and root/mode (uniroot(), optimize()) based functions;
# transf.exp.int and transf.exp.mode are closed forms in metafor
NUMERIC = {'_int', '_mode'}
CLOSED_FORM = {'transf_exp_int', 'transf_exp_mode'}


@cache
def _cases() -> dict[str, dict[str, Any]]:
    with FIXTURE.open() as fh:
        data: dict[str, dict[str, Any]] = json.load(fh)['cases']
    return data


def _num(v: Any) -> float:
    if v is None:
        return np.nan
    if isinstance(v, str):
        return {'Inf': np.inf, '-Inf': -np.inf}[v]
    return float(v)


def _decode(v: Any) -> Any:
    if isinstance(v, list):
        return np.array([_num(e) for e in v], dtype=np.float64)
    if isinstance(v, dict):
        return {k: _decode(e) for k, e in v.items()}
    if isinstance(v, str):
        return _num(v)
    return v


def _func(name: str) -> Callable[..., Any]:
    f: Callable[..., Any] = getattr(tm, name.replace('.', '_'))
    return f


def _tolerance(name: str) -> tuple[float, float]:
    py = name.replace('.', '_')
    if py not in CLOSED_FORM and any(py.endswith(s) for s in NUMERIC):
        return 1e-6, 1e-12
    return 1e-10, 1e-15


@pytest.mark.parametrize('name', sorted(_cases()))
def test_transf_matches_metafor(name: str) -> None:
    case = _cases()[name]
    f = _func(case['fun'])
    args = {k: _decode(v) for k, v in case['args'].items()}

    def call() -> Any:
        if case['each']:
            xi = np.atleast_1d(args.pop('xi'))
            return np.array([f(x, **args)[0] for x in xi])
        return f(**args)

    if 'error' in case:
        with pytest.raises(ValueError, match=r"'targs'|'tau2'|Length|length|'pi'|'pc'|negative|scalar|sample sizes"):
            call()
        return
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # numpy floating-point warnings must not leak
        out = call()
    expected = np.atleast_1d(_decode(case['out']))
    rtol, atol = _tolerance(case['fun'])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol, equal_nan=True)


def test_every_metafor_function_is_covered() -> None:
    covered = {c['fun'].replace('.', '_') for c in _cases().values()}
    assert covered == set(tm.__all__)


def test_scalars_and_shapes() -> None:
    out = tm.transf_ztor(0.5)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(np.tanh(0.5))
    grid = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert tm.transf_logit(grid).shape == (2, 2)
    assert tm.transf_ilogit_int(grid, targs={'tau2': 0.1}).shape == (2, 2)
    assert tm.transf_ztor_mode(grid, targs=0.1).shape == (2, 2)
    np.testing.assert_allclose(tm.transf_pft(grid, 10), tm.transf_pft(grid.ravel(), 10).reshape(2, 2))


def test_targs_forms_agree() -> None:
    x = np.array([-1.0, 0.0, 0.7])
    by_dict = tm.transf_ilogit_int(x, targs={'tau2': 0.3, 'lower': -2, 'upper': 3})
    by_seq = tm.transf_ilogit_int(x, targs=[0.3, -2, 3])
    np.testing.assert_array_equal(by_dict, by_seq)
    np.testing.assert_array_equal(tm.transf_ztor_mode(x, targs=0.2), tm.transf_ztor_mode(x, targs={'tau2': 0.2}))
    # tau2 below machine epsilon is treated as 0 (the plain back-transformation)
    np.testing.assert_array_equal(tm.transf_ztor_int(x, targs=1e-20), np.tanh(x))


def test_mode_with_zero_tau2_is_elementwise() -> None:
    # metafor returns tanh(xi) length(xi) times for vector xi; here the result is elementwise
    x = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(tm.transf_ztor_mode(x, targs=0), np.tanh(x))
    np.testing.assert_array_equal(tm.transf_iabt_mode(x, targs=0), tm.transf_iabt(x))


def test_ipft_hm_accepts_sample_sizes_directly() -> None:
    x = np.array([0.2, 0.5])
    np.testing.assert_array_equal(tm.transf_ipft_hm(x, targs={'ni': [10, 20]}), tm.transf_ipft_hm(x, targs=[10, 20]))
