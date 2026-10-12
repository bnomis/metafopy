"""Numerical parity of rma_uni with R metafor across all fixture cases."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from metafopy import confint, datasets, predict, rma_uni

from reference_data import case_names, load_case

RTOL = 1e-7
ATOL = 1e-9


def _fits() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for case in case_names():
        for i, _ in enumerate(load_case(case)['fits']):
            out.append((case, i))
    return out


def _id(param: tuple[str, int]) -> str:
    case, i = param
    f = load_case(case)['fits'][i]
    return f'{case}-{f["method"]}-{f["test"]}-{"mods" if f["mods"] else "int"}'


def _arr(value: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def _close(actual: Any, expected: Any, what: str, rtol: float = RTOL) -> None:
    np.testing.assert_allclose(_arr(actual), _arr(expected), rtol=rtol, atol=ATOL, equal_nan=True, err_msg=what)


FITS = _fits()


@pytest.mark.parametrize('param', FITS, ids=[_id(p) for p in FITS])
def test_rma_uni_matches_metafor(param: tuple[str, int]) -> None:
    case, i = param
    fx = load_case(case)
    spec = fx['fits'][i]
    ref = spec['result']
    dat = datasets.load(fx['dataset'])
    dat['yi'] = _arr(fx['yi'])
    dat['vi'] = _arr(fx['vi'])

    res = rma_uni(data=dat, mods=spec['mods'], method=spec['method'], test=spec['test'])

    assert res.k == ref['k']
    assert res.p == ref['p']
    names = ref['coef_names']
    assert list(res.coef_names) == ([names] if isinstance(names, str) else names)
    for attr in ('beta', 'se', 'zval', 'pval', 'ci_lb', 'ci_ub', 'tau2', 'QE', 'QEp', 'I2', 'H2'):
        _close(getattr(res, attr), ref[attr], attr)
    _close(res.vb, ref['vb'], 'vb')
    if ref['se_tau2'] is not None:
        _close(res.se_tau2, ref['se_tau2'], 'se_tau2')
    _close(res.QM, ref['QM'], 'QM')
    _close(res.QMp, ref['QMp'], 'QMp')
    if ref['R2'] is None:
        assert res.R2 is None
    else:
        _close(res.R2, ref['R2'], 'R2')
    _close(res.fit_stats_ml.as_tuple(), ref['fit_stats_ml'], 'fitstats ML')
    _close(res.fit_stats_reml.as_tuple(), ref['fit_stats_reml'], 'fitstats REML')
    _close(res.weights_(), ref['weights'], 'weights')
    _close(res.hatvalues(), ref['hat'], 'hat')

    if 'predict' in ref:
        pr = predict(res)
        for attr in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
            _close(getattr(pr, attr), ref['predict'][attr], f'predict {attr}')

    if 'confint_random' in ref:
        ci = confint(res).random
        assert ci is not None
        for col, key in (('estimate', 'estimate'), ('ci.lb', 'ci_lb'), ('ci.ub', 'ci_ub')):
            _close(ci[col].to_numpy(), ref['confint_random'][key], f'confint {col}')
