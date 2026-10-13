"""summary_escalc() against metafor's summary.escalc() (cases in tests/reference/extras/transf.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from metafopy import escalc
from metafopy.effects._summary import summary_escalc
from metafopy.transf import transf_exp_int, transf_ipft, transf_ztor, transf_ztor_int

REF: dict[str, Any] = json.loads((Path(__file__).parent / 'reference' / 'extras' / 'transf.json').read_text())['summary']


def _arr(v: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([np.nan if e is None else e for e in np.atleast_1d(v)], dtype=np.float64)


def _names(v: Any) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def _check(res: pd.DataFrame, ref: dict[str, Any], rtol: float = 1e-10) -> None:
    # metafor's default out.names "ci.lb"/"ci.ub" are "ci_lb"/"ci_ub" here
    cols = [c.replace('.', '_') for c in ref['columns']]
    assert list(res.columns) == cols
    for rname, col in zip(ref['columns'], cols, strict=True):
        np.testing.assert_allclose(res[col].to_numpy(float), _arr(ref[rname]), rtol=rtol, atol=1e-14, equal_nan=True, err_msg=col)
    for key in ('yi_names', 'vi_names', 'sei_names'):
        assert res.attrs[key] == _names(ref['attrs'][key])
    assert res.attrs['ci_lb_names'] == [n.replace('.', '_') for n in _names(ref['attrs']['ci_lb_names'])]


def _escalc(spec: dict[str, Any]) -> pd.DataFrame:
    kwargs: dict[str, Any] = {k: (v if k in ('measure', 'var_names') else _arr(v)) for k, v in spec.items()}
    return escalc(**kwargs)


@pytest.fixture(scope='module')
def pft() -> pd.DataFrame:
    return _escalc(REF['pft']['input'])


@pytest.fixture(scope='module')
def odds() -> pd.DataFrame:
    return _escalc(REF['or']['input'])


@pytest.fixture(scope='module')
def zcor() -> pd.DataFrame:
    return _escalc(REF['zcor']['input'])


def test_default(pft: pd.DataFrame) -> None:
    res = summary_escalc(pft)
    _check(res, REF['pft']['default'])
    assert res.attrs['measure'] == 'PFT'


def test_level_and_h0(pft: pd.DataFrame) -> None:
    _check(summary_escalc(pft, level=90, H0=0.3), REF['pft']['level90_H0'])


def test_transf_with_per_study_argument(pft: pd.DataFrame) -> None:
    np.testing.assert_array_equal(pft.attrs['ni'], _arr(REF['pft']['ni']))
    res = summary_escalc(pft, transf=transf_ipft, ni=pft.attrs['ni'])
    _check(res, REF['pft']['transf_ipft'])
    assert 'measure' not in res.attrs
    _check(summary_escalc(pft, transf=transf_ipft, ni=pft.attrs['ni'], append=False), REF['pft']['transf_noappend'])


def test_out_names_and_no_append(pft: pd.DataFrame) -> None:
    res = summary_escalc(pft, out_names=('se', 'z', 'p', 'lo', 'hi'))
    _check(res, REF['pft']['out_names'])
    _check(summary_escalc(pft, append=False), REF['pft']['noappend'])


def test_transf_exp_olim_and_integral(odds: pd.DataFrame) -> None:
    _check(summary_escalc(odds, transf=np.exp), REF['or']['transf_exp'])
    _check(summary_escalc(odds, transf=np.exp, olim=(1, 0.5)), REF['or']['olim'])
    _check(summary_escalc(odds, transf=transf_exp_int, targs=0.2), REF['or']['exp_int'])


def test_replace_false_fills_only_missing(odds: pd.DataFrame) -> None:
    s1 = summary_escalc(odds)
    s1.loc[1, 'sei'] = np.nan
    s1.loc[[0, 2], 'ci_lb'] = np.nan
    _check(s1, REF['or']['replace_false_input'])
    _check(summary_escalc(s1, replace=False, level=80), REF['or']['replace_false'])


def test_var_names_from_attrs(zcor: pd.DataFrame) -> None:
    _check(summary_escalc(zcor, transf=transf_ztor), REF['zcor']['ztor'])
    _check(summary_escalc(zcor, transf=transf_ztor_int, targs=0.05), REF['zcor']['ztor_int'], rtol=1e-6)


def test_errors(pft: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match='length 5'):
        summary_escalc(pft, out_names=('a', 'b'))
    with pytest.raises(ValueError, match='between 0 and 100'):
        summary_escalc(pft, level=120)
    with pytest.raises(ValueError, match='Cannot find'):
        summary_escalc(pft, var_names=('yi', 'nope'))
    with pytest.raises(ValueError, match="'olim'"):
        summary_escalc(pft, olim=(0,))
