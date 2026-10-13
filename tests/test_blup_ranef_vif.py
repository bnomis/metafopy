"""blup(), ranef() and vif() for rma_uni fits, checked against R metafor (tests/r/generate_blup_vif.R)."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from metafopy import datasets, rma
from metafopy.effects import escalc
from metafopy.methods.blup import BlupResult, blup
from metafopy.methods.ranef import RanefResult, ranef
from metafopy.methods.vif import VIFResult, format_btt, vif

REF: dict[str, Any] = json.loads((Path(__file__).parent / 'reference' / 'extras' / 'blup_vif.json').read_text())
RTOL = 1e-8


def _close(actual: Any, expected: Any, what: str = '') -> None:
    np.testing.assert_allclose(
        np.atleast_1d(np.asarray(actual, dtype=float)),
        np.atleast_1d(np.asarray(expected, dtype=float)),
        rtol=RTOL,
        atol=1e-12,
        equal_nan=True,
        err_msg=what,
    )


def _as_list(value: Any) -> list[Any]:
    # jsonlite's auto_unbox turns length-one vectors into scalars
    return value if isinstance(value, list) else [value]


def _r_print(text: str) -> str:
    # metafor's printed tables end every line with a space
    return '\n'.join(line.rstrip() for line in text.split('\n'))


@pytest.fixture(scope='module')
def dat() -> pd.DataFrame:
    d = datasets.load('bcg')
    a, b, c, e = (d[col].to_numpy(float) for col in ('tpos', 'tneg', 'cpos', 'cneg'))
    d['yi'] = np.log(a / (a + b)) - np.log(c / (c + e))
    d['vi'] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d['n'] = a + b + c + e
    d['label'] = d['author'] + ' ' + d['year'].astype(str)
    return d


# -- BLUPs ------------------------------------------------------------------------

BLUP_CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'blup_REML': ({}, {}),
    'blup_DL': ({'method': 'DL'}, {}),
    'blup_ML': ({'method': 'ML'}, {}),
    'blup_mods': ({'mods': '~ ablat + year'}, {}),
    'blup_EE': ({'method': 'EE'}, {}),
    'blup_knha': ({'mods': '~ ablat', 'test': 'knha'}, {}),
    'blup_level90': ({}, {'level': 90}),
    'blup_level90_fit': ({'level': 90}, {}),
    'blup_exp': ({}, {'transf': np.exp}),
    'blup_slab': ({'slab': 'label'}, {}),
}


def _check_list(out: BlupResult, ref: dict[str, Any], what: str) -> None:
    for attr in ('pred', 'se', 'pi_lb', 'pi_ub'):
        _close(getattr(out, attr), ref[attr], f'{what}: {attr}')
    assert list(out.slab) == ref['slab']


@pytest.mark.parametrize('key', list(BLUP_CASES))
def test_blup_ranef_match_metafor(dat: pd.DataFrame, key: str) -> None:
    fit_kw, call_kw = BLUP_CASES[key]
    res = rma(data=dat, **fit_kw)
    b = blup(res, **call_kw)
    r = ranef(res, **call_kw)
    assert isinstance(b, BlupResult)
    assert isinstance(r, RanefResult)
    _check_list(b, REF[key]['blup'], f'{key} blup')
    _check_list(r, REF[key]['ranef'], f'{key} ranef')


@pytest.fixture(scope='module')
def res_na(dat: pd.DataFrame) -> Any:
    dna = dat.copy()
    dna.loc[2, 'yi'] = np.nan
    dna.loc[6, 'ablat'] = np.nan
    return rma(data=dna, mods='~ ablat')


def test_blup_missing_values_omit(res_na: Any) -> None:
    _check_list(blup(res_na), REF['blup_na_omit']['blup'], 'na omit blup')
    _check_list(ranef(res_na), REF['blup_na_omit']['ranef'], 'na omit ranef')


@pytest.mark.parametrize('na_action', ['exclude', 'pass'])
def test_blup_missing_values_exclude(res_na: Any, na_action: Any) -> None:
    _check_list(blup(res_na, na_action=na_action), REF['blup_na_exclude']['blup'], 'na exclude blup')
    _check_list(ranef(res_na, na_action=na_action), REF['blup_na_exclude']['ranef'], 'na exclude ranef')
    assert str(blup(res_na, na_action=na_action)).splitlines()[4].split() == ['3', 'NA', 'NA', 'NA', 'NA']


def test_blup_missing_values_fail(res_na: Any) -> None:
    with pytest.raises(ValueError, match='Missing values'):
        blup(res_na, na_action='fail')


def test_user_weights_not_supported(dat: pd.DataFrame) -> None:
    assert REF['weights_error'] == {'blup': True, 'ranef': True}
    res = rma(data=dat, weights='n')
    with pytest.raises(ValueError, match='non-standard weights'):
        blup(res)
    with pytest.raises(ValueError, match='non-standard weights'):
        ranef(rma(data=dat, weighted=False))


def test_blup_targs_and_frame(dat: pd.DataFrame) -> None:
    res = rma(data=dat)
    shifted = blup(res, transf=lambda x, a: x + a, targs=1.0)
    _close(shifted.pred, np.asarray(REF['blup_REML']['blup']['pred']) + 1.0)
    assert np.isnan(shifted.se).all()
    frame = ranef(res).to_frame()
    assert list(frame.columns) == ['pred', 'se', 'pi.lb', 'pi.ub']
    assert list(frame.index) == REF['blup_REML']['ranef']['slab']


@pytest.mark.parametrize(
    ('key', 'fit_kw', 'call', 'call_kw'),
    [
        ('print_blup_exp', {}, blup, {'transf': np.exp}),
        ('print_ranef', {}, ranef, {}),
        ('print_blup_mods', {'mods': '~ ablat + year'}, blup, {}),
    ],
)
def test_list_print_matches_metafor(dat: pd.DataFrame, key: str, fit_kw: dict[str, Any], call: Any, call_kw: dict[str, Any]) -> None:
    assert str(call(rma(data=dat, **fit_kw), **call_kw)) == _r_print(REF[key])


# -- VIFs ---------------------------------------------------------------------------

BD_MODS = '~ length + wic + feedback + info + pers + imag + meta'
BD_VARS = ('length', 'wic', 'feedback', 'info', 'pers', 'imag', 'meta')


@pytest.fixture(scope='module')
def fits(dat: pd.DataFrame) -> dict[str, Any]:
    bd = datasets.load('bangertdrowns2004')
    # R's scale() standardises with the NAs removed (ddof=1), as pandas does
    bdu = bd.copy()
    for col in BD_VARS:
        bdu[f'u_{col}'] = (bd[col] - bd[col].mean()) / bd[col].std()
    mc = escalc('ZCOR', ri='ri', ni='ni', data=datasets.load('mcdaniel1994'))
    return {
        'bd': rma(data=bd, mods=BD_MODS),
        'bdu': rma(data=bdu, mods='~ ' + ' + '.join(f'u_{col}' for col in BD_VARS)),
        'mc': rma(data=mc, mods='~ C(type) + C(struct)'),
        'bcg': rma(data=dat, mods='~ ablat + alloc + year'),
        'bcg_knha': rma(data=dat, mods='~ ablat + alloc + year', test='knha'),
        'bcg_ee': rma(data=dat, mods='~ ablat + alloc + year', method='EE'),
    }


# key -> (fit, vif kwargs, compare names and print?)
VIF_CASES: dict[str, tuple[str, dict[str, Any], bool]] = {
    'vif_bd': ('bd', {}, True),
    'vif_bd_reest': ('bd', {'reestimate': True}, True),
    'vif_bdu_reest': ('bdu', {'reestimate': True}, False),
    'vif_bdu_table': ('bdu', {'reestimate': True, 'table': True}, False),
    'vif_mc': ('mc', {}, False),
    'vif_bcg_table': ('bcg', {'table': True}, True),
    'vif_bcg_btt': ('bcg', {'btt': [2, 3]}, True),
    'vif_bcg_btt1': ('bcg', {'btt': 1}, True),
    'vif_bcg_str': ('bcg', {'btt': 'alloc'}, True),
    'vif_bcg_list': ('bcg', {'btt': [1, [2, 3], 4]}, True),
    'vif_bcg_list_reest': ('bcg', {'btt': [1, [2, 3], 4], 'reestimate': True}, True),
    'vif_bcg_reest': ('bcg', {'reestimate': True}, True),
    # R wraps this table at 80 columns, so only the values are compared
    'vif_bcg_knha_table': ('bcg_knha', {'table': True}, False),
    'vif_bcg_ee_reest': ('bcg_ee', {'reestimate': True}, True),
}


@pytest.mark.parametrize('key', list(VIF_CASES))
def test_vif_matches_metafor(fits: dict[str, Any], key: str) -> None:
    fit, kwargs, exact = VIF_CASES[key]
    ref = REF[key]
    out = vif(fits[fit], **kwargs)
    assert isinstance(out, VIFResult)
    _close(out.vifs.to_numpy(), ref['vif'], f'{key}: vif')
    _close(out.sifs.to_numpy(), ref['sif'], f'{key}: sif')
    assert [e.m for e in out.vif] == _as_list(ref['m'])
    assert [e.coefs for e in out.vif] == _as_list(ref['coefs'])
    if 'table' in ref:
        assert out.table is not None
        for col, values in ref['table'].items():
            _close(out.table[col].to_numpy(), values, f'{key}: table {col}')
    else:
        assert out.table is None
    if exact:
        assert out.names == _as_list(ref['names'])
        assert str(out) == _r_print(ref['print'])


def test_vif_frame(fits: dict[str, Any]) -> None:
    frame = vif(fits['bcg'], btt=[1, [2, 3], 4]).to_frame()
    assert list(frame.columns) == ['coefs', 'm', 'vif', 'sif']
    assert list(frame.index) == [0, 1, 2]
    named = vif(fits['bcg'], btt=['ablat', 'alloc']).to_frame()
    assert list(named.columns) == ['spec', 'coefs', 'm', 'vif', 'sif']
    assert list(named['coefs']) == ['2', '3:4']
    assert list(vif(fits['bcg']).to_frame().index) == REF['vif_bcg_table']['names']


def test_vif_errors(dat: pd.DataFrame, fits: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match='intercept-only'):
        vif(rma(data=dat))
    with pytest.raises(ValueError, match="'btt'"):
        vif(fits['bcg'], btt=0)
    with pytest.raises(ValueError, match="'att'"):
        vif(fits['bcg'], att=1)
    with pytest.raises(ValueError, match="'sim'"):
        vif(fits['bcg'], sim=1)


def test_format_btt() -> None:
    assert format_btt([2, 3, 4, 6]) == '2:4,6'
    assert format_btt([5]) == '5'
    assert format_btt([2, 4, 5]) == '2,4:5'


# -- simulation (structural only: R's random draws cannot be reproduced) -------------------


def test_vif_sim_structure(fits: dict[str, Any]) -> None:
    out = vif(fits['bcg'], sim=25, seed=42, progbar=False)
    assert out.sim is not None
    assert out.prop is not None
    assert out.sim.shape == (25, 4)
    assert out.sim_type == 'X'
    assert np.all((out.prop >= 0) & (out.prop <= 1))
    # each prop is the share of simulated values the observed (G)VIF reaches
    for j, e in enumerate(out.vif):
        col = out.sim[:, j]
        col = col[~np.isnan(col)]
        assert out.prop[j] == pytest.approx(np.mean(e.vif >= col))
    again = vif(fits['bcg'], sim=25, seed=42, progbar=False)
    assert again.sim is not None
    np.testing.assert_array_equal(out.sim, again.sim)
    text = str(out)
    assert text.splitlines()[1].split() == ['vif', 'prop']


def test_vif_sim_btt_groups_reestimate_and_executor(fits: dict[str, Any]) -> None:
    kwargs: dict[str, Any] = {'btt': [1, [2, 3], 4], 'sim': 6, 'seed': 7, 'progbar': False}
    serial = vif(fits['bcg'], **kwargs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pooled = vif(fits['bcg'], cl=pool, **kwargs)
    assert serial.sim is not None
    assert pooled.sim is not None
    np.testing.assert_array_equal(serial.sim, pooled.sim)
    assert 'prop' in str(serial).splitlines()[1]
    reest = vif(fits['bcg'], reestimate=True, **kwargs)
    assert reest.sim is not None
    assert reest.sim.shape == (6, 3)
    single = vif(fits['bcg'], btt=[2, 3], sim=4, seed=1, progbar=False)
    assert single.prop is not None
    assert str(single).splitlines()[2].endswith(f', prop = {single.prop[0]:.2f}')
    assert not math.isnan(float(single.prop[0]))
