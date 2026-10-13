"""blup(), ranef() and vif() for rma_uni fits, checked against R metafor (src/tests/r/generate_blup_vif.R)."""

import concurrent.futures
import json
import math
import pathlib
import typing

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.effects
import metafopy.methods.blup
import metafopy.methods.ranef
import metafopy.methods.vif

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'extras' / 'blup_vif.json').read_text())
RTOL = 1e-8


def close(actual: typing.Any, expected: typing.Any, what: str = '') -> None:
    numpy.testing.assert_allclose(
        numpy.atleast_1d(numpy.asarray(actual, dtype=float)),
        numpy.atleast_1d(numpy.asarray(expected, dtype=float)),
        rtol=RTOL,
        atol=1e-12,
        equal_nan=True,
        err_msg=what,
    )


def as_list(value: typing.Any) -> list[typing.Any]:
    # jsonlite's auto_unbox turns length-one vectors into scalars
    return value if isinstance(value, list) else [value]


def r_print(text: str) -> str:
    # metafor's printed tables end every line with a space
    return '\n'.join(line.rstrip() for line in text.split('\n'))


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    a, b, c, e = (d[col].to_numpy(float) for col in ('tpos', 'tneg', 'cpos', 'cneg'))
    d['yi'] = numpy.log(a / (a + b)) - numpy.log(c / (c + e))
    d['vi'] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d['n'] = a + b + c + e
    d['label'] = d['author'] + ' ' + d['year'].astype(str)
    return d


# -- BLUPs ------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
BLUP_CASES: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'blup_REML': ({}, {}),
    'blup_DL': ({'method': 'DL'}, {}),
    'blup_ML': ({'method': 'ML'}, {}),
    'blup_mods': ({'mods': '~ ablat + year'}, {}),
    'blup_EE': ({'method': 'EE'}, {}),
    'blup_knha': ({'mods': '~ ablat', 'test': 'knha'}, {}),
    'blup_level90': ({}, {'level': 90}),
    'blup_level90_fit': ({'level': 90}, {}),
    'blup_exp': ({}, {'transf': numpy.exp}),
    'blup_slab': ({'slab': 'label'}, {}),
}


def check_list(out: metafopy.methods.blup.BlupResult, ref: dict[str, typing.Any], what: str) -> None:
    for attr in ('pred', 'se', 'pi_lb', 'pi_ub'):
        close(getattr(out, attr), ref[attr], f'{what}: {attr}')
    assert list(out.slab) == ref['slab']


@pytest.mark.parametrize('key', list(BLUP_CASES))
def test_blup_ranef_match_metafor(dat: pandas.DataFrame, key: str) -> None:
    fit_kw, call_kw = BLUP_CASES[key]
    res = metafopy.rma(data=dat, **fit_kw)
    b = metafopy.methods.blup.blup(res, **call_kw)
    r = metafopy.methods.ranef.ranef(res, **call_kw)
    assert isinstance(b, metafopy.methods.blup.BlupResult)
    assert isinstance(r, metafopy.methods.ranef.RanefResult)
    check_list(b, REF[key]['blup'], f'{key} blup')
    check_list(r, REF[key]['ranef'], f'{key} ranef')


@pytest.fixture(scope='module')
def res_na(dat: pandas.DataFrame) -> typing.Any:
    dna = dat.copy()
    dna.loc[2, 'yi'] = numpy.nan
    dna.loc[6, 'ablat'] = numpy.nan
    return metafopy.rma(data=dna, mods='~ ablat')


def test_blup_missing_values_omit(res_na: typing.Any) -> None:
    check_list(metafopy.methods.blup.blup(res_na), REF['blup_na_omit']['blup'], 'na omit blup')
    check_list(metafopy.methods.ranef.ranef(res_na), REF['blup_na_omit']['ranef'], 'na omit ranef')


@pytest.mark.parametrize('na_action', ['exclude', 'pass'])
def test_blup_missing_values_exclude(res_na: typing.Any, na_action: typing.Any) -> None:
    check_list(metafopy.methods.blup.blup(res_na, na_action=na_action), REF['blup_na_exclude']['blup'], 'na exclude blup')
    check_list(metafopy.methods.ranef.ranef(res_na, na_action=na_action), REF['blup_na_exclude']['ranef'], 'na exclude ranef')
    assert str(metafopy.methods.blup.blup(res_na, na_action=na_action)).splitlines()[4].split() == ['3', 'NA', 'NA', 'NA', 'NA']


def test_blup_missing_values_fail(res_na: typing.Any) -> None:
    with pytest.raises(ValueError, match='Missing values'):
        metafopy.methods.blup.blup(res_na, na_action='fail')


def test_user_weights_not_supported(dat: pandas.DataFrame) -> None:
    assert REF['weights_error'] == {'blup': True, 'ranef': True}
    res = metafopy.rma(data=dat, weights='n')
    with pytest.raises(ValueError, match='non-standard weights'):
        metafopy.methods.blup.blup(res)
    with pytest.raises(ValueError, match='non-standard weights'):
        metafopy.methods.ranef.ranef(metafopy.rma(data=dat, weighted=False))


def test_blup_targs_and_frame(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat)
    shifted = metafopy.methods.blup.blup(res, transf=lambda x, a: x + a, targs=1.0)
    close(shifted.pred, numpy.asarray(REF['blup_REML']['blup']['pred']) + 1.0)
    assert numpy.isnan(shifted.se).all()
    frame = metafopy.methods.ranef.ranef(res).to_frame()
    assert list(frame.columns) == ['pred', 'se', 'pi.lb', 'pi.ub']
    assert list(frame.index) == REF['blup_REML']['ranef']['slab']


@pytest.mark.parametrize(
    ('key', 'fit_kw', 'call', 'call_kw'),
    [
        ('print_blup_exp', {}, metafopy.methods.blup.blup, {'transf': numpy.exp}),
        ('print_ranef', {}, metafopy.methods.ranef.ranef, {}),
        ('print_blup_mods', {'mods': '~ ablat + year'}, metafopy.methods.blup.blup, {}),
    ],
)
def test_list_print_matches_metafor(dat: pandas.DataFrame, key: str, fit_kw: dict[str, typing.Any], call: typing.Any, call_kw: dict[str, typing.Any]) -> None:
    assert str(call(metafopy.rma(data=dat, **fit_kw), **call_kw)) == r_print(REF[key])


# -- VIFs ---------------------------------------------------------------------------

BD_MODS = '~ length + wic + feedback + info + pers + imag + meta'
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
BD_VARS = ('length', 'wic', 'feedback', 'info', 'pers', 'imag', 'meta')


@pytest.fixture(scope='module')
def fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    bd = metafopy.datasets.load('bangertdrowns2004')
    # R's scale() standardises with the NAs removed (ddof=1), as pandas does
    bdu = bd.copy()
    for col in BD_VARS:
        bdu[f'u_{col}'] = (bd[col] - bd[col].mean()) / bd[col].std()
    mc = metafopy.effects.escalc('ZCOR', ri='ri', ni='ni', data=metafopy.datasets.load('mcdaniel1994'))
    return {
        'bd': metafopy.rma(data=bd, mods=BD_MODS),
        'bdu': metafopy.rma(data=bdu, mods='~ ' + ' + '.join(f'u_{col}' for col in BD_VARS)),
        'mc': metafopy.rma(data=mc, mods='~ C(type) + C(struct)'),
        'bcg': metafopy.rma(data=dat, mods='~ ablat + alloc + year'),
        'bcg_knha': metafopy.rma(data=dat, mods='~ ablat + alloc + year', test='knha'),
        'bcg_ee': metafopy.rma(data=dat, mods='~ ablat + alloc + year', method='EE'),
    }


# key -> (fit, vif kwargs, compare names and print?)
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
VIF_CASES: dict[str, tuple[str, dict[str, typing.Any], bool]] = {
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
def test_vif_matches_metafor(fits: dict[str, typing.Any], key: str) -> None:
    fit, kwargs, exact = VIF_CASES[key]
    ref = REF[key]
    out = metafopy.methods.vif.vif(fits[fit], **kwargs)
    assert isinstance(out, metafopy.methods.vif.VIFResult)
    close(out.vifs.to_numpy(), ref['vif'], f'{key}: vif')
    close(out.sifs.to_numpy(), ref['sif'], f'{key}: sif')
    assert [e.m for e in out.vif] == as_list(ref['m'])
    assert [e.coefs for e in out.vif] == as_list(ref['coefs'])
    if 'table' in ref:
        assert out.table is not None
        for col, values in ref['table'].items():
            close(out.table[col].to_numpy(), values, f'{key}: table {col}')
    else:
        assert out.table is None
    if exact:
        assert out.names == as_list(ref['names'])
        assert str(out) == r_print(ref['print'])


def test_vif_frame(fits: dict[str, typing.Any]) -> None:
    frame = metafopy.methods.vif.vif(fits['bcg'], btt=[1, [2, 3], 4]).to_frame()
    assert list(frame.columns) == ['coefs', 'm', 'vif', 'sif']
    assert list(frame.index) == [0, 1, 2]
    named = metafopy.methods.vif.vif(fits['bcg'], btt=['ablat', 'alloc']).to_frame()
    assert list(named.columns) == ['spec', 'coefs', 'm', 'vif', 'sif']
    assert list(named['coefs']) == ['2', '3:4']
    assert list(metafopy.methods.vif.vif(fits['bcg']).to_frame().index) == REF['vif_bcg_table']['names']


def test_vif_errors(dat: pandas.DataFrame, fits: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError, match='intercept-only'):
        metafopy.methods.vif.vif(metafopy.rma(data=dat))
    with pytest.raises(ValueError, match="'btt'"):
        metafopy.methods.vif.vif(fits['bcg'], btt=0)
    with pytest.raises(ValueError, match="'att'"):
        metafopy.methods.vif.vif(fits['bcg'], att=1)
    with pytest.raises(ValueError, match="'sim'"):
        metafopy.methods.vif.vif(fits['bcg'], sim=1)


def test_format_btt() -> None:
    assert metafopy.methods.vif.format_btt([2, 3, 4, 6]) == '2:4,6'
    assert metafopy.methods.vif.format_btt([5]) == '5'
    assert metafopy.methods.vif.format_btt([2, 4, 5]) == '2,4:5'


# -- simulation (structural only: R's random draws cannot be reproduced) -------------------


def test_vif_sim_structure(fits: dict[str, typing.Any]) -> None:
    out = metafopy.methods.vif.vif(fits['bcg'], sim=25, seed=42, progbar=False)
    assert out.sim is not None
    assert out.prop is not None
    assert out.sim.shape == (25, 4)
    assert out.sim_type == 'X'
    assert numpy.all((out.prop >= 0) & (out.prop <= 1))
    # each prop is the share of simulated values the observed (G)VIF reaches
    for j, e in enumerate(out.vif):
        col = out.sim[:, j]
        col = col[~numpy.isnan(col)]
        assert out.prop[j] == pytest.approx(numpy.mean(e.vif >= col))
    again = metafopy.methods.vif.vif(fits['bcg'], sim=25, seed=42, progbar=False)
    assert again.sim is not None
    numpy.testing.assert_array_equal(out.sim, again.sim)
    text = str(out)
    assert text.splitlines()[1].split() == ['vif', 'prop']


def test_vif_sim_btt_groups_reestimate_and_executor(fits: dict[str, typing.Any]) -> None:
    kwargs: dict[str, typing.Any] = {'btt': [1, [2, 3], 4], 'sim': 6, 'seed': 7, 'progbar': False}
    serial = metafopy.methods.vif.vif(fits['bcg'], **kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pooled = metafopy.methods.vif.vif(fits['bcg'], cl=pool, **kwargs)
    assert serial.sim is not None
    assert pooled.sim is not None
    numpy.testing.assert_array_equal(serial.sim, pooled.sim)
    assert 'prop' in str(serial).splitlines()[1]
    reest = metafopy.methods.vif.vif(fits['bcg'], reestimate=True, **kwargs)
    assert reest.sim is not None
    assert reest.sim.shape == (6, 3)
    single = metafopy.methods.vif.vif(fits['bcg'], btt=[2, 3], sim=4, seed=1, progbar=False)
    assert single.prop is not None
    assert str(single).splitlines()[2].endswith(f', prop = {single.prop[0]:.2f}')
    assert not math.isnan(float(single.prop[0]))
