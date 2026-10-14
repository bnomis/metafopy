"""to_long(), to_table(), to_wide() and contrmat() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R, mostly the help page examples)."""

import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

TL = refutil.load_phase7('to_long')
TT = refutil.load_phase7('to_table')
TW = refutil.load_phase7('to_wide')
TABLE: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg'}
# auditme: accept[non-alphabetical-dict-keys] order follows the argument order of the R function
MEANS: dict[str, typing.Any] = {'m1i': 'm1i', 'sd1i': 'sd1i', 'n1i': 'n1i', 'm2i': 'm2i', 'sd2i': 'sd2i', 'n2i': 'n2i'}
RI = [0.5, 0.4, 0.3, 0.2]


def load(name: str, rows: int | None = None) -> pandas.DataFrame:
    dat = metafopy.datasets.load(name)
    return dat if rows is None else dat.iloc[:rows]


def bcg_zero() -> pandas.DataFrame:
    dat = load('bcg', 4).copy()
    dat.loc[1, 'tpos'] = 0
    dat.loc[2, 'cneg'] = numpy.nan
    return dat


LONG_CASES: list[tuple[str, str, dict[str, typing.Any]]] = [
    ('or', 'OR', {**TABLE, 'data': load('bcg')}),
    ('or_vlong', 'OR', {**TABLE, 'data': load('bcg'), 'vlong': True}),
    ('or_append_names', 'OR', {**TABLE, 'data': load('bcg'), 'append': ['author', 'year']}),
    ('or_append_pos', 'OR', {**TABLE, 'data': load('bcg'), 'append': [1, 2]}),
    ('or_noappend_slab', 'OR', {**TABLE, 'data': load('bcg'), 'append': False, 'slab': 'author'}),
    ('or_all', 'RR', {**TABLE, 'data': bcg_zero(), 'to': 'all', 'add': 1, 'append': False}),
    ('or_if0all', 'RD', {**TABLE, 'data': bcg_zero(), 'to': 'if0all', 'append': False}),
    ('mpor', 'MPOR', {**TABLE, 'data': load('bcg', 3), 'append': False}),
    ('mpor_vlong', 'MPOR', {**TABLE, 'data': load('bcg', 3), 'vlong': True}),
    ('mporc', 'MPORC', {**TABLE, 'data': load('bcg', 3), 'append': False}),
    ('mporc_vlong', 'MPORC', {**TABLE, 'data': load('bcg', 3), 'vlong': True, 'append': False}),
    ('irr', 'IRR', {'x1i': 'x1i', 'x2i': 'x2i', 't1i': 't1i', 't2i': 't2i', 'data': load('hart1999'), 'var_names': ['id', 'group', 'events', 'ptime']}),
    ('md', 'MD', {**MEANS, 'data': load('normand1999'), 'var_names': ['id', 'group', 'mean', 'sd', 'n']}),
    ('cor', 'COR', {'ri': 'ri', 'ni': 'ni', 'data': load('mcdaniel1994', 5)}),
    ('pr', 'PR', {'xi': 'xi', 'ni': 'ni', 'data': load('pritz1997', 4), 'append': False}),
    ('pr_vlong', 'PLO', {'xi': 'xi', 'ni': 'ni', 'data': load('pritz1997', 4), 'vlong': True, 'append': False}),
    ('ir', 'IR', {'xi': 'x1i', 'ti': 't1i', 'data': load('hart1999', 4), 'append': False}),
    ('mn', 'MN', {'mi': 'm1i', 'sdi': 'sd1i', 'ni': 'n1i', 'data': load('normand1999', 4), 'append': False}),
    ('mc', 'MC', {'m1i': 'm1i', 'm2i': 'm2i', 'sd1i': 'sd1i', 'sd2i': 'sd2i', 'ni': 'n1i', 'ri': RI, 'data': load('normand1999', 4), 'append': False}),
    ('smcr', 'SMCR', {'m1i': 'm1i', 'm2i': 'm2i', 'sd1i': 'sd1i', 'ni': 'n1i', 'ri': RI, 'data': load('normand1999', 4), 'append': False}),
    ('araw', 'ARAW', {'ai': [0.8, 0.7, 0.9], 'mi': [10, 12, 8], 'ni': [100, 120, 80]}),
    ('dup_slab', 'OR', {**TABLE, 'data': load('bcg', 4), 'slab': ['a', 'b', 'a', 'c'], 'append': False}),
]


# auditme: accept[literal-duplication] each test case spells out its own R arguments
@pytest.mark.parametrize(('key', 'measure', 'kwargs'), LONG_CASES, ids=[c[0] for c in LONG_CASES])
def test_to_long(key: str, measure: str, kwargs: dict[str, typing.Any]) -> None:
    if key in ('or_all', 'or_if0all'):
        # bcg_zero() has a table with a missing cell, which R drops with the same warning
        with pytest.warns(UserWarning, match='NAs omitted'):
            out = metafopy.to_long(measure, **kwargs)
    else:
        out = metafopy.to_long(measure, **kwargs)
    refutil.assert_frame(out, TL[key], key)


def test_to_long_subset_and_missing() -> None:
    bcg = load('bcg')
    out = metafopy.to_long('OR', ai='tpos', n1i=bcg['tpos'] + bcg['tneg'], ci='cpos', n2i=bcg['cpos'] + bcg['cneg'], data=bcg, subset=range(5))
    refutil.assert_frame(out, TL['or_subset_n1i'], 'subset')
    with pytest.warns(UserWarning, match=TL['or_only0_warn']):
        out = metafopy.to_long('OR', **TABLE, data=bcg_zero(), to='only0')
    refutil.assert_frame(out, TL['or_only0'], 'only0')
    with pytest.raises(metafopy.errors.InputError, match='Missing values'):
        metafopy.to_long('OR', **TABLE, data=bcg_zero(), na_action='fail')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert metafopy.to_long('OR', **TABLE, data=bcg_zero(), na_action='pass', append=False).shape[0] == 8


def test_to_long_factors() -> None:
    out = metafopy.to_long('OR', **TABLE, data=load('bcg'), append=False)
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    assert list(out['group'].cat.categories) == ['2', '1']
    # the study labels sort numerically, as R's factor() of the default labels 1, ..., k does
    assert list(out['study'].cat.categories) == [str(i) for i in range(1, 14)]


def test_to_long_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='not available'):
        metafopy.to_long('VR', m1i=[1], m2i=[1], sd1i=[1], sd2i=[1], n1i=[1], n2i=[1])
    with pytest.raises(metafopy.errors.InputError, match="Unknown 'measure'"):
        metafopy.to_long('XYZ')
    with pytest.raises(metafopy.errors.InputError, match='Cannot compute outcomes'):
        metafopy.to_long('OR', ai=[1], bi=[2])
    with pytest.raises(metafopy.errors.InputError, match='not of length 4'):
        # auditme: accept[literal-duplication] each test case spells out its own R arguments
        metafopy.to_long('OR', **TABLE, data=load('bcg'), var_names=['a', 'b'])


TABLE_CASES: list[tuple[str, str, dict[str, typing.Any]]] = [
    ('or_default', 'OR', {**TABLE, 'data': load('bcg'), 'subset': [0, 2, 4]}),
    ('mprd', 'MPRD', {**TABLE, 'data': load('bcg', 3)}),
    ('mppeto', 'MPPETO', {**TABLE, 'data': load('bcg', 3)}),
    ('md', 'MD', {**MEANS, 'data': load('normand1999'), 'slab': 'source', 'rows': ['Specialized Care', 'Routine Care']}),
    ('cor', 'ZCOR', {'ri': 'ri', 'ni': 'ni', 'data': load('mcdaniel1994', 5)}),
    ('pr', 'PR', {'xi': 'xi', 'ni': 'ni', 'data': load('pritz1997', 4), 'to': 'all'}),
    ('ir', 'IR', {'xi': 'x1i', 'ti': 't1i', 'data': load('hart1999', 4)}),
    ('mn', 'MN', {'mi': 'm1i', 'sdi': 'sd1i', 'ni': 'n1i', 'data': load('normand1999', 4)}),
    ('mc', 'MC', {'m1i': 'm1i', 'm2i': 'm2i', 'sd1i': 'sd1i', 'sd2i': 'sd2i', 'ni': 'n1i', 'ri': RI, 'data': load('normand1999', 4)}),
    ('smcr', 'SMCR', {'m1i': 'm1i', 'm2i': 'm2i', 'sd1i': 'sd1i', 'ni': 'n1i', 'ri': RI, 'data': load('normand1999', 4)}),
    ('araw', 'ARAW', {'ai': [0.8, 0.7, 0.9], 'mi': [10, 12, 8], 'ni': [100, 120, 80]}),
]


def check_table(out: metafopy.TableArray, key: str) -> None:
    ref = TT[key]
    dims = refutil.as_list(ref['dim'])
    values = numpy.array([numpy.nan if v is None else v for v in refutil.as_list(ref['values'])]).reshape(dims, order='F')
    assert out.values.shape == tuple(dims)
    numpy.testing.assert_allclose(out.values, values, rtol=refutil.RTOL, equal_nan=True, err_msg=key)
    assert list(out.rows) == refutil.as_list(ref['rows'])
    assert list(out.cols) == refutil.as_list(ref['cols'])
    assert list(out.slab) == [str(s) for s in refutil.as_list(ref['slab'])]


# auditme: accept[literal-duplication] each test case spells out its own R arguments
@pytest.mark.parametrize(('key', 'measure', 'kwargs'), TABLE_CASES, ids=[c[0] for c in TABLE_CASES])
def test_to_table(key: str, measure: str, kwargs: dict[str, typing.Any]) -> None:
    check_table(metafopy.to_table(measure, **kwargs), key)


def test_to_table_examples() -> None:
    bcg = load('bcg')
    out = metafopy.to_table(
        'OR', **TABLE, data=bcg, slab=(bcg['author'] + ', ' + bcg['year'].astype(str)).tolist(), rows=['Vaccinated', 'Not Vaccinated'], cols=['TB+', 'TB-']
    )
    check_table(out, 'or')
    first = out.table(0)
    assert first.loc['Vaccinated', 'TB+'] == bcg['tpos'].iloc[0]
    assert out.table(out.slab[1]).equals(out.table(1))
    hart = load('hart1999')
    out = metafopy.to_table(
        'IRR', x1i='x1i', x2i='x2i', t1i='t1i', t2i='t2i', data=hart, slab=(hart['study'] + ', ' + hart['year'].astype(str)).tolist(),
        rows=['Warfarin Group', 'Placebo/Control Group'],
    )  # fmt: skip
    check_table(out, 'irr')
    with pytest.warns(UserWarning, match='omitted'):
        check_table(metafopy.to_table('OR', **TABLE, data=bcg_zero(), to='only0'), 'or_only0')
    assert len(out) == hart.shape[0]
    assert ', , AFASAK, 1989' in str(out)


def test_to_wide_and_contrmat() -> None:
    senn = load('senn2013').iloc[:, [0, 3, 2, 1, 4, 5]]
    wide = metafopy.to_wide(senn, study='study', grp='treatment', ref='placebo', grpvars=[3, 4, 5])
    refutil.assert_frame(wide, TW['senn'], 'senn')
    refutil.assert_frame(metafopy.contrmat(wide, grp1='treatment.1', grp2='treatment.2'), TW['senn_contr'], 'senn contrmat')
    out = metafopy.to_wide(senn, study='study', grp='treatment', grpvars=['ni', 'mi', 'sdi'], postfix=['_a', '_b'], addid=False, minlen=4)
    refutil.assert_frame(out, TW['senn_noref'], 'default ref')
    wide = metafopy.to_wide(senn, study=0, grp=2, ref='placebo', grpvars=[3, 4, 5], adddesign=False)
    out = metafopy.contrmat(wide, grp1='treatment.1', grp2='treatment.2', append=False, last=numpy.nan)
    refutil.assert_frame(out, TW['senn_contr_noappend'], 'last=NA')


def test_to_wide_hasselblad() -> None:
    dat = load('hasselblad1998')
    wide = metafopy.to_wide(dat, study='study', grp='trt', ref='no_contact', grpvars=[5, 6])
    refutil.assert_frame(wide, TW['hass'], 'hasselblad')
    refutil.assert_frame(metafopy.contrmat(wide, grp1='trt.1', grp2='trt.2', shorten=True, minlen=3), TW['hass_contr'], 'shorten')
    out = metafopy.contrmat(wide, grp1='trt.1', grp2='trt.2', last='self_help', check=False, append=False)
    refutil.assert_frame(out, TW['hass_contr_last'], 'last')


def test_to_wide_one_arm() -> None:
    out = metafopy.to_wide(refutil.r_frame(TW['onearm_input']), study='study', grp='grp', ref='A', grpvars='y')
    refutil.assert_frame(out, TW['onearm'], 'one-arm study')


def test_to_wide_errors() -> None:
    dat = load('hasselblad1998')
    with pytest.raises(metafopy.errors.InputError, match="'study'"):
        metafopy.to_wide(dat, study='nope', grp='trt', grpvars=[5])
    with pytest.raises(metafopy.errors.InputError, match="'ref'"):
        metafopy.to_wide(dat, study='study', grp='trt', ref='nothing', grpvars=[5])
    with pytest.raises(metafopy.errors.InputError, match="'grpvars'"):
        metafopy.to_wide(dat, study='study', grp='trt', grpvars=[50])
    with pytest.raises(metafopy.errors.InputError, match="'last'"):
        metafopy.contrmat(dat, grp1='trt', grp2='trt', last='nothing')
