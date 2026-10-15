"""fmtx, fmtp, fmtp2, fmtt and setmfopt/getmfopt against R (src/tests/r/generate_formatters.R)."""

import collections.abc
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.mfopt

REF = refutil.load_phase7('formatters')
PV = [0.0002, 0.00008, None, 0.5, 1.2, -0.1, 0.04999, 0.00010]
PV2 = [0.0005, 0.001, 0.002, 0.0423, 0.0543, 0.0578, 0.0623, 0.5329, None, 1]
XS = [1.0002, 2.00008, 3.00004, -1, 1, -0.00001, None, numpy.inf, -numpy.inf, 0.5, 12345.678, 2.5]


def unbox(args: dict[str, list[typing.Any]] | list[typing.Any]) -> dict[str, typing.Any]:
    if isinstance(args, list):
        return {}
    return {k: (v[0] if len(v) == 1 else v) for k, v in args.items()}


@pytest.fixture
def options() -> collections.abc.Iterator[None]:
    saved = dict(metafopy.mfopt.OPTIONS)
    yield
    metafopy.mfopt.OPTIONS.clear()
    metafopy.mfopt.OPTIONS.update(saved)


def test_fmtp() -> None:
    for case in REF['fmtp']:
        assert metafopy.fmtp(PV, **unbox(case['args'])) == case['out'], case['args']
    assert metafopy.fmtp(0.00002) == '<.0001'


def test_fmtp2() -> None:
    for case in REF['fmtp2']:
        assert metafopy.fmtp2(PV2, **unbox(case['args'])) == case['out'], case['args']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.fmtp2(0.5, cutoff=[0.05])


def test_fmtx() -> None:
    for case in REF['fmtx']:
        assert metafopy.fmtx(XS, **unbox(case['args'])) == case['out'], case['args']
    assert metafopy.fmtx([1.5, 2, 2.10, 0.25], digits=2, drop0ifint=True) == REF['fmtx_drop0']
    assert [metafopy.fmtx(1e-9, digits=4, thresh=1e-8), metafopy.fmtx(0.5, digits=4, thresh=1e-8)] == REF['fmtx_thresh']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.fmtx([1.0, 2.0], thresh=1.0)


def test_fmtx_two_dimensional() -> None:
    frame = pandas.DataFrame({'a': [1.0, 2.5], 'b': [0.123456, -3.0]}, index=['r1', 'r2'])
    out = metafopy.fmtx(frame, digits=[1, 3])
    assert out.loc['r2', 'a'] == '2.5'
    assert out.loc['r1', 'b'] == '0.123'
    assert metafopy.fmtx(frame.to_numpy(), digits=2).tolist() == [['1.00', '0.12'], ['2.50', '-3.00']]


def test_fmtt() -> None:
    got = [
        metafopy.fmtt(2.45, 'z', pval=0.01429, digits=2),
        metafopy.fmtt(3.45, 'z', pval=0.00056, digits=2),
        metafopy.fmtt(2.45, 't', df=23, pval=0.02232, digits=2),
        metafopy.fmtt(3.45, 't', df=23, pval=0.00218, digits=2),
        metafopy.fmtt(3.45, 't', df=23, pval=0.00218, digits=2, format=2),
        metafopy.fmtt(46.23, 'Q', df=29, pval=0.0226, digits=2),
        metafopy.fmtt(46.23, 'Q', df=29, pval=0.0226, digits=2, format=2),
        metafopy.fmtt(8.75, 'F', df1=2, df2=35, pval=0.00083, digits=(2, 3)),
        metafopy.fmtt(8.75, 'F', df1=2, df2=35.4567, pval=0.00083, digits=(2, 3), format=2, pname='p'),
        metafopy.fmtt(8.75, 'F', df1=2, df2=35, pval=0.00083, digits=(2, 3), format=2, pname='p', sep=False),
        metafopy.fmtt(-1.5, 'z', pval=0.13, digits=3, flag=' ', sep=False),
    ]
    assert got == REF['fmtt']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.fmtt(1.0, 'z')


def test_r_as_character() -> None:
    assert [metafopy.formatters.r_as_character(v) for v in (23.0, 35.123456789012345, 1e-20, 1e5, 1e15, 123456.7)] == [
        '23',
        '35.1234567890123',
        '1e-20',
        '1e+05',
        '1e+15',
        '123456.7',
    ]


def test_mfopt_store(options: None) -> None:
    assert metafopy.getmfopt('space') is True
    assert metafopy.getmfopt('nosuch', default=3) == 3
    metafopy.setmfopt(digits=2, theme='dark')
    assert metafopy.getmfopt()['digits'] == 2
    metafopy.setmfopt(digits=None)
    assert metafopy.getmfopt('digits') is None
    with pytest.raises(metafopy.errors.InputError):
        metafopy.setmfopt(space='no')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.setmfopt(theme='neon')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.setmfopt(digits='4')


def test_mfopt_printing(options: None) -> None:
    d = refutil.bcg()
    res = metafopy.rma_uni(d['yi'], d['vi'])
    assert refutil.r_print(str(res)) == refutil.r_print('\n'.join(REF['print_default']))
    metafopy.setmfopt(digits=2)
    assert refutil.r_print(str(res)) == refutil.r_print('\n'.join(REF['print_digits2']))
    assert refutil.r_print(res.summary(digits=4, showfit=False)) == refutil.r_print('\n'.join(REF['print_default']))
    metafopy.setmfopt(digits=None, space=False)
    assert refutil.r_print(str(res)) == refutil.r_print('\n'.join(REF['print_nospace']))
