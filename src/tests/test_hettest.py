"""hettest() checked against R metafor 5.2-1 (tests/r/generate_hettest.R), including the bootstrap p-values drawn from R's generator."""

import json
import pathlib
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.hettest

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase11' / 'hettest.json').read_text(encoding='utf-8'))
ALL_TESTS = ('score', 'lrt', 'wald', 'ksn', 'ksx2', 'adn', 'adx2')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def hack() -> pandas.DataFrame:
    return metafopy.datasets.load('hackshaw1998')


def same_text(actual: str, expected: str) -> None:
    assert refutil.r_print(actual).strip() == refutil.r_print(expected).strip()


def check(res: metafopy.methods.hettest.HettestResult, ref: dict[str, typing.Any]) -> None:
    refutil.close(res.statistic, ref['statistic'], 'statistic', rtol=1e-8)
    refutil.close(res.pval, ref['pval'], 'pval', rtol=1e-8)
    assert res.method == ref['method']
    assert list(res.test) == refutil.as_list(ref['test'])
    assert list(res.boot) == refutil.as_list(ref['boot'])
    assert res.iter == ref['iter']
    refutil.close(res.tau2i, ref['tau2i'], 'tau2i', rtol=1e-8)
    if isinstance(ref.get('se.tau2i'), list | float):
        assert res.se_tau2i is not None
        refutil.close(res.se_tau2i, ref['se.tau2i'], 'se.tau2i', rtol=1e-8)
    if isinstance(ref['nboot'], list | int):
        assert res.statistic_boot is not None
        assert list(numpy.sum(~numpy.isnan(res.statistic_boot), axis=0)) == refutil.as_list(ref['nboot'])
        for j, col in enumerate(refutil.as_list(ref['statistic.boot'])):
            refutil.close(res.statistic_boot[:, j], col, f'statistic_boot[{j}]', rtol=1e-7)
    same_text(str(res), ref['print'])


def vec_cases(dat: pandas.DataFrame, hack: pandas.DataFrame) -> dict[str, typing.Callable[[], metafopy.methods.hettest.HettestResult]]:
    return {
        'vec_reml': lambda: metafopy.hettest(dat.yi, dat.vi, test=ALL_TESTS, boot=200, seed=1234),
        'vec_ml': lambda: metafopy.hettest(dat.yi, dat.vi, method='ML', test=ALL_TESTS, boot=200, seed=99),
        'vec_noboot': lambda: metafopy.hettest(dat.yi, dat.vi, test=('score', 'lrt', 'wald'), boot=False, seed=5),
        'vec_noboot_ml': lambda: metafopy.hettest(dat.yi, dat.vi, method='ML', test=('score', 'lrt', 'wald'), boot=False, seed=5),
        'vec_sei': lambda: metafopy.hettest(dat.yi, sei=numpy.sqrt(dat.vi), boot=100, seed=7),
        'vec_mom': lambda: metafopy.hettest(dat.yi, dat.vi, test=('lrt', 'wald'), mom=True, boot=100, seed=11),
        'vec_hack': lambda: metafopy.hettest(hack.yi, hack.vi, test=ALL_TESTS, boot=150, seed=3),
        'vec_hack_ml': lambda: metafopy.hettest(hack.yi, hack.vi, method='ML', test=('score', 'lrt'), boot=100, seed=4),
        'vec_boot_mixed': lambda: metafopy.hettest(dat.yi, dat.vi, test=('score', 'lrt', 'ksn'), boot=(True, False, False), seed=8),
        'vec_control': lambda: metafopy.hettest(dat.yi, dat.vi, test=('lrt', 'wald'), boot=False, control={'tau2i_init': 0.5}),
        'vec_default_seeded': lambda: metafopy.hettest(dat.yi, dat.vi, boot=300, seed=2024),
    }


@pytest.mark.parametrize(
    'name',
    [
        'vec_reml',
        'vec_ml',
        'vec_noboot',
        'vec_noboot_ml',
        'vec_sei',
        'vec_mom',
        'vec_hack',
        'vec_hack_ml',
        'vec_boot_mixed',
        'vec_control',
        'vec_default_seeded',
    ],
)
def test_vectors(name: str, dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    check(vec_cases(dat, hack)[name](), REF[name])


def test_models(dat: pandas.DataFrame, hack: pandas.DataFrame) -> None:
    check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi), test=ALL_TESTS, boot=200, seed=21), REF['mod_reml'])
    check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi, method='ML'), test=ALL_TESTS, boot=200, seed=22), REF['mod_ml'])
    check(metafopy.hettest(metafopy.rma(hack.yi, hack.vi), test=('score', 'lrt', 'wald'), boot=100, seed=23), REF['mod_hack'])
    check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi), test=('score', 'lrt', 'wald'), boot=False), REF['mod_noboot'])
    check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi, test='knha'), test='score', boot=100, seed=26), REF['mod_knha'])


def test_model_not_fitted_with_ml(dat: pandas.DataFrame) -> None:
    with pytest.warns(UserWarning, match='not fitted with ML or REML'):
        check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi, method='DL'), test=('score', 'lrt'), boot=100, seed=24), REF['mod_dl'])
    with pytest.warns(UserWarning, match='based on ML estimation'), pytest.warns(UserWarning, match="Argument 'method' ignored"):
        check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi, method='DL'), method='ML', test=('score', 'lrt'), boot=100, seed=25), REF['mod_dl_ml'])
    with pytest.warns(UserWarning, match='not fitted with ML or REML'):
        check(metafopy.hettest(metafopy.rma(dat.yi, dat.vi, method='FE'), test=('score', 'lrt'), boot=100, seed=27), REF['mod_fe'])


def test_mantel_haenszel_fit() -> None:
    bcg = metafopy.datasets.load('bcg')
    fit = metafopy.rma_mh(ai=bcg.tpos, bi=bcg.tneg, ci=bcg.cpos, di=bcg.cneg, measure='RR')
    with pytest.warns(UserWarning, match='not fitted with ML or REML'):
        check(metafopy.hettest(fit, test='score', boot=60, seed=28), REF['mod_mh'])


def test_missing_values_are_omitted(dat: pandas.DataFrame) -> None:
    yna = dat.yi.copy()
    yna[3] = numpy.nan
    with pytest.warns(UserWarning, match='1 study with NAs omitted from the test'):
        check(metafopy.hettest(yna, dat.vi, boot=100, seed=31), REF['vec_na'])


def test_global_stream(dat: pandas.DataFrame) -> None:
    metafopy.set_seed(42)
    a = metafopy.hettest(dat.yi, dat.vi, boot=80)
    b = metafopy.hettest(dat.yi, dat.vi, boot=80, test='lrt')
    refutil.close(a.pval, REF['stream']['a'])
    refutil.close(b.pval, REF['stream']['b'])
    refutil.close(a.statistic, REF['stream']['a_stat'])
    refutil.close(b.statistic, REF['stream']['b_stat'])


def test_summary_digits(dat: pandas.DataFrame) -> None:
    res = metafopy.hettest(dat.yi, dat.vi, boot=False)
    assert 'X^2(df = 12) = 2.88842' in res.summary(digits=5)


def test_data_argument(dat: pandas.DataFrame) -> None:
    named = metafopy.hettest('yi', 'vi', data=dat, boot=20, seed=1)
    plain = metafopy.hettest(dat.yi, dat.vi, boot=20, seed=1)
    numpy.testing.assert_array_equal(named.pval, plain.pval)


@pytest.mark.parametrize(
    ('name', 'call', 'exc'),
    [
        ('mods', lambda d: metafopy.hettest(metafopy.rma(d.yi, d.vi, mods=d.ablat)), metafopy.errors.InputError),
        ('test', lambda d: metafopy.hettest(d.yi, d.vi, test='foo'), metafopy.errors.InputError),
        ('method', lambda d: metafopy.hettest(d.yi, d.vi, method='DL'), metafopy.errors.InputError),
        ('novi', lambda d: metafopy.hettest(d.yi), metafopy.errors.InputError),
        ('k1', lambda d: metafopy.hettest([1.0], [0.1]), metafopy.errors.InputError),
        ('mv', lambda d: metafopy.hettest(metafopy.rma_mv(d.yi, d.vi, random='~ 1 | trial', data=d)), metafopy.errors.NotSupportedError),
        ('ls', lambda d: metafopy.hettest(metafopy.rma(d.yi, d.vi, scale=numpy.ones((13, 1)))), metafopy.errors.NotSupportedError),
        ('tau2i_init', lambda d: metafopy.hettest(d.yi, d.vi, control={'tau2i_init': [1, 2]}), metafopy.errors.InputError),
    ],
)
@pytest.mark.filterwarnings('ignore')
def test_errors(name: str, call: typing.Callable[[pandas.DataFrame], typing.Any], exc: type[Exception], dat: pandas.DataFrame) -> None:
    with pytest.raises(exc) as info:
        call(dat)
    assert str(info.value).replace('_', '.') == REF['errors'][name]
