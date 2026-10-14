"""conv_fivenum() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R, based on the help page examples)."""

import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF = refutil.load_phase7('conv_fivenum')
# auditme: accept[non-alphabetical-dict-keys] order follows the argument order of the R function
COLS: dict[str, typing.Any] = {'min': 'min', 'q1': 'q1', 'median': 'median', 'q3': 'q3', 'max': 'max', 'n': 'n'}


def check(key: str, source: str, attrs: bool = True, **kwargs: typing.Any) -> pandas.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF[source]), **kwargs)
    ref = REF[key]
    refutil.assert_frame(out, ref['data'], key)
    if attrs:
        m = ref['mean']
        refutil.close(out.attrs['tval'], m['tval'], f'{key}: tval', rtol=1e-12)
        refutil.close(out.attrs['crit'], m['crit'], f'{key}: crit', rtol=1e-12)
        assert [None if pandas.isna(v) else bool(v) for v in out.attrs['sig']] == refutil.as_list(m['sig'])
        assert out.attrs['dist'] == refutil.as_list(m['dist'])
        if m['est'] not in (None, []):
            assert out.attrs['est']['mean'].tolist() == refutil.as_list(m['est'])
            assert out.attrs['est']['sd'].tolist() == refutil.as_list(ref['sd']['est'])
    return out


@pytest.mark.parametrize('method', ['hozo2005', 'wan2014', 'bland2015', 'walter2007'])
def test_methods(method: str) -> None:
    check(method, 'input1', method=method)


def test_default_and_options() -> None:
    check('default', 'input1')
    check('shi2020', 'input1', method=['luo2016', 'shi2020'])
    check('noappend', 'input1', append=False)
    check('include', 'input1', include=[True, False, True, True])
    out = check('names', 'input1', attrs=False, var_names=['m', 's'])
    assert 'est' not in out.attrs


def test_skewness() -> None:
    check('skewed', 'input3')
    check('skewed_notest', 'input3', test=False)
    check('by_n', 'input_n')
    check('by_n_hozo', 'input_n', method='hozo2005', test=False)
    check('by_n_walter', 'input_n', method='walter2007', test=False)


def test_lognormal() -> None:
    check('lnorm', 'input3', dist=['norm', 'norm', 'lnorm', 'norm'], replace='all')
    check('lnorm_all', 'input3', dist='lnorm')
    check('lnorm_notransf', 'input3', dist='lnorm', transf=False)
    with pytest.warns(UserWarning, match='Switching'):
        metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input3']), dist='lnorm', method='wan2014')
    check('lnorm_switch', 'input3', dist='lnorm', method='wan2014')


def test_errors() -> None:
    dat = refutil.r_frame(REF['input1'])
    with pytest.raises(metafopy.errors.NotSupportedError, match='estmeansd'):
        metafopy.conv_fivenum(**COLS, data=dat, method='qe')
    with pytest.raises(metafopy.errors.NotSupportedError, match='metaBLUE'):
        metafopy.conv_fivenum(**COLS, data=dat, method='blue')
    # as in R, a method that exists only for the means (or SDs) must be paired with one for the other
    with pytest.raises(metafopy.errors.InputError, match="Unknown 'method'"):
        metafopy.conv_fivenum(**COLS, data=dat, method='luo2016')
    with pytest.raises(metafopy.errors.InputError, match='not true in row 0'):
        metafopy.conv_fivenum(min=[5.0], median=[4.0], max=[10.0], n=[20])
    with pytest.raises(metafopy.errors.InputError, match="'dist'"):
        metafopy.conv_fivenum(**COLS, data=dat, dist='gamma')
    assert numpy.isnan(metafopy.conv_fivenum(min=[1.0], median=[4.0], max=[10.0], n=[4])['mean'].iloc[0])
