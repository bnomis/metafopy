"""escalc() data handling, attributes and argument checks (values from R metafor 5.2-1)."""

import math
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets


def test_appends_to_data_and_keeps_existing_yi_when_not_replacing() -> None:
    dat = metafopy.datasets.load('bcg')
    dat['yi'] = [numpy.nan, 1.0, *[numpy.nan] * 11]
    res = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, replace=False, add_measure=True, slab='author')
    # R: escalc("RR", ..., data=dat, replace=FALSE, add.measure=TRUE, slab=author)
    numpy.testing.assert_allclose(res['yi'][:3], [-0.889311333920205, 1.0, -1.34807314829969], rtol=1e-12)
    numpy.testing.assert_allclose(res['vi'][:3], [0.325584765003961, 0.194581121398144, 0.415367965367965], rtol=1e-12)
    assert list(res['measure'][:3]) == ['RR', 'RR', 'RR']
    assert list(res.columns[: len(dat.columns)]) == list(dat.columns)
    assert res.attrs['measure'] == 'RR'
    assert res.attrs['slab'][:3] == ['Aronson', 'Ferguson & Simes', 'Rosenthal et al.1']


def test_var_names_and_numeric_subset_are_zero_based() -> None:
    res = metafopy.escalc('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'), var_names=('lor', 'vlor'), subset=[1, 4, 6])
    # R: escalc("OR", ..., var.names=c("lor","vlor"), subset=c(2,5,7))
    assert list(res['trial']) == [2, 5, 7]
    numpy.testing.assert_allclose(res['lor'], [-1.666190729026550, -0.219141085675433, -1.633775838228896], rtol=1e-10)
    numpy.testing.assert_allclose(res.attrs['ni'], [609, 10877, 3174])
    assert res.attrs['yi_names'] == ['lor']
    assert res.attrs['vi_names'] == ['vlor']


def test_without_data_returns_yi_vi_only() -> None:
    res = metafopy.escalc('COR', ri=[0.2, 0.3], ni=[10, 10], slab=['a', 'a'])
    assert list(res.columns) == ['yi', 'vi']
    assert res.attrs['slab'] == ['a.1', 'a.2']


def test_mapping_data_and_scalar_inputs() -> None:
    res = metafopy.escalc(
        'MC',
        m1i='a',
        m2i='b',
        sd1i='s1',
        sd2i='s2',
        ni='n',
        ri=0.5,
        data={'a': [10.0, 12.0], 'b': [9.0, 9.5], 's1': [2.0, 3.0], 's2': [2.0, 2.5], 'n': [20, 30]},
    )
    numpy.testing.assert_allclose(res['yi'], [1.0, 2.5])
    numpy.testing.assert_allclose(res['vi'], [4.0 / 20, (9 + 6.25 - 2 * 0.5 * 3 * 2.5) / 30])


def test_output_feeds_rma() -> None:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))
    res = metafopy.rma(data=dat)
    # metafor: rma(yi, vi, data=dat) for dat.bcg log risk ratios
    assert math.isclose(res.beta[0], -0.714532348365101, rel_tol=1e-8)
    assert math.isclose(res.tau2, 0.313243325980895, rel_tol=1e-7)


def test_large_smd_warning() -> None:
    with pytest.warns(UserWarning, match='unusually large'):
        metafopy.escalc('SMD', m1i=[10, 50], sd1i=[1, 1], n1i=[10, 10], m2i=[9, 1], sd2i=[1, 1], n2i=[10, 10])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        metafopy.escalc('SMD', m1i=[10, 50], sd1i=[1, 1], n1i=[10, 10], m2i=[9, 1], sd2i=[1, 1], n2i=[10, 10], cutoff=math.nan)


def test_infinite_values_are_recoded_with_a_warning() -> None:
    with pytest.warns(UserWarning, match='Recoded to NAs'):
        res = metafopy.escalc('PLO', xi=[0, 3], ni=[10, 10], to='none')
    assert numpy.isnan(res['yi'][0])
    assert numpy.isnan(res['vi'][0])


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'measure': 'XYZ', 'ai': [1]}, "Unknown 'measure'"),
        ({'measure': 'RR', 'ai': [1], 'bi': [2], 'ci': [3], 'di': [4], 'to': 'some'}, "Unknown 'to'"),
        ({'measure': 'RR', 'ai': [1], 'bi': [2], 'ci': [3], 'di': [4], 'vtype': 'XX'}, "Unknown 'vtype'"),
        ({'measure': 'RR', 'ai': [1], 'bi': [2], 'ci': [3], 'di': [4], 'vtype': 'UB'}, "must be either 'LS' or 'AV'"),
        ({'measure': 'RR', 'ai': [1], 'bi': [2], 'ci': [3]}, 'Cannot compute outcomes'),
        ({'measure': 'RR', 'ai': [1, 2], 'bi': [2], 'ci': [3], 'di': [4]}, 'not all of the same length'),
        ({'measure': 'RR', 'ai': [3], 'n1i': [2], 'bi': [1], 'ci': [3], 'di': [4]}, "'n1i' values are not equal"),
        ({'measure': 'COR', 'ri': [1.2], 'ni': [10]}, 'correlations are > 1'),
        ({'measure': 'GEN', 'yi': [0.1]}, "'vi' or 'sei'"),
        ({'measure': 'MD', 'm1i': [1], 'm2i': [1], 'sd1i': [-1], 'sd2i': [1], 'n1i': [10], 'n2i': [10]}, 'standard deviations are negative'),
    ],
)
def test_argument_errors(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        metafopy.escalc(**kwargs)  # type: ignore[arg-type]


RESERVED = ['HD', 'HR']


@pytest.mark.parametrize('measure', RESERVED)
def test_reserved_measures_not_implemented(measure: str) -> None:
    # metafor 5.2-1 accepts 'HR' and 'HD' but has no code to compute them (R fails with 'argument "vi" is missing')
    with pytest.raises(NotImplementedError, match='reserved'):
        metafopy.escalc(measure, ai=[1])


@pytest.mark.parametrize('measure', RESERVED)
def test_reserved_measures_accept_yi_and_vi(measure: str) -> None:
    # as in metafor, given yi and vi (or sei) the values are passed through and labelled with the measure
    dat = metafopy.escalc(measure, yi=[1.0, 2.0], sei=[0.1, 0.2])
    numpy.testing.assert_allclose(dat['vi'], [0.01, 0.04])
    numpy.testing.assert_allclose(dat['yi'], [1.0, 2.0])
    assert dat.attrs['measure'] == measure
    with pytest.raises(ValueError, match=r"Must specify the 'vi' or 'sei' argument\."):
        metafopy.escalc(measure, yi=[1.0, 2.0])


def test_correct_must_be_logical_for_ratio_measures() -> None:
    with pytest.raises(TypeError, match='logical'):
        metafopy.escalc('ROM', m1i=[1], m2i=[1], sd1i=[1], sd2i=[1], n1i=[10], n2i=[10], correct='approx')


def test_cmicalc() -> None:
    # R: cmicalc(c(2, 10, 400))
    numpy.testing.assert_allclose(metafopy.cmicalc([2, 10, 400]), [0.564189583547756, 0.922745608053087, 0.998123631714881], rtol=1e-12)
    assert numpy.isnan(metafopy.cmicalc([1])[0])
    assert metafopy.cmicalc([10], method='approx')[0] == 1 - 3 / 39
    assert metafopy.cmicalc([10], method=False)[0] == 1.0


def test_dataframe_index_is_reset() -> None:
    dat = metafopy.datasets.load('bcg').iloc[::-1]
    res = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    assert isinstance(res, pandas.DataFrame)
    assert list(res.index) == list(range(13))
    assert res['trial'].iloc[0] == 13
