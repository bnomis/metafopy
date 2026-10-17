"""aggregate_escalc() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R, mostly the help page examples)."""

import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

REF = refutil.load_phase7('aggregate_escalc')


@pytest.fixture(scope='module')
def assink() -> pandas.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.escalc('SMD', yi='yi', vi='vi', data=metafopy.datasets.load('assink2016'))


@pytest.fixture(scope='module')
def ishak() -> pandas.DataFrame:
    dat = refutil.r_frame(REF['ishak_long'])
    dat = metafopy.escalc('MD', yi='yi', vi='vi', data=dat)
    dat['obs'] = REF['obs']
    return dat


def test_id_structure_with_addk() -> None:
    out = metafopy.aggregate_escalc(metafopy.datasets.load('konstantopoulos2011'), cluster='district', struct='ID', addk=True)
    refutil.assert_frame(out, REF['id_addk'], 'ID')
    assert out.attrs['yi_names'] == ['yi']
    assert out.attrs['measure'] == 'SMD'
    assert out.attrs['digits'] == 3


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    ('key', 'kwargs'),
    [
        ('cs', {'rho': 0.6}),
        ('cs_unweighted', {'rho': 0.6, 'weighted': False}),
        ('cs_rho_vector', {'rho': REF['rho_vector']}),
        # auditme: accept[literal-duplication] each test case spells out its own R arguments
        ('cs_select', {'rho': 0.6, 'select': ['study', 'yi', 'vi', 'deltype'], 'addk': True}),
    ],
)
def test_compound_symmetry(assink: pandas.DataFrame, key: str, kwargs: dict[str, object]) -> None:
    out = metafopy.aggregate_escalc(assink, cluster='study', **kwargs)  # type: ignore[arg-type]
    refutil.assert_frame(out, REF[key], key)
    assert out.attrs['measure'] == 'SMD'


def test_subset(assink: pandas.DataFrame) -> None:
    out = metafopy.aggregate_escalc(assink, cluster='study', rho=0.6, subset=(assink['pubstatus'] == 1).to_numpy())
    refutil.assert_frame(out, REF['cs_subset'], 'subset')


def test_select_by_position(assink: pandas.DataFrame) -> None:
    out = metafopy.aggregate_escalc(assink, cluster='study', rho=0.6, select=[0, 3, 4, 7])
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    assert list(out.columns) == ['study', 'yi', 'vi', 'deltype']


def test_given_V(assink: pandas.DataFrame) -> None:
    V1 = metafopy.vcalc('vi', cluster='study', obs='esid', data=assink, rho=0.6)
    assert isinstance(V1, numpy.ndarray)
    refutil.assert_frame(metafopy.aggregate_escalc(assink, cluster='study', V=V1), REF['V1'], 'V1')
    refutil.assert_frame(metafopy.aggregate_escalc(assink, cluster='study', V=numpy.array(REF['V2'])), REF['V2_agg'], 'V2')
    refutil.assert_frame(metafopy.aggregate_escalc(assink, cluster='study', V='vi'), REF['V_vector'], 'V vector')


@pytest.mark.parametrize(('key', 'na_rm'), [('na_rm', True), ('na_rm_ff', False), ('na_rm_tf', [True, False])])
def test_missing_values(assink: pandas.DataFrame, key: str, na_rm: bool | list[bool]) -> None:
    dat = assink.copy()
    dat.loc[[1, 9], 'yi'] = numpy.nan
    dat.loc[[0, 2], 'deltype'] = None
    dat.loc[4, 'year'] = numpy.nan
    refutil.assert_frame(metafopy.aggregate_escalc(dat, cluster='study', rho=0.6, na_rm=na_rm), REF[key], key)


def test_var_names(assink: pandas.DataFrame) -> None:
    dat = assink.rename(columns={'yi': 'es', 'vi': 'var'})
    dat.attrs = {}
    refutil.assert_frame(metafopy.aggregate_escalc(dat, cluster='study', rho=0.6, var_names=['es', 'var']), REF['var_names'], 'var_names')


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    ('key', 'kwargs'),
    [
        ('car', {'struct': 'CAR', 'time': 'time', 'phi': 0.9}),
        ('cs_car', {'struct': 'CS+CAR', 'time': 'time', 'phi': 0.9, 'rho': 0.3}),
        ('csxcar', {'struct': 'CS*CAR', 'time': 'time', 'obs': 'obs', 'phi': 0.9, 'rho': 0.3}),
        ('car_unweighted', {'struct': 'CAR', 'time': 'time', 'phi': 0.5, 'weighted': False, 'addk': True}),
    ],
)
def test_autoregressive(ishak: pandas.DataFrame, key: str, kwargs: dict[str, object]) -> None:
    out = metafopy.aggregate_escalc(ishak, cluster='study', **kwargs)  # type: ignore[arg-type]
    # the R data got its 'obs' variable only for the CS*CAR fit
    if 'obs' not in REF[key]['names']:
        out = out.drop(columns='obs')
    refutil.assert_frame(out, REF[key], key)


def test_digits_and_fun(ishak: pandas.DataFrame) -> None:
    out = metafopy.aggregate_escalc(ishak, cluster='study', struct='CAR', time='time', phi=0.5, digits=2)
    # metafor expands digits to one value per kind of number; metafopy keeps the value given
    assert set(refutil.as_list(REF['digits'])) == {out.attrs['digits']}
    first = metafopy.aggregate_escalc(
        ishak, cluster='study', struct='CAR', time='time', phi=0.5, fun=[lambda s: s.iloc[0], lambda s: s.iloc[0], lambda s: s.iloc[0]]
    )
    assert first['time'].tolist() == ishak.groupby('study', sort=False)['time'].first().tolist()


def test_errors(assink: pandas.DataFrame) -> None:
    with pytest.raises(metafopy.errors.InputError, match="Unknown 'struct'"):
        metafopy.aggregate_escalc(assink, cluster='study', struct='UN')
    with pytest.raises(metafopy.errors.InputError, match="Must specify 'rho'"):
        metafopy.aggregate_escalc(assink, cluster='study')
    with pytest.raises(metafopy.errors.InputError, match="Must specify 'phi'"):
        metafopy.aggregate_escalc(assink, cluster='study', struct='CAR')
    with pytest.raises(metafopy.errors.InputError, match='must be in'):
        metafopy.aggregate_escalc(assink, cluster='study', rho=1.5)
    with pytest.raises(metafopy.errors.InputError, match="'fun' must be"):
        metafopy.aggregate_escalc(assink, cluster='study', rho=0.6, fun=[len])
    with pytest.warns(UserWarning, match='not positive definite'), pytest.raises(metafopy.errors.EstimationError, match='non-positive-definite'):
        metafopy.aggregate_escalc(assink, cluster='study', rho=-1)
    V = numpy.diag(assink['vi'].to_numpy())
    V[0, -1] = V[-1, 0] = 0.001
    with pytest.warns(UserWarning, match='non-zero covariances'):
        metafopy.aggregate_escalc(assink, cluster='study', V=V)
