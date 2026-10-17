"""vcov(type=), the data summaries of the fits, coef(summary()) and cbind()/rbind() of escalc frames, checked against R metafor 5.2-1.

The reference values come from tests/r/generate_audit4_vcov.R.
"""

import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_vcov')

NA_ACTIONS = ('omit', 'exclude', 'pass', 'fail')
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')
VCOV_TYPES = ('fixed', 'beta', 'alpha', 'delta', 'obs', 'marginal', 'fitted', 'resid')
# R's own failure for type = "resid" with na.fail (the variable `out` is never defined); metafopy refuses with its own message
R_BUG = "object 'out' not found"
# the tolerance of fits that depend on an optimiser
OPT = 1e-5
TIGHT = 1e-8
LOOSE = {'mv_t': OPT, 'ls_knha': OPT, 'mv': OPT, 'mv_mods': OPT, 'ls': OPT, 'ls_na': OPT, 'sel': OPT, 'sel_mods': OPT, 'rob_mv': OPT, 'rob_cs': OPT, 'tf': 1e-7}


@pytest.fixture(scope='module')
def dat0() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def dat(dat0: pandas.DataFrame) -> pandas.DataFrame:
    out = dat0.copy()
    out.attrs = dict(dat0.attrs)
    out.loc[4, 'ablat'] = numpy.nan
    out.loc[2, 'yi'] = numpy.nan
    return out


@pytest.fixture(scope='module')
def bcg() -> pandas.DataFrame:
    return metafopy.datasets.load('dat.bcg')


@pytest.fixture(scope='module')
def fits(dat0: pandas.DataFrame, dat: pandas.DataFrame, bcg: pandas.DataFrame) -> dict[str, typing.Any]:
    cells: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg'}
    uni0 = metafopy.rma('yi', 'vi', data=dat0)
    return {
        'uni': uni0,
        'uni_mods': metafopy.rma('yi', 'vi', mods='~ablat', data=dat),
        'uni_w': metafopy.rma('yi', 'vi', mods='~ablat', weights=1 / numpy.sqrt(dat0['vi'].to_numpy()), data=dat0),
        'mv': metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat0),
        'mv_mods': metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~ 1 | trial', data=dat),
        'mh': metafopy.rma_mh(data=bcg, **cells),
        'peto': metafopy.rma_peto(data=bcg, **cells),
        'ls': metafopy.rma('yi', 'vi', scale='~ablat', data=dat0),
        'ls_na': metafopy.rma('yi', 'vi', mods='~ablat', scale='~ablat', data=dat),
        'sel': metafopy.selmodel(uni0, type='halfnorm'),
        'sel_mods': metafopy.selmodel(metafopy.rma('yi', 'vi', mods='~ablat', data=dat), type='halfnorm'),
        'rob': metafopy.robust(uni0, cluster='trial'),
        'rob_mods': metafopy.robust(metafopy.rma('yi', 'vi', mods='~ablat', data=dat), cluster='trial'),
        'rob_cs': metafopy.robust(metafopy.rma('yi', 'vi', mods='~ablat', data=dat0), cluster='alloc', club_sandwich=True),
        'rob_mv': metafopy.robust(metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat0), cluster='trial'),
        'tf': metafopy.trimfill(uni0),
    }


def matrix(spec: dict[str, typing.Any]) -> numpy.typing.NDArray[numpy.float64]:
    shape = tuple(refutil.as_list(spec['dim']))
    values = [numpy.nan if v is None else v for v in numpy.asarray(spec['values'], dtype=object).reshape(-1)]
    return numpy.asarray(values, dtype=float).reshape(shape)


def check_matrix(actual: pandas.DataFrame, spec: dict[str, typing.Any], what: str, rtol: float) -> None:
    expected = matrix(spec)
    assert actual.shape == expected.shape, f'{what}: shape {actual.shape} against R {expected.shape}'
    numpy.testing.assert_allclose(actual.to_numpy(), expected, rtol=rtol, atol=1e-10, equal_nan=True, err_msg=what)
    rows = spec['rows']
    if rows is not None:
        assert list(actual.index) == refutil.as_list(rows), f'{what}: row labels'
        assert list(actual.columns) == refutil.as_list(rows), f'{what}: column labels'


def cases() -> list[tuple[str, str, str]]:
    return [(name, ty, na) for name in REF['vcov'] for ty in VCOV_TYPES for na in NA_ACTIONS]


@pytest.mark.parametrize(('name', 'ty', 'na'), cases())
def test_vcov_against_r(fits: dict[str, typing.Any], name: str, ty: str, na: str) -> None:
    ref = REF['vcov'][name][ty][na]
    fit = fits[name]
    what = f'{name} {ty} {na}'
    rtol = LOOSE.get(name, TIGHT)
    if 'error' in ref:
        with pytest.raises(metafopy.errors.MetaforError) as err:
            fit.vcov(ty, na_action=na)
        if ref['error'] != R_BUG:
            assert ref['error'] in str(err.value), what
        return
    out = fit.vcov(ty, na_action=na)
    if 'dim' in ref:
        check_matrix(out, ref, what, rtol)
        return
    assert isinstance(out, dict), what
    assert list(out) == list(ref), what
    for part, spec in ref.items():
        check_matrix(out[part], spec, f'{what} {part}', rtol)


def test_vcov_function_and_partial_matching(fits: dict[str, typing.Any]) -> None:
    ls = fits['ls']
    full = metafopy.vcov(ls)
    assert isinstance(full, dict)
    assert list(full) == ['beta', 'alpha']
    numpy.testing.assert_array_equal(metafopy.vcov(ls, 'alpha').to_numpy(), ls.va)
    partial: typing.Any = metafopy.vcov  # partial matching of `type` is allowed at run time, as in R
    numpy.testing.assert_array_equal(partial(fits['uni'], 'be').to_numpy(), fits['uni'].vb)
    numpy.testing.assert_array_equal(partial(fits['uni'], 'm').to_numpy(), metafopy.vcov(fits['uni'], 'obs').to_numpy())
    with pytest.raises(metafopy.errors.InputError, match='type'):
        partial(fits['uni'], 'f')
    with pytest.raises(metafopy.errors.InputError, match=r'na\.action'):
        metafopy.vcov(fits['uni'], 'obs', na_action='drop')  # type: ignore[call-overload]
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.vcov(numpy.eye(2))


def test_vcov_fitted_na_rows_of_binary_fits(bcg: pandas.DataFrame) -> None:
    mh = metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg.assign(tpos=[float('nan') if i == 1 else v for i, v in enumerate(bcg['tpos'])]))
    for spec, na in zip(REF['vcov_na_tables']['mh_fitted'], ('omit', 'exclude'), strict=True):
        check_matrix(mh.vcov('fitted', na_action=na), spec, f'mh fitted {na}', TIGHT)  # type: ignore[arg-type]


def test_glmm_vcov_fitted(bcg: pandas.DataFrame) -> None:
    """lme4 is not installed where the fixtures are made, so rma_glmm is checked against X vb X' directly."""
    res = metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, measure='OR', model='CM.EL')
    fitted = res.vcov('fitted')
    assert fitted.shape == (13, 13)
    numpy.testing.assert_allclose(fitted.to_numpy(), numpy.ones((13, 1)) @ res.vb @ numpy.ones((1, 13)), rtol=1e-12)
    assert list(fitted.index) == [str(i) for i in range(1, 14)]
    with pytest.raises(metafopy.errors.NotSupportedError, match='marginal var-cov matrix'):
        res.vcov('obs')
    with pytest.raises(metafopy.errors.ArgumentError, match='hatvalues'):
        res.vcov('resid')


@pytest.mark.parametrize('name', list(REF['fields']))
def test_fields_against_r(fits: dict[str, typing.Any], dat0: pandas.DataFrame, dat: pandas.DataFrame, bcg: pandas.DataFrame, name: str) -> None:
    ref = REF['fields'][name]
    if name in ('mh_na', 'peto_na'):
        cells: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg'}
        data = bcg.assign(tpos=[float('nan') if i == 1 else v for i, v in enumerate(bcg['tpos'])])
        fit = metafopy.rma_mh(data=data, **cells) if name == 'mh_na' else metafopy.rma_peto(data=data, **cells)
    else:
        fit = fits[name]
    rtol = LOOSE.get(name, TIGHT)
    for key, expected in ref.items():
        attr = key.replace('.', '_')
        value = getattr(fit, attr)
        if isinstance(expected, dict):
            numpy.testing.assert_allclose(
                numpy.asarray(value, dtype=float).reshape(matrix(expected).shape), matrix(expected), rtol=rtol, atol=1e-12, err_msg=f'{name} {key}'
            )
        elif expected is None:
            assert value is None or numpy.isnan(value), f'{name} {key}'
        else:
            numpy.testing.assert_allclose(
                numpy.asarray(value, dtype=float), numpy.asarray(refutil.as_list(expected), dtype=float), rtol=rtol, err_msg=f'{name} {key}'
            )


def test_b_is_beta(fits: dict[str, typing.Any]) -> None:
    for name in ('uni', 'mv', 'mh', 'peto', 'ls', 'sel', 'rob', 'tf'):
        numpy.testing.assert_array_equal(fits[name].b, fits[name].beta)


def table(spec: dict[str, typing.Any]) -> pandas.DataFrame:
    return pandas.DataFrame(
        {k: [numpy.nan if v is None else v for v in refutil.as_list(spec['values'][k])] for k in spec['names']}, index=refutil.as_list(spec['rows'])
    )


def check_table(actual: pandas.DataFrame, spec: dict[str, typing.Any], what: str, rtol: float) -> None:
    expected = table(spec)
    assert list(actual.columns) == list(expected.columns), what
    assert list(actual.index) == list(expected.index), what
    numpy.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=rtol, atol=1e-10, equal_nan=True, err_msg=what)


COEF_FITS = ('uni', 'uni_mods', 'uni_knha', 'uni_t', 'mv', 'mv_t', 'mh', 'peto', 'ls', 'ls_knha', 'sel', 'sel_mods', 'rob', 'rob_mods', 'rob_cs', 'tf')


@pytest.fixture(scope='module')
def coef_fits(fits: dict[str, typing.Any], dat0: pandas.DataFrame, dat: pandas.DataFrame) -> dict[str, typing.Any]:
    return {
        **fits,
        'uni_mods': metafopy.rma('yi', 'vi', mods='~ablat + alloc', data=dat),
        'uni_knha': metafopy.rma('yi', 'vi', mods='~ablat', data=dat0, test='knha'),
        'uni_t': metafopy.rma('yi', 'vi', data=dat0, test='t'),
        'mv_t': metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~ 1 | trial', data=dat, test='t'),
        'ls_knha': metafopy.rma('yi', 'vi', scale='~ablat', data=dat0, test='knha'),
    }


@pytest.mark.parametrize('name', COEF_FITS)
def test_coef_summary_against_r(coef_fits: dict[str, typing.Any], name: str) -> None:
    ref = REF['coef_summary'][name]
    fit = coef_fits[name]
    rtol = LOOSE.get(name, TIGHT)
    out = metafopy.coef_summary(fit)
    assert fit.coef_summary().equals(out) if isinstance(out, pandas.DataFrame) else list(fit.coef_summary()) == list(out)
    if 'names' in ref:
        assert isinstance(out, pandas.DataFrame)
        check_table(out, ref, name, rtol)
        return
    assert isinstance(out, dict)
    assert list(out) == list(ref)
    for part, spec in ref.items():
        check_table(out[part], spec, f'{name} {part}', rtol)


def test_coef_summary_type(coef_fits: dict[str, typing.Any]) -> None:
    ref = REF['coef_summary_type']
    check_table(metafopy.coef_summary(coef_fits['ls'], 'beta'), ref['ls_beta'], 'ls beta', OPT)
    check_table(metafopy.coef_summary(coef_fits['ls'], 'alpha'), ref['ls_alpha'], 'ls alpha', OPT)
    check_table(metafopy.coef_summary(coef_fits['sel'], 'beta'), ref['sel_beta'], 'sel beta', OPT)
    check_table(metafopy.coef_summary(coef_fits['sel'], 'delta'), ref['sel_delta'], 'sel delta', OPT)
    check_table(metafopy.coef_summary(coef_fits['uni'], 'alpha'), ref['uni_alpha'], 'uni alpha', TIGHT)
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.coef_summary(numpy.eye(2))


def escalc_frames() -> dict[str, pandas.DataFrame]:
    bcg = metafopy.datasets.load('dat.bcg')
    cells: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg'}
    a = metafopy.escalc('RR', data=bcg.iloc[0:5], slab=[f's{i}' for i in range(1, 6)], digits=3, **cells)
    b = metafopy.escalc('RR', data=bcg.iloc[5:9], slab=[f's{i}' for i in range(6, 10)], digits=5, **cells)
    b_dup = metafopy.escalc('RR', data=bcg.iloc[5:9], slab=[f's{i}' for i in range(1, 5)], digits=5, **cells)
    data = pandas.DataFrame({'m1i': [10, 12], 'm2i': [9, 10], 'sd1i': [2, 3], 'sd2i': [2, 3], 'n1i': [20, 30], 'n2i': [20, 30]})
    nia = metafopy.escalc('SMD', m1i='m1i', m2i='m2i', sd1i='sd1i', sd2i='sd2i', n1i='n1i', n2i='n2i', data=data)
    y1 = metafopy.escalc('RR', data=bcg.iloc[0:5], var_names=('y1', 'v1'), **cells)[['y1', 'v1']]
    y1.attrs = {k: v for k, v in metafopy.escalc('RR', data=bcg.iloc[0:5], var_names=('y1', 'v1'), **cells).attrs.items()}
    y2_full = metafopy.escalc('OR', data=bcg.iloc[0:5], var_names=('y2', 'v2'), digits=2, **cells)
    y2 = y2_full[['y2', 'v2']]
    y2.attrs = dict(y2_full.attrs)
    return {'a': a, 'b': b, 'b_dup': b_dup, 'nia': nia, 'y1': y1, 'y2': y2}


def check_bound(actual: pandas.DataFrame, ref: dict[str, typing.Any], what: str, measure: bool = True) -> None:
    assert list(actual.columns) == ref['names'], what
    assert actual.shape[0] == ref['nrow'], what
    for col, values in ref['numeric'].items():
        base, _, suffix = col.rpartition('.')
        if col not in actual.columns and suffix.isdigit():
            found = actual[base].iloc[:, int(suffix)]  # R numbers repeated column names trial, trial.1, ...
        else:
            found = actual[col]
            found = found.iloc[:, 0] if isinstance(found, pandas.DataFrame) else found
        refutil.close(found.to_numpy(dtype=float), values, f'{what} {col}', rtol=1e-12)
    for key, names in ref['attrs'].items():
        assert actual.attrs[key.replace('.', '_')] == refutil.as_list(names), f'{what} {key}'
    assert {str(k) for k in actual.attrs if str(k).endswith('_names')} == {k.replace('.', '_') for k in ref['attrs']}, what
    if ref['digits'] is None:
        assert actual.attrs.get('digits') is None, what
    else:
        assert actual.attrs.get('digits', 4) in (None, ref['digits']), what  # R's default is 4; escalc() here leaves it unset
    if measure:
        assert actual.attrs['measure'] == ref['measure'], what
    slab = ref['slab']
    assert list(actual.attrs.get('slab') or []) == ([] if slab in (None, []) else refutil.as_list(slab)), what
    ni = actual.attrs.get('ni')
    if ref['ni'] is None:
        assert ni is None, what
    else:
        refutil.close(ni, ref['ni'], f'{what} ni')


def test_rbind_escalc() -> None:
    f = escalc_frames()
    for key, args in (('rbind_ab', (f['a'], f['b'])), ('rbind_dup', (f['a'], f['b_dup'])), ('rbind_aa', (f['a'], f['a'])), ('rbind_ni', (f['nia'], f['nia']))):
        out = metafopy.rbind_escalc(*args)
        check_bound(out, REF['bind'][key], key)
        assert out.index.is_unique, key


def test_cbind_escalc() -> None:
    f = escalc_frames()
    summary = metafopy.summary_escalc(f['a'])
    check_bound(metafopy.cbind_escalc(f['a'], summary[['sei', 'zi']]), REF['bind']['cbind_summary'], 'cbind_summary')
    check_bound(metafopy.cbind_escalc(f['a'], z=numpy.arange(1, 6)), REF['bind']['cbind_extra'], 'cbind_extra')
    check_bound(metafopy.cbind_escalc(f['a'], f['a']), REF['bind']['cbind_aa'], 'cbind_aa')
    # R keeps a measure for each column; a frame has one set of attributes, so the measure of the first argument is kept
    check_bound(metafopy.cbind_escalc(f['y1'], f['y2']), REF['bind']['cbind_names'], 'cbind_names', measure=False)


def test_bind_escalc_errors() -> None:
    f = escalc_frames()
    with pytest.raises(metafopy.errors.InputError, match='numbers of columns'):
        metafopy.rbind_escalc(f['a'], f['a'][['yi', 'vi']])
    with pytest.raises(metafopy.errors.InputError, match='names do not match'):
        metafopy.rbind_escalc(f['a'], f['a'].rename(columns={'author': 'who'}))
    with pytest.raises(metafopy.errors.InputError, match='differing number of rows'):
        metafopy.cbind_escalc(f['a'], f['b'])
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.rbind_escalc(f['a'], numpy.zeros(3))  # type: ignore[arg-type]
    plain = f['a'].copy()
    plain.attrs = {}
    out = metafopy.rbind_escalc(f['a'], plain)
    assert out.attrs.get('ni') is None
    assert 'slab' not in out.attrs
    assert out.attrs['yi_names'] == ['yi']
