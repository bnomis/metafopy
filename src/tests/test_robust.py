"""robust() for rma.uni and rma.mv fits (default sandwich estimators and clubSandwich) checked against R metafor 5.2-1 and clubSandwich 0.7.0."""

import json
import pathlib
import typing

import numpy
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase4' / 'robust.json').read_text(encoding='utf-8'))
RTOL = 1e-6

KON = metafopy.datasets.load('konstantopoulos2011')
ASS = metafopy.datasets.load('assink2016')
MODS = 'I(year - 1990)'


def uni(**kwargs: typing.Any) -> metafopy.RMAResult:
    return metafopy.rma_uni('yi', 'vi', data=kwargs.pop('data', KON), **kwargs)


def mv(mods: str | None = None, **kwargs: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods=mods, random=kwargs.pop('random', '~ 1 | district/school'), data=kwargs.pop('data', KON), **kwargs)


def weights_arg() -> numpy.typing.NDArray[numpy.float64]:
    return numpy.asarray(1 / numpy.sqrt(KON['vi'].to_numpy()))


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
MODELS: dict[str, typing.Callable[[], typing.Any]] = {
    'uni_int': lambda: uni(),
    'uni_mods': lambda: uni(mods='~ ' + MODS),
    'uni_fe': lambda: uni(mods='~ ' + MODS, method='FE'),
    'uni_weights': lambda: uni(mods='~ ' + MODS, weights=weights_arg()),
    'uni_subset': lambda: uni(mods='~ ' + MODS, subset='district != 11'),
    'uni_btt': lambda: uni(mods='~ ' + MODS + ' + I((year - 1990)**2)'),
    'uni_level90': lambda: uni(mods='~ ' + MODS, level=90),
    'mv_int': lambda: mv(),
    'mv_mods': lambda: mv('~ ' + MODS),
    'mv_weights': lambda: mv('~ ' + MODS, W=1 / KON['vi'].to_numpy()),
    'mv_sparse': lambda: mv('~ ' + MODS, sparse=True),
    'mv_subset': lambda: mv('~ ' + MODS, subset='district != 11'),
    # auditme: accept[literal-duplication] the fixture names repeat as the R script that generates them repeats them
    'ass': lambda: metafopy.rma_mv('yi', 'vi', mods='~ year', random=['~ 1 | study', '~ 1 | esid'], data=ASS),
}

# (fixture key, model, cluster, keyword arguments)
# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
DEFAULT_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'uni_int': ('uni_int', 'district', {}),
    'uni_mods': ('uni_mods', 'district', {}),
    'uni_mods_cr0': ('uni_mods', 'district', {'adjust': False}),
    'uni_mods_stata1': ('uni_mods', 'district', {'adjust': 'Stata1'}),
    'uni_mods_stata2': ('uni_mods', 'district', {'adjust': 'Stata2'}),
    'uni_fe': ('uni_fe', 'district', {}),
    'uni_weights': ('uni_weights', 'district', {}),
    'uni_subset': ('uni_subset', 'district', {}),
    'mv_int': ('mv_int', 'district', {}),
    'mv_mods': ('mv_mods', 'district', {}),
    'mv_mods_cr0': ('mv_mods', 'district', {'adjust': False}),
    'mv_mods_stata1': ('mv_mods', 'district', {'adjust': 'Stata1'}),
    'mv_mods_stata2': ('mv_mods', 'district', {'adjust': 'Stata2'}),
    'mv_weights': ('mv_weights', 'district', {}),
    'mv_sparse': ('mv_sparse', 'district', {}),
    'mv_subset': ('mv_subset', 'district', {}),
    'mv_school': ('mv_mods', 'school', {}),
    'ass_default': ('ass', 'study', {}),
}

CS_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {}
for family in ('uni', 'mv'):
    for suffix, model in (('int', f'{family}_int'), ('mods', f'{family}_mods')):
        for vcov in ('CR0', 'CR1', 'CR1p', 'CR1S', 'CR2', 'CR3'):
            CS_CASES[f'{family}_cs_{suffix}_{vcov}'] = (model, 'district', {'club_sandwich': True, 'vcov': vcov})
    for ct in ('z', 'naive-t', 'naive-tp'):
        CS_CASES[f'{family}_cs_coef_{ct}'] = (f'{family}_mods', 'district', {'club_sandwich': True, 'coef_test': ct})
    for wt in ('chi-sq', 'Naive-F', 'Naive-Fp'):
        CS_CASES[f'{family}_cs_wald_{wt}'] = (f'{family}_mods', 'district', {'club_sandwich': True, 'wald_test': wt})
    CS_CASES[f'{family}_cs_weights'] = (f'{family}_weights', 'district', {'club_sandwich': True})
    CS_CASES[f'{family}_cs_subset'] = (f'{family}_subset', 'district', {'club_sandwich': True})
CS_CASES['uni_cs_fe'] = ('uni_fe', 'district', {'club_sandwich': True})
CS_CASES['uni_cs_conf'] = ('uni_mods', 'district', {'club_sandwich': True, 'conf_test': 'z'})
CS_CASES['uni_cs_btt'] = ('uni_btt', 'district', {'club_sandwich': True})
CS_CASES['uni_cs_level90'] = ('uni_level90', 'district', {'club_sandwich': True})

ALL_CASES = {**DEFAULT_CASES, **CS_CASES}


def fit(key: str) -> typing.Any:
    model, cluster, kwargs = ALL_CASES[key]
    return metafopy.robust(MODELS[model](), cluster, **kwargs)


def vector(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = 1e-10) -> None:
    refutil.close(numpy.asarray(actual, dtype=float).reshape(-1), numpy.asarray(expected, dtype=float).reshape(-1).tolist(), what, rtol=rtol, atol=atol)


def finite(values: typing.Any) -> numpy.typing.NDArray[numpy.float64]:
    """R's Inf is serialised as null (NaN) in the reference file."""
    a = numpy.asarray(values, dtype=float)
    return numpy.where(numpy.isinf(a), numpy.nan, a)


@pytest.mark.parametrize('key', sorted(ALL_CASES))
def test_numbers(key: str) -> None:
    ref = REF[key]
    res = fit(key)
    # auditme: accept[literal-duplication] the fixture names repeat as the R script that generates them repeats them
    for attr, name in (('se', 'se'), ('zval', 'zval'), ('pval', 'pval'), ('ci_lb', 'ci_lb'), ('ci_ub', 'ci_ub')):
        vector(getattr(res, attr), ref[name], f'{key} {name}')
    vector(res.vb, numpy.array(ref['vb']).reshape(-1), f'{key} vb')
    vector(res.beta, ref['beta'], f'{key} beta')
    vector(res.QM, ref['QM'], f'{key} QM')
    vector(res.QMp, ref['QMp'], f'{key} QMp')
    vector(finite([float(v) for v in res.QMdf]), ref['QMdf'], f'{key} QMdf')
    assert res.robust is not None
    vector(finite(res.robust.dfs), numpy.broadcast_to(numpy.asarray(ref['ddf'], dtype=float), res.robust.dfs.shape), f'{key} dfs')
    assert res.robust.n == ref['n']
    vector(res.robust.tcl, ref['tcl'], f'{key} tcl', rtol=0)
    assert res.robust.vbest == ref['vbest']
    assert res.robust.robumethod == ref['robumethod']
    assert res.test == 't'


def test_ass_clubsandwich_unsupported() -> None:
    """clubSandwich refuses the Assink model in R as well (random effects are not nested in the clusters)."""
    assert 'error' in REF['ass_cs']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.robust(MODELS['ass'](), 'study', club_sandwich=True)


def test_errors() -> None:
    fit_uni = MODELS['uni_mods']()
    with pytest.raises(metafopy.errors.InputError, match='must be larger'):
        metafopy.robust(fit_uni, numpy.zeros(len(KON)))
    with pytest.raises(metafopy.errors.InputError, match='Cannot find'):
        metafopy.robust(fit_uni, 'nope')
    with pytest.raises(metafopy.errors.InputError, match='does not correspond'):
        metafopy.robust(fit_uni, numpy.arange(3))
    with pytest.raises(metafopy.errors.InputError, match='Unknown'):
        metafopy.robust(fit_uni, 'district', club_sandwich=True, vcov='CR9')
    with pytest.raises(metafopy.errors.InputError, match='single cluster'):
        metafopy.robust(fit_uni, numpy.zeros(len(KON)), club_sandwich=True)
    with pytest.raises(metafopy.errors.InputError, match='not supported'):
        metafopy.robust(fit_uni, 'district', club_sandwich=True, coef_test='saddlepoint')


@pytest.mark.parametrize('key', sorted(ALL_CASES))
def test_print(key: str) -> None:
    if key == 'uni_cs_btt':
        pytest.skip('coefficient label of the squared term follows the formula engine (** instead of ^)')
    assert refutil.r_print(fit(key).summary(digits=4)).strip() == refutil.r_print(REF[key]['print']).strip()


@pytest.mark.parametrize(('key', 'model'), [('predict_uni', 'uni_mods'), ('predict_mv', 'mv_mods')])
def test_predict(key: str, model: str) -> None:
    ref = REF[key]
    pr = metafopy.robust(MODELS[model](), 'district').predict([-5, 0, 5])
    # auditme: accept[literal-duplication] the fixture names repeat as the R script that generates them repeats them
    for field in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        vector(getattr(pr, field), ref[field], f'{key} {field}', rtol=1e-5)
    with pytest.raises(metafopy.errors.InputError, match='not supported'):
        metafopy.robust(MODELS[model](), 'district', club_sandwich=True).predict([0])
