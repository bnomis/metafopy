"""robust() with coef_test='saddlepoint' and the HTA, HTB, EDF and EDT Wald tests checked against R metafor 5.2-1 and clubSandwich 0.7.0."""

import json
import pathlib
import typing
import warnings

import numpy
import pytest
import refutil

import metafopy
import metafopy.core.clubsandwich
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase6' / 'robust_extras.json').read_text(encoding='utf-8'))
# the saddlepoint root is found to the tolerance of uniroot() (eps^0.25), which the port reproduces step by step
RTOL = 1e-7

KON = metafopy.datasets.load('konstantopoulos2011').assign(yr=lambda d: d['year'] - 1990)
KON['yr2'] = KON['yr'] ** 2
KON['yr3'] = KON['yr'] ** 3
RANDOM = '~ 1 | district/school'
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
WALD_TESTS = ('chi-sq', 'Naive-F', 'Naive-Fp', 'HTA', 'HTB', 'HTZ', 'EDF', 'EDT')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
VCOVS = ('CR0', 'CR1', 'CR1p', 'CR1S', 'CR3')


def uni(mods: str | None = None, **kwargs: typing.Any) -> metafopy.RMAResult:
    return typing.cast(metafopy.RMAResult, metafopy.rma_uni('yi', 'vi', mods=mods, data=KON, **kwargs))


def mv(mods: str | None = None, **kwargs: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods=mods, random=RANDOM, data=KON, **kwargs)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
MODELS: dict[str, typing.Callable[[], typing.Any]] = {
    'uni_int': lambda: uni(),
    'uni_mods': lambda: uni('~ yr'),
    'uni_two': lambda: uni('~ yr + yr2'),
    'uni_three': lambda: uni('~ yr + yr2 + yr3'),
    'uni_btt23': lambda: uni('~ yr + yr2 + yr3', btt=[1, 2]),
    'uni_btt24': lambda: uni('~ yr + yr2 + yr3', btt=[1, 3]),
    'uni_weights': lambda: uni('~ yr + yr2', weights=1 / numpy.sqrt(KON['vi'].to_numpy())),
    'uni_fe': lambda: uni('~ yr + yr2', method='FE'),
    'mv_int': lambda: mv(),
    'mv_mods': lambda: mv('~ yr'),
    'mv_two': lambda: mv('~ yr + yr2'),
    'mv_btt23': lambda: mv('~ yr + yr2 + yr3', btt=[1, 2]),
    'mv_weights': lambda: mv('~ yr + yr2', W=1 / KON['vi'].to_numpy()),
}

# fixture key -> (model, keyword arguments of robust)
CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {}
for name in MODELS:
    CASES[f'{name}_saddle'] = (name, {'coef_test': 'saddlepoint'})
    CASES[f'{name}_saddle_conf_z'] = (name, {'coef_test': 'saddlepoint', 'conf_test': 'naive-t'})
    for wt in WALD_TESTS:
        CASES[f'{name}_wald_{wt}'] = (name, {'wald_test': wt})
for vcov in VCOVS:
    CASES[f'uni_two_saddle_{vcov}'] = ('uni_two', {'vcov': vcov, 'coef_test': 'saddlepoint'})
    CASES[f'mv_two_saddle_{vcov}'] = ('mv_two', {'vcov': vcov, 'coef_test': 'saddlepoint'})
    for wt in ('HTA', 'HTB', 'EDF', 'EDT'):
        CASES[f'uni_two_{vcov}_{wt}'] = ('uni_two', {'vcov': vcov, 'wald_test': wt})


def fit(key: str) -> typing.Any:
    model, kwargs = CASES[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = metafopy.robust(MODELS[model](), 'district', club_sandwich=True, **kwargs)
    fit.warnings = [str(w.message) for w in caught]  # type: ignore[attr-defined]
    return res


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture helpers
def vector(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = 1e-10) -> None:
    refutil.close(numpy.asarray(actual, dtype=float).reshape(-1), numpy.asarray(expected, dtype=float).reshape(-1).tolist(), what, rtol=rtol, atol=atol)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture helpers
def finite(values: typing.Any) -> numpy.typing.NDArray[numpy.float64]:
    """R's Inf is serialised as null (NaN) in the reference file."""
    a = numpy.asarray(values, dtype=float)
    return numpy.where(numpy.isinf(a), numpy.nan, a)


@pytest.mark.parametrize('key', sorted(CASES))
def test_numbers(key: str) -> None:
    ref = REF[key]
    res = fit(key)
    # auditme: accept[literal-duplication] the fixture names repeat as the R script that generates them repeats them
    for attr, name in (('se', 'se'), ('zval', 'zval'), ('pval', 'pval'), ('ci_lb', 'ci_lb'), ('ci_ub', 'ci_ub')):
        vector(getattr(res, attr), ref[name], f'{key} {name}')
    vector(res.vb, numpy.array(ref['vb']).reshape(-1), f'{key} vb')
    vector(res.QM, ref['QM'], f'{key} QM')
    vector(res.QMp, ref['QMp'], f'{key} QMp')
    vector(finite([float(v) for v in res.QMdf]), ref['QMdf'], f'{key} QMdf')
    assert res.robust is not None
    assert res.robust.coef_test == ref['coef_test']
    assert res.robust.conf_test == ref['conf_test']
    assert res.robust.wald_test == ref['wald_test']
    assert res.robust.vbest == ref['vbest']
    assert sorted(set(fit.warnings)) == sorted(set(refutil.as_list(ref['warnings'])))  # type: ignore[attr-defined]


@pytest.mark.parametrize('key', [k for k in sorted(CASES) if 'saddle' in k and 'conf_z' not in k])
def test_saddlepoint_has_no_degrees_of_freedom(key: str) -> None:
    res = fit(key)
    assert res.robust is not None
    assert numpy.isnan(res.robust.dfs).all()


@pytest.mark.parametrize('key', ['uni_two_saddle', 'mv_two_saddle', 'uni_int_wald_HTA', 'mv_two_wald_EDT', 'uni_three_wald_HTB'])
def test_print(key: str) -> None:
    assert refutil.r_print(fit(key).summary(digits=4, showfit=False)).strip() == refutil.r_print(REF[key]['print']).strip()


def test_explicit_saddlepoint_conf_test_is_rejected() -> None:
    assert 'error' in REF['uni_bad_conf']
    with pytest.raises(metafopy.errors.InputError, match='conf_test'):
        metafopy.robust(MODELS['uni_mods'](), 'district', club_sandwich=True, coef_test='saddlepoint', conf_test='saddlepoint')


def test_unknown_wald_test_is_rejected() -> None:
    with pytest.raises(metafopy.errors.InputError, match='wald_test'):
        metafopy.robust(MODELS['uni_two'](), 'district', club_sandwich=True, wald_test='HTC')


def test_uniroot_matches_r() -> None:
    # R: uniroot(function(x) x^3 - 2, c(0, 3))$root
    root = metafopy.core.clubsandwich.uniroot(lambda x: x**3 - 2, 0.0, 3.0)
    assert root == pytest.approx(2 ** (1 / 3), abs=1e-4)
    with pytest.raises(metafopy.errors.EstimationError, match='opposite sign'):
        metafopy.core.clubsandwich.uniroot(lambda x: x**2 + 1, 0.0, 3.0)


def test_saddlepoint_zero_statistic() -> None:
    s, p = metafopy.core.clubsandwich.saddlepoint_pval(0.0, numpy.eye(3))
    assert numpy.isnan(s)
    assert p == 1.0
