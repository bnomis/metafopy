"""rma_glmm (UM.FS, UM.RS, CM.AL, CM.EL, PLO, IRLN) checked against R metafor 5.2-1 (fixtures from ``r/generate_glmm.R``).

The generalized linear models behind the fixed-effects and the incidence rate ratio fits are reproduced to
machine precision. The mixed models are fitted by R with ``lme4::glmer()``, which stops at a loose
optimiser tolerance and takes its standard errors from a finite-difference Hessian, and the exact conditional
likelihood is evaluated by R with ``BiasedUrn`` to a precision of 1e-10 that ``numDeriv`` amplifies, so those
results agree to the tolerances in ``TOLERANCE`` (fitted at the optimum, metafopy's objective is the same or
better, see ``test_collins_objective``).
"""

import json
import math
import pathlib
import re
import typing
import warnings

import glmm_cases
import numpy
import pytest
import refutil

import metafopy
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase3' / 'glmm.json').read_text(encoding='utf-8'))

EXACT = 1e-6
OPTIM = 5e-4
LOOSE = 5e-2
VERY_LOOSE = 1.5e-1

# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
TOLERANCE: dict[str, float] = {
    'fs_ml': OPTIM,
    'fs_ml_coding1': OPTIM,
    'fs_ml_t': OPTIM,
    'fs_collins': VERY_LOOSE,
    'fs_nw': 5e-3,
    'fs_subset': OPTIM,
    'rs_ml': LOOSE,
    'rs_ml_cor': LOOSE,
    'rs_fe': OPTIM,
    'rs_ml_mod': LOOSE,
    'rs_collins': LOOSE,
    'al_ml': OPTIM,
    'al_ml_mod': OPTIM,
    'al_collins': OPTIM,
    'al_nw': OPTIM,
    'el_ml': LOOSE,
    'el_fe': LOOSE,
    'el_fe_mod': LOOSE,
    'el_collins': LOOSE,
    'el_nw': LOOSE,
    'irr_rs': LOOSE,
    'plo': OPTIM,
    'plo_mod': OPTIM,
    'irln': OPTIM,
    'irln_mod': OPTIM,
}
# the Wald test for heterogeneity of the exact conditional model is dominated by finite-difference noise in R
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
NOISY_QE_WLD = frozenset({'el_ml', 'el_fe', 'el_fe_mod', 'el_collins', 'el_nw'})
# the exact-likelihood standard errors (and with them Wald-type statistics) of R carry finite-difference noise
SCALAR_FIELDS = (
    ('tau2', 'tau2'),
    ('sigma2', 'sigma2'),
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    ('I2', 'I2'),
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    ('H2', 'H2'),
    ('QE_Wld', 'QE_Wld'),
    ('QE_LRT', 'QE_LRT'),
    ('QM', 'QM'),
    ('se_tau2', 'se_tau2'),
)


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """Relative differences scaled by ``max(|expected|, 1e-3)``; NaN must match NaN."""
    exp = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in refutil.as_list(expected)], dtype=float))
    act = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    assert act.shape == exp.shape, what
    assert numpy.array_equal(numpy.isnan(act), numpy.isnan(exp)), f'{what}: {act} vs {exp}'
    mask = ~numpy.isnan(exp)
    scale = numpy.maximum(numpy.abs(exp[mask]), 1e-3)
    worst = float(numpy.max(numpy.abs(act[mask] - exp[mask]) / scale, initial=0.0))
    assert worst <= rtol, f'{what}: relative difference {worst:.2e} > {rtol:.0e}\n{act}\n{exp}'


def mask_numbers(text: str) -> str:
    return re.sub(r' +', ' ', re.sub(r'-?\d+\.\d+|<\.\d+|\d+', '#', refutil.r_print(text)))


def fit(name: str) -> metafopy.GLMMResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return glmm_cases.CASES[name]()


@pytest.mark.parametrize('name', list(glmm_cases.CASES))
def test_fit(name: str) -> None:
    ref = REF[name]
    res = fit(name)
    tol = TOLERANCE.get(name, EXACT)
    close(res.beta, ref['beta'], 'beta', tol)
    close(res.se, ref['se'], 'se', tol)
    close(res.vb.ravel('F'), ref['vb'], 'vb', 2 * tol)
    close(res.zval, ref['zval'], 'zval', tol)
    close(res.ci_lb, ref['ci_lb'], 'ci_lb', tol)
    close(res.ci_ub, ref['ci_ub'], 'ci_ub', tol)
    for key, attr in SCALAR_FIELDS:
        if key == 'QE_Wld' and name in NOISY_QE_WLD:
            assert math.isfinite(res.QE_Wld)
            assert res.QE_Wld > 0
            continue
        close(getattr(res, attr), ref[key], key, tol)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', tol)
    assert (res.k, res.k_f, res.k_yi, res.k_all, res.p, res.parms) == (ref['k'], ref['k_f'], ref['k_yi'], ref['k_all'], ref['p'], ref['parms'])
    assert (res.method, res.model, res.QE_df) == (ref['method'], ref['model'], ref['QE_df'])
    close(res.yi, ref['yi'], 'yi', EXACT)
    close(res.vi, ref['vi'], 'vi', EXACT)
    close(res.X.ravel('F'), ref['X'], 'X', EXACT)


@pytest.mark.parametrize('name', [n for n in glmm_cases.CASES if TOLERANCE.get(n, EXACT) == EXACT])
def test_print_exact(name: str) -> None:
    ref = REF[name]
    res = fit(name)
    assert refutil.same_print(str(res), ref['print']), f'\n{res}\n{ref["print"]}'
    assert refutil.same_print(res.summary(showfit=True), ref['print_showfit']), f'\n{res.summary(showfit=True)}\n{ref["print_showfit"]}'


@pytest.mark.parametrize('name', [n for n in glmm_cases.CASES if TOLERANCE.get(n, EXACT) != EXACT])
def test_print_layout(name: str) -> None:
    """Where the numbers carry optimiser noise, the printed layout (titles, labels, rows) must still be R's."""
    ref = REF[name]
    res = fit(name)
    assert mask_numbers(str(res)) == mask_numbers(ref['print'])
    assert mask_numbers(res.summary(showfit=True)) == mask_numbers(ref['print_showfit'])


@pytest.mark.parametrize('name', ['fs_ml', 'al_ml', 'plo'])
def test_accessors(name: str) -> None:
    res = fit(name)
    ref = REF[name]
    assert list(res.coef().index) == list(res.coef_names)
    close(res.coef().to_numpy(), ref['beta'], 'coef', OPTIM)
    assert res.vcov().shape == (res.p, res.p)
    stats = res.fitstats()
    assert list(stats.index) == ['logLik', 'deviance', 'AIC', 'BIC', 'AICc']
    close(stats.to_numpy(), ref['fit_ml'], 'fitstats', OPTIM)
    assert res.logLik() == stats['logLik']
    assert res.AIC() == stats['AIC']
    assert res.BIC() == stats['BIC']
    assert res.AICc() == stats['AICc']
    assert res.deviance() == stats['deviance']


def test_errors() -> None:
    bcg = glmm_cases.bcg()
    cells: dict[str, typing.Any] = dict(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    errors = REF['errors']
    with pytest.raises(metafopy.errors.InputError, match=re.escape(errors['no_measure'])):
        metafopy.rma_glmm(**cells)
    options: tuple[tuple[str, dict[str, typing.Any]], ...] = (
        ('bad_model', {'model': 'XX'}),
        ('bad_method', {'method': 'REML'}),
        ('bad_coding', {'coding': 2}),
        ('bad_measure', {'measure': 'SMD'}),
    )
    for key, kwargs in options:
        with pytest.raises(metafopy.errors.InputError, match=re.escape(errors[key])):
            metafopy.rma_glmm(**{**cells, 'measure': 'OR', **kwargs})


def test_um_rs_warns_about_nagq() -> None:
    with pytest.warns(UserWarning, match='nAGQ'):
        glmm_cases.CASES['rs_ml']()
