"""Experimental ``rma_glmm`` measures (RR, RD, PR, PLN) and ``predict`` for ``rma_glmm`` fits checked against R metafor 5.2-1.

The fixtures come from ``r/generate_glmm_extras.R``. The fixed-effects fits and the saturated models are
glm() fits and agree to machine precision. The mixed models use lme4's ``glmer()`` with a log or identity
link, whose optimiser often stops early (R then only warns) so those estimates agree to the tolerances in
``TOLERANCE``. Where lme4 fails outright (``rd_rs_ml``, ``rd_collins``) or where the roundoff of the
saturated glm decides whether R can fit it (``rd_collins_fe``) the differences are asserted explicitly.
"""

import functools
import json
import math
import pathlib
import re
import typing
import warnings

import glmm_cases
import glmm_extras_cases
import numpy
import pytest
import refutil
import test_rma_glmm

import metafopy
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase6' / 'glmm_extras.json').read_text(encoding='utf-8'))

EXACT = 1e-6
TIGHT = 2e-3
LOOSE = 5e-2

# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
TOLERANCE: dict[str, float] = {
    'rr_fs_ml': TIGHT,
    'rr_fs_mod': TIGHT,
    'rr_fs_t': TIGHT,
    'rr_fs_mod_na': TIGHT,
    'rr_rs_ml': 2e-2,
    'rr_rs_fe': 1e-4,
    'rr_collins': 1e-1,
    'rr_nw': TIGHT,
    'rd_fs_ml': TIGHT,
    'rd_fs_mod': TIGHT,
    'rd_rs_fe': TIGHT,
    'pr_ml': TIGHT,
    'pr_mod': TIGHT,
    'pln_ml': TIGHT,
    'pln_mod': TIGHT,
    'pln_t': TIGHT,
    'pln_model_arg': TIGHT,
}
# R's glmer stops with a convergence warning, so its estimates are not at the optimum that metafopy finds
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
LME4_FAILS = ('rd_rs_ml', 'rd_collins')
# the roundoff of the saturated model decides whether glm() can start (R cannot), so only the main model is compared
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
SATURATED_ROUNDOFF = ('rd_collins_fe',)
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
FAILS_IN_BOTH = ('rd_nw', 'rd_nw_fe')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
LME4_NOISE = ('Model failed to converge', 'Model is nearly unidentifiable', 'Cannot fit saturated model')
FITS = [n for n in glmm_extras_cases.CASES if n not in LME4_FAILS and n not in FAILS_IN_BOTH]
EXPERIMENTAL = "The use of this 'measure' is experimental - treat results with caution."


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """``test_rma_glmm.close`` where undefined values (NaN, infinite) must match R's NA."""
    act = numpy.asarray(actual, dtype=float)
    test_rma_glmm.close(numpy.where(numpy.isfinite(act), act, numpy.nan), expected, what, rtol)


@functools.cache
def fit_with_warnings(name: str) -> tuple[metafopy.GLMMResult, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = glmm_extras_cases.CASES[name]()
    return res, tuple(str(w.message).split('\n')[0] for w in caught)


@pytest.mark.parametrize('name', FITS)
def test_fit(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    tol = TOLERANCE.get(name, EXACT)
    close(res.beta, ref['beta'], 'beta', tol)
    close(res.se, ref['se'], 'se', tol)
    close(res.vb.ravel('F'), ref['vb'], 'vb', 2 * tol)
    close(res.zval, ref['zval'], 'zval', tol)
    close(res.ci_lb, ref['ci_lb'], 'ci_lb', tol)
    close(res.ci_ub, ref['ci_ub'], 'ci_ub', tol)
    close(res.tau2, ref['tau2'], 'tau2', tol)
    close(res.sigma2, ref['sigma2'], 'sigma2', tol)
    close(res.I2, ref['I2'], 'I2', tol)
    close(res.H2, ref['H2'], 'H2', tol)
    close(res.QM, ref['QM'], 'QM', 4 * tol)
    assert (res.k, res.k_f, res.k_yi, res.k_all, res.p, res.parms) == (ref['k'], ref['k_f'], ref['k_yi'], ref['k_all'], ref['p'], ref['parms'])
    assert (res.method, res.model) == (ref['method'], ref['model'])
    close(res.yi, ref['yi'], 'yi', EXACT)
    close(res.vi, ref['vi'], 'vi', EXACT)
    close(res.X.ravel('F'), ref['X'], 'X', EXACT)
    if name in SATURATED_ROUNDOFF:
        return
    close(res.QE_Wld, ref['QE_Wld'], 'QE_Wld', tol)
    close(res.QE_LRT, ref['QE_LRT'], 'QE_LRT', tol)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', tol)
    assert res.QE_df == ref['QE_df']


@pytest.mark.parametrize('name', FITS)
def test_warnings(name: str) -> None:
    """metafopy raises R's warnings, except those from lme4's optimiser and the roundoff-dependent saturated model."""
    _, issued = fit_with_warnings(name)
    expected = [w.split('\n')[0] for w in refutil.as_list(REF[name]['warnings']) if not w.startswith(LME4_NOISE)]
    assert sorted(set(expected)) == sorted({w for w in issued if not w.startswith(('step size truncated', *LME4_NOISE))})
    assert issued.count(EXPERIMENTAL) == 1


@pytest.mark.parametrize('name', [n for n in FITS if n not in TOLERANCE and n not in SATURATED_ROUNDOFF])
def test_print_exact(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    assert refutil.same_print(str(res), ref['print']), f'\n{res}\n{ref["print"]}'
    assert refutil.same_print(res.summary(showfit=True), ref['print_showfit']), f'\n{res.summary(showfit=True)}\n{ref["print_showfit"]}'


@pytest.mark.parametrize('name', [n for n in FITS if n in TOLERANCE])
def test_print_layout(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    assert test_rma_glmm.mask_numbers(str(res)) == test_rma_glmm.mask_numbers(ref['print'])


@pytest.mark.parametrize('name', FAILS_IN_BOTH)
def test_glm_fails_like_r(name: str) -> None:
    """The log/identity-link ``glm`` cannot start (no valid fitted means), so R and metafopy both stop."""
    assert REF[name]['error'].startswith('Cannot fit FE model')
    with pytest.raises(metafopy.errors.EstimationError, match='Cannot fit FE model'):
        fit_with_warnings(name)


@pytest.mark.parametrize('name', LME4_FAILS)
def test_lme4_failures(name: str) -> None:
    """R stops with ``Cannot fit ML model`` (lme4 does not converge); the maximum-likelihood fit exists and is returned."""
    assert REF[name]['error'].startswith('Cannot fit ML model')
    res, issued = fit_with_warnings(name)
    assert numpy.all(numpy.isfinite(res.beta))
    assert math.isfinite(res.tau2)
    assert res.tau2 > 0
    assert EXPERIMENTAL in issued
    # the fixed-effects estimates are those of the closely related fit that R does produce
    related = REF['rd_fs_ml' if name == 'rd_rs_ml' else 'rd_collins_fe']
    close(res.beta, related['beta'], 'beta', 0.5)


def test_errors() -> None:
    options: tuple[tuple[str, str], ...] = (('rr_cm_el', 'RR'), ('rr_cm_al', 'RR'), ('rd_cm_el', 'RD'), ('rd_cm_al', 'RD'))
    for key, measure in options:
        model = 'CM.EL' if key.endswith('el') else 'CM.AL'
        with pytest.raises(metafopy.errors.InputError, match=re.escape(REF['errors'][key])), pytest.warns(UserWarning, match='experimental'):
            metafopy.rma_glmm(**{**glmm_cases.BCG, 'measure': measure}, data=glmm_cases.bcg(), model=model)


def test_model_argument_warning() -> None:
    _, issued = fit_with_warnings('pln_model_arg')
    assert "Argument 'model' not relevant for this outcome measure." in issued
    assert issued.count(EXPERIMENTAL) == 1


def test_experimental_measures_need_their_data() -> None:
    with pytest.raises(metafopy.errors.InputError), pytest.warns(UserWarning, match='experimental'):
        metafopy.rma_glmm(xi='tpos', mi='tneg', measure='RR', data=glmm_cases.bcg())
    with pytest.raises(metafopy.errors.InputError), pytest.warns(UserWarning, match='experimental'):
        metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', measure='PLN', data=glmm_cases.bcg())


# -- predict, ranef -------------------------------------------------------------------------------------------------

PREDICT_TOLERANCE = {'al': 1e-4, 'el': 1e-3, 'irln': 1e-4, 'or': 1e-3, 'pln': 1e-3, 'plo': 1e-4, 'plo_mod': 1e-4, 'rr_mod': 1e-3, 'rs': 2e-2, 'or_mod_newmods_level': 1e-5}
PREDICT_EXACT = 1e-6


@pytest.mark.parametrize('name', list(glmm_extras_cases.PREDICT))
def test_predict(name: str) -> None:
    ref = REF['predict'][name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = glmm_extras_cases.PREDICT[name]()
    base = name.split('_')[0]
    tol = PREDICT_TOLERANCE.get(name, PREDICT_TOLERANCE.get(base, PREDICT_EXACT) if name.startswith(('or_transf', 'or_level', 'or_targs')) else PREDICT_EXACT)
    # auditme: accept[literal-duplication] the result fields repeat as metafor's predict() returns them
    for key in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        expected = ref.get(key)
        actual = getattr(res, key)
        if expected is None:
            assert numpy.all(numpy.isnan(actual)), key
        else:
            close(actual, expected, key, tol)
    if res.slab is not None:
        assert list(res.slab) == refutil.as_list(ref['slab'])


def test_predict_drops_intervals_like_r() -> None:
    assert REF['predict']['or_fe'].get('pi_lb') is None
    assert REF['predict']['or_transf_ilogit_int'].get('pi_lb') is None
    res = glmm_extras_cases.PREDICT['or_fe']()
    assert numpy.all(numpy.isnan(res.pi_lb))
    assert numpy.all(numpy.isnan(res.pi_ub))


def test_predict_rejects_wrong_newmods() -> None:
    with pytest.raises(metafopy.errors.InputError, match='newmods'):
        metafopy.predict(glmm_extras_cases.model('or_mod2'), newmods=[[1.0, 2.0, 3.0, 4.0]])


def test_ranef_and_blup_do_not_exist_for_glmm() -> None:
    """metafor has no ``ranef`` or ``blup`` method for ``rma.glmm`` objects (R: "no applicable method")."""
    res = glmm_extras_cases.model('or')
    with pytest.raises(metafopy.errors.NotSupportedError, match='no applicable method'):
        metafopy.ranef(res)
    with pytest.raises(metafopy.errors.NotSupportedError, match='no applicable method'):
        metafopy.blup(res)
