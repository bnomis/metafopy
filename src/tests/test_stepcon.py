"""selmodel(type='stepcon') (non-increasing step function weights, fitted with a constrained optimiser) against R metafor 5.2-1.

Fixtures come from tests/r/generate_stepcon.R, where metafor uses Rsolnp::solnp; here the constrained fit uses SLSQP, so estimates agree to the
accuracy of the two optimisers (about 1e-5), not to machine precision.
"""

import functools
import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil
import test_selmodel

import metafopy
import metafopy.errors
import metafopy.models.sel_opt

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase6' / 'stepcon.json').read_text(encoding='utf-8'))
DATA = {'bang': 'bang_data', 'hack': 'hack_data', 'rau': 'rau_data'}

# estimates are optimiser output; the numerical-Hessian quantities inherit the optimum's error
RTOL = 5e-5
RTOL_SE = 5e-4
ATOL = 1e-6
# the fit of rau_stepcon_mods has a flat likelihood in delta (tau^2 = 0), identified to about 1e-4
LOOSE = {'rau_stepcon_mods': 5e-4}

# For these fits the heterogeneity test of metafor is based on a worse fit than ours: it refits the model with tau^2 = 0 with solnp but without the
# inequality constraint (the call is built without ineqfun), and solnp stops at a point whose log-likelihood is up to 0.9 below the constrained optimum
# (that of the equal-effects step model for hack_stepcon_b). We return the optimum, so LRT.tau2 is smaller than R's.
R_SUBOPTIMAL_TAU2_FIT = {
    'bang_stepcon_mods',
    'hack_stepcon_b',
    'hack_stepcon_c',
    'hack_stepcon_level90',
    'hack_stepcon_prec',
    'hack_stepcon_reml',
    'hack_stepcon_subset',
    'hack_stepcon_two',
}

CASES = sorted(k for k, v in REF['fits'].items() if 'error' not in v)
STEPCON = [c for c in CASES if 'stepcon' in c]


def base_model(case: str) -> metafopy.RMAResult:
    spec = REF['specs'][case]
    kwargs: dict[str, typing.Any] = {'ni': 'ni'} if spec['data'] == 'bang' else {}
    return typing.cast(metafopy.RMAResult, metafopy.rma(data=pandas.DataFrame(REF[DATA[spec['data']]]), **kwargs, **spec['rma']))


def sel_kwargs(case: str) -> dict[str, typing.Any]:
    kwargs = {k: test_selmodel.arg(v) for k, v in REF['specs'][case]['sel'].items()}
    if 'subset' in kwargs:
        kwargs['subset'] = numpy.array(kwargs['subset'], dtype=bool)
    return kwargs


@functools.cache
def fitted(case: str) -> metafopy.SelmodelResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return typing.cast(metafopy.SelmodelResult, metafopy.selmodel(base_model(case), **sel_kwargs(case)))


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    refutil.close(actual, expected, what, rtol=rtol, atol=atol)


@pytest.mark.parametrize('case', CASES)
def test_estimates(case: str) -> None:
    ref = REF['fits'][case]
    res = fitted(case)
    rtol = LOOSE.get(case, RTOL)
    close(res.beta, ref['beta'], f'{case} beta', rtol=rtol)
    close(res.tau2, ref['tau2'], f'{case} tau2', rtol=rtol, atol=1e-6)
    close(res.delta, ref['delta'], f'{case} delta', rtol=rtol)
    close(res.ll, ref['ll'], f'{case} ll', rtol=1e-8, atol=1e-6)
    close(res.ll0, ref['ll0'], f'{case} ll0', rtol=1e-8, atol=1e-6)
    close(res.pvals, ref['pvals'], f'{case} pvals', rtol=1e-12)
    close(res.delta_min, ref['delta_min'], f'{case} delta_min')
    close(res.delta_max, ref['delta_max'], f'{case} delta_max')
    assert (res.k, res.k0, res.k1, res.p, res.deltas) == (ref['k'], ref['k0'], ref['k1'], ref['p'], ref['deltas'])
    assert list(res.delta_fix) == refutil.as_list(ref['delta_fix'])
    assert (res.type, res.alternative, res.parms) == (ref['type'], ref['alternative'], ref['parms'])
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{case} fit stats', rtol=RTOL_SE)
    close(res.steps, ref['steps'], f'{case} steps', rtol=1e-12)


@pytest.mark.parametrize('case', STEPCON)
def test_delta_is_non_increasing_and_first_is_one(case: str) -> None:
    res = fitted(case)
    assert res.delta[0] == 1
    assert numpy.all(numpy.diff(res.delta) <= 1e-7)


@pytest.mark.parametrize('case', CASES)
def test_inference(case: str) -> None:
    ref = REF['fits'][case]
    res = fitted(case)
    rtol = LOOSE.get(case, RTOL_SE)
    for name in (*test_selmodel.WALD, 'se_tau2', 'QM', 'QMp', 'LRT', 'LRTp'):
        close(getattr(res, name), ref[name], f'{case} {name}', rtol=rtol)
    if case not in R_SUBOPTIMAL_TAU2_FIT:
        for name in ('LRT_tau2', 'LRTp_tau2'):
            close(getattr(res, name), ref[name], f'{case} {name}', rtol=rtol)
    for name in test_selmodel.WALD:
        close(getattr(res, f'{name}_delta'), ref[f'{name}_delta'], f'{case} {name}_delta', rtol=rtol)
    numpy.testing.assert_allclose(res.vb, test_selmodel.matrix(ref['vb']), rtol=rtol, atol=ATOL, equal_nan=True, err_msg=f'{case} vb')
    numpy.testing.assert_allclose(res.vd, test_selmodel.matrix(ref['vd']), rtol=rtol, atol=ATOL, equal_nan=True, err_msg=f'{case} vd')
    assert res.LRTdf == ref['LRTdf']


@pytest.mark.parametrize('case', sorted(R_SUBOPTIMAL_TAU2_FIT))
def test_heterogeneity_test_is_at_least_as_good_as_r(case: str) -> None:
    res = fitted(case)
    ref = REF['fits'][case]
    assert 0 <= res.LRT_tau2 <= ref['LRT_tau2'] + 1e-6
    assert res.LRTp_tau2 >= ref['LRTp_tau2'] - 1e-6


def test_heterogeneity_test_agrees_with_equal_effects_fit() -> None:
    # with tau^2 = 0 the model is the equal-effects step model, which metafor also fits (with the constraint)
    res = fitted('hack_stepcon_b')
    assert res.LRT_tau2 == pytest.approx(-2 * (fitted('hack_stepcon_ee').ll - res.ll), rel=1e-6)


def without_heterogeneity_test(text: str) -> str:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Test for') and 'Heterogeneity' in line:
            del lines[i + 1]
            break
    return '\n'.join(lines)


@pytest.mark.parametrize('case', CASES)
def test_print(case: str) -> None:
    ref = REF['fits'][case]
    actual, expected = str(fitted(case)), ref['print']
    if case in R_SUBOPTIMAL_TAU2_FIT:
        actual, expected = without_heterogeneity_test(actual), without_heterogeneity_test(expected)
    assert test_selmodel.same_text(actual, expected, LOOSE.get(case, RTOL_SE)), f'{case}\n{fitted(case)}\n---\n{ref["print"]}'


@pytest.mark.parametrize('case', CASES)
def test_ptable(case: str) -> None:
    ref = REF['fits'][case]['ptable']
    res = fitted(case)
    assert res.ptable is not None
    assert [str(n) for n in res.ptable.index] == ref['.rownames']
    assert res.ptable['k'].tolist() == ref['k']


@pytest.mark.parametrize('case', CASES)
def test_warnings(case: str) -> None:
    # solnp also prints its own diagnostics ("solnp-->warning: ..."), which are not R warnings
    expected = [w for w in refutil.as_list(REF['fits'][case]['warnings']) if not w.strip().startswith('solnp')]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        metafopy.selmodel(base_model(case), **sel_kwargs(case))
    messages = {str(w.message).strip() for w in caught}
    for msg in expected:
        assert msg.strip() in messages, f'{case}: missing {msg!r} in {messages}'


def test_ptable_only() -> None:
    table = metafopy.selmodel(base_model('hack_stepcon_a'), 'stepcon', steps=[0.025, 0.1, 0.5], ptable=True)
    assert isinstance(table, pandas.DataFrame)
    assert table['k'].tolist() == REF['ptable']['k']
    assert [str(n) for n in table.index] == REF['ptable']['.rownames']


def test_accessors_and_predict() -> None:
    ref = REF['accessors']
    res = fitted('hack_stepcon_a')
    close(res.coef('delta'), ref['coef_delta'], 'coef delta')
    numpy.testing.assert_allclose(res.vcov('delta').to_numpy(), test_selmodel.matrix(ref['vcov_delta']), rtol=RTOL_SE, atol=ATOL, equal_nan=True)
    close(res.logLik(), ref['logLik'], 'logLik', rtol=1e-8)
    close(res.AIC(), ref['AIC'], 'AIC', rtol=1e-8)
    close(res.BIC(), ref['BIC'], 'BIC', rtol=1e-8)
    assert repr(res).startswith('<SelmodelResult stepcon')
    for name, pred in (
        ('hack_stepcon_a', metafopy.predict(fitted('hack_stepcon_a'))),
        ('bang_stepcon_mods', metafopy.predict(fitted('bang_stepcon_mods'), newmods=numpy.array([[2, 0], [3, 1]]))),
    ):
        # auditme: accept[literal-duplication] the fields of metafor's predict() result
        for field in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
            close(getattr(pred, field), REF['predict'][name][field], f'{name} {field}', rtol=RTOL_SE)


def test_confint_and_profile_not_implemented() -> None:
    # metafor: "Method not currently implemented for this type of model."
    assert REF['confint_error'] == REF['profile_error'] == 'Method not currently implemented for this type of model.'
    res = fitted('hack_stepcon_a')
    with pytest.raises(metafopy.errors.NotSupportedError, match='not currently implemented'):
        metafopy.confint(res)
    with pytest.raises(metafopy.errors.NotSupportedError, match='not currently implemented'):
        metafopy.profile(res, tau2=True)


def test_stepcon_requires_steps_and_matches_stepfun_when_unconstrained() -> None:
    with pytest.raises(metafopy.errors.InputError, match="Must specify the 'steps'"):
        metafopy.selmodel(base_model('hack_stepcon_a'), 'stepcon')
    # the unconstrained step function fits of these data are already non-increasing, so the constraint is not active
    for stepcon, stepfun in (('hack_stepcon_a', 'hack_stepfun_a'), ('hack_stepcon_c', 'hack_stepfun_c'), ('bang_stepcon_int', 'bang_stepfun_int')):
        numpy.testing.assert_allclose(fitted(stepcon).delta, fitted(stepfun).delta, rtol=1e-4, atol=1e-6)
        assert fitted(stepcon).ll == pytest.approx(fitted(stepfun).ll, rel=1e-8)


def test_default_optimizer_is_the_constrained_one() -> None:
    with pytest.warns(UserWarning, match='stepcon'):
        choice = metafopy.models.sel_opt.choose_optimizer('nlminb', 'BFGS', 'stepcon')
    assert choice.optimizer == 'solnp'
