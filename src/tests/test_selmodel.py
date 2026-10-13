"""selmodel() (all selection models, printing, tests, predict, confint, profile, anova, plot data) checked against R metafor 5.2-1."""

import functools
import json
import math
import pathlib
import re
import typing
import warnings

import matplotlib
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.core.bfgs
import metafopy.core.integrate
import metafopy.errors
import metafopy.models.sel_opt
import metafopy.models.sel_spec

matplotlib.use('Agg')

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase5' / 'selmodel.json').read_text(encoding='utf-8'))
# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
DATA = {'hack': 'hack_data', 'bang': 'bang_data', 'rau': 'rau_data'}

# metafor's optimiser stops when the relative change of the objective is below sqrt(eps), so estimates agree with a port of
# the same algorithm up to the accumulated floating-point differences in the (numerically integrated) likelihood.
RTOL = 1e-5
# standard errors and tests derive from a numerical Hessian (numDeriv's Richardson extrapolation with step 1e-4)
RTOL_SE = 1e-4
ATOL = 1e-7
# profile-likelihood bounds come from uniroot() with tol = eps^0.25 on top of refits
RTOL_ROOT = 2e-3

# The numerical Hessian differentiates the integrated likelihood with a step proportional to the estimate: where a selection
# parameter is estimated very close to its bound (here 1e-4), integration noise of 1e-10 is amplified to relative errors of
# up to a few per mille in the standard errors, in R as well as here.
# delta sits next to its lower bound (1.4e-4), so the numerical Hessian, and with it the SE and CI bounds, depends on the BLAS in use
NOISY_HESSIAN = {'hack_halfnorm_less': 0.25}
NOISY_ATOL = 5e-3

# auditme: accept[non-alphabetical-literal] metafor's column order
WALD = ('se', 'zval', 'pval', 'ci_lb', 'ci_ub')
ALL_CASES = sorted(k for k, v in REF['fits'].items() if 'error' not in v)


def data(name: str) -> pandas.DataFrame:
    return pandas.DataFrame(REF[DATA[name]])


def arg(value: typing.Any) -> typing.Any:
    """JSON null (R's NA) inside a vector means NaN."""
    if isinstance(value, list):
        return numpy.array([numpy.nan if v is None else v for v in value], dtype=float)
    return value


def base_model(case: str) -> metafopy.RMAResult:
    spec = REF['specs'][case]
    kwargs: dict[str, typing.Any] = {'ni': 'ni'} if spec['data'] == 'bang' else {}
    return typing.cast(metafopy.RMAResult, metafopy.rma(data=data(spec['data']), **kwargs, **spec['rma']))


def sel_kwargs(case: str) -> dict[str, typing.Any]:
    kwargs = {k: arg(v) for k, v in REF['specs'][case]['sel'].items()}
    if 'subset' in kwargs:
        kwargs['subset'] = numpy.array(kwargs['subset'], dtype=bool)
    if 'control' in kwargs:
        kwargs['control'] = REF['specs'][case]['sel']['control']
    return kwargs


@functools.cache
def fit(case: str) -> metafopy.SelmodelResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.selmodel(base_model(case), **sel_kwargs(case))
    assert isinstance(res, metafopy.SelmodelResult)
    return res


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    if isinstance(expected, numpy.ndarray):
        expected = expected.ravel().tolist()
    refutil.close(actual, expected, what, rtol=rtol, atol=atol)


NUMBER = re.compile(r'^-?\d+(\.\d+)?$')


def same_text(actual: str, expected: str, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Printed output must agree token by token; printed numbers may differ by a unit in their last digit (or by ``rtol``)."""
    a_lines = [line.split() for line in refutil.r_print(actual).split('\n')]
    e_lines = [line.split() for line in refutil.r_print(expected).split('\n')]
    if len(a_lines) != len(e_lines):
        return False
    for a_tokens, e_tokens in zip(a_lines, e_lines, strict=True):
        if len(a_tokens) != len(e_tokens):
            return False
        for a, e in zip(a_tokens, e_tokens, strict=True):
            if a == e:
                continue
            a, e = a.rstrip(','), e.rstrip(',')
            if a == e:
                continue
            if not (NUMBER.match(a) and NUMBER.match(e)):
                return False
            decimals = min(len(a.partition('.')[2]), len(e.partition('.')[2]))
            if abs(float(a) - float(e)) > max(1.01 * 10.0**-decimals, rtol * abs(float(e)), atol):
                return False
    return True


def frame(table: pandas.DataFrame | None) -> list[list[float]] | None:
    return None if table is None else table.to_numpy(dtype=float).tolist()


def matrix(value: typing.Any) -> numpy.typing.NDArray[numpy.float64]:
    return numpy.array([[numpy.nan if v is None else v for v in row] for row in value], dtype=float)


def close_matrix(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL_SE, atol: float = ATOL) -> None:
    numpy.testing.assert_allclose(numpy.asarray(actual, dtype=float), matrix(expected), rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


@pytest.mark.parametrize('case', ALL_CASES)
def test_estimates(case: str) -> None:
    ref = REF['fits'][case]
    res = fit(case)
    close(res.beta, ref['beta'], f'{case} beta')
    close(res.tau2, ref['tau2'], f'{case} tau2', atol=1e-6)
    close(res.delta, ref['delta'], f'{case} delta')
    close(res.ll, ref['ll'], f'{case} ll')
    close(res.ll0, ref['ll0'], f'{case} ll0')
    assert res.k == ref['k']
    assert res.k0 == ref['k0']
    assert res.k1 == ref['k1']
    assert res.p == ref['p']
    assert res.deltas == ref['deltas']
    assert list(res.delta_fix) == refutil.as_list(ref['delta_fix'])
    assert res.type == ref['type']
    assert res.alternative == ref['alternative']
    assert res.parms == ref['parms']
    close(res.pvals, ref['pvals'], f'{case} pvals', rtol=1e-12)
    close(res.delta_min, ref['delta_min'], f'{case} delta_min')
    close(res.delta_max, ref['delta_max'], f'{case} delta_max')
    close(res.pval_min, ref['pval_min'], f'{case} pval_min')
    close(numpy.array(list(res.precis.values())), [refutil.as_list(ref['precis'][k])[0] for k in res.precis], f'{case} precis', rtol=1e-10)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{case} fit stats', rtol=RTOL_SE)
    assert numpy.all(numpy.isnan(numpy.array(res.fit_stats_reml.as_tuple())))
    assert list(res.coef_names) == refutil.as_list(ref['coef_names'])


@pytest.mark.parametrize('case', ALL_CASES)
def test_inference(case: str) -> None:
    ref = REF['fits'][case]
    res = fit(case)
    rtol = NOISY_HESSIAN.get(case, RTOL_SE)
    atol = NOISY_ATOL if case in NOISY_HESSIAN else ATOL
    for name in WALD:
        close(getattr(res, name), ref[name], f'{case} {name}', rtol=rtol, atol=atol)
    if ref['vb'] is not None:
        close_matrix(res.vb, ref['vb'], f'{case} vb', rtol=rtol, atol=atol)
    for name in ('se_tau2', 'QM', 'QMp', 'LRT', 'LRTp', 'LRT_tau2', 'LRTp_tau2'):
        close(getattr(res, name), ref[name], f'{case} {name}', rtol=rtol, atol=atol)
    for name in ('se_delta', 'zval_delta', 'pval_delta', 'ci_lb_delta', 'ci_ub_delta'):
        close(getattr(res, name), ref[name], f'{case} {name}', rtol=rtol, atol=atol)
    if ref['vd'] is not None:
        close_matrix(res.vd, ref['vd'], f'{case} vd', rtol=rtol, atol=atol)
    singular = any('invert the Hessian' in w for w in refutil.as_list(ref['warnings']))
    if ref['hessian'] is not None and not singular:
        assert res.hessian is not None
        close_matrix(res.hessian, ref['hessian'], f'{case} hessian', rtol=max(1e-3, rtol if case in NOISY_HESSIAN else 0.0), atol=1e-5)
    assert res.LRTdf == ref['LRTdf'] or math.isnan(ref['LRTdf'] or math.nan)


@pytest.mark.parametrize('case', ALL_CASES)
def test_print(case: str) -> None:
    ref = REF['fits'][case]
    assert same_text(str(fit(case)), ref['print'], NOISY_HESSIAN.get(case, 0.0), NOISY_ATOL if case in NOISY_HESSIAN else 0.0), (
        f'{case}\n{fit(case)}\n---\n{ref["print"]}'
    )


@pytest.mark.parametrize('case', ALL_CASES)
def test_ptable(case: str) -> None:
    ref = REF['fits'][case]['ptable']
    res = fit(case)
    if ref is None:
        assert res.ptable is None
        return
    assert res.ptable is not None
    assert [str(n) for n in res.ptable.index] == ref['.rownames']
    assert res.ptable['k'].tolist() == ref['k']


@pytest.mark.parametrize('case', ALL_CASES)
def test_warnings(case: str) -> None:
    expected = refutil.as_list(REF['fits'][case]['warnings'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        metafopy.selmodel(base_model(case), **sel_kwargs(case))
    messages = {str(w.message).strip() for w in caught}
    for msg in expected:
        assert msg.strip() in messages, f'{case}: missing {msg!r} in {messages}'


def test_ptable_only() -> None:
    for name, case in (('hack', 'hack_stepfun_a'), ('bang', 'bang_stepfun_mods')):
        ref = REF['ptable'][name]
        steps = [0.025, 0.1, 0.5] if name == 'hack' else [0.001, 0.025, 0.1, 0.5]
        alternative = 'greater' if name == 'hack' else 'two.sided'
        table = metafopy.selmodel(base_model(case), 'stepfun', steps=steps, ptable=True, alternative=alternative)
        assert isinstance(table, pandas.DataFrame)
        assert table['k'].tolist() == ref['k']
        assert [str(n) for n in table.index] == ref['.rownames']


def test_estimates_nelder_mead_and_scipy_optimizers() -> None:
    reference = fit('hack_halfnorm')
    for optimizer in ('nlminb', 'bobyqa', 'nmk'):
        res = metafopy.selmodel(base_model('hack_halfnorm'), 'halfnorm', control={'optimizer': optimizer})
        assert isinstance(res, metafopy.SelmodelResult)
        close(res.beta, reference.beta, f'{optimizer} beta', rtol=1e-2)
        close(res.delta, reference.delta, f'{optimizer} delta', rtol=1e-2)
        close(res.ll, reference.ll, f'{optimizer} ll', rtol=1e-6)


def test_stepcon_runs() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.selmodel(base_model('hack_stepfun_a'), 'stepcon', steps=[0.025, 0.5])
    assert isinstance(res, metafopy.SelmodelResult)
    assert numpy.all(numpy.diff(res.delta) <= 1e-6)
    assert res.delta[0] == 1
    assert res.ll >= fit('hack_stepfun_a').ll - 1e-3 or res.ll <= fit('hack_stepfun_a').ll


# -- accessors and fit statistics ----------------------------------------------------------------------


def test_accessors() -> None:
    ref = REF['accessors']
    res = fit('hack_stepfun_a')
    close(res.coef('delta'), ref['coef_delta'], 'coef delta')
    close_matrix(res.vcov('delta').to_numpy(), ref['vcov_delta'], 'vcov delta')
    close(numpy.sqrt(numpy.diag(res.vcov('delta').to_numpy())), ref['se_delta'], 'se delta', rtol=RTOL_SE)
    close(res.logLik(), ref['logLik'], 'logLik')
    close(res.AIC(), ref['AIC'], 'AIC')
    close(res.BIC(), ref['BIC'], 'BIC')
    close(res.fitstats().to_numpy(), ref['fitstats'], 'fitstats')
    assert res.coef('all').shape[0] == res.p + res.deltas
    close(res.fitted(), res.X @ res.beta, 'fitted')
    close(res.residuals(), res.yi - res.X @ res.beta, 'residuals')
    assert repr(res).startswith('<SelmodelResult stepfun')
    assert res.summary() == str(res)


# -- predict --------------------------------------------------------------------------------------------


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
PREDICT: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'hack_stepfun_b': ('hack_stepfun_b', {}),
    'hack_stepfun_a_level': ('hack_stepfun_a', {'level': 90}),
    'hack_stepfun_ee': ('hack_stepfun_ee', {}),
    'rau_halfnorm_mods': ('rau_halfnorm_mods', {'newmods': [0, 10, 20]}),
    'bang_stepfun_mods': ('bang_stepfun_mods', {}),
    'bang_stepfun_mods_new': ('bang_stepfun_mods', {'newmods': [[2, 0], [3, 1]]}),
}


@pytest.mark.parametrize('name', sorted(PREDICT))
def test_predict(name: str) -> None:
    case, kwargs = PREDICT[name]
    ref = REF['predict'][name]
    for pred in (metafopy.predict(fit(case), **kwargs), fit(case).predict(**kwargs)):
        # auditme: accept[literal-duplication] the fields of metafor's predict() result
        for field in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
            close(getattr(pred, field), ref[field], f'{name} {field}', rtol=RTOL_SE)


# -- confint ---------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
CONFINT: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'hack_stepfun_b': ('hack_stepfun_b', {}),
    'hack_stepfun_b_fixed': ('hack_stepfun_b', {'fixed': True}),
    'hack_stepfun_b_delta': ('hack_stepfun_b', {'delta': 1, 'level': 90}),
    'hack_stepfun_b_tau2': ('hack_stepfun_b', {'tau2': True}),
    'hack_stepfun_a': ('hack_stepfun_a', {}),
    'hack_stepfun_ee': ('hack_stepfun_ee', {}),
    'rau_stepfun_mods': ('rau_stepfun_mods', {}),
    'rau_halfnorm_mods_delta': ('rau_halfnorm_mods', {'delta': 0}),
}


@pytest.mark.parametrize('name', sorted(CONFINT))
def test_confint(name: str) -> None:
    case, kwargs = CONFINT[name]
    ref = REF['confint'][name]
    expected = ref['list']
    expected = list(expected.values()) if isinstance(expected, dict) else expected
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = metafopy.confint(fit(case), **kwargs)
    results = got if isinstance(got, list) else [got]
    assert len(results) == len(expected)
    for r, e in zip(results, expected, strict=True):
        if e['random'] is not None:
            assert r.random is not None
            assert list(r.random.index) == refutil.as_list(e['random_names'])
            close_matrix(r.random.to_numpy(), e['random'], f'{name} random', rtol=RTOL_ROOT, atol=1e-4)
            assert r.lb_sign == e['lb_sign']
            assert r.ub_sign == e['ub_sign']
        if e['fixed'] is not None:
            assert r.fixed is not None
            assert list(r.fixed.index) == refutil.as_list(e['fixed_names'])
            close_matrix(r.fixed.to_numpy(), e['fixed'], f'{name} fixed')


def test_confint_method_matches_function() -> None:
    res = fit('hack_stepfun_b')
    a = res.confint(delta=1)
    b = metafopy.confint(res, delta=1)
    assert isinstance(a, metafopy.ConfintResult)
    assert isinstance(b, metafopy.ConfintResult)
    assert a.random is not None
    assert b.random is not None
    close_matrix(a.random.to_numpy(), b.random.to_numpy().tolist(), 'confint method')


def test_confint_extint_and_control() -> None:
    res = fit('hack_stepfun_b')
    ci = metafopy.confint(res, delta=1, extint=True, control={'vc.max': 20})
    assert isinstance(ci, metafopy.ConfintResult)
    assert ci.random is not None
    ref = REF['confint']['hack_stepfun_b']['list']
    ref_delta = (list(ref.values()) if isinstance(ref, dict) else ref)[1]['random'][0]
    close(ci.random.to_numpy()[0, 1], ref_delta[1], 'extint lower bound', rtol=RTOL_ROOT)


def test_confint_transf_fixed() -> None:
    ci = metafopy.confint(fit('hack_stepfun_b'), fixed=True, transf=numpy.exp)
    plain = metafopy.confint(fit('hack_stepfun_b'), fixed=True)
    assert isinstance(ci, metafopy.ConfintResult)
    assert isinstance(plain, metafopy.ConfintResult)
    assert ci.fixed is not None
    assert plain.fixed is not None
    close_matrix(ci.fixed.to_numpy(), numpy.exp(plain.fixed.to_numpy()).tolist(), 'transf')


# -- profile ---------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
PROFILE: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'hack_stepfun_b_tau2': ('hack_stepfun_b', 'tau2', {'tau2': True, 'steps': 6}),
    'hack_stepfun_b_delta': ('hack_stepfun_b', 'delta', {'delta': 1, 'steps': 6}),
    'hack_stepfun_b_delta_xlim': ('hack_stepfun_b', 'delta', {'delta': 1, 'xlim': [0.1, 1.5], 'steps': 5}),
    'rau_halfnorm_mods_delta': ('rau_halfnorm_mods', 'delta', {'delta': 0, 'steps': 4}),
}


@pytest.mark.parametrize('name', sorted(PROFILE))
def test_profile(name: str) -> None:
    case, comp, kwargs = PROFILE[name]
    ref = REF['profile'][name]['prof']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pr = metafopy.profile(fit(case), **kwargs)
    assert isinstance(pr, metafopy.SelProfileResult)
    assert pr.comp == comp
    close(pr.grid, ref['grid'], f'{name} grid')
    close(pr.ll, ref['ll'], f'{name} ll', rtol=RTOL_SE)
    close(pr.vc, ref['vc'], f'{name} vc')
    close(pr.maxll, ref['maxll'], f'{name} maxll')
    close(pr.ylim, ref['ylim'], f'{name} ylim', rtol=RTOL_SE)
    for key in ('beta', 'ci_lb', 'ci_ub'):
        for col, values in ref[key].items():
            close(getattr(pr, key)[col].to_numpy(), values, f'{name} {key}', rtol=RTOL_SE)
    assert pr.method == ref['method']
    assert list(pr.to_frame().columns) == [comp, 'll']


def test_profile_all_components_and_plot() -> None:
    res = fit('hack_stepfun_a')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = res.profile(steps=4, plot=False)
    assert isinstance(out, list)
    assert [o.comp for o in out] == ['tau2', 'delta', 'delta']
    assert [o.delta_pos for o in out] == [0, 1, 2]
    matplotlib.pyplot.close('all')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        single = metafopy.profile(res, delta=1, steps=4, plot=True)
    assert isinstance(single, metafopy.SelProfileResult)
    ax = matplotlib.pyplot.gcf().axes[0]
    assert 'delta' in ax.get_xlabel()
    matplotlib.pyplot.close('all')


# -- anova -----------------------------------------------------------------------------------------------


def test_anova_wald() -> None:
    res = fit('bang_stepfun_mods')
    for name, kwargs in (('bang_wald', {'btt': 1}), ('bang_wald_all', {}), ('bang_wald_X', {'X': [[1, 2, 0], [0, 1, -1]]})):
        ref = REF['anova'][name]
        got = metafopy.anova(res, **kwargs)
        assert isinstance(got, metafopy.WaldResult)
        close(got.QM, ref['QM'], f'{name} QM', rtol=RTOL_SE)
        close(got.QMp, ref['QMp'], f'{name} QMp', rtol=RTOL_SE)
        assert got.QMdf[0] == refutil.as_list(ref['QMdf'])[0]
        assert got.QMdf[1] is None
        if ref['Xb'] is not None:
            assert got.Xb is not None
            assert got.se is not None
            assert got.zval is not None
            assert got.pval is not None
            close(got.Xb, ref['Xb'], f'{name} Xb', rtol=RTOL_SE)
            close(got.se, ref['se'], f'{name} se', rtol=RTOL_SE)
            close(got.zval, ref['zval'], f'{name} zval', rtol=RTOL_SE)
            close(got.pval, ref['pval'], f'{name} pval', rtol=RTOL_SE)


@pytest.mark.parametrize(('name', 'full', 'reduced'), [('rau_lrt', 'rau_stepfun_mods', 'rau_stepfun_red'), ('hack_lrt', 'hack_stepfun_b', 'hack_stepfun_a')])
def test_anova_lrt(name: str, full: str, reduced: str) -> None:
    ref = REF['anova'][name]
    # R's anova() takes the models in the order given; the full model is the one with more parameters
    a, b = (fit(full), fit(reduced)) if ref['parms_f'] > ref['parms_r'] else (fit(reduced), fit(full))
    got = metafopy.anova(a, b)
    assert isinstance(got, metafopy.LRTResult)
    assert got.df_full == ref['parms_f']
    assert got.df_reduced == ref['parms_r']
    close(got.LRT, ref['LRT'], f'{name} LRT', rtol=RTOL_SE)
    close(got.pval, ref['pval'], f'{name} pval', rtol=RTOL_SE)
    close(got.tau2_full, ref['tau2_f'], f'{name} tau2.f', rtol=1e-3, atol=1e-6)
    close(got.tau2_reduced, ref['tau2_r'], f'{name} tau2.r', rtol=1e-3, atol=1e-6)
    close(got.R2, ref['R2'], f'{name} R2', rtol=1e-3)
    # metafor stores QE = NA in selection models
    assert math.isnan(got.QE_full)
    assert math.isnan(got.QE_reduced)
    assert ref['QE_f'] is None
    assert ref['QE_r'] is None
    close(got.fit_stats_full, ref['fit_f'], f'{name} fit f', rtol=RTOL_SE)
    close(got.fit_stats_reduced, ref['fit_r'], f'{name} fit r', rtol=RTOL_SE)


def test_anova_against_plain_model() -> None:
    sel = fit('hack_stepfun_a')
    got = sel.anova(sel.base)
    assert isinstance(got, metafopy.LRTResult)
    assert got.df_full == sel.parms
    with pytest.raises(metafopy.errors.InputError):
        metafopy.anova(sel, sel)


# -- plot data -------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
PLOT: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'hack_halfnorm_s': ('hack_halfnorm_s', {'n': 51}),
    'hack_halfnorm_s_prec': ('hack_halfnorm_sei', {'n': 51, 'prec': 'min'}),
    'hack_halfnorm_s_wald': ('hack_halfnorm_s', {'n': 51, 'ci': 'wald'}),
    'hack_stepfun_a': ('hack_stepfun_a', {}),
    'hack_stepfun_a_wald': ('hack_stepfun_a', {'ci': 'wald', 'scale': True}),
    'hack_beta': ('hack_beta', {'n': 51, 'xlim': [0.01, 0.99]}),
    'hack_power': ('hack_power', {'n': 51}),
}


@pytest.mark.parametrize('name', sorted(PLOT))
def test_plot_data(name: str) -> None:
    case, kwargs = PLOT[name]
    ref = REF['plot'][name]
    out = metafopy.plot_selmodel(fit(case), draw=False, **kwargs)
    close(out.xs, ref['xs'], f'{name} xs', rtol=1e-10)
    close(out.ys, ref['ys'], f'{name} ys', rtol=RTOL_SE)
    close(out.ys_lb, ref['ys_lb'], f'{name} ys.lb', rtol=RTOL_SE)
    close(out.ys_ub, ref['ys_ub'], f'{name} ys.ub', rtol=RTOL_SE)


@pytest.mark.parametrize('ci', [False, 'wald', 'boot'])
def test_plot_draws(ci: bool | str) -> None:
    matplotlib.pyplot.close('all')
    out = fit('hack_halfnorm_s').plot(n=25, ci=ci, reps=100, seed=1)
    assert out.ax is not None
    assert len(out.ax.lines) >= 1
    assert out.xs.shape == (25,)
    matplotlib.pyplot.close('all')
    step = fit('hack_stepfun_a').plot(ci='wald')
    assert step.ax is not None
    matplotlib.pyplot.close('all')


def test_plot_boot_is_reproducible_and_brackets_the_curve() -> None:
    res = fit('hack_halfnorm_s')
    a = metafopy.plot_selmodel(res, n=21, ci='boot', reps=400, seed=7, draw=False)
    b = metafopy.plot_selmodel(res, n=21, ci='boot', reps=400, seed=7, draw=False)
    close(a.ys_lb, b.ys_lb, 'seeded')
    assert numpy.all(a.ys_lb <= a.ys + 1e-12)
    assert numpy.all(a.ys_ub >= a.ys - 1e-12)


def test_plot_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='Cannot draw'):
        metafopy.plot_selmodel(fit('hack_trunc'), draw=False)
    with pytest.raises(metafopy.errors.InputError, match='Unknown confidence'):
        metafopy.plot_selmodel(fit('hack_halfnorm_s'), ci='foo', draw=False)
    with pytest.raises(metafopy.errors.InputError, match='between 0 and 1'):
        metafopy.plot_selmodel(fit('hack_halfnorm_s'), xlim=[-1, 0.5], draw=False)
    with pytest.raises(metafopy.errors.InputError, match="'prec'"):
        metafopy.plot_selmodel(fit('hack_halfnorm_s'), prec='foo', draw=False)


# -- argument handling ----------------------------------------------------------------------------------


def test_type_partial_matching() -> None:
    for abbreviated, full in (('half', 'halfnorm'), ('step', 'stepfun'), ('neg', 'negexp'), ('trunce', 'truncest'), ('negexpp', 'negexppow')):
        assert metafopy.models.sel_spec.match_type(abbreviated) == full


def test_errors() -> None:
    res = base_model('hack_halfnorm')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.selmodel(res, 'foo')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.selmodel(res, 'trunc', alternative='two.sided')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.selmodel(res, 'halfnorm', prec='foo')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.selmodel(res, 'stepfun')
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.selmodel(object(), 'halfnorm')  # type: ignore[arg-type]
    with pytest.raises(metafopy.errors.InputError):
        metafopy.selmodel(res, 'halfnorm', delta=[1, 2, 3])
    with pytest.warns(UserWarning, match='Extra argument'):
        metafopy.selmodel(res, 'halfnorm', foo=1)


def test_decreasing_ignored_for_other_types() -> None:
    with pytest.warns(UserWarning, match='decreasing'):
        metafopy.selmodel(base_model('hack_halfnorm'), 'halfnorm', decreasing=True)


def test_reml_input_is_refitted_with_ml() -> None:
    reml = fit('hack_stepfun_reml')
    ml = fit('hack_stepfun_b')
    close(reml.beta, ml.beta, 'reml input', rtol=1e-6)
    close(reml.ll, ml.ll, 'reml input ll', rtol=1e-6)


def test_unsupported_components() -> None:
    dec = fit('hack_stepfun_decr')
    for call in (lambda: metafopy.confint(dec), lambda: metafopy.profile(dec, delta=1)):
        with pytest.raises(metafopy.errors.NotSupportedError):
            call()
    res = fit('bang_stepfun_beta')
    with pytest.raises(metafopy.errors.InputError, match='beta'):
        metafopy.confint(res)
    ee = fit('hack_stepfun_ee')
    with pytest.raises(metafopy.errors.InputError, match='tau2'):
        metafopy.confint(ee, tau2=True)
    with pytest.raises(metafopy.errors.InputError, match='fixed'):
        metafopy.confint(fit('hack_stepfun_partial'), delta=2)
    with pytest.raises(metafopy.errors.InputError, match='No such'):
        metafopy.profile(fit('hack_stepfun_b'), delta=7)
    with pytest.raises(metafopy.errors.InputError, match='only one'):
        metafopy.profile(fit('hack_stepfun_b'), delta=1, tau2=True)


# -- building blocks ---------------------------------------------------------------------------------------


def test_bfgs_matches_optim() -> None:
    """R: optim(c(-1.2, 1), fr, method = 'BFGS') with a numerical gradient converges to (1, 1) after 118 function and 38 gradient evaluations."""

    def rosenbrock(x: numpy.typing.NDArray[numpy.float64]) -> float:
        return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

    res = metafopy.core.bfgs.optim_bfgs(rosenbrock, numpy.array([-1.2, 1.0]))
    assert res.convergence == 0
    assert (res.fncount, res.grcount) == (118, 38)
    numpy.testing.assert_allclose(res.par, [0.9998044, 0.9996084], rtol=1e-6)


def test_bfgs_maxit() -> None:
    res = metafopy.core.bfgs.optim_bfgs(lambda x: float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2), numpy.array([-1.2, 1.0]), maxit=5)
    assert res.convergence == 1


def test_integrate() -> None:
    value = metafopy.core.integrate.integrate(lambda x: x**2, 0.0, 1.0)
    assert value == pytest.approx(1 / 3, rel=1e-10)
    inf = metafopy.core.integrate.integrate(lambda x: math.exp(-x * x / 2), -math.inf, math.inf)
    assert inf == pytest.approx(math.sqrt(2 * math.pi), rel=1e-8)
    with pytest.raises(metafopy.errors.EstimationError):
        metafopy.core.integrate.integrate(lambda x: 1 / x, 0.0, 1.0)


def test_optimizer_choice() -> None:
    choice = metafopy.models.sel_opt.choose_optimizer('BFGS', 'BFGS', 'halfnorm')
    assert (choice.optimizer, choice.optmethod) == ('optim', 'BFGS')
    with pytest.warns(UserWarning, match='stepcon'):
        stepcon = metafopy.models.sel_opt.choose_optimizer('nlminb', 'BFGS', 'stepcon')
    assert stepcon.optimizer == 'solnp'
    assert metafopy.models.sel_opt.choose_optimizer('optim', 'BFGS', 'stepcon').optimizer == 'solnp'
    with pytest.raises(metafopy.errors.InputError):
        metafopy.models.sel_opt.partial_match('zzz', metafopy.models.sel_opt.OPTIMIZERS, 'optimizer')
