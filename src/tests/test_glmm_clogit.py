"""``rma_glmm`` with ``control={'optimizer': 'clogit'}`` and ``'clogistic'`` checked against R metafor 5.2-1 (fixtures from ``r/generate_glmm_clogit.R``).

``survival::clogit()`` and ``Epi::clogistic()`` maximise the exact conditional likelihood by Newton-Raphson; metafopy repeats their
iterations (including the Cholesky tolerance that marks aliased columns, the step halving and the stopping rules) on the sums over the
non-central hypergeometric support instead of individual patient data, so the estimates, the covariance matrices and the Wald-type
heterogeneity test (taken from the saturated model's covariance matrix) agree to rounding error. The log-likelihoods are evaluated by R
with ``BiasedUrn`` to a precision of 1e-10, which bounds the agreement of the likelihood ratio test at about 1e-9.
"""

import functools
import json
import pathlib
import re
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase9' / 'glmm_clogit.json').read_text(encoding='utf-8'))

TIGHT = 1e-9
LRT = 1e-9

type Fit = typing.Callable[..., metafopy.GLMMResult]

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
OPTIMIZERS = ('clogit', 'clogistic')


def bcg_small() -> pandas.DataFrame:
    """The five smallest-population trials of bcg that fit in R's ``clogit()`` (it works on individual patient data)."""
    data = metafopy.datasets.load('bcg')
    return data[data['trial'].isin([1, 2, 3, 7, 10])].reset_index(drop=True)


def bcg_na() -> pandas.DataFrame:
    data = bcg_small()
    data.loc[1, 'ablat'] = numpy.nan
    return data


def bcg_fit(data: pandas.DataFrame, model: str = 'CM.EL', **kw: typing.Any) -> metafopy.GLMMResult:
    return metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=data, measure='OR', model=model, **kw)


def table_fit(data: pandas.DataFrame, **kw: typing.Any) -> metafopy.GLMMResult:
    return metafopy.rma_glmm(ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=data, measure='OR', model='CM.EL', **kw)


def collins_fit(**kw: typing.Any) -> metafopy.GLMMResult:
    data = metafopy.datasets.load('collins1985a')
    return metafopy.rma_glmm(ai='b.xci', n1i='nci', ci='b.xti', n2i='nti', data=data, measure='OR', model='CM.EL', **kw)


def nielweise() -> pandas.DataFrame:
    return metafopy.datasets.load('nielweise2007')


def lee() -> pandas.DataFrame:
    return metafopy.datasets.load('lee2004')


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
CASES: dict[str, Fit] = {
    'bcg_fe': lambda **c: bcg_fit(bcg_small(), method='FE', control=c),
    'bcg_ee': lambda **c: bcg_fit(bcg_small(), method='EE', control=c),
    'bcg_mod': lambda **c: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control=c),
    'bcg_mod2': lambda **c: bcg_fit(bcg_small(), mods='~ ablat + year', method='FE', control=c),
    'bcg_mod_na': lambda **c: bcg_fit(bcg_na(), mods='~ ablat', method='FE', control=c),
    'bcg_t': lambda **c: bcg_fit(bcg_small(), mods='~ ablat', method='FE', test='t', level=90, control=c),
    'bcg_noscale': lambda **c: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={**c, 'scaleX': False}),
    'nw_fe': lambda **c: table_fit(nielweise(), method='FE', control=c),
    'nw_mod': lambda **c: table_fit(nielweise(), mods='~ year', method='FE', control=c),
    'lee_fe': lambda **c: table_fit(lee(), method='FE', control=c),
    'lee_mod': lambda **c: table_fit(lee(), mods='~ year', method='FE', control=c),
    'col_fe': lambda **c: collins_fit(method='FE', control=c),
    'col_mod': lambda **c: collins_fit(mods='~ year', method='FE', control=c),
}
# the exact conditional model has zero cells in the saturated fit of some data sets: the study coefficients are then infinite and
# the path of the iterations decides the stopping point, so R's and metafopy's last digits can differ by the roundoff of exp()
INFINITE_ESTIMATES: dict[tuple[str, str], float] = {}
SCALARS = ('QE_Wld', 'QE_LRT', 'QM', 'I2', 'H2', 'tau2')


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """Relative differences scaled by ``max(|expected|, 1e-3)``; NaN must match NaN."""
    exp = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in refutil.as_list(expected)], dtype=float))
    act = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    assert act.shape == exp.shape, what
    assert numpy.array_equal(numpy.isnan(act), numpy.isnan(exp)), f'{what}: {act} vs {exp}'
    mask = ~numpy.isnan(exp)
    worst = float(numpy.max(numpy.abs(act[mask] - exp[mask]) / numpy.maximum(numpy.abs(exp[mask]), 1e-3), initial=0.0))
    assert worst <= rtol, f'{what}: relative difference {worst:.2e} > {rtol:.0e}\n{act}\n{exp}'


@functools.cache
def fit(optimizer: str, name: str) -> tuple[metafopy.GLMMResult, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = CASES[name](optimizer=optimizer)
    return res, tuple(str(w.message).split('\n')[0] for w in caught)


def tolerance(optimizer: str, name: str) -> float:
    return INFINITE_ESTIMATES.get((optimizer, name), TIGHT)


def check(res: metafopy.GLMMResult, ref: dict[str, typing.Any], tol: float) -> None:
    close(res.beta, ref['beta'], 'beta', tol)
    close(res.se, ref['se'], 'se', tol)
    close(res.vb.ravel('F'), ref['vb'], 'vb', 2 * tol)
    close(res.zval, ref['zval'], 'zval', tol)
    close(res.pval, ref['pval'], 'pval', tol)
    close(res.ci_lb, ref['ci_lb'], 'ci_lb', tol)
    close(res.ci_ub, ref['ci_ub'], 'ci_ub', tol)
    for key in SCALARS:
        close(getattr(res, key), ref[key], key, tol if key != 'QE_LRT' else LRT)
    close(res.QEp_Wld, ref['QEp_Wld'], 'QEp_Wld', 10 * tol)
    close(res.QEp_LRT, ref['QEp_LRT'], 'QEp_LRT', 10 * LRT)
    close(res.QMp, ref['QMp'], 'QMp', 10 * tol)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', LRT)
    assert (res.k, res.k_f, res.k_yi, res.p, res.p_eff, res.parms, res.k_eff) == (
        ref['k'],
        ref['k_f'],
        ref['k_yi'],
        ref['p'],
        ref['p_eff'],
        ref['parms'],
        ref['k_eff'],
    )
    assert (res.method, res.model, res.QE_df) == (ref['method'], ref['model'], ref['QE_df'])
    assert list(res.coef_names) == refutil.as_list(ref['coef_names'])


@pytest.mark.parametrize('optimizer', OPTIMIZERS)
@pytest.mark.parametrize('name', list(CASES))
def test_fit(optimizer: str, name: str) -> None:
    ref = REF[optimizer][name]
    assert 'error' not in ref, ref.get('error')
    res, _ = fit(optimizer, name)
    check(res, ref, tolerance(optimizer, name))


@pytest.mark.parametrize('optimizer', OPTIMIZERS)
@pytest.mark.parametrize('name', list(CASES))
def test_print(optimizer: str, name: str) -> None:
    ref = REF[optimizer][name]
    res, _ = fit(optimizer, name)
    assert refutil.same_print(str(res), ref['print']), f'\n{res}\n{ref["print"]}'
    assert refutil.same_print(res.summary(showfit=True), ref['print_showfit']), f'\n{res.summary(showfit=True)}\n{ref["print_showfit"]}'


@pytest.mark.parametrize('optimizer', OPTIMIZERS)
@pytest.mark.parametrize('name', list(CASES))
def test_warnings(optimizer: str, name: str) -> None:
    """The warnings of ``clogit()`` for the fixed-effects model carry over; those of the saturated model are not shown by metafor either."""
    _, issued = fit(optimizer, name)
    expected = {w.split('\n')[0] for w in refutil.as_list(REF[optimizer][name]['warnings'])}
    assert expected == set(issued)


def test_default_optimizer_agrees() -> None:
    """The default optimiser maximises the same likelihood, so the three fits agree to the (loose) tolerance of the numerical Hessian."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = table_fit(nielweise(), method='FE')
    res, _ = fit('clogit', 'nw_fe')
    close(res.beta, default.beta.tolist(), 'beta', 1e-5)
    close(res.se, default.se.tolist(), 'se', 1e-4)
    close(res.QE_LRT, [default.QE_LRT], 'QE_LRT', 1e-6)
    close(res.fit_stats_ml.ll, [default.fit_stats_ml.ll], 'll', 1e-8)


# -- control lists -----------------------------------------------------------------------------------------------------------

CTRL: dict[str, Fit] = {
    'clogit_eps': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'control': {'eps': 1e-5}}}),
    'clogit_init': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'init': [-0.5, 0.1]}}),
    'clogit_iter': lambda: table_fit(nielweise(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'control': {'iter.max': 3}}}),
    'clogit_exact': lambda: bcg_fit(bcg_small(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'exact'}}),
    'ties_efron': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'efron'}}),
    'ties_efron_nw': lambda: table_fit(nielweise(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'efron'}}),
    'ties_breslow': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'breslow'}}),
    'ties_breslow_nw': lambda: table_fit(nielweise(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'breslow'}}),
    'ties_approximate': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'approximate'}}),
    'ties_approximate_nw': lambda: table_fit(nielweise(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'approximate'}}),
    'clogistic_eps': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogistic', 'clogisticCtrl': {'eps': 1e-3}}),
    'clogistic_init': lambda: bcg_fit(bcg_small(), mods='~ ablat', method='FE', control={'optimizer': 'clogistic', 'clogisticCtrl': {'init': [-0.5, 0.1]}}),
    'clogistic_iter': lambda: table_fit(nielweise(), method='FE', control={'optimizer': 'clogistic', 'clogisticCtrl': {'iter.max': 3}}),
}


@pytest.mark.parametrize('name', list(CTRL))
def test_control_lists(name: str) -> None:
    ref = REF['ctrl'][name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'error' in ref:
            # the start values fit the model but not the saturated model, which uses the same control list
            assert ref['error'].startswith('Cannot fit saturated model')
            with pytest.raises(metafopy.errors.EstimationError, match='Cannot fit saturated model'):
                CTRL[name]()
            return
        res = CTRL[name]()
    check(res, ref, TIGHT)


def test_iteration_limit_warnings() -> None:
    """Running out of iterations is reported as the originals do."""
    with pytest.warns(UserWarning, match='Ran out of iterations and did not converge'):
        table_fit(nielweise(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'control': {'iter.max': 3}}})
    with pytest.warns(UserWarning, match='Iteration limit exceeded'):
        table_fit(nielweise(), method='FE', control={'optimizer': 'clogistic', 'clogisticCtrl': {'iter.max': 3}})


# -- arguments and errors ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('key', ['clogit_ml', 'clogistic_ml', 'clogit_ml_um'])
def test_ml_is_not_possible(key: str) -> None:
    optimizer = 'clogistic' if key.startswith('clogistic') else 'clogit'
    model = 'UM.FS' if key.endswith('um') else 'CM.EL'
    with pytest.raises(metafopy.errors.InputError, match=re.escape(REF['errors'][key])):
        bcg_fit(bcg_small(), model=model, control={'optimizer': optimizer})


@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_single_study(optimizer: str) -> None:
    with pytest.raises(metafopy.errors.InputError, match=re.escape(REF['errors'][f'{optimizer}_k1'])):
        bcg_fit(bcg_small().iloc[:1], method='FE', control={'optimizer': optimizer})


def test_unknown_optimizer() -> None:
    assert REF['errors']['bad_optimizer']
    with pytest.raises(metafopy.errors.InputError, match='optimizer'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'bogus'})
    with pytest.raises(metafopy.errors.InputError, match='optimizer'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'BFGS'})


def test_unknown_control_entries() -> None:
    with pytest.raises(metafopy.errors.InputError, match='clogitCtrl'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'bogus': 1}})
    with pytest.raises(metafopy.errors.InputError, match='clogisticCtrl'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'clogistic', 'clogisticCtrl': {'bogus': 1}})
    with pytest.raises(metafopy.errors.InputError, match='method'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'method': 'bogus'}})
    with pytest.raises(metafopy.errors.EstimationError, match='Cannot fit FE model'):
        bcg_fit(bcg_small(), method='FE', control={'optimizer': 'clogit', 'clogitCtrl': {'init': [1.0, 2.0]}})


@pytest.mark.parametrize(('name', 'model', 'optimizer'), [('al', 'CM.AL', 'clogit'), ('um', 'UM.FS', 'clogistic')])
def test_other_models_ignore_the_optimizer(name: str, model: str, optimizer: str) -> None:
    ref = REF['ignored'][name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = bcg_fit(bcg_small(), model=model, method='FE', control={'optimizer': optimizer})
    close(res.beta, ref['beta'], 'beta', 1e-6)
    close(res.se, ref['se'], 'se', 1e-6)
    assert res.model == model


# -- the numerical routines ---------------------------------------------------------------------------------------------------------


def test_cholesky_marks_redundant_columns() -> None:
    import metafopy.core.clogit

    a = numpy.array([[4.0, 2.0, 4.0], [2.0, 2.0, 2.0], [4.0, 2.0, 4.0]])
    fac = a.copy()
    assert metafopy.core.clogit.cholesky2(fac, 1e-9) == 2
    assert fac[2, 2] == 0.0
    inv = metafopy.core.clogit.chinv2(fac)
    assert numpy.all(inv[2] == 0.0)
    rhs = numpy.array([1.0, 1.0, 0.0])
    metafopy.core.clogit.chsolve2(fac, rhs)
    assert rhs[2] == 0.0


def test_aliased_column_is_dropped() -> None:
    """A duplicated study dummy of the saturated design gives a zero pivot, as in ``coxph()``: the coefficient is missing and the others unchanged."""
    import metafopy.core.clogit

    ai, bi, ci, di = (numpy.array(v, dtype=float) for v in ([3, 5, 7, 2], [10, 8, 4, 9], [2, 6, 3, 4], [11, 6, 9, 7]))
    strata = metafopy.core.clogit.Strata.from_counts(ai, bi, ci, di)
    X = numpy.column_stack([numpy.ones(4), numpy.array([0.5, -1.0, 0.2, 0.1])])
    plain = metafopy.core.clogit.coxph_exact(strata, X)
    doubled = metafopy.core.clogit.coxph_exact(strata, numpy.column_stack([X, X[:, 1]]))
    assert bool(numpy.isnan(doubled.coef[2])) or bool(numpy.isnan(doubled.coef[1]))
    kept = ~numpy.isnan(doubled.coef)
    assert int(kept.sum()) == 2
    assert numpy.isclose(doubled.loglik, plain.loglik, rtol=1e-9)
