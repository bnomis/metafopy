"""The ``link`` and ``family`` arguments of ``rma_glmm`` checked against R metafor 5.2-1 (fixtures from ``r/generate_glmm_links.R``).

metafor passes ``link`` and ``family`` through ``...`` to ``glm()`` and ``lme4::glmer()``. The fixed-effects models and the saturated
models are ``glm()`` fits that metafopy reproduces to machine precision, including the link functions themselves (``TestLinks``).
The mixed models use ``glmer()`` with probit, complementary log-log, identity and square root links, whose optimiser stops at a
loose tolerance (R then often only warns), so those estimates agree to ``MIXED`` or the entries of ``TOLERANCE``.
"""

import functools
import json
import pathlib
import re
import typing
import warnings

import glmm_links_cases
import numpy
import pytest
import refutil
import scipy.stats

import metafopy
import metafopy.core.glm
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase9' / 'glmm_links.json').read_text(encoding='utf-8'))

EXACT = 1e-8
MIXED = 5e-3
# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
TOLERANCE: dict[str, float] = {
    # lme4's random-effects fit with the complementary log-log link stops at a larger gradient than the other fits
    'or_fs_cloglog_col': 1e-1,
    'or_rs_probit': 1e-2,
    'or_rs_cloglog': 1e-2,
    'or_rs_probit_fe': 1e-4,
    'el_probit_fe': 1e-3,
}
# the Wald statistic of the exact conditional model is dominated by finite-difference noise in R
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
NOISY_QE_WLD = ('el_probit_fe',)
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
LME4_NOISE = (
    'Model failed to converge',
    'Model is nearly unidentifiable',
    'Cannot fit saturated model',
    'variance-covariance matrix computed from finite-difference Hessian',
    'glm.fit: algorithm did not converge',
)
EXPERIMENTAL = "The use of this 'measure' is experimental - treat results with caution."
FITS = list(glmm_links_cases.CASES)


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """Relative differences scaled by ``max(|expected|, 1e-3)``; NaN (and infinite values) must match R's NA."""
    exp = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in refutil.as_list(expected)], dtype=float))
    act = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    act = numpy.where(numpy.isfinite(act), act, numpy.nan)
    assert act.shape == exp.shape, f'{what}: {act.shape} vs {exp.shape}'
    assert numpy.array_equal(numpy.isnan(act), numpy.isnan(exp)), f'{what}: {act} vs {exp}'
    mask = ~numpy.isnan(exp)
    worst = float(numpy.max(numpy.abs(act[mask] - exp[mask]) / numpy.maximum(numpy.abs(exp[mask]), 1e-3), initial=0.0))
    assert worst <= rtol, f'{what}: relative difference {worst:.2e} > {rtol:.0e}\n{act}\n{exp}'


@functools.cache
def fit_with_warnings(name: str) -> tuple[metafopy.GLMMResult, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = glmm_links_cases.CASES[name]()
    return res, tuple(str(w.message).split('\n')[0] for w in caught)


def is_mixed(ref: dict[str, typing.Any]) -> bool:
    """Whether a mixed-effects model (``glmer``) is behind the estimates: ``method='ML'`` or the random study effects of ``UM.RS``."""
    return bool(ref['method'] == 'ML' or ref['model'] == 'UM.RS')


def tolerance(name: str) -> float:
    return TOLERANCE.get(name, MIXED if is_mixed(REF[name]) else EXACT)


@pytest.mark.parametrize('name', FITS)
def test_fit(name: str) -> None:
    ref = REF[name]
    assert 'error' not in ref, ref.get('error')
    res, _ = fit_with_warnings(name)
    tol = tolerance(name)
    close(res.beta, ref['beta'], 'beta', tol)
    close(res.se, ref['se'], 'se', tol)
    close(res.vb.ravel('F'), ref['vb'], 'vb', 2 * tol)
    close(res.zval, ref['zval'], 'zval', tol)
    close(res.pval, ref['pval'], 'pval', 10 * tol)
    close(res.ci_lb, ref['ci_lb'], 'ci_lb', tol)
    close(res.ci_ub, ref['ci_ub'], 'ci_ub', tol)
    close(res.tau2, ref['tau2'], 'tau2', tol)
    close(res.sigma2, ref['sigma2'], 'sigma2', tol)
    close(res.I2, ref['I2'], 'I2', tol)
    close(res.H2, ref['H2'], 'H2', tol)
    close(res.QM, ref['QM'], 'QM', 4 * tol)
    assert (res.k, res.k_f, res.k_yi, res.p, res.p_eff, res.parms, res.QE_df) == (
        ref['k'],
        ref['k_f'],
        ref['k_yi'],
        ref['p'],
        ref['p_eff'],
        ref['parms'],
        ref['QE_df'],
    )
    assert (res.method, res.model) == (ref['method'], ref['model'])
    qe_tol = tol if res.model == 'UM.RS' else EXACT
    if name not in NOISY_QE_WLD:
        close(res.QE_Wld, ref['QE_Wld'], 'QE_Wld', qe_tol)
    close(res.QE_LRT, ref['QE_LRT'], 'QE_LRT', qe_tol)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', tol)


@pytest.mark.parametrize('name', FITS)
def test_warnings(name: str) -> None:
    """metafopy issues R's warnings, except those from lme4's optimiser and from the roundoff-dependent saturated models."""
    _, issued = fit_with_warnings(name)
    expected = [w.split('\n')[0] for w in refutil.as_list(REF[name]['warnings']) if not w.startswith(LME4_NOISE)]
    assert sorted(set(expected)) == sorted({w for w in issued if not w.startswith(('step size truncated', *LME4_NOISE))})


@pytest.mark.parametrize('name', [n for n in FITS if not is_mixed(REF[n]) and n not in NOISY_QE_WLD])
def test_print_exact(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    assert refutil.same_print(str(res), ref['print']), f'\n{res}\n{ref["print"]}'
    assert refutil.same_print(res.summary(showfit=True), ref['print_showfit']), f'\n{res.summary(showfit=True)}\n{ref["print_showfit"]}'


def mask_numbers(text: str) -> str:
    return re.sub(r' +', ' ', re.sub(r'-?\d+\.\d+|<\.\d+|\d+', '#', refutil.r_print(text)))


@pytest.mark.parametrize('name', [n for n in FITS if is_mixed(REF[n]) or n in NOISY_QE_WLD])
def test_print_layout(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    assert mask_numbers(str(res)) == mask_numbers(ref['print'])


# -- the family and link arguments -----------------------------------------------------------------------------------------------------------


def test_family_equals_link() -> None:
    """``family=binomial(link=...)`` and ``link=...`` give the same model; a family name stands for its default link."""
    by_link, _ = fit_with_warnings('or_fs_probit')
    by_family, _ = fit_with_warnings('or_fam_probit')
    assert numpy.array_equal(by_link.beta, by_family.beta)
    by_name, _ = fit_with_warnings('or_fam_binomial')
    default, _ = fit_with_warnings('or_fs_logit')
    assert numpy.array_equal(by_name.beta, default.beta)


def test_family_replaces_the_link_of_the_measure() -> None:
    """With ``family=binomial`` the risk ratio model has the logit link, not its own log link (R ignores ``link`` then)."""
    ref = REF['rr_fam_binomial']
    res, issued = fit_with_warnings('rr_fam_binomial')
    close(res.beta, ref['beta'], 'beta', MIXED)
    assert EXPERIMENTAL in issued


def test_family_wins_over_link() -> None:
    kwargs: dict[str, typing.Any] = {
        'ai': 'tpos',
        'bi': 'tneg',
        'ci': 'cpos',
        'di': 'cneg',
        'data': metafopy.datasets.load('bcg'),
        'measure': 'OR',
        'model': 'UM.FS',
    }
    both = metafopy.rma_glmm(**kwargs, method='FE', link='cloglog', family=metafopy.core.glm.binomial('probit'))
    family = metafopy.rma_glmm(**kwargs, method='FE', family=metafopy.core.glm.binomial('probit'))
    assert numpy.array_equal(both.beta, family.beta)


@pytest.mark.parametrize('name', ['al_probit', 'al_family', 'el_probit_fe'])
def test_conditional_models_ignore_link_and_family(name: str) -> None:
    ref = REF[name]
    res, _ = fit_with_warnings(name)
    plain = glmm_links_cases.b2(model=res.model, method=res.method)()
    close(res.beta, plain.beta.tolist(), 'beta', 1e-12)
    close(res.beta, ref['beta'], 'beta', MIXED if res.method == 'ML' else TOLERANCE.get(name, EXACT))


@pytest.mark.parametrize('link', ['bogus', '1/mu^2', 'inverse'])
def test_unknown_link(link: str) -> None:
    assert 'not recognised' in REF['errors']['bad_link']
    with pytest.raises(metafopy.errors.InputError, match='Unknown link'):
        glmm_links_cases.b2(model='UM.FS', link=link)()


def test_unknown_family() -> None:
    with pytest.raises(metafopy.errors.InputError, match="Unknown 'family'"):
        glmm_links_cases.b2(model='UM.FS', family='gaussian')()
    with pytest.raises(metafopy.errors.InputError, match='Unknown family'):
        metafopy.core.glm.GLMFamily('gaussian', 'identity')  # type: ignore[arg-type]
    with pytest.raises(metafopy.errors.InputError, match='Unknown link'):
        metafopy.core.glm.binomial(link='bogus')


@pytest.mark.parametrize(
    ('call', 'own'),
    [
        (glmm_links_cases.b2(model='UM.FS', family=metafopy.core.glm.poisson()), 'binomial'),
        (glmm_links_cases.h2(model='UM.FS', family=metafopy.core.glm.binomial()), 'poisson'),
        (glmm_links_cases.p1('PLO', family='poisson'), 'binomial'),
        (glmm_links_cases.r1(family='binomial'), 'poisson'),
    ],
)
def test_family_must_match_the_data(call: typing.Callable[[], metafopy.GLMMResult], own: str) -> None:
    with pytest.raises(metafopy.errors.InputError, match=f"must be '{own}'"):
        call()


def test_family_str() -> None:
    assert str(metafopy.core.glm.binomial('probit')) == 'binomial(link = "probit")'
    assert metafopy.core.glm.poisson() == metafopy.core.glm.GLMFamily('poisson', 'log')


@pytest.mark.parametrize('link', ['probit', 'cloglog'])
@pytest.mark.parametrize('package', ['GLMMadaptive', 'glmmTMB'])
def test_other_packages(package: str, link: str) -> None:
    """GLMMadaptive and glmmTMB are not installed for the R fixtures; they maximise the likelihood that lme4 approximates, so R's lme4 fit is the reference."""
    ref = REF[f'or_fs_{link}']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = glmm_links_cases.b2(model='UM.FS', link=link, control={'package': package})()
    close(res.beta, ref['beta'], 'beta', MIXED)
    close(res.se, ref['se'], 'se', 1e-2)
    close(res.tau2, ref['tau2'], 'tau2', 2e-2)


# -- link functions --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('link', list(REF['link_functions']))
class TestLinks:
    """``linkinv``, ``mu.eta``, ``linkfun`` and ``valideta`` of R's families on a grid."""

    def test_linkinv(self, link: str) -> None:
        ref = REF['link_functions'][link]
        eta = numpy.array(ref['eta'])
        with numpy.errstate(all='ignore'):
            close(metafopy.core.glm.link_inv(link, eta), ref['linkinv'], 'linkinv', 1e-13)  # type: ignore[arg-type]

    def test_mu_eta(self, link: str) -> None:
        ref = REF['link_functions'][link]
        eta = numpy.array(ref['eta'])
        with numpy.errstate(all='ignore'):
            close(metafopy.core.glm.mu_eta(link, eta), ref['mu_eta'], 'mu.eta', 1e-13)  # type: ignore[arg-type]

    def test_linkfun(self, link: str) -> None:
        ref = REF['link_functions'][link]
        mu = numpy.array(ref['mu'])
        close(metafopy.core.glm.link_fun(link, mu), ref['linkfun'], 'linkfun', 1e-12)  # type: ignore[arg-type]

    def test_valideta(self, link: str) -> None:
        ref = REF['link_functions'][link]
        eta = numpy.array(ref['eta'])
        assert metafopy.core.glm.valid_eta(link, eta) == ref['valideta']  # type: ignore[arg-type]
        assert metafopy.core.glm.valid_eta(link, numpy.abs(eta) + 1) == ref['valideta_pos']  # type: ignore[arg-type]


# -- predict ---------------------------------------------------------------------------------------------------------------------------------

type Predict = typing.Callable[[], metafopy.PredictResult]


def predicting(model: typing.Callable[[], metafopy.GLMMResult], **kw: typing.Any) -> Predict:
    fitted = functools.cache(model)

    def call() -> metafopy.PredictResult:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return metafopy.predict(fitted(), **kw)

    return call


PREDICT: dict[str, Predict] = {
    'or_probit_mod': predicting(glmm_links_cases.b2(model='UM.FS', link='probit', mods='~ ablat'), newmods=[10, 30, 50]),
    'or_probit_mod_exp': predicting(glmm_links_cases.b2(model='UM.FS', link='probit', mods='~ ablat'), newmods=[10, 30, 50], transf=numpy.exp),
    'or_probit': predicting(glmm_links_cases.b2(model='UM.FS', link='probit')),
    'or_cloglog_fe': predicting(glmm_links_cases.b2(model='UM.FS', link='cloglog', method='FE'), transf=numpy.exp),
    'plo_probit': predicting(glmm_links_cases.p1('PLO', link='probit'), transf=scipy.stats.norm.cdf),
    'plo_cloglog_mod': predicting(glmm_links_cases.p1('PLO', link='cloglog', mods='~ ablat'), newmods=[10, 30, 50]),
    'irln_sqrt_fe': predicting(glmm_links_cases.r1(link='sqrt', method='FE')),
    'irr_identity_fe': predicting(glmm_links_cases.h2(model='UM.FS', link='identity', method='FE'), transf=numpy.exp),
}
PREDICT_TOLERANCE = {'or_probit_mod': MIXED, 'or_probit_mod_exp': MIXED, 'or_probit': MIXED, 'plo_probit': MIXED, 'plo_cloglog_mod': MIXED}


@pytest.mark.parametrize('name', list(PREDICT))
def test_predict(name: str) -> None:
    ref = REF['predict'][name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = PREDICT[name]()
    tol = PREDICT_TOLERANCE.get(name, EXACT)
    # auditme: accept[literal-duplication] the result fields repeat as metafor's predict() returns them
    for key in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        expected = ref.get(key)
        actual = getattr(res, key)
        if expected is None:
            assert numpy.all(numpy.isnan(actual)), key
        else:
            close(actual, expected, key, tol)
