"""Optimisers, control lists and numerical derivatives (round five of the audit: area D).

``rma_mv(control=)`` (``nearpd``, ``optimizer``, ``hesspack``, ``hessianCtrl``, ``mfmaxit``, ``ncpus``), the optimisers of location-scale
models, ``rma_glmm(control=)`` (``hesspack``, ``hessianCtrl``, ``dnchgcalc``, ``dnchgprec``), the ``vc.min``/``vc.max`` control of
``confint()`` for location-scale fits and the lme4 warning that R does not give. The reference values are R metafor 5.2-1 fixtures that
already exist (``reference/phase7/matreg.json``, ``reference/phase3/glmm.json``) and analytic derivatives.
"""

import json
import math
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil
import scipy.stats

import metafopy
import metafopy.core.nchg
import metafopy.core.numderiv
import metafopy.datasets
import metafopy.errors

REFERENCE = pathlib.Path(__file__).parent / 'reference'
MATREG: dict[str, typing.Any] = json.loads((REFERENCE / 'phase7' / 'matreg.json').read_text(encoding='utf-8'))
GLMM: dict[str, typing.Any] = json.loads((REFERENCE / 'phase3' / 'glmm.json').read_text(encoding='utf-8'))


def colditz_long() -> pandas.DataFrame:
    """The long-format dataset of the example of ``?matreg`` (van Houwelingen et al., 2002)."""
    long = metafopy.to_long(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('colditz1994'), append=False)
    long = metafopy.escalc(measure='PLO', xi='out1', mi='out2', data=long)
    long['group'] = pandas.Categorical(long['group'].astype(str).map({'2': 'con', '1': 'exp'}), categories=['con', 'exp'])
    return long


def colditz_fit(control: dict[str, typing.Any] | None) -> metafopy.RMAMVResult:
    return metafopy.rma_mv(
        'yi', 'vi', mods='~ 0 + group', random='~ group | study', struct='UN', data=colditz_long(), method='ML', cvvc='varcov', control=control
    )


def bcg_trials() -> pandas.DataFrame:
    dat = metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))
    dat['trial'] = range(len(dat))
    return dat


def mv_fit(**control: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=bcg_trials(), method='REML', cvvc=True, control=control or None)


# -- rma_mv(control=) -------------------------------------------------------------------------------------


def test_colditz_cvvc_needs_nearpd_for_vvc() -> None:
    """The example of ``?matreg``: ``control=list(nearpd=TRUE)`` makes the Hessian invertible, so that ``res$vvc`` exists (R: 3 x 3)."""
    with pytest.warns(UserWarning, match='Error when trying to invert the Hessian'):
        assert colditz_fit(None).vvc is None
    res = colditz_fit({'nearpd': True})
    ref = MATREG['colditz']
    assert res.vvc is not None
    refutil.close(res.vvc.ravel(), numpy.array(ref['vvc']).ravel().tolist(), 'vvc', rtol=1e-5)
    assert res.G is not None
    refutil.close(numpy.asarray(res.G).ravel(), numpy.array(ref['G']).ravel().tolist(), 'G', rtol=1e-6)


def test_colditz_vvc_feeds_matreg() -> None:
    res = colditz_fit({'nearpd': True})
    ref = MATREG['colditz']
    assert res.G is not None
    G = pandas.DataFrame(numpy.asarray(ref['G']), index=['con', 'exp'], columns=['con', 'exp'])
    fit = metafopy.matreg(1, [0], R=G, cov=True, means=ref['means'], V=res.vvc)
    refutil.close(fit.tab['beta'], ref['fit_V']['beta'], 'matreg beta', rtol=1e-5)
    refutil.close(fit.tab['se'], ref['fit_V']['se'], 'matreg se', rtol=1e-4)


@pytest.mark.parametrize(
    'control',
    [
        {'optimizer': 'optim'},
        {'optimizer': 'BFGS', 'reltol': 1e-12},
        {'optimizer': 'nlm'},
        {'optimizer': 'bobyqa'},
        {'optimizer': 'hjk'},
        {'optimizer': 'L-BFGS-B'},
        {'optimizer': 'optim', 'optmethod': 'CG'},
        {'ncpus': 2},
        {'parallel': {}, 'cl': None},
        {'nearpd': True},
    ],
)
def test_mv_optimizers_reach_the_same_optimum(control: dict[str, typing.Any]) -> None:
    ref = mv_fit()
    res = mv_fit(**control)
    assert ref.sigma2 is not None
    assert res.sigma2 is not None
    refutil.close(res.sigma2, ref.sigma2.tolist(), 'sigma2', rtol=1e-4)
    refutil.close(res.beta, ref.beta.tolist(), 'beta', rtol=1e-5)


def test_mv_nelder_mead_is_the_port_of_optim() -> None:
    """``optim``'s Nelder-Mead stops at its own relative tolerance (1e-8 on the log-likelihood), not at the exact optimum."""
    ref = mv_fit()
    res = mv_fit(optimizer='Nelder-Mead')
    assert ref.sigma2 is not None
    assert res.sigma2 is not None
    assert res.sigma2[0] == pytest.approx(ref.sigma2[0], rel=1e-3)


def test_mv_unknown_optimizer_and_control() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'optimizer' should be one of"):
        mv_fit(optimizer='gradient-descent')
    with pytest.raises(metafopy.errors.ArgumentError, match='unknown control argument'):
        mv_fit(not_an_argument=1)


def test_mv_mfmaxit_limits_the_evaluations() -> None:
    with pytest.raises(metafopy.errors.EstimationError, match='mfmaxit'):
        mv_fit(mfmaxit=3)
    assert mv_fit(mfmaxit=10**6).sigma2 is not None


def test_mv_hesspack_choices_agree() -> None:
    base = mv_fit().vvc
    assert base is not None
    for control in ({'hesspack': 'pracma'}, {'hesspack': 'calculus'}, {'hessianCtrl': {'r': 6}}, {'hessian.ctrl': {'d': 0.05}}):
        vvc = mv_fit(**control).vvc
        assert vvc is not None
        refutil.close(vvc.ravel(), base.ravel().tolist(), str(control), rtol=1e-3)


def test_mv_hessianctrl_is_used() -> None:
    wide = mv_fit(hesspack='pracma', hessianCtrl={'h': 0.1}).vvc
    narrow = mv_fit(hesspack='pracma').vvc
    assert wide is not None
    assert narrow is not None
    assert not numpy.allclose(wide, narrow, rtol=1e-6)


def test_mv_unknown_hesspack() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'hesspack' should be one of"):
        mv_fit(hesspack='nope')


# -- numerical Hessians -----------------------------------------------------------------------------------


def analytic() -> tuple[typing.Callable[[numpy.ndarray], float], numpy.ndarray, numpy.ndarray]:
    def f(x: numpy.ndarray) -> float:
        return float(numpy.exp(x[0]) * numpy.sin(x[1]) + x[0] * x[1] ** 2 + x[2] ** 3)

    x = numpy.array([0.3, 0.7, 1.1])
    e, s, c = math.exp(0.3), math.sin(0.7), math.cos(0.7)
    h = numpy.array([[e * s, e * c + 1.4, 0.0], [e * c + 1.4, -e * s + 0.6, 0.0], [0.0, 0.0, 6.6]])
    return f, x, h


@pytest.mark.parametrize(
    ('hesspack', 'ctrl', 'tol'), [('numDeriv', {}, 1e-9), ('pracma', {}, 1e-6), ('calculus', {}, 1e-8), ('calculus', {'accuracy': 2, 'stepsize': 1e-3}, 1e-5)]
)
def test_hessian_packages_recover_an_analytic_hessian(hesspack: str, ctrl: dict[str, typing.Any], tol: float) -> None:
    f, x, expected = analytic()
    assert numpy.max(numpy.abs(metafopy.core.numderiv.hessian_by_package(f, x, hesspack, ctrl) - expected)) < tol


def test_hessian_pracma_is_the_central_difference_formula() -> None:
    """``pracma::hessian`` for two parameters: ``(f(x-h) - 2f(x) + f(x+h)) / h^2`` and the four-point mixed difference (hand computed)."""
    h = 0.5

    def f(x: numpy.ndarray) -> float:
        return float(x[0] ** 2 * x[1] + x[1] ** 3)

    x = numpy.array([1.0, 2.0])
    got = metafopy.core.numderiv.hessian_pracma(f, x, h=h)
    assert got[0, 0] == pytest.approx(2 * x[1])
    assert got[1, 1] == pytest.approx(6 * x[1])
    assert got[0, 1] == got[1, 0] == pytest.approx(2 * x[0])


def test_hessian_single_parameter() -> None:
    for hesspack in ('numDeriv', 'pracma', 'calculus'):
        got = metafopy.core.numderiv.hessian_by_package(lambda x: float(x[0] ** 2), numpy.array([1.0]), hesspack, {})
        assert got.shape == (1, 1)
        assert got[0, 0] == pytest.approx(2.0, abs=1e-6)


def test_unknown_hessian_package() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'hesspack' should be one of"):
        metafopy.core.numderiv.hessian_by_package(lambda x: float(x[0]), numpy.array([1.0]), 'x', {})


# -- location-scale models --------------------------------------------------------------------------------


def ls_data() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))


def ls_fit(**kwargs: typing.Any) -> metafopy.RMALSResult:
    res = metafopy.rma('yi', 'vi', data=ls_data(), mods='~ ablat', scale='~ ablat', method='REML', **kwargs)
    assert isinstance(res, metafopy.RMALSResult)
    return res


@pytest.mark.parametrize(
    'optimizer',
    [
        'optim',
        'nlm',
        'uobyqa',
        'newuoa',
        'bobyqa',
        'nloptr',
        'hjk',
        'nmk',
        'mads',
        'ucminf',
        'lbfgsb3c',
        'subplex',
        'BBoptim',
        'optimParallel',
        'Rcgmin',
        'Rvmmin',
        'L-BFGS-B',
        'CG',
    ],
)
def test_ls_optimizers_reach_the_default_optimum(optimizer: str) -> None:
    ref = ls_fit()
    res = ls_fit(control={'optimizer': optimizer})
    refutil.close(res.alpha, ref.alpha.tolist(), f'{optimizer} alpha', rtol=1e-4, atol=1e-5)
    refutil.close(res.beta, ref.beta.tolist(), f'{optimizer} beta', rtol=1e-4, atol=1e-5)


def test_ls_optimizer_settings_and_ncpus() -> None:
    ref = ls_fit()
    res = ls_fit(control={'optimizer': 'optim', 'optmethod': 'Nelder-Mead', 'maxit': 2000, 'reltol': 1e-12})
    refutil.close(res.alpha, ref.alpha.tolist(), 'alpha', rtol=1e-3, atol=1e-4)
    assert ls_fit(control={'ncpus': 2}).alpha[0] == pytest.approx(ref.alpha[0], rel=1e-4)


def test_ls_unknown_optimizer() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'optimizer' should be one of"):
        ls_fit(control={'optimizer': 'gradient-descent'})


def test_ls_log_link_refuses_the_barrier_optimizers() -> None:
    for optimizer in ('constrOptim', 'constrOptim.nl', 'auglag', 'alabama'):
        with pytest.raises(metafopy.errors.InputError, match='when using a log link'):
            ls_fit(control={'optimizer': optimizer})


def test_ls_identity_link_optimizers() -> None:
    dat = ls_data()
    dat['abl'] = (dat['ablat'] - dat['ablat'].mean()) / 10

    def identity_fit(optimizer: str | None) -> metafopy.RMALSResult:
        control = None if optimizer is None else {'optimizer': optimizer}
        res = metafopy.rma('yi', 'vi', data=dat, scale='~ abl', link='identity', method='REML', control=control)
        assert isinstance(res, metafopy.RMALSResult)
        return res

    ref = identity_fit(None)
    for optimizer in ('constrOptim', 'solnp', 'nloptr', 'constrOptim.nl', 'auglag', 'alabama'):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            res = identity_fit(optimizer)
        refutil.close(res.alpha, ref.alpha.tolist(), optimizer, rtol=1e-3, atol=1e-3)
    for optimizer in ('BFGS', 'optim', 'bobyqa'):
        with pytest.warns(UserWarning, match="Can only use optimizers 'constrOptim', 'solnp', 'nloptr', 'constrOptim.nl', or 'auglag'"):
            res = identity_fit(optimizer)
        assert res.alpha[0] == pytest.approx(ref.alpha[0], rel=1e-9)


def test_ls_confint_control_accepts_both_spellings() -> None:
    res = ls_fit()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dotted = metafopy.confint(res, alpha=1, control={'vc.min': -0.5, 'vc.max': 0.1})
        underscored = metafopy.confint(res, alpha=1, control={'vc_min': -0.5, 'vc_max': 0.1})
    assert isinstance(dotted, metafopy.ConfintResult)
    assert isinstance(underscored, metafopy.ConfintResult)
    assert dotted.random is not None
    assert underscored.random is not None
    refutil.close(dotted.random.to_numpy().ravel(), underscored.random.to_numpy().ravel().tolist(), 'ci', rtol=1e-12)
    # the search interval limits the bounds: the upper bound cannot be found beyond vc.max
    assert dotted.ub_sign in ('', '>')
    upper = float(dotted.random.to_numpy()[0, 2])
    assert upper <= 0.1 + 1e-12 or math.isnan(upper)


def test_ls_confint_control_changes_the_search_interval() -> None:
    res = ls_fit()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        narrow = metafopy.confint(res, alpha=1, control={'vc.max': 0.0})
    assert isinstance(narrow, metafopy.ConfintResult)
    assert narrow.random is not None
    assert float(narrow.random.to_numpy()[0, 2]) <= 1e-12
    assert narrow.ub_sign == '>'


def test_ls_confint_unknown_control_entry() -> None:
    res = ls_fit()
    with pytest.raises(metafopy.errors.InputError, match='unknown control parameter'):
        metafopy.confint(res, alpha=1, control={'vc.mni': 0.0})
    with pytest.raises(metafopy.errors.InputError, match='unknown control parameter'):
        metafopy.confint(res, control={'bogus': 1})


# -- rma_glmm(control=) -----------------------------------------------------------------------------------


def bcg_glmm(**kwargs: typing.Any) -> metafopy.GLMMResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'), measure='OR', **kwargs)


def test_glmm_or_mods_has_no_lme4_hessian_warning() -> None:
    """A variance estimated as zero puts a parameter on its bound; the fall back to the covariance matrix from RX is silent (R gives no warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'), measure='OR', mods='~ ablat')
    assert not [w for w in caught if 'falling back' in str(w.message)]


def test_glmm_hesspack_and_hessianctrl() -> None:
    base = bcg_glmm(model='CM.EL', method='FE')
    for control in ({'hesspack': 'pracma'}, {'hesspack': 'calculus'}, {'hessianCtrl': {'r': 8}}, {'hesspack': 'numDeriv', 'hessianCtrl': {'d': 0.05, 'r': 10}}):
        res = bcg_glmm(model='CM.EL', method='FE', control=control)
        refutil.close(res.beta, base.beta.tolist(), f'{control} beta', rtol=1e-9)
        refutil.close(res.se, base.se.tolist(), f'{control} se', rtol=1e-2)


def test_glmm_hessianctrl_changes_the_hessian() -> None:
    narrow = bcg_glmm(model='CM.EL', method='FE', control={'hesspack': 'pracma', 'hessianCtrl': {'h': 1e-2}})
    default = bcg_glmm(model='CM.EL', method='FE', control={'hesspack': 'pracma'})
    assert not numpy.allclose(narrow.se, default.se, rtol=1e-8)


def test_glmm_unknown_dnchgcalc_and_hesspack() -> None:
    with pytest.raises(metafopy.errors.InputError, match=r'Unknown dnchgcalc method specified\.'):
        bcg_glmm(model='CM.EL', method='FE', control={'dnchgcalc': 'x'})
    with pytest.raises(metafopy.errors.InputError, match="'hesspack' should be one of"):
        bcg_glmm(model='CM.EL', method='FE', control={'hesspack': 'x'})


def test_glmm_dnchg_choices() -> None:
    base = bcg_glmm(model='CM.EL', method='FE')
    for control in ({'dnchgcalc': 'dnoncenhypergeom'}, {'dnchgcalc': 'dFNCHypergeo'}, {'dnchgprec': 1e-12}, {'dnchgcalc': 'dFNCHypergeo', 'dnchgprec': 1e-7}):
        res = bcg_glmm(model='CM.EL', method='FE', control=control)
        refutil.close(res.beta, base.beta.tolist(), f'{control} beta', rtol=1e-9)
        # the standard errors of the exact-likelihood model carry finite-difference noise (BiasedUrn's truncation at the given precision)
        refutil.close(res.se, base.se.tolist(), f'{control} se', rtol=5e-2)


@pytest.mark.parametrize(
    ('draws', 'm', 'total', 'x', 'odds'),
    [(7, 40, 120, 3, 2.5), (30, 200, 3000, 4, 0.3), (400, 1500, 3000, 210, 1.7), (12, 12, 40, 12, 1e6), (5, 900, 1500, 0, 1e-6), (60, 4000, 9000, 25, 0.8)],
)
def test_fisher_probability_matches_the_exact_distribution(draws: int, m: int, total: int, x: int, odds: float) -> None:
    """``BiasedUrn::dFNCHypergeo``'s truncation at precision 1e-10 changes the probabilities by less than about 1e-9 (SciPy's exact pmf)."""
    exact = scipy.stats.nchypergeom_fisher.logpmf(x, total, m, draws, odds)
    got = metafopy.core.nchg.fisher_log_probability(draws, m, total, x, odds, 1e-10)
    assert got == pytest.approx(float(exact), abs=1e-8)


def test_fisher_probability_truncation_follows_precision() -> None:
    """A coarse precision cuts the tails earlier, so the normalising constant is smaller and the probabilities larger."""
    fine = metafopy.core.nchg.fisher_log_probability(400, 1500, 3000, 210, 1.7, 1e-12)
    coarse = metafopy.core.nchg.fisher_log_probability(400, 1500, 3000, 210, 1.7, 1e-3)
    assert coarse > fine
    assert coarse - fine < 1e-2


def test_fisher_probability_edge_cases() -> None:
    assert metafopy.core.nchg.fisher_log_probability(0, 5, 10, 0, 2.0, 1e-10) == 0.0
    assert metafopy.core.nchg.fisher_log_probability(3, 5, 10, 9, 2.0, 1e-10) == -math.inf
    expected = scipy.stats.hypergeom.logpmf(2, 10, 5, 3)
    assert metafopy.core.nchg.fisher_log_probability(3, 5, 10, 2, 1.0, 1e-10) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize('optimizer', ['hjk', 'bobyqa', 'nmk', 'subplex'])
def test_glmm_derivative_free_optimizers_reach_the_exact_optimum(optimizer: str) -> None:
    """R's ``hjk``, ``nmk``, ``mads``, ``subplex`` and ``BBoptim`` all reach tau^2 = 0.69346 on nielweise2007; SciPy's default tolerances stopped at 0.7013."""
    res = metafopy.rma_glmm(
        ai='ai', n1i='n1i', ci='ci', n2i='n2i', measure='OR', model='CM.EL', data=metafopy.datasets.load('nielweise2007'), control={'optimizer': optimizer}
    )
    ref = GLMM['el_nw']
    assert res.tau2 == pytest.approx(ref['tau2'], rel=1e-4)
    refutil.close(res.beta, ref['beta'], 'beta', rtol=1e-4)
