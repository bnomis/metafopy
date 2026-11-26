"""Location-scale models (``rma_uni(..., scale=)``) checked against R metafor 5.2-1 (fixtures from tests/r/generate_location_scale.R)."""

import collections.abc
import dataclasses
import json
import math
import pathlib
import re
import types
import typing
import warnings

import correctness.cases
import correctness.formulas
import hypothesis
import hypothesis.strategies
import numpy
import pandas
import pytest
import refutil
import rstrategies
import scipy.stats

import metafopy
import metafopy.core.labelled
import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.ls_anova
import metafopy.methods.ls_permutest
import metafopy.methods.ls_predict
import metafopy.methods.ls_profile
import metafopy.methods.ls_vif
import metafopy.models.rma_ls_fit
import metafopy.models.rma_ls_result
import metafopy.printing

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase5' / 'location_scale.json').read_text(encoding='utf-8'))

CLOSED = 1e-6
OPT = 1e-4
OPT_DEFAULT = OPT
# the scale part of bcg_ablat is only identified to about 1e-4, so permuted alpha statistics move by ~1.4e-4 with the BLAS (seen on Windows CI)
PERM_ALPHA = 5e-4

type Fit = collections.abc.Callable[[], metafopy.RMALSResult]

MODS3 = '~ ni100 + meta'
SCALE3 = '~ ni100 + imag'


any_permutest: collections.abc.Callable[..., typing.Any] = metafopy.permutest
any_profile: collections.abc.Callable[..., typing.Any] = metafopy.profile
any_vif: collections.abc.Callable[..., typing.Any] = metafopy.vif


def flat(value: typing.Any) -> list[float]:
    """A (possibly nested, possibly unboxed) JSON matrix as a flat float array; null is NaN."""
    return typing.cast(list[float], numpy.array([numpy.nan if v is None else v for v in numpy.array(value, dtype=object).ravel()], dtype=float).tolist())


def frame(name: str) -> pandas.DataFrame:
    return pandas.DataFrame(REF[name])


def bd() -> pandas.DataFrame:
    return frame('bd_data')


def bd_na() -> pandas.DataFrame:
    d = bd()
    d.loc[[2, 9], 'imag'] = numpy.nan
    d.loc[19, 'ni100'] = numpy.nan
    return d


def bcg() -> pandas.DataFrame:
    return frame('bcg_data')


def bcg_perm() -> pandas.DataFrame:
    """``dat.bcg`` with ``lat`` and its permutation 433 of the bcglat exact permutest (``perm``), as ``generate_location_scale.R`` builds them."""
    d = bcg()
    d['lat'] = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    d['perm'] = [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    return d


def bcg_perm18() -> pandas.DataFrame:
    """``dat.bcg`` with ``lat`` and its permutation 18 of the bcglat exact permutest (``perm``), as ``generate_location_scale.R`` builds them."""
    d = bcg()
    d['lat'] = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    d['perm'] = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    return d


def bd6() -> pandas.DataFrame:
    return frame('bd6_data')


def mo() -> pandas.DataFrame:
    return frame('mo_data')


def mc() -> pandas.DataFrame:
    return frame('mc_data')


def ls(data: pandas.DataFrame, mods: typing.Any = None, scale: typing.Any = '~ 1', **kwargs: typing.Any) -> metafopy.RMALSResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', mods=mods, scale=scale, data=data, **kwargs)
    assert isinstance(res, metafopy.RMALSResult)
    return res


def bdz() -> numpy.typing.NDArray[numpy.float64]:
    d = bd()
    return numpy.column_stack([numpy.ones(len(d)), d['ni100'], d['imag']])


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FITS: collections.abc.Mapping[str, Fit] = types.MappingProxyType(
    {
        'bd_int': lambda: ls(bd()),
        'bd_int_ml': lambda: ls(bd(), method='ML'),
        'bd_int_identity': lambda: ls(bd(), link='identity'),
        'bd_ni': lambda: ls(bd(), '~ ni100', '~ ni100'),
        'bd_ni_ml': lambda: ls(bd(), '~ ni100', '~ ni100', method='ML'),
        'bd_3': lambda: ls(bd(), MODS3, SCALE3),
        'bd_3_ml': lambda: ls(bd(), MODS3, SCALE3, method='ML'),
        'bd_3_t': lambda: ls(bd(), MODS3, SCALE3, test='t'),
        'bd_3_knha': lambda: ls(bd(), MODS3, SCALE3, test='knha'),
        'bd_3_adhoc': lambda: ls(bd(), MODS3, SCALE3, test='adhoc'),
        'bd_3_btt': lambda: ls(bd(), MODS3, SCALE3, btt=[2], att=[1, 2]),
        'bd_3_level': lambda: ls(bd(), MODS3, SCALE3, level=90),
        'bd_3_unweighted': lambda: ls(bd(), MODS3, SCALE3, weighted=False),
        'bd_3_weights': lambda: ls(bd(), MODS3, SCALE3, weights=bd()['ni'].to_numpy()),
        'bd_3_alpha': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, numpy.nan]),
        'bd_3_alpha_ml': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, numpy.nan], method='ML'),
        'bd_3_alpha_ml_all': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, 0], method='ML'),
        'bd_optbeta': lambda: ls(bd(), MODS3, SCALE3, method='ML', optbeta=True),
        'bd_optbeta_reml': lambda: ls(bd(), '~ ni100', '~ ni100', optbeta=True),
        'bd_optbeta_beta': lambda: ls(bd(), MODS3, '~ 1', optbeta=True, beta=[numpy.nan, numpy.nan, 0.1], method='ML'),
        'bd_identity': lambda: ls(bd(), '~ ni100', '~ imag', link='identity'),
        'bd_identity_ml': lambda: ls(bd(), '~ ni100', '~ imag', link='identity', method='ML'),
        'bd_identity_ni': lambda: ls(bd(), MODS3, '~ meta + imag', link='identity'),
        'bd_na': lambda: ls(bd_na(), MODS3, '~ imag + ni100'),
        'bd_matrix': lambda: ls(bd(), numpy.column_stack([bd()['ni100']]), bdz()),
        'bd_poly': lambda: ls(bd(), '~ ni100', '~ ni100 + I(ni100**2)'),
        'bd_alpha_bounds': lambda: ls(bd(), '~ ni100', '~ ni100', control=metafopy.LSControl(alpha_min=[-5, -3], alpha_max=[0, 3])),
        'bd_alpha_init': lambda: ls(bd(), '~ ni100', '~ ni100', control=metafopy.LSControl(alpha_init=[-2, -0.5])),
        'bd_noscaleZ': lambda: ls(bd(), '~ ni100', SCALE3, control=metafopy.LSControl(scaleZ=False)),
        'bcg_ablat': lambda: ls(bcg(), '~ ablat', '~ ablat'),
        'bcg_ablat_ml': lambda: ls(bcg(), '~ ablat', '~ ablat', method='ML'),
        'bcg_ablat_t': lambda: ls(bcg(), '~ ablat', '~ ablat', test='t'),
        'bcg_int_scale': lambda: ls(bcg(), '~ ablat', '~ 1'),
        'bcg_year_identity': lambda: ls(bcg(), '~ ablat', '~ I(ablat/10)', link='identity', method='ML'),
        # R stops at a lower local maximum here than the restarts reach (spec location-scale-fits-reach-the-higher-maximum), so the port of
        # metafor's single start is checked with restarts=False; test_rma_ls_restarts.py and a correctness case check the restart
        'bcg_perm433': lambda: ls(bcg_perm(), '~ perm', '~ lat', method='ML', control={'restarts': False}),
        # likewise for link='identity' (spec identity-link-location-scale-fits-reach-the-higher-maximum): R stops at its default start, below the face maximum
        'bcg_perm18_identity': lambda: ls(bcg_perm18(), '~ perm', '~ lat', link='identity', method='ML', control={'restarts': False}),
        'bd6_exact': lambda: ls(bd6(), '~ ni100', '~ ni100'),
        'mo_int': lambda: ls(mo()),
        'mo_quality': lambda: ls(mo(), '~ quality', '~ quality', method='ML'),
        'mo_design': lambda: ls(mo(), None, '~ design'),
        'mc_factors': lambda: ls(mc(), '~ type + struct', '~ struct', method='ML'),
    }
)

# The scale coefficients are optimizer output (R: nlminb, here BFGS plus a polish): they agree to about 1e-7, the quantities derived from
# them (location estimates, Hessian-based standard errors) to 1e-4. The log-likelihood at the optimum is closed-form given the optimum.
# Fits whose likelihood is flat in the scale coefficients (tau^2 -> 0 for some studies) only have their log-likelihood compared.
FLAT = {'mo_design'}
# the scale part of these fits is identified to about 1e-3 (the likelihood is flat in alpha), which BLAS differences expose
LOOSE_FIT = {'bcg_ablat', 'bcg_ablat_t'}
# The fits with link='identity': metafor stops about 1e-5 short of the maximum (its constrOptim), metafopy polishes to it, so the values that follow
# from the scale coefficients differ from the fixtures (docs/differences.md, "Location-scale fits with link=\"identity\""). The fixtures stay as R wrote
# them; test_identity_fit_is_the_maximum compares each fit with the independent maximisation of correctness.formulas, and test_fit allows
# IDENTITY_FIT for the rest. A fit whose scale model puts tau^2_i near zero shifts most (bd_identity_ni).
IDENTITY_FITS = {'bcg_year_identity', 'bd_identity', 'bd_identity_ml', 'bd_identity_ni', 'bd_int_identity'}
IDENTITY_FIT = 5e-2
# restarts=False identity fits: the port of metafor's single start, which stops short of the maximum like the fits above but is no maximum itself
IDENTITY_SINGLE_START = {'bcg_perm18_identity'}
STOPS_SHORT = IDENTITY_FITS | IDENTITY_SINGLE_START
IDENTITY_FIT_STATS = 1e-3  # the log-likelihood of the criterion that was not maximised moves with alpha
# column names that come from formulaic's rendering of I() terms or from an unnamed matrix, not from the algorithm
UNNAMED = {'bcg_year_identity', 'bd_matrix', 'bd_poly'}


@pytest.mark.parametrize('name', sorted(set(FITS) - FLAT))
def test_fit(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    OPT = 1e-3 if name in LOOSE_FIT else IDENTITY_FIT if name in STOPS_SHORT else OPT_DEFAULT
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=OPT, atol=1e-7)
    refutil.close(res.alpha, ref['alpha'], f'{name} alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.se, ref['se'], f'{name} se', rtol=OPT, atol=1e-7)
    refutil.close(res.vb.ravel(), flat(ref['vb']), f'{name} vb', rtol=OPT, atol=1e-8)
    refutil.close(res.va.ravel(), flat(ref['va']), f'{name} va', rtol=OPT, atol=1e-6)
    # a tau^2_i on the boundary is R's barrier value (about 1e-8), here 0
    refutil.close(res.tau2, ref['tau2'], f'{name} tau2', rtol=OPT, atol=1e-7 if name in STOPS_SHORT else 1e-8)
    refutil.close(res.pval, ref['pval'], f'{name} pval', rtol=OPT, atol=1e-7)
    refutil.close(res.ci_lb, ref['ci_lb'], f'{name} ci_lb', rtol=OPT, atol=1e-6)
    refutil.close(res.ci_ub, ref['ci_ub'], f'{name} ci_ub', rtol=OPT, atol=1e-6)
    closed = IDENTITY_FIT_STATS if name in STOPS_SHORT else CLOSED
    refutil.close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{name} fit ML', rtol=closed, atol=1e-6)
    refutil.close(res.fit_stats_reml.as_tuple(), ref['fit_reml'], f'{name} fit REML', rtol=closed, atol=1e-6)
    refutil.close(res.QE, ref['QE'], f'{name} QE', rtol=1e-8)
    refutil.close(res.QM, ref['QM'], f'{name} QM', rtol=OPT, atol=1e-7)
    refutil.close(res.QMp, ref['QMp'], f'{name} QMp', rtol=OPT, atol=1e-7)
    refutil.close(res.vt, ref['vt'], f'{name} vt', rtol=1e-8)
    refutil.close(res.I2, ref['I2'], f'{name} I2', rtol=OPT, atol=1e-4 if name in STOPS_SHORT else 1e-6)
    refutil.close(res.H2, ref['H2'], f'{name} H2', rtol=OPT, atol=1e-4 if name in STOPS_SHORT else 1e-6)
    refutil.close(res.se_alpha, ref['se_alpha'], f'{name} se_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.zval_alpha, ref['zval_alpha'], f'{name} zval_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.pval_alpha, ref['pval_alpha'], f'{name} pval_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.QS, ref['QS'], f'{name} QS', rtol=OPT, atol=1e-6)
    refutil.close(res.QSp, ref['QSp'], f'{name} QSp', rtol=OPT, atol=1e-6)
    assert (res.k, res.p, res.q, res.m, res.m_alpha, res.parms) == (ref['k'], ref['p'], ref['q'], ref['m'], ref['m_alpha'], ref['parms'])
    assert (res.method, res.link, res.test) == (ref['method'], ref['link'], ref['test'])
    if name not in UNNAMED:
        assert list(res.coef_names) == refutil.as_list(ref['coef_names'])
    if name not in UNNAMED:
        assert list(res.alpha_names) == refutil.as_list(ref['alpha_names'])
    refutil.close(res.hatvalues(), ref['hat'], f'{name} hat', rtol=OPT, atol=1e-7)
    refutil.close(res.weights_(), ref['weights'], f'{name} weights', rtol=OPT, atol=1e-6)
    refutil.close(res.residuals(), ref['residuals'], f'{name} residuals', rtol=OPT, atol=1e-6)
    fitted = res.fitted()
    refutil.close(fitted['location'], ref['fitted_location'], f'{name} fitted location', rtol=OPT, atol=1e-6)
    refutil.close(fitted['scale'], ref['fitted_scale'], f'{name} fitted scale', rtol=OPT, atol=1e-5)


@pytest.mark.parametrize('name', sorted(IDENTITY_FITS))
def test_identity_fit_is_the_maximum(name: str) -> None:
    """The identity-link fit is the maximum within ``Z alpha >= 0``, which R's fixture is not: the reference's ``alpha``, R's log-likelihood or higher."""
    ref = REF[name]
    res = FITS[name]()
    reference = correctness.formulas.ls_reference_fit(res.yi, res.vi, res.X, res.Z, 'identity', res.method)
    numpy.testing.assert_allclose(res.alpha, numpy.ravel(reference['alpha']), rtol=1e-6, atol=1e-6, err_msg=f'{name} alpha')
    assert not numpy.allclose(ref['alpha'], numpy.ravel(reference['alpha']), rtol=1e-6, atol=1e-6)
    fit_stats, key = (res.fit_stats_reml, 'fit_reml') if res.method == 'REML' else (res.fit_stats_ml, 'fit_ml')
    assert fit_stats.ll >= ref[key][0] - 1e-9
    assert numpy.min(res.Z @ res.alpha) >= 0


def identity_problem(res: metafopy.RMALSResult) -> metafopy.models.rma_ls_fit.LSProblem:
    q = res.Z.shape[1]
    return metafopy.models.rma_ls_fit.LSProblem(
        yi=res.yi,
        vi=res.vi,
        X=res.X,
        Z=res.Z,
        reml=res.method == 'REML',
        REMLf=True,
        link='identity',
        alpha_min=numpy.full(q, -numpy.inf),
        alpha_max=numpy.full(q, numpy.inf),
        tau2_min=0.0,
        tau2_max=numpy.inf,
        optbeta=False,
    )


@pytest.mark.parametrize('method', metafopy.models.rma_ls_fit.LS_METHODS)
def test_identity_score_and_information(method: str) -> None:
    """The score is the gradient of the log-likelihood (the independent ``formulas.ls_score``) and the information its expectation's negative."""
    res = ls(bd(), '~ imag', '~ imag', link='identity', method=method)
    problem = identity_problem(res)
    alpha = numpy.array([0.05, 0.2])
    score = problem.ls_score(alpha)
    numpy.testing.assert_allclose(score, correctness.formulas.ls_score(res.yi, res.vi, res.X, res.Z, alpha, 'identity', method), rtol=1e-8)
    step = 1e-6
    numeric = [
        (
            problem.loglik(alpha + step * e, numpy.full(2, numpy.nan), numpy.full(2, numpy.nan), alpha_transf=False)
            - problem.loglik(alpha - step * e, numpy.full(2, numpy.nan), numpy.full(2, numpy.nan), alpha_transf=False)
        )
        / (2 * step)
        for e in numpy.eye(2)
    ]
    numpy.testing.assert_allclose(score, numeric, rtol=1e-5)
    information = problem.ls_information(alpha)
    assert numpy.all(numpy.linalg.eigvalsh(information) > 0)
    numpy.testing.assert_allclose(information, information.T)


def test_identity_fit_bd_is_a_stationary_point() -> None:
    """On ``dat.bangertdrowns2004`` the optimum is interior: the score is zero and the log-likelihood is the maximum's, which ``constrOptim`` misses by 7e-8."""
    res = ls(bd(), '~ imag', '~ imag', link='identity', method='ML')
    score = correctness.formulas.ls_score(res.yi, res.vi, res.X, res.Z, res.alpha, 'identity', 'ML')
    assert numpy.max(numpy.abs(score)) < 1e-6
    assert res.fit_stats_ml.ll == pytest.approx(-17.058054672888, abs=1e-9)


def test_profile_refit_keeps_the_highest_likelihood_on_bcg() -> None:
    """On the ML fit of ``dat.bcg`` one start jumps to ``tau2 = 0`` at a = 1.1 (``2 (ll_max - ll)`` 3.86864); the best of the starts stays on the branch."""
    res = ls(bcg(), '~ ablat', '~ ablat', method='ML')
    fit = metafopy.methods.ls_profile.profile_fit(res, 0, 1.1)
    assert fit is not None
    assert 2 * (res.logLik() - fit[0]) == pytest.approx(3.13614, abs=1e-4)
    table = metafopy.confint(res, alpha=0, tol=1e-12).random
    assert table is not None
    assert table['ci.ub'].iloc[0] == pytest.approx(1.803009, abs=1e-5)


def test_profile_keeps_the_continuous_branch_of_alpha2_on_bcg() -> None:
    """Every start of the line ends on the ``tau2 = 0`` plateau (3.868641) from a = 1.36 on.

    The continuous branch, whose optimum has ``alpha.1`` near -74, stays below the cut-off until 1.3812.
    """
    res = ls(bcg(), '~ ablat', '~ ablat', method='ML')
    for val, expected in ((1.36, 3.789459), (1.37, 3.813933), (1.38, 3.838488)):
        fit = metafopy.methods.ls_profile.profile_fit(res, 1, val)
        assert fit is not None
        assert 2 * (res.logLik() - fit[0]) == pytest.approx(expected, abs=1e-5)
    ci = metafopy.confint(res, alpha=1, tol=1e-12)
    assert ci.random is not None
    assert ci.random['ci.ub'].iloc[0] == pytest.approx(1.3812075, abs=1e-6)
    assert ci.ub_sign == ''


def test_confint_identity_lower_bound_is_a_crossing() -> None:
    """The brute-force profile (a one-dimensional search over the feasible interval) crosses the cut-off at the lower bound.

    Before the new starts every line start left a ``tau2_i`` negative and was dropped, so the bound was a ``<`` at -0.0820267.
    """
    res = result('bcg_year_identity')
    ci = metafopy.confint(res, alpha=1, tol=1e-12)
    assert ci.random is not None
    assert ci.random['ci.lb'].iloc[0] == pytest.approx(-0.1356308, abs=1e-6)
    assert ci.lb_sign == ''


def test_feasible_start_gives_the_missing_identity_lower_bound_on_bcg() -> None:
    """Below the estimate every start but the default leaves a ``tau2_i`` negative; the feasible start gives the refit a place to begin."""
    res = ls(bcg(), '~ ablat', '~ ablat', link='identity', method='ML')
    start = metafopy.methods.ls_profile.feasible_start(res, 0, -0.2143)
    assert start is not None
    assert numpy.min(res.Z @ start) > 0
    assert any(s is not None and numpy.array_equal(s, start) for s in metafopy.methods.ls_profile.profile_starts(res, 0, -0.2143))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ci = metafopy.confint(res, alpha=0, tol=1e-12)
    assert ci.random is not None
    assert ci.random['ci.lb'].iloc[0] == pytest.approx(-0.214322988803, abs=1e-8)
    assert not any('Cannot obtain lower bound' in str(w.message) for w in caught)


def test_profile_starts_skip_the_feasible_start_when_a_line_start_is_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    res = ls(bcg(), '~ ablat', '~ ablat', link='identity', method='ML')
    before = metafopy.methods.ls_profile.profile_starts(res, 0, 0.1)

    def refuse(*args: typing.Any) -> None:
        raise AssertionError('feasible_start called')

    monkeypatch.setattr(metafopy.methods.ls_profile, 'feasible_start', refuse)
    after = metafopy.methods.ls_profile.profile_starts(res, 0, 0.1)
    assert len(before) == len(after)
    assert all((a is None and b is None) or (a is not None and b is not None and numpy.array_equal(a, b)) for a, b in zip(before, after, strict=True))


def test_identity_fit_bcg_is_a_constrained_maximum() -> None:
    """On ``dat.bcg`` the optimum is on the boundary: ``Z alpha >= 0``, a log-likelihood above R's, and the score zero on the free directions."""
    res = ls(bcg(), '~ ablat', '~ ablat', link='identity', method='ML')
    tau2 = res.Z @ res.alpha
    assert numpy.min(tau2) >= 0
    assert res.fit_stats_ml.ll >= -7.139288984088 + 1e-7
    active = tau2 <= 1e-8
    assert active.any()
    _, sv, vt = numpy.linalg.svd(res.Z[active])
    free_directions = vt[int(numpy.sum(sv > 1e-10 * sv[0])) :].T
    score = correctness.formulas.ls_score(res.yi, res.vi, res.X, res.Z, res.alpha, 'identity', 'ML')
    assert numpy.max(numpy.abs(free_directions.T @ score)) < 1e-6


@pytest.mark.parametrize('name', sorted(FLAT))
def test_fit_flat_likelihood(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    refutil.close(res.fit_stats_reml.ll, ref['fit_reml'][0], f'{name} ll', rtol=1e-6)
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=1e-4, atol=1e-6)


def test_print_names_a_single_location_coefficient() -> None:
    """R drops the row name only for an intercept-only part (audit cases ``construct/ls_*/mods=~ ablat - 1``); R 4.6.2, metafor 5.2-1."""
    lines = [line.rstrip() for line in str(ls(bcg(), mods='~ ablat - 1', scale='~ alloc')).splitlines()]
    assert 'ablat   -0.0189  0.0017  -11.0003  <.0001  -0.0223  -0.0155  ***' in lines
    lines = [line.rstrip() for line in str(ls(bcg(), mods='~ ablat', scale='~ 1')).splitlines()]
    assert ' -2.5725  1.0241  -2.5119  0.0120  -4.5797  -0.5652  *' in lines


def test_print_wraps_a_wide_table_as_r(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank column of significance stars is one character wide in R, so the second block of a wide table keeps it (audit case
    ``construct/ls_bcg/mods=~ relevel(...)``); R 4.6.2, metafor 5.2-1, width 80."""
    monkeypatch.setitem(metafopy.printing.SETTINGS, 'width', 80)
    res = ls(bcg(), mods="~ relevel(factor(alloc), ref='systematic')", scale='~ alloc')
    text = str(res)
    location = text[text.index('Model Results (Location):') : text.index('Test of Scale Coefficients')]
    assert [line.rstrip() for line in location.splitlines() if line.strip()] == [
        'Model Results (Location):',
        '                                                     estimate      se     zval',
        'intrcpt                                               -0.4242  0.3597  -1.1792',
        'relevel(factor(alloc), ref = "systematic")alternate   -0.1166  0.4569  -0.2552',
        'relevel(factor(alloc), ref = "systematic")random      -0.5468  0.4534  -1.2059',
        '                                                       pval    ci.lb   ci.ub',
        'intrcpt                                              0.2383  -1.1293  0.2809',
        'relevel(factor(alloc), ref = "systematic")alternate  0.7986  -1.0120  0.7788',
        'relevel(factor(alloc), ref = "systematic")random     0.2279  -1.4354  0.3419',
    ]


@pytest.mark.parametrize('num', [False, True])
def test_print_one_row_scale_table_without_a_name(num: bool) -> None:
    """R's ``print.rma.uni`` prints any one-row scale table as a vector (``if (nrow(res.table) == 1L)``), not only an intercept-only
    one (audit cases ``construct/*/scale=~ 0 + ablat``); R 4.6.2, metafor 5.2-1."""
    res = ls(bcg(), scale='~ 0 + ablat')
    text = res.summary(num=True) if num else str(res)
    scale = text[text.index('Model Results (Scale):') : text.index('---')]
    assert [line.rstrip() for line in scale.splitlines() if line.strip()] == [
        'Model Results (Scale):',
        'estimate      se     zval    pval    ci.lb    ci.ub',
        ' -0.0338  0.0167  -2.0262  0.0427  -0.0664  -0.0011  *',
    ]


def test_predict_names_rows_by_a_named_newscale() -> None:
    """With one scale predictor a vector ``newscale`` holds one value per prediction, and R names each row by its element (audit case
    ``options/predict/ls_na/newscale={"$named":{"ablat":30,"year":1960}}``); R 4.6.2, metafor 5.2-1:
    ``predict(rma(yi, vi, mods = ~ year, scale = ~ ablat, data = dat), newscale = c(ablat = 30, year = 1960))``."""
    res = ls(bcg(), mods='~ year', scale='~ ablat')
    newscale = {'ablat': 30, 'year': 1960}
    out = metafopy.predict(res, newscale=newscale)
    assert out.slab is not None
    assert list(out.slab) == list(newscale)
    assert out.pred == pytest.approx([-4.4395, 375.2869], abs=2e-3)


@pytest.mark.parametrize('name', ['bd_int', 'bd_ni', 'bd_3', 'bd_3_t', 'bd_3_alpha', 'bd_na', 'bd_optbeta'])
def test_print(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    assert refutil.r_print(str(res)) == refutil.r_print(ref['print'])
    assert refutil.r_print(res.summary(showfit=True)) == refutil.r_print(ref['print_showfit'])


# -- predict ----------------------------------------------------------------------------------------------------------------

M2 = numpy.array([[0.5, 0], [1.5, 1]])
RESULTS: dict[str, metafopy.RMALSResult] = {}


def result(name: str) -> metafopy.RMALSResult:
    if name not in RESULTS:
        RESULTS[name] = FITS[name]()
    return RESULTS[name]


def check_predict(pr: metafopy.PredictResult, ref: dict[str, typing.Any], what: str, rtol: float = OPT) -> None:
    refutil.close(pr.pred, ref['pred'], f'{what} pred', rtol=rtol, atol=1e-7)
    refutil.close(pr.se, ref['se'], f'{what} se', rtol=rtol, atol=1e-7)
    refutil.close(pr.ci_lb, ref['ci_lb'], f'{what} ci_lb', rtol=rtol, atol=1e-7)
    refutil.close(pr.ci_ub, ref['ci_ub'], f'{what} ci_ub', rtol=rtol, atol=1e-7)
    if ref.get('pi_lb') is not None:
        refutil.close(pr.pi_lb, ref['pi_lb'], f'{what} pi_lb', rtol=rtol, atol=1e-7)
        refutil.close(pr.pi_ub, ref['pi_ub'], f'{what} pi_ub', rtol=rtol, atol=1e-7)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PREDICT_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'bd_int:predict_newscale_exp': ('bd_int', 'predict_newscale_exp', {'newscale': 1, 'transf': numpy.exp}),
    'bd_int:predict_newscale': ('bd_int', 'predict_newscale', {'newscale': 1}),
    'bd_int:predict_default': ('bd_int', 'predict_default', {}),
    'bd_int:predict_newmods': ('bd_int', 'predict_newmods', {'newmods': 1}),
    'bd_3:predict_default': ('bd_3', 'predict_default', {}),
    'bd_3:predict_addxz': ('bd_3', 'predict_addxz', {'addx': True, 'addz': True}),
    'bd_3:predict_newmods': ('bd_3', 'predict_newmods', {'newmods': numpy.array([[0.5, 0], [1.5, 1], [2.5, 1]])}),
    'bd_3:predict_newmods_noint': ('bd_3', 'predict_newmods_noint', {'newmods': [1, 1], 'intercept': False}),
    'bd_3:predict_newscale': ('bd_3', 'predict_newscale', {'newscale': M2}),
    'bd_3:predict_newscale_exp': ('bd_3', 'predict_newscale_exp', {'newscale': M2, 'transf': numpy.exp}),
    'bd_3:predict_both': ('bd_3', 'predict_both', {'newmods': M2, 'newscale': M2}),
    'bd_3:predict_both_single': ('bd_3', 'predict_both_single', {'newmods': [1, 1], 'newscale': [1, 1]}),
    'bd_3:predict_both_t': ('bd_3', 'predict_both_t', {'newmods': M2, 'newscale': M2, 'level': 90, 'adjust': True}),
    'bd_3:predict_simple': ('bd_3', 'predict_simple', {'newmods': M2, 'newscale': M2, 'predtype': 'simple'}),
    'bd_3:predict_riley': ('bd_3', 'predict_riley', {'newmods': M2, 'newscale': M2, 'predtype': 'riley'}),
    'bd_3:predict_newvi': ('bd_3', 'predict_newvi', {'newmods': M2, 'newscale': M2, 'newvi': [0.01, 0.02]}),
    'bd_3:predict_prob': ('bd_3', 'predict_prob', {'newmods': M2, 'newscale': M2, 'prob': '>0.2'}),
    'bd_3_t:predict_default': ('bd_3_t', 'predict_default', {}),
    'bd_3_t:predict_both': ('bd_3_t', 'predict_both', {'newmods': M2, 'newscale': M2}),
    'bd_3_t:predict_newscale': ('bd_3_t', 'predict_newscale', {'newscale': M2}),
    'bd_3_t:predict_both_riley': ('bd_3_t', 'predict_both_riley', {'newmods': M2, 'newscale': M2, 'predtype': 'riley'}),
    'bd_3_t:predict_both_tt': ('bd_3_t', 'predict_both_tt', {'newmods': M2, 'newscale': M2, 'predtype': 't'}),
    'bd_identity:predict_newscale': ('bd_identity', 'predict_newscale', {'newscale': [0, 1]}),
    'bd_na:predict_default': ('bd_na', 'predict_default', {}),
    'bcg_ablat:predict_default': ('bcg_ablat', 'predict_default', {}),
    'bcg_ablat:predict_both_exp': ('bcg_ablat', 'predict_both_exp', {'newmods': [20, 40], 'newscale': [20, 40], 'transf': numpy.exp}),
    'mo_int:predict_newscale_exp': ('mo_int', 'predict_newscale_exp', {'newscale': 1, 'transf': numpy.exp}),
}


@pytest.mark.parametrize('case', sorted(PREDICT_CASES))
def test_predict(case: str) -> None:
    name, key, kwargs = PREDICT_CASES[case]
    pr = metafopy.predict(result(name), **kwargs)
    assert isinstance(pr, metafopy.methods.ls_predict.LSPredictResult)
    ref = REF[name][key]
    check_predict(pr, ref, case, IDENTITY_FIT if name in IDENTITY_FITS else OPT)
    assert pr.pred_type == ref['pred_type']
    if ref.get('pi_se') is not None and pr.pi_se is not None:
        # metafor's pi.se is not subset to the studies without missing values
        expected = [v for v in refutil.as_list(ref['pi_se']) if v is not None] if len(pr.pi_se) != len(refutil.as_list(ref['pi_se'])) else ref['pi_se']
        refutil.close(pr.pi_se, expected, f'{case} pi_se', rtol=OPT, atol=1e-7)
    if ref.get('prob') is not None:
        refutil.close(pr.prob, ref['prob'], f'{case} prob', rtol=OPT, atol=1e-7)
    if ref.get('X') is not None:
        refutil.close(numpy.asarray(pr.X).ravel(), flat(ref['X']), f'{case} X', rtol=OPT, atol=1e-8)
    if ref.get('Z') is not None:
        refutil.close(numpy.asarray(pr.Z).ravel(), flat(ref['Z']), f'{case} Z', rtol=OPT, atol=1e-8)


def test_predict_vcov() -> None:
    res = result('bd_3')
    pv = metafopy.predict(res, newmods=M2, newscale=M2, vcov=True)
    assert isinstance(pv, metafopy.methods.ls_predict.LSPredictResult)
    refutil.close(numpy.asarray(pv.vcov).ravel(), flat(REF['bd_3']['predict_vcov']['vcov']), 'vcov', rtol=OPT, atol=1e-8)
    ps = metafopy.predict(res, newscale=M2, vcov=True)
    assert isinstance(ps, metafopy.methods.ls_predict.LSPredictResult)
    refutil.close(numpy.asarray(ps.vcov).ravel(), flat(REF['bd_3']['predict_vcov_scale']['vcov']), 'vcov scale', rtol=OPT, atol=1e-8)


def test_predict_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='without moderators'):
        metafopy.predict(result('bd_int'), newmods=[1, 2])
    with pytest.raises(metafopy.errors.InputError, match='without scale variables'):
        metafopy.predict(result('bd_int'), newscale=[1, 2])
    with pytest.raises(metafopy.errors.InputError, match='do not match'):
        metafopy.predict(result('bd_3'), newmods=[1, 2, 3, 4])


# -- confint and profile ----------------------------------------------------------------------------------------------------


def check_ci(table: pandas.DataFrame, ref: dict[str, typing.Any], what: str, rtol: float = OPT, atol: float = 2e-4) -> None:
    refutil.close(table['estimate'].to_numpy(), ref['estimate'], f'{what} estimate', rtol=rtol, atol=atol)
    refutil.close(table['ci.lb'].to_numpy(), ref['ci_lb'], f'{what} ci.lb', rtol=rtol, atol=atol)
    refutil.close(table['ci.ub'].to_numpy(), ref['ci_ub'], f'{what} ci.ub', rtol=rtol, atol=atol)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
CONFINT_CASES: dict[str, tuple[str, str, int]] = {
    'bd_int': ('bd_int', 'confint_alpha', 0),
    'bd_3_t': ('bd_3_t', 'confint_alpha2', 1),
    'bd_3_alpha': ('bd_3_alpha', 'confint_alpha1', 0),
    'bd_identity': ('bd_identity', 'confint_alpha1', 0),
}


@pytest.mark.parametrize('case', sorted(CONFINT_CASES))
def test_confint_alpha(case: str) -> None:
    name, key, pos = CONFINT_CASES[case]
    ci = metafopy.confint(result(name), alpha=pos)
    assert isinstance(ci, metafopy.ConfintResult)
    assert ci.random is not None
    ref = dict(REF[name][key])
    if case == 'bd_3_alpha':
        # the lower bound is an artifact of metafor's optimizer: the refits at alpha = -19.5 and -17.0 fail to converge to the
        # (flat) optimum, so its profile-likelihood difference jumps; only the well-behaved upper bound is comparable
        ref['ci_lb'] = ci.random['ci.lb'].to_numpy().tolist()
    if case == 'bd_identity' and numpy.isnan(ci.random['ci.lb'].to_numpy()).all():
        # the lower bound sits at the boundary (0); where the refit fails to converge it is not found (macOS BLAS)
        ref['ci_lb'] = ci.random['ci.lb'].to_numpy().tolist()
    check_ci(ci.random, ref, case)


def test_confint_all() -> None:
    ci = metafopy.confint(result('bd_3'))
    assert isinstance(ci, list)
    ref = REF['bd_3']['confint_all']
    assert len(ci) == len(ref) == 3
    for j, item in enumerate(ci):
        assert item.random is not None
        check_ci(item.random, ref[str(j + 1)], f'confint alpha.{j + 1}')
        assert list(item.random.index) == [f'alpha.{j + 1}']


def test_confint_single_free_alpha_is_not_a_list() -> None:
    ci = metafopy.confint(result('bd_int'))
    assert isinstance(ci, metafopy.ConfintResult)
    assert list(ci.random.index) == ['alpha']  # type: ignore[union-attr]


def test_confint_fixed() -> None:
    ci = metafopy.confint(result('bd_int'), fixed=True)
    assert ci.fixed is not None
    check_ci(ci.fixed, REF['bd_int']['confint_fixed'], 'fixed', rtol=1e-6, atol=1e-8)
    ci3 = metafopy.confint(result('bd_3'), fixed=True, alpha=1, level=90)
    ref = REF['bd_3']['confint_fixed_alpha']
    assert ci3.fixed is not None
    assert ci3.random is not None
    refutil.close(ci3.fixed.to_numpy().ravel(), flat(ref['fixed']), 'fixed table', rtol=OPT, atol=1e-7)
    refutil.close(ci3.random.to_numpy().ravel(), ref['random'], 'random row', rtol=OPT, atol=2e-4)


def test_optbeta_with_weights_errors() -> None:
    """metafor's rma.uni stops with "Cannot use custom weights when 'optbeta=TRUE'." for a location-scale fit with user weights."""
    with pytest.raises(metafopy.errors.InputError, match='Cannot use custom weights'):
        metafopy.rma_uni('yi', 'vi', mods=MODS3, scale=SCALE3, data=bd(), weights=bd()['ni'].to_numpy(), optbeta=True)


def test_confint_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='optbeta'):
        metafopy.confint(result('bd_optbeta'))
    with pytest.raises(metafopy.errors.InputError, match='was fixed'):
        metafopy.confint(result('bd_3_alpha'), alpha=1)
    with pytest.raises(metafopy.errors.InputError, match='No such'):
        metafopy.confint(result('bd_3'), alpha=7)


def check_profile(pf: metafopy.methods.ls_profile.LSProfileResult, ref: dict[str, typing.Any], what: str) -> None:
    refutil.close(pf.alpha, ref['alpha'], f'{what} alpha', rtol=OPT, atol=1e-6)
    refutil.close(pf.ll, ref['ll'], f'{what} ll', rtol=CLOSED, atol=1e-5)
    refutil.close(pf.beta.to_numpy().ravel(), flat(ref['beta']), f'{what} beta', rtol=OPT, atol=1e-5)
    refutil.close(pf.ci_lb.to_numpy().ravel(), flat(ref['ci_lb']), f'{what} ci_lb', rtol=OPT, atol=1e-5)
    refutil.close(pf.ci_ub.to_numpy().ravel(), flat(ref['ci_ub']), f'{what} ci_ub', rtol=OPT, atol=1e-5)
    refutil.close(pf.vc, ref['vc'], f'{what} vc', rtol=OPT)
    refutil.close(pf.maxll, ref['maxll'], f'{what} maxll', rtol=CLOSED)
    refutil.close(pf.ylim, ref['ylim'], f'{what} ylim', rtol=OPT, atol=1e-5)


def test_profile() -> None:
    pf = any_profile(result('bd_int'), alpha=0, steps=12)
    assert isinstance(pf, metafopy.methods.ls_profile.LSProfileResult)
    check_profile(pf, REF['bd_int']['profile'], 'profile bd_int')
    pf3 = any_profile(result('bd_3'), alpha=1, steps=10)
    assert isinstance(pf3, metafopy.methods.ls_profile.LSProfileResult)
    check_profile(pf3, REF['bd_3']['profile'], 'profile bd_3')
    assert list(pf3.beta.columns) == list(result('bd_3').coef_names)


def test_profile_all_and_errors() -> None:
    allp = any_profile(result('bd_3'), steps=5)
    assert isinstance(allp, list)
    assert [p.position for p in allp] == [0, 1, 2]
    with pytest.raises(metafopy.errors.InputError, match='optbeta'):
        any_profile(result('bd_optbeta'))
    with pytest.raises(metafopy.errors.InputError, match='was fixed'):
        any_profile(result('bd_3_alpha'), alpha=1)
    with pytest.raises(metafopy.errors.InputError, match='steps'):
        any_profile(result('bd_3'), alpha=0, steps=1)


# -- anova ------------------------------------------------------------------------------------------------------------------

X23 = numpy.array([[0, 1, 0], [0, 0, 1]])


def check_wald(a: metafopy.methods.ls_anova.LSWaldResult, ref: dict[str, typing.Any], what: str) -> None:
    key = 'QM' if a.part == 'location' else 'QS'
    refutil.close(a.stat, ref[key], f'{what} stat', rtol=OPT, atol=1e-7)
    refutil.close(a.pval_omnibus, ref[key + 'p'], f'{what} p', rtol=OPT, atol=1e-7)
    refutil.close(a.df[0], ref[key + 'df'][0], f'{what} df')
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    for mine, theirs in (('estimate', 'Xb' if a.part == 'location' else 'Za'), ('se', 'se'), ('zval', 'zval'), ('pval', 'pval')):
        if theirs in ref:
            refutil.close(getattr(a, mine), ref[theirs], f'{what} {mine}', rtol=OPT, atol=1e-7)
    if 'print' in ref:
        assert refutil.r_print(str(a)) == refutil.r_print(ref['print'])


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
WALD_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'btt': ('bd_3', 'anova_btt', {'btt': [2]}),
    'att': ('bd_3', 'anova_att', {'att': [2]}),
    'att23': ('bd_3', 'anova_att_23', {'att': [1, 2]}),
    'X': ('bd_3', 'anova_X', {'X': X23}),
    'Z': ('bd_3', 'anova_Z', {'Z': X23, 'rhs': [0, 1]}),
    'Z1': ('bd_3', 'anova_Z1', {'Z': [0, 1, -1]}),
    't_att': ('bd_3_t', 'anova_att', {'att': [1, 2]}),
    't_Z': ('bd_3_t', 'anova_Z', {'Z': X23}),
    't_X': ('bd_3_t', 'anova_X', {'X': X23}),
}


@pytest.mark.parametrize('case', sorted(WALD_CASES))
def test_anova_wald(case: str) -> None:
    name, key, kwargs = WALD_CASES[case]
    a = metafopy.anova(result(name), **kwargs)
    assert isinstance(a, metafopy.methods.ls_anova.LSWaldResult)
    check_wald(a, REF[name][key], case)


def test_anova_wald_matches_model_tests() -> None:
    res = result('bd_3')
    a = metafopy.anova(res)
    assert isinstance(a, metafopy.methods.ls_anova.LSWaldResult)
    refutil.close(a.QM, res.QM, 'QM', rtol=1e-12)
    s = metafopy.anova(res, att=[1, 2])
    assert isinstance(s, metafopy.methods.ls_anova.LSWaldResult)
    refutil.close(s.QS, res.QS, 'QS', rtol=1e-12)
    names = metafopy.anova(res, att='ni100|imag')
    assert isinstance(names, metafopy.methods.ls_anova.LSWaldResult)
    assert names.idx == (1, 2)


def test_anova_att_list_and_errors() -> None:
    res = result('bd_3')
    out = metafopy.anova(res, att=[[1], [1, 2]])
    assert isinstance(out, list)
    assert [o.m for o in out] == [1, 2]
    with pytest.raises(metafopy.errors.InputError, match="either 'btt' or 'att'"):
        metafopy.anova(res, btt=[1], att=[1])
    with pytest.raises(metafopy.errors.InputError, match="either 'X' or 'Z'"):
        metafopy.anova(res, X=X23, Z=X23)
    with pytest.raises(metafopy.errors.InputError, match='does not match the number of scale coefficients'):
        metafopy.anova(res, Z=[1, 2, 3, 4])


def bdc() -> pandas.DataFrame:
    d = bd()
    return d[d[['yi', 'vi', 'ni100', 'meta', 'imag']].notna().all(axis=1)].reset_index(drop=True)


def lrt_models() -> dict[str, metafopy.RMALSResult]:
    d = bdc()
    return {
        'full': ls(d, MODS3, SCALE3, method='ML'),
        'scale': ls(d, MODS3, '~ 1', method='ML'),
        'location': ls(d, '~ 1', SCALE3, method='ML'),
        'alpha_fixed': ls(d, MODS3, SCALE3, method='ML', alpha=[numpy.nan, 0, numpy.nan]),
        'full_reml': ls(d, MODS3, SCALE3),
        'scale_reml': ls(d, MODS3, '~ 1'),
    }


@pytest.mark.parametrize(
    ('key', 'first', 'second'),
    [
        ('lrt_scale', 'full', 'scale'),
        ('lrt_location', 'full', 'location'),
        ('lrt_alpha_fixed', 'full', 'alpha_fixed'),
        ('lrt_swapped', 'scale', 'full'),
        ('lrt_reml', 'full_reml', 'scale_reml'),
    ],
)
def test_anova_lrt(key: str, first: str, second: str) -> None:
    models = lrt_models()
    a = metafopy.anova(models[first], models[second])
    assert isinstance(a, metafopy.methods.anova.MVLRTResult)
    ref = REF[key]
    refutil.close(a.LRT, ref['LRT'], f'{key} LRT', rtol=OPT, atol=1e-6)
    refutil.close(a.pval, ref['pval'], f'{key} pval', rtol=OPT, atol=1e-7)
    refutil.close(a.fit_stats_full, ref['fit_stats_f'], f'{key} fit full', rtol=OPT, atol=1e-6)
    refutil.close(a.fit_stats_reduced, ref['fit_stats_r'], f'{key} fit reduced', rtol=OPT, atol=1e-6)
    refutil.close([a.df_full, a.df_reduced], ref['parms'], f'{key} parms')
    refutil.close([a.QE_full, a.QE_reduced], ref['QE'], f'{key} QE', rtol=1e-8)
    assert refutil.r_print(str(a)).strip() == refutil.r_print(ref['print']).strip()


def test_anova_lrt_errors() -> None:
    models = lrt_models()
    with pytest.raises(metafopy.errors.InputError, match='same number of parameters'):
        metafopy.anova(models['full'], models['full'])
    with pytest.raises(metafopy.errors.InputError, match='Mismatch'):
        metafopy.anova(models['full'], models['scale_reml'])
    with pytest.raises(metafopy.errors.InputError, match='not equal'):
        metafopy.anova(models['full'], ls(bd().iloc[:30], MODS3, '~ 1', method='ML'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        metafopy.anova(models['full_reml'], ls(bdc(), '~ 1', '~ 1'))
    assert any('REML comparisons' in str(w.message) for w in caught)
    refit = metafopy.anova(models['full_reml'], ls(bdc(), '~ 1', '~ 1'), refit=True)
    assert isinstance(refit, metafopy.methods.anova.MVLRTResult)


# -- vif --------------------------------------------------------------------------------------------------------------------


def test_vif() -> None:
    v = any_vif(result('bd_3'))
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif beta', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif alpha', rtol=OPT)
    refutil.close(v.beta.sifs.to_numpy(), ref['beta_sif'], 'sif beta', rtol=OPT)
    assert list(v.beta.vifs.index) == ref['beta_names']
    assert list(v.alpha.vifs.index) == ref['alpha_names']
    assert refutil.r_print(str(v)).strip() == refutil.r_print(ref['print']).strip()


def test_vif_reestimate() -> None:
    v = any_vif(result('bd_3'), reestimate=True)
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif_reest']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif reest beta', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif reest alpha', rtol=OPT)


def test_vif_btt_att() -> None:
    v = any_vif(result('bd_3'), btt=1, att=[[1], [1, 2]])
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif_att']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif btt', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif att', rtol=OPT)
    assert refutil.r_print(str(v)).strip() == refutil.r_print(ref['print']).strip()


def test_vif_table() -> None:
    v = any_vif(result('bd_3'), table=True)
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    assert v.beta.table is not None
    assert v.alpha.table is not None
    refutil.close(v.beta.table['estimate'].to_numpy(), result('bd_3').beta.tolist(), 'table beta', rtol=1e-12)
    refutil.close(v.alpha.table['estimate'].to_numpy(), result('bd_3').alpha.tolist(), 'table alpha', rtol=1e-12)
    text = str(v)
    assert 'intrcpt' in text
    assert 'vif' in text


def test_vif_single_part() -> None:
    v = any_vif(result('mc_factors'))
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    refutil.close(v.beta.vifs.to_numpy(), REF['mc_factors']['vif']['beta'], 'vif mc', rtol=OPT)
    only_loc = any_vif(result('bd_ni_ml'))
    assert isinstance(only_loc, metafopy.methods.ls_vif.LSVIFResult)
    loc_only = ls(bd(), MODS3, '~ 1')
    single = any_vif(loc_only)
    assert isinstance(single, metafopy.VIFResult)
    scale_only = ls(bd(), None, SCALE3)
    single_scale = any_vif(scale_only)
    assert isinstance(single_scale, metafopy.VIFResult)
    assert list(single_scale.vifs.index) == ['ni100', 'imag']


def test_vif_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        any_vif(result('bd_int'))
    with pytest.raises(metafopy.errors.InputError, match="'sim'"):
        any_vif(result('bd_3'), sim=1)


# -- blup, ranef, rstandard and the other accessors -------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
DIAG_CASES = {
    'bd_int': ('bd_int', None),
    'bd_3': ('bd_3', None),
    'bd_na': ('bd_na', 'diag'),
    'bcg_ablat': ('bcg_ablat', 'diag'),
    'mo_int': ('mo_int', 'diag'),
    'mc_factors': ('mc_factors', 'diag'),
}


def diag_ref(name: str, key: str | None) -> dict[str, typing.Any]:
    ref = REF[name]
    out: dict[str, typing.Any] = ref if key is None else ref[key]
    return out


@pytest.mark.parametrize('case', sorted(DIAG_CASES))
def test_blup_ranef_rstandard(case: str) -> None:
    name, key = DIAG_CASES[case]
    res = result(name)
    ref = diag_ref(name, key)
    b = metafopy.blup(res)
    refutil.close(b.pred, ref['blup']['pred'], f'{case} blup pred', rtol=OPT, atol=1e-7)
    refutil.close(b.se, ref['blup']['se'], f'{case} blup se', rtol=OPT, atol=1e-7)
    refutil.close(b.pi_lb, ref['blup']['pi_lb'], f'{case} blup pi_lb', rtol=OPT, atol=1e-7)
    refutil.close(b.pi_ub, ref['blup']['pi_ub'], f'{case} blup pi_ub', rtol=OPT, atol=1e-7)
    # metafor's slab holds the labels of all supplied studies; the BLUPs only those used in the fit
    assert list(b.slab) == list(numpy.asarray(ref['slab'], dtype=object)[res.data.keep])
    assert len(refutil.r_print(str(b)).split('\n')) == len(refutil.r_print(ref['blup']['print']).split('\n'))
    r = metafopy.ranef(res)
    refutil.close(r.pred, ref['ranef']['pred'], f'{case} ranef pred', rtol=OPT, atol=1e-7)
    refutil.close(r.se, ref['ranef']['se'], f'{case} ranef se', rtol=OPT, atol=1e-7)
    refutil.close(r.pi_lb, ref['ranef']['pi_lb'], f'{case} ranef pi_lb', rtol=OPT, atol=1e-7)
    s = metafopy.rstandard(res)
    refutil.close(s.resid, ref['rstandard']['resid'], f'{case} rstandard resid', rtol=OPT, atol=1e-7)
    refutil.close(s.se, ref['rstandard']['se'], f'{case} rstandard se', rtol=OPT, atol=1e-7)
    refutil.close(s.z, ref['rstandard']['z'], f'{case} rstandard z', rtol=OPT, atol=1e-6)


def test_blup_options_and_conditional_residuals() -> None:
    res = result('bd_na')
    padded = metafopy.blup(res, na_action='exclude')
    assert len(padded.pred) == 48
    assert int(numpy.isnan(padded.pred).sum()) == 48 - res.k
    cond = metafopy.rstandard(res, type='conditional')
    assert len(cond.resid) == res.k
    with pytest.raises(metafopy.errors.InputError, match='Missing values'):
        metafopy.rstandard(res, na_action='fail')
    with pytest.raises(metafopy.errors.InputError, match='non-standard weights'):
        metafopy.blup(result('bd_3_weights'))
    exp = metafopy.blup(result('bd_3'), transf=numpy.exp)
    assert bool(numpy.all(numpy.isnan(exp.se)))


def test_accessors() -> None:
    res = result('bd_3')
    ref = REF['bd_3']
    refutil.close(res.coef('beta').to_numpy(), ref['coef']['beta'], 'coef beta', rtol=OPT)  # type: ignore[union-attr]
    refutil.close(res.coef('alpha').to_numpy(), ref['coef']['alpha'], 'coef alpha', rtol=OPT)  # type: ignore[union-attr]
    refutil.close(res.vcov('alpha').to_numpy().ravel(), flat(ref['coef']['vcov_alpha']), 'vcov alpha', rtol=OPT, atol=1e-7)
    assert set(res.coef()) == {'alpha', 'beta'}
    fs = res.fitstats()
    refutil.close(fs.to_numpy(), list(res.fit_stats_reml.as_tuple()), 'fitstats', rtol=1e-12)
    ml = res.fitstats(REML=False)
    refutil.close(ml.to_numpy(), list(res.fit_stats_ml.as_tuple()), 'fitstats ML', rtol=1e-12)
    refutil.close(res.hatvalues(), ref['hat'], 'hat', rtol=OPT)
    refutil.close(result('bd_int').fitstats().to_numpy(), REF['bd_int']['fitstats'], 'fitstats int', rtol=1e-6)
    assert repr(res).startswith('<RMALSResult')


def test_unsupported_methods_raise() -> None:
    res = result('bd_3')
    for func in (
        metafopy.influence,
        metafopy.cooks_distance,
        metafopy.dfbetas,
        metafopy.rstudent,
        metafopy.leave1out,
        metafopy.cumul,
        metafopy.robust,
    ):
        with pytest.raises(metafopy.errors.ArgumentError):
            func(res)
    with pytest.raises(metafopy.errors.NotSupportedError, match='Method not available'):
        metafopy.gosh(res)


# -- permutest --------------------------------------------------------------------------------------------------------------

# The permutation tests compare with metafor's single-start fits, so their models are fitted with restarts=False: a restart reaches a higher
# maximum on some permuted refits, which moves the permutation p-values (spec location-scale-fits-reach-the-higher-maximum).
NO_RESTARTS = {'restarts': False}
PERM_FITS: collections.abc.Mapping[str, Fit] = types.MappingProxyType(
    {
        'bcg_ablat': lambda: ls(bcg(), '~ ablat', '~ ablat', control=NO_RESTARTS),
        'bcg_ablat_t': lambda: ls(bcg(), '~ ablat', '~ ablat', test='t', control=NO_RESTARTS),
        'bcg_int_scale': lambda: ls(bcg(), '~ ablat', '~ 1', control=NO_RESTARTS),
        'bd6_exact': lambda: ls(bd6(), '~ ni100', '~ ni100', control=NO_RESTARTS),
    }
)

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PERM_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'both': ('bcg_ablat', 'permutest', {'iter': 100, 'seed': 1234}),
    'beta': ('bcg_ablat', 'permutest_beta', {'iter': 60, 'seed': 99, 'skip_alpha': True}),
    'alpha': ('bcg_ablat', 'permutest_alpha', {'iter': 60, 'seed': 99, 'skip_beta': True}),
    'btt': ('bcg_ablat', 'permutest_btt', {'iter': 40, 'seed': 5, 'btt': [1], 'att': [1]}),
    'less': ('bcg_ablat', 'permutest_less', {'iter': 50, 'seed': 7, 'control': {'alternative': 'less', 'stat': 'coef'}}),
    'px2': ('bcg_ablat', 'permutest_px2', {'iter': 50, 'seed': 7, 'control': {'p2defn': 'px2'}}),
    't': ('bcg_ablat_t', 'permutest', {'iter': 50, 'seed': 321}),
    'int_scale': ('bcg_int_scale', 'permutest', {'iter': 50, 'seed': 11}),
}


@pytest.mark.parametrize('case', sorted(PERM_CASES))
def test_permutest(case: str) -> None:
    name, key, kwargs = PERM_CASES[case]
    pt = any_permutest(PERM_FITS[name](), **kwargs)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    ref = REF[name][key]
    # p-values of a skipped part come from the model fit (optimizer precision), the others are proportions of permutations
    tol_beta = OPT if pt.skip_beta else 1e-8
    tol_alpha = PERM_ALPHA if pt.skip_alpha else 1e-8
    refutil.close(pt.pval, ref['pval'], f'{case} pval', rtol=tol_beta, atol=1e-8)
    refutil.close(pt.QMp, ref['QMp'], f'{case} QMp', rtol=tol_beta, atol=1e-8)
    refutil.close(pt.QM, ref['QM'], f'{case} QM', rtol=OPT, atol=1e-7)
    assert pt.skip_beta == ref['skip_beta']
    assert pt.skip_alpha == ref['skip_alpha']
    refutil.close([pt.exact_iter, pt.Z_exact_iter], [ref['exact_iter'], ref['Z_exact_iter']], f'{case} exact iter')
    if case != 'px2':
        refutil.close(pt.pval_alpha, ref['pval_alpha'], f'{case} pval_alpha', rtol=tol_alpha, atol=1e-8)
    if ref['QSp'] is not None:
        refutil.close(pt.QSp, ref['QSp'], f'{case} QSp', rtol=tol_alpha, atol=1e-8)
    if not pt.skip_beta:
        refutil.close(pt.QM_perm, ref['QM_perm'], f'{case} QM_perm', rtol=OPT, atol=1e-6)
        refutil.close(pt.zval_perm.to_numpy().ravel(), flat(ref['zval_perm']), f'{case} zval_perm', rtol=OPT, atol=1e-6)
    if not pt.skip_alpha:
        refutil.close(pt.QS_perm, ref['QS_perm'], f'{case} QS_perm', rtol=PERM_ALPHA, atol=1e-6)
        refutil.close(pt.zval_alpha_perm.to_numpy().ravel(), flat(ref['zval_alpha_perm']), f'{case} zval_alpha_perm', rtol=PERM_ALPHA, atol=1e-6)


PRINTED_DECIMAL = re.compile(r'-?\d+\.\d+')


def same_loose(actual: str, expected: str) -> bool:
    """Printed output agrees up to the numbers, which may differ by a few units in the third decimal.

    The scale part of ``bcg_ablat`` is only identified to about 1e-4, so the last printed digit depends on the BLAS in use.
    """
    a_text = refutil.r_print(actual).strip()
    e_text = refutil.r_print(expected).strip()
    if PRINTED_DECIMAL.sub('#', a_text) != PRINTED_DECIMAL.sub('#', e_text):
        return False
    a_nums = [float(x) for x in PRINTED_DECIMAL.findall(a_text)]
    e_nums = [float(x) for x in PRINTED_DECIMAL.findall(e_text)]
    return bool(numpy.allclose(a_nums, e_nums, rtol=2e-3, atol=2.5e-3))


@pytest.mark.parametrize('case', ['both', 'beta', 'alpha', 't', 'int_scale'])
def test_permutest_print(case: str) -> None:
    name, key, kwargs = PERM_CASES[case]
    pt = any_permutest(PERM_FITS[name](), **kwargs)
    assert same_loose(str(pt), REF[name][key]['print'])


def test_permutest_exact_iter_and_exact() -> None:
    assert any_permutest(PERM_FITS['bcg_ablat'](), exact='i') == REF['bcg_ablat']['permutest_iter']
    small = PERM_FITS['bd6_exact']()
    assert any_permutest(small, exact='i') == REF['bd6_exact']['permutest_iter']
    pt = any_permutest(small, exact=True)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    ref = REF['bd6_exact']['permutest']
    assert (pt.exact_iter, pt.Z_exact_iter) == (ref['exact_iter'], ref['Z_exact_iter'])
    # metafor's optimizer fails for 11 of the 720 permuted fits (it drops them from the p-values), ours converges for all of them;
    # where both succeed the statistics agree, so the p-values agree up to the share of dropped permutations
    expected = numpy.array([[numpy.nan if v is None else v for v in row] for row in ref['zval_perm']])
    ok = ~numpy.isnan(expected).any(axis=1)
    assert int(ok.sum()) == 709
    refutil.close(pt.zval_perm.to_numpy()[ok].ravel(), expected[ok].ravel().tolist(), 'exact zval_perm', rtol=OPT, atol=1e-6)
    refutil.close(pt.pval, ref['pval'], 'exact pval', rtol=0, atol=11 / 720)
    refutil.close(pt.QMp, ref['QMp'], 'exact QMp', rtol=0, atol=11 / 720)
    refutil.close(pt.pval_alpha, ref['pval_alpha'], 'exact pval_alpha', rtol=0, atol=11 / 720)
    refutil.close(pt.QSp, ref['QSp'], 'exact QSp', rtol=0, atol=11 / 720)


def test_permutest_errors() -> None:
    res = result('bcg_ablat')
    with pytest.raises(metafopy.errors.InputError, match='at least one part'):
        any_permutest(res, skip_beta=True, skip_alpha=True)
    with pytest.raises(metafopy.errors.InputError, match='iter'):
        any_permutest(res, iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pt = any_permutest(result('bcg_int_scale'), iter=20, seed=2)
    assert any('intercept-only scale model' in str(w.message) for w in caught)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    assert pt.skip_alpha


NOT_CROSSED = ('<', '>')


def test_confint_scale_interval_is_not_capped_at_ten_times_the_estimate() -> None:
    """Table 1 of Viechtbauer and Lopez-Lopez (2022): the social sciences bounds are not crossed in ``10 crit se``; ``xlim`` gives metafor 5.2-1's."""
    case = next(item for item in correctness.cases.load_cases('published') if item.id == 'published.vl2022.table1.plci.size')
    res = typing.cast('metafopy.models.rma_ls_result.RMALSResult', correctness.cases.run_case(case))
    pos = 3
    vc = float(res.alpha[pos])
    ci = metafopy.confint(res, alpha=pos)
    assert isinstance(ci, metafopy.ConfintResult)
    assert (ci.lb_sign, ci.ub_sign) == NOT_CROSSED
    table = ci.random
    assert table is not None
    crit = float(scipy.stats.norm.isf(0.025))
    se = float(res.se_alpha[pos])
    assert table['ci.lb'].iloc[0] == pytest.approx(vc - 10 * crit * se, abs=1e-6)
    assert table['ci.ub'].iloc[0] == pytest.approx(vc + 10 * crit * se, abs=1e-6)
    capped = metafopy.confint(res, alpha=pos, xlim=(vc - 10 * abs(vc), vc + 10 * abs(vc)))
    assert isinstance(capped, metafopy.ConfintResult)
    assert capped.random is not None
    assert (capped.lb_sign, capped.ub_sign) == NOT_CROSSED
    assert capped.random['ci.lb'].iloc[0] == pytest.approx(-3.6100, abs=5e-5)
    assert capped.random['ci.ub'].iloc[0] == pytest.approx(4.4122, abs=5e-5)


def test_hat_matrix_diagonal_is_hatvalues() -> None:
    res = FITS['bd_3']()
    assert isinstance(res, metafopy.RMALSResult)
    hat = res.hat_matrix()
    numpy.testing.assert_allclose(numpy.diag(hat), res.hatvalues(), rtol=1e-10)
    numpy.testing.assert_allclose(hat @ hat, hat, atol=1e-8)


def test_post_init_labels_the_matrices_of_a_replaced_fit() -> None:
    res = FITS['bd_3']()
    assert isinstance(res, metafopy.RMALSResult)
    again = dataclasses.replace(res, H=numpy.asarray(res.H), X=numpy.asarray(res.X), Z=numpy.asarray(res.Z))
    assert isinstance(again.H, metafopy.core.labelled.LabelledMatrix)
    assert isinstance(res.H, metafopy.core.labelled.LabelledMatrix)
    assert again.H.rownames == res.H.rownames
    assert isinstance(again.X, metafopy.core.labelled.LabelledMatrix)
    assert isinstance(again.Z, metafopy.core.labelled.LabelledMatrix)
    assert again.X.colnames == tuple(res.coef_names)
    assert again.Z.colnames == tuple(res.alpha_names)
    assert again.X.rownames == tuple(res.data.rownames)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    stat=rstrategies.finite_floats(0.0, 1e6),
    m=hypothesis.strategies.integers(min_value=1, max_value=50),
    ddf_den=rstrategies.finite_floats(0.5, 1e4),
)
def test_omnibus_property_f_test_scales_the_statistic_and_bounds_the_p_value(stat: float, m: int, ddf_den: float) -> None:
    """Candidate ``src/metafopy/methods/ls_anova.py::omnibus``: with ``ddf_den`` the statistic is divided by ``m`` and p is in [0, 1]."""
    scaled, df, p = metafopy.methods.ls_anova.omnibus(stat, m, ddf_den)
    assert scaled == stat / m
    assert df == (m, ddf_den)
    assert 0.0 <= p <= 1.0


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(stat=rstrategies.finite_floats(0.0, 1e6), m=hypothesis.strategies.integers(min_value=1, max_value=50))
def test_omnibus_property_without_ddf_den_is_the_chi_square_tail(stat: float, m: int) -> None:
    """Candidate ``src/metafopy/methods/ls_anova.py::omnibus``: ``ddf_den = None`` leaves the statistic unscaled and gives the chi-square tail."""
    unscaled, df, p = metafopy.methods.ls_anova.omnibus(stat, m, None)
    assert unscaled == stat
    assert df == (m, None)
    assert p == float(scipy.stats.chi2.sf(stat, m))


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    stat=rstrategies.finite_floats(0.0, 1e6),
    m=hypothesis.strategies.integers(min_value=1, max_value=50),
    ddf_den=rstrategies.finite_floats(-100.0, 0.0),
)
def test_omnibus_property_p_value_is_nan_without_denominator_degrees_of_freedom(stat: float, m: int, ddf_den: float) -> None:
    """Candidate ``src/metafopy/methods/ls_anova.py::omnibus``: p is NaN when ``ddf_den <= 0``."""
    assert math.isnan(metafopy.methods.ls_anova.omnibus(stat, m, ddf_den)[2])


def leverage_one_data() -> pandas.DataFrame:
    """``dat.bcg`` with three studies' counts or moderator missing, as in the location-scale interaction example of ``docs/differences.md``."""
    dat = metafopy.datasets.load('dat.bcg').copy()
    for row, column in [(2, 'tpos'), (6, 'cpos'), (4, 'ablat')]:
        dat.loc[dat.index[row], column] = numpy.nan
    return metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)


@pytest.mark.parametrize('draw', range(10))
def test_study_of_leverage_one_is_refused_on_every_platform(draw: int) -> None:
    """A study with ``1 - h_ii < LEVERAGE_ONE`` has a zero residual in the scale start, so R's error is raised, unperturbed and at rounding noise.

    Draw 0 is the unperturbed input, which fitted on the development Mac before the check; draws 1 to 9 perturb ``yi`` by ``1e-15`` noise as the CI
    runners' arithmetic does.
    """
    dat = leverage_one_data()
    if draw > 0:
        rng = numpy.random.default_rng(1)
        noise = numpy.array([rng.standard_normal(dat.shape[0]) for _ in range(draw)])[-1]
        dat['yi'] = dat['yi'] * (1 + 1e-15 * noise)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(metafopy.errors.InputError, match="No missing values allowed in 'alpha_init'"):
            metafopy.rma(yi='yi', vi='vi', data=dat, mods='~ ablat * alloc', scale='~ ablat')


def test_restarts_control_defaults_on_and_is_stored() -> None:
    """``LSControl.restarts`` is on by default and ``control={'restarts': False}`` is kept in ``res.control``."""
    assert ls(bd(), mods=MODS3, scale=SCALE3).control.restarts is True
    assert ls(bd(), mods=MODS3, scale=SCALE3, control={'restarts': False}).control.restarts is False


def test_restarts_control_is_carried_into_refits() -> None:
    """A refit through ``refit_ls`` keeps ``restarts=False``, as the permutation test and the profile need."""
    res = ls(bd(), mods=MODS3, scale=SCALE3, control={'restarts': False})
    refit = metafopy.methods.ls_common.refit_ls(res)
    assert refit is not None
    assert refit.control.restarts is False


def test_restarts_control_refuses_a_non_bool() -> None:
    """A truthy string such as ``'FALSE'`` would turn the restarts on, so only a ``bool`` is accepted."""
    with pytest.raises(metafopy.errors.InputError, match='restarts'):
        ls(bd(), mods=MODS3, scale=SCALE3, control={'restarts': 'FALSE'})
