"""Numerical agreement with R where the optimisers stop (tests/r/generate_audit4_numerics.R).

``rma_mv(struct='GEN')`` on a ridge of the likelihood, ``permutest()`` of a location-scale fit with permuted scale fits on the boundary, and the
scale standard errors of an identity-link location-scale fit at two Richardson depths.
"""

import sys
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets

any_permutest: typing.Callable[..., typing.Any] = metafopy.permutest

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_numerics')


def kon() -> pandas.DataFrame:
    return pandas.DataFrame(REF['gen']['kon'])


def bcg() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def gen_fit(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', random='~ year | district', struct='GEN', data=kon(), **kwargs)


def test_gen_converges_to_the_end_of_the_ridge() -> None:
    """R's default run stops at logLik -31.47369 on the ridge; the fit here reaches its end, where R started there stays (-30.98466)."""
    ref = REF['gen']
    res = gen_fit(mods='~ year')
    ll = res.fit_stats(True).as_tuple()[0]
    assert ll >= ref['default_ll']
    refutil.close([ll], [ref['hot_ll']], 'GEN logLik', rtol=1e-7)
    refutil.close(res.tau2, ref['hot_tau2'], 'GEN tau2', rtol=1e-5)
    refutil.close(res.beta, ref['hot_beta'], 'GEN beta', rtol=1e-5)
    assert res.rho is not None
    assert res.rho[0] == pytest.approx(-1.0, abs=1e-9)


def test_gen_without_moderators_converges() -> None:
    ref = REF['gen']
    ll = gen_fit().fit_stats(True).as_tuple()[0]
    assert ll >= ref['nomods_default_ll']
    assert ll == pytest.approx(ref['nomods_hot_ll'], abs=1e-3)


def test_gen_ml_converges() -> None:
    tau2 = gen_fit(method='ML').tau2
    assert tau2 is not None
    assert numpy.isfinite(tau2).all()


def test_permutest_ls_boundary_fits_match_r() -> None:
    """Permuted scale fits on the boundary stop where nlminb does (relative change below 1e-10), so undefined standard errors and p-values agree."""
    ref = REF['permutest']
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ alloc', data=bcg())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pt = any_permutest(res, iter=50, seed=1)
    refutil.close(pt.pval, ref['pval'], 'pval', rtol=1e-12)
    refutil.close([pt.QMp], [ref['QMp']], 'QMp', rtol=1e-12)
    # where a permuted scale fit runs to the boundary the stopping point depends on the platform's libm, so the number of permutations with an
    # undefined standard error (and with it the p-values) is only exact on Linux
    exact = 1e-12 if sys.platform == 'linux' else 0.05
    refutil.close(pt.pval_alpha, ref['pval_alpha'], 'pval_alpha', rtol=exact)
    refutil.close([pt.QSp], [ref['QSp']], 'QSp', rtol=exact)
    if sys.platform != 'linux':
        return
    r_z = numpy.array([[numpy.nan if v is None else v for v in col] for col in ref['zval_alpha_perm'].values()]).T
    r_a = numpy.array([[numpy.nan if v is None else v for v in col] for col in ref['alpha_perm'].values()]).T
    z = pt.zval_alpha_perm.to_numpy()
    a = pt.alpha_perm.to_numpy()
    assert (numpy.isnan(z) == numpy.isnan(r_z)).all()
    interior = numpy.abs(r_a).max(axis=1) < 10
    refutil.close(a[interior].ravel(), r_a[interior].ravel().tolist(), 'alpha_perm (interior)', rtol=2e-4, atol=1e-6)
    refutil.close(z[interior].ravel(), r_z[interior].ravel().tolist(), 'zval_alpha_perm (interior)', rtol=2e-4, atol=1e-6)
    # the coefficient of a permuted fit that ran to the boundary has a standard error of 1e3 or more in R, so its z value is within 0.01 of zero
    # there and here; the other coefficients of that fit move with where the search stopped, so they only agree roughly
    on_boundary = numpy.abs(r_z) < 0.01
    assert (numpy.abs(z[on_boundary & ~numpy.isnan(z)]) < 0.01).all()
    rest = ~on_boundary & ~numpy.isnan(r_z)
    refutil.close(z[rest], r_z[rest].tolist(), 'zval_alpha_perm (other coefficients)', rtol=2e-3, atol=1e-3)


def identity_fit(r: int | None) -> metafopy.RMALSResult:
    control = None if r is None else {'hessian_r': r}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', scale='~ ablat', link='identity', data=bcg(), control=control)
    assert isinstance(res, metafopy.RMALSResult)
    return res


def test_identity_link_scale_se_depth_4() -> None:
    """At depth 4 numDeriv's steps are still above rounding error, so the standard errors agree with R's to 1e-5."""
    ref = REF['identity']['r4']
    res = identity_fit(4)
    refutil.close(res.alpha, ref['alpha'], 'alpha', rtol=1e-6)
    refutil.close(res.se_alpha, ref['se_alpha'], 'se_alpha', rtol=1e-5)


def test_identity_link_scale_se_depth_8() -> None:
    """At depth 8 the steps for the coefficient -3.8e-5 are 3e-8 and the second differences are rounding error: R's and our noise differ (5e-4)."""
    ref = REF['identity']['r8']
    res = identity_fit(8)
    refutil.close(res.alpha, ref['alpha'], 'alpha', rtol=1e-6)
    refutil.close(res.se_alpha, ref['se_alpha'], 'se_alpha', rtol=2e-3)
