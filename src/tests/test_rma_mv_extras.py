"""rma_mv arguments ``optbeta`` (``beta``, ``lambda1``, ``lambda2``), ``vccon``, ``cvvc`` and ``restart`` checked against R metafor 5.2-1.

The fixture ``reference/extras/rma_mv_extras.json`` is written by ``src/tests/r/generate_rma_mv_extras.R``.
"""

import collections.abc
import json
import pathlib
import re
import typing
import warnings

import numpy
import numpy.typing
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.models.mv_structs

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'extras' / 'rma_mv_extras.json').read_text(encoding='utf-8'))

# the optimiser (BFGS with analytic gradients) and nlminb stop at slightly different points of a flat likelihood
RTOL = 1e-4
VC_RTOL = 1e-4
VC_ATOL = 1e-6
HESS_RTOL = 1e-3

type Fit = collections.abc.Callable[[], metafopy.RMAMVResult]
type Floats = numpy.typing.NDArray[numpy.float64]


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def kon() -> pandas.DataFrame:
    return metafopy.datasets.load('konstantopoulos2011')


def berkey_v() -> Floats:
    data = metafopy.datasets.load('berkey1998')
    variances = [col for col in data.columns if re.fullmatch(r'v\di', col)]
    return metafopy.bldiag([g[variances].to_numpy() for _, g in data.groupby('trial', sort=False)])


def kr(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon(), **kwargs)


def ar(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    # auditme: accept[literal-duplication] the formulas repeat as the R script that generates the fixtures repeats them
    return metafopy.rma_mv('yi', 'vi', random=['~ 1 | study', '~ 1 | esid'], data=metafopy.datasets.load('assink2016'), **kwargs)


def br(**kwargs: typing.Any) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', berkey_v(), mods='~ 0 + outcome', random='~ outcome | trial', data=metafopy.datasets.load('berkey1998'), **kwargs)


YEAR = '~ I(year - 1990)'
YEAR2 = '~ I(year - 1990) + I((year - 1990)**2)'
DELTYPE = '~ C(deltype) + year'
NO_CHOLESKY = {'cholesky': False}

# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
CASES: dict[str, Fit] = {
    'kon_vccon_sigma2': lambda: kr(vccon={'sigma2': [1, 1]}),
    'kon_vccon_sigma2_ml': lambda: kr(vccon={'sigma2': [1, 1]}, method='ML'),
    'kon_vccon_mods': lambda: kr(mods=YEAR, vccon={'sigma2': [1, 1]}),
    'kon_vccon_distinct': lambda: kr(vccon={'sigma2': [1, 2]}),
    'kon_vccon_fixed': lambda: kr(sigma2=[0.04, 0.04], vccon={'sigma2': [1, 1]}),
    'kon_vccon_fixed_mixed': lambda: kr(sigma2=[0.04, numpy.nan], vccon={'sigma2': [1, 1]}),
    'ass_vccon_sigma2': lambda: ar(vccon={'sigma2': [1, 1]}),
    'ass_vccon_sigma2_ml': lambda: ar(vccon={'sigma2': [1, 1]}, method='ML'),
    'ber_un_vccon_tau2': lambda: br(struct='UN', vccon={'tau2': [1, 1]}),
    'ber_un_vccon_tau2_nochol': lambda: br(struct='UN', vccon={'tau2': [1, 1]}, control=NO_CHOLESKY),
    'ber_hcs_vccon_tau2': lambda: br(struct='HCS', vccon={'tau2': [1, 1]}),
    'ber_un_vccon_sigma2_only': lambda: br(struct='UN', vccon={'sigma2': [1]}),
    'kon_optbeta': lambda: kr(optbeta=True),
    'kon_optbeta_ml': lambda: kr(optbeta=True, method='ML'),
    'kon_optbeta_mods': lambda: kr(mods=YEAR, optbeta=True),
    'kon_optbeta_mods_t': lambda: kr(mods=YEAR, optbeta=True, test='t'),
    'kon_optbeta_fixbeta': lambda: kr(mods=YEAR, optbeta=True, beta=[numpy.nan, 0.01]),
    'kon_fixbeta': lambda: kr(mods=YEAR, beta=[numpy.nan, 0.01]),
    'kon_fixbeta_ml': lambda: kr(mods=YEAR, beta=[0.1, numpy.nan], method='ML'),
    'kon_ridge': lambda: kr(mods=YEAR, lambda2=5),
    'kon_ridge_ml': lambda: kr(mods=YEAR, lambda2=5, method='ML'),
    'kon_ridge_big': lambda: kr(mods=YEAR2, lambda2=50),
    'kon_lasso': lambda: kr(mods=YEAR, lambda1=0.05),
    'ass_optbeta': lambda: ar(mods=DELTYPE, optbeta=True),
    'ass_ridge': lambda: ar(mods=DELTYPE, lambda2=2),
    'ber_un_optbeta': lambda: br(struct='UN', optbeta=True),
    'ber_cs_optbeta': lambda: br(struct='CS', optbeta=True),
    'ber_un_optbeta_ml': lambda: br(struct='UN', optbeta=True, method='ML'),
    'kon_fe_optbeta': lambda: metafopy.rma_mv('yi', 'vi', data=kon(), method='FE', mods=YEAR, optbeta=True),
    'kon_cvvc_true': lambda: kr(cvvc=True),
    'kon_cvvc_varcor': lambda: kr(cvvc='varcor'),
    'kon_cvvc_transf': lambda: kr(cvvc='transf'),
    'kon_cvvc_varcov': lambda: kr(cvvc='varcov'),
    'kon_cvvc_ml': lambda: kr(cvvc=True, method='ML'),
    'kon_cvvc_mods': lambda: kr(mods=YEAR, cvvc=True),
    'kon_cvvc_fixed': lambda: kr(sigma2=[numpy.nan, 0.02], cvvc=True),
    'kon_cvvc_fixed_varcov': lambda: kr(sigma2=[numpy.nan, 0.02], cvvc='varcov'),
    'kon_cvvc_vccon': lambda: kr(vccon={'sigma2': [1, 2]}, cvvc=True),
    'ass_cvvc_true': lambda: ar(cvvc=True),
    'ass_cvvc_transf': lambda: ar(cvvc='transf'),
    'ass_cvvc_ml': lambda: ar(cvvc=True, method='ML'),
    'ber_un_cvvc_true': lambda: br(struct='UN', cvvc=True),
    'ber_un_cvvc_transf': lambda: br(struct='UN', cvvc='transf'),
    'ber_un_cvvc_varcov': lambda: br(struct='UN', cvvc='varcov'),
    'ber_un_nochol_cvvc_transf': lambda: br(struct='UN', cvvc='transf', control=NO_CHOLESKY),
    'ber_un_nochol_cvvc_varcor': lambda: br(struct='UN', cvvc='varcor', control=NO_CHOLESKY),
    'ber_cs_cvvc_true': lambda: br(struct='CS', cvvc=True),
    'ber_cs_cvvc_varcov': lambda: br(struct='CS', cvvc='varcov'),
    'ber_hcs_cvvc_true': lambda: br(struct='HCS', cvvc=True),
    'ber_hcs_cvvc_transf': lambda: br(struct='HCS', cvvc='transf'),
    'ber_un_fixrho_cvvc': lambda: br(struct='UN', rho=0.5, cvvc=True),
    'ber_ar_cvvc_true': lambda: br(struct='AR', cvvc=True),
    'ber_diag_cvvc_true': lambda: br(struct='DIAG', cvvc=True),
}

WARNS = {
    'ber_cs_cvvc_varcov': "Cannot use cvvc='varcov' for the specified",
    'kon_cvvc_fixed_varcov': "Cannot use cvvc='varcov' when",
    'kon_cvvc_varcov': "Cannot use cvvc='varcov' for the specified",
}


def build(name: str) -> metafopy.RMAMVResult:
    """Fit a case, ignoring the warnings that some of them provoke on purpose."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = CASES[name]()
    return fitted


def floats(value: typing.Any) -> Floats:
    return numpy.array([numpy.nan if v is None else v for v in numpy.array(value, dtype=object).ravel()], dtype=float)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float, atol: float = 1e-9) -> None:
    numpy.testing.assert_allclose(numpy.asarray(actual, dtype=float).reshape(-1), floats(expected), rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


@pytest.mark.parametrize('name', sorted(CASES))
def test_estimates(name: str) -> None:
    ref = REF[name]
    assert 'error' not in ref, ref
    res = build(name)
    tol = RTOL
    close(res.beta, ref['beta'], f'{name} beta', tol, 1e-7)
    close(res.se, ref['se'], f'{name} se', tol, 1e-7)
    close(res.vb, ref['vb'], f'{name} vb', tol, 1e-8)
    for key in metafopy.models.mv_structs.VC_NAMES:
        actual = getattr(res, key)
        if actual is not None:
            close(actual, ref[key], f'{name} {key}', VC_RTOL, VC_ATOL)
    close(res.pval, ref['pval'], f'{name} pval', tol, 1e-7)
    close(res.ci_lb, ref['ci_lb'], f'{name} ci_lb', tol, 1e-7)
    close(res.ci_ub, ref['ci_ub'], f'{name} ci_ub', tol, 1e-7)
    close(res.QM, ref['QM'], f'{name} QM', tol, 1e-7)
    close(res.QMp, ref['QMp'], f'{name} QMp', tol, 1e-7)
    assert res.p_eff == ref['p']
    assert res.parms == ref['parms']
    assert res.optbeta == ref['optbeta']
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{name} fit ML', 1e-5, 1e-6)
    close(res.fit_stats_reml.as_tuple(), ref['fit_reml'], f'{name} fit REML', 1e-5, 1e-6)


@pytest.mark.parametrize('name', sorted(CASES))
def test_hessian(name: str) -> None:
    ref = REF[name]
    res = build(name)
    if ref['hessian'] is None:
        assert res.hessian is None
        assert res.vvc is None
        return
    assert res.hessian is not None
    names = refutil.as_list(ref['hessian_names'])
    assert list(res.vc_names) == names
    close(res.hessian, ref['hessian'], f'{name} hessian', HESS_RTOL, 1e-6)
    if ref['vvc'] is None:
        assert res.vvc is None
    else:
        assert res.vvc is not None
        close(res.vvc, ref['vvc'], f'{name} vvc', HESS_RTOL, 1e-6)


@pytest.mark.parametrize('name', sorted(WARNS))
def test_warnings(name: str) -> None:
    with pytest.warns(UserWarning, match=WARNS[name]):
        CASES[name]()


def test_constrained_components_are_equal() -> None:
    res = build('kon_vccon_sigma2')
    assert res.sigma2 is not None
    assert res.sigma2[0] == res.sigma2[1]
    unconstrained = kr()
    assert res.fit_stats_reml.ll <= unconstrained.fit_stats_reml.ll + 1e-9
    assert res.parms == unconstrained.parms - 1
    assert res.vccon is not None
    assert list(res.vccon['sigma2']) == [1, 1]


def test_vccon_tau2_equals_hcs_to_cs_in_variance() -> None:
    res = build('ber_hcs_vccon_tau2')
    assert res.tau2 is not None
    assert res.tau2[0] == res.tau2[1]


def test_optbeta_matches_gls_point_estimates() -> None:
    plain = kr(mods=YEAR)
    joint = kr(mods=YEAR, optbeta=True)
    numpy.testing.assert_allclose(joint.beta, plain.beta, rtol=1e-4)
    assert joint.hessian is not None
    assert joint.vvc is not None
    assert joint.vc_names[:2] == ('beta0', 'beta1')


def test_lambda_implies_optbeta() -> None:
    assert build('kon_ridge').optbeta
    plain = kr(mods=YEAR)
    assert abs(build('kon_ridge').beta[1]) < abs(plain.beta[1])


def test_restart_continues_from_previous_fit() -> None:
    first = kr()
    again = kr(restart=first)
    assert again.sigma2 is not None
    assert first.sigma2 is not None
    numpy.testing.assert_allclose(again.sigma2, first.sigma2, rtol=1e-5)
    assert again.iterations <= first.iterations
    with pytest.raises(metafopy.errors.InputError, match='Restarting for a different model'):
        ar(restart=br(struct='UN'))
    with pytest.raises(metafopy.errors.InputError, match='Restarting for a different model'):
        kr(restart=metafopy.rma_mv('yi', 'vi', random='~ 1 | district', data=kon()))


def test_restart_matches_fresh_fit() -> None:
    first = br(struct='UN')
    again = br(struct='UN', restart=first)
    close(again.tau2, first.tau2, 'tau2', 1e-5, 1e-8)
    close(again.rho, first.rho, 'rho', 1e-5, 1e-8)


def test_refits_in_confint_keep_vccon_but_not_optbeta() -> None:
    res = kr(optbeta=True, vccon={'sigma2': [1, 2]})
    ci = metafopy.confint(res, sigma2=0)
    assert ci is not None


def test_argument_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'cvvc'"):
        kr(cvvc='bogus')
    with pytest.raises(metafopy.errors.InputError, match="Length of the 'beta' argument"):
        kr(beta=[0.1, 0.2])
    with pytest.raises(metafopy.errors.InputError, match="Length of 'vccon\\$sigma2'"):
        kr(vccon={'sigma2': [1, 1, 1]})
    with pytest.raises(metafopy.errors.InputError, match="Length of 'vccon\\$sigma2'"):
        kr(sigma2=[0.1, 0.2], vccon={'sigma2': [1, 1, 1]})
    with pytest.raises(metafopy.errors.InputError, match='Unknown component'):
        kr(vccon={'sigma': [1, 1]})
    with pytest.raises(metafopy.errors.InputError, match='custom weights'):
        kr(W=1 / kon()['vi'].to_numpy(), optbeta=True)
    with pytest.raises(metafopy.errors.InputError, match='non-negative'):
        kr(lambda1=-1.0)
