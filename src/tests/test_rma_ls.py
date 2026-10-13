"""Location-scale models (``rma_uni(..., scale=)``) checked against R metafor 5.2-1 (fixtures from tests/r/generate_location_scale.R)."""

import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase5' / 'location_scale.json').read_text(encoding='utf-8'))

CLOSED = 1e-6
OPT = 1e-4

type Fit = typing.Callable[[], metafopy.RMALSResult]

MODS3 = '~ ni100 + meta'
SCALE3 = '~ ni100 + imag'


def flat(value: typing.Any) -> list[float]:
    """A (possibly nested, possibly unboxed) JSON matrix as a flat float array; null is NaN."""
    return numpy.array([numpy.nan if v is None else v for v in numpy.array(value, dtype=object).ravel()], dtype=float).tolist()


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


def bcg6() -> pandas.DataFrame:
    return bcg().iloc[[0, 2, 4, 7, 9, 11]].reset_index(drop=True)


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


FITS: dict[str, Fit] = {
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
    'bcg6_exact': lambda: ls(bcg6(), '~ ablat', '~ ablat'),
    'mo_int': lambda: ls(mo()),
    'mo_quality': lambda: ls(mo(), '~ quality', '~ quality', method='ML'),
    'mo_design': lambda: ls(mo(), None, '~ design'),
    'mc_factors': lambda: ls(mc(), '~ type + struct', '~ struct', method='ML'),
}

# The scale coefficients are optimizer output (R: nlminb, here BFGS plus a polish): they agree to about 1e-7, the quantities derived from
# them (location estimates, Hessian-based standard errors) to 1e-4. The log-likelihood at the optimum is closed-form given the optimum.
# Fits whose likelihood is flat in the scale coefficients (tau^2 -> 0 for some studies) only have their log-likelihood compared.
FLAT = {'bcg6_exact', 'mo_design'}
# column names that come from formulaic's rendering of I() terms or from an unnamed matrix, not from the algorithm
UNNAMED = {'bcg_year_identity', 'bd_matrix', 'bd_poly'}


@pytest.mark.parametrize('name', sorted(set(FITS) - FLAT))
def test_fit(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=OPT, atol=1e-7)
    refutil.close(res.alpha, ref['alpha'], f'{name} alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.se, ref['se'], f'{name} se', rtol=OPT, atol=1e-7)
    refutil.close(res.vb.ravel(), flat(ref['vb']), f'{name} vb', rtol=OPT, atol=1e-8)
    refutil.close(res.va.ravel(), flat(ref['va']), f'{name} va', rtol=OPT, atol=1e-6)
    refutil.close(res.tau2, ref['tau2'], f'{name} tau2', rtol=OPT, atol=1e-8)
    refutil.close(res.pval, ref['pval'], f'{name} pval', rtol=OPT, atol=1e-7)
    refutil.close(res.ci_lb, ref['ci_lb'], f'{name} ci_lb', rtol=OPT, atol=1e-6)
    refutil.close(res.ci_ub, ref['ci_ub'], f'{name} ci_ub', rtol=OPT, atol=1e-6)
    refutil.close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{name} fit ML', rtol=CLOSED, atol=1e-6)
    refutil.close(res.fit_stats_reml.as_tuple(), ref['fit_reml'], f'{name} fit REML', rtol=CLOSED, atol=1e-6)
    refutil.close(res.QE, ref['QE'], f'{name} QE', rtol=1e-8)
    refutil.close(res.QM, ref['QM'], f'{name} QM', rtol=OPT, atol=1e-7)
    refutil.close(res.QMp, ref['QMp'], f'{name} QMp', rtol=OPT, atol=1e-7)
    refutil.close(res.vt, ref['vt'], f'{name} vt', rtol=1e-8)
    refutil.close(res.I2, ref['I2'], f'{name} I2', rtol=OPT, atol=1e-6)
    refutil.close(res.H2, ref['H2'], f'{name} H2', rtol=OPT, atol=1e-6)
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


@pytest.mark.parametrize('name', sorted(FLAT))
def test_fit_flat_likelihood(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    refutil.close(res.fit_stats_reml.ll, ref['fit_reml'][0], f'{name} ll', rtol=1e-6)
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('name', ['bd_int', 'bd_ni', 'bd_3', 'bd_3_t', 'bd_3_alpha', 'bd_na', 'bd_optbeta'])
def test_print(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    assert refutil.r_print(str(res)) == refutil.r_print(ref['print'])
    assert refutil.r_print(res.summary(showfit=True)) == refutil.r_print(ref['print_showfit'])
