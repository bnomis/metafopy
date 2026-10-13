"""Numerical parity of rma_uni with R metafor across all fixture cases."""

import typing

import numpy
import pytest

import metafopy
import metafopy.datasets

import reference_data

RTOL = 1e-7
ATOL = 1e-9


def fits() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for case in reference_data.case_names():
        for i, _ in enumerate(reference_data.load_case(case)['fits']):
            out.append((case, i))
    return out


def case_id(param: tuple[str, int]) -> str:
    case, i = param
    f = reference_data.load_case(case)['fits'][i]
    return f'{case}-{f["method"]}-{f["test"]}-{"mods" if f["mods"] else "int"}'


def arr(value: typing.Any) -> numpy.ndarray:
    return numpy.atleast_1d(numpy.asarray(value, dtype=float))


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL) -> None:
    numpy.testing.assert_allclose(arr(actual), arr(expected), rtol=rtol, atol=ATOL, equal_nan=True, err_msg=what)


FITS = fits()


@pytest.mark.parametrize('param', FITS, ids=[case_id(p) for p in FITS])
def test_rma_uni_matches_metafor(param: tuple[str, int]) -> None:
    case, i = param
    fx = reference_data.load_case(case)
    spec = fx['fits'][i]
    ref = spec['result']
    dat = metafopy.datasets.load(fx['dataset'])
    dat['yi'] = arr(fx['yi'])
    dat['vi'] = arr(fx['vi'])

    res = metafopy.rma_uni(data=dat, mods=spec['mods'], method=spec['method'], test=spec['test'])

    assert res.k == ref['k']
    assert res.p == ref['p']
    names = ref['coef_names']
    assert list(res.coef_names) == ([names] if isinstance(names, str) else names)
    for attr in ('beta', 'se', 'zval', 'pval', 'ci_lb', 'ci_ub', 'tau2', 'QE', 'QEp', 'I2', 'H2'):
        close(getattr(res, attr), ref[attr], attr)
    close(res.vb, ref['vb'], 'vb')
    if ref['se_tau2'] is not None:
        close(res.se_tau2, ref['se_tau2'], 'se_tau2')
    close(res.QM, ref['QM'], 'QM')
    close(res.QMp, ref['QMp'], 'QMp')
    if ref['R2'] is None:
        assert res.R2 is None
    else:
        close(res.R2, ref['R2'], 'R2')
    close(res.fit_stats_ml.as_tuple(), ref['fit_stats_ml'], 'fitstats ML')
    close(res.fit_stats_reml.as_tuple(), ref['fit_stats_reml'], 'fitstats REML')
    close(res.weights_(), ref['weights'], 'weights')
    close(res.hatvalues(), ref['hat'], 'hat')

    if 'predict' in ref:
        pr = metafopy.predict(res)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        for attr in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
            close(getattr(pr, attr), ref['predict'][attr], f'predict {attr}')

    if 'confint_random' in ref:
        ci = metafopy.confint(res).random
        assert ci is not None
        for col, key in (('estimate', 'estimate'), ('ci.lb', 'ci_lb'), ('ci.ub', 'ci_ub')):
            close(ci[col].to_numpy(), ref['confint_random'][key], f'confint {col}')
