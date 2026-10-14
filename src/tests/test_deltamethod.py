"""deltamethod() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R, based on the help page examples)."""

import collections.abc
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF = refutil.load_phase7('deltamethod')
B = numpy.array(REF['b'])
VB = numpy.array(REF['vb'])
BZ = numpy.array(REF['bz'])
VBZ = numpy.array(REF['vbz'])
# auditme: accept[literal-duplication] test cases spell out their own R arguments
NAMES = ['acog', 'asom', 'conf']


def regression(r: numpy.typing.NDArray[numpy.float64]) -> pandas.Series:
    """The standardized regression coefficients of perf on acog, asom and conf from the correlations ``r``."""
    R = numpy.asarray(metafopy.vec2mat(r))
    return pandas.Series(numpy.linalg.solve(R[1:, 1:], R[1:4, 0]), index=NAMES)


def regression_args(r1: float, r2: float, r3: float, r4: float, r5: float, r6: float) -> pandas.Series:
    return regression(numpy.array([r1, r2, r3, r4, r5, r6]))


def ztor_diff(z: numpy.typing.NDArray[numpy.float64]) -> float:
    return float(numpy.tanh(z[0]) - numpy.tanh(z[1]))


def ztor_diff_args(z1: float, z2: float, z3: float, z4: float, z5: float, z6: float) -> float:
    return float(numpy.tanh(z1) - numpy.tanh(z2))


def check(key: str, res: metafopy.DeltaMethodResult, rtol: float = 1e-8) -> None:
    ref = REF[key]
    coef = res.coef()
    assert list(coef.index) == refutil.as_list(ref['names'])
    refutil.close(coef.to_numpy(), ref['coef'], f'{key}: coef', rtol=rtol)
    refutil.close(res.vcov().to_numpy().reshape(-1), numpy.array(ref['vcov']).reshape(-1).tolist(), f'{key}: vcov', rtol=rtol, atol=1e-14)
    # auditme: accept[literal-duplication] the columns of the R table, also listed by other tests
    for col in ('se', 'zval', 'pval', 'ci_lb', 'ci_ub'):
        refutil.close(res.tab[col].to_numpy(), ref[col], f'{key}: {col}', rtol=rtol, atol=1e-14)
    expected = '\n'.join(ref['print']).replace('\u2018', "'").replace('\u2019', "'")
    assert refutil.same_print(str(res), expected), f'{key}:\n{res}\n---\n{expected}'


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    ('key', 'fun', 'kwargs'),
    [
        ('reg_args', regression_args, {}),
        ('reg_vec', regression, {}),
        ('reg_vec_o2', regression, {'order': 2}),
        ('reg_vec_o3', regression, {'order': 3}),
        ('reg_level', regression, {'level': 90, 'H0': [0.1, 0.2, 0.3], 'digits': 3}),
    ],
)
def test_regression_coefficients(key: str, fun: collections.abc.Callable[..., typing.Any], kwargs: dict[str, typing.Any]) -> None:
    check(key, metafopy.deltamethod(B, VB, fun=fun, **kwargs))


@pytest.mark.parametrize(
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    ('key', 'fun', 'kwargs'),
    [
        ('diff_args', ztor_diff_args, {}),
        ('diff_vec', ztor_diff, {}),
        ('diff_o2', ztor_diff, {'order': 2}),
        ('diff_o3', ztor_diff, {'order': 3}),
    ],
)
def test_correlation_difference(key: str, fun: collections.abc.Callable[..., typing.Any], kwargs: dict[str, typing.Any]) -> None:
    check(key, metafopy.deltamethod(BZ, VBZ, fun=fun, **kwargs))


def test_model() -> None:
    dat = refutil.bcg()
    res = metafopy.rma('yi', 'vi', data=dat, mods='~ ablat')
    check('model', metafopy.deltamethod(res, fun=lambda b: {'ratio': b[1] / b[0], 'expb0': numpy.exp(b[0])}))
    check('model_o2', metafopy.deltamethod(res, fun=lambda b0, b1: numpy.exp(b0 + 30 * b1), order=2))


def test_scalar_and_diagonal() -> None:
    check('scalar', metafopy.deltamethod(0.5, 0.04, fun=numpy.exp))
    check('scalar_o3', metafopy.deltamethod(0.5, 0.04, fun=lambda x: numpy.exp(x) + x**3, order=3))
    check('diagvec', metafopy.deltamethod([0.2, 1.5], [0.01, 0.09], fun=lambda a, b: pandas.Series({'prod': a * b, 'quot': a / b})))


def test_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'fun'"):
        metafopy.deltamethod(B, VB)
    with pytest.raises(metafopy.errors.InputError, match="'order'"):
        metafopy.deltamethod(B, VB, fun=regression, order=4)
    with pytest.raises(metafopy.errors.InputError, match='function arguments'):
        metafopy.deltamethod(B, VB, fun=lambda a, b: a + b)
    with pytest.raises(metafopy.errors.InputError, match="'vcov'"):
        metafopy.deltamethod(B, VB[:3, :3], fun=regression)
    with pytest.raises(metafopy.errors.InputError, match="'H0'"):
        metafopy.deltamethod(B, VB, fun=regression, H0=[0, 1])
    with pytest.raises(metafopy.errors.EstimationError, match='applying the function'):
        metafopy.deltamethod(B, VB, fun=lambda r: r[10])
