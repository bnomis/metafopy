"""metafopy.transf against metafor: every case in src/tests/reference/extras/transf.json (see src/tests/r/generate_transf.R)."""

import collections.abc
import functools
import json
import pathlib
import typing
import warnings

import numpy
import pytest

import metafopy
import metafopy.transf

FIXTURE = pathlib.Path(__file__).parent / 'reference' / 'extras' / 'transf.json'

# integral (integrate()) and root/mode (uniroot(), optimize()) based functions;
# transf.exp.int and transf.exp.mode are closed forms in metafor
NUMERIC = {'_int', '_mode'}
CLOSED_FORM = {'exp_int', 'exp_mode'}


@functools.cache
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def cases() -> dict[str, dict[str, typing.Any]]:
    with FIXTURE.open() as fh:
        data: dict[str, dict[str, typing.Any]] = json.load(fh)['cases']
    return data


def num(v: typing.Any) -> float:
    if v is None:
        return numpy.nan
    if isinstance(v, str):
        return {'Inf': numpy.inf, '-Inf': -numpy.inf}[v]
    return float(v)


def decode(v: typing.Any) -> typing.Any:
    if isinstance(v, list):
        return numpy.array([num(e) for e in v], dtype=numpy.float64)
    if isinstance(v, dict):
        return {k: decode(e) for k, e in v.items()}
    if isinstance(v, str):
        return num(v)
    return v


def py_name(name: str) -> str:
    """``transf.ipft.hm`` is ``ipft_hm`` in ``metafopy.transf``."""
    return name.removeprefix('transf.').replace('.', '_')


def func(name: str) -> collections.abc.Callable[..., typing.Any]:
    f: collections.abc.Callable[..., typing.Any] = getattr(metafopy.transf, py_name(name))
    return f


def tolerance(name: str) -> tuple[float, float]:
    py = py_name(name)
    if py not in CLOSED_FORM and any(py.endswith(s) for s in NUMERIC):
        return 1e-6, 1e-12
    return 1e-10, 1e-15


@pytest.mark.parametrize('name', sorted(cases()))
def test_transf_matches_metafor(name: str) -> None:
    case = cases()[name]
    f = func(case['fun'])
    args = {k: decode(v) for k, v in case['args'].items()}

    def call() -> typing.Any:
        if case['each']:
            xi = numpy.atleast_1d(args.pop('xi'))
            return numpy.array([f(x, **args)[0] for x in xi])
        return f(**args)

    if 'error' in case:
        with pytest.raises(ValueError, match=r"'targs'|'tau2'|Length|length|'pi'|'pc'|negative|scalar|sample sizes"):
            call()
        return
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # numpy floating-point warnings must not leak
        out = call()
    expected = numpy.atleast_1d(decode(case['out']))
    rtol, atol = tolerance(case['fun'])
    assert isinstance(out, numpy.ndarray)
    assert out.dtype == numpy.float64
    numpy.testing.assert_allclose(out, expected, rtol=rtol, atol=atol, equal_nan=True)


def test_every_metafor_function_is_covered() -> None:
    covered = {py_name(c['fun']) for c in cases().values()}
    assert covered == set(metafopy.transf.__all__)


def test_top_level_names_are_the_module_functions() -> None:
    # transf.ilogit translates mechanically to metafopy.transf_ilogit, the same function as metafopy.transf.ilogit
    for name in metafopy.transf.__all__:
        assert getattr(metafopy, f'transf_{name}') is getattr(metafopy.transf, name)
        assert f'transf_{name}' in metafopy.__all__


def test_scalars_and_shapes() -> None:
    out = metafopy.transf.ztor(0.5)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(numpy.tanh(0.5))
    grid = numpy.array([[0.1, 0.2], [0.3, 0.4]])
    assert metafopy.transf.logit(grid).shape == (2, 2)
    assert metafopy.transf.ilogit_int(grid, targs={'tau2': 0.1}).shape == (2, 2)
    assert metafopy.transf.ztor_mode(grid, targs=0.1).shape == (2, 2)
    numpy.testing.assert_allclose(metafopy.transf.pft(grid, 10), metafopy.transf.pft(grid.ravel(), 10).reshape(2, 2))


def test_targs_forms_agree() -> None:
    x = numpy.array([-1.0, 0.0, 0.7])
    by_dict = metafopy.transf.ilogit_int(x, targs={'tau2': 0.3, 'lower': -2, 'upper': 3})
    by_seq = metafopy.transf.ilogit_int(x, targs=[0.3, -2, 3])
    numpy.testing.assert_array_equal(by_dict, by_seq)
    numpy.testing.assert_array_equal(metafopy.transf.ztor_mode(x, targs=0.2), metafopy.transf.ztor_mode(x, targs={'tau2': 0.2}))
    # tau2 below machine epsilon is treated as 0 (the plain back-transformation)
    numpy.testing.assert_array_equal(metafopy.transf.ztor_int(x, targs=1e-20), numpy.tanh(x))


def test_mode_with_zero_tau2_is_elementwise() -> None:
    # metafor returns tanh(xi) length(xi) times for vector xi; here the result is elementwise
    x = numpy.array([0.1, 0.2, 0.3])
    numpy.testing.assert_array_equal(metafopy.transf.ztor_mode(x, targs=0), numpy.tanh(x))
    numpy.testing.assert_array_equal(metafopy.transf.iabt_mode(x, targs=0), metafopy.transf.iabt(x))


def test_ipft_hm_accepts_sample_sizes_directly() -> None:
    x = numpy.array([0.2, 0.5])
    numpy.testing.assert_array_equal(metafopy.transf.ipft_hm(x, targs={'ni': [10, 20]}), metafopy.transf.ipft_hm(x, targs=[10, 20]))
