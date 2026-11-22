"""summary_escalc() against metafor's summary.escalc() (cases in src/tests/reference/extras/transf.json)."""

import json
import pathlib
import typing

import hypothesis
import hypothesis.strategies
import numpy
import pandas
import pytest

import metafopy
import metafopy.effects.summary
import metafopy.errors
import metafopy.transf

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'extras' / 'transf.json').read_text())['summary']


def arr(v: typing.Any) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    return numpy.array([numpy.nan if e is None else e for e in numpy.atleast_1d(v)], dtype=numpy.float64)


def names(v: typing.Any) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def check(res: pandas.DataFrame, ref: dict[str, typing.Any], rtol: float = 1e-10) -> None:
    assert list(res.columns) == ref['columns']
    for col in ref['columns']:
        numpy.testing.assert_allclose(res[col].to_numpy(float), arr(ref[col]), rtol=rtol, atol=1e-14, equal_nan=True, err_msg=col)
    for key in ('yi_names', 'vi_names', 'sei_names'):
        assert res.attrs[key] == names(ref['attrs'][key])
    assert res.attrs['ci_lb_names'] == names(ref['attrs']['ci_lb_names'])


def escalc_frame(spec: dict[str, typing.Any]) -> pandas.DataFrame:
    kwargs: dict[str, typing.Any] = {k: (v if k in ('measure', 'var_names') else arr(v)) for k, v in spec.items()}
    return metafopy.escalc(**kwargs)


@pytest.fixture(scope='module')
def pft() -> pandas.DataFrame:
    return escalc_frame(REF['pft']['input'])


@pytest.fixture(scope='module')
def odds() -> pandas.DataFrame:
    return escalc_frame(REF['or']['input'])


@pytest.fixture(scope='module')
def zcor() -> pandas.DataFrame:
    return escalc_frame(REF['zcor']['input'])


def test_default(pft: pandas.DataFrame) -> None:
    res = metafopy.effects.summary.summary_escalc(pft)
    check(res, REF['pft']['default'])
    assert res.attrs['measure'] == 'PFT'


def test_level_and_h0(pft: pandas.DataFrame) -> None:
    check(metafopy.effects.summary.summary_escalc(pft, level=90, H0=0.3), REF['pft']['level90_H0'])


def test_transf_with_per_study_argument(pft: pandas.DataFrame) -> None:
    numpy.testing.assert_array_equal(pft.attrs['ni'], arr(REF['pft']['ni']))
    res = metafopy.effects.summary.summary_escalc(pft, transf=metafopy.transf.ipft, ni=pft.attrs['ni'])
    check(res, REF['pft']['transf_ipft'])
    assert 'measure' not in res.attrs
    check(metafopy.effects.summary.summary_escalc(pft, transf=metafopy.transf.ipft, ni=pft.attrs['ni'], append=False), REF['pft']['transf_noappend'])


def test_out_names_and_no_append(pft: pandas.DataFrame) -> None:
    res = metafopy.effects.summary.summary_escalc(pft, out_names=('se', 'z', 'p', 'lo', 'hi'))
    check(res, REF['pft']['out_names'])
    check(metafopy.effects.summary.summary_escalc(pft, append=False), REF['pft']['noappend'])


def test_transf_exp_olim_and_integral(odds: pandas.DataFrame) -> None:
    check(metafopy.effects.summary.summary_escalc(odds, transf=numpy.exp), REF['or']['transf_exp'])
    check(metafopy.effects.summary.summary_escalc(odds, transf=numpy.exp, olim=(1, 0.5)), REF['or']['olim'])
    check(metafopy.effects.summary.summary_escalc(odds, transf=metafopy.transf.exp_int, targs=0.2), REF['or']['exp_int'])


def test_replace_false_fills_only_missing(odds: pandas.DataFrame) -> None:
    s1 = metafopy.effects.summary.summary_escalc(odds)
    s1.loc[1, 'sei'] = numpy.nan
    s1.loc[[0, 2], 'ci.lb'] = numpy.nan
    check(s1, REF['or']['replace_false_input'])
    check(metafopy.effects.summary.summary_escalc(s1, replace=False, level=80), REF['or']['replace_false'])


def test_var_names_from_attrs(zcor: pandas.DataFrame) -> None:
    check(metafopy.effects.summary.summary_escalc(zcor, transf=metafopy.transf.ztor), REF['zcor']['ztor'])
    check(metafopy.effects.summary.summary_escalc(zcor, transf=metafopy.transf.ztor_int, targs=0.05), REF['zcor']['ztor_int'], rtol=1e-6)


def test_errors(pft: pandas.DataFrame) -> None:
    with pytest.raises(ValueError, match='length 5'):
        # auditme: accept[literal-duplication] test cases spell out their own R arguments
        metafopy.effects.summary.summary_escalc(pft, out_names=('a', 'b'))
    with pytest.raises(ValueError, match='between 0 and 100'):
        metafopy.effects.summary.summary_escalc(pft, level=120)
    with pytest.raises(ValueError, match='Cannot find'):
        metafopy.effects.summary.summary_escalc(pft, var_names=('yi', 'nope'))
    with pytest.raises(ValueError, match="'olim'"):
        metafopy.effects.summary.summary_escalc(pft, olim=(0,))


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(level=hypothesis.strategies.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_summary_level_to_alpha_is_a_probability(level: float) -> None:
    """effects/summary.py::level_to_alpha: 0 gives 1, 1 to 100 a percent, up to 0.5 an alpha already, above 0.5 a proportion; always in [0, 1]."""
    alpha = metafopy.effects.summary.level_to_alpha(level)
    assert 0.0 <= alpha <= 1.0
    if level == 0:
        assert alpha == 1.0
    elif level >= 1:
        assert alpha == pytest.approx((100 - level) / 100, abs=1e-12)
    elif level > 0.5:
        assert alpha == pytest.approx(1 - level, abs=1e-12)
    else:
        assert alpha == level


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(prop=hypothesis.strategies.floats(min_value=0.51, max_value=0.99, allow_nan=False))
def test_summary_level_to_alpha_percent_matches_proportion(prop: float) -> None:
    """effects/summary.py::level_to_alpha: 95 and 0.95 give the same alpha."""
    assert metafopy.effects.summary.level_to_alpha(100 * prop) == pytest.approx(metafopy.effects.summary.level_to_alpha(prop), abs=1e-12)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    level=hypothesis.strategies.floats(min_value=100.0, max_value=1e9, exclude_min=True, allow_nan=False)
    | hypothesis.strategies.floats(min_value=-1e9, max_value=0.0, exclude_max=True, allow_nan=False)
)
def test_summary_level_to_alpha_rejects_levels_outside_0_to_100(level: float) -> None:
    """effects/summary.py::level_to_alpha: a level above 100 or below 0 raises InputError."""
    with pytest.raises(metafopy.errors.InputError):
        metafopy.effects.summary.level_to_alpha(level)
