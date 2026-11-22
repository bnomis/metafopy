"""The shared strategies draw only values inside their documented domains."""

import math
import re

import hypothesis
import rstrategies

R_NAME = re.compile(r'^(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*$')


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(p=rstrategies.probabilities())
def test_probabilities_open_interval(p: float) -> None:
    """probabilities() draws from the open interval (0, 1)."""
    assert 0.0 < p < 1.0


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(p=rstrategies.probabilities(open_interval=False))
def test_probabilities_closed_interval(p: float) -> None:
    """probabilities(open_interval=False) draws from [0, 1] and never NaN."""
    assert 0.0 <= p <= 1.0


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(df=rstrategies.df_values())
def test_df_values_finite_positive(df: float) -> None:
    """df_values() draws finite values from 0.5 to 1e6."""
    assert math.isfinite(df)
    assert 0.5 <= df <= 1e6


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(df=rstrategies.df_values(allow_inf=True))
def test_df_values_allow_inf(df: float) -> None:
    """df_values(allow_inf=True) draws values from 0.5 to 1e6, or inf."""
    assert df == math.inf or 0.5 <= df <= 1e6


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats())
def test_finite_floats_default_bounds(x: float) -> None:
    """finite_floats() draws finite values in [-1e12, 1e12]."""
    assert math.isfinite(x)
    assert -1e12 <= x <= 1e12


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-3.0, 7.5))
def test_finite_floats_given_bounds(x: float) -> None:
    """finite_floats(lo, hi) draws finite values in [lo, hi]."""
    assert math.isfinite(x)
    assert -3.0 <= x <= 7.5


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(name=rstrategies.r_identifiers())
def test_r_identifiers_are_syntactic(name: str) -> None:
    """r_identifiers() draws syntactic R names that are not reserved words."""
    assert R_NAME.match(name)
    assert name not in rstrategies.R_RESERVED_WORDS
