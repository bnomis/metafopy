"""Property tests of the mapping helpers of ``metafopy.models.sel_lik``."""

import math

import hypothesis
import hypothesis.strategies
import pytest
import rstrategies

import metafopy.errors
import metafopy.models.sel_lik

NAMES = ('I', 'exp', 'log', 'plogis', 'qlogis')
bounds = hypothesis.strategies.tuples(rstrategies.finite_floats(-100.0, 100.0), hypothesis.strategies.floats(min_value=0.1, max_value=100.0)).map(
    lambda t: (t[0], t[0] + t[1])
)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-700.0, 700.0))
def test_apply_named_exp_log_round_trip(x: float) -> None:
    """``apply_named``: log(exp(x)) recovers x."""
    e = metafopy.models.sel_lik.apply_named('exp', x)
    assert metafopy.models.sel_lik.apply_named('log', e) == pytest.approx(x, rel=1e-9, abs=1e-9)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=hypothesis.strategies.floats(min_value=1e-300, max_value=1e300))
def test_apply_named_log_exp_round_trip(x: float) -> None:
    """``apply_named``: exp(log(x)) recovers x for x > 0 within the range where exp does not overflow."""
    lg = metafopy.models.sel_lik.apply_named('log', x)
    hypothesis.assume(lg < 709.0)
    assert metafopy.models.sel_lik.apply_named('exp', lg) == pytest.approx(x, rel=1e-9)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-15.0, 15.0))
def test_apply_named_logit_round_trip_and_identity(x: float) -> None:
    """``apply_named``: qlogis(plogis(x)) recovers x, and I is the identity."""
    p = metafopy.models.sel_lik.apply_named('plogis', x)
    assert 0.0 < p < 1.0
    assert metafopy.models.sel_lik.apply_named('qlogis', p) == pytest.approx(x, abs=1e-8)
    assert metafopy.models.sel_lik.apply_named('I', x) == x


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(name=hypothesis.strategies.text(max_size=8).filter(lambda s: s not in NAMES), x=rstrategies.finite_floats())
def test_apply_named_rejects_unknown_names(name: str, x: float) -> None:
    """``apply_named``: an unknown function name raises InputError."""
    with pytest.raises(metafopy.errors.InputError, match='unknown mapping function'):
        metafopy.models.sel_lik.apply_named(name, x)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1000.0, 1000.0), y=rstrategies.finite_floats(-1000.0, 1000.0))
def test_exp_safe(x: float, y: float) -> None:
    """``exp_safe``: non-negative, math.exp below 709, inf from there and non-decreasing."""
    ex = metafopy.models.sel_lik.exp_safe(x)
    assert ex >= 0.0
    if x < 709.0:
        assert ex == math.exp(x)
    else:
        assert ex == math.inf
    lo, hi = min(x, y), max(x, y)
    assert metafopy.models.sel_lik.exp_safe(lo) <= metafopy.models.sel_lik.exp_safe(hi)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-30.0, 30.0), lb_ub=bounds)
def test_map_delta_within_bounds(x: float, lb_ub: tuple[float, float]) -> None:
    """``map_delta``: for lb < ub the output lies within [lb, ub] (4 ulps of slack), by default and for a named map."""
    lb, ub = lb_ub
    slack = 4 * math.ulp(max(abs(lb), abs(ub)))
    for fun in (None, 'I', 'plogis'):
        v = metafopy.models.sel_lik.map_delta(x, lb, ub, fun)
        assert lb - slack <= v <= ub + slack


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-10.0, 10.0), lb_ub=bounds)
def test_map_delta_map_inverse_recover_x(x: float, lb_ub: tuple[float, float]) -> None:
    """``map_delta``: map_inverse(map_delta(x)) recovers x where the logistic map does not saturate."""
    lb, ub = lb_ub
    y = metafopy.models.sel_lik.map_delta(x, lb, ub, None)
    assert metafopy.models.sel_lik.map_inverse(y, lb, ub, None) == pytest.approx(x, abs=1e-6)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(frac=hypothesis.strategies.floats(min_value=1e-3, max_value=1 - 1e-3), lb_ub=bounds)
def test_map_inverse_round_trip(frac: float, lb_ub: tuple[float, float]) -> None:
    """``map_inverse``: for x strictly inside (lb, ub), map_delta of the result recovers x, for the default and the named maps."""
    lb, ub = lb_ub
    x = lb + frac * (ub - lb)
    tol = 1e-8 * max(1.0, abs(lb), abs(ub))
    assert metafopy.models.sel_lik.map_delta(metafopy.models.sel_lik.map_inverse(x, lb, ub, None), lb, ub, None) == pytest.approx(x, abs=tol)
    assert metafopy.models.sel_lik.map_delta(metafopy.models.sel_lik.map_inverse(x, lb, ub, 'I'), lb, ub, 'I') == pytest.approx(x, abs=tol)
    q = metafopy.models.sel_lik.map_inverse(frac, 0.0, 1.0, 'qlogis')
    assert metafopy.models.sel_lik.map_delta(q, 0.0, 1.0, 'plogis') == pytest.approx(frac, abs=1e-8)
