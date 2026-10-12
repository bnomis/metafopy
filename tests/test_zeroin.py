"""The uniroot port must reproduce R's iterates exactly, not just find a root."""

from __future__ import annotations

import math

import pytest

from metafopy.core.zeroin import uniroot


def test_matches_r_default_tolerance() -> None:
    # R: uniroot(function(x) x^2 - 2, c(0, 2))
    res = uniroot(lambda x: x * x - 2, 0, 2)
    assert res.root == 1.4142133199955025
    assert res.iterations == 6


def test_matches_r_tight_tolerance() -> None:
    # R: uniroot(function(x) cos(x) - x, c(-1, 3), tol=1e-12)
    res = uniroot(lambda x: math.cos(x) - x, -1, 3, tol=1e-12)
    assert res.root == 0.73908513321526159
    assert res.iterations == 8


def test_root_at_endpoint() -> None:
    assert uniroot(lambda x: x, 0, 1).root == 0


def test_same_sign_rejected() -> None:
    with pytest.raises(ValueError, match='opposite sign'):
        uniroot(lambda x: x * x + 1, -1, 1)


def test_maxiter_reports_nonconvergence() -> None:
    res = uniroot(lambda x: x * x - 2, 0, 2, tol=1e-15, maxiter=2)
    assert not res.converged
