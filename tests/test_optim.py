"""The ports of R's optimize(), optim(method="Nelder-Mead") and optimhess must reproduce R's iterates."""

from __future__ import annotations

import math

import numpy as np

from metafopy.core.bvn import bvn_rect
from metafopy.core.optim import fmin, nelder_mead, optimhess

INF = math.inf


def test_fmin_matches_r() -> None:
    # R: optimize(function(x) (x-1/3)^2 + sin(5*x), c(-1, 2))$minimum
    assert fmin(lambda x: (x - 1 / 3) ** 2 + math.sin(5 * x), -1, 2) == 0.89699447458266623


def _rosenbrock(x: np.ndarray) -> float:
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def test_nelder_mead_matches_r() -> None:
    # R: optim(c(-1.2, 1), fr, hessian=TRUE) with fr the Rosenbrock function
    res = nelder_mead(_rosenbrock, [-1.2, 1.0])
    np.testing.assert_allclose(res.par, [1.0002601387256695, 1.0005059993037650], rtol=1e-14)
    assert math.isclose(res.value, 8.8252410967227472e-08, rel_tol=1e-12)
    assert res.fncount == 195
    assert res.fail == 0
    hess = optimhess(_rosenbrock, res.par)
    np.testing.assert_allclose(hess, [[802.42281442667115, -400.10405549025830], [-400.10405549025830, 199.99999999998931]], rtol=1e-10)


def test_bvn_matches_mvtnorm() -> None:
    # R: mvtnorm::pmvnorm(..., corr=matrix(c(1, r, r, 1), 2))
    assert math.isclose(bvn_rect((-INF, -INF), (0.3, -0.2), 0.5), 0.33619843701551877, rel_tol=1e-14)
    assert math.isclose(bvn_rect((0.3, -INF), (INF, -0.2), 0.95), 0.0030896714018696747, rel_tol=1e-13)
    assert math.isclose(bvn_rect((-INF, -0.2), (0.3, INF), -0.97), 0.5576283936677815, rel_tol=1e-14)
    assert math.isclose(bvn_rect((0.3, -0.2), (INF, INF), 0.1), 0.23621914480234804, rel_tol=1e-14)


def test_bvn_degenerate_limits() -> None:
    assert bvn_rect((-INF, -INF), (INF, INF), 0.3) == 1.0
    assert math.isclose(bvn_rect((-INF, -1.0), (INF, 1.0), 0.3), 0.6826894921370859, rel_tol=1e-14)
