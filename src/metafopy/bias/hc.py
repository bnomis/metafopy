"""Henmi & Copas confidence interval (metafor's ``hc``).

Henmi & Copas (2010), doi:10.1002/sim.4029. See https://wviechtb.github.io/metafor/reference/hc.html.
"""

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

from metafopy.bias._inputs import fx
from metafopy.core.zeroin import uniroot
from metafopy.methods._common import psort
from metafopy.methods.blup import format_r_frame
from metafopy.models.rma_uni import RMAResult
from metafopy.transf import apply_transf

__all__ = ['HcResult', 'hc']

type FloatArray = NDArray[np.float64]

_EPS = sys.float_info.epsilon


@dataclass(frozen=True, slots=True)
class HcResult:
    beta: float
    se: float
    ci_lb: float
    ci_ub: float
    beta_rma: float
    se_rma: float
    ci_lb_rma: float
    ci_ub_rma: float
    method: str
    method_rma: str
    tau2: float
    tau2_rma: float
    digits: int = 4

    def summary(self, digits: int | None = None) -> str:
        d = self.digits if digits is None else digits
        cols: list[tuple[str, list[str]]] = [
            ('method', [self.method_rma, self.method]),
            ('tau2', [fx(self.tau2_rma, d), fx(self.tau2, d)]),
            ('estimate', [fx(self.beta_rma, d), fx(self.beta, d)]),
        ]
        if not math.isnan(self.se):
            cols.append(('se', [fx(self.se_rma, d), fx(self.se, d)]))
        cols += [('ci.lb', [fx(self.ci_lb_rma, d), fx(self.ci_lb, d)]), ('ci.ub', [fx(self.ci_ub_rma, d), fx(self.ci_ub, d)])]
        lines = [line + ' ' for line in format_r_frame(cols, ['rma', 'hc'])]
        return '\n' + '\n'.join(lines) + '\n'

    def __str__(self) -> str:
        return self.summary()


def hc(
    obj: RMAResult,
    digits: int | None = None,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    control: dict[str, Any] | None = None,
) -> HcResult:
    """Henmi-Copas CI for a random-effects model (no moderators); ``control`` takes ``tol`` and ``maxiter`` for the root search."""
    x = obj
    if not x.int_only:
        raise ValueError('Method only applicable to models without moderators.')
    k = x.k
    if k == 1:
        raise ValueError('Stopped because k = 1.')
    if not x.allvipos:
        raise ValueError('Cannot use method when one or more sampling variances are non-positive.')
    con = {'tol': _EPS**0.25, 'maxiter': 1000}
    con.update({key: v for key, v in (control or {}).items() if key in con})
    level = x.level
    yi, vi = x.yi, x.vi
    wi = 1 / vi
    w1 = float(np.sum(wi))
    w2 = float(np.sum(wi**2)) / w1
    w3 = float(np.sum(wi**3)) / w1
    w4 = float(np.sum(wi**4)) / w1
    beta = float(np.sum(wi * yi)) / w1
    q = float(np.sum(wi * (yi - beta) ** 2))
    tau2 = max(0.0, (q - (k - 1)) / (w1 - w2))
    vb = (tau2 * w2 + 1) / w1
    se = math.sqrt(vb)
    vr = 1 + tau2 * w2
    sdr = math.sqrt(vr)

    def eq(r: float) -> float:
        return (k - 1) + tau2 * (w1 - w2) + (tau2**2) * ((1 / vr**2) * r**2 - 1 / vr) * (w3 - w2**2)

    def vq(r: float) -> float:
        rsq = r**2
        rv2 = 1 / vr**2
        return (
            2 * (k - 1)
            + 4 * tau2 * (w1 - w2)
            + 2 * tau2**2 * (w1 * w2 - 2 * w3 + w2**2)
            + 4 * tau2**2 * (rv2 * rsq - 1 / vr) * (w3 - w2**2)
            + 4 * tau2**3 * (rv2 * rsq - 1 / vr) * (w4 - 2 * w2 * w3 + w2**3)
            + 2 * tau2**4 * (rv2 - 2 * (1 / vr**3) * rsq) * (w3 - w2**2) ** 2
        )

    def finv(f: float) -> float:
        return (w1 / w2 - 1) * (f**2 - 1) + (k - 1)

    def eqn(xx: float) -> float:
        def integrand(r: float) -> float:
            rr = sdr * r
            e, v_ = eq(rr), vq(rr)
            with np.errstate(divide='ignore', invalid='ignore'):
                arg = finv(float(np.float64(r) / np.float64(xx)))
            return float(stats.gamma.cdf(arg, a=e**2 / v_, scale=v_ / e) * stats.norm.pdf(r))

        val, _ = integrate.quad(integrand, xx, np.inf, epsabs=_EPS**0.25, epsrel=_EPS**0.25, limit=100)
        return float(val) - level / 2

    try:
        t0 = uniroot(eqn, 0.0, 2.0, tol=con['tol'], maxiter=int(con['maxiter'])).root
    except (ValueError, ArithmeticError) as exc:
        raise ValueError('Error in uniroot().') from exc
    u0 = sdr * t0
    ci_lb = beta - u0 * se
    ci_ub = beta + u0 * se
    beta_rma, se_rma, ci_lb_rma, ci_ub_rma = float(x.beta[0]), float(x.se[0]), float(x.ci_lb[0]), float(x.ci_ub[0])
    if transf is not None:
        tz = x.tau2
        beta, ci_lb, ci_ub, beta_rma, ci_lb_rma, ci_ub_rma = (
            float(apply_transf(a, transf, targs, tz)[0]) for a in (beta, ci_lb, ci_ub, beta_rma, ci_lb_rma, ci_ub_rma)
        )
        se = se_rma = math.nan
    lo, hi = psort(np.array([ci_lb]), np.array([ci_ub]))
    lo_r, hi_r = psort(np.array([ci_lb_rma]), np.array([ci_ub_rma]))
    return HcResult(
        beta=beta,
        se=se,
        ci_lb=float(lo[0]),
        ci_ub=float(hi[0]),
        beta_rma=beta_rma,
        se_rma=se_rma,
        ci_lb_rma=float(lo_r[0]),
        ci_ub_rma=float(hi_r[0]),
        method='DL',
        method_rma=x.method,
        tau2=tau2,
        tau2_rma=x.tau2,
        digits=4 if digits is None else digits,
    )
