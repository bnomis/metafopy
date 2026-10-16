"""Henmi & Copas confidence interval (metafor's ``hc``).

Henmi & Copas (2010), doi:10.1002/sim.4029. See https://wviechtb.github.io/metafor/reference/hc.html.
"""

import collections.abc
import dataclasses
import math
import sys
import typing

import numpy
import numpy.typing
import scipy.integrate
import scipy.stats

import metafopy.bias.inputs
import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.transf

__all__ = ['HcResult', 'hc']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = sys.float_info.epsilon


@dataclasses.dataclass(frozen=True, slots=True)
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
        d = metafopy.mfopt.option_digits(digits, self.digits)
        cols: list[tuple[str, list[str]]] = [
            ('method', [self.method_rma, self.method]),
            ('tau2', [metafopy.bias.inputs.fx(self.tau2_rma, d), metafopy.bias.inputs.fx(self.tau2, d)]),
            ('estimate', [metafopy.bias.inputs.fx(self.beta_rma, d), metafopy.bias.inputs.fx(self.beta, d)]),
        ]
        if not math.isnan(self.se):
            cols.append(('se', [metafopy.bias.inputs.fx(self.se_rma, d), metafopy.bias.inputs.fx(self.se, d)]))
        cols += [
            ('ci.lb', [metafopy.bias.inputs.fx(self.ci_lb_rma, d), metafopy.bias.inputs.fx(self.ci_lb, d)]),
            ('ci.ub', [metafopy.bias.inputs.fx(self.ci_ub_rma, d), metafopy.bias.inputs.fx(self.ci_ub, d)]),
        ]
        lines = [line + ' ' for line in metafopy.methods.blup.format_r_frame(cols, ['rma', 'hc'])]
        return '\n' + '\n'.join(lines) + '\n'

    def __str__(self) -> str:
        return self.summary()


def hc(
    obj: metafopy.models.rma_uni.RMAResult,
    digits: int | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    control: dict[str, typing.Any] | None = None,
) -> HcResult:
    """Henmi-Copas CI for a random-effects model (no moderators); ``control`` takes ``tol`` and ``maxiter`` for the root search.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.fitview.check_method(obj, 'hc')
    metafopy.methods.fitview.check_class(obj, must='rma.uni', notav=('rma.ls', 'rma.gen', 'rma.uni.selmodel'))
    x = obj
    if not x.int_only:
        raise metafopy.errors.InputError('Method only applicable to models without moderators.')
    k = x.k
    if k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    if not x.allvipos:
        raise metafopy.errors.InputError('Cannot use method when one or more sampling variances are non-positive.')
    con = {'tol': EPS**0.25, 'maxiter': 1000}
    con.update({key: v for key, v in (control or {}).items() if key in con})
    level = x.level
    yi, vi = x.yi, x.vi
    wi = 1 / vi
    w1 = float(numpy.sum(wi))
    w2 = float(numpy.sum(wi**2)) / w1
    w3 = float(numpy.sum(wi**3)) / w1
    w4 = float(numpy.sum(wi**4)) / w1
    beta = float(numpy.sum(wi * yi)) / w1
    q = float(numpy.sum(wi * (yi - beta) ** 2))
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
            with numpy.errstate(divide='ignore', invalid='ignore'):
                arg = finv(float(numpy.float64(r) / numpy.float64(xx)))
            return float(scipy.stats.gamma.cdf(arg, a=e**2 / v_, scale=v_ / e) * scipy.stats.norm.pdf(r))

        val, _ = scipy.integrate.quad(integrand, xx, numpy.inf, epsabs=EPS**0.25, epsrel=EPS**0.25, limit=100)
        return float(val) - level / 2

    try:
        t0 = metafopy.core.zeroin.uniroot(eqn, 0.0, 2.0, tol=con['tol'], maxiter=int(con['maxiter'])).root
    except (ValueError, ArithmeticError) as exc:
        raise metafopy.errors.InputError('Error in uniroot().') from exc
    u0 = sdr * t0
    ci_lb = beta - u0 * se
    ci_ub = beta + u0 * se
    beta_rma, se_rma, ci_lb_rma, ci_ub_rma = float(x.beta[0]), float(x.se[0]), float(x.ci_lb[0]), float(x.ci_ub[0])
    if transf is not None:
        tz = x.tau2
        beta, ci_lb, ci_ub, beta_rma, ci_lb_rma, ci_ub_rma = (
            float(metafopy.transf.apply_transf(a, transf, targs, tz)[0]) for a in (beta, ci_lb, ci_ub, beta_rma, ci_lb_rma, ci_ub_rma)
        )
        se = se_rma = math.nan
    lo, hi = metafopy.methods.common.psort(numpy.array([ci_lb]), numpy.array([ci_ub]))
    lo_r, hi_r = metafopy.methods.common.psort(numpy.array([ci_lb_rma]), numpy.array([ci_ub_rma]))
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
        digits=metafopy.mfopt.stored_digits(digits, 4),
    )
