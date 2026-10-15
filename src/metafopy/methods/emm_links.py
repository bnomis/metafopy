"""Transformations of the estimates that emmeans can back-transform (``tran`` of ``emmprep``).

The links mirror R's ``make.link`` and emmeans' ``make.tran``, so that ``type='response'`` summaries give the numbers emmeans does.
R sources: https://github.com/rvlenth/emmeans/blob/master/R/transformations.R and
https://github.com/wch/r-source/blob/trunk/src/library/stats/R/family.R. The measure-to-transformation table is that of metafor's
``emmprep``: https://github.com/wviechtb/metafor/blob/master/R/emmprep.r.
"""

import collections.abc
import dataclasses
import math
import sys

import numpy
import numpy.typing
import scipy.special

import metafopy.errors
import metafopy.transf

__all__ = ['LOG_LIKE', 'Link', 'bias_adjusted', 'make_link', 'measure_link']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Func = collections.abc.Callable[[FloatArray], FloatArray]

EPS = sys.float_info.epsilon
THRESH = 30.0
PROBIT_THRESH = -float(scipy.special.ndtri(EPS))
LOG_LIKE = ('log', 'log2', 'log10')
LOG_MEASURES = (
    'CVLN',
    'CVR',
    'CVRC',
    'D2OR',
    'D2ORL',
    'D2ORN',
    'HR',
    'IRLN',
    'IRR',
    'MNLN',
    'MPOR',
    'MPORC',
    'MPORM',
    'MPPETO',
    'MPRR',
    'OR',
    'PETO',
    'PLN',
    'REH',
    'ROM',
    'ROMC',
    'RR',
    'SDLN',
    'VR',
    'VRC',
)
ZCOR_MEASURES = ('ZBIS', 'ZCOR', 'ZPB', 'ZPCOR', 'ZPHI', 'ZSPCOR', 'ZTET')


@dataclasses.dataclass(frozen=True, slots=True)
class Link:
    """A link function ``eta = linkfun(mu)`` with its inverse and the derivative ``d mu / d eta`` (emmeans' ``mu.eta``)."""

    name: str
    linkfun: Func
    linkinv: Func
    mu_eta: Func


def logit_inv(eta: FloatArray) -> FloatArray:
    tmp = numpy.where(eta < -THRESH, EPS, numpy.where(eta > THRESH, 1 / EPS, numpy.exp(numpy.clip(eta, -THRESH - 1, THRESH + 1))))
    return metafopy.transf.arr(tmp / (1 + tmp))


def logit_mu_eta(eta: FloatArray) -> FloatArray:
    opexp = 1 + numpy.exp(numpy.clip(eta, -THRESH - 1, THRESH + 1))
    return metafopy.transf.arr(numpy.where(numpy.abs(eta) > THRESH, EPS, numpy.exp(numpy.clip(eta, -THRESH - 1, THRESH + 1)) / (opexp * opexp)))


def probit_mu_eta(eta: FloatArray) -> FloatArray:
    return metafopy.transf.arr(numpy.maximum(metafopy.transf.dnorm(eta), EPS))


def asin_sqrt(alpha: float) -> Link:
    return Link(
        'asin(sqrt(mu))',
        lambda mu: metafopy.transf.arr(numpy.arcsin(numpy.sqrt(mu / alpha))),
        lambda eta: metafopy.transf.arr(alpha * numpy.sin(numpy.clip(eta, 0, math.pi / 2)) ** 2),
        lambda eta: metafopy.transf.arr(alpha * numpy.sin(2 * numpy.clip(eta, 0, math.pi / 2))),
    )


def make_links() -> dict[str, Link]:
    return {
        'identity': Link('identity', metafopy.transf.arr, metafopy.transf.arr, numpy.ones_like),
        'log': Link(
            'log',
            lambda mu: metafopy.transf.arr(numpy.log(mu)),
            lambda eta: metafopy.transf.arr(numpy.maximum(numpy.exp(eta), EPS)),
            lambda eta: metafopy.transf.arr(numpy.maximum(numpy.exp(eta), EPS)),
        ),
        'log2': Link(
            'log2',
            lambda mu: metafopy.transf.arr(numpy.log2(mu)),
            lambda eta: metafopy.transf.arr(2.0**eta),
            lambda eta: metafopy.transf.arr(math.log(2) * 2.0**eta),
        ),
        'log10': Link(
            'log10',
            lambda mu: metafopy.transf.arr(numpy.log10(mu)),
            lambda eta: metafopy.transf.arr(10.0**eta),
            lambda eta: metafopy.transf.arr(math.log(10) * 10.0**eta),
        ),
        'logit': Link('logit', lambda mu: metafopy.transf.arr(scipy.special.logit(mu)), logit_inv, logit_mu_eta),
        'probit': Link(
            'probit', lambda mu: metafopy.transf.qnorm(mu), lambda eta: metafopy.transf.pnorm(numpy.clip(eta, -PROBIT_THRESH, PROBIT_THRESH)), probit_mu_eta
        ),
        'sqrt': Link(
            'sqrt',
            lambda mu: metafopy.transf.arr(numpy.sqrt(mu)),
            lambda eta: metafopy.transf.arr(numpy.maximum(0, eta) ** 2),
            lambda eta: metafopy.transf.arr(2 * numpy.maximum(0, eta)),
        ),
        'inverse': Link(
            'inverse',
            lambda mu: metafopy.transf.arr(1 / mu),
            lambda eta: metafopy.transf.arr(1 / numpy.maximum(0, eta)),
            lambda eta: metafopy.transf.arr(-1 / numpy.maximum(0, eta) ** 2),
        ),
        '1/mu^2': Link(
            '1/mu^2',
            lambda mu: metafopy.transf.arr(1 / mu**2),
            lambda eta: metafopy.transf.arr(1 / numpy.sqrt(numpy.maximum(0, eta))),
            lambda eta: metafopy.transf.arr(-1 / (2 * numpy.maximum(0, eta) ** 1.5)),
        ),
        'asin.sqrt': asin_sqrt(1.0),
        'r-to-z': Link('r-to-z', metafopy.transf.transf_rtoz, metafopy.transf.transf_ztor, lambda eta: metafopy.transf.arr(1 / numpy.cosh(eta) ** 2)),
        'r2-to-z': Link(
            'r-to-z', metafopy.transf.transf_r2toz, metafopy.transf.transf_ztor2, lambda eta: metafopy.transf.arr(2 * numpy.sinh(eta) / numpy.cosh(eta) ** 3)
        ),
        'ahw': Link('ahw', metafopy.transf.transf_ahw, metafopy.transf.transf_iahw, lambda eta: metafopy.transf.arr(3 * (1 - eta) ** 2)),
        'abt': Link('abt', metafopy.transf.transf_abt, metafopy.transf.transf_iabt, lambda eta: metafopy.transf.arr(1 / (1 - eta))),
    }


LINKS = make_links()


def make_link(name: str) -> Link:
    """The link called ``name`` (``'log'``, ``'logit'``, ``'probit'``, ``'sqrt'``, ``'asin.sqrt'``, ``'r-to-z'``, ...).

    Raises:
        metafopy.errors.InputError: the transformation is unknown.
    """
    if name not in LINKS:
        raise metafopy.errors.InputError(f'Unknown transformation {name!r}; use one of {", ".join(sorted(LINKS))}.')
    return LINKS[name]


def measure_link(measure: str) -> Link | None:
    """The transformation emmprep sets for an effect size measure (``None`` for measures it leaves untransformed)."""
    if measure in LOG_MEASURES:
        return LINKS['log']
    if measure == 'PLO':
        return LINKS['logit']
    if measure == 'PRZ':
        return LINKS['probit']
    # metafor 5.2-1 also meant to set an arcsine square-root transformation for 'PAS', but its call has no effect
    if measure == 'IRS':
        return LINKS['sqrt']
    if measure in ZCOR_MEASURES:
        return LINKS['r-to-z']
    if measure in ('ZR2', 'ZR2F'):
        return LINKS['r2-to-z']
    if measure == 'AHW':
        return LINKS['ahw']
    if measure == 'ABT':
        return LINKS['abt']
    return None


def bias_adjusted(link: Link, sigma: float) -> Link:
    """The link with the second-order bias adjustment of emmeans' ``.make.bias.adj.link`` for an error SD of ``sigma``."""
    sigma22 = sigma**2 / 2

    def der2(eta: FloatArray) -> FloatArray:
        return metafopy.transf.arr(1000 * (link.mu_eta(eta + 5e-04) - link.mu_eta(eta - 5e-04)))

    return Link(
        link.name,
        link.linkfun,
        lambda eta: metafopy.transf.arr(link.linkinv(eta) + sigma22 * der2(eta)),
        lambda eta: metafopy.transf.arr(link.mu_eta(eta) + 1000 * sigma22 * (der2(eta + 5e-04) - der2(eta - 5e-04))),
    )
