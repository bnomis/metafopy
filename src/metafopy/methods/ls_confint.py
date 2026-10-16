"""Confidence intervals for location-scale models (metafor's ``confint.rma.ls``).

Profile-likelihood CIs for the scale coefficients (Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1562) and Wald-type CIs
for the location coefficients.
"""

import collections.abc
import math
import typing
import warnings

import numpy
import pandas
import scipy.stats

import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.confint
import metafopy.methods.ls_common
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.transf

__all__ = ['confint_rma_ls']

# auditme: accept[literal-duplication] the labels repeat those of the confint tables
COLUMNS = ('estimate', 'ci.lb', 'ci.ub')  # auditme: accept[non-alphabetical-literal] the order follows the metafor column order


def search_limits(
    x: metafopy.models.rma_ls_result.RMALSResult, pos: int, alpha_level: float, control: collections.abc.Mapping[str, typing.Any]
) -> tuple[float, float]:
    """The interval searched for the bounds of the CI of ``alpha[pos]`` (``vc.min`` and ``vc.max``)."""
    vc = float(x.alpha[pos])
    se = float(x.se_alpha[pos])
    if math.isnan(se):
        lo, hi = vc - 10 * abs(vc), vc + 10 * abs(vc)
    else:
        crit = float(scipy.stats.norm.isf(alpha_level / 2))
        lo, hi = max(vc - 10 * abs(vc), vc - 10 * crit * se), min(vc + 10 * abs(vc), vc + 10 * crit * se)
    amin = numpy.broadcast_to(numpy.asarray(x.control.alpha_min, dtype=numpy.float64), (x.q,))
    amax = numpy.broadcast_to(numpy.asarray(x.control.alpha_max, dtype=numpy.float64), (x.q,))
    lo = max(lo, float(amin[pos]))
    hi = min(hi, float(amax[pos]))
    if control.get('vc_min') is not None:
        lo = float(control['vc_min'])
    if control.get('vc_max') is not None:
        hi = float(control['vc_max'])
    return lo, hi


def profile_lrt(x: metafopy.models.rma_ls_result.RMALSResult, pos: int, val: float, objective: float) -> float:
    """``-2 (ll(alpha[pos] = val) - ll) - objective`` from a refit with ``alpha[pos]`` fixed (metafor's ``.profile.rma.ls``).

    Raises:
        metafopy.errors.EstimationError: the refit failed.
    """
    arg = numpy.where(x.alpha_fix, x.alpha, numpy.nan)
    arg[pos] = val
    fit = metafopy.methods.ls_common.refit_ls(x, alpha=arg)
    if fit is None:
        raise metafopy.errors.EstimationError('the model could not be fitted')
    return -2 * (fit.logLik() - x.logLik()) - objective


@metafopy.methods.confint.confint.register
def confint_rma_ls(
    model: metafopy.models.rma_ls_result.RMALSResult,
    *,
    fixed: bool = False,
    alpha: int | None = None,
    level: float | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    tol: float = metafopy.core.zeroin.DEFAULT_TOL,
    maxiter: int = 1000,
    eptries: int = 10,
    xlim: collections.abc.Sequence[float] | None = None,
) -> metafopy.methods.confint.ConfintResult | metafopy.methods.confint.ConfintList:
    """CIs for a location-scale fit (``confint.rma.ls``).

    ``alpha`` (a 0-based position) requests a profile-likelihood CI for that scale coefficient; ``fixed=True`` gives Wald-type
    CIs for the location coefficients (``transf`` and ``targs`` apply to these). With neither, a profile-likelihood CI is
    computed for every estimated scale coefficient and a list is returned (a single result if there is one). ``control`` may
    hold ``vc_min`` and ``vc_max`` (also written ``vc.min`` and ``vc.max``), the interval searched for the bounds, and ``tol``,
    ``maxiter``, ``eptries`` and ``verbose``; its entries replace the keyword arguments of the same name. ``xlim`` (metafor's ``...`` argument) is the interval
    ``(vc_min, vc_max)`` and wins over ``control``.

    Raises:
        metafopy.errors.InputError: an argument is invalid, ``control`` has an unknown entry or the model has no such coefficient.
    """
    x = model
    settings = metafopy.methods.confint.control_values(
        control, {'vc_min': None, 'vc_max': None, 'tol': tol, 'maxiter': maxiter, 'eptries': eptries, 'verbose': False}
    )
    tol, maxiter, eptries = settings['tol'], settings['maxiter'], settings['eptries']
    if x.optbeta:
        raise metafopy.errors.InputError("CI calculation not yet implemented for models fitted with 'optbeta=True'.")
    level_alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    con = {key: value for key, value in settings.items() if key in ('vc_min', 'vc_max')}
    if xlim is not None:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
        con['vc_min'], con['vc_max'] = float(xlim[0]), float(xlim[1])

    if not fixed and alpha is None:
        free = numpy.flatnonzero(~x.alpha_fix)
        if free.size == 0:
            raise metafopy.errors.InputError('No components for which a CI can be obtained.')
        results = [confint_rma_ls(x, alpha=int(pos), level=level, control=settings, tol=tol, maxiter=maxiter, eptries=eptries, xlim=xlim) for pos in free]
        out = [r for r in results if isinstance(r, metafopy.methods.confint.ConfintResult)]
        return out[0] if len(out) == 1 else metafopy.methods.confint.ConfintList(out)

    res_random: pandas.DataFrame | None = None
    lb_sign = ub_sign = ''
    if alpha is not None:
        if bool(numpy.all(x.alpha_fix)):
            raise metafopy.errors.InputError("Model does not contain any estimated 'alpha' components.")
        if not 0 <= alpha < x.q:
            raise metafopy.errors.InputError("No such 'alpha' component in the model.")
        if x.alpha_fix[alpha]:
            raise metafopy.errors.InputError("Specified 'alpha' component was fixed.")
        pos = alpha
        vc = float(x.alpha[pos])
        lo, hi = search_limits(x, pos, level_alpha, con)
        if lo > vc:
            raise metafopy.errors.InputError('Lower bound of interval to be searched must be <= estimated value of component.')
        if hi < vc:
            raise metafopy.errors.InputError('Upper bound of interval to be searched must be >= estimated value of component.')
        objective = float(scipy.stats.chi2.isf(level_alpha, 1))

        def diff(val: float) -> float:
            return profile_lrt(x, pos, val, objective)

        vc_lb = vc_ub = math.nan
        lb_conv = ub_conv = False
        step = abs(lo - vc) / eptries
        for _ in range(eptries):
            res = metafopy.methods.confint.try_profile(diff, lo)
            if not math.isnan(res):
                if res < 0:
                    vc_lb, lb_conv, lb_sign = lo, True, '<'
                else:
                    root, ok = metafopy.methods.confint.find_root(diff, lo, vc, tol, maxiter, check_conv=True)
                    if ok:
                        vc_lb, lb_conv = root, True
                break
            lo += step
        step = abs(hi - vc) / eptries
        for _ in range(eptries):
            res = metafopy.methods.confint.try_profile(diff, hi)
            if not math.isnan(res):
                if res < 0:
                    vc_ub, ub_conv, ub_sign = hi, True, '>'
                else:
                    root, ok = metafopy.methods.confint.find_root(diff, vc, hi, tol, maxiter, check_conv=True)
                    if ok:
                        vc_ub, ub_conv = root, True
                break
            hi -= step
        if not lb_conv:
            warnings.warn('Cannot obtain lower bound of profile likelihood CI due to convergence problems.', stacklevel=2)
        if not ub_conv:
            warnings.warn('Cannot obtain upper bound of profile likelihood CI due to convergence problems.', stacklevel=2)
        label = 'alpha' if x.q == 1 else f'alpha.{pos + 1}'
        res_random = pandas.DataFrame([[vc, vc_lb, vc_ub]], index=[label], columns=COLUMNS)

    res_fixed: pandas.DataFrame | None = None
    if fixed:
        # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
        if x.test in ('knha', 'adhoc', 't'):
            ddf = float(x.ddf) if x.ddf is not None else math.nan
            crit = float(scipy.stats.t.isf(level_alpha / 2, ddf)) if ddf > 0 else math.nan
        else:
            crit = float(scipy.stats.norm.isf(level_alpha / 2))
        beta, lb, ub = x.beta, x.beta - crit * x.se, x.beta + crit * x.se
        if transf is not None:
            beta, lb, ub = (metafopy.transf.apply_transf(a, transf, targs) for a in (beta, lb, ub))
        res_fixed = pandas.DataFrame({'estimate': beta, 'ci.lb': numpy.minimum(lb, ub), 'ci.ub': numpy.maximum(lb, ub)}, index=list(x.coef_names))

    return metafopy.methods.confint.ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign)
