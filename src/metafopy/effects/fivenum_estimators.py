"""Estimators behind the ``'qe'``, ``'bc'``, ``'mln'`` and ``'blue'`` methods of ``conv_fivenum``.

Ports of the R packages estmeansd 1.0.1 (``qe.mean.sd``, ``bc.mean.sd``, ``mln.mean.sd``;
https://cran.r-project.org/package=estmeansd) and metaBLUE 1.0.0 (``Luo.mean``, ``Wan.std``, ``BLUE_s``;
https://cran.r-project.org/package=metaBLUE), which metafor calls (https://github.com/wviechtb/metafor/blob/master/R/conv.fivenum.r):

- QE: McGrath et al. (2020), doi:10.1177/0962280219889080. Fits candidate distributions to the quantiles with ``L-BFGS-B``
  (here SciPy's, with R's finite-difference gradient) and keeps the best fit.
- BC: Box-Cox method of McGrath et al. (2023), doi:10.1002/jrsm.1664. Draws a normal sample with R's generator, so
  results reproduce R's for the same seed.
- MLN: Box-Cox with maximum likelihood, McGrath et al. (2023), doi:10.1002/jrsm.1664. Also uses R's generator.
- BLUE: Yang, Hutson and Wang (2022), doi:10.1080/02664763.2021.1967890, after Lloyd (1952), doi:10.1093/biomet/39.1-2.88.

The quantiles are given for one of three scenarios: ``'S1'`` (minimum, median, maximum), ``'S2'`` (first quartile, median,
third quartile) or ``'S3'`` (all five).
"""

import collections.abc
import dataclasses
import itertools
import math
import warnings

import numpy
import numpy.typing
import scipy.optimize
import scipy.special
import scipy.stats

import metafopy.core.optim
import metafopy.core.rng
import metafopy.errors

__all__ = ['BlueResult', 'Estimate', 'bc_mean_sd', 'blue_s', 'estimate', 'luo_mean', 'mln_mean_sd', 'qe_mean_sd', 'wan_std']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] order is estmeansd's, which breaks ties between equally good fits
DISTRIBUTIONS = ('normal', 'log-normal', 'gamma', 'weibull', 'beta')
INDEX_FRACTIONS = {'S1': (0.0, 0.5, 1.0), 'S2': (0.25, 0.5, 0.75), 'S3': (0.0, 0.25, 0.5, 0.75, 1.0)}


@dataclasses.dataclass(frozen=True)
class Estimate:
    """The estimated mean and SD (and, for QE, the selected distribution)."""

    mean: float
    sd: float
    dist: str | None = None


@dataclasses.dataclass(frozen=True)
class BlueResult:
    """The BLUE of the location and scale from ``blue_s``, with the expected values and covariance matrix of the order statistics."""

    alpha: FloatArray
    B: FloatArray
    muhat: float
    sigmahat: float
    var_mu: float
    var_sigma: float


def qnorm(p: float) -> float:
    return float(scipy.stats.norm.ppf(p))


def sum_stat_index(n: float, kind: str) -> list[int]:
    """The ranks (1-based) of the order statistics a scenario (or ``'tertiles'``, ...) reports."""
    if kind == 'S1':
        return [1, math.floor(n / 2) + 1, int(n)]
    if kind == 'S2':
        return [math.floor(n / 4) + 1, math.floor(n / 2) + 1, math.floor(3 * n / 4) + 1]
    if kind == 'S3':
        return [1, math.floor(n / 4) + 1, math.floor(n / 2) + 1, math.floor(3 * n / 4) + 1, int(n)]
    raise metafopy.errors.InputError(f"Unknown 'type' specified: {kind}.")


def luo_mean(x: collections.abc.Sequence[float] | FloatArray, n: float, kind: str) -> float:
    """metaBLUE's ``Luo.mean``: Luo et al. (2018), doi:10.1177/0962280216669183, mean estimator."""
    v = [float(a) for a in x]
    if kind == 'S1':
        return float(4 / (4 + n**0.75) * (v[0] + v[2]) / 2 + n**0.75 / (4 + n**0.75) * v[1])
    if kind == 'S2':
        return float((0.7 + 0.39 / n) * (v[0] + v[2]) / 2 + (0.3 - 0.39 / n) * v[1])
    if kind == 'S3':
        return float(
            2.2 / (2.2 + n**0.75) * (v[0] + v[4]) / 2 + (0.7 - 0.72 / n**0.55) * (v[1] + v[3]) / 2 + (0.3 + 0.72 / n**0.55 - 2.2 / (2.2 + n**0.75)) * v[2]
        )
    raise metafopy.errors.InputError(f"Unknown 'type' specified: {kind}.")


def wan_std(x: collections.abc.Sequence[float] | FloatArray, n: float, kind: str) -> float:
    """metaBLUE's ``Wan.std``: Wan et al. (2014), doi:10.1186/1471-2288-14-135, SD estimator."""
    v = [float(a) for a in x]
    if kind == 'S1':
        return (v[2] - v[0]) / (2 * qnorm((n - 0.375) / (n + 0.25)))
    if kind == 'S2':
        return (v[2] - v[0]) / (2 * qnorm((0.75 * n - 0.125) / (n + 0.25)))
    if kind == 'S3':
        tao = 2 * qnorm((n - 0.375) / (n + 0.25))
        eta = 2 * qnorm((0.75 * n - 0.125) / (n + 0.25))
        return 0.5 * (v[4] - v[0]) / tao + 0.5 * (v[3] - v[1]) / eta
    raise metafopy.errors.InputError(f"Unknown 'type' specified: {kind}.")


def blue_s(x: collections.abc.Sequence[float] | FloatArray, n: float, kind: str) -> BlueResult:
    """metaBLUE's ``BLUE_s``: the BLUE of the location and scale of a normal distribution from order statistics.

    The expected values and covariances of the standard normal order statistics use the series approximations of
    Yang et al. (2022) (the ``Qr_d*`` derivatives of the quantile function). ``kind`` is ``'S1'``, ``'S2'`` or ``'S3'``.

    Raises:
        metafopy.errors.EstimationError: the covariance matrix of the order statistics is singular.
    """
    index = sum_stat_index(n, kind)
    m = len(index)

    def inverse_erfc(u: float) -> float:
        return -1 / math.sqrt(2) * qnorm(u / 2)

    def d1(r: int) -> float:
        return math.exp(inverse_erfc(2 * r / (n + 1)) ** 2) * math.sqrt(2 * math.pi)

    def d2(r: int) -> float:
        e = inverse_erfc(2 * r / (n + 1))
        return -2 * math.sqrt(2) * math.exp(2 * e**2) * math.pi * e

    def d3(r: int) -> float:
        e = inverse_erfc(2 * r / (1 + n))
        return float(2 * math.sqrt(2) * math.exp(3 * e**2) * math.pi ** (3 / 2) * (1 + 4 * e**2))

    def d4(r: int) -> float:
        e = inverse_erfc(2 * r / (1 + n))
        return -4 * math.sqrt(2) * math.exp(4 * e**2) * math.pi**2 * e * (7 + 12 * e**2)

    def expected(r: int) -> float:
        p = r / (n + 1)
        return qnorm(p) + d2(r) * p * (1 - p) / (2 * (n + 2)) + (p * (1 - p) / (n + 2) ** 2) * ((1 - p - p) * d3(r) / 3 + p * (1 - p) * d4(r) / 8)

    def variance(r: int) -> float:
        p = r / (n + 1)
        return p * (1 - p) * d1(r) ** 2 / (n + 2) + (p * (1 - p) / (n + 2) ** 2) * (
            2 * (1 - p - p) * d1(r) * d2(r) + p * (1 - p) * (d1(r) * d3(r) + d2(r) ** 2 / 2)
        )

    def covariance(r: int, s: int) -> float:
        p = r / (n + 1)
        q = s / (n + 1)
        return p * (1 - q) * d1(r) * d1(s) / (n + 2) + (p * (1 - q) / (n + 2) ** 2) * (
            (1 - p - p) * d2(r) * d1(s)
            + (1 - q - q) * d1(r) * d2(s)
            + p * (1 - p) * d3(r) * d1(s) / 2
            + q * (1 - q) * d3(s) * d1(r) / 2
            + p * (1 - q) * d2(r) * d2(s) / 2
        )

    alpha = numpy.array([expected(r) for r in index])
    scovar = numpy.empty((m, m))
    for ii, rr in enumerate(index):
        for jj, ss in enumerate(index):
            if rr < ss:
                scovar[ii, jj] = covariance(rr, ss)
            elif rr > ss:
                scovar[ii, jj] = scovar[jj, ii]
            else:
                scovar[ii, jj] = variance(rr)

    ones = numpy.ones(m)
    A = numpy.column_stack([ones, alpha])
    try:
        omega = numpy.linalg.inv(scovar)
    except numpy.linalg.LinAlgError as e:
        raise metafopy.errors.EstimationError('The covariance matrix of the order statistics is singular.') from e
    delta = float(numpy.linalg.det(A.T @ omega @ A))
    gamma = omega @ (numpy.outer(ones, alpha) - numpy.outer(alpha, ones)) @ omega / delta
    c_mu = -alpha @ gamma
    c_sigma = ones @ gamma
    values = numpy.asarray(x, dtype=numpy.float64)
    muhat = float(c_mu @ values)
    sigmahat = float(c_sigma @ values)
    var_mu = float(alpha @ omega @ alpha * sigmahat**2 / delta)
    var_sigma = float(ones @ omega @ ones * sigmahat**2 / delta)
    return BlueResult(alpha, scovar, muhat, sigmahat, var_mu, var_sigma)


def check_errors(quants: FloatArray, n: float) -> None:
    """estmeansd's ``check_errors``.

    Raises:
        metafopy.errors.InputError: ``n`` is outside 3 to 1,000,000 or the quantiles are not increasing.
    """
    if math.isnan(n) or n < 3 or n > 1e6:
        raise metafopy.errors.InputError('Value of n must be between 3 and 1,000,000')
    if any(b < a for a, b in itertools.pairwise(quants)):
        raise metafopy.errors.InputError('The quantiles are not in increasing order.')
    if len(set(quants.tolist())) < len(quants):
        warnings.warn('Some of the quantiles are equal to each other. This can result in unexpected behaviour of the method.', stacklevel=3)


def rnorm(rng: metafopy.core.rng.RRandom, n: int, mu: float, sigma: float) -> FloatArray:
    """R's ``rnorm(n, mu, sigma)``: no draws are made when ``mu`` or ``sigma`` is not usable."""
    if math.isnan(mu) or math.isnan(sigma) or not math.isfinite(sigma) or sigma < 0:
        return numpy.full(n, numpy.nan)
    if sigma == 0 or not math.isfinite(mu):
        return numpy.full(n, mu)
    return mu + sigma * rng.rnorm(n)


def r_mean(x: FloatArray) -> float:
    return float(numpy.mean(x, dtype=numpy.longdouble)) if x.size else math.nan


def r_sd(x: FloatArray) -> float:
    return float(numpy.std(x, ddof=1, dtype=numpy.longdouble)) if x.size > 1 else math.nan


def sign_pow(a: FloatArray, b: float) -> FloatArray:
    with numpy.errstate(all='ignore'):
        return numpy.sign(a) * numpy.abs(a) ** b


def smooth_bc_transform(vals: FloatArray, lam: float) -> FloatArray:
    if abs(lam) < 1e-8:
        with numpy.errstate(all='ignore'):
            return numpy.log(vals)
    return (sign_pow(vals, lam) - 1) / lam


def inv_smooth_bc_transform(transformed: FloatArray, lam: float) -> FloatArray:
    if abs(lam) < 1e-8:
        with numpy.errstate(all='ignore'):
            return numpy.exp(transformed)
    return sign_pow(transformed * lam + 1, 1 / lam)


def round_lambda(lam: float) -> float:
    return 0.0 if abs(lam) < 1e-8 else max(0.0, lam)


def lbfgsb(
    fn: collections.abc.Callable[[FloatArray], float],
    start: collections.abc.Sequence[float],
    lower: collections.abc.Sequence[float],
    upper: collections.abc.Sequence[float],
) -> tuple[FloatArray, float] | None:
    """R's ``optim(method='L-BFGS-B')`` with default control: SciPy's L-BFGS-B with R's bound-aware finite-difference gradient.

    Returns ``None`` where R signals an error (a non-finite objective or gradient).
    """
    lo = numpy.array(lower, dtype=numpy.float64)
    hi = numpy.array(upper, dtype=numpy.float64)
    ndeps = 1e-3

    def objective(p: FloatArray) -> float:
        with numpy.errstate(all='ignore'):
            val = float(fn(p))
        if not math.isfinite(val):
            raise metafopy.errors.EstimationError("L-BFGS-B needs finite values of 'fn'")
        return val

    def gradient(p: FloatArray) -> FloatArray:
        df = numpy.empty_like(p)
        for i in range(p.shape[0]):
            eps_up = eps_down = ndeps
            x = p.copy()
            x[i] = p[i] + ndeps
            if x[i] > hi[i]:
                x[i] = hi[i]
                eps_up = x[i] - p[i]
            val1 = objective(x)
            x[i] = p[i] - ndeps
            if x[i] < lo[i]:
                x[i] = lo[i]
                eps_down = p[i] - x[i]
            val2 = objective(x)
            df[i] = (val1 - val2) / (eps_up + eps_down)
        return df

    x0 = numpy.clip(numpy.array(start, dtype=numpy.float64), lo, hi)
    try:
        res = scipy.optimize.minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=list(zip(lo, hi, strict=True)),
            options={'maxcor': 5, 'ftol': 1e7 * numpy.finfo(float).eps, 'gtol': 0.0, 'maxiter': 100, 'maxfun': 15000},
        )
    except metafopy.errors.EstimationError, ValueError, ArithmeticError:
        return None
    return numpy.asarray(res.x, dtype=numpy.float64), float(res.fun)


def weibull_shape_start(mean_hat: float, sd_hat: float) -> float:
    """The starting value of the Weibull shape: ``nlminb`` matches the coefficient of variation, from ``(mean/sd)^1.086``."""
    start = (mean_hat / sd_hat) ** 1.086

    def objective(k: float) -> float:
        temp1 = float(scipy.special.gamma((k + 2) / k))
        temp2 = float(scipy.special.gamma((k + 1) / k))
        return float((sd_hat / mean_hat - math.sqrt(temp1 / temp2**2 - 1)) ** 2)

    fit = lbfgsb(lambda p: objective(float(p[0])), [start], [1e-3], [math.inf])
    if fit is None or not math.isfinite(fit[1]):
        return float(start)
    return float(fit[0][0])


def qe_fit(quants: FloatArray, n: float, scenario: str) -> dict[str, tuple[FloatArray, float] | None]:
    """estmeansd's ``qe.fit``: the L-BFGS-B fits of the candidate distributions (``None`` where a fit fails)."""
    probs = {
        'S1': numpy.array([1 / n, 0.5, 1 - 1 / n]),
        'S2': numpy.array([0.25, 0.5, 0.75]),
        'S3': numpy.array([1 / n, 0.25, 0.5, 0.75, 1 - 1 / n]),
    }[scenario]
    quants = quants.copy()
    if numpy.all(quants == 0):
        quants[:] = 1e-2

    lo_i, hi_i = (0, 2) if scenario != 'S3' else (1, 3)
    med = quants[lo_i + 1] if scenario != 'S3' else quants[2]
    positive = bool(numpy.min(quants) > 0)
    norm_mu_bounds = (quants[lo_i], quants[hi_i])
    lnorm_mu_bounds = (math.log(quants[lo_i]), math.log(quants[hi_i])) if positive else None
    del med

    mean_hat = luo_mean(quants, n, scenario)
    sd_hat = wan_std(quants, n, scenario)
    with numpy.errstate(all='ignore'):
        lnorm_start = (math.log(mean_hat / math.sqrt(1 + (sd_hat / mean_hat) ** 2)), math.sqrt(math.log(1 + (sd_hat / mean_hat) ** 2)))
        gamma_start = (mean_hat**2 / sd_hat**2, mean_hat / sd_hat**2)
        beta1 = mean_hat * (mean_hat * (1 - mean_hat) / sd_hat**2 - 1)
        beta_start = (beta1, beta1 * (1 - mean_hat) / mean_hat)
        weibull_shape = weibull_shape_start(mean_hat, sd_hat)
        weibull_start = (weibull_shape, mean_hat / float(scipy.special.gamma((weibull_shape + 1) / weibull_shape)))
    norm_start = (mean_hat, sd_hat)

    def sse(values: FloatArray) -> float:
        return float(numpy.sum((values - quants) ** 2))

    def lnorm_bounds() -> tuple[list[float], list[float]]:
        # R recycles bounds that are too short: with no log-scale bounds for the mean, both parameters get the sigma bounds
        if lnorm_mu_bounds is None:
            return [1e-3, 1e-3], [10.0, 10.0]
        return [lnorm_mu_bounds[0], 1e-3], [lnorm_mu_bounds[1], 10.0]

    fits: dict[str, tuple[FloatArray, float] | None] = {name: None for name in DISTRIBUTIONS}
    fits['normal'] = lbfgsb(lambda t: sse(scipy.stats.norm.ppf(probs, loc=t[0], scale=t[1])), norm_start, [norm_mu_bounds[0], 1e-3], [norm_mu_bounds[1], 50.0])
    if numpy.min(quants) >= 0:
        lo, hi = lnorm_bounds()
        fits['log-normal'] = lbfgsb(lambda t: sse(scipy.stats.lognorm.ppf(probs, s=t[1], scale=math.exp(t[0]))), lnorm_start, lo, hi)
        fits['gamma'] = lbfgsb(lambda t: sse(scipy.stats.gamma.ppf(probs, a=t[0], scale=1 / t[1])), gamma_start, [1e-3, 1e-3], [1e2, 1e2])
        fits['weibull'] = lbfgsb(lambda t: sse(scipy.stats.weibull_min.ppf(probs, c=t[0], scale=t[1])), weibull_start, [1e-3, 1e-3], [1e2, 1e2])
        fits['beta'] = lbfgsb(lambda t: sse(scipy.stats.beta.ppf(probs, t[0], t[1])), beta_start, [1e-3, 1e-3], [40.0, 40.0])
    return fits


def mean_sd(par: FloatArray, family: str) -> tuple[float, float]:
    """estmeansd's ``get.mean.sd``: the mean and SD of a fitted distribution."""
    a, b = float(par[0]), float(par[1])
    with numpy.errstate(all='ignore'):
        if family == 'normal':
            return a, b
        if family == 'log-normal':
            return math.exp(a + b**2 / 2), math.sqrt((math.exp(b**2) - 1) * math.exp(2 * a + b**2))
        if family == 'gamma':
            return a / b, math.sqrt(a / b**2)
        if family == 'weibull':
            g1 = float(scipy.special.gamma(1 + 1 / a))
            return b * g1, math.sqrt(b**2 * (float(scipy.special.gamma(1 + 2 / a)) - g1**2))
        return a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


def qe_mean_sd(quants: FloatArray, n: float, scenario: str) -> Estimate:
    """estmeansd's ``qe.mean.sd``: fits the candidate distributions and takes the mean and SD of the best fit.

    Raises:
        metafopy.errors.InputError: ``n`` or the quantiles are invalid.
        metafopy.errors.EstimationError: no distribution could be fitted.
    """
    check_errors(quants, n)
    fits = qe_fit(quants, n, scenario)
    best: str | None = None
    for name in DISTRIBUTIONS:
        fit = fits[name]
        if fit is not None and (best is None or fit[1] < fits[best][1]):  # type: ignore[index]
            best = name
    if best is None or fits[best] is None:
        raise metafopy.errors.EstimationError('No distribution could be fitted.')
    par = fits[best][0]  # type: ignore[index]
    est_mean, est_sd = mean_sd(par, best)
    return Estimate(est_mean, est_sd, best)


def bc_mean_sd(quants: FloatArray, n: float, scenario: str, rng: metafopy.core.rng.RRandom) -> Estimate:
    """estmeansd's ``bc.mean.sd`` (default ``preserve.tail=FALSE``, ``avoid.mc=FALSE``): the Box-Cox method.

    Raises:
        metafopy.errors.InputError: ``n`` or the quantiles are invalid.
    """
    check_errors(quants, n)
    with numpy.errstate(all='ignore'):
        logs = numpy.log(quants)
        mid = len(quants) // 2
        ratio_diff = {
            'S1': (logs[2] - logs[1]) / (logs[1] - logs[0]) - 1,
            'S2': (logs[2] - logs[1]) / (logs[1] - logs[0]) - 1,
        }.get(scenario)
        if scenario == 'S3':
            temp1 = ((logs[3] - logs[2]) / (logs[2] - logs[1]) - 1) ** 2
            temp2 = ((logs[4] - logs[2]) / (logs[2] - logs[0]) - 1) ** 2
            diff = temp1 + temp2
        else:
            assert ratio_diff is not None
            diff = ratio_diff**2
    del mid

    def lambda_error(lam: float) -> float:
        t = smooth_bc_transform(quants, lam)
        with numpy.errstate(all='ignore'):
            if scenario == 'S3':
                return float(((t[3] - t[2]) / (t[2] - t[1]) - 1) ** 2 + ((t[4] - t[2]) / (t[2] - t[0]) - 1) ** 2)
            return float(((t[2] - t[1]) / (t[1] - t[0]) - 1) ** 2)

    lam = 0.0 if abs(diff) < 1e-8 else metafopy.core.optim.fmin(lambda_error, -2.0, 3.0)
    lam = round_lambda(lam)
    transformed = smooth_bc_transform(quants, lam)
    location = luo_mean(transformed, n, scenario)
    scale = wan_std(transformed, n, scenario)

    trans_norm = rnorm(rng, int(n + 1e3), location, scale)
    if lam >= 1e-8:
        trans_norm = trans_norm[(trans_norm >= -1 / lam) & (trans_norm <= 2 * location + 1 / lam)]
    bc_rvs = inv_smooth_bc_transform(trans_norm, lam)
    return Estimate(r_mean(bc_rvs), r_sd(bc_rvs))


def mln_mean_sd(quants: FloatArray, n: float, scenario: str, rng: metafopy.core.rng.RRandom) -> Estimate:
    """estmeansd's ``mln.mean.sd``: the Box-Cox transformation and normal parameters by maximum likelihood.

    Raises:
        metafopy.errors.InputError: ``n`` or the quantiles are invalid.
    """
    check_errors(quants, n)
    h = math.floor(0.25 * n + 1)
    j = math.floor(0.5 * n + 1)
    k = math.floor(0.75 * n + 1)

    def boxcox(lam: float) -> FloatArray:
        with numpy.errstate(all='ignore'):
            return numpy.log(quants) if lam == 0 else (quants**lam - 1) / lam

    def loglik(t: FloatArray, mu: float, sigma: float) -> float:
        norm = scipy.stats.norm
        with numpy.errstate(all='ignore'):
            dens = float(numpy.sum(norm.logpdf(t, mu, sigma)))
            cdf = norm.cdf(t, mu, sigma)
            if scenario == 'S1':
                return float(dens + (j - 2) * math.log(cdf[1] - cdf[0]) + (n - j - 1) * math.log(cdf[2] - cdf[1]))
            if scenario == 'S2':
                return float(
                    dens
                    + (h - 1) * norm.logcdf(t[0], mu, sigma)
                    + (j - h - 1) * math.log(cdf[1] - cdf[0])
                    + (k - j - 1) * math.log(cdf[2] - cdf[1])
                    + (n - k) * norm.logsf(t[2], mu, sigma)
                )
            return float(
                dens
                + (h - 2) * math.log(cdf[1] - cdf[0])
                + (j - h - 1) * math.log(cdf[2] - cdf[1])
                + (k - j - 1) * math.log(cdf[3] - cdf[2])
                + (n - k - 1) * math.log(cdf[4] - cdf[3])
            )

    def profile(lam: float) -> float:
        t = boxcox(lam)
        return loglik(t, luo_mean(t, n, scenario), wan_std(t, n, scenario))

    lam_hat = round_lambda(metafopy.core.optim.fmin(lambda lam: -profile(lam), 0.0, 10.0, tol=0.001))
    t = boxcox(lam_hat)
    start = [luo_mean(t, n, scenario), wan_std(t, n, scenario)]
    mle = metafopy.core.optim.nelder_mead(lambda theta: -loglik(t, float(theta[0]), float(theta[1])), start)
    mu, sigma = float(mle.par[0]), float(mle.par[1])

    data = rnorm(rng, 10000, mu, sigma)
    if lam_hat != 0:
        data = data[(data > -1 / lam_hat) & (data < 2 * mu + 1 / lam_hat)]
    with numpy.errstate(all='ignore'):
        rvs = numpy.exp(data) if lam_hat == 0 else (lam_hat * data + 1) ** (1 / lam_hat)
    return Estimate(r_mean(rvs), r_sd(rvs))


def estimate(method: str, quants: FloatArray, n: float, scenario: str) -> Estimate:
    """The mean and SD of one study with ``method`` (``'qe'``, ``'bc'``, ``'mln'`` or ``'blue'``).

    ``'bc'`` and ``'mln'`` draw from the module-level generator.
    """
    if method == 'qe':
        return qe_mean_sd(quants, n, scenario)
    if method == 'bc':
        return bc_mean_sd(quants, n, scenario, metafopy.core.rng.get_rng())
    if method == 'mln':
        return mln_mean_sd(quants, n, scenario, metafopy.core.rng.get_rng())
    res = blue_s(quants, n, scenario)
    return Estimate(res.muhat, res.sigmahat)
