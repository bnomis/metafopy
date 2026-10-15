"""Estimate means and standard deviations from five-number summaries (metafor's ``conv.fivenum``).

R source: https://github.com/wviechtb/metafor/blob/master/R/conv.fivenum.r. Documentation:
https://wviechtb.github.io/metafor/reference/conv.fivenum.html. From the minimum, median and maximum
(case 1), the quartiles and median (case 2) or all five values (case 3) with the sample size, after a
test for skewness (Shi et al. 2023, doi:10.1177/09622802221139233):

- means: Luo et al. (2016), doi:10.1177/0962280216669183 (``'luo/wan/shi'``, ``'luo2016'``), Hozo et
  al. (2005), doi:10.1186/1471-2288-5-13, Wan et al. (2014), doi:10.1186/1471-2288-14-135, Bland
  (2015) and Walter and Yao (2007), doi:10.1016/j.jclinepi.2006.09.004;
- SDs: Wan et al. (2014), Shi et al. (2020), doi:10.1002/jrsm.1429 (``'shi2020'``), Hozo et al.
  (2005), Bland (2015) and Walter and Yao (2007);
- ``dist='lnorm'`` assumes log-normal data and back-transforms (Shi et al. 2020,
  doi:10.1177/0962280220913853);
- ``'qe'`` (McGrath et al. 2020, doi:10.1177/0962280219889080), ``'bc'`` and ``'mln'`` (McGrath et al. 2023,
  doi:10.1002/jrsm.1664) from the R package estmeansd, and ``'blue'`` (Yang et al. 2022,
  doi:10.1080/02664763.2021.1967890) from metaBLUE, in :mod:`metafopy.effects.fivenum_estimators`. ``'bc'`` and
  ``'mln'`` simulate with R's random number generator, so ``seed`` gives R's results exactly.

The skewness test results are kept in ``DataFrame.attrs`` (``tval``, ``crit``, ``sig``,
``dist``); ``attrs['est']`` maps each existing output column to which of its values were estimated.
"""

import builtins
import collections.abc
import itertools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.integrate
import metafopy.core.linalg
import metafopy.core.rng
import metafopy.effects.conv
import metafopy.effects.fivenum_estimators
import metafopy.effects.measures
import metafopy.errors

__all__ = ['conv_fivenum']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Arg = numpy.typing.ArrayLike | str | None

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
METHODS1 = ('default', 'luo/wan/shi', 'qe', 'bc', 'mln', 'blue', 'hozo2005', 'wan2014', 'bland2015', 'luo2016', 'walter2007')
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
METHODS2 = ('default', 'luo/wan/shi', 'qe', 'bc', 'mln', 'blue', 'hozo2005', 'wan2014', 'bland2015', 'shi2020', 'walter2007')


EXTERNAL = ('bc', 'blue', 'mln', 'qe')
SCENARIOS = {(True, False): 'S1', (False, True): 'S2', (True, True): 'S3'}


def qnorm(p: float) -> float:
    return float(scipy.stats.norm.ppf(p))


def match_method(method: str | collections.abc.Sequence[str]) -> tuple[str, str]:
    """The methods for the means and the SDs (R: the first option that starts with the given string).

    Raises:
        metafopy.errors.InputError: a method is unknown or ``'qe'``, ``'bc'``, ``'mln'`` or ``'blue'`` is not used for both.
    """
    given = [method] if isinstance(method, str) else list(method)
    given = [m.lower() for m in given]
    if len(given) == 1:
        given = given * 2
    out = []
    for m, options in zip(given[:2], (METHODS1, METHODS2), strict=True):
        hits = [opt for opt in options if opt.startswith(m)]
        if not hits:
            raise metafopy.errors.InputError("Unknown 'method' specified.")
        out.append('luo/wan/shi' if hits[0] == 'default' else hits[0])
    if out[0] in EXTERNAL and out[0] != out[1]:
        raise metafopy.errors.InputError("Must use the same 'method' for estimating means and SDs.")
    return out[0], out[1]


def match_dist(dist: typing.Any, k: int, data: pandas.DataFrame | None) -> list[str]:
    """Raises:
    metafopy.errors.InputError: a value is not (a prefix of) ``'norm'`` or ``'lnorm'``.
    """
    if isinstance(dist, str) and data is not None and dist in data.columns:
        values = [str(v) for v in data[dist]]
    elif isinstance(dist, str):
        values = [dist]
    else:
        values = [str(v) for v in dist]
    if len(values) == 1:
        values = values * k
    if len(values) != k:
        raise metafopy.errors.InputError(f"Length of the 'dist' argument ({len(values)}) does not match the length of the data ({k}).")
    out = []
    for v in values:
        hits = [opt for opt in ('norm', 'lnorm') if v and opt.startswith(v)]
        if len(hits) != 1:
            raise metafopy.errors.InputError("Unknown 'dist' value specified (should either be 'norm' or 'lnorm').")
        out.append(hits[0])
    return out


def walter_denominator(n: float) -> float:
    """``integrate(function(x) 1 - (1 - pnorm(x))^n - pnorm(x)^n, -Inf, Inf)``: the expected range of ``n`` standard normal values."""

    def f(x: float) -> float:
        alpha = float(scipy.stats.norm.cdf(x))
        return float(1 - (1 - alpha) ** n - alpha**n)

    return float(metafopy.core.integrate.integrate(f, -math.inf, math.inf))


def conv_fivenum(
    min: Arg = None,
    q1: Arg = None,
    median: Arg = None,
    q3: Arg = None,
    max: Arg = None,
    n: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    include: numpy.typing.ArrayLike | str | None = None,
    method: str | collections.abc.Sequence[str] = 'default',
    dist: str | collections.abc.Sequence[str] = 'norm',
    transf: bool = True,
    test: bool = True,
    var_names: collections.abc.Sequence[str] = ('mean', 'sd'),
    append: bool = True,
    replace: metafopy.effects.conv.Replace = 'ifna',
    seed: int | None = None,
    verbose: bool = False,
) -> pandas.DataFrame:
    """Estimate means and SDs from the minimum, first quartile, median, third quartile and maximum.

    ``method`` is one method for both or a pair (means, SDs); ``dist`` (``'norm'`` or ``'lnorm'``, one
    value or one per study) the assumed distribution, where ``transf`` back-transforms the estimates of
    log-normal data. With ``test=True`` no estimates are computed when the test for skewness is
    significant. Studies with ``n < 5`` are skipped.

    The ``'qe'``, ``'bc'`` and ``'mln'`` methods (estmeansd) never skip a study after the test for skewness, except that
    ``'qe'`` tests studies with negative values when ``test=True``; they need the same method for the means and SDs. ``'bc'``
    and ``'mln'`` simulate with R's random number generator: ``seed`` is R's ``set.seed()`` argument and reproduces R's
    results exactly, otherwise the module-level generator (see :func:`metafopy.set_seed`) is used. ``verbose=True`` lets
    the warnings of these methods through. The ``dist`` entry of ``attrs`` holds the distribution selected by ``'qe'``
    (``'normal'``, ``'lnorm'``, ``'gamma'``, ``'weibull'`` or ``'beta'``, ``None`` when the fit failed).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: ``'blue'`` met a singular covariance matrix.
    """
    if min is None and q1 is None and median is None and q3 is None and max is None:
        raise metafopy.errors.InputError("Must specify at least some of these arguments: 'min', 'q1', 'median', 'q3', 'max'.")
    repl = metafopy.effects.conv.match_replace(replace)
    if len(var_names) != 2:
        raise metafopy.errors.InputError("Argument 'var_names' must be of length 2.")
    x = metafopy.effects.conv.as_frame(data)
    inp = metafopy.effects.measures.Inputs(x, dict(min=min, q1=q1, median=median, q3=q3, max=max, n=n))
    names = ('min', 'q1', 'median', 'q3', 'max', 'n')
    raw = {name: inp.get(name) for name in names}
    if not metafopy.effects.measures.equal_length(*raw.values()):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    k = builtins.max(v.shape[0] for v in raw.values() if v is not None)
    v = {name: metafopy.effects.conv.fill(val, k) for name, val in raw.items()}
    dists_in = match_dist(dist, k, x)

    inc = metafopy.effects.conv.include_mask(include, k, x).copy()
    with numpy.errstate(invalid='ignore'):
        inc[v['n'] < 5] = False
    for name in names:
        v[name][~inc] = numpy.nan
    lo, a1, med, a3, hi, nn = (v[name] for name in names)

    case1 = ~numpy.isnan(lo) & numpy.isnan(a1) & numpy.isnan(a3) & ~numpy.isnan(hi)
    case2 = numpy.isnan(lo) & ~numpy.isnan(a1) & ~numpy.isnan(a3) & numpy.isnan(hi)
    case3 = ~numpy.isnan(lo) & ~numpy.isnan(a1) & ~numpy.isnan(a3) & ~numpy.isnan(hi)

    m1, m2 = match_method(method)
    if 'lnorm' in dists_in and not (m1 == 'luo/wan/shi' and m2 == 'luo/wan/shi'):
        m1 = m2 = 'luo/wan/shi'
        warnings.warn("Switching to method='luo/wan/shi' (since dist='lnorm' for one or more studies).", stacklevel=2)

    if seed is not None:
        metafopy.core.rng.set_seed(seed)
    test_now = False if m1 in ('bc', 'mln', 'qe') else test

    means = numpy.full(k, numpy.nan)
    sds = numpy.full(k, numpy.nan)
    tval = numpy.full(k, numpy.nan)
    crit = numpy.full(k, numpy.nan)
    sig: list[bool | None] = [None] * k
    dists: list[str | None] = ['norm'] * k

    def apply_external(i: int, quants: list[float], scenario: str, ni: float) -> None:
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter('ignore')
            try:
                est = metafopy.effects.fivenum_estimators.estimate(m1, numpy.array(quants), ni, scenario)
            except metafopy.errors.InputError, metafopy.errors.EstimationError, ArithmeticError, ValueError:
                if m1 == 'blue':
                    raise
                est = None
        if m1 == 'qe':
            dists[i] = None if est is None else est.dist
        if est is not None:
            means[i], sds[i] = est.mean, est.sd

    for i in range(k):
        vals = [lo[i], a1[i], med[i], a3[i], hi[i]]
        present = [val for val in vals if not math.isnan(val)]
        if m1 in ('bc', 'mln') and any(val <= 0 for val in present):
            raise metafopy.errors.InputError(f'Cannot use method with non-positive values (found in row {i}).')
        if m1 == 'qe' and test and any(val < 0 for val in present):
            test_now = True
        if any(b < a for a, b in itertools.pairwise(present)):
            raise metafopy.errors.InputError(f"Found 'min <= q1 <= median <= q3 <= max' not true in row {i}.")
        mn, q1i, md, q3i, mx, ni = lo[i], a1[i], med[i], a3[i], hi[i], nn[i]
        if dists_in[i] == 'lnorm':
            if any(val <= 0 for val in present):
                raise metafopy.errors.InputError(f'Cannot assume a log-normal distribution with non-positive values (found in row {i}).')
            mn, q1i, md, q3i, mx = (math.log(val) if not math.isnan(val) else val for val in (mn, q1i, md, q3i, mx))
            dists[i] = 'lnorm'
        lnorm = dists_in[i] == 'lnorm'

        with numpy.errstate(all='ignore'):
            if case1[i]:
                tval[i] = abs((mn + mx - 2 * md) / (mx - mn))
                crit[i] = 1 / math.log(ni + 9) + 2.5 / (ni + 1)
                sig[i] = bool(tval[i] >= crit[i])
                if test_now and sig[i]:
                    continue
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                if m1 in ('luo/wan/shi', 'luo2016'):
                    weight = 4 / (4 + ni**0.75)
                    means[i] = weight * (mn + mx) / 2 + (1 - weight) * md
                if m1 == 'hozo2005':
                    means[i] = math.nan if math.isnan(ni) else (mn + 2 * md + mx) / 4 if ni <= 25 else md
                if m1 == 'wan2014':
                    means[i] = (mn + 2 * md + mx) / 4
                if m1 == 'walter2007':
                    means[i] = md
                if m1 in EXTERNAL:
                    apply_external(i, [mn, md, mx], 'S1', ni)
                xi = 2 * qnorm((ni - 0.375) / (ni + 0.25))
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                if m2 in ('luo/wan/shi', 'wan2014'):
                    z1 = 1.0 if not lnorm else 1.01 + 0.25 / math.log(ni) ** 2
                    sds[i] = (mx - mn) / xi * (1 / math.sqrt(z1))
                if m2 == 'hozo2005':
                    if math.isnan(ni):
                        sds[i] = math.nan
                    elif ni <= 15:
                        sds[i] = 1 / math.sqrt(12) * math.sqrt((mn - 2 * md + mx) ** 2 / 4 + (mx - mn) ** 2)
                    elif ni <= 70:
                        sds[i] = (mx - mn) / 4
                    else:
                        sds[i] = (mx - mn) / 6
                if m2 == 'walter2007':
                    try:
                        f = walter_denominator(ni)
                    except metafopy.errors.EstimationError, ArithmeticError, ValueError:
                        continue
                    sds[i] = (mx - mn) / f
                if lnorm and transf:
                    s41 = ((mx - mn) / xi) ** 4 / (1 + 2.23 / math.log(ni) ** 2)
                    phi1 = 1 + 0.565 * sds[i] ** 2 / ni + 0.37 * s41 / ni
                    btmean = math.exp(means[i] + sds[i] ** 2 / 2) * (1 / phi1)
                    phi11 = 1 + 2.26 * sds[i] ** 2 / ni + 5.92 * s41 / ni
                    phi12 = 1 + 2.26 * sds[i] ** 2 / ni + 1.48 * s41 / ni
                    btsd = metafopy.core.linalg.rsqrt(
                        math.exp(2 * means[i] + 2 * sds[i] ** 2) * (1 / phi11) - math.exp(2 * means[i] + sds[i] ** 2) * (1 / phi12)
                    )
                    means[i], sds[i] = btmean, btsd

            if case2[i]:
                tval[i] = abs((q1i + q3i - 2 * md) / (q3i - q1i))
                crit[i] = 2.65 / math.sqrt(ni) - 6 / ni**2
                sig[i] = bool(tval[i] >= crit[i])
                if test_now and sig[i]:
                    continue
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                if m1 in ('luo/wan/shi', 'luo2016'):
                    weight = 0.7 + 0.39 / ni
                    means[i] = weight * (q1i + q3i) / 2 + (1 - weight) * md
                if m1 == 'wan2014':
                    means[i] = (q1i + md + q3i) / 3
                if m1 in EXTERNAL:
                    apply_external(i, [q1i, md, q3i], 'S2', ni)
                eta = 2 * qnorm((0.75 * ni - 0.125) / (ni + 0.25))
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                if m2 in ('luo/wan/shi', 'wan2014'):
                    z2 = 1.0 if not lnorm else 1 + 1.58 / ni
                    sds[i] = (q3i - q1i) / eta * (1 / math.sqrt(z2))
                if lnorm and transf:
                    s42 = ((q3i - q1i) / eta) ** 4 / (1 + 19.2 / ni**1.2)
                    phi2 = 1 + 0.57 * sds[i] ** 2 / ni + 0.75 * s42 / ni
                    btmean = math.exp(means[i] + sds[i] ** 2 / 2) * (1 / phi2)
                    phi21 = 1 + 2.28 * sds[i] ** 2 / ni + 12 * s42 / ni
                    phi22 = 1 + 2.28 * sds[i] ** 2 / ni + 3 * s42 / ni
                    btsd = metafopy.core.linalg.rsqrt(
                        math.exp(2 * means[i] + 2 * sds[i] ** 2) * (1 / phi21) - math.exp(2 * means[i] + sds[i] ** 2) * (1 / phi22)
                    )
                    means[i], sds[i] = btmean, btsd

            if case3[i]:
                tval[i] = float(
                    numpy.max([2.65 * math.log(0.6 * ni) / math.sqrt(ni) * abs((mn + mx - 2 * md) / (mx - mn)), abs((q1i + q3i - 2 * md) / (q3i - q1i))])
                )
                crit[i] = 3 / math.sqrt(ni) - 40 / ni**3
                sig[i] = bool(tval[i] >= crit[i])
                if test_now and sig[i]:
                    continue
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                if m1 in ('luo/wan/shi', 'luo2016'):
                    weight1 = 2.2 / (2.2 + ni**0.75)
                    weight2 = 0.7 - 0.72 / ni**0.55
                    means[i] = weight1 * (mn + mx) / 2 + weight2 * (q1i + q3i) / 2 + (1 - weight1 - weight2) * md
                if m1 in ('wan2014', 'bland2015'):
                    means[i] = (mn + 2 * q1i + 2 * md + 2 * q3i + mx) / 8
                if m1 in EXTERNAL:
                    apply_external(i, [mn, q1i, md, q3i, mx], 'S3', ni)
                xi = 2 * qnorm((ni - 0.375) / (ni + 0.25))
                eta = 2 * qnorm((0.75 * ni - 0.125) / (ni + 0.25))
                weight = 1 / (1 + 0.07 * ni**0.6)
                if m2 in ('luo/wan/shi', 'shi2020'):
                    z3 = 1.0 if not lnorm else 1 + 0.28 / math.log(ni) ** 2
                    sds[i] = (weight * (mx - mn) / xi + (1 - weight) * (q3i - q1i) / eta) * (1 / math.sqrt(z3))
                if m2 == 'wan2014':
                    sds[i] = 1 / 2 * ((mx - mn) / xi + (q3i - q1i) / eta)
                if m2 == 'bland2015':
                    sds[i] = metafopy.core.linalg.rsqrt(
                        (mn**2 + 2 * q1i**2 + 2 * md**2 + 2 * q3i**2 + mx**2) / 16
                        + (mn * q1i + q1i * md + md * q3i + q3i * mx) / 8
                        - (mn + 2 * q1i + 2 * md + 2 * q3i + mx) ** 2 / 64
                    )
                if lnorm and transf:
                    s43 = (weight * (mx - mn) / xi + (1 - weight) * (q3i - q1i) / eta) ** 4 / (1 + 3.93 / ni)
                    phi3 = 1 + 0.405 * sds[i] ** 2 / ni + 0.315 * s43 / ni
                    btmean = math.exp(means[i] + sds[i] ** 2 / 2) * (1 / phi3)
                    phi31 = 1 + 1.62 * sds[i] ** 2 / ni + 5.04 * s43 / ni
                    phi32 = 1 + 1.62 * sds[i] ** 2 / ni + 1.26 * s43 / ni
                    btsd = metafopy.core.linalg.rsqrt(
                        math.exp(2 * means[i] + 2 * sds[i] ** 2) * (1 / phi31) - math.exp(2 * means[i] + sds[i] ** 2) * (1 / phi32)
                    )
                    means[i], sds[i] = btmean, btsd

        if m1 == 'qe':
            test_now = False

    est: dict[str, numpy.typing.NDArray[numpy.bool_]] = {}
    if x is not None and append:
        for name, values in zip(var_names, (means, sds), strict=True):
            if name in x.columns:
                current = numpy.isnan(x[name].to_numpy(dtype=numpy.float64))
                est[name] = (current & ~numpy.isnan(values)) if repl == 'ifna' else ~numpy.isnan(values)
            metafopy.effects.conv.replace_column(x, name, values, repl)
        out = x
    else:
        out = pandas.DataFrame({var_names[0]: metafopy.effects.conv.quiet(means), var_names[1]: metafopy.effects.conv.quiet(sds)})
    out.attrs['tval'] = tval
    out.attrs['crit'] = crit
    out.attrs['sig'] = pandas.array(sig, dtype='boolean')
    out.attrs['dist'] = ['lnorm' if d == 'log-normal' else d for d in dists]
    if est:
        out.attrs['est'] = est
    return out
