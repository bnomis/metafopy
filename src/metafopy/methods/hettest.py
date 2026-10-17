"""Test for heteroscedastic heterogeneity (metafor's ``hettest``, new in metafor 5.2-1).

Tests whether the amount of heterogeneity differs across the studies of a random-effects model, :math:`H_0: \\tau^2_1 = \\dots = \\tau^2_k`,
with a likelihood ratio, Wald-type or score test (all with a parametric bootstrap p-value by default), or by checking the Pearson residuals with
Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit tests (Kolmogorov 1933, Anderson & Darling 1952, doi:10.1214/aoms/1177729437).
The bootstrap draws from R's generator (see :func:`metafopy.set_seed`), so ``seed=`` reproduces ``set.seed(seed)`` in R.
See https://wviechtb.github.io/metafor/reference/hettest.html and
https://github.com/wviechtb/metafor/blob/master/R/hettest.r.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.bias.inputs
import metafopy.core.parallel
import metafopy.core.rng
import metafopy.errors
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.control_names
import metafopy.models.rma_uni

__all__ = ['HettestResult', 'hettest']

type FloatArray = numpy.typing.NDArray[numpy.float64]

TESTS = ('adn', 'adx2', 'ksn', 'ksx2', 'lrt', 'score', 'wald')
GOODNESS_OF_FIT = ('adn', 'adx2', 'ksn', 'ksx2')
MODEL_TESTS = ('lrt', 'score', 'wald')
TAU2I_TESTS = ('lrt', 'wald')
ESTIMATION_METHODS = ('ML', 'REML')
# auditme: accept[non-alphabetical-dict-keys] order follows metafor's print method
TEST_LABEL = {
    'lrt': 'Likelihood ratio test',
    'wald': 'Wald-type test',
    'score': 'Score test',
    'ksn': 'Kolmogorov-Smirnov test (normal)',
    'ksx2': 'Kolmogorov-Smirnov test (chi-squared)',
    'adn': 'Anderson-Darling test (normal)',
    'adx2': 'Anderson-Darling test (chi-squared)',
}
EPS = float(numpy.finfo(numpy.float64).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class HettestResult:
    """Result of :func:`hettest`: one entry per requested test in ``statistic``, ``df`` (``NaN`` for goodness-of-fit tests), ``pval``, ``test``, ``boot``.

    ``tau2i`` are the estimated study-specific heterogeneity variances (``None`` where they could not be estimated), ``se_tau2i`` their standard errors
    (only with the Wald-type test) and ``statistic_boot`` the bootstrapped statistics, one column per test (``NaN`` for iterations that failed).
    """

    statistic: FloatArray
    df: FloatArray
    pval: FloatArray
    method: str
    test: tuple[str, ...]
    boot: tuple[bool, ...]
    iter: int
    digits: int
    tau2i: FloatArray | None
    se_tau2i: FloatArray | None = None
    statistic_boot: FloatArray | None = None

    def summary(self, digits: int | None = None) -> str:
        d = metafopy.mfopt.option_digits(digits, self.digits)
        method = 'Maximum likelihood' if self.method == 'ML' else 'Restricted maximum likelihood'
        out = ''
        for j, test in enumerate(self.test):
            boot = 'No'
            if self.boot[j]:
                done = 0 if self.statistic_boot is None else int(numpy.sum(~numpy.isnan(self.statistic_boot[:, j])))
                boot = f'Yes ({done}/{self.iter} iterations)'
            pval = metafopy.bias.inputs.fp(float(self.pval[j]), d, equal=True, sep=True, pname='p ')
            stat = metafopy.bias.inputs.fx(float(self.statistic[j]), d)
            result = f'X^2(df = {int(self.df[j])}) = {stat}, {pval}' if test in MODEL_TESTS else f'statistic = {stat}, {pval}'
            out += (
                '\nTest for Heteroscedastic Heterogeneity\n\n'
                f'Estimation method: {method}\n'
                f'Test type:         {TEST_LABEL[test]}\n'
                f'Bootstrapping:     {boot}\n\n'
                f'{result}\n\n'
            )
        return out

    def __str__(self) -> str:
        return self.summary()


@dataclasses.dataclass(frozen=True, slots=True)
class NullFit:
    """What the tests need of the random-effects model: ``beta``, ``tau2``, its log-likelihood and the raw residuals and leverages."""

    beta: float
    tau2: float
    ll: float
    ei: FloatArray
    hi: FloatArray


def solve_or_none(a: FloatArray) -> FloatArray | None:
    """The inverse of ``a``, or ``None`` where R's ``solve()`` fails (singular, or reciprocal condition number below machine epsilon)."""
    try:
        rcond = 1 / numpy.linalg.cond(a, 1)
        if not math.isfinite(rcond) or rcond < EPS:
            return None
        out: FloatArray = numpy.linalg.inv(a)
    except numpy.linalg.LinAlgError:
        return None
    return out


def re_fit_quick(yi: FloatArray, vi: FloatArray, method: str, threshold: float, maxiter: int) -> NullFit | None:
    """metafor's ``.re.fit.quick``: the ML or REML estimate of tau^2 by Fisher scoring from the DerSimonian-Laird start; ``None`` where it does not converge."""
    k = yi.shape[0]
    wi = 1 / vi
    sumwi = float(numpy.sum(wi))
    theta = float(numpy.sum(wi * yi)) / sumwi
    q = float(numpy.sum(wi * (yi - theta) ** 2))
    tau2 = max(0.0, (q - (k - 1)) / (sumwi - float(numpy.sum(wi**2)) / sumwi))
    tau2 = max(0.01, tau2)
    diff = 1.0
    it = 1
    mu = theta
    while diff > threshold:
        if it > maxiter:
            return None
        old = tau2
        wi = 1 / (tau2 + vi)
        sumwi = float(numpy.sum(wi))
        mu = float(numpy.sum(wi * yi)) / sumwi
        tau2 = float(numpy.sum(wi**2 * ((yi - mu) ** 2 - vi)) / numpy.sum(wi**2))
        if method == 'REML':
            tau2 += 1 / sumwi
        tau2 = max(0.0, tau2)
        diff = abs(tau2 - old)
        it += 1
    wi = 1 / (tau2 + vi)
    sumwi = float(numpy.sum(wi))
    mu = float(numpy.sum(wi * yi)) / sumwi
    ll = float(numpy.sum(scipy.stats.norm.logpdf(yi, loc=mu, scale=numpy.sqrt(vi + tau2))))
    if method == 'REML':
        ll += 0.5 * math.log(2 * k * math.pi) - 0.5 * math.log(sumwi)
    return NullFit(beta=mu, tau2=tau2, ll=ll, ei=yi - mu, hi=wi / sumwi)


def est_tau2i(
    yi: FloatArray, vi: FloatArray, method: str, res0: NullFit, mom: bool, tau2i_init: FloatArray, threshold: float, maxiter: int
) -> FloatArray | None:
    """metafor's ``.hettest.esttau2i``: the study-specific heterogeneity variances (``None`` where the iterations do not converge)."""
    if mom:
        return numpy.maximum(0.0, res0.ei**2 / (1 - res0.hi) - vi)
    tau2i = tau2i_init.copy()
    diffs = numpy.full(yi.shape[0], 10.0)
    it = 1
    while numpy.any(diffs > threshold):
        if it > maxiter:
            return None
        old = tau2i
        wi = 1 / (tau2i + vi)
        sumwi = float(numpy.sum(wi))
        mu = float(numpy.sum(wi * yi)) / sumwi
        tau2i = (yi - mu) ** 2 - vi
        if method == 'REML':
            tau2i = tau2i + 1 / sumwi
        tau2i = numpy.maximum(tau2i, 0.0)
        diffs = numpy.abs(tau2i - old)
        it += 1
    return tau2i


type TestOut = tuple[float, float, float]


def lrt(yi: FloatArray, vi: FloatArray, method: str, res0: NullFit, tau2i: FloatArray) -> TestOut:
    k = yi.shape[0]
    vari = tau2i + vi
    wi = 1 / vari
    mu = float(numpy.sum(wi * yi) / numpy.sum(wi))
    llsat = float(numpy.sum(scipy.stats.norm.logpdf(yi, loc=mu, scale=numpy.sqrt(vari))))
    if method == 'REML':
        llsat += 0.5 * math.log(2 * k * math.pi) - 0.5 * math.log(float(numpy.sum(wi)))
    x2 = -2 * (res0.ll - llsat)
    return x2, k - 1, float(scipy.stats.chi2.sf(x2, k - 1))


def reml_information(vari: FloatArray) -> FloatArray:
    wi = 1 / vari
    s = float(numpy.sum(wi))
    a = 1 / (math.sqrt(2) * s * vari**2)
    info: FloatArray = numpy.outer(a, a)
    numpy.fill_diagonal(info, 0.5 * (wi - 1 / (s * vari**2)) ** 2)
    return info


def wald(yi: FloatArray, vi: FloatArray, method: str, tau2i: FloatArray) -> tuple[TestOut, FloatArray]:
    k = yi.shape[0]
    vari = tau2i + vi
    nan_se = numpy.full(k, math.nan)
    if method == 'ML':
        v_tau2 = numpy.diag(2 * vari**2)
    else:
        v_tau2_inv = solve_or_none(reml_information(vari))
        if v_tau2_inv is None:
            return (math.nan, k - 1, math.nan), nan_se
        v_tau2 = v_tau2_inv
    se = numpy.sqrt(numpy.diag(v_tau2))
    x = numpy.hstack([numpy.ones((k - 1, 1)), -numpy.eye(k - 1)])
    diff = x @ tau2i
    w_diff = solve_or_none(x @ v_tau2 @ x.T)
    if w_diff is None:
        return (math.nan, k - 1, math.nan), se
    x2 = float(diff @ w_diff @ diff)
    return (x2, k - 1, float(scipy.stats.chi2.sf(x2, k - 1))), se


def score(yi: FloatArray, vi: FloatArray, method: str, res0: NullFit) -> TestOut:
    k = yi.shape[0]
    vari = res0.tau2 + vi
    wi = 1 / vari
    if method == 'ML':
        x2 = 0.5 * float(numpy.sum((wi * (yi - res0.beta) ** 2 - 1) ** 2))
    else:
        s = float(numpy.sum(wi))
        ui = 0.5 * wi**2 * (yi - res0.beta) ** 2 - 0.5 * wi + 0.5 * wi**2 / s
        inv = solve_or_none(reml_information(vari))
        if inv is None:
            return math.nan, k - 1, math.nan
        x2 = float(ui @ inv @ ui)
    return x2, k - 1, float(scipy.stats.chi2.sf(x2, k - 1))


def ks(x: FloatArray, cdf: collections.abc.Callable[[FloatArray], FloatArray]) -> TestOut:
    k = x.shape[0]
    fx = cdf(numpy.sort(x))
    dplus = float(numpy.max(numpy.arange(1, k + 1) / k - fx))
    dminus = float(numpy.max(fx - numpy.arange(0, k) / k))
    return max(dplus, dminus), math.nan, math.nan


def ad(x: FloatArray, cdf: collections.abc.Callable[[FloatArray], FloatArray]) -> TestOut:
    k = x.shape[0]
    fx = numpy.clip(cdf(numpy.sort(x)), EPS, 1 - EPS)
    a2 = -k - float(numpy.mean((2 * numpy.arange(1, k + 1) - 1) * (numpy.log(fx) + numpy.log(1 - fx[::-1]))))
    return a2, math.nan, math.nan


def pnorm(x: FloatArray) -> FloatArray:
    out: FloatArray = scipy.stats.norm.cdf(x)
    return out


def pchisq1(x: FloatArray) -> FloatArray:
    out: FloatArray = scipy.stats.chi2.cdf(x, 1)
    return out


def run_test(
    name: str, yi: FloatArray, vi: FloatArray, method: str, res0: NullFit, tau2i: FloatArray | None, ri: FloatArray
) -> tuple[TestOut, FloatArray | None]:
    """One test on the data ``yi``; ``tau2i`` is only used (and required) by the likelihood ratio and Wald-type tests."""
    if name == 'lrt':
        return lrt(yi, vi, method, res0, typing.cast(FloatArray, tau2i)), None
    if name == 'wald':
        return wald(yi, vi, method, typing.cast(FloatArray, tau2i))
    if name == 'score':
        return score(yi, vi, method, res0), None
    if name == 'ksn':
        return ks(ri, pnorm), None
    if name == 'ksx2':
        return ks(ri**2, pchisq1), None
    if name == 'adn':
        return ad(ri, pnorm), None
    return ad(ri**2, pchisq1), None


def expand_boot(boot: bool | int | collections.abc.Sequence[bool], tests: tuple[str, ...]) -> tuple[list[bool], int]:
    """The per-test bootstrap flags and the number of iterations; a number is the iteration count (bootstrapping all tests).

    Raises:
        metafopy.errors.InputError: ``boot`` has neither one entry nor one per test.
    """
    ntest = len(tests)
    if not isinstance(boot, bool) and isinstance(boot, int | float):
        return [True] * ntest, max(1, round(boot))
    flags = [bool(boot)] * ntest if isinstance(boot, bool) else [bool(b) for b in boot]
    if len(flags) == 1:
        flags *= ntest
    if len(flags) != ntest:
        raise metafopy.errors.InputError(f"Length of the 'boot' argument ({len(flags)}) does not match the number of tests ({ntest}).")
    return [True if t in GOODNESS_OF_FIT else f for t, f in zip(tests, flags, strict=True)], 1000


def hettest(
    x: metafopy.methods.fitview.Fit | numpy.typing.ArrayLike | str,
    vi: numpy.typing.ArrayLike | str | None = None,
    *,
    sei: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    data: pandas.DataFrame | None = None,
    method: str | None = None,
    test: str | collections.abc.Sequence[str] = 'score',
    boot: bool | int | collections.abc.Sequence[bool] = True,
    progbar: bool = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    seed: int | None = None,
    mom: bool = False,
    model: metafopy.models.rma_uni.RMAResult | None = None,
) -> HettestResult:
    """Test for heteroscedastic heterogeneity in a random-effects model; ``x`` is a vector of estimates (with ``vi`` or ``sei``) or a model without moderators.

    ``test`` is one or more of ``'score'`` (default), ``'lrt'``, ``'wald'`` and the goodness-of-fit tests of the Pearson residuals ``'ksn'``, ``'ksx2'``,
    ``'adn'`` and ``'adx2'``. The p-values come from a parametric bootstrap (``boot=True`` uses 1000 samples, a number sets the count; the
    goodness-of-fit tests always bootstrap), the first sample being the data. ``method`` is ``'REML'`` (default) or ``'ML'``; for a model it is that of the
    model. ``control`` takes ``tau2i_init``, ``threshold`` (1e-8) and ``maxiter`` (1000) and ``mom=True`` estimates the tau^2_i by the method of moments.
    ``model`` is metafor's undocumented argument for a vector ``x``: an intercept-only model already fitted to the same data, used as the null model
    instead of fitting it again (its ``beta`` and ``tau2`` and the log-likelihood of the ``method`` are used).
    Unlike metafor, ``progbar`` is off by default. Without ``seed`` the bootstrap uses the generator of :func:`metafopy.set_seed`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, or the null model or the tau^2_i could not be estimated.
        metafopy.errors.NotSupportedError: ``x`` is a model of a class the test is not available for.
    """
    tests = (test.lower(),) if isinstance(test, str) else tuple(t.lower() for t in test)
    if not tests or any(t not in TESTS for t in tests):
        raise metafopy.errors.InputError("Unknown option specified for the 'test' argument.")
    if method is not None and method not in ESTIMATION_METHODS:
        raise metafopy.errors.InputError('Test must be based on ML/REML estimation.')
    unknown = {key.replace('.', '_') for key in control or {} if not metafopy.models.control_names.read_elsewhere(key)} - {'maxiter', 'tau2i_init', 'threshold'}
    if unknown:
        raise metafopy.errors.InputError(f'Unknown control argument {sorted(unknown)[0]!r}.')
    con = {key.replace('.', '_'): value for key, value in (control or {}).items()}
    threshold = float(con.get('threshold', 1e-8))
    maxiter = int(con.get('maxiter', 1000))
    if metafopy.methods.fitview.is_rma(x):
        if model is not None:
            warnings.warn("Argument 'model' ignored when 'x' is a model object.", stacklevel=2)
        model = None
        metafopy.methods.fitview.check_class(x, notav=('rma.glmm', 'rma.mv', 'robust.rma', 'rma.ls', 'rma.gen', 'rma.uni.selmodel'))
        view = metafopy.methods.fitview.fit_view(x)
        if not view.int_only:
            raise metafopy.errors.InputError('Test only applicable to models without moderators.')
        if vi is not None or sei is not None or subset is not None:
            warnings.warn("Arguments 'vi', 'sei', and 'subset' ignored when 'x' is a model object.", stacklevel=2)
        if method is not None:
            warnings.warn("Argument 'method' ignored when 'x' is a model object.", stacklevel=2)
        yi, v = view.yi, view.vi
        if view.method in ESTIMATION_METHODS:
            fit_method = view.method
            model = typing.cast(metafopy.models.rma_uni.RMAResult, x)
        else:
            fit_method = 'REML' if method is None else method
            warnings.warn(f'Model was not fitted with ML or REML estimation, but the test will be based on {fit_method} estimation.', stacklevel=2)
        stored_digits = getattr(x, 'digits', 4)
    else:
        fit_method = 'REML' if method is None else method
        yi, v_in, _ = metafopy.bias.inputs.vector_inputs(typing.cast(numpy.typing.ArrayLike | str, x), vi, sei, subset=subset, data=data, what='the test')
        if v_in is None:
            raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
        v = v_in
        stored_digits = 4
        if model is not None and (not isinstance(model, metafopy.models.rma_uni.RMAResult) or model.k != yi.shape[0]):
            raise metafopy.errors.InputError("Argument 'model' must be a model fitted to the same estimates as 'x'.")
    k = yi.shape[0]
    if k <= 1:
        raise metafopy.errors.InputError('Stopped because k <= 1.')
    boot_flags, iterations = expand_boot(boot, tests)
    any_boot = any(boot_flags)

    if model is None:
        res0 = re_fit_quick(yi, v, fit_method, threshold, maxiter)
        if res0 is None:
            raise metafopy.errors.InputError('Could not fit the null model.')
    else:
        beta = float(model.beta[0])
        wi = 1 / (v + model.tau2)
        res0 = NullFit(beta=beta, tau2=float(model.tau2), ll=model.fit_stats(fit_method == 'REML').ll, ei=yi - beta, hi=wi / float(numpy.sum(wi)))

    init = numpy.full(k, max(res0.tau2, 0.01))
    if con.get('tau2i_init') is not None:
        given = numpy.atleast_1d(numpy.asarray(con['tau2i_init'], dtype=numpy.float64))
        if given.shape[0] not in (1, k):
            raise metafopy.errors.InputError(
                f"Length of the 'tau2i_init' argument ({given.shape[0]}) does not match the actual number of variance components ({k})."
            )
        if numpy.isnan(given).any():
            raise metafopy.errors.InputError("No missing values allowed in 'tau2i_init'.")
        if (given < 0).any():
            raise metafopy.errors.InputError("Value(s) of 'tau2i_init' must be >= 0.")
        init = numpy.broadcast_to(given, (k,)).copy()

    vari = v + res0.tau2
    ri = (yi - res0.beta) / numpy.sqrt(vari)

    sim: FloatArray | None = None
    if any_boot:
        gen = metafopy.core.rng.set_seed(seed) if seed is not None else metafopy.core.rng.get_rng()
        sim = numpy.column_stack([res0.beta + numpy.sqrt(vari) * gen.rnorm(k) for _ in range(iterations)])
        sim[:, 0] = yi

    tau2i = est_tau2i(yi, v, fit_method, res0, mom, init, threshold, maxiter)
    if tau2i is None and any(t in TAU2I_TESTS for t in tests):
        raise metafopy.errors.InputError('Could not estimate the tau^2_i values.')

    ntest = len(tests)
    statistic = numpy.zeros(ntest)
    df = numpy.zeros(ntest)
    pval = numpy.zeros(ntest)
    se_tau2i: FloatArray | None = None
    for j, name in enumerate(tests):
        (statistic[j], df[j], pval[j]), se = run_test(name, yi, v, fit_method, res0, tau2i, ri)
        if name == 'wald':
            se_tau2i = se

    statistic_boot: FloatArray | None = None
    if any_boot and sim is not None:
        sim_data = sim
        statistic_boot = numpy.full((iterations, ntest), math.nan)

        def one(b: int) -> FloatArray | None:
            sb = sim_data[:, b]
            res0_boot = re_fit_quick(sb, v, fit_method, threshold, maxiter)
            if res0_boot is None:
                return None
            ri_boot = (sb - res0_boot.beta) / numpy.sqrt(v + res0_boot.tau2)
            tau2i_boot: FloatArray | None = None
            if any(t in TAU2I_TESTS for t in tests):
                tau2i_boot = est_tau2i(sb, v, fit_method, res0_boot, mom, init, threshold, maxiter)
                if tau2i_boot is None:
                    return None
            row = numpy.full(ntest, math.nan)
            for j, name in enumerate(tests):
                if boot_flags[j]:
                    row[j] = run_test(name, sb, v, fit_method, res0_boot, tau2i_boot, ri_boot)[0][0]
            return row

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for b, row in enumerate(metafopy.core.parallel.progress((one(b) for b in range(iterations)), iterations, progbar)):
                if row is not None:
                    statistic_boot[b] = row
        if not math.isnan(statistic[-1]):
            for j in range(ntest):
                if boot_flags[j]:
                    col = statistic_boot[:, j]
                    col = col[~numpy.isnan(col)]
                    pval[j] = float(numpy.mean(col > statistic[j])) if col.size else math.nan

    return HettestResult(
        statistic=statistic,
        df=df,
        pval=pval,
        method=fit_method,
        test=tests,
        boot=tuple(boot_flags),
        iter=iterations,
        digits=metafopy.mfopt.stored_digits(digits, stored_digits),
        tau2i=tau2i,
        se_tau2i=se_tau2i,
        statistic_boot=statistic_boot,
    )
