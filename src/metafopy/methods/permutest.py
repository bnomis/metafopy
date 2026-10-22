"""Permutation tests for ``rma.uni`` models (metafor's ``permutest.rma.uni``).

The test statistics of the fitted model are compared with their distribution
under random permutations of the rows of the model matrix (or, for models
without moderators, random sign flips of the observed effects); an exact test
enumerates all (unique) permutations. See Follmann & Proschan (1999),
doi:10.1111/j.0006-341x.1999.00732.x, and Good (2009, *Permutation, Parametric
and Bootstrap Tests of Hypotheses*, 3rd ed.), and
https://wviechtb.github.io/metafor/reference/permutest.html.

Random permutations are drawn with R's own Mersenne-Twister
(:mod:`metafopy.core.rng`), so ``set_seed(s); permutest(res)`` reproduces
R's ``set.seed(s); permutest(res)`` and ``permutest(res, seed=s)`` that of
``permutest(res, seed=s)``.
"""

import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.core.parallel
import metafopy.core.rng
import metafopy.core.timing
import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.models.control_names
import metafopy.models.rma_uni
import metafopy.printing

if typing.TYPE_CHECKING:
    import metafopy.methods.ls_permutest

__all__ = ['PermutestResult', 'permutest']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class PermutestResult(metafopy.printing.HtmlRepr):
    """Permutation-based p-values (and optionally CI bounds) for the coefficients of a model, as returned by [permutest][metafopy.permutest]. ``to_frame()``
    and ``coef()`` give the coefficient table and ``summary()`` the printout.

    Attributes:
        pval: Permutation p-values of the coefficients.
        QMdf: Degrees of freedom of the omnibus test, ``(m, None)`` for the z test and ``(m, ddf)`` for the t test.
        QMp: Permutation p-value of the omnibus test.
        beta: Estimates of the coefficients.
        se: Standard errors of the coefficients.
        zval: Test statistics of the coefficients.
        ci_lb: Lower confidence interval bounds (permutation-based with ``permci``, else the usual ones).
        ci_ub: Upper confidence interval bounds.
        QM: Test statistic of the omnibus test.
        k: Number of studies.
        p: Number of coefficients.
        btt: Coefficients (0-based) the omnibus test refers to.
        m: Number of coefficients in the omnibus test.
        test: Test used by the model: ``'z'``, ``'t'``, ``'knha'`` or ``'adhoc'``.
        ddf: Residual degrees of freedom of the t test, or None for the z test.
        int_only: Whether the model has only an intercept.
        int_incl: Whether the model has an intercept.
        coef_names: Names of the coefficients.
        exact_iter: Number of iterations an exact test needs (``inf`` if it exceeds the largest float).
        permci: Whether permutation-based CI bounds were computed (False if ``permci`` was requested but the number of permutations is too low for
            the confidence level).
        alternative: Alternative hypothesis of the test (the ``alternative`` control entry; with permutation-based CI bounds, the one of the last bound
            search, as in metafor).
        p2defn: Definition of the two-sided p-value.
        stat: Statistic used for the coefficients.
        QM_perm: Omnibus statistics of every permutation (for the approximate test the first one is the observed one, as in metafor).
        zval_perm: Test statistics of every permutation, one column per coefficient.
        beta_perm: Estimates of every permutation, one column per coefficient.
        digits: Number of decimals of the printout.
    """

    pval: FloatArray
    QMdf: tuple[int, int | None]
    QMp: float
    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    QM: float
    k: int
    p: int
    btt: tuple[int, ...]
    m: int
    test: str
    ddf: int | None
    int_only: bool
    int_incl: bool
    coef_names: tuple[str, ...]
    exact_iter: int | float
    permci: bool
    alternative: str
    p2defn: str
    stat: str
    QM_perm: FloatArray
    zval_perm: pandas.DataFrame
    beta_perm: pandas.DataFrame
    digits: int = 4

    def to_frame(self) -> pandas.DataFrame:
        cols: dict[str, typing.Any] = {'estimate': self.beta, 'se': self.se, ('tval' if self.ddf is not None else 'zval'): self.zval}
        if self.ddf is not None:
            cols['df'] = numpy.full(self.p, self.ddf)
        cols.update({'pval': self.pval, 'ci.lb': self.ci_lb, 'ci.ub': self.ci_ub})
        return pandas.DataFrame(cols, index=list(self.coef_names))

    @property
    def dfs(self) -> int | None:
        """R's ``dfs``: the residual degrees of freedom (``ddf``) of the test."""
        return self.ddf

    @property
    def skip_beta(self) -> bool:
        """R's ``skip.beta``: the plain model never skips the coefficients."""
        return False

    def coef(self) -> pandas.DataFrame:
        """The table of the coefficients with the permutation p-values (``estimate``, ``se``, ``zval``, ``pval``, ``ci.lb``, ``ci.ub``), as
        metafor's ``coef.permutest.rma.uni`` (``tval`` and ``df`` for the t-test)."""
        return self.to_frame()

    def summary(self, digits: int | None = None, *, signif_stars: bool | None = None, signif_legend: bool | None = None, num: bool = False) -> str:
        """The printout of ``print(x, digits=, signif.stars=, signif.legend=, num=)``: ``digits`` default to those of the fit; ``num`` numbers the rows
        of the coefficient table."""
        stars, legend = metafopy.mfopt.option_signif(signif_stars, signif_legend)
        return format_permutest(self, self.digits if digits is None else metafopy.mfopt.option_digits(digits, self.digits), stars, legend, num)

    def __str__(self) -> str:
        return format_permutest(self)


@functools.singledispatch
def permutest(
    x: typing.Any, *args: typing.Any, **kwargs: typing.Any
) -> PermutestResult | int | float | metafopy.methods.ls_permutest.LSPermutestResult | list[float]:
    """Permutation tests for the coefficients of a fitted model (metafor's ``permutest``).

    This is a single-dispatch function: the implementation is chosen by the class of ``x``. See
    [permutest_rma_uni][metafopy.methods.permutest.permutest_rma_uni] and [permutest_rma_ls][metafopy.methods.ls_permutest.permutest_rma_ls].

    Args:
        x: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation (``exact``, ``iter``, ...).
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        A [PermutestResult][metafopy.PermutestResult] (a location-scale result for a location-scale model); with ``exact='i'`` the number of iterations an
        exact test needs.

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``x``.
        TypeError: the implementation does not accept a keyword argument.
    """
    raise metafopy.errors.ArgumentError(f'permutest() not implemented for {type(x).__name__}')


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
CONTROL_DEFAULTS: dict[str, typing.Any] = {
    'comptol': EPS**0.5,
    'tol': EPS**0.25,
    'maxiter': 100,
    'alternative': 'two.sided',
    'p2defn': 'abs',
    'stat': 'test',
    'cialt': 'one.sided',
    'distfac': 1.0,
    'extendInt': 'no',
}


def unique_perms_count(X: FloatArray) -> tuple[int | float, numpy.typing.NDArray[numpy.intp], FloatArray]:
    """Number of distinct row permutations of ``X``, the group labels of its sorted rows, and the sorted ``X``."""
    k = X.shape[0]
    order = numpy.lexsort(tuple(X[:, j] for j in range(X.shape[1] - 1, -1, -1)))
    Xs = X[order]
    new = numpy.ones(k, dtype=numpy.bool_)
    new[1:] = numpy.any(Xs[1:] != Xs[:-1], axis=1) if k > 1 else True
    # R: X.indices is the position of the first row of each run of identical rows
    # (duplicated() flags any earlier identical row, runs are contiguous after sorting)
    starts = numpy.flatnonzero(new)
    run_len = numpy.diff(numpy.append(starts, k))
    labels = numpy.repeat(starts + 1, run_len)
    counts = numpy.array(run_len)
    big = int(numpy.argmax(counts))
    rest = numpy.delete(counts, big)
    num = math.prod(range(int(counts.max()) + 1, k + 1))
    den = math.prod(math.factorial(int(c)) for c in rest)
    total: int | float = num // den
    if total > 1.7e308:
        total = math.inf
    return total, labels, Xs


def multiset_perms(labels: numpy.typing.NDArray[numpy.intp]) -> list[tuple[int, ...]]:
    """All distinct permutations of ``labels`` in lexicographic order (metafor's ``.genuperms``)."""
    items = sorted(labels.tolist())
    out: list[tuple[int, ...]] = []
    perm = list(items)
    while True:
        out.append(tuple(perm))
        i = len(perm) - 2
        while i >= 0 and perm[i] >= perm[i + 1]:
            i -= 1
        if i < 0:
            break
        j = len(perm) - 1
        while perm[j] <= perm[i]:
            j -= 1
        perm[i], perm[j] = perm[j], perm[i]
        perm[i + 1 :] = reversed(perm[i + 1 :])
    return out


def permutest_control(control: collections.abc.Mapping[str, typing.Any] | None) -> dict[str, typing.Any]:
    con = dict(CONTROL_DEFAULTS)
    for key, value in (control or {}).items():
        match = [name for name in con if name.startswith(key.replace('.', '_').replace('extend_int', 'extendInt'))]
        if key in con:
            match = [key]
        if len(match) != 1:
            if metafopy.models.control_names.read_elsewhere(key):
                continue
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        con[match[0]] = value
    if con['alternative'] not in ('two.sided', 'less', 'greater'):
        raise metafopy.errors.InputError("control 'alternative' must be 'two.sided', 'less' or 'greater'")
    if con['p2defn'] not in ('abs', 'px2'):
        raise metafopy.errors.InputError("control 'p2defn' must be 'abs' or 'px2'")
    if con['stat'] not in ('test', 'coef'):
        raise metafopy.errors.InputError("control 'stat' must be 'test' or 'coef'")
    return con


def nanmean(a: numpy.typing.NDArray[numpy.bool_] | FloatArray, valid: numpy.typing.NDArray[numpy.bool_]) -> float:
    n = int(valid.sum())
    return float(numpy.sum(numpy.asarray(a, dtype=numpy.float64)[valid])) / n if n else math.nan


@permutest.register
@metafopy.core.timing.timed
def permutest_rma_uni(
    x: metafopy.models.rma_uni.RMAResult,
    exact: bool | typing.Literal['i'] = False,
    iter: int = 1000,
    btt: metafopy.models.rma_uni.Btt = None,
    permci: bool | collections.abc.Sequence[int] = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    *,
    seed: int | None = None,
    tol: float | None = None,
    fixed: bool = False,
    progbar: bool = False,
    verbose: bool = False,
) -> PermutestResult | int | float:
    """Permutation test for an [rma_uni][metafopy.rma_uni] fit (metafor's ``permutest.rma.uni``).

    For a model without moderators the signs of the estimates are permuted, otherwise the rows of the model matrix. The p-value is the proportion of
    permutations whose statistic is as extreme as the observed one (the first permutation of an approximate test is the data, which avoids p-values of 0).
    With few studies the smallest possible p-value can exceed 0.05.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/permutest.rma.uni.r>; reference page:
    <https://wviechtb.github.io/metafor/reference/permutest.html>.

    Args:
        x: Model fitted by [rma_uni][metafopy.rma_uni] (not a [robust][metafopy.robust] result).
        exact: ``True`` carries out an exact test (all unique permutations; also used if ``iter`` exceeds their number); ``'i'`` returns the number of
            iterations an exact test needs; ``False`` (default) uses ``iter`` random permutations.
        iter: Number of random permutations for an approximate test (at least 2).
        btt: Coefficients the omnibus test refers to: 0-based indices or a string searched for (Python ``re.search``) in the coefficient names.
            Default: that of the model.
        permci: ``True``, or 0-based coefficient indices, to also find permutation-based confidence interval bounds (slow). A bound that cannot be found
            is NaN.
        digits: Number of decimals of the printout. Default: the ``digits`` option, else that of the model.
        control: Mapping with metafor's control values: ``comptol`` (tolerance of the comparisons), ``tol`` and ``maxiter`` (root finding for ``permci``),
            ``alternative`` (``'two.sided'``, ``'less'`` or ``'greater'``), ``p2defn`` (``'abs'`` or ``'px2'``), ``stat`` (``'test'`` or ``'coef'``),
            ``cialt`` (alternative for the CI bounds) and ``distfac`` (factor widening the search interval for the CI bounds).
        seed: Seed of the random permutations. Default: drawn from the R-compatible generator of [set_seed][metafopy.set_seed], which is left in the state
            the permutations ended in, as R's global generator is.
        tol: Old spelling of the ``comptol`` entry of ``control``.
        fixed: Match a string ``btt`` literally instead of as a regular expression.
        progbar: Print a progress bar to stderr while the permutations are run (default False; metafor's default is ``TRUE``).
        verbose: Accepted and ignored (metafor shows the errors of refits that fail during the permutations).

    Returns:
        A [PermutestResult][metafopy.PermutestResult] with the coefficient table (``beta``, ``se``, ``zval``, ``pval``, ``ci_lb``, ``ci_ub``), the omnibus
        test (``QM``, ``QMp``) and the permutation distributions. With ``exact='i'``, the number of iterations an exact test needs.

    Raises:
        metafopy.errors.InputError: ``iter`` is below 2; ``btt`` or ``permci`` names a coefficient that does not exist; a ``control`` entry is not
            accepted; an exact test needs more iterations than the largest float.
        metafopy.errors.ArgumentError: ``control`` has an unknown entry.
        metafopy.errors.NotSupportedError: ``x`` is a [robust][metafopy.robust] result.
        ValueError: an exact test needs more iterations than can be allocated (R stops with an error as well).
    """
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.NOT_ROBUST)
    iter = iter
    if iter <= 1:
        raise metafopy.errors.InputError("Argument 'iter' must be >= 2.")
    if btt is not None:
        bt = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names) if not (fixed and isinstance(btt, str)) else fixed_btt(btt, x)
        refitted = metafopy.methods.common.refit_arrays(x, btt=list(bt))
        if refitted is None:
            raise metafopy.errors.InputError('could not refit the model with the requested btt')
        x = refitted
    k, p = x.k, x.p

    if x.int_only:
        exact_iter: int | float = 2**k
        labels = numpy.arange(1, k + 1)
        Xs = x.X
    else:
        exact_iter, labels, Xs = unique_perms_count(x.X)
    if isinstance(exact, str) and exact == 'i':
        return exact_iter
    use_exact = bool(exact) or exact_iter <= iter
    n_iter = int(exact_iter) if use_exact else iter
    if use_exact and math.isinf(exact_iter):
        raise metafopy.errors.InputError('Too many iterations required for an exact permutation test.')

    con = permutest_control(control)
    if tol is not None:
        con['comptol'] = tol
    digits_ = metafopy.mfopt.stored_digits(digits, x.digits)

    gen: metafopy.core.rng.RRandom | None = None
    if not use_exact:
        seed_ = int(metafopy.core.rng.get_rng().unif_rand() * 2e9)
        gen = metafopy.core.rng.set_seed(seed_ if seed is None else seed)

    zval_perm, beta_perm, QM_perm = run_permutations(x, Xs, labels, n_iter, use_exact, gen, progbar)
    if not use_exact:
        beta_perm[0] = x.beta
        zval_perm[0] = x.zval
        QM_perm[0] = x.QM

    pval = pvalues(x, zval_perm, beta_perm, con)
    QMp = nanmean(QM_perm >= x.QM - con['comptol'], ~numpy.isnan(QM_perm))

    ci_lb, ci_ub = x.ci_lb.copy(), x.ci_ub.copy()
    did_permci = False
    if permci is not False and permci is not None:
        did_permci = True
        alpha = x.level
        per = 1 if con['cialt'] == 'one.sided' else 2
        if 1 / n_iter > alpha / per:
            did_permci = False
            warnings.warn(f'Cannot obtain {100 * (1 - x.level):g}% permutation-based CI; number of permutations ({n_iter}) too low.', stacklevel=2)
        else:
            if permci is True:
                coefs = list(range(p))
            else:
                coefs = sorted({round(c) for c in permci})
                if any(c < 0 or c >= p for c in coefs):
                    raise metafopy.errors.InputError("Non-existent coefficients specified via 'permci'.")
            ci_lb = numpy.full(p, numpy.nan)
            ci_ub = numpy.full(p, numpy.nan)
            for j in coefs:
                ci_lb[j] = permci_bound(x, j, use_exact, n_iter, alpha, con, lower=True, base_ci=x.ci_lb[j], base_ub=x.ci_ub[j])
                ci_ub[j] = permci_bound(x, j, use_exact, n_iter, alpha, con, lower=False, base_ci=x.ci_lb[j], base_ub=x.ci_ub[j])

    return PermutestResult(
        pval=pval,
        QMdf=(x.QMdf[0], None if x.QMdf[1] is None else int(x.QMdf[1])),
        QMp=QMp,
        beta=x.beta,
        se=x.se,
        zval=x.zval,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        QM=x.QM,
        k=k,
        p=p,
        btt=x.btt,
        m=x.m,
        test=x.test,
        ddf=x.ddf,
        int_only=x.int_only,
        int_incl=x.int_incl,
        coef_names=x.coef_names,
        exact_iter=exact_iter,
        permci=did_permci,
        alternative=con['alternative'] if not did_permci else final_alt(con),
        p2defn=con['p2defn'],
        stat=con['stat'],
        QM_perm=QM_perm,
        zval_perm=pandas.DataFrame(zval_perm, columns=list(x.coef_names)),
        beta_perm=pandas.DataFrame(beta_perm, columns=list(x.coef_names)),
        digits=digits_,
    )


def final_alt(con: collections.abc.Mapping[str, typing.Any]) -> str:
    # metafor overwrites con$alternative while searching for the CI bounds; the last search is for the upper bound
    return 'less' if con['cialt'] == 'one.sided' else 'two.sided'


def fixed_btt(btt: str, x: metafopy.models.rma_uni.RMAResult) -> tuple[int, ...]:
    idx = tuple(i for i, n in enumerate(x.coef_names) if btt in n)
    if not idx:
        raise metafopy.errors.InputError("Cannot identify coefficient(s) corresponding to the specified 'btt' string.")
    return idx


def run_permutations(
    x: metafopy.models.rma_uni.RMAResult,
    Xs: FloatArray,
    labels: numpy.typing.NDArray[numpy.intp],
    n_iter: int,
    exact: bool,
    gen: metafopy.core.rng.RRandom | None,
    progbar: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    k, p = x.k, x.p
    zval = numpy.full((n_iter, p), numpy.nan)
    beta = numpy.full((n_iter, p), numpy.nan)
    QM = numpy.full(n_iter, numpy.nan)

    def store(i: int, fit: metafopy.models.rma_uni.RMAResult | None) -> bool:
        if fit is None:
            return False
        beta[i] = fit.beta
        zval[i] = fit.zval
        QM[i] = fit.QM
        return True

    if x.int_only:
        one = numpy.ones((k, 1))
        if exact:
            for i in metafopy.core.parallel.progress(range(n_iter), n_iter, progbar):
                signs = numpy.array([1.0 if not (i >> j) & 1 else -1.0 for j in range(k)])
                store(i, metafopy.methods.common.refit_arrays(x, yi=signs * x.yi, X=one, btt=[0]))
        else:
            assert gen is not None
            i = 0
            while i < n_iter:
                signs = numpy.where(gen.sample_int(2, k, replace=True) == 1, -1.0, 1.0)
                if store(i, metafopy.methods.common.refit_arrays(x, yi=signs * x.yi, X=one, btt=[0])):
                    i += 1
                    metafopy.core.parallel.tick(i, n_iter, progbar)
        return zval, beta, QM

    btt = list(x.btt)
    if exact:
        perms = multiset_perms(labels)
        for i in metafopy.core.parallel.progress(range(n_iter), n_iter, progbar):
            store(i, metafopy.methods.common.refit_arrays(x, X=Xs[numpy.array(perms[i]) - 1], btt=btt))
    else:
        assert gen is not None
        i = 0
        while i < n_iter:
            perm = gen.sample_int(k) - 1
            if store(i, metafopy.methods.common.refit_arrays(x, X=Xs[perm], btt=btt)):
                i += 1
                metafopy.core.parallel.tick(i, n_iter, progbar)
    return zval, beta, QM


def pvalues(x: metafopy.models.rma_uni.RMAResult, zval_perm: FloatArray, beta_perm: FloatArray, con: collections.abc.Mapping[str, typing.Any]) -> FloatArray:
    comptol = con['comptol']
    use_test = con['stat'] == 'test'
    stat_perm = zval_perm if use_test else beta_perm
    stat_obs = x.zval if use_test else x.beta
    pval = numpy.full(x.p, numpy.nan)
    for j in range(x.p):
        s = stat_perm[:, j]
        valid = ~numpy.isnan(s)
        obs = stat_obs[j]
        if con['alternative'] == 'two.sided':
            if con['p2defn'] == 'abs':
                pval[j] = nanmean(numpy.abs(s) >= abs(obs) - comptol, valid)
            elif obs > float(numpy.nanmedian(s)):
                pval[j] = 2 * nanmean(s >= obs - comptol, valid)
            else:
                pval[j] = 2 * nanmean(s <= obs + comptol, valid)
        elif con['alternative'] == 'less':
            pval[j] = nanmean(s <= obs + comptol, valid)
        else:
            pval[j] = nanmean(s >= obs - comptol, valid)
    return numpy.where(pval > 1, 1.0, pval)


def permci_bound(
    x: metafopy.models.rma_uni.RMAResult,
    j: int,
    exact: bool,
    n_iter: int,
    alpha: float,
    con: collections.abc.Mapping[str, typing.Any],
    *,
    lower: bool,
    base_ci: float,
    base_ub: float,
) -> float:
    """Search for a permutation-based CI bound of coefficient ``j`` (metafor's ``.permci`` with ``uniroot``).

    Raises:
        metafopy.errors.InputError: the model cannot be refitted.
    """
    c = dict(con)
    one_sided = c['cialt'] == 'one.sided'
    if lower:
        c['alternative'] = 'greater' if one_sided else 'two.sided'
        interval = (base_ci - c['distfac'] * (x.beta[j] - base_ci), float(x.beta[j]))
    else:
        c['alternative'] = 'less' if one_sided else 'two.sided'
        interval = (float(x.beta[j]), base_ub + c['distfac'] * (base_ub - x.beta[j]))

    def f(val: float) -> float:
        fit = metafopy.methods.common.refit_arrays(x, yi=x.yi - val * x.X[:, j])
        if fit is None:
            raise metafopy.errors.InputError('refit failed')
        res = permutest_rma_uni(fit, exact=exact, iter=n_iter, control=c)
        assert isinstance(res, PermutestResult)
        diff = float(res.pval[j]) - alpha / (1 if c['alternative'] == 'two.sided' else 2)
        return diff * 10 if diff < 0 else diff

    try:
        # as metafor: any extendInt other than 'no' extends the interval upwards for the lower bound and downwards for the upper bound
        extend_int: typing.Literal['no', 'upX', 'downX'] = 'no' if c['extendInt'] == 'no' else 'upX' if lower else 'downX'
        return metafopy.core.zeroin.uniroot(f, interval[0], interval[1], tol=c['tol'], maxiter=c['maxiter'], extend_int=extend_int).root
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError:
        return math.nan


def format_permutest(x: PermutestResult, d: int | None = None, stars: bool = True, legend: bool = True, num: bool = False) -> str:
    d = x.digits if d is None else d
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    out: list[str] = ['']
    if not x.int_only:
        out.append(f'Test of Moderators (coefficient{"s" if x.m > 1 else ""} {metafopy.models.rma_uni.coef_span(x.btt)}):¹')
        if x.ddf is not None:
            out.append(f'F(df1 = {x.QMdf[0]}, df2 = {x.QMdf[1]}) = {metafopy.printing.fmt_num(x.QM, d)}, p-val {pfmt(x.QMp, d)}')
        else:
            out.append(f'QM(df = {x.QMdf[0]}) = {metafopy.printing.fmt_num(x.QM, d)}, p-val {pfmt(x.QMp, d)}')
        out.append('')
    out.append('Model Results:')
    out.append('')
    header = ['estimate', 'se', 'tval' if x.ddf is not None else 'zval']
    if x.ddf is not None:
        header.append('df')
    header += ['pval¹', *(['ci.lb¹', 'ci.ub¹'] if x.permci else ['ci.lb', 'ci.ub']), '']
    rows: list[list[str]] = []
    for j in range(x.p):
        row = [metafopy.printing.fmt_num(x.beta[j], d), metafopy.printing.fmt_num(x.se[j], d), metafopy.printing.fmt_num(x.zval[j], d)]
        if x.ddf is not None:
            row.append(f'{x.ddf:g}')
        # metafor's .addfootsym() also appends a space to the cells of the footnoted columns
        pad = ' ' if x.permci else ''
        row += [
            metafopy.printing.fmt_pval(x.pval[j], d) + ' ',
            metafopy.printing.fmt_num(x.ci_lb[j], d) + pad,
            metafopy.printing.fmt_num(x.ci_ub[j], d) + pad,
            metafopy.printing.signif_stars(x.pval[j]),
        ]
        rows.append(row)
    header, rows = metafopy.printing.drop_stars(header, rows, stars)
    if x.int_only:
        out.extend(metafopy.methods.vif.format_vector(header, rows[0]))
    else:
        out.append(metafopy.printing.fmt_table(header, rows, row_names=metafopy.printing.number_rows(x.coef_names) if num else list(x.coef_names)))
    what = f'p-value{"" if x.int_only else "s"}{" and CI bounds" if x.permci else ""}'
    out += metafopy.printing.signif_footer(legend, (f'1) {what} based on permutation testing',))
    return '\n'.join(out)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def pfmt(p: float, digits: int) -> str:
    s = metafopy.printing.fmt_pval(p, digits)
    return f'< {s[1:]}' if s.startswith('<') else f'= {s}'
