"""Permutation tests for ``rma.uni`` models (metafor's ``permutest.rma.uni``).

The test statistics of the fitted model are compared with their distribution
under random permutations of the rows of the model matrix (or, for models
without moderators, random sign flips of the observed effects); an exact test
enumerates all (unique) permutations. See Follmann & Proschan (1999),
doi:10.1111/j.0006-341X.1999.00254.x, and Good (2005, *Permutation, Parametric
and Bootstrap Tests of Hypotheses*), and
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

import metafopy.core.rng
import metafopy.core.zeroin
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.vif
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['PermutestResult', 'permutest']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class PermutestResult:
    """Permutation-based p-values (and optionally CI bounds) for the coefficients of a model.

    ``QM_perm``, ``zval_perm`` and ``beta_perm`` hold the statistics of every
    permutation (for the approximate test, the first row is the observed one,
    as in metafor).
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

    def __str__(self) -> str:
        return format_permutest(self)


@functools.singledispatch
def permutest(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> PermutestResult | int | float:
    """Permutation test for a fitted model; see :func:`_permutest_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
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
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        con[match[0]] = value
    if con['alternative'] not in ('two.sided', 'less', 'greater'):
        raise metafopy.errors.InputError("control 'alternative' must be 'two.sided', 'less' or 'greater'")
    if con['p2defn'] not in ('abs', 'px2'):
        raise metafopy.errors.InputError("control 'p2defn' must be 'abs' or 'px2'")
    if con['stat'] not in ('test', 'coef'):
        raise metafopy.errors.InputError("control 'stat' must be 'test' or 'coef'")
    if con['extendInt'] != 'no':
        raise metafopy.errors.NotSupportedError("control extendInt other than 'no' is not supported")
    return con


def nanmean(a: numpy.typing.NDArray[numpy.bool_] | FloatArray, valid: numpy.typing.NDArray[numpy.bool_]) -> float:
    n = int(valid.sum())
    return float(numpy.sum(numpy.asarray(a, dtype=numpy.float64)[valid])) / n if n else math.nan


@permutest.register
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
) -> PermutestResult | int | float:
    """Permutation test for an ``rma.uni`` fit.

    ``exact=True`` (or enough iterations to enumerate all permutations) runs an
    exact test; ``exact='i'`` returns the number of iterations an exact test
    needs. ``iter`` is the number of random permutations otherwise. ``btt``
    (0-based) changes the coefficients the omnibus test refers to. ``permci``
    (``True`` or 0-based coefficient indices) also searches for permutation-based
    confidence interval bounds. ``seed`` plays the role of ``seed=`` in
    metafor; without it the seed is drawn from the module's R-compatible
    generator (see :func:`metafopy.set_seed`), which like R's global generator
    is left in the state the permutations ended in. ``control`` takes metafor's
    list (``comptol``, ``tol``, ``maxiter``, ``alternative``, ``p2defn``,
    ``stat``, ``cialt``, ``distfac``).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
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
    digits_ = 4 if digits is None else digits

    gen: metafopy.core.rng.RRandom | None = None
    if not use_exact:
        seed_ = int(metafopy.core.rng.get_rng().unif_rand() * 2e9)
        gen = metafopy.core.rng.set_seed(seed_ if seed is None else seed)

    zval_perm, beta_perm, QM_perm = run_permutations(x, Xs, labels, n_iter, use_exact, gen)
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
        QMdf=x.QMdf,
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
            for i in range(n_iter):
                signs = numpy.array([1.0 if not (i >> j) & 1 else -1.0 for j in range(k)])
                store(i, metafopy.methods.common.refit_arrays(x, yi=signs * x.yi, X=one, btt=[0]))
        else:
            assert gen is not None
            i = 0
            while i < n_iter:
                signs = numpy.where(gen.sample_int(2, k, replace=True) == 1, -1.0, 1.0)
                if store(i, metafopy.methods.common.refit_arrays(x, yi=signs * x.yi, X=one, btt=[0])):
                    i += 1
        return zval, beta, QM

    btt = list(x.btt)
    if exact:
        perms = multiset_perms(labels)
        for i in range(n_iter):
            store(i, metafopy.methods.common.refit_arrays(x, X=Xs[numpy.array(perms[i]) - 1], btt=btt))
    else:
        assert gen is not None
        i = 0
        while i < n_iter:
            perm = gen.sample_int(k) - 1
            if store(i, metafopy.methods.common.refit_arrays(x, X=Xs[perm], btt=btt)):
                i += 1
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
        return metafopy.core.zeroin.uniroot(f, interval[0], interval[1], tol=c['tol'], maxiter=c['maxiter']).root
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError:
        return math.nan


def format_permutest(x: PermutestResult) -> str:
    d = x.digits
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
    if x.int_only:
        out.extend(metafopy.methods.vif.format_vector(header, rows[0]))
    else:
        out.append(metafopy.printing.fmt_table(header, rows, row_names=list(x.coef_names)))
    out.append('')
    out.append('---')
    out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    out.append('')
    what = f'p-value{"" if x.int_only else "s"}{" and CI bounds" if x.permci else ""}'
    out.append(f'1) {what} based on permutation testing')
    out.append('')
    return '\n'.join(out)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def pfmt(p: float, digits: int) -> str:
    s = metafopy.printing.fmt_pval(p, digits)
    return f'< {s[1:]}' if s.startswith('<') else f'= {s}'
