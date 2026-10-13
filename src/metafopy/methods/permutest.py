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

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from metafopy._print import fmt_num, fmt_pval, fmt_table, signif_stars
from metafopy.core.rng import RRandom, get_rng, set_seed
from metafopy.core.zeroin import uniroot
from metafopy.methods._common import refit_arrays
from metafopy.methods.vif import _format_vector
from metafopy.models.rma_uni import Btt, RMAResult, coef_span, set_btt

__all__ = ['PermutestResult', 'permutest']

type FloatArray = NDArray[np.float64]

_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
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
    zval_perm: pd.DataFrame
    beta_perm: pd.DataFrame
    digits: int = 4

    def to_frame(self) -> pd.DataFrame:
        cols: dict[str, Any] = {'estimate': self.beta, 'se': self.se, ('tval' if self.ddf is not None else 'zval'): self.zval}
        if self.ddf is not None:
            cols['df'] = np.full(self.p, self.ddf)
        cols.update({'pval': self.pval, 'ci.lb': self.ci_lb, 'ci.ub': self.ci_ub})
        return pd.DataFrame(cols, index=list(self.coef_names))

    def __str__(self) -> str:
        return _format(self)


@singledispatch
def permutest(x: Any, *args: Any, **kwargs: Any) -> PermutestResult | int | float:
    """Permutation test for a fitted model; see :func:`_permutest_rma_uni`."""
    raise TypeError(f'permutest() not implemented for {type(x).__name__}')


_CONTROL_DEFAULTS: dict[str, Any] = {
    'comptol': _EPS**0.5,
    'tol': _EPS**0.25,
    'maxiter': 100,
    'alternative': 'two.sided',
    'p2defn': 'abs',
    'stat': 'test',
    'cialt': 'one.sided',
    'distfac': 1.0,
    'extendInt': 'no',
}


def _unique_perms_count(X: FloatArray) -> tuple[int | float, NDArray[np.intp], FloatArray]:
    """Number of distinct row permutations of ``X``, the group labels of its sorted rows, and the sorted ``X``."""
    k = X.shape[0]
    order = np.lexsort(tuple(X[:, j] for j in range(X.shape[1] - 1, -1, -1)))
    Xs = X[order]
    new = np.ones(k, dtype=np.bool_)
    new[1:] = np.any(Xs[1:] != Xs[:-1], axis=1) if k > 1 else True
    # R: X.indices is the position of the first row of each run of identical rows
    # (duplicated() flags any earlier identical row, runs are contiguous after sorting)
    starts = np.flatnonzero(new)
    run_len = np.diff(np.append(starts, k))
    labels = np.repeat(starts + 1, run_len)
    counts = np.array(run_len)
    big = int(np.argmax(counts))
    rest = np.delete(counts, big)
    num = math.prod(range(int(counts.max()) + 1, k + 1))
    den = math.prod(math.factorial(int(c)) for c in rest)
    total: int | float = num // den
    if total > 1.7e308:
        total = math.inf
    return total, labels, Xs


def _multiset_perms(labels: NDArray[np.intp]) -> list[tuple[int, ...]]:
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


def _control(control: Mapping[str, Any] | None) -> dict[str, Any]:
    con = dict(_CONTROL_DEFAULTS)
    for key, value in (control or {}).items():
        match = [name for name in con if name.startswith(key.replace('.', '_').replace('extend_int', 'extendInt'))]
        if key in con:
            match = [key]
        if len(match) != 1:
            raise TypeError(f'unknown control argument {key!r}')
        con[match[0]] = value
    if con['alternative'] not in ('two.sided', 'less', 'greater'):
        raise ValueError("control 'alternative' must be 'two.sided', 'less' or 'greater'")
    if con['p2defn'] not in ('abs', 'px2'):
        raise ValueError("control 'p2defn' must be 'abs' or 'px2'")
    if con['stat'] not in ('test', 'coef'):
        raise ValueError("control 'stat' must be 'test' or 'coef'")
    if con['extendInt'] != 'no':
        raise NotImplementedError("control extendInt other than 'no' is not supported")
    return con


def _nanmean(a: NDArray[np.bool_] | FloatArray, valid: NDArray[np.bool_]) -> float:
    n = int(valid.sum())
    return float(np.sum(np.asarray(a, dtype=np.float64)[valid])) / n if n else math.nan


@permutest.register
def _permutest_rma_uni(
    x: RMAResult,
    exact: bool | Literal['i'] = False,
    iter: int = 1000,
    btt: Btt = None,
    permci: bool | Sequence[int] = False,
    digits: int | None = None,
    control: Mapping[str, Any] | None = None,
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
    """
    iter = iter
    if iter <= 1:
        raise ValueError("Argument 'iter' must be >= 2.")
    if btt is not None:
        bt = set_btt(btt, x.p, x.int_incl, x.coef_names) if not (fixed and isinstance(btt, str)) else _fixed_btt(btt, x)
        refitted = refit_arrays(x, btt=list(bt))
        if refitted is None:
            raise ValueError('could not refit the model with the requested btt')
        x = refitted
    k, p = x.k, x.p

    if x.int_only:
        exact_iter: int | float = 2**k
        labels = np.arange(1, k + 1)
        Xs = x.X
    else:
        exact_iter, labels, Xs = _unique_perms_count(x.X)
    if isinstance(exact, str) and exact == 'i':
        return exact_iter
    use_exact = bool(exact) or exact_iter <= iter
    n_iter = int(exact_iter) if use_exact else iter
    if use_exact and math.isinf(exact_iter):
        raise ValueError('Too many iterations required for an exact permutation test.')

    con = _control(control)
    if tol is not None:
        con['comptol'] = tol
    digits_ = 4 if digits is None else digits

    gen: RRandom | None = None
    if not use_exact:
        seed_ = int(get_rng().unif_rand() * 2e9)
        gen = set_seed(seed_ if seed is None else seed)

    zval_perm, beta_perm, QM_perm = _run_permutations(x, Xs, labels, n_iter, use_exact, gen)
    if not use_exact:
        beta_perm[0] = x.beta
        zval_perm[0] = x.zval
        QM_perm[0] = x.QM

    pval = _pvalues(x, zval_perm, beta_perm, con)
    QMp = _nanmean(QM_perm >= x.QM - con['comptol'], ~np.isnan(QM_perm))

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
                    raise ValueError("Non-existent coefficients specified via 'permci'.")
            ci_lb = np.full(p, np.nan)
            ci_ub = np.full(p, np.nan)
            for j in coefs:
                ci_lb[j] = _permci_bound(x, j, use_exact, n_iter, alpha, con, lower=True, base_ci=x.ci_lb[j], base_ub=x.ci_ub[j])
                ci_ub[j] = _permci_bound(x, j, use_exact, n_iter, alpha, con, lower=False, base_ci=x.ci_lb[j], base_ub=x.ci_ub[j])

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
        alternative=con['alternative'] if not did_permci else _final_alt(con),
        p2defn=con['p2defn'],
        stat=con['stat'],
        QM_perm=QM_perm,
        zval_perm=pd.DataFrame(zval_perm, columns=list(x.coef_names)),
        beta_perm=pd.DataFrame(beta_perm, columns=list(x.coef_names)),
        digits=digits_,
    )


def _final_alt(con: Mapping[str, Any]) -> str:
    # metafor overwrites con$alternative while searching for the CI bounds; the last search is for the upper bound
    return 'less' if con['cialt'] == 'one.sided' else 'two.sided'


def _fixed_btt(btt: str, x: RMAResult) -> tuple[int, ...]:
    idx = tuple(i for i, n in enumerate(x.coef_names) if btt in n)
    if not idx:
        raise ValueError("Cannot identify coefficient(s) corresponding to the specified 'btt' string.")
    return idx


def _run_permutations(
    x: RMAResult,
    Xs: FloatArray,
    labels: NDArray[np.intp],
    n_iter: int,
    exact: bool,
    gen: RRandom | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    k, p = x.k, x.p
    zval = np.full((n_iter, p), np.nan)
    beta = np.full((n_iter, p), np.nan)
    QM = np.full(n_iter, np.nan)

    def store(i: int, fit: RMAResult | None) -> bool:
        if fit is None:
            return False
        beta[i] = fit.beta
        zval[i] = fit.zval
        QM[i] = fit.QM
        return True

    if x.int_only:
        one = np.ones((k, 1))
        if exact:
            for i in range(n_iter):
                signs = np.array([1.0 if not (i >> j) & 1 else -1.0 for j in range(k)])
                store(i, refit_arrays(x, yi=signs * x.yi, X=one, btt=[0]))
        else:
            assert gen is not None
            i = 0
            while i < n_iter:
                signs = np.where(gen.sample_int(2, k, replace=True) == 1, -1.0, 1.0)
                if store(i, refit_arrays(x, yi=signs * x.yi, X=one, btt=[0])):
                    i += 1
        return zval, beta, QM

    btt = list(x.btt)
    if exact:
        perms = _multiset_perms(labels)
        for i in range(n_iter):
            store(i, refit_arrays(x, X=Xs[np.array(perms[i]) - 1], btt=btt))
    else:
        assert gen is not None
        i = 0
        while i < n_iter:
            perm = gen.sample_int(k) - 1
            if store(i, refit_arrays(x, X=Xs[perm], btt=btt)):
                i += 1
    return zval, beta, QM


def _pvalues(x: RMAResult, zval_perm: FloatArray, beta_perm: FloatArray, con: Mapping[str, Any]) -> FloatArray:
    comptol = con['comptol']
    use_test = con['stat'] == 'test'
    stat_perm = zval_perm if use_test else beta_perm
    stat_obs = x.zval if use_test else x.beta
    pval = np.full(x.p, np.nan)
    for j in range(x.p):
        s = stat_perm[:, j]
        valid = ~np.isnan(s)
        obs = stat_obs[j]
        if con['alternative'] == 'two.sided':
            if con['p2defn'] == 'abs':
                pval[j] = _nanmean(np.abs(s) >= abs(obs) - comptol, valid)
            elif obs > float(np.nanmedian(s)):
                pval[j] = 2 * _nanmean(s >= obs - comptol, valid)
            else:
                pval[j] = 2 * _nanmean(s <= obs + comptol, valid)
        elif con['alternative'] == 'less':
            pval[j] = _nanmean(s <= obs + comptol, valid)
        else:
            pval[j] = _nanmean(s >= obs - comptol, valid)
    return np.where(pval > 1, 1.0, pval)


def _permci_bound(
    x: RMAResult,
    j: int,
    exact: bool,
    n_iter: int,
    alpha: float,
    con: Mapping[str, Any],
    *,
    lower: bool,
    base_ci: float,
    base_ub: float,
) -> float:
    """Search for a permutation-based CI bound of coefficient ``j`` (metafor's ``.permci`` with ``uniroot``)."""
    c = dict(con)
    one_sided = c['cialt'] == 'one.sided'
    if lower:
        c['alternative'] = 'greater' if one_sided else 'two.sided'
        interval = (base_ci - c['distfac'] * (x.beta[j] - base_ci), float(x.beta[j]))
    else:
        c['alternative'] = 'less' if one_sided else 'two.sided'
        interval = (float(x.beta[j]), base_ub + c['distfac'] * (base_ub - x.beta[j]))

    def f(val: float) -> float:
        fit = refit_arrays(x, yi=x.yi - val * x.X[:, j])
        if fit is None:
            raise ValueError('refit failed')
        res = _permutest_rma_uni(fit, exact=exact, iter=n_iter, control=c)
        assert isinstance(res, PermutestResult)
        diff = float(res.pval[j]) - alpha / (1 if c['alternative'] == 'two.sided' else 2)
        return diff * 10 if diff < 0 else diff

    try:
        return uniroot(f, interval[0], interval[1], tol=c['tol'], maxiter=c['maxiter']).root
    except ValueError, ArithmeticError, np.linalg.LinAlgError:
        return math.nan


def _format(x: PermutestResult) -> str:
    d = x.digits
    out: list[str] = ['']
    if not x.int_only:
        out.append(f'Test of Moderators (coefficient{"s" if x.m > 1 else ""} {coef_span(x.btt)}):¹')
        if x.ddf is not None:
            out.append(f'F(df1 = {x.QMdf[0]}, df2 = {x.QMdf[1]}) = {fmt_num(x.QM, d)}, p-val {_pfmt(x.QMp, d)}')
        else:
            out.append(f'QM(df = {x.QMdf[0]}) = {fmt_num(x.QM, d)}, p-val {_pfmt(x.QMp, d)}')
        out.append('')
    out.append('Model Results:')
    out.append('')
    header = ['estimate', 'se', 'tval' if x.ddf is not None else 'zval']
    if x.ddf is not None:
        header.append('df')
    header += ['pval¹', *(['ci.lb¹', 'ci.ub¹'] if x.permci else ['ci.lb', 'ci.ub']), '']
    rows: list[list[str]] = []
    for j in range(x.p):
        row = [fmt_num(x.beta[j], d), fmt_num(x.se[j], d), fmt_num(x.zval[j], d)]
        if x.ddf is not None:
            row.append(f'{x.ddf:g}')
        # metafor's .addfootsym() also appends a space to the cells of the footnoted columns
        pad = ' ' if x.permci else ''
        row += [fmt_pval(x.pval[j], d) + ' ', fmt_num(x.ci_lb[j], d) + pad, fmt_num(x.ci_ub[j], d) + pad, signif_stars(x.pval[j])]
        rows.append(row)
    if x.int_only:
        out.extend(_format_vector(header, rows[0]))
    else:
        out.append(fmt_table(header, rows, row_names=list(x.coef_names)))
    out.append('')
    out.append('---')
    out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    out.append('')
    what = f'p-value{"" if x.int_only else "s"}{" and CI bounds" if x.permci else ""}'
    out.append(f'1) {what} based on permutation testing')
    out.append('')
    return '\n'.join(out)


def _pfmt(p: float, digits: int) -> str:
    s = fmt_pval(p, digits)
    return f'< {s[1:]}' if s.startswith('<') else f'= {s}'
