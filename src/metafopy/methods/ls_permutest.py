"""Permutation tests for location-scale models (metafor's ``permutest.rma.ls``).

The test statistics of the location part are compared with their distribution under random permutations of the rows of the
location model matrix (random sign flips of the observed effects for an intercept-only location model), those of the scale part
under permutations of the rows of the scale model matrix; an exact test enumerates all unique permutations. See
Viechtbauer & Lopez-Lopez (2022), doi:10.1002/jrsm.1562, and :mod:`metafopy.methods.permutest`.

The permuted fits use the Richardson depth ``r = 8`` of ``numDeriv::hessian`` whatever ``hessian_r`` the fit has, and the
optimiser stops like ``nlminb`` on a likelihood that is flat towards a boundary (see :func:`metafopy.models.rma_ls_opt.minimise`).

Random permutations are drawn with R's own Mersenne-Twister (:mod:`metafopy.core.rng`): each part draws
``int(runif(1) * 2e9)`` from the generator and seeds it with that (or with ``seed``) before permuting, as metafor does.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.core.parallel
import metafopy.core.rng
import metafopy.errors
import metafopy.methods.ls_common
import metafopy.methods.permutest
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['LSPermutestResult', 'permutest_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# Richardson depth of the Hessian in the permuted fits: metafor's numDeriv default. Where a permuted scale fit runs to the boundary, the depth
# decides whether the Hessian is still positive definite (a huge standard error, z near 0) or not (undefined), so the 4 of an ordinary fit
# (``LSControl.hessian_r``) would give 13 undefined standard errors for 50 permutations of the bcg example where R has 3.
HESSIAN_R = 8


@dataclasses.dataclass(frozen=True, slots=True)
class LSPermutestResult:
    """Permutation-based p-values for the location (``pval``) and scale (``pval_alpha``) coefficients of a location-scale model.

    ``QM_perm``, ``zval_perm`` and ``beta_perm`` (``QS_perm``, ``zval_alpha_perm`` and ``alpha_perm``) hold the statistics of
    every permutation of the location (scale) part; for an approximate test the first row is the observed one, as in metafor.
    A part that was skipped (``skip_beta``, ``skip_alpha``) keeps the p-values of the model fit and has no permutations.
    """

    pval: FloatArray
    QMdf: tuple[int, float | None]
    QMp: float
    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    QM: float
    pval_alpha: FloatArray
    QSdf: tuple[int, float | None]
    QSp: float
    alpha: FloatArray
    se_alpha: FloatArray
    zval_alpha: FloatArray
    ci_lb_alpha: FloatArray
    ci_ub_alpha: FloatArray
    QS: float
    k: int
    p: int
    q: int
    btt: tuple[int, ...]
    att: tuple[int, ...]
    m: int
    m_alpha: int
    test: str
    ddf: int | None
    ddf_alpha: int | None
    int_only: bool
    Z_int_only: bool
    coef_names: tuple[str, ...]
    alpha_names: tuple[str, ...]
    exact_iter: float
    Z_exact_iter: float
    alternative: str
    p2defn: str
    stat: str
    skip_beta: bool
    skip_alpha: bool
    QM_perm: FloatArray
    zval_perm: pandas.DataFrame
    beta_perm: pandas.DataFrame
    QS_perm: FloatArray
    zval_alpha_perm: pandas.DataFrame
    alpha_perm: pandas.DataFrame
    digits: int = 4
    int_incl: bool = True
    Z_int_incl: bool = True

    @property
    def dfs(self) -> int | None:
        """R's ``dfs``: the residual degrees of freedom (``ddf``) of the location test."""
        return self.ddf

    @property
    def permci(self) -> bool:
        """R's ``permci``: permutation CIs are not computed for location-scale models."""
        return False

    def to_frame(self) -> dict[str, pandas.DataFrame]:
        """The tables of the location (``'beta'``) and scale (``'alpha'``) coefficients with the permutation p-values (``coef.permutest.rma.uni``)."""
        t_test = self.ddf is not None

        def table(
            names: tuple[str, ...], est: FloatArray, se: FloatArray, z: FloatArray, ddf: int | None, pval: FloatArray, lb: FloatArray, ub: FloatArray
        ) -> pandas.DataFrame:
            cols: dict[str, typing.Any] = {'estimate': est, 'se': se, ('tval' if t_test else 'zval'): z}
            if t_test:
                cols['df'] = numpy.full(len(names), numpy.nan if ddf is None else ddf)
            cols.update({'pval': pval, 'ci.lb': lb, 'ci.ub': ub})
            return pandas.DataFrame(cols, index=list(names))

        return {
            'beta': table(self.coef_names, self.beta, self.se, self.zval, self.ddf, self.pval, self.ci_lb, self.ci_ub),
            'alpha': table(self.alpha_names, self.alpha, self.se_alpha, self.zval_alpha, self.ddf_alpha, self.pval_alpha, self.ci_lb_alpha, self.ci_ub_alpha),
        }

    def coef(self, type: typing.Literal['alpha', 'beta'] | None = None) -> pandas.Series | dict[str, pandas.Series]:
        """The estimated coefficients, indexed by name: ``{'beta': ..., 'alpha': ...}``, or one of them with ``type``."""
        both = {
            'beta': pandas.Series(self.beta, index=list(self.coef_names), name='estimate'),
            'alpha': pandas.Series(self.alpha, index=list(self.alpha_names), name='estimate'),
        }
        return both if type is None else both[type]

    def __str__(self) -> str:
        return format_permutest(self)


def perm_pvalues(obs: FloatArray, perm: FloatArray, con: collections.abc.Mapping[str, typing.Any]) -> FloatArray:
    """Permutation p-values of the statistics ``obs`` given their permutation distributions (the columns of ``perm``)."""
    comptol = con['comptol']
    out = numpy.full(obs.shape[0], numpy.nan)
    for j in range(obs.shape[0]):
        s = perm[:, j]
        valid = ~numpy.isnan(s)
        o = float(obs[j])
        if con['alternative'] == 'two.sided':
            if con['p2defn'] == 'abs':
                out[j] = metafopy.methods.permutest.nanmean(numpy.abs(s) >= abs(o) - comptol, valid)
            elif o > float(numpy.nanmedian(s)):
                out[j] = 2 * metafopy.methods.permutest.nanmean(s >= o - comptol, valid)
            else:
                out[j] = 2 * metafopy.methods.permutest.nanmean(s <= o + comptol, valid)
        elif con['alternative'] == 'less':
            out[j] = metafopy.methods.permutest.nanmean(s <= o + comptol, valid)
        else:
            out[j] = metafopy.methods.permutest.nanmean(s >= o - comptol, valid)
    return numpy.where(out > 1, 1.0, out)


def omnibus_p(perm: FloatArray, observed: float, comptol: float) -> float:
    return metafopy.methods.permutest.nanmean(perm >= observed - comptol, ~numpy.isnan(perm))


def draw_generator(seed: int | None) -> metafopy.core.rng.RRandom:
    """Draw ``int(runif(1) * 2e9)`` from R's generator and seed it with that (or with ``seed``), as the permutation tests do."""
    drawn = int(metafopy.core.rng.get_rng().unif_rand() * 2e9)
    return metafopy.core.rng.set_seed(drawn if seed is None else seed)


def beta_permutations(
    x: metafopy.models.rma_ls_result.RMALSResult,
    n_iter: int,
    exact: bool,
    labels: numpy.typing.NDArray[numpy.intp],
    Xs: FloatArray,
    seed: int | None,
    progbar: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Statistics (``zval``, ``beta``, ``QM``) of the location part for every permutation."""
    k, p = x.k, x.p
    alpha_arg = numpy.where(x.alpha_fix, x.alpha, numpy.nan)
    beta_arg = numpy.where(x.beta_fix, x.beta, numpy.nan) if x.optbeta else None
    # the Hessian is only needed for the location tests when the location coefficients are optimized with the scale coefficients
    skiphes = not x.optbeta
    gen = None if exact else draw_generator(seed)
    n_col = 1 if x.int_only else p
    zval = numpy.full((n_iter, n_col), numpy.nan)
    beta = numpy.full((n_iter, n_col), numpy.nan)
    QM = numpy.full(n_iter, numpy.nan)

    def store(i: int, fit: metafopy.models.rma_ls_result.RMALSResult | None) -> bool:
        if fit is None:
            return False
        beta[i] = fit.beta
        zval[i] = fit.zval
        QM[i] = fit.QM
        return True

    if x.int_only:
        one = numpy.ones((k, 1))

        def fit_signs(signs: FloatArray) -> metafopy.models.rma_ls_result.RMALSResult | None:
            return metafopy.methods.ls_common.refit_ls(x, yi=signs * x.yi, X=one, btt=[0], alpha=alpha_arg, beta=beta_arg, skiphes=True, polish_converged=False)

        if gen is None:
            for i in metafopy.core.parallel.progress(range(n_iter), n_iter, progbar):
                store(i, fit_signs(numpy.array([1.0 if not (i >> j) & 1 else -1.0 for j in range(k)])))
        else:
            i = 0
            while i < n_iter:
                signs = numpy.where(gen.sample_int(2, k, replace=True) == 1, -1.0, 1.0)
                if store(i, fit_signs(signs)):
                    i += 1
                    metafopy.core.parallel.tick(i, n_iter, progbar)
        return zval, beta, QM

    def fit_rows(rows: numpy.typing.NDArray[numpy.intp]) -> metafopy.models.rma_ls_result.RMALSResult | None:
        return metafopy.methods.ls_common.refit_ls(
            x, X=Xs[rows], btt=list(x.btt), alpha=alpha_arg, beta=beta_arg, skiphes=skiphes, hessian_r=HESSIAN_R, polish_converged=False
        )

    if gen is None:
        perms = metafopy.methods.permutest.multiset_perms(labels)
        for i in metafopy.core.parallel.progress(range(n_iter), n_iter, progbar):
            store(i, fit_rows(numpy.array(perms[i]) - 1))
    else:
        i = 0
        while i < n_iter:
            if store(i, fit_rows(gen.sample_int(k) - 1)):
                i += 1
                metafopy.core.parallel.tick(i, n_iter, progbar)
    return zval, beta, QM


def alpha_permutations(
    x: metafopy.models.rma_ls_result.RMALSResult,
    n_iter: int,
    exact: bool,
    labels: numpy.typing.NDArray[numpy.intp],
    Zs: FloatArray,
    seed: int | None,
    progbar: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Statistics (``zval_alpha``, ``alpha``, ``QS``) of the scale part for every permutation."""
    k, q = x.k, x.q
    alpha_arg = numpy.where(x.alpha_fix, x.alpha, numpy.nan)
    beta_arg = numpy.where(x.beta_fix, x.beta, numpy.nan) if x.optbeta else None
    gen = None if exact else draw_generator(seed)
    zval = numpy.full((n_iter, q), numpy.nan)
    alpha = numpy.full((n_iter, q), numpy.nan)
    QS = numpy.full(n_iter, numpy.nan)

    def store(i: int, fit: metafopy.models.rma_ls_result.RMALSResult | None) -> bool:
        if fit is None:
            return False
        alpha[i] = fit.alpha
        zval[i] = fit.zval_alpha
        QS[i] = fit.QS
        return True

    def fit_rows(rows: numpy.typing.NDArray[numpy.intp]) -> metafopy.models.rma_ls_result.RMALSResult | None:
        return metafopy.methods.ls_common.refit_ls(
            x, Z=Zs[rows], att=list(x.att), alpha=alpha_arg, beta=beta_arg, skiphes=False, hessian_r=HESSIAN_R, polish_converged=False
        )

    if gen is None:
        perms = metafopy.methods.permutest.multiset_perms(labels)
        for i in metafopy.core.parallel.progress(range(n_iter), n_iter, progbar):
            store(i, fit_rows(numpy.array(perms[i]) - 1))
    else:
        i = 0
        while i < n_iter:
            if store(i, fit_rows(gen.sample_int(k) - 1)):
                i += 1
                metafopy.core.parallel.tick(i, n_iter, progbar)
    return zval, alpha, QS


def sorted_rows(M: FloatArray) -> tuple[float, numpy.typing.NDArray[numpy.intp], FloatArray]:
    """Number of unique row permutations, the group labels of the sorted rows, and the sorted matrix."""
    count, labels, sorted_M = metafopy.methods.permutest.unique_perms_count(M)
    return float(count), labels, sorted_M


@metafopy.methods.permutest.permutest.register
def permutest_rma_ls(
    x: metafopy.models.rma_ls_result.RMALSResult,
    exact: bool | typing.Literal['i'] = False,
    iter: int = 1000,
    btt: metafopy.models.rma_uni.Btt = None,
    att: metafopy.models.rma_uni.Btt = None,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    *,
    seed: int | None = None,
    tol: float | None = None,
    skip_beta: bool = False,
    skip_alpha: bool = False,
    progbar: bool = False,
) -> LSPermutestResult | list[float]:
    """Permutation tests for the location and scale coefficients of a location-scale fit.

    ``exact=True`` (or enough iterations to enumerate all unique permutations) runs an exact test; ``exact='i'`` returns the
    number of iterations an exact test needs for the location and the scale part. ``btt`` and ``att`` (0-based) change the
    coefficients the omnibus tests refer to; ``skip_beta`` and ``skip_alpha`` skip a part. ``seed`` seeds the generator for each
    approximate part; without it the seeds are drawn from the module's R-compatible generator (see :func:`metafopy.set_seed`).
    ``control`` takes ``comptol``, ``alternative``, ``p2defn`` and ``stat`` as in :func:`metafopy.permutest`. ``progbar=True`` prints a progress bar
    to stderr for each part (metafor's default is ``TRUE``).

    The scale part with ``control={'p2defn': 'px2'}`` uses the observed test statistics; metafor 5.2-1 has a typo there that
    makes these p-values ``NaN``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if iter <= 1:
        raise metafopy.errors.InputError("Argument 'iter' must be >= 2.")
    if x.Z_int_only:
        skip_alpha = True
        warnings.warn('Cannot carry out a permutation test for an intercept-only scale model.', stacklevel=2)
    if skip_beta and skip_alpha:
        raise metafopy.errors.InputError('Must run permutation test for at least one part of the model.')
    con = metafopy.methods.permutest.permutest_control(control)
    if tol is not None:
        con['comptol'] = tol
    if isinstance(exact, str) and exact == 'i':
        skip_beta = skip_alpha = True

    if btt is not None or att is not None:
        refitted = metafopy.methods.ls_common.refit_ls(
            x,
            btt=metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, x.coef_names) if btt is not None else list(x.btt),
            att=metafopy.models.rma_uni.set_btt(att, x.q, x.Z_int_incl, x.alpha_names) if att is not None else list(x.att),
            alpha=numpy.where(x.alpha_fix, x.alpha, numpy.nan),
            beta=numpy.where(x.beta_fix, x.beta, numpy.nan) if x.optbeta else None,
            skiphes=False,
        )
        if refitted is None:
            raise metafopy.errors.InputError('could not refit the model with the requested btt and att')
        x = refitted

    k = x.k
    if x.int_only:
        x_exact_iter: float = 2.0**k
        labels = numpy.arange(1, k + 1)
        Xs = x.X
    else:
        x_exact_iter, labels, Xs = sorted_rows(x.X)
    z_exact_iter, z_labels, Zs = sorted_rows(x.Z)
    if x.Z_int_only:
        z_exact_iter = math.nan
    if isinstance(exact, str) and exact == 'i':
        return [x_exact_iter, z_exact_iter]

    digits_ = metafopy.mfopt.stored_digits(digits, 4)
    comptol = float(con['comptol'])
    n_col = 1 if x.int_only else x.p

    empty = pandas.DataFrame([[math.nan]])
    pval = x.pval
    QMp = x.QMp
    QM_perm = numpy.array([math.nan])
    zval_perm, beta_perm = empty, empty
    if not skip_beta:
        use_exact = bool(exact) or x_exact_iter <= iter
        n_iter = int(x_exact_iter) if use_exact else iter
        if use_exact and math.isinf(x_exact_iter):
            raise metafopy.errors.InputError('Too many iterations required for an exact permutation test of the location model.')
        zv, bv, QM_perm = beta_permutations(x, n_iter, use_exact, labels, Xs, seed, progbar)
        if not use_exact:
            zv[0], bv[0], QM_perm[0] = x.zval[:n_col], x.beta[:n_col], x.QM
        pval = perm_pvalues(x.zval if con['stat'] == 'test' else x.beta, zv if con['stat'] == 'test' else bv, con)
        if x.int_only:
            pval = pval[:1]
        QMp = omnibus_p(QM_perm, x.QM, comptol)
        zval_perm = pandas.DataFrame(zv, columns=list(x.coef_names)[:n_col])
        beta_perm = pandas.DataFrame(bv, columns=list(x.coef_names)[:n_col])

    pval_alpha = x.pval_alpha
    QSp = math.nan
    QS_perm = numpy.array([math.nan])
    zval_alpha_perm, alpha_perm = empty, empty
    if not skip_alpha:
        use_exact = bool(exact) or z_exact_iter <= iter
        n_iter = int(z_exact_iter) if use_exact else iter
        if use_exact and math.isinf(z_exact_iter):
            raise metafopy.errors.InputError('Too many iterations required for an exact permutation test of the scale model.')
        za, av, QS_perm = alpha_permutations(x, n_iter, use_exact, z_labels, Zs, seed, progbar)
        if not use_exact:
            za[0], av[0], QS_perm[0] = x.zval_alpha, x.alpha, x.QS
        pval_alpha = perm_pvalues(x.zval_alpha if con['stat'] == 'test' else x.alpha, za if con['stat'] == 'test' else av, con)
        pval_alpha = numpy.where(x.alpha_fix, math.nan, pval_alpha)
        QSp = omnibus_p(QS_perm, x.QS, comptol)
        zval_alpha_perm = pandas.DataFrame(za, columns=list(x.alpha_names))
        alpha_perm = pandas.DataFrame(av, columns=list(x.alpha_names))

    return LSPermutestResult(
        int_incl=x.int_incl,
        Z_int_incl=x.Z_int_incl,
        pval=pval,
        QMdf=x.QMdf,
        QMp=QMp,
        beta=x.beta,
        se=x.se,
        zval=x.zval,
        ci_lb=x.ci_lb,
        ci_ub=x.ci_ub,
        QM=x.QM,
        pval_alpha=pval_alpha,
        QSdf=x.QSdf,
        QSp=QSp,
        alpha=x.alpha,
        se_alpha=x.se_alpha,
        zval_alpha=x.zval_alpha,
        ci_lb_alpha=x.ci_lb_alpha,
        ci_ub_alpha=x.ci_ub_alpha,
        QS=x.QS,
        k=k,
        p=x.p,
        q=x.q,
        btt=x.btt,
        att=x.att,
        m=x.m,
        m_alpha=x.m_alpha,
        test=x.test,
        ddf=x.ddf,
        ddf_alpha=x.ddf_alpha,
        int_only=x.int_only,
        Z_int_only=x.Z_int_only,
        coef_names=tuple(x.coef_names),
        alpha_names=tuple(x.alpha_names),
        exact_iter=0.0 if skip_beta else x_exact_iter,
        Z_exact_iter=0.0 if skip_alpha else z_exact_iter,
        alternative=con['alternative'],
        p2defn=con['p2defn'],
        stat=con['stat'],
        skip_beta=skip_beta,
        skip_alpha=skip_alpha,
        QM_perm=QM_perm,
        zval_perm=zval_perm,
        beta_perm=beta_perm,
        QS_perm=QS_perm,
        zval_alpha_perm=zval_alpha_perm,
        alpha_perm=alpha_perm,
        digits=digits_,
    )


def part_table(
    est: FloatArray,
    se: FloatArray,
    zval: FloatArray,
    pval: FloatArray,
    ci_lb: FloatArray,
    ci_ub: FloatArray,
    names: tuple[str, ...],
    ddf: int | None,
    permuted: bool,
    single: bool,
    digits: int,
) -> list[str]:
    """The coefficient table of one part; the p-value column carries the footnote symbol when it is based on permutations."""
    fmt = metafopy.printing.fmt_num
    header = ['estimate', 'se', 'tval' if ddf is not None else 'zval']
    if ddf is not None:
        header.append('df')
    header += ['pval¹' if permuted else 'pval', 'ci.lb', 'ci.ub']
    stars = metafopy.methods.ls_common.center_stars(float(v) for v in pval)
    rows: list[list[str]] = []
    for j in range(est.shape[0]):
        row = [fmt(float(est[j]), digits), fmt(float(se[j]), digits), fmt(float(zval[j]), digits)]
        if ddf is not None:
            row.append(f'{ddf:g}')
        # metafor's .addfootsym() also appends a space to the cells of the footnoted column
        row += [metafopy.printing.fmt_pval(float(pval[j]), digits) + (' ' if permuted else ''), fmt(float(ci_lb[j]), digits), fmt(float(ci_ub[j]), digits)]
        rows.append([*row, stars[j]])
    header.append('')
    if single:
        return metafopy.methods.vif.format_vector(header, rows[0])
    return metafopy.printing.fmt_table(header, rows, row_names=list(names)).split('\n')


def format_permutest(x: LSPermutestResult) -> str:
    d = x.digits
    pfmt = metafopy.models.rma_uni.pfmt
    fmt = metafopy.printing.fmt_num
    t_test = x.ddf is not None
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    out: list[str] = ['']
    if not x.int_only:
        out.append(
            f'Test of Location Coefficients (coefficient{"s" if x.m > 1 else ""} {metafopy.models.rma_uni.coef_span(x.btt)}):{"" if x.skip_beta else "¹"}'
        )
        if t_test:
            out.append(f'F(df1 = {x.QMdf[0]}, df2 = {int(x.QMdf[1] or 0)}) = {fmt(x.QM, d)}, p-val {pfmt(x.QMp, d)}')
        else:
            out.append(f'QM(df = {x.QMdf[0]}) = {fmt(x.QM, d)}, p-val {pfmt(x.QMp, d)}')
        out.append('')
    out += ['Model Results (Location):', '']
    out += part_table(x.beta, x.se, x.zval, x.pval, x.ci_lb, x.ci_ub, x.coef_names, x.ddf, not x.skip_beta, x.int_only, d)
    out.append('')
    if not x.Z_int_only:
        out.append(
            f'Test of Scale Coefficients (coefficient{"s" if x.m_alpha > 1 else ""} {metafopy.models.rma_uni.coef_span(x.att)}):{"" if x.skip_alpha else "¹"}'
        )
        if t_test:
            out.append(f'F(df1 = {x.QSdf[0]}, df2 = {int(x.QSdf[1] or 0)}) = {fmt(x.QS, d)}, p-val {pfmt(x.QSp, d)}')
        else:
            out.append(f'QS(df = {x.QSdf[0]}) = {fmt(x.QS, d)}, p-val {pfmt(x.QSp, d)}')
        out.append('')
    out += ['Model Results (Scale):', '']
    out += part_table(
        x.alpha, x.se_alpha, x.zval_alpha, x.pval_alpha, x.ci_lb_alpha, x.ci_ub_alpha, x.alpha_names, x.ddf_alpha, not x.skip_alpha, x.Z_int_only, d
    )
    out += ['', '---', "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", '', '1) p-values based on permutation testing', '']
    return '\n'.join(out)
