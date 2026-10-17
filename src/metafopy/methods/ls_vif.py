"""Variance inflation factors for location-scale models (metafor's ``vif.rma`` for ``rma.ls`` objects).

The (generalised) variance inflation factors of the location coefficients and, when the scale model has predictors, of the
scale coefficients (Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1562; Fox & Monette, 1992,
doi:10.1080/01621459.1992.10475190), computed as for :func:`metafopy.vif`. With ``sim`` the (G)VIFs are also computed after permuting
the columns of the location (scale) model matrix, with R's generator (:class:`metafopy.core.rng.RRandom`), as metafor's ``.compvifsim`` does.
See https://github.com/wviechtb/metafor/blob/master/R/vif.rma.r and https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.vif.r.
"""

import collections.abc
import concurrent.futures
import dataclasses
import functools
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.linalg
import metafopy.core.parallel
import metafopy.core.rng
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.ls_common
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni

__all__ = ['LSVIFResult', 'vif_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LSVIFResult:
    """(G)VIFs of the location (``beta``) and scale (``alpha``) coefficients of a location-scale model."""

    beta: metafopy.methods.vif.VIFResult
    alpha: metafopy.methods.vif.VIFResult

    def __str__(self) -> str:
        return f'\nLocation Coefficients:\n{self.beta}\nScale Coefficients:\n{self.alpha}'


def coef_table(
    estimate: FloatArray,
    se: FloatArray,
    zval: FloatArray,
    pval: FloatArray,
    ci_lb: FloatArray,
    ci_ub: FloatArray,
    names: collections.abc.Sequence[str],
    ddf: int | None,
) -> pandas.DataFrame:
    cols: dict[str, typing.Any] = {'estimate': estimate, 'se': se}
    if ddf is not None:
        cols['tval'] = zval
        cols['df'] = numpy.full(len(names), ddf)
    else:
        cols['zval'] = zval
    cols.update({'pval': pval, 'ci.lb': ci_lb, 'ci.ub': ci_ub})
    return pandas.DataFrame(cols, index=list(names))


def compvif_ls(
    group: collections.abc.Sequence[int],
    vcov: FloatArray,
    int_incl: bool,
    refit: collections.abc.Callable[[FloatArray], FloatArray | None] | None,
    X: FloatArray,
    intercept: bool = False,
) -> tuple[float, list[int]]:
    """GVIF of the coefficients ``group`` (0-based) of one part of the model and the indices actually used.

    Without ``refit`` the GVIF comes from the correlation matrix of the estimates; with ``refit`` (a function returning the
    variance-covariance matrix of the part after refitting the model with a given model matrix) the model is refitted with
    only those coefficients (plus the intercept) and the variance-covariance blocks are compared.
    """
    return metafopy.methods.vif.compvif_matrix(group, vcov, int_incl, refit, X, intercept)


def part_vif(
    res: metafopy.models.rma_ls_result.RMALSResult,
    *,
    which: str,
    spec_arg: metafopy.methods.vif.BttArg,
    table: bool,
    reestimate: bool,
    intercept: bool = False,
    fixed: bool = False,
    join: tuple[typing.Any, ...] | None = None,
    sim: int = 0,
    seed: int | None = None,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    progbar: bool = False,
) -> metafopy.methods.vif.VIFResult:
    """The (G)VIFs of the location (``which='beta'``) or the scale (``which='alpha'``) coefficients.

    With ``sim >= 2`` the model is refitted ``sim`` times with independently permuted columns of the model matrix of this part.

    Raises:
        metafopy.errors.InputError: a coefficient set is invalid.
    """
    if which == 'beta':
        n, int_incl, names, vcov, M = res.p, res.int_incl, res.coef_names, res.vb, res.X
    else:
        n, int_incl, names, vcov, M = res.q, res.Z_int_incl, res.alpha_names, res.va, res.Z
    arg = 'btt' if which == 'beta' else 'att'
    missing = spec_arg is None
    specs: list[metafopy.methods.vif.BttSpec]
    if spec_arg is None:
        specs = [j for j in range(1 if int_incl and not intercept else 0, n)]
    elif isinstance(spec_arg, str | int):
        specs = [spec_arg]
    elif any(isinstance(b, str | collections.abc.Sequence) for b in spec_arg):
        specs = list(spec_arg)
    else:
        specs = [list(spec_arg)]  # type: ignore[arg-type]
    groups = [metafopy.models.rma_uni.set_btt(s, n, int_incl, names, fixed) for s in specs]
    if int_incl and not intercept and any(g == (0,) for g in groups):
        raise metafopy.errors.InputError(f"Cannot compute VIF(s) for the specified '{arg}' argument.")

    refit = make_refit(res, which) if reestimate else None

    entries: list[metafopy.methods.vif.VIFEntry] = []
    for spec, g in zip(specs, groups, strict=True):
        gvif, used = compvif_ls(g, vcov, int_incl, refit, M, intercept)
        m = len(used)
        entries.append(
            metafopy.methods.vif.VIFEntry(
                spec=metafopy.methods.vif.format_spec(spec),
                coefs=metafopy.methods.vif.format_btt(i + 1 for i in used),
                coefname=names[used[0]] if m == 1 else '',
                m=m,
                vif=gvif,
                sif=gvif ** (1 / (2 * m)),
            )
        )
    tab: pandas.DataFrame | None = None
    if table and missing:
        if which == 'beta':
            tab = coef_table(res.beta, res.se, res.zval, res.pval, res.ci_lb, res.ci_ub, res.coef_names, res.ddf)
        else:
            tab = coef_table(res.alpha, res.se_alpha, res.zval_alpha, res.pval_alpha, res.ci_lb_alpha, res.ci_ub_alpha, res.alpha_names, res.ddf_alpha)
    sims: FloatArray | None = None
    prop: FloatArray | None = None
    vifs = numpy.array([e.vif for e in entries])
    # simulating is pointless when every (G)VIF is already 1
    if sim >= 2 and not numpy.all(vifs[~numpy.isnan(vifs)] == 1):
        seeds = [seed + i for i in range(1, sim + 1)] if seed is not None else [int(metafopy.core.rng.get_rng().unif_index(2**31 - 1)) for _ in range(sim)]
        one = functools.partial(sim_one, res, which, tuple(groups), reestimate, intercept, join)
        rows = metafopy.core.parallel.map_sims(one, seeds, parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
        sims = numpy.vstack(rows)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            ok = ~numpy.isnan(sims)
            hits = (vifs[None, :] >= sims) & ok
            prop = hits.sum(axis=0) / ok.sum(axis=0)
    return metafopy.methods.vif.VIFResult(
        vif=tuple(entries),
        bttspec=not missing,
        table=tab,
        test=res.test if tab is not None else None,
        sim=sims,
        sim_type='X' if sims is not None else None,
        prop=prop,
    )


def make_refit(fit: metafopy.models.rma_ls_result.RMALSResult, which: str) -> collections.abc.Callable[[FloatArray], FloatArray | None]:
    """A function that refits ``fit`` with a replacement model matrix for one part and returns the variance-covariance matrix of that part."""
    mask = numpy.where(fit.alpha_fix, fit.alpha, numpy.nan)

    def refit_part(Mb: FloatArray) -> FloatArray | None:
        beta_arg = numpy.where(fit.beta_fix, fit.beta, numpy.nan) if fit.optbeta and which == 'alpha' else None
        refitted = metafopy.methods.ls_common.refit_ls(
            fit,
            alpha=mask if which == 'beta' else None,
            beta=beta_arg,
            skiphes=False,
            X=Mb if which == 'beta' else None,
            Z=Mb if which == 'alpha' else None,
        )
        if refitted is None:
            return None
        return refitted.vb if which == 'beta' else refitted.va

    return refit_part


def sim_one(
    res: metafopy.models.rma_ls_result.RMALSResult,
    which: str,
    groups: tuple[tuple[int, ...], ...],
    reestimate: bool,
    intercept: bool,
    join: tuple[typing.Any, ...] | None,
    seed: int,
) -> FloatArray:
    """One iteration of metafor's ``.compvifsim``: (G)VIFs after permuting every column of the model matrix of one part.

    The generator is R's, seeded with ``seed`` (metafor calls ``set.seed(seed + l)`` for iteration ``l``), and each column is shuffled
    with ``sample()`` in turn, so the draws are those of metafor when the model matrix was given as a matrix.
    """
    rng = metafopy.core.rng.RRandom(seed)
    M = res.X if which == 'beta' else res.Z
    perm = metafopy.methods.vif.permute_joined(M, rng, join) if join is not None else metafopy.methods.vif.permute_design(None, None, M, rng)
    drop = metafopy.core.linalg.drop_redundant(perm)
    if drop.any():
        # a set containing a coefficient that became redundant is infinitely inflated
        return numpy.array([math.inf if drop[list(g)].any() else math.nan for g in groups])
    alpha = numpy.where(res.alpha_fix, res.alpha, numpy.nan)
    beta = numpy.where(res.beta_fix, res.beta, numpy.nan) if res.optbeta else None
    fit = metafopy.methods.ls_common.refit_ls(
        res, alpha=alpha, beta=beta, skiphes=False, X=perm if which == 'beta' else None, Z=perm if which == 'alpha' else None
    )
    if fit is None:
        return numpy.full(len(groups), math.nan)
    vcov = fit.vb if which == 'beta' else fit.va
    int_incl = res.int_incl if which == 'beta' else res.Z_int_incl
    refit = make_refit(fit, which) if reestimate else None
    try:
        with numpy.errstate(invalid='ignore'):
            return numpy.array([compvif_ls(g, vcov, int_incl, refit, perm, intercept)[0] for g in groups])
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError:
        return numpy.full(len(groups), math.nan)


@metafopy.methods.vif.vif.register  # type: ignore[arg-type]  # returns a result class of its own
@metafopy.core.timing.timed
def vif_rma_ls(
    res: metafopy.models.rma_ls_result.RMALSResult,
    btt: metafopy.methods.vif.BttArg = None,
    att: metafopy.methods.vif.BttArg = None,
    table: bool = False,
    reestimate: bool = False,
    sim: bool | int = False,
    progbar: bool = True,
    seed: int | None = None,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    digits: int | None = None,
    intercept: bool = False,
    fixed: bool = False,
    joinb: collections.abc.Sequence[typing.Any] | None = None,
    joina: collections.abc.Sequence[typing.Any] | None = None,
    LB: bool = False,
) -> metafopy.methods.vif.VIFResult | LSVIFResult:
    """(G)VIFs for the location and, if the scale model has predictors, the scale coefficients of a location-scale model.

    ``btt`` and ``att`` (0-based) select sets of location and scale coefficients as ``btt`` does in :func:`metafopy.vif`;
    ``table=True`` adds the coefficient table; ``reestimate=True`` refits the model with only the coefficients in each set.
    A single :class:`metafopy.VIFResult` is returned when only one of the two parts applies, otherwise an
    :class:`LSVIFResult` with both.

    ``sim=True`` (1000 iterations) or ``sim=n`` refits the model ``n`` times with independently permuted columns of the model matrix
    (of the location and, separately, of the scale part), giving ``sim`` and ``prop`` in the results. Iteration ``l`` shuffles the
    columns with R's generator seeded by ``seed + l`` (as metafor), so the draws are reproducible and do not depend on ``parallel``;
    without ``seed`` the seeds are drawn from :func:`metafopy.core.rng.get_rng`. Unlike metafor, a model given by formulas has its model
    matrix columns permuted (``sim_type == 'X'``), not the underlying data variables. ``parallel`` (``'snow'``/``'multicore'``) runs the
    iterations in a process pool of ``ncpus`` workers, or on the :class:`concurrent.futures.Executor` given as ``cl``.

    ``intercept=True`` includes the intercepts and ``fixed=True`` matches string ``btt``/``att`` literally (metafor's ``...`` arguments).
    ``joinb`` and ``joina`` give one label per column of the location and scale model matrix; with ``sim`` the columns that share a label are
    permuted together. ``LB`` is accepted as in metafor and changes nothing.

    Raises:
        metafopy.errors.InputError: an argument is invalid or the model has no predictors.
    """
    loc = not res.int_only
    scale = not res.Z_int_only
    if not loc and not scale:
        raise metafopy.errors.InputError('VIFs not applicable to intercept-only models.')
    if isinstance(sim, bool):
        n_sim = 1000 if sim else 0
    else:
        n_sim = round(sim)
        if n_sim <= 1:
            raise metafopy.errors.InputError("Argument 'sim' must be >= 2.")
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    joins = {'beta': metafopy.methods.vif.join_labels(joinb), 'alpha': metafopy.methods.vif.join_labels(joina)}
    sim_args: dict[str, typing.Any] = {'sim': n_sim, 'seed': seed, 'parallel': parallel, 'ncpus': ncpus, 'cl': cl, 'progbar': progbar}
    out_beta = (
        part_vif(res, which='beta', spec_arg=btt, table=table, reestimate=reestimate, intercept=intercept, fixed=fixed, join=joins['beta'], **sim_args)
        if loc
        else None
    )
    out_alpha = (
        part_vif(res, which='alpha', spec_arg=att, table=table, reestimate=reestimate, intercept=intercept, fixed=fixed, join=joins['alpha'], **sim_args)
        if scale
        else None
    )
    result: metafopy.methods.vif.VIFResult | LSVIFResult
    if out_beta is not None and out_alpha is not None:
        result = LSVIFResult(beta=out_beta, alpha=out_alpha)
    else:
        single = out_beta if out_beta is not None else out_alpha
        if single is None:
            raise metafopy.errors.InternalError('no VIFs were computed')
        result = single
    if digits is not None:
        result = apply_digits(result, digits)
    return result


def apply_digits(result: metafopy.methods.vif.VIFResult | LSVIFResult, digits: int) -> metafopy.methods.vif.VIFResult | LSVIFResult:
    digits = metafopy.mfopt.reduce_digits(digits)
    if isinstance(result, LSVIFResult):
        return LSVIFResult(beta=dataclasses.replace(result.beta, digits=digits), alpha=dataclasses.replace(result.alpha, digits=digits))
    return dataclasses.replace(result, digits=digits)
