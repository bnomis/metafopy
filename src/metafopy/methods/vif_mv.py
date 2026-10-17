"""Variance inflation factors for ``rma.mv`` and ``selmodel`` fits with moderators (metafor's ``vif.rma``).

The (generalised) variance inflation factor of a set of coefficients is computed as for :func:`metafopy.vif` from the
variance-covariance matrix of the coefficients (Fox & Monette, 1992, doi:10.1080/01621459.1992.10475190). For ``rma.mv`` fits
``reestimate=True`` refits the model with only the coefficients in each set and ``sim`` compares the (G)VIFs with those obtained
after permuting the moderators; for ``selmodel`` fits neither is available, as in metafor. See
https://github.com/wviechtb/metafor/blob/master/R/vif.rma.r and https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.vif.r.
"""

import collections.abc
import concurrent.futures
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
import metafopy.errors
import metafopy.methods.mv_diagnostics
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_mv
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['vif_rma_mv', 'vif_selmodel']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type MV = metafopy.models.mv_result.RMAMVResult
type Refit = collections.abc.Callable[[FloatArray], FloatArray | None]

# the arguments of the call that the refits of metafor's ``.compvif`` do not pass on
NOT_PASSED = (*metafopy.methods.mv_diagnostics.DROPPED, 'optbeta')


def refit_mods(x: MV, X: FloatArray) -> MV | None:
    """``x`` refitted with the model matrix ``X`` and no intercept added (metafor's ``rma.mv(..., mods=X, intercept=FALSE)`` of ``.compvif``).

    The estimated variance components are re-estimated, the fixed ones kept; ``None`` if the fit fails.
    """
    spec = dict(x.spec)
    for name in NOT_PASSED:
        spec.pop(name, None)
    used = numpy.array(x.ids, dtype=numpy.intp)[x.not_na]
    data = spec.get('data')
    spec['data'] = data.reset_index(drop=True).iloc[used].reset_index(drop=True) if isinstance(data, pandas.DataFrame) else None
    spec.update(
        yi=x.yi,
        V=metafopy.methods.mv_diagnostics.dense(x.V),
        W=None if spec.get('W') is None or x.W is None else metafopy.methods.mv_diagnostics.dense(x.W),
        mods=X,
        intercept=False,
        subset=None,
        slab=None,
    )
    current = metafopy.methods.mv_diagnostics.vc_values(x)
    for name in metafopy.models.mv_structs.VC_NAMES:
        flags = x.vc_fix[name]
        value = current[name]
        if flags is not None and value is not None:
            spec[name] = numpy.where(flags, value, numpy.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return metafopy.models.rma_mv.rma_mv(**spec)
    except metafopy.methods.mv_diagnostics.REFIT_ERRORS:
        return None


def make_refit(fit: MV) -> Refit:
    """A function that refits ``fit`` with a replacement model matrix and returns the variance-covariance matrix of the coefficients."""

    def refit_with(Xb: FloatArray) -> FloatArray | None:
        out = refit_mods(fit, Xb)
        return None if out is None else out.vb

    return refit_with


def sim_one(
    x: MV,
    groups: tuple[tuple[int, ...], ...],
    reestimate: bool,
    intercept: bool,
    formula: str | None,
    rows: pandas.DataFrame | None,
    join: tuple[typing.Any, ...] | None,
    seed: int | None,
) -> FloatArray:
    """One iteration of metafor's ``.compvifsim``: (G)VIFs after permuting the moderators (``seed=None`` continues the global stream)."""
    rng = metafopy.core.rng.get_rng() if seed is None else metafopy.core.rng.RRandom(seed)
    perm = metafopy.methods.vif.permute_design(formula, rows, x.X, rng, join)
    fit = refit_mods(x, perm)
    if fit is None:
        return numpy.full(len(groups), math.nan)
    if any(fit.coef_na):
        return numpy.array([math.inf if any(fit.coef_na[i] for i in g) else math.nan for g in groups])
    refit = make_refit(fit) if reestimate else None
    try:
        return numpy.array([metafopy.methods.vif.compvif_matrix(g, fit.vb, x.int_incl, refit, perm, intercept)[0] for g in groups])
    except ValueError, ArithmeticError, numpy.linalg.LinAlgError:
        return numpy.full(len(groups), math.nan)


def coef_table(res: MV | metafopy.models.sel_result.SelmodelResult) -> pandas.DataFrame:
    """metafor's ``coef(summary(x))`` (``type="beta"``)."""
    cols: dict[str, typing.Any] = {'estimate': res.beta, 'se': res.se}
    if res.test != 'z':
        cols['tval'] = res.zval
        cols['df'] = numpy.broadcast_to(numpy.asarray(res.ddf, dtype=numpy.float64), (res.p,)).copy()
    else:
        cols['zval'] = res.zval
    cols.update({'pval': res.pval, 'ci.lb': res.ci_lb, 'ci.ub': res.ci_ub})
    return pandas.DataFrame(cols, index=list(res.coef_names))


def check_arguments(res: MV | metafopy.models.sel_result.SelmodelResult, att: typing.Any, sim: bool | int) -> int:
    """metafor's argument checks; returns the number of simulations."""
    if res.int_only:
        raise metafopy.errors.InputError('VIFs not applicable to intercept-only models.')
    if isinstance(sim, bool):
        n_sim = 1000 if sim else 0
    else:
        n_sim = round(sim)
        if n_sim <= 1:
            raise metafopy.errors.InputError("Argument 'sim' must be >= 2.")
    if att is not None:
        raise metafopy.errors.InputError("Argument 'att' only relevant for location-scale models.")
    return n_sim


def vif_coefficients(
    res: MV | metafopy.models.sel_result.SelmodelResult,
    btt: metafopy.methods.vif.BttArg,
    table: bool,
    refit: Refit | None,
    intercept: bool,
    fixed: bool,
    digits: int | None,
) -> tuple[metafopy.methods.vif.VIFResult, list[tuple[int, ...]]]:
    """The (G)VIFs of the sets of coefficients in ``btt`` (without the simulations) and the 0-based sets."""
    int_incl, p, names = res.int_incl, res.p, res.coef_names
    btt_missing = btt is None
    specs: list[metafopy.methods.vif.BttSpec]
    if btt is None:
        specs = [j for j in range(1 if int_incl and not intercept else 0, p)]
    elif isinstance(btt, str | int):
        specs = [btt]
    elif any(isinstance(b, str | collections.abc.Sequence) for b in btt):
        specs = list(btt)
    else:
        specs = [list(btt)]  # type: ignore[arg-type]
    groups = [metafopy.models.rma_uni.set_btt(s, p, int_incl, names, fixed) for s in specs]
    if int_incl and not intercept and any(g == (0,) for g in groups):
        raise metafopy.errors.InputError("Cannot compute VIF(s) for the specified 'btt' argument.")
    entries: list[metafopy.methods.vif.VIFEntry] = []
    for spec, g in zip(specs, groups, strict=True):
        gvif, used = metafopy.methods.vif.compvif_matrix(g, res.vb, int_incl, refit, res.X, intercept)
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
    tab = coef_table(res) if table and btt_missing else None
    result = metafopy.methods.vif.VIFResult(
        vif=tuple(entries), bttspec=not btt_missing, digits=metafopy.mfopt.stored_digits(digits, 4), table=tab, test=res.test if tab is not None else None
    )
    return result, groups


@metafopy.methods.vif.vif.register
@metafopy.core.timing.timed
def vif_rma_mv(
    res: metafopy.models.mv_result.RMAMVResult,
    btt: metafopy.methods.vif.BttArg = None,
    att: typing.Any = None,
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
    LB: bool = False,
    joina: collections.abc.Sequence[typing.Any] | None = None,
) -> metafopy.methods.vif.VIFResult:
    """(G)VIFs for the coefficients of an ``rma.mv`` meta-regression, as :func:`metafopy.vif` for ``rma.uni`` fits.
    ``joina`` is accepted and unused, as in metafor.

    ``btt`` (0-based; an int, a regular expression, a list of ints or a list of sets) selects sets of coefficients, ``table=True``
    adds the coefficient table, ``intercept=True`` includes the intercept and ``fixed=True`` matches a string ``btt`` literally.
    ``reestimate=True`` refits the model for each set with only its coefficients, the estimated variance components being
    re-estimated. ``sim=True`` (1000 iterations) or ``sim=n`` refits the model ``n`` times after permuting the moderators (the variables
    of the data when the moderators were given as a formula, otherwise the columns of the model matrix) with R's generator seeded
    with ``seed + l`` in iteration ``l`` (otherwise from the stream of :func:`metafopy.set_seed`), giving ``sim`` and ``prop``.
    ``parallel`` (``'snow'``/``'multicore'``) runs the iterations in a process pool of ``ncpus`` workers, or on the
    :class:`concurrent.futures.Executor` given as ``cl``. ``joinb`` gives one label per column of the model matrix and the columns that share
    a label are permuted together (the columns, not the variables of the data, are then permuted); ``LB`` is accepted as in metafor and changes nothing.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    n_sim = check_arguments(res, att, sim)
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    result, groups = vif_coefficients(res, btt, table, make_refit(res) if reestimate else None, intercept, fixed, digits)
    vifs = numpy.array([e.vif for e in result.vif])
    # simulating is pointless when every (G)VIF is already 1
    if n_sim >= 2 and any(res.coef_na):
        warnings.warn("Cannot use 'sim' when some redundant predictors were dropped from the model.", stacklevel=2)
        n_sim = 0
    if n_sim >= 2 and not numpy.all(vifs[~numpy.isnan(vifs)] == 1):
        seeds: list[int | None]
        if seed is not None:
            seeds = [seed + i for i in range(1, n_sim + 1)]
        elif parallel == 'no':
            seeds = [None] * n_sim
        else:
            seeds = [int(metafopy.core.rng.get_rng().unif_index(2**31 - 1)) for _ in range(n_sim)]
        formula = res.formula_mods
        data = res.spec.get('data')
        rows = None
        if formula is not None and joinb is None and isinstance(data, pandas.DataFrame):
            rows = data.reset_index(drop=True).iloc[numpy.array(res.ids, dtype=numpy.intp)[res.not_na]]
        one = functools.partial(sim_one, res, tuple(groups), reestimate, intercept, formula, rows, None if joinb is None else tuple(joinb))
        sims = numpy.vstack(metafopy.core.parallel.map_sims(one, seeds, parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar))
        with numpy.errstate(invalid='ignore', divide='ignore'):
            ok = ~numpy.isnan(sims)
            prop = ((vifs[None, :] >= sims) & ok).sum(axis=0) / ok.sum(axis=0)
        result = dataclasses.replace(result, sim=sims, sim_type='data' if rows is not None else 'X', prop=prop)
    return result


@metafopy.methods.vif.vif.register
@metafopy.core.timing.timed
def vif_selmodel(
    res: metafopy.models.sel_result.SelmodelResult,
    btt: metafopy.methods.vif.BttArg = None,
    att: typing.Any = None,
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
    LB: bool = False,
    joina: collections.abc.Sequence[typing.Any] | None = None,
) -> metafopy.methods.vif.VIFResult:
    """(G)VIFs for the coefficients of a ``selmodel`` fit with moderators, from the variance-covariance matrix of the coefficients.
    ``joina`` is accepted and unused.

    The arguments are those of :func:`metafopy.vif`; as in metafor, ``sim`` and ``reestimate`` are not available for these fits
    (``progbar``, ``seed``, ``parallel``, ``ncpus``, ``cl``, ``joinb`` and ``LB`` are accepted and unused).

    Raises:
        metafopy.errors.InputError: an argument is invalid or ``sim`` or ``reestimate`` was requested.
    """
    n_sim = check_arguments(res, att, sim)
    if n_sim >= 2:
        raise metafopy.errors.InputError("Cannot use 'sim' with models of class 'rma.uni.selmodel'.")
    if reestimate:
        raise metafopy.errors.InputError("Cannot use 'restimate=TRUE' with models of class 'rma.uni.selmodel'.")
    result, _ = vif_coefficients(res, btt, table, None, intercept, fixed, digits)
    return result
