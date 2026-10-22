"""Graphical display of study heterogeneity (metafor's ``gosh.rma``).

The model is fitted to a large number of subsets of the ``k`` studies (all
``2^k - p - ...`` of them if feasible, otherwise a random sample) and the
pooled estimate(s) and heterogeneity statistics of each fit are collected, as
proposed by Olkin, Dahabreh & Trikalinos (2012), doi:10.1002/jrsm.1033. See
https://wviechtb.github.io/metafor/reference/gosh.rma.html.
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

import metafopy.core.dbinom
import metafopy.core.parallel
import metafopy.core.rng
import metafopy.core.timing
import metafopy.effects.datatools
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

__all__ = ['GoshResult', 'gosh']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# subsets fitted per task (and per step of the progress bar)
CHUNK = 1000


@dataclasses.dataclass(frozen=True, slots=True)
class GoshResult:
    """Statistics of the model fitted to each subset of studies.

    ``res`` has one row per subset (sorted by the number of studies ``k``) with the
    columns ``k``, ``QE``, ``I2``, ``H2``, ``tau2``, ``tau`` and the coefficient estimates;
    ``incl`` is the corresponding matrix of inclusion indicators, ``fit`` marks successful fits.

    Attributes:
        res: Data frame with one row per subset of studies.
        incl: Boolean matrix with one row per subset and one column per study; true where the study is in the subset.
        fit: Boolean flags for the subsets whose model fit succeeded.
        k: Number of studies of the model.
        int_only: Whether the model has no moderators.
        method: The ``tau^2`` estimator of the model.
        measure: The effect size measure of the model.
        digits: Number of decimals used when the result is printed.
    """

    res: pandas.DataFrame
    incl: numpy.typing.NDArray[numpy.bool_]
    fit: numpy.typing.NDArray[numpy.bool_]
    k: int
    int_only: bool
    method: str
    measure: str
    digits: int = 4

    def summary(self) -> str:
        return format_gosh(self)

    def __str__(self) -> str:
        return format_gosh(self)


def format_gosh(x: GoshResult) -> str:
    d = x.digits
    res = x.res
    stats_: list[tuple[str, collections.abc.Callable[[FloatArray], float]]] = [
        ('mean', lambda c: float(numpy.nanmean(c))),
        ('min', lambda c: float(numpy.nanmin(c))),
        ('q1', lambda c: float(numpy.nanquantile(c, 0.25))),
        ('median', lambda c: float(numpy.nanquantile(c, 0.5))),
        ('q3', lambda c: float(numpy.nanquantile(c, 0.75))),
        ('max', lambda c: float(numpy.nanmax(c))),
    ]
    names = list(res.columns)
    rows: list[list[str]] = [[metafopy.printing.fmt_num(f(res[c].to_numpy()), d) for _, f in stats_] for c in names]
    labels = list(names)
    if len(names) == 6:
        labels[1] = 'Q'
    if len(names) > 6:
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        rows = [*rows[:5], [''] * 6, *rows[5:]]
        labels = [*labels[:5], '', *labels[5:]]
    if x.method in metafopy.models.tau2.FIXED_EFFECTS:
        del rows[4]
        del labels[4]
    cols = [(h, [r[j] for r in rows]) for j, (h, _) in enumerate(stats_)]
    head = ['', f'Model fits attempted: {len(x.fit)}', f'Model fits succeeded: {int(x.fit.sum())}', '']
    return '\n'.join([*head, *metafopy.methods.blup.format_r_frame(cols, labels), ''])


@functools.singledispatch
def gosh(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> GoshResult:
    """GOSH (graphical display of study heterogeneity) analysis of a fitted model.

    This is a single-dispatch function: the implementation is chosen from the class of ``x``. See
    [gosh_rma_uni][metafopy.methods.gosh.gosh_rma_uni]. See <https://wviechtb.github.io/metafor/reference/gosh.html>.

    Args:
        x: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``x``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``x``.

    Returns:
        What the implementation returns, a [GoshResult][metafopy.GoshResult].

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
    raise metafopy.errors.ArgumentError(f'gosh() not implemented for {type(x).__name__}')


@gosh.register
@metafopy.core.timing.timed
def gosh_rma_uni(
    x: metafopy.models.rma_uni.RMAResult,
    subsets: int | None = None,
    *,
    seed: int | None = None,
    rng: metafopy.core.rng.RRandom | None = None,
    progbar: bool = False,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    LB: bool = False,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> GoshResult:
    """Fit the model to subsets of the studies (metafor's ``gosh.rma``).

    Olkin, Dahabreh & Trikalinos (2012, [doi:10.1002/jrsm.1033](https://doi.org/10.1002/jrsm.1033)) proposed examining the pooled estimate and the
    heterogeneity statistics of the model fitted to all subsets of the studies; Viechtbauer (2021) generalised this to models with moderators.
    By default all subsets (with at least ``p`` studies, ``p`` being the number of coefficients) are used if there are at most ``10**6`` of
    them, otherwise ``10**6`` random subsets are drawn. The subsets are drawn before the fits, so the result equals that of a serial run.
    See <https://wviechtb.github.io/metafor/reference/gosh.html>.

    Args:
        x: A model fitted by [rma_uni][metafopy.rma_uni], with or without moderators; not allowed for a model passed through
            [robust][metafopy.robust], or for location-scale and selection models.
        subsets: Number of subsets to use. ``None`` (the default) uses all subsets if there are at most ``10**6``, else ``10**6`` random ones.
            If it is at least the number of possible subsets, all subsets are used. Must be 2 or more.
        seed: Seed of the R-compatible random number generator used to draw random subsets, so that ``gosh(res, seed=s)`` draws the same
            subset sizes and study indices as R's ``set.seed(s)``. ``None`` uses the global generator (see [set_seed][metafopy.set_seed]).
            Only used when random subsets are drawn.
        rng: An R-compatible generator to draw the random subsets from, instead of ``seed`` or the global generator.
        progbar: Print a progress bar (per block of subsets) to stderr (default ``False``).
        parallel: ``'no'`` (the default), ``'snow'`` or ``'multicore'``; fit the subsets in a process pool (see ``ncpus``).
        ncpus: Number of worker processes (default 1; a value above 1 turns parallel processing on).
        cl: An executor to fit the subsets on, instead of one created for the call.
        LB: Accepted as in metafor (load-balanced scheduling) and changes no result.
        na_action: Stands for R's ``options(na.action=)``; metafor only checks that it is one of ``'omit'`` (the default), ``'exclude'``,
            ``'pass'`` or ``'fail'``, and so does this function.

    Returns:
        A [GoshResult][metafopy.GoshResult]. Its ``res`` data frame has one row per subset (sorted by the number of studies) with the columns
        ``k``, ``QE``, ``I2``, ``H2``, ``tau2``, ``tau`` and the coefficient estimates (``estimate`` for a model without moderators);
        ``incl`` is the matrix of inclusion indicators.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model is of a class that is not supported (see ``x``).
    """
    metafopy.methods.common.check_na_action(na_action)
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
    k, p = x.k, x.p
    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    if k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    incl = inclusion_matrix(k, p, subsets, seed, rng)

    model = 0
    if x.method in metafopy.models.tau2.FIXED_EFFECTS and x.weighted and x.weights is None and x.int_only:
        model = 1
    if x.method == 'DL' and x.weighted and x.weights is None and x.int_only:
        model = 2

    n = incl.shape[0]
    blocks = [incl[i : i + CHUNK] for i in range(0, n, CHUNK)]
    one = functools.partial(gosh_block, x, model)
    parts = metafopy.core.parallel.map_sims(one, blocks, parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
    beta = numpy.concatenate([b for b, _ in parts])
    het = numpy.concatenate([h for _, h in parts])
    # auditme: accept[literal-duplication] R's column name for an intercept-only model
    return assemble(beta, het, incl, ['estimate'] if x.int_only else list(x.coef_names), x.int_only, x.method, x.measure)


def inclusion_matrix(k: int, p: int, subsets: int | None, seed: int | None, rng: metafopy.core.rng.RRandom | None) -> numpy.typing.NDArray[numpy.bool_]:
    """The subsets of the ``k`` studies (with at least ``p`` of them) that ``gosh.rma`` fits the model to, one per row.

    Raises:
        metafopy.errors.InputError: ``subsets`` is smaller than 2.
    """
    n_tot = sum(math.comb(k, m) for m in range(p, k + 1))
    if subsets is None:
        exact = n_tot <= 10**6
        if not exact:
            n_tot = 10**6
    else:
        subsets = round(subsets)
        if subsets <= 1:
            raise metafopy.errors.InputError("Argument 'subsets' must be >= 2.")
        exact = n_tot <= subsets
        if not exact:
            n_tot = subsets
    if exact:
        r = numpy.arange(2**k)
        incl_all = ((r[:, None] >> numpy.arange(k)[None, :]) & 1).astype(numpy.bool_)
        out: numpy.typing.NDArray[numpy.bool_] = incl_all[incl_all.sum(axis=1) >= p]
        return out
    gen = rng if rng is not None else (metafopy.core.rng.set_seed(seed) if seed is not None else metafopy.core.rng.get_rng())
    sizes = numpy.arange(p, k + 1)
    j = sizes[gen.sample_int(len(sizes), n_tot, replace=True, prob=metafopy.core.dbinom.dbinom(sizes, k, 0.5)) - 1]
    incl = numpy.zeros((n_tot, k), dtype=numpy.bool_)
    for i, m in enumerate(j):
        incl[i, gen.sample_int(k, int(m)) - 1] = True
    return incl


def assemble(
    beta: FloatArray, het: FloatArray, incl: numpy.typing.NDArray[numpy.bool_], names: list[str], int_only: bool, method: str, measure: str
) -> GoshResult:
    """The :class:`GoshResult` of the fits to the subsets in ``incl`` (``het`` holds ``k``, ``QE``, ``I2``, ``H2`` and ``tau2``).

    Raises:
        metafopy.errors.InputError: all fits failed.
    """
    ok = ~numpy.isnan(het).any(axis=1)
    if not ok.any():
        raise metafopy.errors.InputError('All model fits failed.')
    het[:, 0] = incl.sum(axis=1)
    cols = ['k', 'QE', 'I2', 'H2', 'tau2']
    het6 = numpy.column_stack([het, numpy.sqrt(het[:, 4])])
    frame = pandas.DataFrame(
        numpy.column_stack([het6, beta]), columns=metafopy.effects.datatools.make_names([*cols, 'tau', *[str(n) for n in names]], unique=True)
    )
    order = numpy.argsort(frame['k'].to_numpy(), kind='stable')
    frame = frame.iloc[order].reset_index(drop=True)
    incl = incl[order]
    fit_ok = ~frame.isna().any(axis=1).to_numpy()
    return GoshResult(res=frame, incl=incl, fit=fit_ok, k=incl.shape[1], int_only=int_only, method=method, measure=measure)


def gosh_block(x: metafopy.models.rma_uni.RMAResult, model: int, incl: numpy.typing.NDArray[numpy.bool_]) -> tuple[FloatArray, FloatArray]:
    """The coefficients and heterogeneity statistics (k, QE, I2, H2, tau2) of the fits to the subsets in the rows of ``incl``."""
    n = incl.shape[0]
    beta = numpy.full((n, x.p), numpy.nan)
    het = numpy.full((n, 5), numpy.nan)
    for i in range(n):
        mask = incl[i]
        if model:
            beta[i], het[i] = closed_form(x, mask, model)
            continue
        fit = metafopy.methods.common.refit(x, mask)
        if fit is None:
            continue
        beta[i] = fit.beta
        het[i] = (fit.k, fit.QE, fit.I2, fit.H2, fit.tau2)
    return beta, het


def closed_form(x: metafopy.models.rma_uni.RMAResult, mask: numpy.typing.NDArray[numpy.bool_], model: int) -> tuple[FloatArray, FloatArray]:
    """The EE/FE and DL fits to a subset of an intercept-only model in closed form (metafor's ``model`` shortcut)."""
    yi, vi = x.yi[mask], x.vi[mask]
    k = yi.shape[0]
    wi = 1.0 / vi
    sumwi = wi.sum()
    est = float(numpy.sum(wi * yi) / sumwi)
    Q, I2, H2, tau2 = 0.0, 0.0, 1.0, 0.0
    if k > 1:
        Q = float(numpy.sum(wi * (yi - est) ** 2))
        I2 = max(0.0, 100 * (Q - (k - 1)) / Q)
        H2 = Q / (k - 1)
        if model == 2:
            tau2 = max(0.0, (Q - (k - 1)) / (sumwi - float(numpy.sum(wi**2)) / sumwi))
            wi = 1.0 / (vi + tau2)
            est = float(numpy.sum(wi * yi) / wi.sum())
    return numpy.array([est]), numpy.array([k, Q, I2, H2, tau2], dtype=numpy.float64)
