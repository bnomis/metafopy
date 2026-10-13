"""Variance inflation factors for ``rma.uni`` meta-regressions (metafor's ``vif.rma``).

The (generalised) variance inflation factor of a set of coefficients ``b``
is ``det(R_bb) det(R_rr) / det(R)``, with ``R`` the correlation matrix of the
coefficient estimates (intercept removed) and ``r`` the remaining coefficients
(Fox & Monette 1992, doi:10.1080/01621459.1992.10475190). With
``reestimate=True`` the model is refitted with only the coefficients in ``b``
and the GVIF is the ratio of the determinants of the two variance-covariance
blocks. ``sim`` compares each (G)VIF against those obtained after permuting the
columns of the model matrix, so that the moderators are independent. See
https://wviechtb.github.io/metafor/reference/vif.html.
"""

import collections.abc
import concurrent.futures
import dataclasses
import functools
import math
import multiprocessing
import sys
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.core.linalg
import metafopy.errors
import metafopy.methods.blup
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

__all__ = ['VIFEntry', 'VIFResult', 'vif']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BttSpec = int | str | collections.abc.Sequence[int]
type BttArg = int | str | collections.abc.Sequence[int | str | collections.abc.Sequence[int]] | None
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type Parallel = typing.Literal['no', 'snow', 'multicore']

FIT_ERRORS = (metafopy.errors.EstimationError, ValueError, ArithmeticError, numpy.linalg.LinAlgError)


@dataclasses.dataclass(frozen=True, slots=True)
class VIFEntry:
    """(G)VIF of one set of coefficients; ``coefs`` is 1-based, as metafor prints it."""

    spec: str
    coefs: str
    coefname: str
    m: int
    vif: float
    sif: float


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class VIFResult:
    """(G)VIFs and (G)SIFs (metafor's ``vif.rma`` object).

    ``sim`` holds the simulated (G)VIFs (one row per iteration, one column
    per coefficient set) and ``prop`` the proportion of them that the observed
    value is at least as large as.
    """

    vif: tuple[VIFEntry, ...]
    bttspec: bool
    digits: int = 4
    table: pandas.DataFrame | None = None
    test: str | None = None
    sim: FloatArray | None = None
    sim_type: str | None = None
    prop: FloatArray | None = None

    @property
    def names(self) -> list[str]:
        return [e.coefs if self.bttspec else e.coefname for e in self.vif]

    @property
    def vifs(self) -> pandas.Series:
        """(G)VIFs indexed by coefficient name (or by ``coefs`` when ``btt`` was given)."""
        return pandas.Series([e.vif for e in self.vif], index=self.names, name='vif')

    @property
    def sifs(self) -> pandas.Series:
        return pandas.Series([e.sif for e in self.vif], index=self.names, name='sif')

    def to_frame(self) -> pandas.DataFrame:
        """As metafor's ``as.data.frame.vif.rma``."""
        tab = pandas.DataFrame(
            {
                'spec': [e.spec for e in self.vif],
                'coefs': [e.coefs for e in self.vif],
                'm': [e.m for e in self.vif],
                'vif': [e.vif for e in self.vif],
                'sif': [e.sif for e in self.vif],
            },
            index=None if self.bttspec else self.names,
        )
        if self.prop is not None:
            tab['prop'] = self.prop
        if numeric_specs(self.vif):
            tab = tab.drop(columns='spec')
        return tab

    def __str__(self) -> str:
        d = self.digits
        if self.bttspec:
            if len(self.vif) == 1:
                e = self.vif[0]
                if e.m == 1:
                    head, line = (
                        f'Collinearity Diagnostics (coefficient {e.coefs}):',
                        f'VIF = {metafopy.printing.fmt_num(e.vif, d)}, SIF = {metafopy.printing.fmt_num(e.sif, d)}',
                    )
                else:
                    head, line = (
                        f'Collinearity Diagnostics (coefficients {e.coefs}):',
                        f'GVIF = {metafopy.printing.fmt_num(e.vif, d)}, GSIF = {metafopy.printing.fmt_num(e.sif, d)}',
                    )
                if self.prop is not None:
                    line += f', prop = {metafopy.printing.fmt_num(float(self.prop[0]), 2)}'
                return '\n'.join(['', head, line, ''])
            cols: list[tuple[str, list[str]]] = []
            if not numeric_specs(self.vif):
                cols.append(('spec', [e.spec for e in self.vif]))
            cols += [
                ('coefs', [e.coefs for e in self.vif]),
                ('m', [str(e.m) for e in self.vif]),
                ('vif', [metafopy.printing.fmt_num(e.vif, d) for e in self.vif]),
                ('sif', [metafopy.printing.fmt_num(e.sif, d) for e in self.vif]),
            ]
            if self.prop is not None:
                cols.append(('prop', [metafopy.printing.fmt_num(float(v), 2) for v in self.prop]))
            return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, [str(i + 1) for i in range(len(self.vif))]), ''])

        vifs = [e.vif for e in self.vif]
        sifs = [e.sif for e in self.vif]
        prop = None if self.prop is None else [float(v) for v in self.prop]
        if self.table is None:
            if prop is None:
                return '\n'.join(['', *format_vector(self.names, [metafopy.printing.fmt_num(v, d) for v in vifs]), ''])
            cols = [('vif', [metafopy.printing.fmt_num(v, d) for v in vifs]), ('prop', [metafopy.printing.fmt_num(v, 2) for v in prop])]
            return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, self.names, gap=2), ''])

        tab = self.table
        if len(vifs) != len(tab):
            # no (G)VIF for the intercept
            vifs, sifs = [math.nan, *vifs], [math.nan, *sifs]
            prop = None if prop is None else [math.nan, *prop]
        stat = 'tval' if 'tval' in tab.columns else 'zval'
        cols = [
            ('estimate', [metafopy.printing.fmt_num(v, d) for v in tab['estimate']]),
            ('se', [metafopy.printing.fmt_num(v, d) for v in tab['se']]),
            (stat, [metafopy.printing.fmt_num(v, d) for v in tab[stat]]),
        ]
        if 'df' in tab.columns:
            cols.append(('df', [f'{round(float(v), 2):g}' for v in tab['df']]))
        cols += [
            ('pval', [metafopy.printing.fmt_pval(v, d) for v in tab['pval']]),
            ('ci.lb', [metafopy.printing.fmt_num(v, d) for v in tab['ci.lb']]),
            ('ci.ub', [metafopy.printing.fmt_num(v, d) for v in tab['ci.ub']]),
            ('vif', [metafopy.printing.fmt_num(v, d) for v in vifs]),
            ('sif', [metafopy.printing.fmt_num(v, d) for v in sifs]),
        ]
        if prop is not None:
            cols.append(('prop', [metafopy.printing.fmt_num(v, 2) for v in prop]))
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, [str(n) for n in tab.index]), ''])


def numeric_specs(entries: collections.abc.Sequence[VIFEntry]) -> bool:
    # metafor drops the spec column when every specification starts with a digit 1-9
    return all(e.spec[:1] in '123456789' and e.spec[:1] != '' for e in entries)


def format_vector(names: collections.abc.Sequence[str], values: collections.abc.Sequence[str], gap: int = 2) -> list[str]:
    """metafor's ``.print.vector``: names over values, right aligned."""
    widths = [max(len(n), len(v)) for n, v in zip(names, values, strict=True)]
    sep = ' ' * gap
    return [sep.join(n.rjust(w) for n, w in zip(names, widths, strict=True)), sep.join(v.rjust(w) for v, w in zip(values, widths, strict=True))]


def format_btt(idx: collections.abc.Iterable[int]) -> str:
    """metafor's ``.format.btt``: ``[2, 3, 4, 6]`` -> ``'2:4,6'`` (values as given)."""
    vals = sorted(idx)
    parts: list[str] = []
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[j] == 1:
            j += 1
        parts.append(f'{vals[i]}:{vals[j]}' if j > i else str(vals[i]))
        i = j + 1
    return ','.join(parts)


@functools.singledispatch
def vif(res: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> VIFResult:
    """Variance inflation factors for a fitted model; see :func:`_vif_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'vif() not implemented for {type(res).__name__}')


@vif.register
def vif_rma_uni(
    res: metafopy.models.rma_uni.RMAResult,
    btt: BttArg = None,
    att: typing.Any = None,
    table: bool = False,
    reestimate: bool = False,
    sim: bool | int = False,
    progbar: bool = True,
    seed: int | None = None,
    parallel: Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    digits: int | None = None,
) -> VIFResult:
    """(G)VIFs for the coefficients of an ``rma.uni`` meta-regression.

    ``btt`` takes 0-based indices: an int, a regex or a list of ints gives one
    set of coefficients (one GVIF); a list whose elements include lists or
    strings gives one set per element (``btt=[1, [2, 3], 4]``, like metafor's
    ``btt=list(2, 3:4, 5)``). Without ``btt`` each coefficient except the
    intercept gets its own VIF. ``sim=True`` (1000 iterations) or ``sim=n``
    refits the model ``n`` times with independently permuted model-matrix
    columns; ``seed`` seeds :func:`numpy.random.default_rng`, from which each
    iteration gets its own spawned generator, so results do not depend on
    ``parallel``. ``parallel`` (``'snow'``/``'multicore'``) runs the
    iterations in a process pool of ``ncpus`` workers, or on the
    :class:`concurrent.futures.Executor` given as ``cl``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if att is not None:
        raise metafopy.errors.InputError("Argument 'att' only relevant for location-scale models.")
    if res.int_only:
        raise metafopy.errors.InputError('VIFs not applicable to intercept-only models.')

    if isinstance(sim, bool):
        n_sim = 1000 if sim else 0
    else:
        n_sim = round(sim)
        if n_sim <= 1:
            raise metafopy.errors.InputError("Argument 'sim' must be >= 2.")

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if parallel not in ('no', 'snow', 'multicore'):
        raise metafopy.errors.InputError(f'unknown parallel option {parallel!r}')
    if ncpus < 1:
        raise metafopy.errors.InputError("Argument 'ncpus' must be >= 1.")
    if parallel == 'no' and ncpus > 1:
        parallel = 'snow'
    if cl is not None:
        parallel = 'snow'
    elif parallel != 'no' and ncpus < 2:
        parallel = 'no'

    int_incl = res.int_incl
    p = res.p
    names = res.coef_names

    btt_missing = btt is None
    specs: list[BttSpec]
    if btt is None:
        specs = [j for j in range(1 if int_incl else 0, p)]
    elif isinstance(btt, (str, int)):
        specs = [btt]
    elif any(isinstance(b, (str, collections.abc.Sequence)) for b in btt):
        specs = list(btt)
    else:
        specs = [list(btt)]  # type: ignore[arg-type]
    groups = [metafopy.models.rma_uni.set_btt(s, p, int_incl, names) for s in specs]
    if int_incl and any(g == (0,) for g in groups):
        raise metafopy.errors.InputError("Cannot compute VIF(s) for the specified 'btt' argument.")

    entries: list[VIFEntry] = []
    for spec, g in zip(specs, groups, strict=True):
        gvif, used = compvif(g, res.vb, int_incl, res if reestimate else None)
        m = len(used)
        entries.append(
            VIFEntry(
                spec=format_spec(spec),
                coefs=format_btt(i + 1 for i in used),
                coefname=names[used[0]] if m == 1 else '',
                m=m,
                vif=gvif,
                sif=gvif ** (1 / (2 * m)),
            )
        )

    tab: pandas.DataFrame | None = None
    if table and btt_missing:
        tab = coef_table(res)

    sims: FloatArray | None = None
    prop: FloatArray | None = None
    vifs = numpy.array([e.vif for e in entries])
    # simulating is pointless when every (G)VIF is already 1
    if n_sim >= 2 and not numpy.all(vifs[~numpy.isnan(vifs)] == 1):
        rngs = numpy.random.default_rng(seed).spawn(n_sim)
        one = functools.partial(vif_sim_one, res, tuple(groups), reestimate)
        if parallel == 'no':
            rows = list(progress(map(one, rngs), n_sim, progbar))
        else:
            # 'multicore' forks the workers (as R's mclapply); 'snow' starts fresh ones
            ctx = multiprocessing.get_context('fork' if parallel == 'multicore' and 'fork' in multiprocessing.get_all_start_methods() else None)
            pool = cl if cl is not None else concurrent.futures.ProcessPoolExecutor(max_workers=ncpus, mp_context=ctx)
            try:
                rows = list(progress(pool.map(one, rngs), n_sim, progbar))
            finally:
                if cl is None:
                    pool.shutdown()
        sims = numpy.vstack(rows)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            ok = ~numpy.isnan(sims)
            hits = (vifs[None, :] >= sims) & ok
            prop = hits.sum(axis=0) / ok.sum(axis=0)

    return VIFResult(
        vif=tuple(entries),
        bttspec=not btt_missing,
        digits=4 if digits is None else digits,
        table=tab,
        test=res.test if tab is not None else None,
        sim=sims,
        sim_type='X' if sims is not None else None,
        prop=prop,
    )


def format_spec(spec: BttSpec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, int):
        return str(spec + 1)
    return format_btt(int(i) + 1 for i in spec)


def cov2cor(V: FloatArray) -> FloatArray:
    with numpy.errstate(invalid='ignore', divide='ignore'):
        d = 1 / numpy.sqrt(numpy.diag(V))
    out: FloatArray = V * numpy.outer(d, d)
    return out


def compvif(group: collections.abc.Sequence[int], vb: FloatArray, int_incl: bool, fit: metafopy.models.rma_uni.RMAResult | None) -> tuple[float, list[int]]:
    """GVIF of the coefficients ``group`` (0-based) and the indices actually used.

    Without ``fit`` the GVIF comes from the correlation matrix of the
    estimates; with ``fit`` the model is refitted with only those coefficients
    (plus the intercept) and the variance-covariance blocks are compared.
    """
    if fit is None:
        if int_incl:
            V = vb[1:, 1:]
            b = [i - 1 for i in group if i > 0]
        else:
            V = vb
            b = list(group)
        rb = cov2cor(V)
        rest = [i for i in range(V.shape[0]) if i not in b]
        gvif = float(numpy.linalg.det(rb[numpy.ix_(b, b)]) * numpy.linalg.det(rb[numpy.ix_(rest, rest)]) / numpy.linalg.det(rb))
        return gvif, [i + 1 for i in b] if int_incl else b

    b = [i for i in group if i > 0] if int_incl else list(group)
    Xb = fit.X[:, b]
    if int_incl:
        Xb = numpy.column_stack([numpy.ones(fit.k), Xb])
    try:
        sub = refit(fit, Xb)
    except FIT_ERRORS:
        return math.nan, b
    vb_sub = sub.vb[1:, 1:] if int_incl else sub.vb
    return float(numpy.linalg.det(vb[numpy.ix_(b, b)]) / numpy.linalg.det(vb_sub)), b


def refit(res: metafopy.models.rma_uni.RMAResult, X: FloatArray) -> metafopy.models.rma_uni.RMAResult:
    """Refit ``res`` with model matrix ``X`` (metafor's ``rma.uni(..., mods=X, intercept=FALSE)``)."""
    md = dataclasses.replace(res.data, yi=res.yi, vi=res.vi, X=X, X_names=tuple(f'X{j + 1}' for j in range(X.shape[1])), weights=res.weights, intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.models.rma_uni.fit_rma_uni(
            md,
            method=res.method,
            weighted=res.weighted,
            test=res.test,
            level=100 * (1 - res.level),
            tau2=res.tau2 if res.tau2_fix else None,
            control=res.control,
            skip_r2=True,
        )


def vif_sim_one(res: metafopy.models.rma_uni.RMAResult, groups: tuple[tuple[int, ...], ...], reestimate: bool, rng: numpy.random.Generator) -> FloatArray:
    """(G)VIFs after permuting each column of the model matrix independently."""
    X = res.X.copy()
    for j in range(X.shape[1]):
        X[:, j] = rng.permutation(X[:, j])
    drop = metafopy.core.linalg.drop_redundant(X)
    if drop.any():
        # a set containing a coefficient that became redundant is infinitely inflated
        return numpy.array([math.inf if drop[list(g)].any() else math.nan for g in groups])
    try:
        fit = refit(res, X)
    except FIT_ERRORS:
        return numpy.full(len(groups), math.nan)
    return numpy.array([compvif(g, fit.vb, res.int_incl, fit if reestimate else None)[0] for g in groups])


def coef_table(res: metafopy.models.rma_uni.RMAResult) -> pandas.DataFrame:
    """metafor's ``coef(summary(x))``."""
    cols: dict[str, typing.Any] = {'estimate': res.beta, 'se': res.se}
    if res.ddf is not None:
        cols['tval'] = res.zval
        cols['df'] = numpy.full(res.p, res.ddf)
    else:
        cols['zval'] = res.zval
    cols.update({'pval': res.pval, 'ci.lb': res.ci_lb, 'ci.ub': res.ci_ub})
    return pandas.DataFrame(cols, index=list(res.coef_names))


def progress(rows: collections.abc.Iterable[FloatArray], total: int, show: bool) -> collections.abc.Iterator[FloatArray]:
    for i, row in enumerate(rows, start=1):
        if show:
            print(f'\r  |{"+" * (50 * i // total):<50}| {100 * i // total:3d}%', end='' if i < total else '\n', file=sys.stderr)
        yield row
