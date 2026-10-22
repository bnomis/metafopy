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
import re
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.core.linalg
import metafopy.core.parallel
import metafopy.core.rng
import metafopy.core.timing
import metafopy.errors
import metafopy.formula
import metafopy.methods.blup
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

__all__ = ['VIFEntry', 'VIFResult', 'vif']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BttSpec = int | str | collections.abc.Sequence[int]
type BttArg = int | str | collections.abc.Sequence[int | str | collections.abc.Sequence[int]] | None

FIT_ERRORS = (metafopy.errors.EstimationError, ValueError, ArithmeticError, numpy.linalg.LinAlgError)


@dataclasses.dataclass(frozen=True, slots=True)
class VIFEntry:
    """The (G)VIF of one coefficient or one set of coefficients, an element of [VIFResult][metafopy.VIFResult].

    Attributes:
        spec: The specification of the set, formatted as in metafor's printout.
        coefs: The coefficients of the set, 1-based as metafor prints them.
        coefname: Name of the coefficient (empty for a set of several).
        m: Number of coefficients in the set.
        vif: The (generalized) variance inflation factor.
        sif: The (generalized) standard error inflation factor.
    """

    spec: str
    coefs: str
    coefname: str
    m: int
    vif: float
    sif: float


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class VIFResult:
    """(Generalized) variance inflation factors and standard error inflation factors, as returned by [vif][metafopy.vif] (metafor's ``vif.rma`` object).
    ``vifs`` and ``sifs`` give them as Series and ``to_frame()`` as a data frame.

    Attributes:
        vif: One [VIFEntry][metafopy.VIFEntry] per coefficient or set of coefficients.
        bttspec: Whether the sets were chosen with ``btt``.
        digits: Number of decimals of the printout.
        table: The coefficient table with the VIFs added (``table=True``), or None.
        test: Test of the model, ``'z'`` or ``'t'`` (and variants), for the table.
        sim: Simulated (G)VIFs under independence: one row per iteration, one column per coefficient set (``sim`` was given), or None.
        sim_type: What was permuted in the simulation (``'X'`` for model matrix columns).
        prop: Proportion of the simulated (G)VIFs that the observed value is at least as large as.
        attspec: For the scale part of a location-scale model: whether the sets were chosen with ``att``.
    """

    vif: tuple[VIFEntry, ...]
    bttspec: bool
    digits: int = 4
    table: pandas.DataFrame | None = None
    test: str | None = None
    sim: FloatArray | None = None
    sim_type: str | None = None
    prop: FloatArray | None = None
    attspec: bool | None = None
    """``not alpha_missing`` of the scale part of a location-scale model; metafor stores it as ``attspec`` there (and ``bttspec`` for the location part)."""

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
        return self.summary()

    def summary(self, digits: int | None = None, *, num: bool = False) -> str:
        """The printout of ``print(x, digits=, num=)``: ``num`` numbers the rows of the table of coefficients (``table=True``)."""
        d = metafopy.mfopt.option_digits(digits, self.digits) if digits is not None else self.digits
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
        rows = metafopy.printing.number_rows([str(n) for n in tab.index]) if num else [str(n) for n in tab.index]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, rows), ''])


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
    """Variance inflation factors for a fitted meta-regression model (metafor's ``vif``).

    This is a single-dispatch function: the implementation is chosen by the class of ``res``. See [vif_rma_uni][metafopy.methods.vif.vif_rma_uni],
    [vif_rma_mv][metafopy.methods.vif_mv.vif_rma_mv], [vif_rma_ls][metafopy.methods.ls_vif.vif_rma_ls] and
    [vif_selmodel][metafopy.methods.vif_mv.vif_selmodel].

    Args:
        res: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation.
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        The (G)VIFs, a [VIFResult][metafopy.VIFResult].

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``res``, or an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'vif() not implemented for {type(res).__name__}')


def join_labels(join: typing.Any) -> tuple[typing.Any, ...] | None:
    """The labels of ``joinb`` or ``joina``; metafor reads them only for the simulation, so a value that is no vector (``joinb = TRUE``) is ignored."""
    if join is None or isinstance(join, bool | int | float | str):
        return None
    return tuple(join)


@vif.register
@metafopy.core.timing.timed
def vif_rma_uni(
    res: metafopy.models.rma_uni.RMAResult,
    btt: BttArg = None,
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
) -> VIFResult:
    """(Generalized) variance inflation factors for the coefficients of an [rma_uni][metafopy.rma_uni] meta-regression (metafor's ``vif.rma``).

    A VIF of a coefficient (or a GVIF of a set of coefficients) shows how much the variance of its estimate is inflated by correlation with the other
    moderators; the SIF is the corresponding inflation of the standard error.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/vif.rma.r>; reference page: <https://wviechtb.github.io/metafor/reference/vif.html>.

    Args:
        res: Model fitted by [rma_uni][metafopy.rma_uni] with moderators.
        btt: Sets of location coefficients to compute a (generalized) VIF for, 0-based: an int, a regular expression, or a list of ints gives one set (one
            GVIF); a list with lists or strings among its elements gives one set per element (``btt=[1, [2, 3], 4]``, like metafor's ``btt=list(2, 3:4,
            5)``). Default: each coefficient except the intercept gets its own VIF.
        att: Accepted for compatibility with metafor; it is only used for location-scale models.
        table: Add the VIFs to the coefficient table (only when ``btt`` is not given).
        reestimate: Refit the model with only the coefficients of each set when computing its (G)VIF.
        sim: Simulate the distribution of each (G)VIF under independence: ``True`` for 1000 iterations or a number of iterations. The model is refitted
            with independently permuted columns of the model matrix; the result then has ``sim`` and ``prop``. Iteration ``l`` is seeded with ``seed + l``
            when ``seed`` is given, as in metafor, otherwise from the stream of [set_seed][metafopy.set_seed].
        progbar: Print a progress bar to stderr while simulating (``sim``).
        seed: Seed for the simulation (``sim``). Default: drawn from the generator of [set_seed][metafopy.set_seed].
        parallel: Parallel simulation: ``'no'`` (default), ``'snow'`` or ``'multicore'``; both use a process pool of ``ncpus`` workers.
        ncpus: Number of worker processes when ``parallel`` is not ``'no'``.
        cl: Executor to run the iterations on instead of a process pool.
        digits: Number of decimals of the printout. Default: the ``digits`` option, else that of the model.
        intercept: Include the intercept among the coefficients (it is left out by default).
        fixed: Match a string ``btt`` literally instead of as a regular expression.
        joinb: One label per column of the model matrix; with ``sim`` the columns that share a label are permuted together.
        LB: Accepted as in metafor, where it balances the load of a ``'snow'`` cluster; it changes nothing here.
        joina: Accepted for compatibility with metafor, which reads it for every fit but uses it for location-scale models only.

    Returns:
        The (G)VIFs, a [VIFResult][metafopy.VIFResult] with one [VIFEntry][metafopy.VIFEntry] per coefficient or set (``vif``, ``sif``, ``m``), the
        optional coefficient ``table`` and, with ``sim``, the simulated values ``sim`` and the proportions ``prop``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid (for example the model has no moderators).
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

    if res.robust is not None and n_sim >= 2:
        raise metafopy.errors.InputError("Cannot use 'sim' with models of class 'robust.rma'.")
    if res.robust is not None and reestimate:
        raise metafopy.errors.InputError("Cannot use 'restimate=TRUE' with models of class 'robust.rma'.")

    parallel = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    join = join_labels(joinb)

    int_incl = res.int_incl
    p = res.p
    names = res.coef_names

    btt_missing = btt is None
    specs: list[BttSpec]
    if btt is None:
        specs = [j for j in range(1 if int_incl and not intercept else 0, p)]
    elif isinstance(btt, (str, int)):
        specs = [btt]
    elif any(isinstance(b, (str, collections.abc.Sequence)) for b in btt):
        specs = list(btt)
    else:
        specs = [list(btt)]  # type: ignore[arg-type]
    groups = [metafopy.models.rma_uni.set_btt(s, p, int_incl, names, fixed) for s in specs]
    if int_incl and not intercept and any(g == (0,) for g in groups):
        raise metafopy.errors.InputError("Cannot compute VIF(s) for the specified 'btt' argument.")

    entries: list[VIFEntry] = []
    for spec, g in zip(specs, groups, strict=True):
        gvif, used = compvif(g, res.vb, int_incl, res if reestimate else None, intercept)
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
        seeds: list[int | None]
        if seed is not None:
            seeds = [seed + i for i in range(1, n_sim + 1)]
        elif parallel == 'no':
            seeds = [None] * n_sim  # each iteration continues the global stream, as in R
        else:
            seeds = [int(metafopy.core.rng.get_rng().unif_index(2**31 - 1)) for _ in range(n_sim)]
        one = functools.partial(vif_sim_one, res, tuple(groups), reestimate, intercept, join)
        rows = metafopy.core.parallel.map_sims(one, seeds, parallel=parallel, ncpus=ncpus, cl=cl, progbar=progbar)
        sims = numpy.vstack(rows)
        with numpy.errstate(invalid='ignore', divide='ignore'):
            ok = ~numpy.isnan(sims)
            hits = (vifs[None, :] >= sims) & ok
            prop = hits.sum(axis=0) / ok.sum(axis=0)

    return VIFResult(
        vif=tuple(entries),
        bttspec=not btt_missing,
        digits=metafopy.mfopt.stored_digits(digits, 4),
        table=tab,
        test=res.test if tab is not None else None,
        sim=sims,
        sim_type=('data' if res.data.formula_mods is not None and join is None else 'X') if sims is not None else None,
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


def compvif(
    group: collections.abc.Sequence[int],
    vb: FloatArray,
    int_incl: bool,
    fit: metafopy.models.rma_uni.RMAResult | None,
    intercept: bool = False,
) -> tuple[float, list[int]]:
    """GVIF of the coefficients ``group`` (0-based) and the indices actually used.

    Without ``fit`` the GVIF comes from the correlation matrix of the
    estimates; with ``fit`` the model is refitted with only those coefficients
    (plus the intercept) and the variance-covariance blocks are compared.
    With ``intercept=True`` the intercept is treated as any other coefficient.
    """
    if fit is None:
        return compvif_matrix(group, vb, int_incl, None, None, intercept)

    def refit_with(Xb: FloatArray) -> FloatArray | None:
        try:
            return refit(fit, Xb).vb
        except FIT_ERRORS:
            return None

    return compvif_matrix(group, vb, int_incl, refit_with, fit.X, intercept)


def compvif_matrix(
    group: collections.abc.Sequence[int],
    vb: FloatArray,
    int_incl: bool,
    refit_with: collections.abc.Callable[[FloatArray], FloatArray | None] | None,
    X: FloatArray | None,
    intercept: bool = False,
) -> tuple[float, list[int]]:
    """metafor's ``.compvif``: the GVIF of ``group`` (0-based) from ``vb`` and the indices actually used.

    ``refit_with`` (given together with ``X``, the model matrix) returns the variance-covariance matrix of the model refitted
    with a replacement model matrix, or ``None`` if the refit fails (the GVIF is then ``nan``). ``intercept=True`` keeps the intercept in.
    """
    drop = int_incl and not intercept
    if refit_with is None or X is None:
        V = vb[1:, 1:] if drop else vb
        b = [i - 1 for i in group if i > 0] if drop else list(group)
        rb = cov2cor(V)
        rest = [i for i in range(V.shape[0]) if i not in b]
        with numpy.errstate(invalid='ignore', divide='ignore'):
            gvif = float(numpy.linalg.det(rb[numpy.ix_(b, b)]) * numpy.linalg.det(rb[numpy.ix_(rest, rest)]) / numpy.linalg.det(rb))
        return gvif, [i + 1 for i in b] if drop else b

    b = [i for i in group if i > 0] if drop else list(group)
    Xb = X[:, b]
    if drop:
        Xb = numpy.column_stack([numpy.ones(X.shape[0]), Xb])
    vb_sub = refit_with(Xb)
    if vb_sub is None:
        return math.nan, b
    if drop:
        vb_sub = vb_sub[1:, 1:]
    with numpy.errstate(invalid='ignore', divide='ignore'):
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


def permuted_design(res: metafopy.models.rma_uni.RMAResult, rng: metafopy.core.rng.RRandom, join: tuple[typing.Any, ...] | None = None) -> FloatArray:
    """The model matrix after shuffling, as metafor's ``.compvifsim`` does.

    A model given as a formula has the variables of the data it uses shuffled, in order of appearance in the formula, and the model matrix is
    rebuilt from them (``type="data"``); otherwise every column of the model matrix is shuffled (``type="X"``). Each shuffle is R's ``sample()``.
    """
    md = res.data
    rows = md.data.loc[md.keep] if md.formula_mods is not None and md.data is not None else None
    return permute_design(md.formula_mods, rows, res.X, rng, join)


def permute_design(
    formula: str | None, rows: pandas.DataFrame | None, X: FloatArray, rng: metafopy.core.rng.RRandom, join: collections.abc.Sequence[typing.Any] | None = None
) -> FloatArray:
    """Shuffle the variables of ``formula`` found in ``rows`` (the data rows used in the fit) and rebuild the model matrix, or else the columns of ``X``."""
    k = X.shape[0]
    if join is not None:
        return permute_joined(X, rng, join)
    if formula is not None and rows is not None:
        found = [(m.start(), c) for c in rows.columns if (m := re.search(rf'(?<![\w.]){re.escape(str(c))}(?![\w.])', formula)) is not None]
        frame = rows[[c for _, c in sorted(found, key=lambda t: t[0])]].reset_index(drop=True)
        if len(frame) == k and not frame.empty:
            shuffled = frame.apply(lambda col: col.iloc[rng.sample_int(k) - 1].reset_index(drop=True))
            Xnew, _, _ = metafopy.formula.design_matrix(formula, shuffled)
            if Xnew.shape == X.shape:
                return Xnew
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        Xs[:, j] = Xs[rng.sample_int(k) - 1, j]
    return Xs


def permute_joined(X: FloatArray, rng: metafopy.core.rng.RRandom, join: collections.abc.Sequence[typing.Any]) -> FloatArray:
    """metafor's ``.permXvif``: the columns of ``X`` with the same label in ``join`` are shuffled together, the labels in order of first appearance.

    Raises:
        metafopy.errors.InputError: ``join`` does not have one label per column.
    """
    if len(join) != X.shape[1]:
        raise metafopy.errors.InputError(
            f"Length of the 'join' argument ({len(join)}) does not match the number of columns of the model matrix ({X.shape[1]})."
        )
    Xs = X.copy()
    for label in dict.fromkeys(join):
        pos = [j for j, b in enumerate(join) if b == label]
        Xs[:, pos] = X[rng.sample_int(X.shape[0]) - 1][:, pos]
    return Xs


def vif_sim_one(
    res: metafopy.models.rma_uni.RMAResult,
    groups: tuple[tuple[int, ...], ...],
    reestimate: bool,
    intercept: bool,
    join: tuple[typing.Any, ...] | None,
    seed: int | None,
) -> FloatArray:
    """One iteration of metafor's ``.compvifsim``: (G)VIFs after permuting the model matrix (``seed=None`` continues the global stream)."""
    rng = metafopy.core.rng.get_rng() if seed is None else metafopy.core.rng.RRandom(seed)
    X = permuted_design(res, rng, join)
    drop = metafopy.core.linalg.drop_redundant(X)
    if drop.any():
        # a set containing a coefficient that became redundant is infinitely inflated
        return numpy.array([math.inf if drop[list(g)].any() else math.nan for g in groups])
    try:
        fit = refit(res, X)
    except FIT_ERRORS:
        return numpy.full(len(groups), math.nan)
    return numpy.array([compvif(g, fit.vb, res.int_incl, fit if reestimate else None, intercept)[0] for g in groups])


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
