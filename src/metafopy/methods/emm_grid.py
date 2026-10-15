"""Reference grids and estimated marginal means (metafor's ``emmprep`` and the core of R's emmeans package).

``emmprep`` packages the coefficients, their variance-covariance matrix, the degrees of freedom and the model formula of a fit
as a reference grid, as metafor does for emmeans' ``qdrg``: every combination of the levels of the factors in the model, with
the numeric predictors at their means (or at the values in ``at``). :func:`emmeans` averages the grid over the variables not
asked for (equal or proportional weights), :func:`contrast` and :func:`pairs` form linear combinations of the means, and
:func:`regrid` replaces them by their back-transformed values.

R sources: https://github.com/wviechtb/metafor/blob/master/R/emmprep.r,
https://github.com/rvlenth/emmeans/blob/master/R/ref-grid.R, https://github.com/rvlenth/emmeans/blob/master/R/emmeans.R and
https://github.com/rvlenth/emmeans/blob/master/R/contrast.R; reference: https://wviechtb.github.io/metafor/reference/emmprep.html and
https://rvlenth.github.io/emmeans/. The grid of the numeric predictors and factors follows Searle, Speed and Milliken (1980),
https://doi.org/10.1080/00031305.1980.10483031.
"""

import collections
import collections.abc
import dataclasses
import itertools
import math
import re
import typing

import formulaic
import numpy
import numpy.typing
import pandas

import metafopy.core.verbose
import metafopy.errors
import metafopy.methods.accessors
import metafopy.methods.emm_contrasts
import metafopy.methods.emm_links
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

if typing.TYPE_CHECKING:
    import metafopy.methods.emm_summary

__all__ = ['EmmGrid', 'contrast', 'emmeans', 'emmprep', 'pairs', 'regrid']

OVERALL_KEY = '1'
OVERALL_LABEL = 'overall'
PARENS = re.compile(r'-|\+|/|\*')

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Names = str | collections.abc.Sequence[str]
type Reduce = collections.abc.Callable[[FloatArray], float] | collections.abc.Mapping[str, collections.abc.Callable[[FloatArray], float]]
type Fit = (
    metafopy.models.rma_uni.RMAResult
    | metafopy.models.mv_result.RMAMVResult
    | metafopy.models.glmm_result.GLMMResult
    | metafopy.models.binary.FixedBinaryResult
    | metafopy.models.rma_ls_result.RMALSResult
    | metafopy.models.sel_result.SelmodelResult
)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class EmmGrid:
    """A reference grid, a set of estimated marginal means, or contrasts of them (emmeans' ``emmGrid``).

    ``grid`` has one row per estimate and one column per variable; the estimates are ``linfct @ bhat`` with the variance-covariance
    matrix ``linfct @ vb @ linfct.T`` and degrees of freedom ``df`` (infinite for z tests). ``wgt`` holds the number of studies
    behind each row of a reference grid, which the weights of :func:`emmeans` use. ``link`` is the transformation of the estimates
    (``type='response'`` of :meth:`summary` back-transforms them), ``inv_label`` the name of the back-transformed estimate and
    ``sigma`` the error SD used for the bias adjustment.
    """

    grid: pandas.DataFrame
    levels: dict[str, tuple[typing.Any, ...]]
    numeric: tuple[str, ...]
    linfct: FloatArray
    bhat: FloatArray
    vb: FloatArray
    df: float
    wgt: FloatArray | None
    est_name: str
    est_type: str
    by: tuple[str, ...]
    adjust: str
    fam_size: int
    infer: tuple[bool, bool]
    link: metafopy.methods.emm_links.Link | None
    inv_label: str | None
    log_contrast: bool
    sigma: float | None
    avgd_over: tuple[str, ...]
    messages: tuple[str, ...]
    pairby: tuple[str, ...] | None
    pri_vars: tuple[str, ...]
    level: float = 0.95
    measure: str = 'GEN'

    @property
    def estimate(self) -> FloatArray:
        """The estimates on the link scale."""
        return numpy.asarray(self.linfct @ self.bhat, dtype=numpy.float64)

    @property
    def vcov(self) -> FloatArray:
        """The variance-covariance matrix of the estimates."""
        return numpy.asarray(self.linfct @ self.vb @ self.linfct.T, dtype=numpy.float64)

    def summary(self, **kwargs: typing.Any) -> metafopy.methods.emm_summary.EmmSummary:
        """The estimates with standard errors, intervals and tests; see :func:`metafopy.methods.emm_summary.summary`."""
        # auditme: accept[in-function-import] emm_summary imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.emm_summary

        return metafopy.methods.emm_summary.summary(self, **kwargs)

    def confint(self, **kwargs: typing.Any) -> metafopy.methods.emm_summary.EmmSummary:
        """The summary with confidence intervals only."""
        return self.summary(infer=(True, False), **kwargs)

    def test(self, **kwargs: typing.Any) -> metafopy.methods.emm_summary.EmmSummary:
        """The summary with tests only."""
        return self.summary(infer=(False, True), **kwargs)

    def __str__(self) -> str:
        return str(self.summary())


@dataclasses.dataclass(frozen=True, slots=True)
class FitInfo:
    """What the reference grid needs from a fit."""

    formula: str
    beta: FloatArray
    vb: FloatArray
    coef_names: tuple[str, ...]
    ddf: float
    data: pandas.DataFrame
    measure: str
    sigma: float
    X: FloatArray


def fit_frame(x: Fit, data: pandas.DataFrame | None) -> pandas.DataFrame:
    """The rows of the data the model was fitted to (after ``subset`` and removal of missing values).

    Raises:
        metafopy.errors.InputError: the fit does not hold its data and none was given.
    """
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        return fit_frame(x.base, data)
    if isinstance(x, metafopy.models.binary.FixedBinaryResult):
        return data if data is not None else pandas.DataFrame(index=range(x.k))
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        frame = data if data is not None else metafopy.methods.accessors.call_arguments(x).get('data')
        if frame is None:
            if x.int_only:
                return pandas.DataFrame(index=range(x.k))
            raise metafopy.errors.InputError('Cannot use function when the model object does not contain the original data.')
        return (
            frame.iloc[list(x.ids)].loc[x.not_na.tolist()].reset_index(drop=True)
            if len(x.ids) != len(frame) or not x.not_na.all()
            else frame.reset_index(drop=True)
        )
    if isinstance(x, metafopy.models.glmm_result.GLMMResult):
        frame = data if data is not None else x.data
        if frame is None:
            if x.int_only:
                return pandas.DataFrame(index=range(x.k))
            raise metafopy.errors.InputError('Cannot use function when the model object does not contain the original data.')
        tab = x.tables
        if tab.subset is not None:
            frame = frame[numpy.asarray(tab.subset)]
        return pandas.DataFrame(frame.loc[numpy.asarray(tab.not_na).tolist()].reset_index(drop=True))
    meta = x.data
    frame = data if data is not None else meta.data
    if frame is None:
        if x.int_only:
            return pandas.DataFrame(index=range(x.k))
        raise metafopy.errors.InputError('Cannot use function when the model object does not contain the original data.')
    return pandas.DataFrame(frame.loc[meta.keep.tolist()].reset_index(drop=True))


def fit_ddf(x: Fit) -> float:
    ddf = getattr(x, 'ddf', None)
    if ddf is None:
        return math.inf
    values = numpy.atleast_1d(numpy.asarray(ddf, dtype=numpy.float64))
    return math.inf if bool(numpy.isnan(values[0])) else float(values.min())


def fit_info(x: Fit, data: pandas.DataFrame | None) -> FitInfo:
    metafopy.methods.accessors.check_fit(x, 'emmprep')
    base = x.base if isinstance(x, metafopy.models.sel_result.SelmodelResult) else x
    if any(getattr(base, 'coef_na', ())):
        raise metafopy.errors.InputError('Cannot use function when some redundant predictors were dropped from the model.')
    formula = metafopy.methods.accessors.formula_of(x)
    if formula is None and base.int_only:
        formula = '~ 1'
    if formula is None:
        raise metafopy.errors.InputError('Cannot use function when model was fitted without a formula specification.')
    beta = numpy.asarray(x.beta, dtype=numpy.float64).reshape(-1)
    vb = numpy.atleast_2d(numpy.asarray(x.vb, dtype=numpy.float64))
    names = tuple(base.coef_names)
    tau2 = getattr(x, 'tau2', math.nan)
    no_sigma = isinstance(x, metafopy.models.rma_ls_result.RMALSResult | metafopy.models.mv_result.RMAMVResult)
    sigma = 0.0 if no_sigma or not isinstance(tau2, float) or math.isnan(tau2) else math.sqrt(tau2)
    frame = fit_frame(x, data)
    X = numpy.ones((base.k, 1)) if isinstance(base, metafopy.models.binary.FixedBinaryResult) else numpy.asarray(base.X, dtype=numpy.float64)
    return FitInfo(formula, beta, vb, names, fit_ddf(x), frame, getattr(x, 'measure', 'GEN'), sigma, X)


def appearance(formula: str, variables: collections.abc.Iterable[str]) -> list[str]:
    """The variables in the order they first appear in the formula."""
    pos: dict[str, int] = {}
    for var in variables:
        hit = re.search(rf'(?<![\w.]){re.escape(var)}(?![\w.])', formula)
        pos[var] = hit.start() if hit is not None else len(formula)
    return sorted(pos, key=lambda v: pos[v])


def is_factor(var: str, formula: str, column: pandas.Series) -> bool:
    wrapped = re.search(rf'(?<![\w.])C\(\s*{re.escape(var)}\s*[,)]', formula) is not None
    return wrapped or not pandas.api.types.is_numeric_dtype(column) or pandas.api.types.is_bool_dtype(column)


def factor_levels(column: pandas.Series) -> list[typing.Any]:
    if isinstance(column.dtype, pandas.CategoricalDtype):
        return list(column.cat.categories)
    return sorted(column.dropna().unique().tolist())


def reduce_covariate(var: str, values: FloatArray, cov_reduce: Reduce | None) -> float:
    if cov_reduce is None:
        return float(numpy.mean(values))
    if isinstance(cov_reduce, collections.abc.Mapping):
        return float(cov_reduce[var](values)) if var in cov_reduce else float(numpy.mean(values))
    return float(cov_reduce(values))


def make_link(tran: metafopy.methods.emm_links.Link | str | None, measure: str) -> metafopy.methods.emm_links.Link | None:
    if tran is None:
        return metafopy.methods.emm_links.measure_link(measure)
    if isinstance(tran, metafopy.methods.emm_links.Link):
        return tran
    if tran == 'none':
        return None
    return metafopy.methods.emm_links.make_link(tran)


def emmprep(
    x: Fit,
    verbose: bool = False,
    *,
    data: pandas.DataFrame | None = None,
    df: float | None = None,
    sigma: float | None = None,
    tran: metafopy.methods.emm_links.Link | str | None = None,
    at: collections.abc.Mapping[str, typing.Any] | None = None,
    cov_reduce: Reduce | None = None,
) -> EmmGrid:
    """Prepare a reference grid from a fitted model, as metafor's ``emmprep`` does for emmeans' ``qdrg``.

    The grid has one row per combination of the levels of the factors in the model formula, with each numeric predictor at its
    mean over the studies (``cov_reduce``) or at the values given in ``at`` (a mapping from variable names to values, or, for
    factors, to subsets of the levels). ``df`` and ``sigma`` override the degrees of freedom (the smallest ``ddf`` of the fit,
    infinite for z tests) and the error SD used for bias adjustment (:math:`\\tau`, or 0 for ``rma_mv`` and ``rma_ls`` fits).
    ``tran`` overrides the transformation derived from the effect size measure (``'log'`` for ratio measures, ``'logit'``
    for ``PLO``, and so on; ``'none'`` for none).

    Raises:
        metafopy.errors.InputError: the fit cannot be used (redundant predictors, no formula or data) or an argument is invalid.
    """
    info = fit_info(x, data)
    ddf = info.ddf if df is None else float(df)
    s = info.sigma if sigma is None else float(sigma)
    link = make_link(tran, info.measure)
    names = tuple('(Intercept)' if n == 'intrcpt' else n for n in info.coef_names)
    if verbose:
        metafopy.core.verbose.cat(f'\nExtracted formula:  {info.formula}\n')
        if math.isfinite(ddf):
            metafopy.core.verbose.cat(f'Degrees of freedom: {round(ddf, 2)}\n')
        if s > 0:
            metafopy.core.verbose.cat(f'Value of tau^2:     {round(s**2, 4)}\n')
        if link is not None:
            metafopy.core.verbose.cat(f'Transformation:     {link.name}\n')
        metafopy.core.verbose.cat('\n')
    grid = reference_grid(info, names, at, cov_reduce)
    return dataclasses.replace(
        grid,
        df=ddf,
        sigma=s,
        link=link,
        inv_label='response' if link is not None else None,
        measure=info.measure,
    )


def reference_grid(info: FitInfo, names: tuple[str, ...], at: collections.abc.Mapping[str, typing.Any] | None, cov_reduce: Reduce | None) -> EmmGrid:
    rhs = info.formula.split('~', 1)[1].strip() if '~' in info.formula else info.formula.strip()
    data = info.data
    model_matrix = formulaic.model_matrix(rhs, data)
    spec = model_matrix.model_spec
    required = [] if spec is None else [v for v in spec.required_variables if v in data.columns]
    variables = appearance(rhs, required)
    at = {} if at is None else dict(at)
    unknown = [k for k in at if k not in variables]
    if unknown:
        raise metafopy.errors.InputError(f'Variables in `at` that are not in the model: {", ".join(unknown)}.')
    raw: dict[str, list[typing.Any]] = {}
    labels: dict[str, tuple[typing.Any, ...]] = {}
    covariates: list[str] = []
    for var in variables:
        column = data[var]
        if is_factor(var, rhs, column):
            levels = factor_levels(column)
            if var in at:
                keep = [at[var]] if isinstance(at[var], str) else list(at[var])
                missing = [v for v in keep if str(v) not in {str(lev) for lev in levels}]
                if missing:
                    raise metafopy.errors.InputError(f'Levels {missing} of {var!r} in `at` are not among its levels: {levels}.')
                levels = [lev for lev in levels if str(lev) in {str(v) for v in keep}]
            raw[var] = levels
            labels[var] = tuple(str(lev) for lev in levels)
        else:
            covariates.append(var)
            values = (
                numpy.atleast_1d(numpy.asarray(at[var], dtype=numpy.float64))
                if var in at
                else numpy.array([reduce_covariate(var, column.to_numpy(dtype=float), cov_reduce)])
            )
            raw[var] = values.tolist()
            labels[var] = tuple(float(v) for v in values)
    sizes = [len(raw[v]) for v in variables]
    index = list(itertools.product(*[range(s) for s in reversed(sizes)]))
    rows = [tuple(reversed(i)) for i in index] if variables else [()]
    pred = pandas.DataFrame({v: [raw[v][r[j]] for r in rows] for j, v in enumerate(variables)}, index=range(len(rows)))
    for var in variables:
        if var not in covariates:
            pred[var] = pandas.Categorical(pred[var], categories=raw[var])
    X = numpy.asarray(spec.get_model_matrix(pred).to_numpy(dtype=float), dtype=numpy.float64) if spec is not None else numpy.ones((len(rows), 1))
    fit_X = numpy.asarray(model_matrix.to_numpy(dtype=float), dtype=numpy.float64)
    if fit_X.shape != info.X.shape or not numpy.allclose(fit_X, info.X, rtol=1e-8, atol=1e-10):
        raise metafopy.errors.InputError('Cannot match the model matrix of the fit with that rebuilt from the formula and the data; the fit is not supported.')
    if X.shape[1] != info.beta.shape[0]:
        raise metafopy.errors.InputError('The number of coefficients does not match the model formula.')
    factor_vars = [v for v in variables if v not in covariates]
    counts: collections.Counter[tuple[typing.Any, ...]] = (
        collections.Counter(zip(*[data[v].tolist() for v in factor_vars], strict=True)) if factor_vars else collections.Counter()
    )
    wgt = numpy.array([float(counts[tuple(raw[v][r[j]] for j, v in enumerate(variables) if v in factor_vars)]) if factor_vars else 1.0 for r in rows])
    shown = pandas.DataFrame({v: [labels[v][r[j]] for r in rows] for j, v in enumerate(variables)}, index=range(len(rows)))
    return EmmGrid(
        grid=shown,
        levels={v: labels[v] for v in variables},
        numeric=tuple(covariates),
        linfct=X,
        bhat=info.beta,
        vb=info.vb,
        df=info.ddf,
        wgt=wgt,
        est_name='prediction',
        est_type='prediction',
        by=(),
        adjust='none',
        fam_size=len(rows),
        infer=(True, False),
        link=None,
        inv_label=None,
        log_contrast=False,
        sigma=info.sigma,
        avgd_over=(),
        messages=(),
        pairby=None,
        pri_vars=tuple(variables),
        measure=info.measure,
    )


def parse_specs(specs: Names, by: Names | None) -> tuple[list[str], list[str]]:
    """Variable names from ``'alloc'``, ``['alloc', 'lat']``, ``'~ alloc * lat | ablat'`` and the ``by`` argument."""
    by_vars = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if isinstance(specs, str):
        text = specs.split('~', 1)[1] if '~' in specs else specs
        main, _, after = text.partition('|')
        names = [s.strip() for s in re.split(r'[*+:,]', main) if s.strip()]
        by_vars += [s.strip() for s in re.split(r'[*+:,]', after) if s.strip()]
    else:
        names = list(specs)
    by_vars = [b for b in dict.fromkeys(by_vars) if b not in names]
    return names, by_vars


def emmeans(
    object: EmmGrid | Fit,
    specs: Names,
    by: Names | None = None,
    *,
    weights: str | collections.abc.Sequence[float] = 'equal',
    at: collections.abc.Mapping[str, typing.Any] | None = None,
    cov_reduce: Reduce | None = None,
    **prep: typing.Any,
) -> EmmGrid:
    """Estimated marginal means: the reference grid averaged over the variables not named in ``specs`` (and ``by``).

    ``object`` is a reference grid from :func:`emmprep` or a fit (``at``, ``cov_reduce`` and further keywords then go to
    :func:`emmprep`). ``specs`` names the variables (``'alloc'``, ``['alloc', 'lat']`` or an R-style ``'~ alloc | lat'``; ``'1'``
    or ``'~ 1'`` for the overall mean) and ``by`` the variables the results are grouped by. ``weights`` averages over
    the other factors with ``'equal'`` weights, ``'proportional'`` (frequencies in the data), ``'outer'`` (products of the
    marginal frequencies), ``'cells'`` or ``'flat'`` (cell frequencies, the latter ignoring empty cells), or a vector with one
    weight per combination of those factors.

    Raises:
        metafopy.errors.InputError: a variable is not in the grid or the weights do not fit.
    """
    if isinstance(object, EmmGrid):
        if at is not None or cov_reduce is not None or prep:
            raise metafopy.errors.InputError('Give `at`, `cov_reduce` and the other options to emmprep() when passing a reference grid.')
        rg = object
    else:
        rg = emmprep(object, at=at, cov_reduce=cov_reduce, **prep)
    names, by_vars = parse_specs(specs, by)
    facs = list(dict.fromkeys([*names, *by_vars]))
    overall = facs == [OVERALL_KEY]
    if overall:
        facs = []
    for f in facs:
        if f not in rg.levels:
            raise metafopy.errors.InputError(f'No variable named {f!r} in the reference grid.')
    variables = list(rg.levels)
    dims = [len(rg.levels[v]) for v in variables]
    row_idx = numpy.arange(rg.linfct.shape[0]).reshape(dims, order='F')
    use = [variables.index(f) for f in facs]
    avgd = [i for i, d in enumerate(dims) if d > 1 and i not in use]
    wgt = rg.wgt if rg.wgt is not None else numpy.ones(rg.linfct.shape[0])
    if numpy.allclose(numpy.round(wgt, 8), 0):
        wgt = wgt + 1
    avg_dims = [dims[i] for i in avgd]
    n_avg = int(numpy.prod(avg_dims)) if avgd else 1

    def margin(combo: tuple[int, ...]) -> FloatArray:
        index: list[typing.Any] = [slice(None)] * len(dims)
        for axis, pos in zip(use, combo, strict=True):
            index[axis] = pos
        return numpy.asarray(row_idx[tuple(index)], dtype=numpy.intp).reshape(-1, order='F')

    combos = [tuple(reversed(c)) for c in itertools.product(*[range(dims[i]) for i in reversed(use)])] if use else [()]
    weight_vec: FloatArray | None = None
    cell_mode: str | None = None
    if avgd:
        if isinstance(weights, str):
            options = ('equal', 'proportional', 'outer', 'cells', 'flat')
            hits = [o for o in options if o.startswith(weights)]
            if len(hits) != 1:
                raise metafopy.errors.InputError(f"Invalid 'weights' option: {weights!r}.")
            mode = hits[0]
            frequencies = numpy.reshape(wgt[row_idx.reshape(-1, order='F')], dims, order='F').sum(axis=tuple(i for i in range(len(dims)) if i not in avgd))
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            if mode == 'equal':
                weight_vec = numpy.ones(n_avg)
            elif mode == 'proportional':
                weight_vec = frequencies.reshape(-1, order='F')
            elif mode == 'outer':
                total = float(frequencies.sum())
                outer: FloatArray = numpy.array(total)
                for axis in range(len(avg_dims)):
                    margins = frequencies.sum(axis=tuple(a for a in range(len(avg_dims)) if a != axis)) / total
                    outer = numpy.multiply.outer(outer, margins)
                weight_vec = outer.reshape(-1, order='F')
            else:
                cell_mode = mode
        else:
            weight_vec = numpy.asarray(weights, dtype=numpy.float64)
            if weight_vec.shape != (n_avg,):
                raise metafopy.errors.InputError(f'Nonconforming number of weights -- need {n_avg}')
    rows_k: list[FloatArray] = []
    cell_wgt: list[float] = []
    for combo in combos:
        idx = margin(combo)
        coefs = rg.linfct[idx]
        w_cell = wgt[idx]
        if cell_mode is not None:
            fq = (w_cell > 0).astype(numpy.float64) if cell_mode == 'flat' else w_cell
            rows_k.append((coefs * fq[:, None]).sum(axis=0) / fq.sum())
        elif weight_vec is not None:
            total_w = float(weight_vec.sum())
            rows_k.append((coefs * weight_vec[:, None]).sum(axis=0) / (total_w if total_w > 1e-8 else math.nan))
        else:
            rows_k.append(coefs.mean(axis=0))
        cell_wgt.append(float(w_cell.sum()))
    levels = {f: rg.levels[f] for f in facs}
    if overall:
        levels = {OVERALL_KEY: (OVERALL_LABEL,)}
        grid = pandas.DataFrame({OVERALL_KEY: [OVERALL_LABEL]})
    else:
        grid = pandas.DataFrame({f: [rg.levels[f][c[j]] for c in combos] for j, f in enumerate(facs)}, index=range(len(combos)))
    avgd_over = tuple(dict.fromkeys([*rg.avgd_over, *[variables[i] for i in avgd]]))
    messages = list(rg.messages)
    return dataclasses.replace(
        rg,
        grid=grid,
        levels=levels,
        numeric=tuple(f for f in facs if f in rg.numeric),
        linfct=numpy.vstack(rows_k) if rows_k else numpy.zeros((0, rg.linfct.shape[1])),
        wgt=numpy.array(cell_wgt),
        est_name='emmean',
        est_type='prediction',
        by=tuple(by_vars),
        adjust='none',
        fam_size=len(combos),
        infer=(True, False),
        avgd_over=avgd_over,
        messages=tuple(messages),
        pri_vars=tuple(f for f in (levels if overall else facs) if f not in by_vars),
        pairby=None,
    )


def level_labels(grid: EmmGrid, columns: list[str], enhance: bool) -> list[str]:
    """The labels of the rows by pasting the values of ``columns`` (numeric covariates get their name in front, as emmeans does)."""
    out: list[str] = []
    for i in range(len(grid.grid)):
        parts: list[str] = []
        for c in columns:
            value = grid.grid[c].iloc[i]
            if c in grid.numeric and not isinstance(value, str):
                parts.append((c if enhance else '') + format(float(value), '.15g'))
            else:
                parts.append(str(value))
        out.append(' '.join(parts))
    return out


def group_rows(grid: EmmGrid, by: tuple[str, ...]) -> list[list[int]]:
    """Row positions of each combination of the ``by`` variables, in order of first appearance."""
    if not by:
        return [list(range(len(grid.grid)))]
    groups: dict[tuple[typing.Any, ...], list[int]] = {}
    for i in range(len(grid.grid)):
        groups.setdefault(tuple(grid.grid[b].iloc[i] for b in by), []).append(i)
    return list(groups.values())


def zap_true_contrast(coef: FloatArray) -> bool:
    sums = numpy.abs(coef.sum(axis=0) / (1e-06 + numpy.abs(coef).max(axis=0)))
    return bool(numpy.all(sums < 1e-06))


def contrast(
    object: EmmGrid,
    method: str | collections.abc.Mapping[str, collections.abc.Sequence[float]] = 'eff',
    *,
    by: Names | None = None,
    interaction: bool | str | collections.abc.Sequence[str] = False,
    name: str = 'contrast',
    adjust: str | None = None,
    ref: metafopy.methods.emm_contrasts.Key | None = None,
    exclude: metafopy.methods.emm_contrasts.Key | None = None,
    include: metafopy.methods.emm_contrasts.Key | None = None,
    reverse: bool = False,
    wts: collections.abc.Sequence[float] | None = None,
    max_degree: int | None = None,
    enhance_levels: bool = True,
) -> EmmGrid:
    """Contrasts of the estimates of a reference grid or of marginal means (emmeans' ``contrast``).

    ``method`` is one of :data:`metafopy.methods.emm_contrasts.METHODS` (``'pairwise'``, ``'revpairwise'``, ``'trt_vs_ctrl'``,
    ``'trt_vs_ctrlk'``, ``'dunnett'``, ``'eff'``, ``'del_eff'``, ``'consec'``, ``'mean_chg'``, ``'poly'``, ``'identity'``) or a
    mapping from contrast names to coefficient vectors, one per row of the grid. The contrasts are formed separately for each
    combination of the ``by`` variables (those of ``object`` by default). ``ref``, ``exclude`` and ``include`` are 0-based
    indices or level labels. With ``interaction`` (``True`` for ``method`` on every factor, or one method per factor) contrasts
    of contrasts are formed, one per combination of the factors.

    Raises:
        metafopy.errors.InputError: the method or an argument does not fit the grid.
    """
    by_vars = tuple(object.by if by is None else ([by] if isinstance(by, str) else list(by)))
    if interaction is not False:
        return interaction_contrast(
            object, method, interaction, by_vars, adjust, ref=ref, exclude=exclude, include=include, reverse=reverse, wts=wts, max_degree=max_degree
        )
    groups = group_rows(object, by_vars)
    if len({len(g) for g in groups}) > 1:
        raise metafopy.errors.InputError('`by` groups are of irregular size; currently not supported except with nested structures')
    columns = [c for c in object.grid.columns if c not in by_vars]
    first = groups[0]
    sub = dataclasses.replace(object, grid=object.grid.iloc[first].reset_index(drop=True))
    levs = [f'({lev})' if PARENS.search(lev) else lev for lev in level_labels(sub, columns, enhance_levels)]
    cset = metafopy.methods.emm_contrasts.build_contrasts(
        method, levs, ref=ref, exclude=exclude, include=include, reverse=reverse, wts=wts, max_degree=max_degree
    )
    if cset.coef.shape[0] != len(levs):
        raise metafopy.errors.InputError('Nonconforming number of contrast coefficients')
    n_con = len(cset.names)
    blocks = [cset.coef.T @ object.linfct[g] for g in groups]
    linfct = numpy.vstack(blocks)
    prev = [c for c in object.grid.columns if re.match(rf'^{re.escape(name)}[0-9]*$', str(c))]
    con_name = name if not prev else f'{name}{len(prev)}'
    frame_cols: dict[str, list[typing.Any]] = {con_name: [n for _ in groups for n in cset.names]}
    for b in by_vars:
        frame_cols[b] = [object.grid[b].iloc[g[0]] for g in groups for _ in range(n_con)]
    grid = pandas.DataFrame(frame_cols)
    est_type = cset.est_type if cset.est_type is not None else ('contrast' if bool(numpy.all(numpy.abs(cset.coef.sum(axis=0)) < 0.001)) else 'prediction')
    true_con = zap_true_contrast(cset.coef)
    link = object.link
    inv_label = object.inv_label
    log_contrast = False
    messages = list(object.messages)
    if link is not None:
        if true_con and link.name in (*metafopy.methods.emm_links.LOG_LIKE, 'logit'):
            log_contrast = True
            inv_label = 'odds.ratio' if link.name == 'logit' else 'ratio'
            link = metafopy.methods.emm_links.make_link('log') if link.name == 'logit' else link
        else:
            messages.append(
                f'Note: contrasts are still on the {link.name} scale. Consider using\n     regrid() if you want contrasts of back-transformed estimates.'
            )
            link = None
            inv_label = None
    return dataclasses.replace(
        object,
        grid=grid,
        levels={c: tuple(dict.fromkeys(grid[c].tolist())) for c in grid.columns},
        numeric=(),
        linfct=linfct,
        wgt=None,
        est_name='estimate',
        est_type=est_type,
        by=by_vars,
        adjust=adjust if adjust is not None else cset.adjust,
        fam_size=cset.fam_size if cset.fam_size is not None else len(first),
        infer=(False, True),
        link=link,
        inv_label=inv_label,
        log_contrast=log_contrast,
        sigma=None if true_con else object.sigma,
        messages=tuple(messages),
        pairby=by_vars if est_type == 'pairs' else None,
        pri_vars=tuple(c for c in grid.columns if c not in by_vars),
    )


def interaction_contrast(
    object: EmmGrid,
    method: str | collections.abc.Mapping[str, collections.abc.Sequence[float]],
    interaction: bool | str | collections.abc.Sequence[str],
    by: tuple[str, ...],
    adjust: str | None,
    **kwargs: typing.Any,
) -> EmmGrid:
    variables = [c for c in object.grid.columns if c not in by]
    k = len(variables)
    methods: list[typing.Any]
    if interaction is True:
        methods = [method] * k
    elif isinstance(interaction, str):
        methods = [interaction] * k
    elif isinstance(interaction, collections.abc.Sequence):
        methods = list(itertools.islice(itertools.cycle(interaction), k))
    else:
        raise metafopy.errors.InputError('`interaction` must be True, a method or a sequence of methods.')
    names = list(variables)
    result = object
    for i in range(k - 1, -1, -1):
        label = f'{names[i]}_{methods[i]}' if isinstance(methods[i], str) else f'{names[i]}_custom'
        others = tuple(c for c in result.grid.columns if c != names[i] and c not in ('.wgt.',))
        result = contrast(result, methods[i], by=others, name=label, enhance_levels=False, **kwargs)
        names[i] = label
    return dataclasses.replace(result, by=by, adjust=adjust if adjust is not None else 'none', pairby=None)


def pairs(object: EmmGrid, reverse: bool = False, *, by: Names | None = None, adjust: str | None = None) -> EmmGrid:
    """All pairwise differences of the estimates (``contrast(object, 'pairwise')``; ``reverse=True`` gives later minus earlier)."""
    return contrast(object, 'revpairwise' if reverse else 'pairwise', by=by, adjust=adjust)


def regrid(object: EmmGrid, *, bias_adjust: bool = False, sigma: float | None = None) -> EmmGrid:
    """Back-transform the estimates, so that later contrasts are of the back-transformed values (emmeans' ``regrid``).

    The variance-covariance matrix of the transformed estimates follows by the delta method. Without a transformation the grid
    is returned as is, with an identity ``linfct``.

    Raises:
        metafopy.errors.InputError: ``bias_adjust`` without a valid ``sigma``.
    """
    est = object.estimate
    vcov = object.vcov
    link = object.link
    grid = object.grid
    est_name = object.est_name
    if link is not None:
        if bias_adjust:
            s = object.sigma if sigma is None else sigma
            if s is None or s < 0 or math.isnan(s):
                raise metafopy.errors.InputError("Bias adjustment requires a valid 'sigma'.")
            link = metafopy.methods.emm_links.bias_adjusted(link, s)
        d = link.mu_eta(est)
        vcov = vcov * numpy.outer(d, d)
        est = link.linkinv(est)
        if object.inv_label is not None:
            est_name = object.inv_label if object.est_name == 'prediction' else est_name
            if object.log_contrast:
                grid = grid.apply(lambda col: col.str.replace(' - ', ' / ') if pandas.api.types.is_string_dtype(col) else col)
    n = est.shape[0]
    return dataclasses.replace(
        object,
        grid=grid,
        linfct=numpy.eye(n),
        bhat=est,
        vb=vcov,
        link=None,
        inv_label=None,
        log_contrast=False,
        sigma=None if object.link is not None else object.sigma,
        wgt=None,
        est_name=est_name,
    )
