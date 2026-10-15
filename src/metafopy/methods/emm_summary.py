"""Summaries of reference grids, marginal means and contrasts (emmeans' ``summary.emmGrid``).

R source: https://github.com/rvlenth/emmeans/blob/master/R/summary.R; reference:
https://rvlenth.github.io/emmeans/reference/summary.emmGrid.html.
"""

import collections.abc
import dataclasses
import math

import numpy
import numpy.typing
import pandas

import metafopy.core.padjust
import metafopy.errors
import metafopy.methods.emm_adjust
import metafopy.methods.emm_grid
import metafopy.methods.emm_links

__all__ = ['EmmSummary', 'summary']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Infer = bool | collections.abc.Sequence[bool]


@dataclasses.dataclass(frozen=True, slots=True)
class EmmSummary:
    """The table of a summary with the notes emmeans prints under it (``messages``).

    ``frame`` has a column for each variable, the estimate (``emmean``, ``estimate``, ``response``, ``ratio``, ...), ``SE``,
    ``df``, the confidence limits (``lower.CL``/``upper.CL``, or ``asymp.LCL``/``asymp.UCL`` for z tests), ``null``, the
    ``t.ratio`` (``z.ratio``) and the ``p.value``, as far as they were asked for.
    """

    frame: pandas.DataFrame
    est_name: str
    cl_names: tuple[str, ...]
    by: tuple[str, ...]
    adjust: str
    type: str
    messages: tuple[str, ...]

    def to_frame(self) -> pandas.DataFrame:
        return self.frame.copy()

    def __str__(self) -> str:
        frame = self.frame
        shown = frame.copy()
        for col in shown.columns:
            if pandas.api.types.is_float_dtype(shown[col]):
                shown[col] = [fmt_value(v, col) for v in shown[col]]
        text = shown.to_string(index=False) if not self.by else '\n\n'.join(group_text(shown, self.by))
        notes = '\n'.join(self.messages)
        return f'{text}\n\n{notes}' if notes else text


def fmt_value(value: float, column: str) -> str:
    if math.isnan(value):
        return 'NA'
    if column == 'p.value':
        return '<.0001' if value < 0.0001 else f'{value:.4f}'
    if math.isinf(value):
        return 'Inf'
    return f'{value:.4g}' if abs(value) < 1e4 else f'{value:.0f}'


def group_text(frame: pandas.DataFrame, by: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key, part in frame.groupby(list(by), sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        head = ', '.join(f'{b} = {k}' for b, k in zip(by, keys, strict=True))
        out.append(f'{head}:\n{part.drop(columns=list(by)).to_string(index=False)}')
    return out


def zapsmall(x: FloatArray, digits: int = 7) -> FloatArray:
    mx = numpy.nanmax(numpy.abs(x)) if x.size else 0.0
    if not mx > 0:
        return numpy.round(x, digits)
    return numpy.round(x, max(0, digits - math.ceil(math.log10(mx))))


def infer_pair(infer: Infer | None, default: tuple[bool, bool]) -> tuple[bool, bool]:
    if infer is None:
        return default
    if isinstance(infer, bool):
        return (infer, infer)
    values = list(infer)
    if len(values) == 1:
        return (bool(values[0]), bool(values[0]))
    return (bool(values[0]), bool(values[1]))


def summary(
    object: metafopy.methods.emm_grid.EmmGrid,
    *,
    infer: Infer | None = None,
    level: float | None = None,
    adjust: str | None = None,
    by: str | collections.abc.Sequence[str] | None = None,
    cross_adjust: str = 'none',
    type: str | None = None,
    df: float | None = None,
    null: float = 0.0,
    bias_adjust: bool = False,
    sigma: float | None = None,
) -> EmmSummary:
    """Estimates with standard errors, confidence intervals and tests.

    A reference grid or marginal means show estimates and intervals by default, contrasts estimates and tests; ``infer`` is
    ``(intervals, tests)`` or a single flag for both. ``adjust`` is the multiplicity adjustment (``'tukey'``, ``'sidak'``,
    ``'bonferroni'``, ``'scheffe'``, ``'dunnettx'``, ``'mvt'`` or any method of ``p_adjust``), applied within each combination of
    the ``by`` variables; ``cross_adjust`` adjusts the p-values across those combinations. ``type='response'`` back-transforms
    estimates and intervals (the tests stay on the link scale) and ``bias_adjust`` applies the second-order adjustment based on
    ``sigma``. ``null`` is the null value of the tests on the link scale and ``df`` overrides the degrees of freedom.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    chosen = 'link' if type is None else type
    if chosen not in ('link', 'response'):
        raise metafopy.errors.InputError("'type' must be 'link' or 'response'.")
    lev = object.level if level is None else level
    do_ci, do_test = infer_pair(infer, object.infer)
    method = object.adjust if adjust is None else adjust
    by_vars = tuple(object.by if by is None else ([by] if isinstance(by, str) else list(by)))
    est_type = object.est_type
    if est_type == 'pairs' and object.pairby != by_vars:
        est_type = 'contrast'
    deg = object.df if df is None else float(df)
    s = object.sigma if sigma is None else float(sigma)
    grid = object.grid
    n = len(grid)
    est = object.estimate
    vcov = object.vcov
    se = numpy.sqrt(numpy.diag(vcov))
    dfs = numpy.full(n, deg)
    z_flag = bool(numpy.all(numpy.isinf(dfs) | numpy.isnan(dfs)))
    link = object.link
    inv = chosen == 'response' and link is not None
    messages: list[str] = list(object.messages)
    if not inv and link is not None:
        messages.append(f'Results are given on the {link.name} (not the response) scale.')
    bias = False
    out_link = link
    if inv and bias_adjust and link is not None:
        if s is None or s < 0 or math.isnan(s):
            messages.append('Bias adjustment skipped: No valid sigma provided')
        else:
            out_link = metafopy.methods.emm_links.bias_adjusted(link, s)
            bias = True
    groups = metafopy.methods.emm_grid.group_rows(object, by_vars) if by_vars else [list(range(n))]
    by_size = n // len(groups) if len({len(g) for g in groups}) == 1 else n
    fam = metafopy.methods.emm_adjust.Family(
        float(object.fam_size),
        by_size,
        {'prediction': 1, 'contrast': 2, 'pairs': 3}.get(est_type, 2),
        tuple(tuple(g) for g in groups),
        tuple(int(numpy.linalg.matrix_rank(zapsmall(object.linfct[g]))) for g in groups) if method.lower().startswith('sch') else tuple(1 for _ in groups),
        None,
    )
    if method.lower().startswith('mvt') or method.lower() == 'mvt':
        sd = numpy.sqrt(numpy.diag(vcov))
        fam = dataclasses.replace(fam, corr=vcov / numpy.outer(sd, sd))
    est_name = object.est_name
    if inv and object.inv_label is not None:
        est_name = object.inv_label
    elif inv:
        est_name = 'response'
    labels = grid.copy()
    if inv and object.log_contrast:
        labels = labels.apply(lambda col: col.str.replace(' - ', ' / ') if pandas.api.types.is_string_dtype(col) else col)
    result: dict[str, FloatArray] = {est_name: est, 'SE': se, 'df': dfs}
    cl_names: tuple[str, ...] = ()
    adj_method = method
    adj_ci = None
    if do_ci:
        adj_ci = metafopy.methods.emm_adjust.critical_values(est, lev, dfs, method, fam)
        cv = adj_ci.values
        cl_names = ('asymp.LCL', 'asymp.UCL') if z_flag else ('lower.CL', 'upper.CL')
        lower = est - cv * se
        upper = est + cv * se
        messages.extend([f'Confidence level used: {lev:g}'])
        if adj_ci.message:
            messages.append(adj_ci.message)
        messages.extend(adj_ci.notes)
        if inv and out_link is not None:
            clims = numpy.column_stack([out_link.linkinv(lower), out_link.linkinv(upper)])
            diffs = numpy.where(numpy.isnan(clims[:, 1] - clims[:, 0]), 0, clims[:, 1] - clims[:, 0])
            flip = not bool(numpy.all(diffs >= 0))
            lower, upper = (clims[:, 1], clims[:, 0]) if flip else (clims[:, 0], clims[:, 1])
            messages.append(f'Intervals are back-transformed from the {link.name if link is not None else ""} scale')
        result[cl_names[0]] = lower
        result[cl_names[1]] = upper
    if do_test:
        tnm = 'z.ratio' if z_flag else 't.ratio'
        null_col = out_link.linkinv(numpy.full(n, null)) if inv and out_link is not None else numpy.full(n, null)
        if not numpy.all(null_col == 0):
            result['null'] = null_col
        t_ratio = (est - null) / se
        result[tnm] = t_ratio
        adj_p = metafopy.methods.emm_adjust.p_values(t_ratio, dfs, method, fam)
        adj_method = adj_p.adjust
        result['p.value'] = adj_p.values
        pvals = adj_p.values.copy()
        if adj_p.message:
            messages.append(adj_p.message)
        messages.extend(adj_p.notes)
        cross = cross_adjust
        if len(groups) > 1 and len({len(g) for g in groups}) == 1 and cross.lower() != 'none':
            mat = numpy.column_stack([pvals[g] for g in groups])
            adjusted = numpy.empty_like(mat)
            for i in range(mat.shape[0]):
                if cross.lower() == 'sidak':
                    adjusted[i] = 1 - (1 - mat[i]) ** mat.shape[1]
                else:
                    adjusted[i] = metafopy.core.padjust.p_adjust(mat[i], cross)
            for j, g in enumerate(groups):
                pvals[g] = adjusted[:, j]
            result['p.value'] = pvals
            messages.append(f'Cross-group P-value adjustment: {cross} method for {mat.shape[1]} tests')
        if inv:
            messages.append(f'Tests are performed on the {link.name if link is not None else ""} scale')
    if inv and out_link is not None:
        result['SE'] = numpy.abs(out_link.mu_eta(est) * se)
        result[est_name] = out_link.linkinv(est)
        if bias:
            messages.append(f'Bias adjustment applied based on sigma = {s:.5g}')
    if object.avgd_over:
        messages.insert(0, f'Results are averaged over the levels of: {", ".join(object.avgd_over)}')
    frame = pandas.concat([labels.reset_index(drop=True), pandas.DataFrame(result)], axis=1)
    return EmmSummary(frame, est_name, cl_names, by_vars, adj_method, chosen, tuple(dict.fromkeys(messages)))
