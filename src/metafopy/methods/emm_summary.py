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

import metafopy.core.labelled
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

    Attributes:
        frame: The table, a data frame.
        est_name: Name of the estimate column.
        cl_names: Names of the lower and upper confidence limit columns.
        by: Variables the results are grouped by.
        adjust: Multiplicity adjustment used.
        type: The scale shown, ``'link'`` or ``'response'``.
        messages: Notes printed under the table.
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
        return format_summary(self)


TEST_STATISTICS = ('t.ratio', 'z.ratio', 'F.ratio', 'T.square')
PVAL_DIGITS = 4


def r_zapsmall(x: FloatArray, digits: int = 7) -> FloatArray:
    """R's ``zapsmall()``; an infinite value makes it round to whole numbers, as it does in R."""
    present = x[~numpy.isnan(x)]
    if present.size == 0:
        return x
    mx = float(numpy.max(numpy.abs(present)))
    places = max(0, digits - math.ceil(math.log10(mx))) if mx > 0 and math.isfinite(mx) else (0 if mx > 0 else digits)
    return numpy.asarray(numpy.round(x, places), dtype=numpy.float64)


def opt_dig(row: FloatArray) -> int:
    """emmeans' ``.opt.dig()``: the digits that show the interval of two standard errors around the estimate."""
    if not numpy.all(numpy.isfinite(row)) or numpy.max(numpy.abs(row)) == 0:
        return 3
    z = numpy.array([numpy.min(row), numpy.max(row)]) / numpy.max(numpy.abs(row))
    pair = numpy.array([z[1] - z[0], z[0], z[1]])
    dz = r_zapsmall(pair, 8)[0]
    if dz == 0:
        return 3
    return max(round(1.51 - math.log10(dz)), 3)


def format_summary(res: EmmSummary) -> str:
    """``print.summary_emm``: estimates shown with the digits that carry two standard errors, test statistics to 3 and p values to 4 decimals, the rows of
    each ``by`` group under a ``by = level:`` line, and the notes under the table (https://github.com/rvlenth/emmeans/blob/master/R/summary.R)."""
    frame = res.frame.reset_index(drop=True)
    names = [str(c) for c in frame.columns]
    numeric = {c for c in names if pandas.api.types.is_float_dtype(frame[c]) or pandas.api.types.is_integer_dtype(frame[c])}
    values: dict[str, FloatArray] = {c: r_zapsmall(frame[c].to_numpy(dtype=numpy.float64)) for c in names if c in numeric}
    if 'df' in values:
        values['df'] = numpy.round(values['df'], 2)
    text: dict[str, list[str]] = {}
    for stat in TEST_STATISTICS:
        if stat in values:
            text[stat] = metafopy.core.labelled.format_real(list(numpy.round(values[stat], 3)), nsmall=3)
    if 'p.value' in values:
        p = values['p.value']
        shown = metafopy.core.labelled.format_real(list(numpy.round(p, PVAL_DIGITS)), nsmall=PVAL_DIGITS)
        floor = 10.0**-PVAL_DIGITS
        text['p.value'] = ['NA' if math.isnan(v) else (f'<{floor:.{PVAL_DIGITS}f}' if v < floor else txt) for v, txt in zip(p, shown, strict=True)]
    est = values.get(res.est_name)
    digits: int | None = None
    if est is not None:
        if 'SE' in values:
            values['SE'] = numpy.array([float(f'{v:.3g}') if math.isfinite(v) else v for v in values['SE']])
            rows = est[:, None] + values['SE'][:, None] * numpy.array([-2.0, 0.0, 2.0])
        elif res.cl_names and res.cl_names[0].startswith('lower.'):
            rows = numpy.column_stack([values[res.cl_names[0]], est, values[res.cl_names[1]]])
        else:
            rows = None
        if rows is not None:
            digits = max((opt_dig(r) for r in rows), default=3)
    cells: dict[str, list[str]] = {}
    for c in names:
        if c in text:
            cells[c] = text[c]
        elif c in numeric:
            cells[c] = metafopy.core.labelled.format_real(list(values[c]), digits if digits is not None else metafopy.core.labelled.SIGNIF)
        else:
            cells[c] = [str(v) for v in frame[c]]
    if est is not None:
        cells[res.est_name] = ['nonEst' if math.isnan(v) else txt for v, txt in zip(est, cells[res.est_name], strict=True)]
    # format.data.frame pads the columns to a common width: numbers on the left, labels on the right
    for c in names:
        w = max((len(v) for v in cells[c]), default=0)
        cells[c] = [v.rjust(w) if c in numeric else v.ljust(w) for v in cells[c]]
    shown_names = [c for c in names if c not in res.by]
    width = {c: max(len(c), *(len(v) for v in cells[c])) if cells[c] else len(c) for c in names}

    def block(rows: list[int]) -> list[str]:
        head = ''.join(' ' + (c.rjust(width[c]) if c in numeric else c.ljust(width[c])) for c in shown_names)
        body = [''.join(' ' + (cells[c][i].rjust(width[c]) if c in numeric else cells[c][i].ljust(width[c])) for c in shown_names) for i in rows]
        return [head, *body]

    lines: list[str] = []
    everything = list(range(len(frame)))
    if not res.by:
        lines.extend(block(everything))
        lines.append('')
    else:
        keys = []
        for b in res.by:
            col = frame[b]
            keys.append(col.to_numpy(dtype=numpy.float64) if b in numeric else pandas.Index(pandas.unique(col)).get_indexer(pandas.Index(col)))
        order = [int(i) for i in numpy.lexsort(keys)]
        seen: dict[str, list[int]] = {}
        for i in order:
            seen.setdefault(', '.join(str(frame[b].iloc[i]) for b in res.by), []).append(i)
        for rows in seen.values():
            lines.append(', '.join(f'{b} = {cells[b][rows[0]]}' for b in res.by) + ':')
            lines.extend(block(rows))
            lines.append('')
    lines.extend(f'{m} ' for m in res.messages)
    return '\n'.join(lines)


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
    null: float | None = None,
    delta: float | None = None,
    side: int | float | str | None = None,
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

    ``side`` makes the tests and intervals one-sided (``'<'`` or ``'left'``, ``'>'`` or ``'right'``; see :func:`metafopy.methods.emm_adjust.parse_side`) and
    ``delta`` is a threshold: with both sides it gives an equivalence test of :math:`|\\hat\\theta - \\theta_0| < \\delta`, with
    ``side='<'`` a non-superiority and with ``side='>'`` a non-inferiority test (Schuirmann, 1987,
    https://doi.org/10.1007/BF01068419). The Tukey, Scheffe and Dunnett adjustments become Sidak for one-sided inferences, as in
    emmeans. Without them the values stored on the object by :func:`emmeans` or :func:`contrast` apply.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    chosen = (object.summary_type or 'link') if type is None else type
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
    null_value = object.null if null is None else float(null)
    delta_value = abs(object.delta if delta is None else float(delta))
    side_code = metafopy.methods.emm_adjust.parse_side(object.side if side is None else side)
    tail = side_code if side_code != 0 else (-1 if delta_value > 0 else 0)
    shown = numpy.arange(object.linfct.shape[0]) if object.display is None else numpy.flatnonzero(numpy.asarray(object.display, dtype=bool))
    grid = object.grid.iloc[shown].reset_index(drop=True)
    linfct = object.linfct[shown]
    n = len(grid)
    est = numpy.asarray(linfct @ object.bhat, dtype=numpy.float64)
    vcov = numpy.asarray(linfct @ object.vb @ linfct.T, dtype=numpy.float64)
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
    groups = metafopy.methods.emm_grid.group_frame(grid, by_vars) if by_vars else [list(range(n))]
    sizes = {len(g) for g in groups}
    by_size = n // len(groups) if len(sizes) == 1 else n
    ranks = tuple(int(numpy.linalg.matrix_rank(zapsmall(linfct[g]))) for g in groups) if method.lower().startswith('sch') else tuple(1 for _ in groups)
    fam = metafopy.methods.emm_adjust.Family(
        float(object.fam_size),
        by_size,
        {'prediction': 1, 'contrast': 2, 'pairs': 3}.get(est_type, 2),
        tuple(tuple(g) for g in groups),
        ranks,
        None,
        len(sizes) > 1 or len(set(ranks)) > 1,
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
        adj_ci = metafopy.methods.emm_adjust.critical_values(est, lev, dfs, method, fam, side_code)
        cv = adj_ci.values
        cl_names = ('asymp.LCL', 'asymp.UCL') if z_flag else ('lower.CL', 'upper.CL')
        lower = est - cv * se if side_code >= 0 else numpy.full(n, -numpy.inf)
        upper = est + cv * se if side_code <= 0 else numpy.full(n, numpy.inf)
        messages.extend([f'Confidence level used: {lev:g}'])
        if adj_ci.message:
            messages.append(adj_ci.message)
        messages.extend(adj_ci.notes)
        if inv and out_link is not None:
            with numpy.errstate(all='ignore'):
                clims = numpy.column_stack([out_link.linkinv(lower), out_link.linkinv(upper)])
            diffs = numpy.where(numpy.isnan(clims[:, 1] - clims[:, 0]), 0, clims[:, 1] - clims[:, 0])
            flip = not bool(numpy.all(diffs >= 0))
            lower, upper = (clims[:, 1], clims[:, 0]) if flip else (clims[:, 0], clims[:, 1])
            messages.append(f'Intervals are back-transformed from the {link.name if link is not None else ""} scale')
        result[cl_names[0]] = lower
        result[cl_names[1]] = upper
    if do_test:
        tnm = 'z.ratio' if z_flag else 't.ratio'
        null_col = out_link.linkinv(numpy.full(n, null_value)) if inv and out_link is not None else numpy.full(n, null_value)
        if not numpy.all(null_col == 0):
            result['null'] = null_col
        if side_code == 0:
            t_ratio = (est - null_value) / se if delta_value == 0 else (numpy.abs(est - null_value) - delta_value) / se
        else:
            t_ratio = (est - null_value + side_code * delta_value) / se
        result[tnm] = t_ratio
        adj_p = metafopy.methods.emm_adjust.p_values(t_ratio, dfs, method, fam, tail)
        adj_method = adj_p.adjust
        result['p.value'] = adj_p.values
        pvals = adj_p.values.copy()
        if adj_p.message:
            messages.append(adj_p.message)
        messages.extend(adj_p.notes)
        cross = cross_adjust
        if len(groups) > 1 and len(sizes) == 1 and cross.lower() != 'none':
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
        if delta_value > 0:
            kind = ('nonsuperiority', 'equivalence', 'noninferiority')[side_code + 1]
            messages.append(f'Statistics are tests of {kind} with a threshold of {delta_value:.5g}')
        if tail != 0:
            messages.append(f'P values are {"left" if tail < 0 else "right"}-tailed')
        if inv:
            messages.append(f'Tests are performed on the {link.name if link is not None else ""} scale')
    if inv and out_link is not None:
        result['SE'] = numpy.abs(out_link.mu_eta(est) * se)
        result[est_name] = out_link.linkinv(est)
        if bias:
            messages.append(f'Bias adjustment applied based on sigma = {s:.5g}')
    if object.avgd_over:
        messages.insert(0, f'Results are averaged over{object.avgd_qualifier} the levels of: {", ".join(object.avgd_over)}')
    frame = pandas.concat([labels.reset_index(drop=True), pandas.DataFrame(result)], axis=1)
    return EmmSummary(frame, est_name, cl_names, by_vars, adj_method, chosen, tuple(dict.fromkeys(messages)))
