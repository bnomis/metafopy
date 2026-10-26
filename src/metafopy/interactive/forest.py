"""Interactive forest chart.

Adapted from the forest plot of ``docs/notebooks/showcase.ipynb``; the arithmetic is in :mod:`metafopy.interactive.data`. The matplotlib counterpart is
:func:`metafopy.forest`, after https://wviechtb.github.io/metafor/reference/forest.rma.html.
"""

import collections.abc
import json
import typing

import altair
import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.interactive.data
import metafopy.models.mv_result
import metafopy.models.rma_uni

__all__ = ['forest_chart', 'inline_data']

STUDY_COLOUR = '#268BD2'
POOLED_COLOUR = '#DC322F'
REFLINE_COLOUR = '#93A1A1'
TEXT_COLOUR = '#3D3F44'
STUDY_ROW_HEIGHT = 22
POOLED_ROW_HEIGHT = 26


def inline_data(frame: pandas.DataFrame) -> altair.InlineData:
    """``frame`` as data inlined in the specification, a missing value (``None`` or NaN) leaving its field out of the row."""
    records = json.loads(frame.to_json(orient='records'))
    return altair.InlineData(values=[{k: v for k, v in record.items() if v is not None} for record in records])


def unique_labels(labels: collections.abc.Sequence[str]) -> list[str]:
    """``labels`` with a repeated label suffixed, so each study keeps its own row."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f'{label} ({seen[label]})')
    return out


def forest_chart(
    x: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult | metafopy.interactive.data.ChartData,
    *,
    transf: collections.abc.Callable[[numpy.typing.ArrayLike], numpy.ndarray] | None = None,
    xlab: str | None = None,
    refline: float | None = 0.0,
    order: typing.Literal['obs', 'weight', 'input'] = 'input',
    width: int = 560,
    **data_kwargs: typing.Any,
) -> altair.VConcatChart:
    """Return an interactive forest plot: a row per study, then the pooled rows.

    Each study is a rule from ``ci_lb`` to ``ci_ub`` and a square sized by weight; each pooled row is a diamond with its prediction interval as a thin
    rule. Hovering a study shows its estimate, interval and weight; clicking a linked study follows its ``href``, and clicking a study highlights it
    (``selection_point`` on ``slab``). When ``x`` is a fit, the remaining keyword arguments go to
    [chart_data][metafopy.interactive.data.chart_data]. The chart inlines its data and uses fixed colours, so it renders alike in any embedding.

    Args:
        x: An intercept-only fit, or a [ChartData][metafopy.interactive.data.ChartData] (hand-built or from ``chart_data()``).
        transf: Transformation of the estimates and intervals (for example ``numpy.exp``). Required for a ``ChartData`` whose intervals are
            transformed, so the study estimates are drawn on the same scale.
        xlab: Title of the x axis, default the effect size measure.
        refline: Position of the reference line on the transformed scale; ``None`` draws none.
        order: Order of the study rows: ``'input'`` (as given), ``'obs'`` (by estimate) or ``'weight'`` (largest weight first).
        width: Plot width in pixels.
        **data_kwargs: Passed to ``chart_data()`` when ``x`` is a fit (``slab``, ``href``, ``groups``, ``group_fits``, ``level``).

    Returns:
        The chart; ``chart.to_dict()`` is the Vega-Lite specification.

    Raises:
        metafopy.errors.InputError: ``x`` is a ``ChartData`` missing a column, or with transformed intervals and no ``transf``, or keyword arguments
            for ``chart_data()`` are given with a ``ChartData``.
    """
    if isinstance(x, metafopy.interactive.data.ChartData):
        if data_kwargs:
            raise metafopy.errors.InputError(f'Arguments {", ".join(sorted(data_kwargs))} apply to a fit; x is already a ChartData.')
        data = x
        metafopy.interactive.data.check_chart_data(data)
        if data.transformed and transf is None:
            raise metafopy.errors.InputError("A ChartData with transformed intervals needs the 'transf' argument to place the study estimates.")
    else:
        data = metafopy.interactive.data.chart_data(x, transf=transf, **data_kwargs)
    studies = data.studies.copy()
    yi = studies['yi'].to_numpy(dtype=float)
    studies['est'] = numpy.asarray(transf(yi), dtype=float) if transf is not None else yi
    studies['row'] = unique_labels([str(s) for s in studies['slab']])
    if order == 'obs':
        studies = studies.sort_values('est', kind='stable')
    elif order == 'weight':
        studies = studies.sort_values('weight', ascending=False, kind='stable')
    else:
        studies = studies.sort_values('order', kind='stable')
    rows = list(studies['row'])
    pooled = data.pooled.copy()
    pooled['est_text'] = [f'{e:.2f} [{lo:.2f}, {hi:.2f}]' for e, lo, hi in zip(pooled['estimate'], pooled['ci_lb'], pooled['ci_ub'], strict=True)]
    title = xlab if xlab is not None else data.measure
    free_x = altair.Scale(zero=False)
    pick = altair.selection_point(fields=['slab'], on='click', clear='dblclick', toggle='true')
    dim = altair.condition(pick, altair.value(1), altair.value(0.3))
    y_study = altair.Y('row:N', sort=rows, title=None, axis=altair.Axis(labelLimit=200))
    study_tooltip = [
        altair.Tooltip('slab:N', title='Study'),
        altair.Tooltip('est:Q', title='Estimate', format='.3f'),
        altair.Tooltip('ci_lb:Q', title='CI lower', format='.3f'),
        altair.Tooltip('ci_ub:Q', title='CI upper', format='.3f'),
        altair.Tooltip('weight:Q', title='Weight %', format='.1f'),
    ]
    base = altair.Chart(inline_data(studies))
    study_ci = base.mark_rule(strokeWidth=2, color=STUDY_COLOUR).encode(x=altair.X('ci_lb:Q', title=title, scale=free_x), x2='ci_ub:Q', y=y_study, opacity=dim)
    study_pt = (
        base.mark_square(color=STUDY_COLOUR)
        .encode(
            x=altair.X('est:Q', scale=free_x),
            y=y_study,
            size=altair.Size('weight:Q', scale=altair.Scale(range=[30, 300]), legend=None),
            opacity=dim,
            tooltip=study_tooltip,
            href=altair.Href('href:N'),
        )
        .add_params(pick)
    )
    layers = [study_ci, study_pt]
    if refline is not None:
        layers.insert(0, altair.Chart(altair.InlineData(values=[{'x': refline}])).mark_rule(color=REFLINE_COLOUR, strokeDash=[4, 3]).encode(x='x:Q'))
    study_chart = altair.layer(*layers).properties(width=width, height=STUDY_ROW_HEIGHT * max(len(rows), 1))
    y_pooled = altair.Y('label:N', sort=list(pooled['label']), title=None, axis=altair.Axis(labelLimit=200, labelFontWeight='bold'))
    pbase = altair.Chart(inline_data(pooled))
    pooled_tooltip = [
        altair.Tooltip('label:N', title='Pooled'),
        altair.Tooltip('estimate:Q', title='Estimate', format='.3f'),
        altair.Tooltip('ci_lb:Q', title='CI lower', format='.3f'),
        altair.Tooltip('ci_ub:Q', title='CI upper', format='.3f'),
        altair.Tooltip('pi_lb:Q', title='PI lower', format='.3f'),
        altair.Tooltip('pi_ub:Q', title='PI upper', format='.3f'),
        altair.Tooltip('k:Q', title='k'),
        altair.Tooltip('tau2:Q', title='tau^2', format='.3f'),
        altair.Tooltip('I2:Q', title='I^2 %', format='.1f'),
    ]
    pooled_pi = pbase.mark_rule(strokeWidth=1, color=POOLED_COLOUR).encode(x=altair.X('pi_lb:Q', title=title, scale=free_x), x2='pi_ub:Q', y=y_pooled)
    pooled_ci = pbase.mark_rule(strokeWidth=3, color=POOLED_COLOUR).encode(x='ci_lb:Q', x2='ci_ub:Q', y=y_pooled)
    pooled_pt = pbase.mark_point(shape='diamond', filled=True, size=160, color=POOLED_COLOUR).encode(x='estimate:Q', y=y_pooled, tooltip=pooled_tooltip)
    pooled_text = pbase.mark_text(align='left', dx=8, fontSize=11, color=TEXT_COLOUR).encode(x='pi_ub:Q', y=y_pooled, text='est_text:N')
    pooled_chart = altair.layer(pooled_pi, pooled_ci, pooled_pt, pooled_text).properties(width=width, height=POOLED_ROW_HEIGHT * len(pooled))
    chart: altair.VConcatChart = (
        altair.vconcat(study_chart, pooled_chart, spacing=4)
        .resolve_scale(x='shared')
        .configure_view(stroke=None)
        .configure_axis(labelColor=TEXT_COLOUR, titleColor=TEXT_COLOUR)
    )
    return chart
