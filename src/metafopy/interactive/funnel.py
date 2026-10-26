"""Interactive funnel chart.

Adapted from the funnel plots of ``docs/notebooks/showcase.ipynb``; the arithmetic is in :mod:`metafopy.interactive.data`. The matplotlib counterpart is
:func:`metafopy.funnel`, after https://wviechtb.github.io/metafor/reference/funnel.rma.html.
"""

import collections.abc
import typing

import altair
import numpy
import pandas

import metafopy
import metafopy.errors
import metafopy.interactive.data
import metafopy.interactive.forest
import metafopy.models.mv_result
import metafopy.models.rma_uni

__all__ = ['funnel_chart']

REGION_POINTS = 60
AXIS_MARGIN = 1.05


def funnel_chart(
    x: metafopy.models.rma_uni.RMAResult | metafopy.TrimfillResult | metafopy.interactive.data.ChartData,
    *,
    yaxis: typing.Literal['sei', 'vi'] = 'sei',
    level: float | None = None,
    href: collections.abc.Sequence[str | None] | None = None,
    slab: collections.abc.Sequence[str] | None = None,
    width: int = 420,
) -> altair.LayerChart:
    """Return an interactive funnel plot of a single-level model.

    The pseudo-confidence region is ``estimate +/- z(level) * sei`` with the y axis reversed, as metafor's ``funnel()`` draws it. Each observed study
    is a filled circle with a tooltip, a link to its ``href`` and a ``selection_point`` on ``slab``. A
    ``TrimfillResult`` draws its imputed studies (``fill``) as open circles after the observed ones. A
    [ChartData][metafopy.interactive.data.ChartData] draws its studies on the untransformed ``yi`` scale against the first pooled row's estimate, with
    its ``fill`` column marking imputed studies. The chart inlines its data and uses fixed colours.

    Args:
        x: An intercept-only single-level fit, a trim-and-fill result, or an untransformed ``ChartData``.
        yaxis: ``'sei'`` (standard error) or ``'vi'`` (sampling variance) on the y axis.
        level: Confidence level in percent of the region, default that of ``x`` (95 for a ``ChartData``).
        href: A link per study, for a fit (see [chart_data][metafopy.interactive.data.chart_data]).
        slab: Study labels, for a fit.
        width: Plot width in pixels.

    Returns:
        The chart; ``chart.to_dict()`` is the Vega-Lite specification.

    Raises:
        metafopy.errors.NotSupportedError: ``x`` is a multilevel fit, for which metafor's ``funnel()`` has no single-level pseudo-confidence region.
        metafopy.errors.InputError: ``x`` is a ``ChartData`` with no pooled row, with transformed intervals, or missing a column, or ``href`` or ``slab``
            is given with a ``ChartData``.
    """
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        raise metafopy.errors.NotSupportedError('funnel_chart() needs a single-level model; a multilevel fit has no single-level pseudo-confidence region.')
    if isinstance(x, metafopy.interactive.data.ChartData):
        if href is not None or slab is not None:
            raise metafopy.errors.InputError('Arguments href and slab apply to a fit; x is already a ChartData.')
        data = x
        metafopy.interactive.data.check_chart_data(data)
        if data.pooled.empty:
            raise metafopy.errors.InputError("ChartData.pooled has no row; the funnel is centred on the first pooled row's estimate.")
        if data.transformed:
            raise metafopy.errors.InputError('A funnel is drawn on the untransformed scale; the ChartData has transformed intervals and pooled estimates.')
        centre = float(data.pooled['estimate'].iloc[0])
        z = metafopy.interactive.data.zcrit(level)
    else:
        data = metafopy.interactive.data.chart_data(x, slab=slab, href=href, level=level)
        centre = float(numpy.ravel(x.beta)[0])
        z = metafopy.interactive.data.zcrit(level, x)
    studies = data.studies.copy()
    studies['y'] = studies['sei'] if yaxis == 'sei' else studies['vi']
    top = float(studies['y'].max()) * AXIS_MARGIN
    se_top = top if yaxis == 'sei' else float(numpy.sqrt(top))
    se = numpy.linspace(0, se_top, REGION_POINTS)
    region = pandas.DataFrame({'y': se if yaxis == 'sei' else se**2, 'lo': centre - z * se, 'hi': centre + z * se})
    y_title = 'Standard error' if yaxis == 'sei' else 'Sampling variance'
    y_scale = altair.Scale(reverse=True, domain=[0, top])
    x_scale = altair.Scale(zero=False)
    pick = altair.selection_point(fields=['slab'], on='click', clear='dblclick', toggle='true')
    dim = altair.condition(pick, altair.value(1), altair.value(0.3))
    tooltip = [
        altair.Tooltip('slab:N', title='Study'),
        altair.Tooltip('yi:Q', title=data.measure, format='.3f'),
        altair.Tooltip('sei:Q', title='SE', format='.3f'),
        altair.Tooltip('vi:Q', title='Variance', format='.4f'),
        altair.Tooltip('weight:Q', title='Weight %', format='.1f'),
    ]
    x_enc = altair.X('yi:Q', scale=x_scale, title=data.measure)
    y_enc = altair.Y('y:Q', scale=y_scale, title=y_title)
    shade = (
        altair.Chart(metafopy.interactive.forest.inline_data(region))
        .mark_area(color=metafopy.interactive.forest.REFLINE_COLOUR, opacity=0.15)
        .encode(x=altair.X('lo:Q', scale=x_scale, title=data.measure), x2='hi:Q', y=altair.Y('y:Q', scale=y_scale, title=y_title))
    )
    centre_line = (
        altair.Chart(altair.InlineData(values=[{'x': centre}]))
        .mark_rule(color=metafopy.interactive.forest.REFLINE_COLOUR)
        .encode(x=altair.X('x:Q', scale=x_scale))
    )
    observed = studies[~studies['fill'].astype(bool)]
    imputed = studies[studies['fill'].astype(bool)]
    observed_marks = (
        altair.Chart(metafopy.interactive.forest.inline_data(observed))
        .mark_circle(size=90, color=metafopy.interactive.forest.STUDY_COLOUR, stroke='white', strokeWidth=1)
        .encode(x=x_enc, y=y_enc, opacity=dim, tooltip=tooltip, href=altair.Href('href:N'))
        .add_params(pick)
    )
    layers = [shade, centre_line, observed_marks]
    if not imputed.empty:
        layers.append(
            altair.Chart(metafopy.interactive.forest.inline_data(imputed))
            .mark_point(shape='circle', filled=False, size=90, strokeWidth=1.5, color=metafopy.interactive.forest.POOLED_COLOUR)
            .encode(x=x_enc, y=y_enc, opacity=dim, tooltip=tooltip)
        )
    chart: altair.LayerChart = (
        altair.layer(*layers)
        .properties(width=width, height=int(width * 0.75))
        .configure_view(stroke=None)
        .configure_axis(labelColor=metafopy.interactive.forest.TEXT_COLOUR, titleColor=metafopy.interactive.forest.TEXT_COLOUR)
    )
    return chart
