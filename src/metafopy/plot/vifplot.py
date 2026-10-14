"""Plot of simulated (G)VIFs (metafor's ``plot.vif.rma``).

For each coefficient (or set of coefficients) the histogram of the (G)VIFs simulated under independence of the moderators, with the
observed (G)VIF as a dashed line and the tail beyond it shaded; the share of simulated values beyond the observed one is its p-value.
See https://wviechtb.github.io/metafor/reference/plot.vif.rma.html and Beisbart et al. (2024), doi:10.1002/jrsm.1706.
"""

import collections.abc
import dataclasses

import matplotlib.axes
import numpy
import numpy.typing

import metafopy.errors
import metafopy.methods.vif
import metafopy.plot.draw
import metafopy.plot.histplot
import metafopy.plot.rutils

__all__ = ['PlotVifResult', 'plot_vif']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class PlotVifResult:
    """The axes of the panels drawn, one per coefficient (or set of coefficients) of the result."""

    axes: tuple[matplotlib.axes.Axes, ...]
    names: tuple[str, ...]


def plot_vif(
    x: metafopy.methods.vif.VIFResult,
    breaks: object = 'Scott',
    freq: bool = False,
    col: str | None = None,
    border: str | None = None,
    col_out: str | None = None,
    col_density: str | None = None,
    trim: float = 0.0,
    adjust: float = 1.0,
    lwd: float | collections.abc.Sequence[float] = (2, 0),
    *,
    tail: str = 'upper',
    ax: matplotlib.axes.Axes | None = None,
) -> PlotVifResult:
    """Plot the simulated (G)VIFs of a :func:`~metafopy.vif` result obtained with ``sim=True`` (``plot.vif.rma``).

    ``breaks`` is the ``breaks`` argument of ``hist()``, ``freq`` draws counts instead of densities, ``trim`` drops that fraction of the
    largest simulated values from the histogram and ``lwd`` gives the widths of the dashed line at the observed (G)VIF and of the kernel
    density estimate (of bandwidth adjustment ``adjust``); a width of 0 leaves the element out. A (G)VIF beyond the right edge of the
    plot is shown at the edge, with an arrow.

    Returns a :class:`PlotVifResult` with the axes of the panels.

    Raises:
        metafopy.errors.InputError: the result has no simulated values (``sim=True`` was not used) or an argument is invalid.
    """
    if x.sim is None:
        raise metafopy.errors.InputError("Can only plot 'vif' objects when 'sim=True' was used.")
    if trim >= 0.5:
        raise metafopy.errors.InputError("The value of 'trim' must be < 0.5.")
    if tail not in ('lower', 'upper'):
        raise metafopy.errors.InputError("tail must be 'lower' or 'upper'.")
    widths = [float(lwd), float(lwd)] if isinstance(lwd, int | float) else [float(v) for v in lwd][:2]
    if freq:
        widths[1] = 0.0
    axx = metafopy.plot.rutils.get_ax(ax)
    col_ = metafopy.plot.histplot.COL if col is None else col
    border_ = metafopy.plot.histplot.BORDER if border is None else border
    col_out_ = metafopy.plot.histplot.COL_OUT if col_out is None else col_out
    col_density_ = metafopy.plot.histplot.COL_DENSITY if col_density is None else col_density
    n_panels = len(x.vif)
    drawn, context = metafopy.plot.histplot.panels(axx, n_panels)
    with context:
        for i, (panel, entry) in enumerate(zip(drawn, x.vif, strict=True)):
            sims = numpy.asarray(x.sim[:, i], dtype=float)
            sims = sims[numpy.isfinite(sims)]
            dx, dy = metafopy.plot.rutils.density_r(sims, adjust)
            shown = metafopy.plot.histplot.apply_trim(sims, trim, two_sided=False)
            # as in metafor, the vertical limits come from the densities even when counts are drawn (the bars are then cut off)
            plain = metafopy.plot.histplot.histogram(shown, breaks)
            ylim = (0.0, float(max(float(dy.max()) if widths[1] > 0 else 0.0, float(plain.density.max()))))
            main = f'Coefficient{"s" if entry.m > 1 else ""}: {x.names[i]}'
            h = metafopy.plot.histplot.draw_histogram(panel, shown, breaks, col=col_, border=border_, main=main, xlab='Value of VIF', freq=freq, ylim=ylim)
            metafopy.plot.histplot.draw_tail(panel, h, float(entry.vif), tail=tail, col=col_out_, border=border_, freq=freq)
            lo, hi = panel.get_xlim()
            ylo, yhi = panel.get_ylim()
            value = float(entry.vif)
            if value > hi and widths[0] > 0:
                ticks = metafopy.plot.rutils.ax_ticks(ylo, yhi)
                ya = float(ticks.min() + ticks.max()) / 2
                height_inches = panel.get_position().height * panel.figure.bbox.height / panel.figure.dpi
                head = 0.02 * height_inches * 72.0
                panel.annotate(
                    '',
                    xy=(hi - 0.01 * (hi - lo), ya),
                    xytext=(hi - 0.08 * (hi - lo), ya),
                    arrowprops={'arrowstyle': '->,head_length=1.0,head_width=0.577', 'mutation_scale': head, 'color': 'black'},
                    annotation_clip=False,
                )
            if widths[0] > 0:
                line = metafopy.plot.draw.seg(panel, min(value, hi), ylo, min(value, hi), yhi, lty='dashed', lw=widths[0], clip=False)
                line.set_clip_on(False)
            if widths[1] > 0:
                metafopy.plot.draw.line(panel, dx, dy, col=col_density_, lw=widths[1])
    return PlotVifResult(tuple(drawn), tuple(x.names))
