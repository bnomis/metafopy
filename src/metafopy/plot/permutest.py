"""Plot of the permutation distributions of a permutation test (metafor's ``plot.permutest.rma.uni``).

Each panel is the histogram of the test statistics (or coefficients) of the permutations of one coefficient (or the omnibus test), with the
observed value as a dashed line, the tails beyond it shaded and, optionally, the theoretical null distribution and a kernel density
estimate. See https://wviechtb.github.io/metafor/reference/plot.permutest.rma.uni.html and Follmann & Proschan (1999),
doi:10.1111/j.0006-341X.1999.00151.x.
"""

import collections.abc
import dataclasses
import typing

import matplotlib.axes
import matplotlib.lines
import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.permutest
import metafopy.plot.draw
import metafopy.plot.histplot
import metafopy.plot.rutils

__all__ = ['PlotPermutestResult', 'plot_permutest']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class PlotPermutestResult:
    """The axes of the panels drawn, in order: the omnibus test (when ``QM=True``) and then one per coefficient."""

    axes: tuple[matplotlib.axes.Axes, ...]
    names: tuple[str, ...]


def select_coefficients(x: metafopy.methods.permutest.PermutestResult, beta: collections.abc.Sequence[int | str] | int | str | None) -> list[int]:
    """The 0-based positions of the coefficients to plot (all but the intercept by default)."""
    names = list(x.coef_names)
    if beta is None:
        if x.int_only:
            return [0]
        return list(range(1, x.p)) if x.int_incl else list(range(x.p))
    items = [beta] if isinstance(beta, int | str) else list(beta)
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            if item not in names:
                raise metafopy.errors.InputError(f"Could not find the coefficient '{item}'.")
            out.append(names.index(item))
        else:
            if item < 0 or item >= x.p:
                raise metafopy.errors.InputError('Coefficient positions must be between 0 and the number of coefficients minus one.')
            out.append(int(item))
    return out


def plot_permutest(
    x: metafopy.methods.permutest.PermutestResult,
    beta: collections.abc.Sequence[int | str] | int | str | None = None,
    QM: bool = False,
    breaks: object = 'Scott',
    freq: bool = False,
    col: str | None = None,
    border: str | None = None,
    col_out: str | None = None,
    col_ref: str | None = None,
    col_density: str | None = None,
    trim: float = 0.0,
    adjust: float = 1.0,
    lwd: float | collections.abc.Sequence[float] = (2, 0, 0, 4),
    legend: bool | str = False,
    *,
    alternative: str | None = None,
    p2defn: str | None = None,
    stat: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> PlotPermutestResult:
    """Plot the permutation distributions of a :func:`~metafopy.permutest` result (``plot.permutest.rma.uni``).

    ``beta`` selects the coefficients (names or 0-based positions; all but the intercept by default) and ``QM=True`` adds the omnibus test.
    ``breaks`` is the ``breaks`` argument of ``hist()``, ``freq`` draws counts instead of densities, ``trim`` drops that fraction of the most
    extreme permutations from the histogram, and ``lwd`` gives the widths of the dashed line at the observed value, of the theoretical
    null distribution (the density of the reference ``z``, ``t``, chi-square or ``F`` distribution), of the kernel density estimate of
    the permutation distribution (``adjust`` is its bandwidth adjustment) and of the vertical line at zero; a width of 0 leaves the element out.
    ``alternative``, ``p2defn`` and ``stat`` default to those of the test. ``legend`` (``True`` or a location) labels the two curves in the first panel.

    Returns a :class:`PlotPermutestResult` with the axes of the panels.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    if trim >= 0.5:
        raise metafopy.errors.InputError("The value of 'trim' must be < 0.5.")
    alt = x.alternative if alternative is None else alternative
    p2 = x.p2defn if p2defn is None else p2defn
    st = x.stat if stat is None else stat
    if alt not in ('two.sided', 'less', 'greater'):
        raise metafopy.errors.InputError("alternative must be 'two.sided', 'less' or 'greater'.")
    if p2 not in ('abs', 'px2'):
        raise metafopy.errors.InputError("p2defn must be 'abs' or 'px2'.")
    if st not in ('test', 'coef'):
        raise metafopy.errors.InputError("stat must be 'test' or 'coef'.")
    widths = [float(lwd)] * 3 + [4.0] if isinstance(lwd, int | float) else [float(v) for v in lwd]
    if len(widths) == 2:
        widths = [widths[0], widths[1], widths[1], 4.0]
    if len(widths) == 3:
        widths = [*widths, 4.0]
    if freq:
        widths[1] = widths[2] = 0.0
    axx = metafopy.plot.rutils.get_ax(ax)
    col_ = metafopy.plot.histplot.COL if col is None else col
    border_ = metafopy.plot.histplot.BORDER if border is None else border
    col_out_ = metafopy.plot.histplot.COL_OUT if col_out is None else col_out
    col_ref_ = metafopy.plot.histplot.COL_REF if col_ref is None else col_ref
    col_density_ = metafopy.plot.histplot.COL_DENSITY if col_density is None else col_density
    chosen = select_coefficients(x, beta)
    use_qm = QM and not x.int_only
    n_panels = len(chosen) + int(use_qm)
    if n_panels == 0:
        raise metafopy.errors.InputError('Must select at least one element to plot.')
    perm = x.zval_perm if st == 'test' else x.beta_perm
    observed = x.zval if st == 'test' else x.beta
    ddf = x.ddf
    drawn, context = metafopy.plot.histplot.panels(axx, n_panels)
    titles: list[str] = []
    legend_pending = [bool(legend)]

    def legend_for(panel: matplotlib.axes.Axes) -> None:
        # metafor's addlegend() sets legend to FALSE once drawn, so only the first panel has a legend
        if not legend_pending[0] or not any(w > 0 for w in widths[1:3]):
            return
        legend_pending[0] = False
        entries = [
            ('Kernel Density Estimate of\nthe Permutation Distribution', col_density_, widths[2]),
            ('Theoretical Null Distribution', col_ref_, widths[1]),
        ]
        shown = [(label, color, width) for label, color, width in entries if width > 0]
        handles = [matplotlib.lines.Line2D([], [], color=metafopy.plot.draw.rcolor(color), linewidth=width) for _, color, width in shown]
        location = {'topright': 'upper right', 'topleft': 'upper left', 'bottomright': 'lower right', 'bottomleft': 'lower left'}
        panel.legend(
            handles,
            [label for label, _, _ in shown],
            loc=typing.cast(typing.Any, location.get(legend, legend) if isinstance(legend, str) else 'upper right'),
            framealpha=1.0,
        )

    def reference_and_density(
        values: FloatArray, lower: float, upper: float, pdf: collections.abc.Callable[[FloatArray], FloatArray]
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        xs = numpy.linspace(lower, upper, 1000)
        dx, dy = metafopy.plot.rutils.density_r(values[numpy.isfinite(values)], adjust, 8192)
        return xs, numpy.asarray(pdf(xs), dtype=numpy.float64), dx, dy

    def tail_bars(panel: matplotlib.axes.Axes, h: metafopy.plot.histplot.Hist, val: float, side: str, mult: float = 1.0) -> None:
        metafopy.plot.histplot.draw_tail(panel, h, val, tail=side, mult=mult, col=col_out_, border=border_, freq=freq)

    def histogram(panel: matplotlib.axes.Axes, shown: FloatArray, ys: FloatArray, dy: FloatArray, main: str, xlab: str) -> metafopy.plot.histplot.Hist:
        ylim = None
        if widths[1] > 0 or widths[2] > 0:
            plain = metafopy.plot.histplot.histogram(shown, breaks)
            ylim = (0.0, float(max(float(ys.max()) if widths[1] > 0 else 0.0, float(dy.max()) if widths[2] > 0 else 0.0, float(plain.density.max()))))
        return metafopy.plot.histplot.draw_histogram(panel, shown, breaks, col=col_, border=border_, main=main, xlab=xlab, freq=freq, ylim=ylim)

    with context:
        panel_iter = iter(drawn)
        if use_qm:
            panel = next(panel_iter)
            pdist = x.QM_perm
            df1 = len(x.btt)
            top = max(float(scipy.stats.chi2.ppf(0.995, df1) if ddf is None else scipy.stats.f.ppf(0.995, df1, ddf)), float(numpy.nanmax(pdist)))
            xs, ys, dx, dy = reference_and_density(pdist, 0.0, top, scipy.stats.chi2(df1).pdf if ddf is None else scipy.stats.f(df1, ddf).pdf)
            shown = metafopy.plot.histplot.apply_trim(pdist, trim, two_sided=False)
            h = histogram(panel, shown, ys, dy, 'Omnibus Test of Coefficients', 'Value of Test Statistic')
            tail_bars(panel, h, float(x.QM), 'upper')
            if widths[0] > 0:
                metafopy.plot.draw.vline(panel, float(x.QM), lty='dashed', lw=widths[0])
            if widths[1] > 0:
                metafopy.plot.draw.line(panel, xs, ys, col=col_ref_, lw=widths[1])
            if widths[2] > 0:
                metafopy.plot.draw.line(panel, dx, dy, col=col_density_, lw=widths[2])
            legend_for(panel)
            titles.append('QM')
        for j in chosen:
            panel = next(panel_iter)
            pdist = numpy.asarray(perm.iloc[:, j], dtype=float)
            obs = float(observed[j])
            finite = pdist[numpy.isfinite(pdist)]
            crit = float(scipy.stats.norm.ppf(0.995) if ddf is None else scipy.stats.t.ppf(0.995, ddf))
            xs, ys, dx, dy = reference_and_density(
                pdist, min(-crit, float(finite.min())), max(crit, float(finite.max())), scipy.stats.norm.pdf if ddf is None else scipy.stats.t(ddf).pdf
            )
            shown = metafopy.plot.histplot.apply_trim(pdist, trim, two_sided=True)
            xlab = 'Value of Test Statistic' if st == 'test' else 'Value of Coefficient'
            h = histogram(panel, shown, ys, dy, '' if n_panels == 1 else f'Coefficient: {x.coef_names[j]}', xlab)
            if alt == 'two.sided':
                if p2 == 'abs':
                    tail_bars(panel, h, -abs(obs), 'lower')
                    tail_bars(panel, h, abs(obs), 'upper')
                    if widths[0] > 0:
                        for v in (-obs, obs):
                            metafopy.plot.draw.vline(panel, v, lty='dashed', lw=widths[0])
                elif obs > float(numpy.median(finite)):
                    tail_bars(panel, h, abs(obs), 'upper', 2.0)
                    if widths[0] > 0:
                        metafopy.plot.draw.vline(panel, obs, lty='dashed', lw=widths[0])
                else:
                    tail_bars(panel, h, -abs(obs), 'lower', 2.0)
                    if widths[0] > 0:
                        metafopy.plot.draw.vline(panel, -abs(obs), lty='dashed', lw=widths[0])
            else:
                tail_bars(panel, h, obs, 'lower' if alt == 'less' else 'upper')
                if widths[0] > 0:
                    metafopy.plot.draw.vline(panel, obs, lty='dashed', lw=widths[0])
            if widths[1] > 0:
                metafopy.plot.draw.line(panel, xs, ys, col=col_ref_, lw=widths[1])
            if widths[2] > 0:
                metafopy.plot.draw.line(panel, dx, dy, col=col_density_, lw=widths[2])
            if widths[3] > 0:
                metafopy.plot.draw.vline(panel, 0.0, lw=widths[3])
            legend_for(panel)
            titles.append(x.coef_names[j])
    return PlotPermutestResult(tuple(drawn), tuple(titles))
