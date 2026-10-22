"""Plot of the permutation distributions of a permutation test (metafor's ``plot.permutest.rma.uni``).

Each panel is the histogram of the test statistics (or coefficients) of the permutations of one coefficient (or the omnibus test), with the
observed value as a dashed line, the tails beyond it shaded and, optionally, the theoretical null distribution and a kernel density
estimate. See https://wviechtb.github.io/metafor/reference/plot.permutest.rma.uni.html and Follmann & Proschan (1999),
doi:10.1111/j.0006-341X.1999.00732.x.
"""

import collections.abc
import dataclasses
import typing
import warnings

import matplotlib.axes
import matplotlib.lines
import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.ls_permutest
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


def select_coefficients(
    x: typing.Any,
    beta: collections.abc.Sequence[int | str] | int | str | None,
    *,
    names: collections.abc.Sequence[str] | None = None,
    p: int | None = None,
    int_only: bool | None = None,
    int_incl: bool | None = None,
) -> list[int]:
    """The 0-based positions of the coefficients to plot (all but the intercept by default).

    ``names``, ``p``, ``int_only`` and ``int_incl`` default to those of the location coefficients of ``x``.

    Raises:
        metafopy.errors.InputError: a name or position does not match a coefficient.
    """
    names = list(x.coef_names if names is None else names)
    p = x.p if p is None else p
    int_only = x.int_only if int_only is None else int_only
    int_incl = x.int_incl if int_incl is None else int_incl
    if beta is None:
        if int_only:
            return [0]
        return list(range(1, p)) if int_incl else list(range(p))
    items = [beta] if isinstance(beta, int | str) else list(beta)
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            if item not in names:
                raise metafopy.errors.InputError(f"Could not find the coefficient '{item}'.")
            out.append(names.index(item))
        else:
            if item < 0 or item >= p:
                raise metafopy.errors.InputError('Coefficient positions must be between 0 and the number of coefficients minus one.')
            out.append(int(item))
    return out


def plot_permutest(
    x: metafopy.methods.permutest.PermutestResult | metafopy.methods.ls_permutest.LSPermutestResult,
    beta: collections.abc.Sequence[int | str] | int | str | None = None,
    alpha: collections.abc.Sequence[int | str] | int | str | None = None,
    QM: bool = False,
    QS: bool = False,
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
    xlim: collections.abc.Sequence[float] | None = None,
    layout: object = None,
) -> PlotPermutestResult:
    """Plot the permutation distributions of a [permutest][metafopy.permutest] result (``plot.permutest.rma.uni``).

    Mirrors metafor's ``plot.permutest.rma.uni()``, see <https://wviechtb.github.io/metafor/reference/plot.permutest.rma.uni.html>.

    One panel is drawn per selected element, in the order of metafor: the omnibus test of the location coefficients (``QM=True``), the
    location coefficients, the omnibus test of the scale coefficients (``QS=True``) and the scale coefficients. Each panel is the histogram of
    the permuted test statistics (or coefficients), the observed value is a dashed line and the tails beyond it are filled with ``col_out``.
    The theoretical null distribution and the kernel density estimate are omitted by default (their ``lwd`` is 0). A single-coefficient plot
    has no panel title.

    Trimming only affects the histogram; the kernel density estimate uses all permutations.

    For an omnibus test with a single degree of freedom (``QM=True`` and one coefficient tested) the theoretical density is infinite at 0. That
    point is left out of the curve and of the y range, whereas metafor stops with an infinite ``ylim`` there.

    Args:
        x: The result of [permutest][metafopy.permutest]; a location-scale permutation test gives ``LSPermutestResult``.
        beta: Location coefficients to plot, as names or 0-based positions (R: 1-based). A single name or position is allowed. Default
            ``None``: all coefficients but the intercept (the intercept only when it is the only coefficient). ``[]`` selects none.
        alpha: Scale coefficients to plot, selected as ``beta``. Only used for ``LSPermutestResult``; default ``None``: all but the intercept.
        QM: Add the panel of the omnibus test of the location coefficients (ignored for an intercept-only model). Default: ``False``.
        QS: Add the panel of the omnibus test of the scale coefficients (``LSPermutestResult`` only). Default: ``False``.
        breaks: The ``breaks`` argument of R's ``hist()``: a number of bins, a rule name or a vector of break points. Default: ``'Scott'``.
        freq: Draw counts instead of densities. Forces ``lwd[1]`` and ``lwd[2]`` to 0, so no curves are drawn. Default: ``False``.
        col: Fill colour of the histogram bars. Default: ``'#B3B3B3'`` (light grey).
        border: Border colour of the bars. Default: ``'#E5E5E5'`` (lighter grey).
        col_out: Fill colour of the bars in the tails beyond the observed value. Default: ``'#FF000080'`` (semi-transparent red).
        col_ref: Colour of the theoretical null distribution. Default: ``'#4C4C4C'`` (dark grey). The dashed line at the observed value and the
            vertical line at zero are always black.
        col_density: Colour of the kernel density estimate. Default: ``'blue'``.
        trim: Fraction of the permutations dropped from the histogram, which must be below 0.5. The omnibus panels drop the upper ``trim``
            fraction, the coefficient panels drop ``trim / 2`` from each tail. Default: 0.
        adjust: Bandwidth adjustment of the kernel density estimate, as the ``adjust`` argument of R's ``density()``. Default: 1.
        lwd: Line widths of the dashed line at the observed value, of the theoretical null distribution, of the kernel density estimate and
            of the vertical line at zero (coefficient panels only). A width of 0 leaves the element out. A single number is used for the
            first three and 4 for the last; two or three values are completed to four (the second is repeated for the third, the last
            defaults to 4). Default: ``(2, 0, 0, 4)``.
        legend: ``True`` or a location (for example ``'topleft'``) for a legend naming the curves; drawn in the first panel only and
            only when a curve is drawn. ``True`` places it at the top right. Default: ``False``.
        alternative: ``'two.sided'``, ``'greater'`` or ``'less'``. Default: that of the test.
        p2defn: Definition of the two-sided p-value, ``'abs'`` or ``'px2'``. Default: that of the test.
        stat: Statistic plotted, ``'test'`` (z or t values) or ``'coef'`` (coefficients). Default: that of the test.
        ax: Matplotlib axes to draw into; the panels are laid out inside it. Default: the current axes.
        xlim: Limits of the horizontal axis of every panel. Default: the range of the histogram breaks.
            The plotted axis extends 4 percent beyond the limits, as in R.
        layout: Deprecated and ignored; a warning is issued, as in metafor.

    Returns:
        A ``PlotPermutestResult`` with the fields ``axes`` (the axes of the panels drawn) and ``names`` (``'QM'``, ``'QS'`` or the
        coefficient name of each panel, in the same order).

    Raises:
        metafopy.errors.InputError: ``trim`` is 0.5 or more; ``alternative``, ``p2defn`` or ``stat`` is not a valid value; a coefficient
            name or position in ``beta`` or ``alpha`` does not exist; no panel is selected.
    """
    if layout is not None:
        warnings.warn("Argument 'layout' has been deprecated.", stacklevel=2)
    if trim >= 0.5:
        raise metafopy.errors.InputError("The value of 'trim' must be < 0.5.")
    alt = x.alternative if alternative is None else alternative
    p2 = x.p2defn if p2defn is None else p2defn
    st = x.stat if stat is None else stat
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if alt not in ('two.sided', 'less', 'greater'):
        raise metafopy.errors.InputError("alternative must be 'two.sided', 'less' or 'greater'.")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if p2 not in ('abs', 'px2'):
        raise metafopy.errors.InputError("p2defn must be 'abs' or 'px2'.")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
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
    is_ls = isinstance(x, metafopy.methods.ls_permutest.LSPermutestResult)
    skip_beta = is_ls and typing.cast(metafopy.methods.ls_permutest.LSPermutestResult, x).skip_beta
    chosen = [] if skip_beta else select_coefficients(x, beta)
    use_qm = QM and not x.int_only and not skip_beta
    chosen_alpha: list[int] = []
    use_qs = False
    ls = typing.cast(metafopy.methods.ls_permutest.LSPermutestResult, x)
    if is_ls and not ls.skip_alpha:
        chosen_alpha = select_coefficients(ls, alpha, names=ls.alpha_names, p=ls.q, int_only=ls.Z_int_only, int_incl=ls.Z_int_incl)
        use_qs = QS
    n_panels = len(chosen) + int(use_qm) + len(chosen_alpha) + int(use_qs)
    if n_panels == 0:
        raise metafopy.errors.InputError('Must select at least one element to plot.')
    perm = x.zval_perm if st == 'test' else x.beta_perm
    observed = x.zval if st == 'test' else x.beta
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
        panel.legend(
            handles,
            [label for label, _, _ in shown],
            loc=typing.cast(typing.Any, metafopy.plot.draw.LEGEND_POSITIONS.get(legend, legend) if isinstance(legend, str) else 'upper right'),
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
            # the chi-square density with one degree of freedom is infinite at 0: the y range ignores that point (R stops with an infinite ylim)
            finite_ys = ys[numpy.isfinite(ys)]
            ylim = (
                0.0,
                float(
                    max(
                        float(finite_ys.max()) if widths[1] > 0 and finite_ys.size else 0.0,
                        float(dy.max()) if widths[2] > 0 else 0.0,
                        float(plain.density.max()),
                    )
                ),
            )
        return metafopy.plot.histplot.draw_histogram(panel, shown, breaks, col=col_, border=border_, main=main, xlab=xlab, freq=freq, ylim=ylim, xlim=xlim)

    def omnibus(panel: matplotlib.axes.Axes, pdist: FloatArray, obs: float, df1: int, df2: int | None, main: str) -> None:
        top = max(float(scipy.stats.chi2.ppf(0.995, df1) if df2 is None else scipy.stats.f.ppf(0.995, df1, df2)), float(numpy.nanmax(pdist)))
        xs, ys, dx, dy = reference_and_density(pdist, 0.0, top, scipy.stats.chi2(df1).pdf if df2 is None else scipy.stats.f(df1, df2).pdf)
        shown = metafopy.plot.histplot.apply_trim(pdist, trim, two_sided=False)
        h = histogram(panel, shown, ys, dy, main, 'Value of Test Statistic')
        tail_bars(panel, h, obs, 'upper')
        if widths[0] > 0:
            metafopy.plot.draw.vline(panel, obs, lty='dashed', lw=widths[0])
        if widths[1] > 0:
            metafopy.plot.draw.line(panel, xs[numpy.isfinite(ys)], ys[numpy.isfinite(ys)], col=col_ref_, lw=widths[1])
        if widths[2] > 0:
            metafopy.plot.draw.line(panel, dx, dy, col=col_density_, lw=widths[2])
        legend_for(panel)

    def coefficient(panel: matplotlib.axes.Axes, pdist: FloatArray, obs: float, ddf: int | None, main: str) -> None:
        finite = pdist[numpy.isfinite(pdist)]
        crit = float(scipy.stats.norm.ppf(0.995) if ddf is None else scipy.stats.t.ppf(0.995, ddf))
        xs, ys, dx, dy = reference_and_density(
            pdist, min(-crit, float(finite.min())), max(crit, float(finite.max())), scipy.stats.norm.pdf if ddf is None else scipy.stats.t(ddf).pdf
        )
        shown = metafopy.plot.histplot.apply_trim(pdist, trim, two_sided=True)
        xlab = 'Value of Test Statistic' if st == 'test' else 'Value of Coefficient'
        h = histogram(panel, shown, ys, dy, '' if n_panels == 1 else main, xlab)
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
            metafopy.plot.draw.line(panel, xs[numpy.isfinite(ys)], ys[numpy.isfinite(ys)], col=col_ref_, lw=widths[1])
        if widths[2] > 0:
            metafopy.plot.draw.line(panel, dx, dy, col=col_density_, lw=widths[2])
        if widths[3] > 0:
            metafopy.plot.draw.vline(panel, 0.0, lw=widths[3])
        legend_for(panel)

    location = 'Location Coefficient: ' if is_ls else 'Coefficient: '
    with context:
        panel_iter = iter(drawn)
        if use_qm:
            omnibus(
                next(panel_iter),
                x.QM_perm,
                float(x.QM),
                len(x.btt),
                x.ddf,
                'Omnibus Test of Location Coefficients' if is_ls else 'Omnibus Test of Coefficients',
            )
            titles.append('QM')
        for j in chosen:
            coefficient(next(panel_iter), numpy.asarray(perm.iloc[:, j], dtype=float), float(observed[j]), x.ddf, f'{location}{x.coef_names[j]}')
            titles.append(x.coef_names[j])
        if use_qs:
            omnibus(next(panel_iter), ls.QS_perm, float(ls.QS), len(ls.att), ls.ddf_alpha, 'Omnibus Test of Scale Coefficients')
            titles.append('QS')
        for j in chosen_alpha:
            perm2 = ls.zval_alpha_perm if st == 'test' else ls.alpha_perm
            obs2 = ls.zval_alpha if st == 'test' else ls.alpha
            coefficient(next(panel_iter), numpy.asarray(perm2.iloc[:, j], dtype=float), float(obs2[j]), ls.ddf_alpha, f'Scale Coefficient: {ls.alpha_names[j]}')
            titles.append(ls.alpha_names[j])
    return PlotPermutestResult(tuple(drawn), tuple(titles))
