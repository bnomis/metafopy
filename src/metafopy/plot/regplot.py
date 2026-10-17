"""Meta-analytic scatterplot (metafor's ``regplot.rma``) and its ``points`` method (``points.regplot``).

Plots the observed effects against one moderator of a meta-regression model, with the size of each point given by its
weight (or a precision), the fitted regression line and, optionally, the pointwise confidence and prediction bands.
See https://wviechtb.github.io/metafor/reference/regplot.html, https://github.com/wviechtb/metafor/blob/master/R/regplot.rma.r,
https://github.com/wviechtb/metafor/blob/master/R/points.regplot.r and Thompson & Higgins (2002), doi:10.1002/sim.1187.
"""

import collections.abc
import dataclasses
import math
import re
import typing
import warnings

import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.predict
import metafopy.models.rma_uni
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils

__all__ = ['RegplotResult', 'points_regplot', 'regplot']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Func = collections.abc.Callable[..., typing.Any]

LEGEND_LOC = {'topright': 'upper right', 'topleft': 'upper left', 'bottomright': 'lower right', 'bottomleft': 'lower left'}


@dataclasses.dataclass(frozen=True, slots=True)
class RegplotResult:
    """The plotted points (what metafor's ``regplot()`` returns invisibly) and the axes drawn on.

    ``xi`` and ``yi`` are the coordinates of the studies drawn (``yi`` after any transformation), ``psize`` the
    symbol sizes (the ``cex`` of the points), ``label`` which studies were labelled, ``pred`` the fitted values and
    ``order`` the order in which the points were drawn (largest symbols first). ``ids`` are the 0-based positions of the studies among
    the rows supplied, ``pch``, ``col`` and ``bg`` the symbols and colours of the points, and ``offset`` and ``labsize`` those of the
    labels, so that :func:`points_regplot` can redraw the points and labels (metafor's ``points()`` of a ``regplot`` object).
    """

    ax: matplotlib.axes.Axes
    xi: FloatArray
    yi: FloatArray
    psize: FloatArray
    pred: FloatArray | None
    label: numpy.typing.NDArray[numpy.bool_]
    slab: tuple[str, ...]
    order: numpy.typing.NDArray[numpy.intp]
    offset: tuple[float, ...]
    labsize: float
    ids: numpy.typing.NDArray[numpy.intp]
    pch: tuple[int, ...]
    col: tuple[typing.Any, ...]
    bg: tuple[typing.Any, ...]

    def to_frame(self) -> pandas.DataFrame:
        out = pandas.DataFrame(
            {
                'slab': list(self.slab),
                'ids': self.ids,
                'xi': self.xi,
                'yi': self.yi,
                'pch': list(self.pch),
                'psize': self.psize,
                'col': list(self.col),
                'bg': list(self.bg),
                'label': self.label,
            }
        )
        if self.pred is not None:
            out['pred'] = self.pred
        out['order'] = self.order
        return out

    def points(self, ax: matplotlib.axes.Axes | None = None) -> None:
        """Redraw the points and labels; see :func:`points_regplot`."""
        points_regplot(self, ax=ax)


def expand(v: typing.Any, n: int, name: str) -> list[typing.Any]:
    """An argument that is one value or one value per study (``.expand1()`` and the length check of metafor)."""
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * n
    out = list(v)
    if len(out) != n:
        raise metafopy.errors.InputError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the number of studies ({n}).")
    return out


def find_moderator(
    x: metafopy.methods.fitview.FitView, names: collections.abc.Sequence[str], mod: str | int | None, grep: bool = False, fixed: bool = False
) -> int:
    """The 0-based position of the moderator among the coefficients (metafor's ``mod``, with its ``grep`` and ``fixed`` arguments).

    A string is matched against the coefficient names by ``charmatch()`` (exactly, or as the start of a single name); with ``grep=True`` it is
    searched for in the names as a regular expression (``fixed=True``: as a literal string) and must be found in exactly one.
    """
    p = x.X.shape[1]
    if mod is None:
        if p == 2 and x.int_incl:
            return 1
        if p == 1:
            return 0
        raise metafopy.errors.InputError("Must specify the 'mod' argument for models with multiple predictors.")
    if isinstance(mod, str):
        if grep:
            found = [i for i, name in enumerate(names) if (mod in name if fixed else re.search(mod, name))]
        elif mod in names:
            found = [list(names).index(mod)]
        else:
            found = [i for i, name in enumerate(names) if name.startswith(mod)]
        if len(found) != 1:
            raise metafopy.errors.InputError("Could not find or uniquely identify the moderator variable specified via the 'mod' argument.")
        return found[0]
    pos = int(mod)
    if pos < 0 or pos >= p:
        raise metafopy.errors.InputError("Specified position of 'mod' variable does not exist in the model.")
    return pos


def symbol_sizes(psize: typing.Any, weights: FloatArray, vi: FloatArray, plim: collections.abc.Sequence[float]) -> FloatArray:
    """The ``cex`` of the points: the given sizes, or the weights (or ``'seinv'``/``'vinv'``) rescaled to ``plim``."""
    k = vi.shape[0]
    if psize is not None and not isinstance(psize, str):
        return numpy.asarray(expand(psize, k, 'psize'), dtype=float)
    if len(plim) < 2:
        raise metafopy.errors.InputError("Argument 'plim' must be of length 2 or 3.")
    if psize is None:
        wi = numpy.sqrt(weights)
    elif psize == 'seinv':
        wi = 1 / numpy.sqrt(vi)
    elif psize == 'vinv':
        wi = 1 / vi
    else:
        raise metafopy.errors.InputError("Argument 'psize' must be 'seinv', 'vinv' or numeric.")
    lo, hi = plim[0], plim[1]
    out = numpy.full(k, numpy.nan)
    if not math.isnan(lo) and not math.isnan(hi):
        rng = float(numpy.nanmax(wi) - numpy.nanmin(wi))
        out = numpy.ones(k) if rng <= math.sqrt(numpy.finfo(float).eps) else (wi - numpy.nanmin(wi)) / rng * (hi - lo) + lo
    elif math.isnan(lo) and not math.isnan(hi):
        out = wi / numpy.nanmax(wi) * hi
        if len(plim) == 3:
            out[out <= plim[2]] = plim[2]
    elif not math.isnan(lo) and math.isnan(hi):
        out = wi / numpy.nanmin(wi) * lo
        if len(plim) == 3:
            out[out >= plim[2]] = plim[2]
    return numpy.ones(k) if bool(numpy.all(numpy.isnan(out))) else out


def sorted_pair(lb: FloatArray, ub: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``.psort()``: the smaller of each pair first (transformations may reverse the order)."""
    return numpy.minimum(lb, ub), numpy.maximum(lb, ub)


def regplot(
    x: metafopy.methods.fitview.Fit,
    mod: str | int | None = None,
    pred: bool | metafopy.methods.predict.PredictResult = True,
    ci: bool = True,
    pi: bool = False,
    shade: bool | str | collections.abc.Sequence[str] = True,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    predlim: collections.abc.Sequence[float] | None = None,
    olim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    at: numpy.typing.ArrayLike | None = None,
    digits: float | collections.abc.Sequence[float] = 2,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    pch: int | collections.abc.Sequence[int] = 21,
    psize: float | str | collections.abc.Sequence[float] | None = None,
    plim: collections.abc.Sequence[float] = (0.5, 3.0),
    col: str | collections.abc.Sequence[str] | None = None,
    bg: str | collections.abc.Sequence[str] | None = None,
    slab: collections.abc.Sequence[typing.Any] | None = None,
    grid: bool | str = False,
    refline: float | None = None,
    label: bool | str | collections.abc.Sequence[typing.Any] = False,
    offset: float | collections.abc.Sequence[float] = (1.0, 1.0),
    labsize: float = 1.0,
    lcol: str | collections.abc.Sequence[str] | None = None,
    lwd: float | collections.abc.Sequence[float] | None = None,
    lty: str | int | collections.abc.Sequence[str | int] | None = None,
    legend: bool | str | dict[str, typing.Any] = False,
    xvals: numpy.typing.ArrayLike | None = None,
    *,
    bty: str = 'o',
    las: int = 0,
    grep: bool = False,
    fixed: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    xaxt: str = 's',
    na_action: metafopy.methods.common.NAAction = 'omit',
    at_lab: collections.abc.Sequence[str] | None = None,
    box_lty: str = 'solid',
) -> RegplotResult:
    """Meta-analytic scatterplot of a meta-regression model (``regplot.rma``).

    ``mod`` is the moderator on the horizontal axis (its name, or its 0-based position among the coefficients; it may
    be left out when the model has a single moderator). The regression line (``pred``) and the confidence (``ci``) and
    prediction (``pi``) bands are drawn for the other moderators set to their means; ``xvals`` gives the moderator values
    to predict at, and ``pred`` may be a :class:`~metafopy.methods.predict.PredictResult` for those values. ``transf``
    transforms the effects and the bands, ``atransf`` only the labels of the vertical axis, and ``olim`` limits the
    values drawn. The points have the ``cex`` ``psize`` (or the square roots of the weights rescaled to ``plim``; give
    ``'seinv'`` or ``'vinv'`` to scale the inverse standard errors or variances instead). ``label`` labels the studies
    (``True``/``'all'``, ``'ciout'``/``'piout'`` for those outside the bands, a boolean array or 0-based indices), with
    ``offset`` the distance from the point in percent of the vertical range and ``labsize`` the text size. ``legend``
    adds a legend (``True``, a location such as ``'topright'`` or a dict with ``x`` and ``cex``);
    ``box_lty`` is the line type of its frame (``'blank'`` for none). ``at_lab`` gives the labels of the ticks ``at`` of the vertical axis
    (one per value of ``at``, or per tick when ``at`` is not given) instead of the formatted values.

    ``bty`` (``'o'`` or ``'l'``) is the box type and ``las`` the orientation of the labels of the vertical axis (0 = parallel to it, 1 = horizontal).
    ``xaxt='n'`` leaves out the horizontal axis.
    ``grep=True`` makes a string ``mod`` a regular expression searched for in the coefficient names (which must find exactly one) and
    ``fixed=True`` a literal string instead (the ``grep`` and ``fixed`` arguments metafor reads through ``...``).
    ``na_action`` stands for R's ``options(na.action=)``: ``'fail'`` stops when the regression line is drawn (``pred=True``) and studies were left out of the
    fit, the other choices give the same plot.

    ``x`` is a fit of :func:`metafopy.rma`, including location-scale models, of :func:`metafopy.selmodel`, of :func:`metafopy.rma_mv`
    or of :func:`metafopy.rma_glmm`; the bands are the predictions of the model, so no prediction interval is drawn where
    :func:`metafopy.predict` gives none (location-scale models). The sizes of the points are equal for the models without weights
    (``selmodel`` and ``rma_glmm`` fits).

    Returns a :class:`RegplotResult` with the plotted coordinates and the axes; :func:`points_regplot` redraws its points.

    Raises:
        metafopy.errors.InputError: the model has no moderator or an argument is invalid, or ``na_action='fail'`` and studies were left out.
    """
    metafopy.methods.fitview.check_class(x, notav=('rma.mh', 'rma.peto'))
    metafopy.methods.common.check_na_action(na_action)
    view = metafopy.methods.fitview.fit_view(x)
    if na_action == 'fail' and pred is True and view.na_rows is not None and view.na_rows.any():
        raise metafopy.errors.InputError('Missing values in results.')
    if view.int_only:
        raise metafopy.errors.InputError('Cannot draw plot for intercept-only models.')
    if transf is not None and atransf is not None:
        raise metafopy.errors.InputError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    axx = metafopy.plot.rutils.get_ax(ax)
    k = view.k
    names = view.model.coef_names
    lvl = view.level if level is None else metafopy.plot.rutils.level_to_alpha(level)
    fgc = metafopy.plot.draw.FG
    col_l = expand(fgc if col is None else col, k, 'col')
    bg_l = expand(metafopy.plot.rutils.coladj(-0.35) if bg is None else bg, k, 'bg')
    pch_l = [int(v) for v in expand(pch, k, 'pch')]
    slab_l = [str(s) for s in (view.slab if slab is None else slab)]
    if slab is not None and len(slab_l) == view.k_all != k:
        # metafor takes one label per row of the original dataset and drops those of the studies left out of the fit
        slab_l = [slab_l[i] for i in numpy.flatnonzero(view.keep)]
    if len(slab_l) != k:
        raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(slab_l)}) does not correspond to the number of studies ({k}).")
    shade_c = [metafopy.plot.rutils.coladj(-0.15), metafopy.plot.rutils.coladj(-0.05)]
    if isinstance(shade, str):
        shade_c = [shade, shade]
    elif not isinstance(shade, bool):
        shade_c = list(shade) if len(shade) == 2 else [shade[0], shade[0]]
    do_shade = bool(shade)
    grid_c = [grid, grid] if isinstance(grid, str) else [metafopy.plot.rutils.coladj(-0.2), metafopy.plot.rutils.coladj(0.6)]
    do_grid = bool(grid)
    # R: .expand1(x, 4) repeats a single value for all four lines
    line_cols = [fgc, fgc, fgc, metafopy.plot.rutils.coladj(-0.5)] if lcol is None else ([lcol] if isinstance(lcol, str) else list(lcol))
    if len(line_cols) == 1:
        line_cols = line_cols * 4
    elif len(line_cols) == 2:
        line_cols = [line_cols[0], line_cols[1], line_cols[1], metafopy.plot.rutils.coladj(-0.5)]
    elif len(line_cols) == 3:
        line_cols = [*line_cols, metafopy.plot.rutils.coladj(-0.5)]
    line_types: list[typing.Any] = ['solid', 'dashed', 'dotted', 'solid'] if lty is None else ([lty] if isinstance(lty, str | int) else list(lty))
    if len(line_types) == 1:
        line_types = line_types * 4
    elif len(line_types) == 2:
        line_types = [line_types[0], line_types[1], line_types[1], 'solid']
    elif len(line_types) == 3:
        line_types = [*line_types, 'solid']
    line_widths = [3.0, 1.0, 1.0, 2.0] if lwd is None else ([float(lwd)] if isinstance(lwd, int | float) else [float(v) for v in lwd])
    if len(line_widths) == 1:
        line_widths = line_widths * 4
    elif len(line_widths) == 2:
        line_widths = [line_widths[0], line_widths[1], line_widths[1], 2.0]
    elif len(line_widths) == 3:
        line_widths = [*line_widths, 2.0]
    dg = int(numpy.atleast_1d(digits)[0])  # R: digits[[1]]; R's 1 is a double

    pos = find_moderator(view, names, mod, grep, fixed)
    xmat = view.X
    xi = xmat[:, pos].copy()
    yi = view.yi.copy()
    vi = view.vi
    # metafor uses equal weights where weights() does not work (selmodel and rma.glmm fits)
    weights = numpy.ones(k) if view.weights is None else view.weights

    def predict_at(values: FloatArray) -> metafopy.methods.predict.PredictResult:
        newx = numpy.tile(xmat.mean(axis=0), (values.shape[0], 1))
        newx[:, pos] = values
        if view.int_incl:
            newx = newx[:, 1:]
        return metafopy.methods.predict.predict(view.model, newmods=newx, level=100 * (1 - lvl))

    ref_pred = isinstance(pred, metafopy.methods.predict.PredictResult)
    draw_pred = True if ref_pred else bool(pred)
    label_mode: str | None = label if isinstance(label, str) else None
    if label_mode is not None and label_mode not in ('all', 'ciout', 'piout'):
        raise metafopy.errors.InputError("Argument 'label' must be 'all', 'ciout' or 'piout' when a string.")
    at_obs: tuple[FloatArray, ...] | None
    if isinstance(pred, metafopy.methods.predict.PredictResult):
        if xvals is None:
            raise metafopy.errors.InputError("Must specify the 'xvals' argument when passing an object from predict() to 'pred'.")
        xs = numpy.atleast_1d(numpy.asarray(xvals, dtype=float))
        if xs.shape[0] != pred.pred.shape[0]:
            raise metafopy.errors.InputError(
                f"Length of the 'xvals' argument ({xs.shape[0]}) does not correspond to the number of predicted values ({pred.pred.shape[0]})."
            )
        fit, ci_lb, ci_ub = pred.pred, pred.ci_lb, pred.ci_ub
        if numpy.any(numpy.isnan(pred.pi_lb)):
            pi_lb, pi_ub = ci_lb, ci_ub
            if pi:
                warnings.warn("Object passed to 'pred' argument does not contain prediction interval information.", stacklevel=2)
            pi = False
        else:
            pi_lb, pi_ub = pred.pi_lb, pred.pi_ub
        if label_mode in ('ciout', 'piout'):
            warnings.warn("Cannot label points based on the confidence/prediction interval when passing an object to 'pred'.", stacklevel=2)
            label_mode = None
            label = False
        at_obs = None
        plim_x = (float(xs.min()), float(xs.max()))
    else:
        if xvals is not None:
            xs = numpy.atleast_1d(numpy.asarray(xvals, dtype=float))
            plim_x = (float(xs.min()), float(xs.max()))
        else:
            if predlim is None:
                rng = float(xi.max() - xi.min())
                plim_x = (float(xi.min()) - 0.04 * rng, float(xi.max()) + 0.04 * rng)
            else:
                if len(predlim) != 2:
                    raise metafopy.errors.InputError("Argument 'predlim' must be of length 2.")
                plim_x = (float(predlim[0]), float(predlim[1]))
            xs = numpy.linspace(plim_x[0], plim_x[1], 1000)
        band = predict_at(xs)
        fit, ci_lb, ci_ub = band.pred, band.ci_lb, band.ci_ub
        if numpy.any(numpy.isnan(band.pi_lb)):
            pi_lb, pi_ub = ci_lb, ci_ub
            if pi:
                warnings.warn('Cannot draw prediction interval for the given model.', stacklevel=2)
            pi = False
        else:
            pi_lb, pi_ub = band.pi_lb, band.pi_ub
        obs = predict_at(xi)
        has_pi = not numpy.any(numpy.isnan(obs.pi_lb))
        if not has_pi and label_mode == 'piout':
            warnings.warn('Cannot label points based on the prediction interval for the given model.', stacklevel=2)
            label_mode = None
            label = False
        at_obs = (obs.pred, obs.ci_lb, obs.ci_ub, obs.pi_lb if has_pi else obs.ci_lb, obs.pi_ub if has_pi else obs.ci_ub)

    def tr(v: FloatArray) -> FloatArray:
        return v if transf is None else numpy.asarray(metafopy.methods.common.apply_each(v, transf, targs), dtype=float)

    yi_t = tr(yi)
    fit_t, ci_lb_t, ci_ub_t, pi_lb_t, pi_ub_t = (tr(v) for v in (fit, ci_lb, ci_ub, pi_lb, pi_ub))
    ci_lb_t, ci_ub_t = sorted_pair(ci_lb_t, ci_ub_t)
    pi_lb_t, pi_ub_t = sorted_pair(pi_lb_t, pi_ub_t)
    # the bands at the observed moderator values, to label the studies outside them (absent when a predict() result was passed)
    at_obs_t: tuple[FloatArray, ...] | None = None
    if at_obs is not None:
        obs_t = [tr(v) for v in at_obs]
        at_obs_t = (obs_t[0], *sorted_pair(obs_t[1], obs_t[2]), *sorted_pair(obs_t[3], obs_t[4]))
    yi_pred_t = None if at_obs_t is None else at_obs_t[0]
    if olim is not None:
        if len(olim) != 2:
            raise metafopy.errors.InputError("Argument 'olim' must be of length 2.")
        lo_o, hi_o = sorted(float(v) for v in olim)
        yi_t, ci_lb_t, ci_ub_t, fit_t, pi_lb_t, pi_ub_t = (numpy.clip(v, lo_o, hi_o) for v in (yi_t, ci_lb_t, ci_ub_t, fit_t, pi_lb_t, pi_ub_t))

    sizes = symbol_sizes(psize, weights, vi, plim)

    xlab_c = (names[pos] or 'Moderator') if xlab is None else xlab
    ylab_c = metafopy.plot.rutils.setlab(view.measure, transf, atransf, 1) if ylab is None else ylab
    xl = (float(xi.min()), float(xi.max())) if xlim is None else (float(xlim[0]), float(xlim[1]))
    if ylim is None:
        if pi:
            ylim_c = (float(min(yi_t.min(), pi_lb_t.min())), float(max(yi_t.max(), pi_ub_t.max())))
        elif ci:
            ylim_c = (float(min(yi_t.min(), ci_lb_t.min())), float(max(yi_t.max(), ci_ub_t.max())))
        else:
            ylim_c = (float(yi_t.min()), float(yi_t.max()))
    else:
        if len(ylim) != 2:
            raise metafopy.errors.InputError("Argument 'ylim' must be of length 2.")
        ylim_c = (float(ylim[0]), float(ylim[1]))
    at_in = None if at is None else numpy.asarray(at, dtype=float)
    if at_in is not None:
        ylim_c = (min(ylim_c[0], float(at_in.min())), max(ylim_c[1], float(at_in.max())))

    ux = metafopy.plot.rutils.usr_range(*xl)
    uy = metafopy.plot.rutils.usr_range(*ylim_c)
    xt = metafopy.plot.rutils.ax_ticks(*ux)
    if at_in is None:
        yt = metafopy.plot.rutils.ax_ticks(*uy)
    else:
        yt = at_in[(at_in > uy[0]) & (at_in < uy[1])]
    yvals = metafopy.methods.common.apply_each(yt, atransf, targs) if atransf is not None else yt
    ylabels = [metafopy.plot.rutils.fmtx1(float(v), dg) for v in yvals]  # R: drop0trailing needs an integer `digits`, not the default double
    if at_lab is not None:
        labels_given = [str(v) for v in at_lab]
        if at_in is not None and len(labels_given) == at_in.shape[0]:
            labels_given = [lab for lab, v in zip(labels_given, at_in, strict=True) if uy[0] < v < uy[1]]
        if len(labels_given) != len(yt):
            raise metafopy.errors.InputError(f"Length of the 'at_lab' argument ({len(labels_given)}) does not match the number of ticks ({len(yt)}).")
        ylabels = labels_given
    metafopy.plot.draw.plain_axes(
        axx, ux, uy, None if xaxt == 'n' else xt, yt, None if xaxt == 'n' else metafopy.plot.rutils.rformat(xt), ylabels, bty=bty, thin_ylabels=True
    )
    if las == 0:
        axx.tick_params(axis='y', labelrotation=90)
        for label_ in axx.get_yticklabels():
            label_.set_verticalalignment('center')
    axx.set_xlabel(xlab_c, fontsize=metafopy.plot.draw.base_fontsize())
    axx.set_ylabel(ylab_c, fontsize=metafopy.plot.draw.base_fontsize())

    xpoly = numpy.concatenate([xs, xs[::-1]])
    if do_shade:
        if pi:
            metafopy.plot.draw.poly(axx, xpoly, numpy.concatenate([pi_lb_t, pi_ub_t[::-1]]), col=shade_c[1], border=None, zorder=0.6)
        if ci:
            metafopy.plot.draw.poly(axx, xpoly, numpy.concatenate([ci_lb_t, ci_ub_t[::-1]]), col=shade_c[0], border=None, zorder=0.65)
    if ci:
        for bound in (ci_lb_t, ci_ub_t):
            metafopy.plot.draw.line(axx, xs, bound, lty=line_types[1], col=line_cols[1], lw=line_widths[1]).set_zorder(1.0)
    if pi:
        for bound in (pi_lb_t, pi_ub_t):
            metafopy.plot.draw.line(axx, xs, bound, lty=line_types[2], col=line_cols[2], lw=line_widths[2]).set_zorder(1.0)
    if do_grid:
        # R recycles the two grid colours over the lines of each axis
        for j, v in enumerate(xt):
            metafopy.plot.draw.grid(axx, [float(v)], [], grid_c[j % 2])
        for j, v in enumerate(yt):
            metafopy.plot.draw.grid(axx, [], [float(v)], grid_c[j % 2])
    if refline is not None and not math.isnan(refline):
        metafopy.plot.draw.hline(axx, refline, lty=line_types[3], col=line_cols[3], lw=line_widths[3]).set_zorder(1.0)
    if draw_pred:
        metafopy.plot.draw.line(axx, xs, fit_t, lty=line_types[0], col=line_cols[0], lw=line_widths[0]).set_zorder(1.1)

    order = numpy.argsort(-sizes, kind='stable')
    draw_points(axx, xi, yi_t, pch_l, sizes, col_l, bg_l, order)

    flags = numpy.zeros(k, dtype=bool)
    if label_mode == 'all' or (isinstance(label, bool) and label):
        flags[:] = True
    elif label_mode in ('ciout', 'piout'):
        assert at_obs_t is not None
        lb, ub = (at_obs_t[1], at_obs_t[2]) if label_mode == 'ciout' else (at_obs_t[3], at_obs_t[4])
        flags = (yi_t < lb) | (yi_t > ub)
        flags[(xi < plim_x[0]) | (xi > plim_x[1])] = False
    elif label is not False and not isinstance(label, bool | str):
        sel = numpy.asarray(label)
        if sel.dtype == numpy.bool_:
            if sel.shape[0] != k:
                raise metafopy.errors.InputError(f"Length of the 'label' argument ({sel.shape[0]}) does not correspond to the number of studies ({k}).")
            flags = sel.copy()
        else:
            flags[sel.astype(numpy.intp)] = True
    yrange = ylim_c[1] - ylim_c[0]
    off = (float(offset),) if isinstance(offset, int | float) else tuple(float(v) for v in offset)
    if label is False:
        # as in metafor, the offset is only converted (to units of the vertical axis) when labels were requested
        off3 = off
    elif len(off) == 2:
        off3 = (off[0] / 100 * yrange, off[1] / 100 * yrange, 1.0)
    elif len(off) == 1:
        off3 = (0.0, off[0] / 100 * yrange, 1.0)
    else:
        off3 = off
    if flags.any():
        draw_labels(axx, xi, yi_t, yi_pred_t, sizes, flags, slab_l, off3, labsize)

    regplot_legend(axx, legend, draw_pred, ci, pi, lvl, dg, line_types, line_cols, line_widths, shade_c, col_l, bg_l, pch_l, box_lty)
    return RegplotResult(axx, xi, yi_t, sizes, yi_pred_t, flags, tuple(slab_l), order, off3, labsize, view.ids, tuple(pch_l), tuple(col_l), tuple(bg_l))


def draw_points(
    ax: matplotlib.axes.Axes,
    xi: FloatArray,
    yi: FloatArray,
    pch: collections.abc.Sequence[int],
    psize: FloatArray,
    col: collections.abc.Sequence[typing.Any],
    bg: collections.abc.Sequence[typing.Any],
    order: numpy.typing.NDArray[numpy.intp],
) -> None:
    """Draw the points in ``order`` (the largest symbols first, so that the small ones are on top)."""
    for i in order:
        metafopy.plot.draw.pts(ax, float(xi[i]), float(yi[i]), pch=pch[i], cex=float(psize[i]), col=col[i], bg=bg[i])


def draw_labels(
    ax: matplotlib.axes.Axes,
    xi: FloatArray,
    yi: FloatArray,
    pred: FloatArray | None,
    psize: FloatArray,
    flags: numpy.typing.NDArray[numpy.bool_],
    slab: collections.abc.Sequence[str],
    offset: collections.abc.Sequence[float],
    labsize: float,
) -> None:
    """Label the flagged points above the point where the observed value is above the fitted one and below otherwise."""
    for i in numpy.flatnonzero(flags):
        above = pred is not None and bool(yi[i] > pred[i])
        dy = offset[0] + offset[1] * float(psize[i]) ** offset[2]
        metafopy.plot.draw.txt(ax, float(xi[i]), float(yi[i]) + (dy if above else -dy), slab[int(i)], cex=labsize)


def points_regplot(x: RegplotResult, ax: matplotlib.axes.Axes | None = None) -> None:
    """Redraw the points (and the labels, if there were any) of a :func:`regplot` (metafor's ``points.regplot``).

    The points are drawn on ``ax`` (default: the axes of the plot) in the order of ``x.order``, which puts the points with the
    smallest symbols on top again after other elements were added to the plot.

    Raises:
        metafopy.errors.InputError: ``x`` is not the result of :func:`regplot`.
    """
    if not isinstance(x, RegplotResult):
        raise metafopy.errors.InputError('Argument \'x\' must be an object of class "regplot".')
    axx = x.ax if ax is None else ax
    draw_points(axx, x.xi, x.yi, x.pch, x.psize, x.col, x.bg, x.order)
    if bool(x.label.any()):
        draw_labels(axx, x.xi, x.yi, x.pred, x.psize, x.label, x.slab, x.offset, x.labsize)


def dash(name: str) -> typing.Any:
    return metafopy.plot.draw.LTY.get(name, name)


def regplot_legend(
    ax: matplotlib.axes.Axes,
    legend: bool | str | dict[str, typing.Any],
    addpred: bool,
    ci: bool,
    pi: bool,
    level: float,
    digits: int,
    lty: list[str],
    lcol: list[str],
    lwd: list[float],
    shade: list[str],
    col: list[typing.Any],
    bg: list[typing.Any],
    pch: list[int],
    box_lty: str = 'solid',
) -> None:
    """The legend of ``regplot()``: studies, regression line and the confidence and prediction intervals."""
    opts: dict[str, typing.Any] = {'x': 'topright', 'cex': 1.0}
    if isinstance(legend, dict):
        opts.update({key: val for key, val in legend.items() if key in opts})
    elif isinstance(legend, str):
        opts['x'] = legend
    elif not legend:
        return
    handles: list[typing.Any] = []
    labels: list[str] = []
    cex = float(opts['cex'])
    if len(set(pch)) == 1 and len({str(c) for c in col}) == 1 and len({str(b) for b in bg}) == 1:
        marker = metafopy.plot.draw.PCH.get(pch[0], ('o', True, False))[0]
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                linestyle='None',
                marker=marker,
                markerfacecolor=metafopy.plot.draw.rcolor(bg[0]),
                markeredgecolor=metafopy.plot.draw.rcolor(col[0]),
                markersize=5.5 * 1.5,
            )
        )
        labels.append('Studies')
    percent = f'{round(100 * (1 - level), digits):g}%'
    if addpred:
        handles.append(matplotlib.lines.Line2D([], [], linestyle=dash(lty[0]), color=metafopy.plot.draw.rcolor(lcol[0]), linewidth=lwd[0]))
        labels.append('Regression Line')
    if ci:
        handles.append(
            matplotlib.patches.Patch(
                facecolor=metafopy.plot.draw.rcolor(shade[0]),
                edgecolor=metafopy.plot.draw.rcolor(lcol[1]),
                linewidth=lwd[1],
                linestyle=dash(lty[1]),
            )
        )
        labels.append(f'{percent} Confidence Interval')
    if pi:
        handles.append(
            matplotlib.patches.Patch(
                facecolor=metafopy.plot.draw.rcolor(shade[1]),
                edgecolor=metafopy.plot.draw.rcolor(lcol[2]),
                linewidth=lwd[2],
                linestyle=dash(lty[2]),
            )
        )
        labels.append(f'{percent} Prediction Interval')
    if handles:
        box = ax.legend(
            handles,
            labels,
            loc=typing.cast(typing.Any, LEGEND_LOC.get(str(opts['x']), str(opts['x']))),
            fontsize=metafopy.plot.draw.base_fontsize() * cex,
            framealpha=1.0,
        )
        if box_lty == 'blank':
            box.get_frame().set_edgecolor('none')
        else:
            box.get_frame().set_linestyle(dash(box_lty))
