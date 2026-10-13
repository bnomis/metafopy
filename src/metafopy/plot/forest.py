"""Forest plots (metafor's ``forest.rma``, ``forest.default``, ``addpoly.rma`` and ``addpoly.default``).

The layout logic follows the R code line by line: study rows are numbered ``k:1`` from the top,
the summary polygon sits at row ``-1`` (prediction interval at ``-2``), the axis limits are
``alim``/``xlim`` with the label and annotation areas reserved left and right of the plot, and the
sizes of the arrow heads, CI end caps and summary polygons are fractions of the plot height.

See https://wviechtb.github.io/metafor/reference/forest.rma.html and
https://wviechtb.github.io/metafor/reference/addpoly.html, and Viechtbauer (2010),
doi:10.18637/jss.v036.i03.
"""

import collections.abc
import dataclasses
import math
import re
import typing
import warnings
import weakref

import matplotlib.axes
import matplotlib.colors
import matplotlib.text
import numpy
import numpy.typing
import pandas
import scipy.special
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.influence
import metafopy.methods.predict
import metafopy.models.rma_uni
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils
import metafopy.transf

__all__ = ['ForestResult', 'addpoly', 'forest']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Func = collections.abc.Callable[..., typing.Any]
type PredStyle = typing.Literal['line', 'polygon', 'bar', 'shade', 'dist']
type FontsArg = metafopy.plot.draw.FontSpec | collections.abc.Sequence[metafopy.plot.draw.FontSpec] | collections.abc.Mapping[str, int] | None
type PredDist = tuple[numpy.typing.ArrayLike, numpy.typing.ArrayLike] | collections.abc.Mapping[str, typing.Any]

POLYLEN = 10000
FOREST_STATE: weakref.WeakKeyDictionary[matplotlib.axes.Axes, dict[str, typing.Any]] = weakref.WeakKeyDictionary()


@dataclasses.dataclass(frozen=True, slots=True)
class ForestResult:
    """What metafor's ``forest()`` returns invisibly, plus the axes drawn on.

    ``xlim``/``ylim`` are the limits of the plot region, ``alim`` the limits of the x axis,
    ``at`` the tick locations, ``rows`` the row of each plotted estimate, ``textpos`` the x
    positions of the study labels and annotations, ``areas`` the relative widths of the
    label, plot and annotation areas (``forest`` of a fitted model only).
    """

    ax: matplotlib.axes.Axes
    xlim: tuple[float, float]
    alim: tuple[float, float]
    at: FloatArray
    ylim: tuple[float, float]
    rows: FloatArray
    cex: float
    cex_lab: float
    cex_axis: float
    ilab_xpos: FloatArray | None
    ilab_pos: list[int] | None
    textpos: tuple[float, float]
    areas: tuple[float, float, float] | None = None


def astr(v: typing.Any) -> str:
    if isinstance(v, float | numpy.floating):
        f = float(v)
        if math.isnan(f):
            return 'NA'
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f'{f:.15g}'
    return str(v)


def ilab_matrix(ilab: typing.Any) -> numpy.typing.NDArray[numpy.object_] | None:
    if ilab is None:
        return None
    if isinstance(ilab, pandas.DataFrame):
        arr = ilab.to_numpy(dtype=object)
    else:
        arr = numpy.asarray(ilab, dtype=object)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def floats(v: float | collections.abc.Sequence[float]) -> list[float]:
    return [float(a) for a in numpy.atleast_1d(numpy.asarray(v, dtype=float))]


def expand(v: typing.Any, k: int, name: str) -> list[typing.Any]:
    """R's ``.expand1()`` followed by the length check.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * k
    out = list(v)
    if len(out) == 1:
        out = out * k
    if len(out) != k:
        raise metafopy.errors.InputError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the number of outcomes ({k}).")
    return out


def pervec(v: typing.Any, keep: numpy.typing.NDArray[numpy.bool_], k: int, k_all: int, name: str) -> list[typing.Any]:
    """A per-study argument of a fitted model: a scalar is recycled, a vector may have one entry per supplied study."""
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * k
    out = list(v)
    if len(out) == 1:
        return out * k
    return list(per_study(out, keep, k, k_all, name))


def order_by(order: typing.Any, decreasing: bool, k: int) -> numpy.typing.NDArray[numpy.intp]:
    arr = numpy.asarray(order)
    if arr.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of the 'order' argument ({arr.shape[0]}) does not correspond to the number of outcomes ({k}).")
    if decreasing:
        if arr.dtype.kind in 'fiub':
            return numpy.argsort(-arr.astype(float), kind='stable')
        return numpy.argsort(arr[::-1], kind='stable')[::-1][::-1]
    return numpy.argsort(arr, kind='stable')


def lexorder(primary: FloatArray, secondary: FloatArray) -> numpy.typing.NDArray[numpy.intp]:
    return numpy.lexsort((secondary, primary))


def apply_olim(x: FloatArray, olim: tuple[float, float] | None) -> FloatArray:
    if olim is None:
        return x
    out = x.copy()
    out[out < olim[0]] = olim[0]
    out[out > olim[1]] = olim[1]
    return out


def compress_rows(rows: FloatArray, not_na: numpy.typing.NDArray[numpy.bool_]) -> FloatArray:
    """Close the gaps left by omitted (missing) estimates, as the ``na.omit`` branch of ``forest`` does."""
    rows_new = rows.copy()
    for rna in rows[~not_na]:
        rows_new[rows >= rna] -= 1
    return rows_new[not_na]


def point_sizes(wi: FloatArray, plim: collections.abc.Sequence[float], k: int) -> FloatArray:
    if len(plim) < 2:
        raise metafopy.errors.InputError("Argument 'plim' must be of length 2 or 3.")
    p1, p2 = plim[0], plim[1]
    psize = numpy.full(k, numpy.nan)
    if not math.isnan(p1) and not math.isnan(p2):
        rng = numpy.nanmax(wi) - numpy.nanmin(wi)
        if rng <= math.sqrt(numpy.finfo(float).eps):
            psize = numpy.ones(k)
        else:
            psize = (wi - numpy.nanmin(wi)) / rng
            psize = psize * (p2 - p1) + p1
    if math.isnan(p1) and not math.isnan(p2):
        psize = wi / numpy.nanmax(wi) * p2
        if len(plim) == 3:
            psize[psize <= plim[2]] = plim[2]
    if not math.isnan(p1) and math.isnan(p2):
        psize = wi / numpy.nanmin(wi) * p1
        if len(plim) == 3:
            psize[psize >= plim[2]] = plim[2]
    if numpy.all(numpy.isnan(psize)):
        psize = numpy.ones(k)
    return psize


def strheight(ax: matplotlib.axes.Axes, ylim: tuple[float, float]) -> float:
    """``strheight("O")`` in user coordinates (cap height of 12pt-scaled text relative to the axes height)."""
    fs = metafopy.plot.draw.base_fontsize()
    fig = ax.figure
    h_in = ax.get_position().height * fig.get_figheight()  # type: ignore[union-attr]
    return 0.716 * fs / 72.0 / h_in * (ylim[1] - ylim[0])


def arrow(ax: matplotlib.axes.Axes, side: typing.Literal['l', 'r'], alim: tuple[float, float], row: float, aw: float, ah: float, col: str) -> None:
    if side == 'l':
        xs = [alim[0], alim[0] + aw, alim[0] + aw, alim[0]]
    else:
        xs = [alim[1], alim[1] - aw, alim[1] - aw, alim[1]]
    metafopy.plot.draw.poly(ax, xs, [row, row + ah, row - ah, row], col=col, border=col)


def diamond(xlo: float, mid: float, xhi: float, row: float, height: float) -> tuple[FloatArray, FloatArray]:
    poladds = numpy.arange(POLYLEN) * (height / (POLYLEN - 1))
    xs = numpy.concatenate(
        [
            numpy.linspace(xlo, mid, POLYLEN),
            numpy.linspace(mid, xhi, POLYLEN),
            numpy.linspace(xhi, mid, POLYLEN),
            numpy.linspace(mid, xlo, POLYLEN),
        ]
    )
    ys = numpy.concatenate([row + poladds, row + height - poladds, row - poladds, row - height + poladds])
    return xs, ys


def anno_symbol(annosym: collections.abc.Sequence[str] | None) -> list[str]:
    if annosym is None:
        return [' [', ', ', ']', '-', ' ']
    a = list(annosym)
    if len(a) == 3:
        a += ['-', ' ']
    if len(a) == 4:
        a += [' ']
    if len(a) != 5:
        raise metafopy.errors.InputError("Argument 'annosym' must be a vector of length 3 (or 4 or 5).")
    return a


def build_annotext(
    est: FloatArray,
    lb: FloatArray,
    ub: FloatArray,
    digits: collections.abc.Sequence[int],
    annosym: list[str],
    width: collections.abc.Sequence[int] | int | None,
    weights: FloatArray | None,
    pi_row: bool,
    atransf: Func | None,
    targs: typing.Any,
    weight_pad_rows: tuple[int, ...] = (),
) -> tuple[list[str], list[int]]:
    """The annotation strings (``est [lb, ub]``) and their column widths.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    cols: list[FloatArray] = [est, lb, ub]
    if atransf is not None:
        cols = [metafopy.methods.common.apply_each(c, atransf, targs) for c in cols]
        lo, hi = metafopy.methods.common.psort(cols[1], cols[2])
        cols[1], cols[2] = lo, hi
    if weights is not None:
        mat_cols = [weights, *cols]
        dig = [digits[2], digits[0], digits[0], digits[0]]
    else:
        mat_cols = cols
        dig = [digits[0]] * 3
    strs = [[metafopy.plot.rutils.fmtx1(float(v), dg) for v in c] for c, dg in zip(mat_cols, dig, strict=True)]
    for r in weight_pad_rows:  # the "100" of the summary row
        strs[0][r] = '100'
    if width is None:
        wid = [max(len(s) for s in col) for col in strs]
    else:
        wid = [int(width)] * len(strs) if isinstance(width, int | float) else [int(w) for w in width]
        if len(wid) == 1:
            wid = wid * len(strs)
        if len(wid) != len(strs):
            raise metafopy.errors.InputError(f"Length of the 'width' argument ({len(wid)}) does not match the number of annotation columns ({len(strs)}).")
    # formatC() pads a character vector to the largest of the requested width and the longest string
    strs = [[s.rjust(max(w, *(len(t) for t in col))) for s in col] for col, w in zip(strs, wid, strict=True)]
    out: list[str] = []
    for i in range(len(strs[0])):
        if weights is not None:
            row = f'{strs[0][i]}%{annosym[0][0] * 3}{strs[1][i]}{annosym[0]}{strs[2][i]}{annosym[1]}{strs[3][i]}{annosym[2]}'
        else:
            row = f'{strs[0][i]}{annosym[0]}{strs[1][i]}{annosym[1]}{strs[2][i]}{annosym[2]}'
        out.append(row)
    return out, (wid[1:] if weights is not None else wid)


def finish_annotext(raw: list[str], annosym: list[str], extra_blank: bool, pi_row: bool) -> list[str]:
    out = list(raw)
    isna = ['NA' in s for s in out]
    if pi_row:
        isna = isna[:-1]
        for i, f in enumerate(isna):
            if f:
                out[i] = ''
        out[-1] = out[-1].replace('NA', '').replace('%', '')
    else:
        for i, f in enumerate(isna):
            if f:
                out[i] = ''
    return [s.replace('-', annosym[3]).replace(' ', annosym[4]) for s in out]


def default_mlab(method: str) -> str:
    return {'FE': 'Fixed-Effect Model', 'EE': 'Equal-Effects Model', 'CE': 'Common-Effect Model'}.get(method, 'Random-Effects Model')


# auditme: accept[passthrough-shim] sets the family and font of ``fonts``, with ``bold`` as the explicit font=2 of metafor
def ftxt(ax: matplotlib.axes.Axes, x: float, y: float, s: str, fnt: metafopy.plot.draw.Font, *, bold: bool = False, **kw: typing.Any) -> matplotlib.text.Text:
    """``text()`` in the font ``(family, font)`` of ``fonts``; ``bold=True`` is an explicit ``font=2``, which wins over the font in ``fnt``."""
    return metafopy.plot.draw.txt(ax, x, y, s, family=fnt[0], font=2 if bold else fnt[1], **kw)


def check_pred(predstyle: str) -> PredStyle:
    """The validated ``predstyle``.

    Raises:
        metafopy.errors.InputError: the style is not one of the allowed names.
    """
    for name in typing.get_args(PredStyle.__value__):
        if predstyle == name:
            return typing.cast(PredStyle, name)
    raise metafopy.errors.InputError("'predstyle' should be one of 'line', 'polygon', 'bar', 'shade', 'dist'.")


# -- prediction distribution (predstyle 'shade' and 'dist') --------------------------------------------------------------

PRED_POINTS = 10000
DENSITY_STYLES = ('dist', 'shade')  # the styles that draw the predictive distribution
type BoolArray = numpy.typing.NDArray[numpy.bool_]


@dataclasses.dataclass(frozen=True, slots=True)
class TransfRule:
    """How the density of a prediction is carried over to the scale of a back-transformation (the change of variables of metafor's ``funlist``)."""

    functions: tuple[Func, ...]
    inverse: Func
    jacobian: Func
    lo: float
    hi: float


def jac_exp(xs: FloatArray) -> FloatArray:
    return xs


def jac_tanh(xs: FloatArray) -> FloatArray:
    return 1 - xs**2


def jac_logit(xs: FloatArray) -> FloatArray:
    return xs * (1 - xs)


def jac_arcsin(xs: FloatArray) -> FloatArray:
    return 2 * numpy.sqrt(xs * (1 - xs))


def jac_probit(xs: FloatArray) -> FloatArray:
    return numpy.asarray(metafopy.transf.dnorm(metafopy.transf.qnorm(xs)), dtype=float)


def jac_ahw(xs: FloatArray) -> FloatArray:
    return 3 * (1 - xs) ** (2 / 3)


def jac_abt(xs: FloatArray) -> FloatArray:
    return 1 - xs


# the transformations metafor can draw a prediction distribution for (the order of its ``funlist``)
TRANSF_RULES: tuple[TransfRule, ...] = (
    TransfRule((numpy.exp, math.exp), numpy.log, jac_exp, 0.01, math.inf),
    TransfRule((metafopy.transf.transf_ztor, numpy.tanh, math.tanh), numpy.arctanh, jac_tanh, -0.99, 0.99),
    TransfRule((metafopy.transf.transf_ilogit, metafopy.transf.plogis, scipy.special.expit), metafopy.transf.qlogis, jac_logit, 0.01, 0.99),
    TransfRule((metafopy.transf.transf_iarcsin,), metafopy.transf.transf_arcsin, jac_arcsin, 0.01, 0.99),
    TransfRule(
        (metafopy.transf.transf_iprobit, metafopy.transf.pnorm, scipy.stats.norm.cdf, scipy.special.ndtr), metafopy.transf.qnorm, jac_probit, 0.01, 0.99
    ),
    TransfRule((metafopy.transf.transf_iahw,), metafopy.transf.transf_ahw, jac_ahw, 0.01, 0.99),
    TransfRule((metafopy.transf.transf_iabt,), metafopy.transf.transf_abt, jac_abt, 0.01, 0.99),
)


def transf_rule(transf: Func | None) -> TransfRule | None:
    """The :class:`TransfRule` of ``transf`` (``None`` without a transformation).

    Raises:
        metafopy.errors.InputError: the transformation is not one metafor can draw a prediction distribution for.
    """
    if transf is None:
        return None
    for rule in TRANSF_RULES:
        if any(transf is f for f in rule.functions):
            return rule
    raise metafopy.errors.InputError("Chosen transformation not (currently) possible with this 'predstyle'.")


@dataclasses.dataclass(frozen=True, slots=True)
class PredCurve:
    """The density of the predictive distribution on the plotted scale: ``xs``, ``ys``, the parts below (``l0``) and above (``g0``) ``predtail``.

    ``limited`` is whether ``predlim`` was given (``'dist'`` then keeps the whole density instead of dropping the part below 0.5% of the peak).
    """

    xs: FloatArray
    ys: FloatArray
    l0: BoolArray
    g0: BoolArray
    lower_tail: bool
    limited: bool = False


def parse_predtail(predtail: float | str | None, pred: float) -> tuple[float, bool]:
    """metafor's ``predtail`` (a number or a string such as ``'<2'``/``'>2'``) as ``(value, lower_tail)``.

    Raises:
        metafopy.errors.InputError: the value cannot be interpreted.
    """
    if predtail is None:
        return 0.0, pred > 0
    text = str(predtail)
    try:
        return float(re.sub('[<=>]', '', text)), '<' in text
    except ValueError:
        raise metafopy.errors.InputError(f"Cannot interpret the 'predtail' argument {predtail!r}.") from None


def parse_preddist(preddist: PredDist | None) -> tuple[FloatArray, FloatArray, dict[str, typing.Any]] | None:
    """A user-supplied predictive distribution as ``(xs, ys, extras)`` (``extras`` may hold ``pi_lb``, ``pi_ub`` and ``level``).

    Raises:
        metafopy.errors.InputError: the distribution is invalid.
    """
    if preddist is None:
        return None
    msg = "Argument 'preddist' must be a tuple (values, densities) or a mapping with the keys 'x' and 'y'."
    if isinstance(preddist, collections.abc.Mapping):
        if 'x' not in preddist or 'y' not in preddist:
            raise metafopy.errors.InputError(msg)
        xs, ys = preddist['x'], preddist['y']
        extras = {k: v for k, v in preddist.items() if k in ('pi_lb', 'pi_ub', 'level') and v is not None}
    else:
        if len(preddist) < 2:
            raise metafopy.errors.InputError(msg)
        xs, ys = preddist[0], preddist[1]
        extras = {}
    xa, ya = numpy.atleast_1d(numpy.asarray(xs, dtype=float)), numpy.atleast_1d(numpy.asarray(ys, dtype=float))
    if xa.shape != ya.shape:
        raise metafopy.errors.InputError("Length of the values of 'preddist' does not match the length of its densities.")
    return xa, ya, extras


def pred_curve(
    style: str,
    pred: float,
    pi_se: float | None,
    pi_dist: str,
    pi_ddf: float | None,
    pi_level: float,
    transf: Func | None,
    predlim: collections.abc.Sequence[float] | None,
    preddist: tuple[FloatArray, FloatArray] | None,
    predtail: float | str | None,
) -> PredCurve:
    """The predictive distribution to draw for ``predstyle='shade'`` and ``'dist'`` (metafor's code in ``forest.rma`` and ``addpoly.default``).

    ``pred`` is the (untransformed) predicted value, ``pi_se``, ``pi_dist`` (``'norm'`` or ``'t'``) and ``pi_ddf`` describe the distribution,
    ``pi_level`` is the alpha level of the prediction interval, ``predlim`` the limits of the distribution to draw (on the plotted scale),
    ``preddist`` a distribution supplied by the user (the values and the densities) and ``predtail`` the cut for the shading of ``'dist'``.

    Raises:
        metafopy.errors.InputError: the distribution cannot be drawn.
    """
    rule = transf_rule(transf)
    if preddist is None:
        if pi_se is None or math.isnan(pi_se):
            raise metafopy.errors.InputError('Cannot extract the SE of the prediction interval.')
        if pi_dist != 'norm' and (pi_ddf is None or pi_ddf <= 1):
            raise metafopy.errors.InputError('Cannot shade/draw prediction distribution when df <= 1.')
    if style == 'shade':
        q_lo, q_hi = pi_level / 2, 1 - pi_level / 2
    else:
        q_lo, q_hi = 0.0001, 0.9999

    ddf = math.inf if pi_ddf is None else pi_ddf

    def density(xs: FloatArray) -> FloatArray:
        assert pi_se is not None
        if pi_dist == 'norm':
            return numpy.asarray(scipy.stats.norm.pdf(xs, loc=pred, scale=pi_se), dtype=numpy.float64)
        return numpy.asarray(scipy.stats.t.pdf((xs - pred) / pi_se, df=ddf) / pi_se, dtype=numpy.float64)

    def check_lim() -> None:
        if predlim is None or len(predlim) != 2:
            raise metafopy.errors.InputError("Argument 'predlim' must be of length 2.")

    with numpy.errstate(all='ignore'):
        if preddist is None:
            assert pi_se is not None
            if predlim is None or style == 'shade':
                if pi_dist == 'norm':
                    lo, hi = (float(v) for v in numpy.atleast_1d(scipy.stats.norm.ppf([q_lo, q_hi], loc=pred, scale=pi_se)))
                else:
                    lo, hi = (float(v) * pi_se + pred for v in numpy.atleast_1d(scipy.stats.t.ppf([q_lo, q_hi], df=ddf)))
                xs = numpy.linspace(lo, hi, PRED_POINTS)
            else:
                check_lim()
                assert predlim is not None
                xs = numpy.linspace(float(predlim[0]), float(predlim[1]), PRED_POINTS)
                if rule is not None:
                    xs = numpy.asarray(rule.inverse(xs), dtype=float)
                    xs = xs[numpy.isfinite(xs)]
            ys = density(xs)
        else:
            xs, ys = numpy.asarray(preddist[0], dtype=float), numpy.asarray(preddist[1], dtype=float)
            if predlim is not None:
                check_lim()
                lim = numpy.asarray(predlim, dtype=float)
                if rule is not None:
                    lim = numpy.asarray(rule.inverse(lim), dtype=float)
                keep = (xs > lim[0]) & (xs < lim[1])
                xs, ys = xs[keep], ys[keep]
        lower_tail = False
        sel_l0 = numpy.zeros(xs.shape[0], dtype=bool)
        sel_g0 = numpy.zeros(xs.shape[0], dtype=bool)
        if style == 'dist':
            tail, lower_tail = parse_predtail(predtail, pred)
            sel_l0, sel_g0 = xs < tail, xs > tail
        if rule is not None:
            xs = numpy.asarray(transf(xs), dtype=numpy.float64)  # type: ignore[misc]
            ys = ys / rule.jacobian(xs)
            if predlim is None:
                keep = (xs > rule.lo) & (xs < rule.hi)
                sel_l0, sel_g0, ys, xs = sel_l0[keep], sel_g0[keep], ys[keep], xs[keep]
    return PredCurve(xs, ys, sel_l0, sel_g0, lower_tail, predlim is not None)


def ramp_colors(col_from: str, col_to: str, intensity: FloatArray) -> list[str]:
    """R's ``colorRamp(c(col_from, col_to))`` applied to ``intensity`` and written with ``rgb(..., maxColorValue = 255)`` (which truncates)."""
    c1 = numpy.array(matplotlib.colors.to_rgb(metafopy.plot.draw.rcolor(col_from)))
    c2 = numpy.array(matplotlib.colors.to_rgb(metafopy.plot.draw.rcolor(col_to)))
    out: list[str] = []
    for t in intensity:
        if math.isnan(t):
            out.append('none')
            continue
        r, g, b = (int(v) for v in (c1 + (c2 - c1) * t) * 255)
        out.append(f'#{r:02X}{g:02X}{b:02X}')
    return out


def draw_shade(
    ax: matplotlib.axes.Axes,
    curve: PredCurve,
    row: float,
    barheight: float,
    alim: tuple[float, float],
    olim: tuple[float, float] | None,
    col_from: str | None,
    col_to: str | None,
) -> None:
    """``predstyle='shade'``: rectangles along the prediction interval, shaded from ``col_from`` (the centre) to ``col_to`` (the ends).

    Raises:
        metafopy.errors.InputError: a colour is missing.
    """
    if col_from is None or col_to is None:
        raise metafopy.errors.InputError("The 'col' argument needs the second and third colour for predstyle='shade'.")
    xs, ys = curve.xs, curve.ys
    with numpy.errstate(all='ignore'):
        intensity = 1 - (ys - numpy.min(ys)) / (numpy.max(ys) - numpy.min(ys))
    sel = (xs >= alim[0]) & (xs <= alim[1])
    if olim is not None:
        sel = sel & (xs > olim[0]) & (xs < olim[1])
    xs, intensity = xs[sel], intensity[sel]
    metafopy.plot.draw.rects(ax, xs[1:], row - barheight, xs[:-1], row + barheight, cols=ramp_colors(col_from, col_to, intensity)[: xs.shape[0] - 1])


def draw_dist(
    ax: matplotlib.axes.Axes,
    curve: PredCurve,
    drow: float,
    pi_lb: float,
    pi_ub: float,
    alim: tuple[float, float],
    olim: tuple[float, float] | None,
    ef4: float,
    col: collections.abc.Sequence[str | None],
    border: collections.abc.Sequence[str | None],
) -> None:
    """``predstyle='dist'``: the predictive density, the tails beyond the interval and the parts either side of ``predtail`` shaded."""
    xs = curve.xs
    with numpy.errstate(all='ignore'):
        ys = curve.ys / numpy.max(curve.ys)
    sel = numpy.ones(ys.shape[0], dtype=bool) if curve.limited else ys > 0.005
    ys = ys * ef4
    sel = sel & (xs >= alim[0]) & (xs <= alim[1])
    if olim is not None:
        sel = sel & (xs > olim[0]) & (xs < olim[1])
    xs_l0, xs_g0 = xs[curve.l0 & sel], xs[curve.g0 & sel]
    ys_l0, ys_g0 = ys[curve.l0 & sel] + drow, ys[curve.g0 & sel] + drow
    xs, ys = xs[sel], ys[sel] + drow

    def fill(px: FloatArray, py: FloatArray, c: str | None) -> None:
        metafopy.plot.draw.poly(
            ax, numpy.concatenate([px, px[::-1]]), numpy.concatenate([py, numpy.full(py.shape[0], drow)]), col=c, border=None if c is None else border[1]
        )

    if curve.lower_tail:
        fill(xs_g0, ys_g0, col[3])
        fill(xs_l0, ys_l0, col[2])
    else:
        fill(xs_g0, ys_g0, col[2])
        fill(xs_l0, ys_l0, col[3])
    lo = xs <= pi_lb
    fill(xs[lo], ys[lo], col[1])
    hi = xs >= pi_ub
    fill(xs[hi], ys[hi], col[1])
    metafopy.plot.draw.line(ax, xs, numpy.full(xs.shape[0], drow), col=border[1] or metafopy.plot.draw.FG)
    metafopy.plot.draw.line(ax, xs, ys, col=border[1] or metafopy.plot.draw.FG)


def forest(
    x: metafopy.models.rma_uni.RMAResult | numpy.typing.ArrayLike,
    vi: numpy.typing.ArrayLike | None = None,
    sei: numpy.typing.ArrayLike | None = None,
    ci_lb: numpy.typing.ArrayLike | None = None,
    ci_ub: numpy.typing.ArrayLike | None = None,
    annotate: bool = True,
    addfit: bool = True,
    addpred: bool | None = None,
    predstyle: str | None = None,
    preddist: PredDist | None = None,
    showweights: bool = False,
    header: bool | str | collections.abc.Sequence[str] = True,
    xlim: collections.abc.Sequence[float] | None = None,
    alim: collections.abc.Sequence[float] | None = None,
    olim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    predlim: collections.abc.Sequence[float] | None = None,
    at: numpy.typing.ArrayLike | None = None,
    steps: int = 5,
    level: float | None = None,
    refline: float | None = 0,
    digits: int | collections.abc.Sequence[int] = 2,
    width: int | collections.abc.Sequence[int] | None = None,
    xlab: str | collections.abc.Sequence[str] | None = None,
    slab: collections.abc.Sequence[typing.Any] | float | None = None,
    mlab: str | collections.abc.Sequence[str] | None = None,
    ilab: typing.Any = None,
    ilab_lab: collections.abc.Sequence[str] | None = None,
    ilab_xpos: numpy.typing.ArrayLike | None = None,
    ilab_pos: int | collections.abc.Sequence[int] | None = None,
    order: str | numpy.typing.ArrayLike | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    rows: numpy.typing.ArrayLike | float | None = None,
    efac: float | collections.abc.Sequence[float] = 1,
    pch: int | collections.abc.Sequence[int] = 15,
    psize: float | numpy.typing.ArrayLike | None = None,
    plim: collections.abc.Sequence[float] = (0.5, 1.5),
    colout: str | collections.abc.Sequence[str] | None = None,
    colci: str | collections.abc.Sequence[str] | None = None,
    col: str | collections.abc.Sequence[str] | None = None,
    border: str | collections.abc.Sequence[str] | None = None,
    shade: bool | str | numpy.typing.ArrayLike | None = None,
    colshade: str | None = None,
    lty: str | collections.abc.Sequence[str] | None = None,
    fonts: FontsArg = None,
    cex: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
    predtail: float | str | None = None,
    annosym: collections.abc.Sequence[str] | None = None,
    textpos: collections.abc.Sequence[float] | None = None,
    rowadj: float | collections.abc.Sequence[float] | None = None,
    top: float = 3,
    at_lab: collections.abc.Sequence[str] | None = None,
    xlabadj: float | collections.abc.Sequence[float] | None = None,
    xlabfont: int | collections.abc.Sequence[int] = 1,
    decreasing: bool = False,
    measure: str | None = None,
) -> ForestResult:
    """Forest plot of a fitted model (``forest.rma``) or of estimates with their variances (``forest.default``).

    ``x`` is a fitted :class:`~metafopy.RMAResult` or the vector of estimates; in the second case
    give the sampling variances (``vi``), standard errors (``sei``) or CI bounds (``ci_lb`` and
    ``ci_ub``). Arguments follow metafor's names (``.`` becomes ``_``): ``slab``, ``ilab`` (with
    ``ilab_lab``, ``ilab_xpos``, ``ilab_pos``), ``order`` (``'obs'``, ``'prec'``, ``'fit'``,
    ``'resid'``, ... for a fitted model, or values to sort by), ``rows``, ``xlim``/``alim``/``at``,
    ``refline``, ``annotate``, ``header``, ``transf``/``atransf``, ``psize``/``plim``, ``shade`` and
    ``efac``. For a fitted model ``colout`` colours the markers and ``col``/``border`` the summary
    polygon (``col`` of the numeric form colours each study); ``addpred``/``predstyle`` add a
    prediction interval to the summary (``'line'``, ``'polygon'``, ``'bar'``, ``'shade'`` (rectangles shaded by the
    density of the predictive distribution) or ``'dist'`` (the density itself, with the tails beyond the interval and
    the parts either side of ``predtail`` shaded)). ``predlim`` limits the distribution drawn by ``'dist'`` and
    ``preddist`` supplies a distribution (a tuple of the values and densities, or a mapping with the keys
    ``'x'``, ``'y'`` and optionally ``'pi_lb'``, ``'pi_ub'`` and ``'level'``); ``'shade'`` and ``'dist'`` need the ``transf`` to be
    one of ``numpy.exp``, ``transf_ztor``, ``numpy.tanh``, ``transf_ilogit``, ``plogis``, ``transf_iarcsin``,
    ``transf_iprobit``, ``pnorm``, ``transf_iahw`` or ``transf_iabt``. ``fonts`` sets the font family and font of the
    study labels, the annotations and the ``ilab`` columns (see :func:`metafopy.plot.draw.parse_fonts`: a family such as
    ``'serif'``, ``'sans'`` or ``'mono'``, a ``(family, font)`` pair with font 1 to 4 for plain, bold, italic, bold italic, a
    ``{family: font}`` mapping, or a sequence of up to three of these). The remaining metafor options passed
    through ``...`` are the keyword-only ``predtail``, ``annosym``, ``textpos``, ``rowadj``, ``top``, ``at_lab``,
    ``xlabadj``, ``xlabfont`` (one font or two, the second for the middle one of three labels) and ``decreasing``;
    for the numeric form ``measure`` (or the ``attrs['measure']`` of a pandas Series) sets the label of the axis and header.

    Indices are 0-based where they refer to positions; ``rows`` are plot rows as in R. Vector
    arguments (``slab``, ``ilab``, ``psize``, ...) must have one entry per study of the fit (or per
    supplied study if some were dropped for missing values).

    Returns a :class:`ForestResult` (metafor returns the same values invisibly); the markers, CI
    segments, polygons and texts are ordinary matplotlib artists on ``result.ax``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    is_fit = isinstance(x, metafopy.models.rma_uni.RMAResult)
    if transf is not None and atransf is not None:
        raise metafopy.errors.InputError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    axx = metafopy.plot.rutils.get_ax(ax)
    dig = [digits] * 3 if isinstance(digits, int) else list(digits)
    drop0 = all(isinstance(v, int | numpy.integer) for v in dig)  # R: drop trailing zeros of the axis labels only if digits is an integer (2L)
    if len(dig) == 1:
        dig = dig * 3
    if len(dig) == 2:
        dig = [*dig, dig[0]]
    ef = floats(efac)
    if len(ef) == 1:
        ef = ef * 4
    if is_fit:
        if len(ef) == 2:
            ef = [ef[0], ef[0], ef[1], ef[1]]
        if len(ef) == 3:
            ef = [*ef, ef[2]]
    else:
        if len(ef) == 2:
            ef = [*ef, 1.0, 1.0]
        if len(ef) == 3:
            ef = [*ef, ef[2]]
    efv = [math.nan if v == 0 else v for v in ef]
    asym = anno_symbol(annosym)
    ol: tuple[float, float] | None = None
    if olim is not None:
        if len(olim) != 2:
            raise metafopy.errors.InputError("Argument 'olim' must be of length 2.")
        ol = (float(min(olim)), float(max(olim)))
    if rowadj is None:
        radj = [0.0, 0.0, 0.0]
    else:
        radj = floats(rowadj)
        if len(radj) == 1:
            radj = [radj[0], radj[0], 0.0]
        if len(radj) == 2:
            radj.append(0.0)
    if xlabadj is None:
        xladj: list[float | None] = [None, None]
    else:
        xla = floats(xlabadj)
        if len(xla) == 1:
            xla = [xla[0], 1 - xla[0]]
        xladj = [xla[0], xla[1]]

    # -- inputs -------------------------------------------------------------------------------------------
    pred_ci_lb: FloatArray
    pred_ci_ub: FloatArray
    pred: FloatArray
    X: FloatArray | None = None
    ci_lb_a: FloatArray
    ci_ub_a: FloatArray
    int_only = False
    res: metafopy.models.rma_uni.RMAResult | None = None
    sort_extra: dict[str, FloatArray] = {}
    if is_fit:
        res = x  # type: ignore[assignment]
        assert res is not None
        lvl = res.level if level is None else metafopy.plot.rutils.level_to_alpha(level)
        yi = res.yi.copy()
        vi_a = res.vi.copy()
        X = res.X
        k = int(yi.shape[0])
        k_all = res.data.k_all
        keep = res.data.keep
        int_only = res.int_only
        measure = res.measure
        slab_null = False
        # slab: stored labels; metafor uses "Study <id>" when no slab was given
        slab_l: list[typing.Any] = list(res.slab)
        if slab is None:
            slab_null = all(s == str(i + 1) for s, i in zip(res.slab, numpy.flatnonzero(keep), strict=True))
            if slab_null:
                slab_l = [f'Study {i + 1}' for i in numpy.flatnonzero(keep)]
        else:
            slab_l = per_study(list(slab) if not isinstance(slab, float) else [slab], keep, k, k_all, 'slab')
            slab_null = False
        ilab_m = ilab_matrix(ilab)
        if ilab_m is not None:
            ilab_m = per_study(ilab_m, keep, k, k_all, 'ilab')
        pch_l = pervec(pch, keep, k, k_all, 'pch')
        psize_l = None if psize is None else pervec(psize, keep, k, k_all, 'psize')
        colout_l = pervec(metafopy.plot.draw.FG if colout is None else colout, keep, k, k_all, 'colout')
        colci_l = colout_l if colci is None else pervec(colci, keep, k, k_all, 'colci')
        pr = None if int_only else metafopy.methods.predict.predict(res, level=level)
        wts = res.weights_()
        if int_only:
            pred = numpy.full(k, float(res.beta[0]))
            pred_ci_lb = numpy.full(k, numpy.nan)
            pred_ci_ub = numpy.full(k, numpy.nan)
        else:
            assert pr is not None
            pred = pr.pred
            use_pi = bool(addpred)
            pred_ci_lb = pr.pi_lb if use_pi else pr.ci_lb
            pred_ci_ub = pr.pi_ub if use_pi else pr.ci_ub
        resid_z = None
        if order is not None and isinstance(order, str) and order in ('rstandard', 'abs.rstandard'):
            resid_z = metafopy.methods.influence.rstandard(res).z
        if resid_z is not None:
            sort_extra['z'] = resid_z
        shade_a = shade
        sh_k_all = k_all
        crit0 = float(scipy.stats.norm.isf(lvl / 2))
        ci_lb_a = yi - crit0 * numpy.sqrt(vi_a)
        ci_ub_a = yi + crit0 * numpy.sqrt(vi_a)
    else:
        yi = numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64)).copy()
        k = int(yi.shape[0])
        lvl = 0.05 if level is None else metafopy.plot.rutils.level_to_alpha(level)
        crit = float(scipy.stats.norm.isf(lvl / 2))
        if ci_lb is not None and ci_ub is not None:
            lb0 = numpy.atleast_1d(numpy.asarray(ci_lb, dtype=numpy.float64))
            ub0 = numpy.atleast_1d(numpy.asarray(ci_ub, dtype=numpy.float64))
            if lb0.shape != ub0.shape:
                raise metafopy.errors.InputError("Length of 'ci_lb' and 'ci_ub' are not the same.")
            if vi is None and sei is None:
                vi_a = ((ub0 - lb0) / (2 * crit)) ** 2
            else:
                vi_a = numpy.asarray(vi, dtype=numpy.float64) if vi is not None else numpy.asarray(sei, dtype=numpy.float64) ** 2
            if lb0.shape[0] != vi_a.shape[0]:
                raise metafopy.errors.InputError("Length of 'vi' (or 'sei') does not match the length of ('ci_lb','ci_ub').")
            ci_lb_a, ci_ub_a = lb0, ub0
        else:
            if vi is None:
                if sei is None:
                    raise metafopy.errors.InputError("Must specify either 'vi', 'sei', or ('ci_lb','ci_ub').")
                vi_a = numpy.asarray(sei, dtype=numpy.float64) ** 2
            else:
                vi_a = numpy.asarray(vi, dtype=numpy.float64)
            if yi.shape[0] != vi_a.shape[0]:
                raise metafopy.errors.InputError("Length of 'vi' (or 'sei') does not match the length of 'yi'.")
            ci_lb_a = yi - crit * numpy.sqrt(vi_a)
            ci_ub_a = yi + crit * numpy.sqrt(vi_a)
        vi_a = numpy.atleast_1d(vi_a).astype(numpy.float64)
        attrs = getattr(x, 'attrs', {})
        if measure is None:
            measure = str(attrs['measure']) if isinstance(attrs, dict) and attrs.get('measure') else 'GEN'
        slab_null = False
        if slab is None:
            slab_l = [f'Study {i + 1}' for i in range(k)]
            slab_null = True
        else:
            if numpy.ndim(slab) == 0 and not isinstance(slab, str) and math.isnan(float(slab)):  # type: ignore[arg-type]
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                slab_l = [''] * k  # slab=NA in R: no study labels
                slab_null = True
            else:
                slab_l = list(slab)  # type: ignore[arg-type]
        if len(slab_l) != k:
            raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(slab_l)}) does not correspond to the number of outcomes ({k}).")
        ilab_m = ilab_matrix(ilab)
        if ilab_m is not None and ilab_m.shape[0] != k:
            raise metafopy.errors.InputError(f"Length of the 'ilab' argument ({ilab_m.shape[0]}) does not correspond to the number of outcomes ({k}).")
        pch_l = expand(pch, k, 'pch')
        psize_l = None if psize is None else expand(psize, k, 'psize')
        col_l = expand(metafopy.plot.draw.FG if col is None else col, k, 'col')
        colci_l = col_l if colci is None else expand(colci, k, 'colci')
        colout_l = col_l
        wts = numpy.full(k, numpy.nan)
        pred = numpy.full(k, numpy.nan)
        pred_ci_lb = numpy.full(k, numpy.nan)
        pred_ci_ub = numpy.full(k, numpy.nan)
        shade_a = shade
        sh_k_all = k
        keep = numpy.ones(k, dtype=numpy.bool_)
        k_all = k
    if subset is not None and is_fit:
        raise metafopy.errors.InputError('Function does not have a subset argument.')

    # -- shade type ------------------------------------------------------------------------------------------
    shade_type = 'none'
    shade_chr = ''
    shade_arr: typing.Any = None
    if isinstance(shade_a, str):
        shade_type, shade_chr = 'character', shade_a
        if shade_chr not in ('zebra', 'zebra1', 'zebra2', 'all'):
            raise metafopy.errors.InputError("Unknown option specified for 'shade' argument.")
    elif isinstance(shade_a, bool | numpy.bool_):
        shade_type, shade_chr = 'character', 'zebra'
    elif shade_a is not None:
        sa = numpy.asarray(shade_a)
        if sa.dtype == numpy.bool_:
            shade_type = 'logical'
            shade_arr = numpy.asarray(per_study(list(sa), keep, k, sh_k_all, 'shade'), dtype=bool) if is_fit else sa
        else:
            shade_type = 'numeric'
            shade_arr = sa.astype(float)

    # -- subset / order --------------------------------------------------------------------------------------
    def reorder(idx: numpy.typing.NDArray[numpy.intp]) -> None:
        nonlocal yi, vi_a, X, slab_l, ilab_m, pred, pred_ci_lb, pred_ci_ub, wts, pch_l, psize_l, colout_l, colci_l, shade_arr, ci_lb_a, ci_ub_a
        yi, vi_a = yi[idx], vi_a[idx]
        if X is not None:
            X = X[idx]
        slab_l = [slab_l[i] for i in idx]
        if ilab_m is not None:
            ilab_m = ilab_m[idx]
        pred, pred_ci_lb, pred_ci_ub, wts = pred[idx], pred_ci_lb[idx], pred_ci_ub[idx], wts[idx]
        pch_l = [pch_l[i] for i in idx]
        if psize_l is not None:
            psize_l = [psize_l[i] for i in idx]
        colout_l = [colout_l[i] for i in idx]
        colci_l = [colci_l[i] for i in idx]
        if shade_type == 'logical':
            shade_arr = shade_arr[idx]
        ci_lb_a, ci_ub_a = ci_lb_a[idx], ci_ub_a[idx]

    sub_mask: numpy.typing.NDArray[numpy.bool_] | None = None
    if subset is not None:
        sa_ = numpy.asarray(subset)
        sub_mask = sa_.astype(bool) if sa_.dtype == numpy.bool_ else numpy.isin(numpy.arange(k), sa_.astype(numpy.intp))
    if not is_fit:
        if order is not None:
            if isinstance(order, str):
                if order not in ('est', 'obs', 'yi', 'prec', 'vi'):
                    raise metafopy.errors.InputError("'order' should be one of 'est', 'obs', 'yi', 'prec', 'vi'.")
                # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
                idx = numpy.argsort(yi, kind='stable') if order in ('est', 'obs', 'yi') else lexorder(vi_a, yi)
            else:
                idx = order_by(order, decreasing, k)
            reorder(idx)
            if sub_mask is not None:
                sub_mask = sub_mask[idx]
        if sub_mask is not None:
            reorder(numpy.flatnonzero(sub_mask).astype(numpy.intp))
    elif order is not None:
        assert res is not None
        if isinstance(order, str):
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            if order in ('est', 'obs', 'yi'):
                idx = numpy.argsort(yi, kind='stable')
            elif order == 'fit':
                idx = numpy.argsort(pred, kind='stable')
            elif order in ('prec', 'vi'):
                idx = lexorder(vi_a, yi)
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            elif order == 'resid':
                idx = lexorder(yi - pred, yi)
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            elif order == 'rstandard':
                idx = lexorder(sort_extra['z'], yi)
            elif order == 'abs.resid':
                idx = lexorder(numpy.abs(yi - pred), yi)
            elif order == 'abs.rstandard':
                idx = lexorder(numpy.abs(sort_extra['z']), yi)
            else:
                raise metafopy.errors.InputError(f'unknown order {order!r}')
        else:
            idx = order_by(per_study(list(numpy.asarray(order)), keep, k, k_all, 'order'), decreasing, k)
        reorder(idx)
    k = int(yi.shape[0])

    # -- rows ----------------------------------------------------------------------------------------------
    if rows is None:
        rows_a = numpy.arange(k, 0, -1, dtype=float)
    elif numpy.ndim(rows) == 0:
        r0 = float(rows)  # type: ignore[arg-type]
        rows_a = numpy.arange(r0, r0 - k, -1, dtype=float)
    else:
        rows_a = numpy.asarray(rows, dtype=float)
    if rows_a.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of the 'rows' argument ({rows_a.shape[0]}) does not correspond to the number of outcomes ({k}).")
    rev = numpy.arange(k)[::-1]
    yi, vi_a = yi[rev], vi_a[rev]
    if X is not None:
        X = X[rev]
    slab_l = [slab_l[i] for i in rev]
    if ilab_m is not None:
        ilab_m = ilab_m[rev]
    pred, pred_ci_lb, pred_ci_ub, wts = pred[rev], pred_ci_lb[rev], pred_ci_ub[rev], wts[rev]
    pch_l = [pch_l[i] for i in rev]
    if psize_l is not None:
        psize_l = [psize_l[i] for i in rev]
    colout_l = [colout_l[i] for i in rev]
    colci_l = [colci_l[i] for i in rev]
    rows_a = rows_a[rev]
    if shade_type == 'logical':
        shade_arr = shade_arr[rev]
    ci_lb_a, ci_ub_a = ci_lb_a[rev], ci_ub_a[rev]
    na_mask = numpy.isnan(yi) | numpy.isnan(vi_a)
    if X is not None:
        na_mask |= numpy.isnan(X).any(axis=1)
    if na_mask.any():
        ok = ~na_mask
        yi, vi_a = yi[ok], vi_a[ok]
        if X is not None:
            X = X[ok]
        slab_l = [s for s, o in zip(slab_l, ok, strict=True) if o]
        if ilab_m is not None:
            ilab_m = ilab_m[ok]
        pred, pred_ci_lb, pred_ci_ub, wts = pred[ok], pred_ci_lb[ok], pred_ci_ub[ok], wts[ok]
        pch_l = [s for s, o in zip(pch_l, ok, strict=True) if o]
        if psize_l is not None:
            psize_l = [s for s, o in zip(psize_l, ok, strict=True) if o]
        colout_l = [s for s, o in zip(colout_l, ok, strict=True) if o]
        colci_l = [s for s, o in zip(colci_l, ok, strict=True) if o]
        if shade_type == 'logical':
            shade_arr = shade_arr[ok]
        ci_lb_a, ci_ub_a = ci_lb_a[ok], ci_ub_a[ok]
        rows_a = compress_rows(rows_a, ok)
    k = int(yi.shape[0])

    crit = float(scipy.stats.norm.isf(lvl / 2))
    # -- transformation --------------------------------------------------------------------------------------
    if transf is not None:
        yi = metafopy.methods.common.apply_each(yi, transf, targs)
        ci_lb_a = metafopy.methods.common.apply_each(ci_lb_a, transf, targs)
        ci_ub_a = metafopy.methods.common.apply_each(ci_ub_a, transf, targs)
        pred = metafopy.methods.common.apply_each(pred, transf, targs)
        pred_ci_lb = metafopy.methods.common.apply_each(pred_ci_lb, transf, targs)
        pred_ci_ub = metafopy.methods.common.apply_each(pred_ci_ub, transf, targs)
    ci_lb_a, ci_ub_a = metafopy.methods.common.psort(ci_lb_a, ci_ub_a)
    pred_ci_lb, pred_ci_ub = metafopy.methods.common.psort(pred_ci_lb, pred_ci_ub)
    if ol is not None:
        yi, ci_lb_a, ci_ub_a = (apply_olim(v, ol) for v in (yi, ci_lb_a, ci_ub_a))
        pred, pred_ci_lb, pred_ci_ub = (apply_olim(v, ol) for v in (pred, pred_ci_lb, pred_ci_ub))

    if showweights and not is_fit:
        wts = 1 / vi_a
        wts = 100 * wts / numpy.nansum(wts)

    # -- point sizes ------------------------------------------------------------------------------------------
    if psize_l is None:
        if is_fit:
            psz = point_sizes(numpy.sqrt(wts), plim, k)
        elif numpy.any(vi_a <= 0):
            psz = numpy.ones(k)
        else:
            psz = point_sizes(1 / numpy.sqrt(vi_a), plim, k)
    else:
        psz = numpy.asarray(psize_l, dtype=float)

    # -- alim / at -----------------------------------------------------------------------------------------------
    at_in: FloatArray | None = None
    if at is not None:
        at_in = numpy.asarray(at, dtype=float)
        if numpy.isnan(at_in).any():
            raise metafopy.errors.InputError("Argument 'at' cannot contain NAs.")
        if numpy.isinf(at_in).any():
            raise metafopy.errors.InputError("Argument 'at' cannot contain +-Inf values.")
    alim_spec = True
    lbmin, ubmax = float(numpy.nanmin(ci_lb_a)), float(numpy.nanmax(ci_ub_a))
    if alim is None:
        if at_in is None:
            pr_ = metafopy.plot.rutils.pretty([lbmin, ubmax], steps - 1)
            alim_t = (float(pr_.min()), float(pr_.max()))
            alim_spec = False
        else:
            alim_t = (float(at_in.min()), float(at_in.max()))
    else:
        if len(alim) != 2:
            raise metafopy.errors.InputError("Argument 'alim' must be of length 2.")
        alim_t = (float(min(alim)), float(max(alim)))
    if math.isnan(alim_t[0]) or math.isnan(alim_t[1]):
        raise metafopy.errors.InputError("Argument 'alim' cannot contain NAs.")
    if at_in is None:
        at_a = numpy.linspace(alim_t[0], alim_t[1], steps) if alim_spec else metafopy.plot.rutils.pretty([lbmin, ubmax], steps - 1)
    else:
        at_a = numpy.array(list(dict.fromkeys(numpy.clip(at_in, alim_t[0], alim_t[1]).tolist())), dtype=float)
    if at_lab is None:
        if atransf is not None:
            at_lab_l = [metafopy.plot.rutils.fmtx1(float(v), dig[1], drop0) for v in metafopy.methods.common.apply_each(at_a, atransf, targs)]
        else:
            at_lab_l = [metafopy.plot.rutils.fmtx1(float(v), dig[1], drop0) for v in at_a]
    else:
        at_lab_l = list(at_lab)

    # -- areas / xlim ----------------------------------------------------------------------------------------------
    ncol_ilab = 0 if ilab_m is None else int(ilab_m.shape[1])
    area_slab = 25.0 if slab_null else 40.0
    area_anno = (30.0 if showweights else 25.0) if annotate else 10.0
    iadd = 5.0
    area_slab += iadd * ncol_ilab
    area_forest = 100 + iadd * ncol_ilab - area_slab - area_anno
    area_slab /= 100 + iadd * ncol_ilab
    area_anno /= 100 + iadd * ncol_ilab
    area_forest /= 100 + iadd * ncol_ilab
    mult_l = area_slab / area_forest
    mult_r = area_anno / area_forest
    if xlim is None:
        f1 = alim_t[0] if lbmin < alim_t[0] else lbmin
        f2 = alim_t[1] if ubmax > alim_t[1] else ubmax
        rng = f2 - f1
        xl = (metafopy.plot.rutils.rround(f1 - rng * mult_l, dig[1]), metafopy.plot.rutils.rround(f2 + rng * mult_r, dig[1]))
    else:
        if len(xlim) != 2:
            raise metafopy.errors.InputError("Argument 'xlim' must be of length 2.")
        xl = (float(xlim[0]), float(xlim[1]))
    xl = (min(xl), max(xl))
    tp = [xl[0], xl[1]] if textpos is None else [float(textpos[0]), float(textpos[1])]
    if len(tp) != 2:
        raise metafopy.errors.InputError("Argument 'textpos' must be of length 2.")
    if math.isnan(tp[0]):
        tp[0] = xl[0]
    if math.isnan(tp[1]):
        tp[1] = xl[1]

    pstyle: PredStyle
    if predstyle is None:
        pstyle = 'line'
        if addpred is None:
            addpred = False
    else:
        pstyle = check_pred(predstyle)
        addpred = True  # specifying predstyle sets addpred=TRUE automatically
        if is_fit and not int_only and pstyle != 'line':
            warnings.warn("Argument 'predstyle' not relevant for meta-regression models.", stacklevel=2)
    fnt = metafopy.plot.draw.parse_fonts(fonts, 3)
    pdist = parse_preddist(preddist)
    fit_row = is_fit and int_only and addfit
    mrk = max(rows_a) if rows_a.size else 0.0
    if ylim is None:
        yl = (-2.0 - (0.0 if pstyle == 'line' else 1.0), mrk + top) if fit_row else (0.0, mrk + top)
    elif len(ylim) == 1:
        yl = (float(ylim[0]), mrk + top)
    else:
        yl = (float(min(ylim)), float(max(ylim)))

    # -- the plot ----------------------------------------------------------------------------------------------------
    metafopy.plot.draw.only_xaxis(axx)
    axx.set_xlim(xl)
    axx.set_ylim(yl)
    axx.set_autoscale_on(False)
    colshade_c = metafopy.plot.rutils.coladj(-0.1) if colshade is None else colshade
    if shade_type == 'character':
        if shade_chr in ('zebra', 'zebra1'):
            sh = numpy.resize(numpy.array([True, False]), k)
        elif shade_chr == 'zebra2':
            sh = numpy.resize(numpy.array([False, True]), k)
        else:
            sh = numpy.ones(k, dtype=bool)
        shade_arr = sh
    if shade_type in ('character', 'logical'):
        for i in range(k):
            if shade_arr[i]:
                metafopy.plot.draw.rect(axx, xl[0], rows_a[i] - 0.5, xl[1], rows_a[i] + 0.5, col=colshade_c, border=colshade_c, zorder=0.5)
    if shade_type == 'numeric':
        for s in shade_arr:
            metafopy.plot.draw.rect(axx, xl[0], float(s) - 0.5, xl[1], float(s) + 0.5, col=colshade_c, border=colshade_c, zorder=0.5)
    if lty is None:
        lt = ['solid', 'dotted', 'solid'] if is_fit else ['solid', 'solid']
    else:
        lt = [lty] if isinstance(lty, str) else list(lty)
        if is_fit:
            if len(lt) == 1:
                lt = [lt[0], 'dotted', 'solid']
            if len(lt) == 2:
                lt = [*lt, 'solid']
        elif len(lt) == 1:
            lt = [lt[0], 'solid']
    hl_lty = lt[2] if is_fit else lt[1]
    metafopy.plot.draw.hline(axx, yl[1] - (top - 1), lty=hl_lty)
    if refline is not None:
        metafopy.plot.draw.seg(axx, refline, yl[0], refline, yl[1] - (top - 1), lty='dotted')
    height = yl[1] - yl[0]
    cex_adj = 1.0
    if cex is None:
        lheight = strheight(axx, yl)
        cex_adj = height / (1.25 * k * lheight) if k * lheight > height * 0.8 else 1.0
        cx = cex_adj
        cex_l = cex_adj if cex_lab is None else cex_lab
        cex_a = cex_adj if cex_axis is None else cex_axis
    else:
        cx = cex
        cex_l = cex if cex_lab is None else cex_lab
        cex_a = cex if cex_axis is None else cex_axis
    del cex_adj

    polheight = (height / 100) * cx * efv[2]
    if is_fit and addfit and not int_only:
        col_mr = metafopy.plot.rutils.coladj(-0.2) if col is None else (col if isinstance(col, str) else col[0])
        bord_mr = metafopy.plot.rutils.coladj(-0.3) if border is None else (border if isinstance(border, str) else border[0])
        for i in range(k):
            if math.isnan(pred[i]):
                continue
            xs, ys = diamond(pred_ci_lb[i], pred[i], pred_ci_ub[i], rows_a[i], polheight)
            sel = (xs > alim_t[0]) & (xs < alim_t[1])
            metafopy.plot.draw.poly(axx, xs[sel], ys[sel], col=col_mr, border=bord_mr)
    ciendheight = height / 150 * cx * efv[0]
    arrowwidth = 1.4 / 100 * cx * (xl[1] - xl[0])
    arrowheight = height / 150 * cx * efv[1]
    barheight = min(0.25, height / 150 * cx * efv[3])
    pipolheight = (height / 100) * cx * efv[3]

    beta = beta_lb = beta_ub = pi_lb = pi_ub = math.nan
    if fit_row:
        assert res is not None
        pr2 = metafopy.methods.predict.predict(res, level=100 * (1 - lvl))
        beta, beta_lb, beta_ub = float(pr2.pred[0]), float(pr2.ci_lb[0]), float(pr2.ci_ub[0])
        pi_lb, pi_ub = float(pr2.pi_lb[0]), float(pr2.pi_ub[0])
        predlevel = lvl
        if pdist is not None:
            pdxs, pdys, pextra = pdist
            cdf = numpy.cumsum(numpy.diff(pdxs) * (pdys[1:] + pdys[:-1]) / 2)
            cdf = cdf / numpy.max(cdf)
            pi_lb = float(pextra['pi_lb']) if 'pi_lb' in pextra else float(pdxs[numpy.argmin(numpy.abs(cdf - lvl / 2))])
            pi_ub = float(pextra['pi_ub']) if 'pi_ub' in pextra else float(pdxs[numpy.argmin(numpy.abs(cdf - (1 - lvl / 2)))])
            if 'level' in pextra:
                predlevel = metafopy.plot.rutils.level_to_alpha(float(pextra['level']))
        if transf is not None:
            beta, beta_lb, beta_ub, pi_lb, pi_ub = (
                float(metafopy.methods.common.apply_each(numpy.array([v]), transf, targs)[0]) for v in (beta, beta_lb, beta_ub, pi_lb, pi_ub)
            )
        beta_lb, beta_ub = (float(v[0]) for v in metafopy.methods.common.psort(numpy.array([beta_lb]), numpy.array([beta_ub])))
        pi_lb, pi_ub = (float(v[0]) for v in metafopy.methods.common.psort(numpy.array([pi_lb]), numpy.array([pi_ub])))
        if ol is not None:
            beta, beta_lb, beta_ub, pi_lb, pi_ub = (float(apply_olim(numpy.array([v]), ol)[0]) for v in (beta, beta_lb, beta_ub, pi_lb, pi_ub))
        colv = summary_cols(col, border, pstyle)
        pcol, pborder = colv
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if res.method not in ('FE', 'EE', 'CE') and addpred:
            curve: PredCurve | None = None
            if pstyle in DENSITY_STYLES:
                assert pr2.pi_se is not None
                pddf = None if pr2.pi_ddf is None else float(numpy.atleast_1d(pr2.pi_ddf)[0])
                curve = pred_curve(
                    pstyle,
                    float(pr2.pred[0]),
                    float(pr2.pi_se[0]),
                    pr2.pi_dist or 'norm',
                    pddf,
                    lvl,
                    transf,
                    predlim,
                    None if pdist is None else (pdist[0], pdist[1]),
                    predtail,
                )
            draw_pred(
                axx,
                pstyle,
                pi_lb,
                pi_ub,
                beta,
                -1.0 if pstyle == 'line' else -2.0,
                PredGeometry(alim_t, ciendheight, arrowwidth, arrowheight, barheight, pipolheight, efv[3], ol),
                lt[1],
                pcol,
                pborder,
                curve,
            )
        xs, ys = diamond(beta_lb, beta, beta_ub, -1.0, polheight)
        sel = (xs > alim_t[0]) & (xs < alim_t[1])
        metafopy.plot.draw.poly(axx, xs[sel], ys[sel], col=pcol[0], border=pborder[0])
        ml = [default_mlab(res.method)] if mlab is None else ([mlab] if isinstance(mlab, str) else list(mlab))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if len(ml) == 1 and pstyle != 'line':
            what = 'Predictive Distribution' if pstyle == 'dist' else 'Prediction Interval'
            ml.append(f'{what}{asym[0]}{metafopy.plot.rutils.rround(100 * (1 - predlevel), dig[0]):g}% PI{asym[2]}')
        ftxt(axx, tp[0], -1 + radj[0], ml[0], fnt[0], pos=4, cex=cx)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if pstyle != 'line':
            ftxt(axx, tp[0], -2 + radj[0], ml[1], fnt[0], pos=4, cex=cx)

    metafopy.plot.draw.xaxis(axx, at_a, at_lab_l, cex=cex_a, family=fnt[0][0])
    measure_r = measure
    if xlab is None:
        xlab_l = [metafopy.plot.rutils.setlab(measure_r, transf, atransf, 1)]
    else:
        xlab_l = [xlab] if isinstance(xlab, str) else list(xlab)
    if len(xlab_l) not in (1, 2, 3):
        raise metafopy.errors.InputError("Argument 'xlab' argument must be of length 1, 2, or 3.")
    amin, amax = float(at_a.min()), float(at_a.max())
    amid = amin + (amax - amin) / 2
    # the font of the left, right and single label is xlabfont[1], that of the middle of three labels xlabfont[2] (the font set by fonts if only one is given)
    xlf = [int(xlabfont)] if isinstance(xlabfont, int | numpy.integer) else [int(v) for v in xlabfont]
    xlf = [xlf[0], xlf[1] if len(xlf) > 1 else (xlf[0] if len(xlab_l) != 3 else fnt[0][1])]
    ha_l = 'center' if xladj[0] is None else ('left' if xladj[0] == 0 else ('right' if xladj[0] == 1 else 'center'))
    ha_r = 'center' if xladj[1] is None else ('left' if xladj[1] == 0 else ('right' if xladj[1] == 1 else 'center'))
    if len(xlab_l) == 1:
        metafopy.plot.draw.xlab_below(axx, amid, xlab_l[0], cex=cex_l, family=fnt[0][0], font=xlf[0])
    elif len(xlab_l) == 2:
        metafopy.plot.draw.xlab_below(axx, amin, xlab_l[0], cex=cex_l, ha=ha_l, family=fnt[0][0], font=xlf[0])
        metafopy.plot.draw.xlab_below(axx, amax, xlab_l[1], cex=cex_l, ha=ha_r, family=fnt[0][0], font=xlf[0])
    else:
        metafopy.plot.draw.xlab_below(axx, amin, xlab_l[0], cex=cex_l, ha=ha_l, family=fnt[0][0], font=xlf[0])
        metafopy.plot.draw.xlab_below(axx, amid, xlab_l[1], cex=cex_l, family=fnt[0][0], font=xlf[1])
        metafopy.plot.draw.xlab_below(axx, amax, xlab_l[2], cex=cex_l, ha=ha_r, family=fnt[0][0], font=xlf[0])

    for i in range(k):
        ci_c = colci_l[i]
        if numpy.isnan(yi[i]) or numpy.isnan(ci_lb_a[i]) or numpy.isnan(ci_ub_a[i]) or ci_c is None:
            continue
        lo_i, hi_i = float(ci_lb_a[i]), float(ci_ub_a[i])
        if lo_i >= alim_t[1]:
            arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
            continue
        if hi_i <= alim_t[0]:
            arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
            continue
        metafopy.plot.draw.seg(axx, max(lo_i, alim_t[0]), rows_a[i], min(hi_i, alim_t[1]), rows_a[i], lty=lt[0], col=ci_c)
        if lo_i >= alim_t[0]:
            metafopy.plot.draw.seg(axx, lo_i, rows_a[i] - ciendheight, lo_i, rows_a[i] + ciendheight, col=ci_c)
        else:
            arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
        if hi_i <= alim_t[1]:
            metafopy.plot.draw.seg(axx, hi_i, rows_a[i] - ciendheight, hi_i, rows_a[i] + ciendheight, col=ci_c)
        else:
            arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)

    for i in range(k):
        ftxt(axx, tp[0], rows_a[i] + radj[0], astr(slab_l[i]), fnt[0], pos=4, cex=cx, col=colout_l[i] if not is_fit else metafopy.plot.draw.FG)

    ilab_xpos_out: FloatArray | None = None
    ilab_pos_out: list[int] | None = None
    if ilab_m is not None:
        if ilab_xpos is None:
            dist = float(numpy.nanmin(ci_lb_a)) - xl[0]
            if ncol_ilab == 1:
                ixp = numpy.array([xl[0] + dist * 0.75])
            elif ncol_ilab == 2:
                ixp = xl[0] + dist * numpy.array([0.65, 0.85])
            elif ncol_ilab == 3:
                ixp = xl[0] + dist * numpy.array([0.6, 0.75, 0.9])
            else:
                ixp = numpy.linspace(xl[0] + dist * 0.5, xl[0] + dist * 0.9, ncol_ilab)
        else:
            ixp = numpy.atleast_1d(numpy.asarray(ilab_xpos, dtype=float))
        if ixp.shape[0] != ncol_ilab:
            raise metafopy.errors.InputError(f"Number of 'ilab' columns ({ncol_ilab}) does not match the length of the 'ilab_xpos' argument ({ixp.shape[0]}).")
        ilab_xpos_out = ixp
        ipos: list[int | None]
        if ilab_pos is None:
            ipos = [None] * ncol_ilab
        else:
            ipos = [int(ilab_pos)] * ncol_ilab if numpy.ndim(ilab_pos) == 0 else [int(v) for v in numpy.atleast_1d(ilab_pos)]  # type: ignore[arg-type]
            ilab_pos_out = [int(v) for v in ipos if v is not None]
        if ilab_lab is not None and len(ilab_lab) != ncol_ilab:
            raise metafopy.errors.InputError(f"Number of 'ilab' columns ({ncol_ilab}) does not match the length of the 'ilab_lab' argument ({len(ilab_lab)}).")
        for lcol in range(ncol_ilab):
            for i in range(k):
                ftxt(axx, float(ixp[lcol]), rows_a[i] + radj[2], astr(ilab_m[i, lcol]), fnt[2], pos=ipos[lcol], cex=cx)
            if ilab_lab is not None:
                ftxt(axx, float(ixp[lcol]), yl[1] - (top - 1) + 1 + radj[2], ilab_lab[lcol], fnt[2], pos=ipos[lcol], cex=cx, bold=True)

    width_out: list[int] | None = None
    if annotate:
        est_all, lb_all, ub_all = yi, ci_lb_a, ci_ub_a
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        pi_row = fit_row and pstyle != 'line'
        wts_all: FloatArray | None = None
        pad: tuple[int, ...] = ()
        if fit_row:
            est_all = numpy.concatenate([yi, [beta], [math.nan] if pi_row else []])
            lb_all = numpy.concatenate([ci_lb_a, [beta_lb], [pi_lb] if pi_row else []])
            ub_all = numpy.concatenate([ci_ub_a, [beta_ub], [pi_ub] if pi_row else []])
        if showweights:
            if fit_row:
                wts_all = numpy.concatenate([wts, [100.0], [math.nan] if pi_row else []])
                pad = (k,)
            else:
                wts_all = wts
        raw, width_out = build_annotext(est_all, lb_all, ub_all, dig, asym, width, wts_all, pi_row, atransf, targs, pad)
        annotext = finish_annotext(raw, asym, False, pi_row)
        ypos = list(rows_a + radj[1])
        if fit_row:
            ypos += [-1 + radj[1]] + ([-2 + radj[1]] if pi_row else [])
        for yy, s in zip(ypos, annotext, strict=True):
            ftxt(axx, tp[1], float(yy), s, fnt[1], pos=2, cex=cx)

    for i in range(k):
        if numpy.isnan(yi[i]):
            continue
        if alim_t[0] <= yi[i] <= alim_t[1]:
            metafopy.plot.draw.pts(axx, yi[i], rows_a[i], pch=int(pch_l[i]), cex=cx * float(psz[i]), col=colout_l[i])
    if fit_row:
        metafopy.plot.draw.hline(axx, 0, lty=lt[2])

    # headers
    est_lab = metafopy.plot.rutils.setlab(measure_r, transf, atransf, 3, short=True)
    ci_pct = f'{metafopy.plot.rutils.rround(100 * (1 - lvl), dig[0]):g}'
    hdr_right: str | None = f'{est_lab}{asym[0]}{ci_pct}% CI{asym[2]}'
    if isinstance(header, bool):
        if header:
            hdr_left: str | None = 'Study'
        else:
            hdr_left, hdr_right = None, None
    elif isinstance(header, str):
        hdr_left = header
    else:
        hdr_left = header[0]
        hdr_right = header[1] if len(header) > 1 else hdr_right
    if not annotate:
        hdr_right = None
    if hdr_left is not None:
        ftxt(axx, tp[0], yl[1] - (top - 1) + 1 + radj[0], hdr_left, fnt[0], pos=4, cex=cx, bold=True)
    if hdr_right is not None:
        ftxt(axx, tp[1], yl[1] - (top - 1) + 1 + radj[1], hdr_right, fnt[0], pos=2, cex=cx, bold=True)

    FOREST_STATE[axx] = {
        'level': lvl,
        'annotate': annotate,
        'digits': dig[0],
        'width': width_out,
        'transf': transf,
        'atransf': atransf,
        'targs': targs,
        'alim': alim_t,
        'olim': ol,
        'efac': ef,
        'rowadj': radj,
        'annosym': asym,
        'cex': cx,
        'textpos': (tp[0], tp[1]),
        'fonts': fnt[:2],
    }
    return ForestResult(
        ax=axx,
        xlim=xl,
        alim=alim_t,
        at=at_a,
        ylim=yl,
        rows=rows_a,
        cex=cx,
        cex_lab=cex_l,
        cex_axis=cex_a,
        ilab_xpos=ilab_xpos_out,
        ilab_pos=ilab_pos_out,
        textpos=(tp[0], tp[1]),
        areas=(area_slab, area_forest, area_anno) if is_fit else None,
    )


def summary_cols(col: typing.Any, border: typing.Any, predstyle: str = 'line') -> tuple[list[str | None], list[str | None]]:
    """The colours of a summary polygon and its prediction interval: ``col`` (polygon, interval, shade end or below ``predtail``, above it) and ``border``."""
    fg = metafopy.plot.draw.FG
    col2 = metafopy.plot.rutils.coladj(-0.6) if predstyle == 'dist' else fg
    col3 = metafopy.plot.rutils.coladj(-0.05) if predstyle == 'shade' else metafopy.plot.rutils.coladj(-0.2)
    c: list[str | None] = [fg, col2, col3, None]
    if col is not None:
        cl = [col] if isinstance(col, str) else list(col)
        if len(cl) == 1:
            cl = [cl[0], col2, col3, None]
        if len(cl) == 2:
            cl = [*cl, col3, None]
        if len(cl) == 3:
            cl = [*cl, None]
        c = cl
    b: list[str | None] = [fg, fg]
    if border is not None:
        bl = [border] if isinstance(border, str) else list(border)
        if len(bl) == 1:
            bl = [bl[0], fg]
        b = bl
    return c, b


@dataclasses.dataclass(frozen=True, slots=True)
class PredGeometry:
    """The sizes and limits used to draw a prediction interval (see :func:`draw_pred`)."""

    alim: tuple[float, float]
    ciendheight: float
    arrowwidth: float
    arrowheight: float
    barheight: float
    pipolheight: float
    ef4: float
    olim: tuple[float, float] | None = None


def draw_pred(
    ax: matplotlib.axes.Axes,
    style: PredStyle,
    lb: float,
    ub: float,
    mid: float,
    row: float,
    geo: PredGeometry,
    lty: str,
    col: collections.abc.Sequence[str | None],
    border: collections.abc.Sequence[str | None],
    curve: PredCurve | None = None,
) -> None:
    """The prediction interval of a summary (``line``, ``polygon``, ``bar``, ``shade`` or ``dist``; ``col`` and ``border`` from :func:`summary_cols`).

    Raises:
        metafopy.errors.InputError: ``shade`` is drawn without the second and third colour.
    """
    alim, aw, ah, bh = geo.alim, geo.arrowwidth, geo.arrowheight, geo.barheight
    c = col[1] if col[1] is not None else metafopy.plot.draw.FG
    if style == 'line':
        metafopy.plot.draw.seg(ax, max(lb, alim[0]), row, min(ub, alim[1]), row, lty=lty, col=c)
        if lb >= alim[0]:
            metafopy.plot.draw.seg(ax, lb, row - geo.ciendheight, lb, row + geo.ciendheight, col=c)
        else:
            arrow(ax, 'l', alim, row, aw, ah, c)
        if ub <= alim[1]:
            metafopy.plot.draw.seg(ax, ub, row - geo.ciendheight, ub, row + geo.ciendheight, col=c)
        else:
            arrow(ax, 'r', alim, row, aw, ah, c)
    elif style == 'polygon':
        xs, ys = diamond(lb, mid, ub, row, geo.pipolheight)
        sel = (xs > alim[0]) & (xs < alim[1])
        metafopy.plot.draw.poly(ax, xs[sel], ys[sel], col=col[1], border=border[1])
    elif style == 'bar':
        if lb >= alim[0]:
            metafopy.plot.draw.rect(ax, lb, row - bh, mid, row + bh, col=col[1], border=border[1])
        else:
            metafopy.plot.draw.rect(ax, alim[0] + aw, row - bh, mid, row + bh, col=col[1], border=border[1])
            metafopy.plot.draw.poly(ax, [alim[0], alim[0] + aw, alim[0] + aw, alim[0]], [row, row + bh, row - bh, row], col=col[1], border=col[1])
        if ub <= alim[1]:
            metafopy.plot.draw.rect(ax, ub, row - bh, mid, row + bh, col=col[1], border=border[1])
        else:
            metafopy.plot.draw.rect(ax, alim[1] - aw, row - bh, mid, row + bh, col=col[1], border=border[1])
            metafopy.plot.draw.poly(ax, [alim[1], alim[1] - aw, alim[1] - aw, alim[1]], [row, row + bh, row - bh, row], col=col[1], border=col[1])
    elif style == 'shade':
        assert curve is not None
        draw_shade(ax, curve, row, bh, alim, geo.olim, col[1], col[2])
    elif style == 'dist':
        assert curve is not None
        draw_dist(ax, curve, row - 0.5, lb, ub, alim, geo.olim, geo.ef4, col, border)
    else:
        typing.assert_never(style)


def per_study(values: typing.Any, keep: numpy.typing.NDArray[numpy.bool_], k: int, k_all: int, name: str) -> typing.Any:
    """Select the entries of a per-study argument that correspond to the studies used in the fit.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    n = len(values) if not isinstance(values, numpy.ndarray) else values.shape[0]
    if n == k:
        return values
    if n == k_all:
        if isinstance(values, numpy.ndarray):
            return values[keep]
        return [v for v, kp in zip(values, keep, strict=True) if kp]
    raise metafopy.errors.InputError(f"Length of the '{name}' argument ({n}) does not correspond to the size of the original dataset ({k_all}).")


def addpoly(
    x: metafopy.models.rma_uni.RMAResult | numpy.typing.ArrayLike,
    vi: numpy.typing.ArrayLike | None = None,
    sei: numpy.typing.ArrayLike | None = None,
    ci_lb: numpy.typing.ArrayLike | None = None,
    ci_ub: numpy.typing.ArrayLike | None = None,
    pi_lb: numpy.typing.ArrayLike | None = None,
    pi_ub: numpy.typing.ArrayLike | None = None,
    rows: numpy.typing.ArrayLike | float | None = -1,
    row: float = -2,
    level: float | None = None,
    annotate: bool | None = None,
    addpred: bool = False,
    predstyle: str | None = None,
    predlim: collections.abc.Sequence[float] | None = None,
    digits: int | None = None,
    width: int | collections.abc.Sequence[int] | None = None,
    mlab: str | collections.abc.Sequence[str] | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    efac: float | collections.abc.Sequence[float] | None = None,
    col: str | collections.abc.Sequence[str] | None = None,
    border: str | collections.abc.Sequence[str] | None = None,
    lty: str | collections.abc.Sequence[str] | None = None,
    fonts: FontsArg = None,
    cex: float | None = None,
    constarea: bool = False,
    *,
    ax: matplotlib.axes.Axes | None = None,
    preddist: PredDist | None = None,
    predtail: float | str | None = None,
    pi_se: float | None = None,
    pi_dist: str = 'norm',
    pi_ddf: float | None = None,
    pi_level: float | None = None,
    alim: collections.abc.Sequence[float] | None = None,
    olim: collections.abc.Sequence[float] | None = None,
    annosym: collections.abc.Sequence[str] | None = None,
    textpos: collections.abc.Sequence[float] | None = None,
    rowadj: float | collections.abc.Sequence[float] | None = None,
    lcol: str | collections.abc.Sequence[str] | None = None,
) -> matplotlib.axes.Axes:
    """Add polygons (summary estimates) to a forest plot (``addpoly.rma`` and ``addpoly.default``).

    For a fitted model without moderators the pooled estimate and its CI are drawn at ``row``
    (default ``-2``; the prediction interval with ``addpred=True`` or ``predstyle``). For numbers
    give the estimates ``x`` with ``vi``/``sei`` or ``ci_lb``/``ci_ub`` (and ``pi_lb``/``pi_ub``)
    and the ``rows`` to draw them in. Unspecified options (``alim``, ``digits``, ``annotate``,
    ``transf``, ``atransf``, ``efac``, ``cex``, ``textpos``, ...) are taken from the last
    :func:`forest` call on the same axes. Returns the axes.

    ``predstyle`` is ``'line'``, ``'polygon'``, ``'bar'``, ``'shade'`` or ``'dist'`` (the last two draw the predictive
    distribution, see :func:`forest`; they need ``predlim``/``preddist``/``predtail`` as there). For numbers the predictive
    distribution is described by the keyword-only ``pi_se`` (its standard error), ``pi_dist`` (``'norm'`` or ``'t'``),
    ``pi_ddf`` (degrees of freedom of the ``'t'``) and ``pi_level`` (the level of the interval; the ``level`` by default),
    which metafor reads from the attributes of ``pi.lb``, or by ``preddist``. ``fonts`` is as in :func:`forest` (the first
    entry for the labels, the second for the annotations) and defaults to that of the last :func:`forest` call.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    axx = metafopy.plot.rutils.get_ax(ax)
    st = FOREST_STATE.get(axx) or {}
    ps: PredStyle = 'line' if predstyle is None else check_pred(predstyle)
    if predstyle is not None:
        addpred = True
    pdist = parse_preddist(preddist)
    if pdist is not None and isinstance(x, metafopy.models.rma_uni.RMAResult):
        raise metafopy.errors.InputError("Can only use 'preddist' with the numeric form (estimates and intervals).")
    fn = metafopy.plot.draw.parse_fonts(st.get('fonts') if fonts is None else fonts, 2)
    if isinstance(x, metafopy.models.rma_uni.RMAResult):
        if not x.int_only:
            raise metafopy.errors.InputError('Fitted model should not contain moderators.')
        lvl_pct = None if level is None else level
        pr = metafopy.methods.predict.predict(x, level=lvl_pct)
        yi = numpy.asarray(pr.pred, dtype=float)
        cl, cu = numpy.asarray(pr.ci_lb, dtype=float), numpy.asarray(pr.ci_ub, dtype=float)
        if addpred:
            pl, pu = numpy.asarray(pr.pi_lb, dtype=float), numpy.asarray(pr.pi_ub, dtype=float)
        else:
            pl = pu = numpy.array([math.nan])
        mlab_use: typing.Any = default_mlab(x.method) if mlab is None else mlab
        rows_use: typing.Any = row
        lvl = x.level if level is None else metafopy.plot.rutils.level_to_alpha(level)
        if pr.pi_se is not None:
            pi_se = float(pr.pi_se[0])
        pi_dist = pr.pi_dist or 'norm'
        pi_ddf = None if pr.pi_ddf is None else float(numpy.atleast_1d(pr.pi_ddf)[0])
        pi_alpha = lvl
    else:
        yi = numpy.atleast_1d(numpy.asarray(x, dtype=float))
        lvl = st.get('level', 0.05) if level is None else metafopy.plot.rutils.level_to_alpha(level)
        crit = float(scipy.stats.norm.isf(lvl / 2))
        if ci_lb is not None and ci_ub is not None:
            cl, cu = numpy.atleast_1d(numpy.asarray(ci_lb, dtype=float)), numpy.atleast_1d(numpy.asarray(ci_ub, dtype=float))
        else:
            v = numpy.asarray(vi, dtype=float) if vi is not None else (numpy.asarray(sei, dtype=float) ** 2 if sei is not None else None)
            if v is None:
                raise metafopy.errors.InputError("Must specify either 'vi', 'sei', or ('ci_lb','ci_ub').")
            cl, cu = yi - crit * numpy.sqrt(v), yi + crit * numpy.sqrt(v)
        pi_alpha = lvl if pi_level is None else metafopy.plot.rutils.level_to_alpha(pi_level)
        if pdist is not None:
            if yi.shape[0] != 1:
                raise metafopy.errors.InputError("Can only use 'preddist' when plotting a single estimate.")
            pdxs, pdys, pextra = pdist
            if 'level' in pextra:
                pi_alpha = metafopy.plot.rutils.level_to_alpha(float(pextra['level']))
            cdf = numpy.cumsum(numpy.diff(pdxs) * (pdys[1:] + pdys[:-1]) / 2)
            cdf = cdf / numpy.max(cdf)
            if 'pi_lb' in pextra:
                pl = numpy.array([float(pextra['pi_lb'])])
            elif pi_lb is not None:
                pl = numpy.atleast_1d(numpy.asarray(pi_lb, dtype=float))
            else:
                pl = numpy.array([float(pdxs[numpy.argmin(numpy.abs(cdf - pi_alpha / 2))])])
            if 'pi_ub' in pextra:
                pu = numpy.array([float(pextra['pi_ub'])])
            elif pi_ub is not None:
                pu = numpy.atleast_1d(numpy.asarray(pi_ub, dtype=float))
            else:
                pu = numpy.array([float(pdxs[numpy.argmin(numpy.abs(cdf - (1 - pi_alpha / 2)))])])
        elif pi_lb is not None and pi_ub is not None:
            pl, pu = numpy.atleast_1d(numpy.asarray(pi_lb, dtype=float)), numpy.atleast_1d(numpy.asarray(pi_ub, dtype=float))
        else:
            if ps != 'line':
                raise metafopy.errors.InputError('Cannot draw prediction interval if pi_lb and pi_ub are unspecified.')
            pl = pu = numpy.full(yi.shape[0], math.nan)
        mlab_use = mlab
        rows_use = rows
    k = int(yi.shape[0])
    alim_t = st.get('alim') if alim is None else (float(min(alim)), float(max(alim)))
    if alim_t is None:
        xl0 = axx.get_xlim()
        alim_t = (float(xl0[0]), float(xl0[1]))
    ol = st.get('olim') if olim is None else (float(min(olim)), float(max(olim)))
    do_annotate = st.get('annotate', True) if annotate is None else annotate
    dg = st.get('digits', 2) if digits is None else digits
    wd = st.get('width') if width is None else width
    tr = st.get('transf') if transf is None else transf
    atr = st.get('atransf') if atransf is None else atransf
    targs_ = st.get('targs') if targs is None else targs
    if tr is not None and atr is not None:
        raise metafopy.errors.InputError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    asym = anno_symbol(st.get('annosym') if annosym is None else annosym)
    if efac is None:
        e0 = st.get('efac', [1.0] * 4)
        ef = [float(v) for v in (numpy.atleast_1d(e0) if numpy.ndim(e0) else [e0])]
        ef = ef * 4 if len(ef) == 1 else ef
    else:
        ef = floats(efac)
        if len(ef) == 1:
            ef = ef * 4
        if ps == 'line':
            if len(ef) == 2:
                ef = [ef[1], ef[1], ef[0], ef[0]]
            if len(ef) == 3:
                ef = [ef[1], ef[2], ef[0], ef[0]]
        else:
            if len(ef) == 2:
                ef = [ef[0], ef[0], ef[0], ef[1]]
            if len(ef) == 3:
                ef = [ef[2], ef[2], ef[0], ef[1]]
    efv = [math.nan if v == 0 else v for v in ef]
    if rowadj is None:
        radj = list(st.get('rowadj', [0.0, 0.0, 0.0]))
    else:
        radj = floats(rowadj)
        if len(radj) == 1:
            radj = [radj[0], radj[0]]
    fg = metafopy.plot.draw.FG
    if k == 1:
        c0, b0 = summary_cols(col, border, ps)
        col_l: list[typing.Any] = c0
        bord_l: list[typing.Any] = b0
    else:
        if ps != 'line':
            raise metafopy.errors.InputError(f"Can only use predstyle='{ps}' when plotting a single polygon.")
        col_l = expand(fg if col is None else col, k, 'col')
        bord_l = expand(fg if border is None else border, k, 'border')
    lcol_l = expand(fg if lcol is None else lcol, k, 'lcol')
    lt = ['dotted', 'solid'] if lty is None else ([lty, 'solid'] if isinstance(lty, str) else list(lty))
    if ps == 'line':
        ml: list[str] | None
        if mlab_use is None:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            ml = [''] * k
        else:
            ml = expand(mlab_use, k, 'mlab')
    else:
        ml = None
        what = 'Predictive Distribution' if ps == 'dist' else 'Prediction Interval'
        pi_text = f'{what}{asym[0]}{metafopy.plot.rutils.rround(100 * (1 - pi_alpha), dg):g}% PI{asym[2]}'
        if mlab_use is None:
            ml = ['', pi_text]
        else:
            ml = [mlab_use] if isinstance(mlab_use, str) else list(mlab_use)
            if len(ml) == 1:
                ml.append(pi_text)
    if rows_use is None:
        rows_a = -numpy.arange(1, k + 1, dtype=float)
    elif numpy.ndim(rows_use) == 0:
        r0 = float(rows_use)
        rows_a = numpy.arange(r0, r0 - k, -1, dtype=float)
    else:
        rows_a = numpy.asarray(rows_use, dtype=float)
    if ps == 'line':
        if rows_a.shape[0] != k:
            raise metafopy.errors.InputError(
                f"Length of the 'rows' argument ({rows_a.shape[0]}) does not correspond to the number of polygons to be plotted ({k})."
            )
    elif rows_a.shape[0] == 1:
        rows_a = numpy.array([rows_a[0], rows_a[0] - 1])
    na = numpy.isnan(yi) | numpy.isnan(cl) | numpy.isnan(cu)
    if na.any():
        ok = ~na
        yi, cl, cu, pl, pu = yi[ok], cl[ok], cu[ok], pl[ok], pu[ok]
        if ps == 'line':
            ml = [m for m, o in zip(ml or [], ok, strict=True) if o]
            rows_new = rows_a.copy()
            for rna in rows_a[~ok]:
                rows_new[rows_a <= rna] += 1
            rows_a = rows_new[ok]
    k = int(yi.shape[0])
    if k == 0:
        raise metafopy.errors.InputError('Processing terminated since k = 0.')
    yi_raw = yi.copy()
    if tr is not None:
        yi, cl, cu, pl, pu = (metafopy.methods.common.apply_each(v, tr, targs_) for v in (yi, cl, cu, pl, pu))
    cl, cu = metafopy.methods.common.psort(cl, cu)
    pl, pu = metafopy.methods.common.psort(pl, pu)
    if ol is not None:
        yi, cl, cu, pl, pu = (apply_olim(v, ol) for v in (yi, cl, cu, pl, pu))
    ylim = axx.get_ylim()
    xlim = axx.get_xlim()
    height = ylim[1] - ylim[0]
    cex_adj = min(1.0, 20 / height)
    cx = float(st.get('cex', cex_adj)) if cex is None else cex
    if cex is None and 'cex' not in st:
        cx = cex_adj
    tp = list(st.get('textpos', (xlim[0], xlim[1]))) if textpos is None else [float(textpos[0]), float(textpos[1])]
    if math.isnan(tp[0]):
        tp[0] = xlim[0]
    if math.isnan(tp[1]):
        tp[1] = xlim[1]
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    pstyle_nonline = ps != 'line'
    if do_annotate:
        est_a, lb_a, ub_a = (numpy.concatenate([yi, [math.nan]]), numpy.concatenate([cl, pl]), numpy.concatenate([cu, pu])) if pstyle_nonline else (yi, cl, cu)
        raw, _ = build_annotext(est_a, lb_a, ub_a, [dg, dg, dg], asym, wd, None, False, atr, targs_)
        if pstyle_nonline:
            raw[1] = raw[1].replace('NA', '')
        annotext = [s.replace('-', asym[3]).replace(' ', asym[4]) for s in raw]
        if pstyle_nonline:
            for yy, s in zip([rows_a[0], rows_a[1]], annotext, strict=True):
                ftxt(axx, tp[1], float(yy + radj[1]), s, fn[1], pos=2, cex=cx)
        else:
            for yy, s in zip(rows_a, annotext, strict=True):
                ftxt(axx, tp[1], float(yy + radj[1]), s, fn[1], pos=2, cex=cx)
    if constarea:
        area = (cu - cl) * (height / 100) * cx * efv[0]
        area = area / numpy.nanmin(area)
        polheight = (height / 100) * cx * efv[2] * (1 / area)
    else:
        polheight = numpy.full(k, (height / 100) * cx * efv[2])
    ciendheight = height / 150 * cx * efv[0]
    arrowwidth = 1.4 / 100 * cx * (xlim[1] - xlim[0])
    arrowheight = height / 150 * cx * efv[1]
    barheight = min(0.25, height / 150 * cx * efv[3])
    pipolheight = (height / 100) * cx * efv[3]
    for i in range(k):
        if ps == 'line':
            lo_x = math.nan if math.isnan(pl[i]) else max(float(pl[i]), alim_t[0])
            hi_x = math.nan if math.isnan(pu[i]) else min(float(pu[i]), alim_t[1])
            metafopy.plot.draw.seg(axx, lo_x, rows_a[i], hi_x, rows_a[i], lty=lt[0], col=lcol_l[i])
            if not math.isnan(pl[i]):
                if pl[i] >= alim_t[0]:
                    metafopy.plot.draw.seg(axx, pl[i], rows_a[i] - ciendheight, pl[i], rows_a[i] + ciendheight, lty=lt[1], col=lcol_l[i])
                else:
                    arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, lcol_l[i])
            if not math.isnan(pu[i]):
                if pu[i] <= alim_t[1]:
                    metafopy.plot.draw.seg(axx, pu[i], rows_a[i] - ciendheight, pu[i], rows_a[i] + ciendheight, lty=lt[1], col=lcol_l[i])
                else:
                    arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, lcol_l[i])
        else:
            curve: PredCurve | None = None
            if ps in DENSITY_STYLES:
                curve = pred_curve(
                    ps,
                    float(yi_raw[i]),
                    pi_se,
                    pi_dist,
                    pi_ddf,
                    pi_alpha,
                    tr,
                    predlim,
                    None if pdist is None else (pdist[0], pdist[1]),
                    predtail,
                )
            draw_pred(
                axx,
                ps,
                float(pl[i]),
                float(pu[i]),
                float(yi[i]),
                float(rows_a[1]),
                PredGeometry(alim_t, ciendheight, arrowwidth, arrowheight, barheight, pipolheight, efv[3], ol),
                lt[1],
                col_l,
                bord_l,
                curve,
            )
        xs, ys = diamond(cl[i], yi[i], cu[i], rows_a[i], float(polheight[i]))
        sel = (xs > alim_t[0]) & (xs < alim_t[1])
        metafopy.plot.draw.poly(axx, xs[sel], ys[sel], col=col_l[i], border=bord_l[i])
        if ml is not None:
            ftxt(axx, tp[0], float(rows_a[i] + radj[0]), ml[i], fn[0], pos=4, cex=cx)
            if pstyle_nonline:
                ftxt(axx, tp[0], float(rows_a[1] + radj[0]), ml[1], fn[0], pos=4, cex=cx)
    return axx
