"""Forest plots (metafor's ``forest.rma``, ``forest.default``, ``addpoly.rma`` and ``addpoly.default``).

The layout logic follows the R code line by line: study rows are numbered ``k:1`` from the top,
the summary polygon sits at row ``-1`` (prediction interval at ``-2``), the axis limits are
``alim``/``xlim`` with the label and annotation areas reserved left and right of the plot, and the
sizes of the arrow heads, CI end caps and summary polygons are fractions of the plot height.

See https://wviechtb.github.io/metafor/reference/forest.rma.html and
https://wviechtb.github.io/metafor/reference/addpoly.html, and Viechtbauer (2010),
doi:10.18637/jss.v036.i03.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.methods._common import apply_each, psort
from metafopy.methods.influence import rstandard
from metafopy.methods.predict import predict
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._rutils import coladj, fmtx1, get_ax, level_to_alpha, pretty, rround, setlab

__all__ = ['ForestResult', 'addpoly', 'forest']

type FloatArray = NDArray[np.float64]
type Func = Callable[..., Any]

_POLYLEN = 10000
_FOREST_STATE: weakref.WeakKeyDictionary[Axes, dict[str, Any]] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class ForestResult:
    """What metafor's ``forest()`` returns invisibly, plus the axes drawn on.

    ``xlim``/``ylim`` are the limits of the plot region, ``alim`` the limits of the x axis,
    ``at`` the tick locations, ``rows`` the row of each plotted estimate, ``textpos`` the x
    positions of the study labels and annotations, ``areas`` the relative widths of the
    label, plot and annotation areas (``forest`` of a fitted model only).
    """

    ax: Axes
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


def _astr(v: Any) -> str:
    if isinstance(v, float | np.floating):
        f = float(v)
        if math.isnan(f):
            return 'NA'
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f'{f:.15g}'
    return str(v)


def _ilab_matrix(ilab: Any) -> NDArray[np.object_] | None:
    if ilab is None:
        return None
    if isinstance(ilab, pd.DataFrame):
        arr = ilab.to_numpy(dtype=object)
    else:
        arr = np.asarray(ilab, dtype=object)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _floats(v: float | Sequence[float]) -> list[float]:
    return [float(a) for a in np.atleast_1d(np.asarray(v, dtype=float))]


def _expand(v: Any, k: int, name: str) -> list[Any]:
    """R's ``.expand1()`` followed by the length check."""
    if isinstance(v, str) or np.ndim(v) == 0:
        return [v] * k
    out = list(v)
    if len(out) == 1:
        out = out * k
    if len(out) != k:
        raise ValueError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the number of outcomes ({k}).")
    return out


def _pervec(v: Any, keep: NDArray[np.bool_], k: int, k_all: int, name: str) -> list[Any]:
    """A per-study argument of a fitted model: a scalar is recycled, a vector may have one entry per supplied study."""
    if isinstance(v, str) or np.ndim(v) == 0:
        return [v] * k
    out = list(v)
    if len(out) == 1:
        return out * k
    return list(_per_study(out, keep, k, k_all, name))


def _order_by(order: Any, decreasing: bool, k: int) -> NDArray[np.intp]:
    arr = np.asarray(order)
    if arr.shape[0] != k:
        raise ValueError(f"Length of the 'order' argument ({arr.shape[0]}) does not correspond to the number of outcomes ({k}).")
    if decreasing:
        if arr.dtype.kind in 'fiub':
            return np.argsort(-arr.astype(float), kind='stable')
        return np.argsort(arr[::-1], kind='stable')[::-1][::-1]
    return np.argsort(arr, kind='stable')


def _lexorder(primary: FloatArray, secondary: FloatArray) -> NDArray[np.intp]:
    return np.lexsort((secondary, primary))


def _apply_olim(x: FloatArray, olim: tuple[float, float] | None) -> FloatArray:
    if olim is None:
        return x
    out = x.copy()
    out[out < olim[0]] = olim[0]
    out[out > olim[1]] = olim[1]
    return out


def _compress_rows(rows: FloatArray, not_na: NDArray[np.bool_]) -> FloatArray:
    """Close the gaps left by omitted (missing) estimates, as the ``na.omit`` branch of ``forest`` does."""
    rows_new = rows.copy()
    for rna in rows[~not_na]:
        rows_new[rows >= rna] -= 1
    return rows_new[not_na]


def _point_sizes(wi: FloatArray, plim: Sequence[float], k: int) -> FloatArray:
    if len(plim) < 2:
        raise ValueError("Argument 'plim' must be of length 2 or 3.")
    p1, p2 = plim[0], plim[1]
    psize = np.full(k, np.nan)
    if not math.isnan(p1) and not math.isnan(p2):
        rng = np.nanmax(wi) - np.nanmin(wi)
        if rng <= math.sqrt(np.finfo(float).eps):
            psize = np.ones(k)
        else:
            psize = (wi - np.nanmin(wi)) / rng
            psize = psize * (p2 - p1) + p1
    if math.isnan(p1) and not math.isnan(p2):
        psize = wi / np.nanmax(wi) * p2
        if len(plim) == 3:
            psize[psize <= plim[2]] = plim[2]
    if not math.isnan(p1) and math.isnan(p2):
        psize = wi / np.nanmin(wi) * p1
        if len(plim) == 3:
            psize[psize >= plim[2]] = plim[2]
    if np.all(np.isnan(psize)):
        psize = np.ones(k)
    return psize


def _strheight(ax: Axes, ylim: tuple[float, float]) -> float:
    """``strheight("O")`` in user coordinates (cap height of 12pt-scaled text relative to the axes height)."""
    fs = d.base_fontsize()
    fig = ax.figure
    h_in = ax.get_position().height * fig.get_figheight()  # type: ignore[union-attr]
    return 0.716 * fs / 72.0 / h_in * (ylim[1] - ylim[0])


def _arrow(ax: Axes, side: Literal['l', 'r'], alim: tuple[float, float], row: float, aw: float, ah: float, col: str) -> None:
    if side == 'l':
        xs = [alim[0], alim[0] + aw, alim[0] + aw, alim[0]]
    else:
        xs = [alim[1], alim[1] - aw, alim[1] - aw, alim[1]]
    d.poly(ax, xs, [row, row + ah, row - ah, row], col=col, border=col)


def _diamond(xlo: float, mid: float, xhi: float, row: float, height: float) -> tuple[FloatArray, FloatArray]:
    poladds = np.arange(_POLYLEN) * (height / (_POLYLEN - 1))
    xs = np.concatenate(
        [
            np.linspace(xlo, mid, _POLYLEN),
            np.linspace(mid, xhi, _POLYLEN),
            np.linspace(xhi, mid, _POLYLEN),
            np.linspace(mid, xlo, _POLYLEN),
        ]
    )
    ys = np.concatenate([row + poladds, row + height - poladds, row - poladds, row - height + poladds])
    return xs, ys


def _annosym(annosym: Sequence[str] | None) -> list[str]:
    if annosym is None:
        return [' [', ', ', ']', '-', ' ']
    a = list(annosym)
    if len(a) == 3:
        a += ['-', ' ']
    if len(a) == 4:
        a += [' ']
    if len(a) != 5:
        raise ValueError("Argument 'annosym' must be a vector of length 3 (or 4 or 5).")
    return a


def _build_annotext(
    est: FloatArray,
    lb: FloatArray,
    ub: FloatArray,
    digits: Sequence[int],
    annosym: list[str],
    width: Sequence[int] | int | None,
    weights: FloatArray | None,
    pi_row: bool,
    atransf: Func | None,
    targs: Any,
    weight_pad_rows: tuple[int, ...] = (),
) -> tuple[list[str], list[int]]:
    """The annotation strings (``est [lb, ub]``) and their column widths."""
    cols: list[FloatArray] = [est, lb, ub]
    if atransf is not None:
        cols = [apply_each(c, atransf, targs) for c in cols]
        lo, hi = psort(cols[1], cols[2])
        cols[1], cols[2] = lo, hi
    if weights is not None:
        mat_cols = [weights, *cols]
        dig = [digits[2], digits[0], digits[0], digits[0]]
    else:
        mat_cols = cols
        dig = [digits[0]] * 3
    strs = [[fmtx1(float(v), dg) for v in c] for c, dg in zip(mat_cols, dig, strict=True)]
    for r in weight_pad_rows:  # the "100" of the summary row
        strs[0][r] = '100'
    if width is None:
        wid = [max(len(s) for s in col) for col in strs]
    else:
        wid = [int(width)] * len(strs) if isinstance(width, int | float) else [int(w) for w in width]
        if len(wid) == 1:
            wid = wid * len(strs)
        if len(wid) != len(strs):
            raise ValueError(f"Length of the 'width' argument ({len(wid)}) does not match the number of annotation columns ({len(strs)}).")
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


def _finish_annotext(raw: list[str], annosym: list[str], extra_blank: bool, pi_row: bool) -> list[str]:
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


def _default_mlab(method: str) -> str:
    return {'FE': 'Fixed-Effect Model', 'EE': 'Equal-Effects Model', 'CE': 'Common-Effect Model'}.get(method, 'Random-Effects Model')


def _check_pred(predstyle: str) -> None:
    if predstyle not in ('line', 'polygon', 'bar', 'shade', 'dist'):
        raise ValueError("'predstyle' should be one of 'line', 'polygon', 'bar', 'shade', 'dist'.")
    if predstyle in ('shade', 'dist'):
        raise NotImplementedError("predstyle='shade' and predstyle='dist' are not implemented; use 'line', 'polygon' or 'bar'.")


def forest(
    x: RMAResult | ArrayLike,
    vi: ArrayLike | None = None,
    sei: ArrayLike | None = None,
    ci_lb: ArrayLike | None = None,
    ci_ub: ArrayLike | None = None,
    annotate: bool = True,
    addfit: bool = True,
    addpred: bool | None = None,
    predstyle: str | None = None,
    showweights: bool = False,
    header: bool | str | Sequence[str] = True,
    xlim: Sequence[float] | None = None,
    alim: Sequence[float] | None = None,
    olim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    at: ArrayLike | None = None,
    steps: int = 5,
    level: float | None = None,
    refline: float | None = 0,
    digits: int | Sequence[int] = 2,
    width: int | Sequence[int] | None = None,
    xlab: str | Sequence[str] | None = None,
    slab: Sequence[Any] | float | None = None,
    mlab: str | Sequence[str] | None = None,
    ilab: Any = None,
    ilab_lab: Sequence[str] | None = None,
    ilab_xpos: ArrayLike | None = None,
    ilab_pos: int | Sequence[int] | None = None,
    order: str | ArrayLike | None = None,
    subset: ArrayLike | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: Any = None,
    rows: ArrayLike | float | None = None,
    efac: float | Sequence[float] = 1,
    pch: int | Sequence[int] = 15,
    psize: float | ArrayLike | None = None,
    plim: Sequence[float] = (0.5, 1.5),
    colout: str | Sequence[str] | None = None,
    colci: str | Sequence[str] | None = None,
    col: str | Sequence[str] | None = None,
    border: str | Sequence[str] | None = None,
    shade: bool | str | ArrayLike | None = None,
    colshade: str | None = None,
    lty: str | Sequence[str] | None = None,
    cex: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    *,
    ax: Axes | None = None,
    annosym: Sequence[str] | None = None,
    textpos: Sequence[float] | None = None,
    rowadj: float | Sequence[float] | None = None,
    top: float = 3,
    at_lab: Sequence[str] | None = None,
    xlabadj: float | Sequence[float] | None = None,
    xlabfont: int = 1,
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
    prediction interval to the summary (``'line'``, ``'polygon'`` or ``'bar'``; the ``'shade'``
    and ``'dist'`` styles of metafor are not implemented). The remaining metafor options passed
    through ``...`` are the keyword-only ``annosym``, ``textpos``, ``rowadj``, ``top``, ``at_lab``,
    ``xlabadj``, ``xlabfont`` and ``decreasing``;
    for the numeric form ``measure`` (or the ``attrs['measure']`` of a pandas Series) sets the label of the axis and header.
    Font families (``fonts``) are not supported.

    Indices are 0-based where they refer to positions; ``rows`` are plot rows as in R. Vector
    arguments (``slab``, ``ilab``, ``psize``, ...) must have one entry per study of the fit (or per
    supplied study if some were dropped for missing values).

    Returns a :class:`ForestResult` (metafor returns the same values invisibly); the markers, CI
    segments, polygons and texts are ordinary matplotlib artists on ``result.ax``.
    """
    is_fit = isinstance(x, RMAResult)
    if transf is not None and atransf is not None:
        raise ValueError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    axx = get_ax(ax)
    dig = [digits] * 3 if isinstance(digits, int) else list(digits)
    drop0 = all(isinstance(v, int | np.integer) for v in dig)  # R: drop trailing zeros of the axis labels only if digits is an integer (2L)
    if len(dig) == 1:
        dig = dig * 3
    if len(dig) == 2:
        dig = [*dig, dig[0]]
    ef = _floats(efac)
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
    asym = _annosym(annosym)
    ol: tuple[float, float] | None = None
    if olim is not None:
        if len(olim) != 2:
            raise ValueError("Argument 'olim' must be of length 2.")
        ol = (float(min(olim)), float(max(olim)))
    if rowadj is None:
        radj = [0.0, 0.0, 0.0]
    else:
        radj = _floats(rowadj)
        if len(radj) == 1:
            radj = [radj[0], radj[0], 0.0]
        if len(radj) == 2:
            radj.append(0.0)
    if xlabadj is None:
        xladj: list[float | None] = [None, None]
    else:
        xla = _floats(xlabadj)
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
    res: RMAResult | None = None
    sort_extra: dict[str, FloatArray] = {}
    if is_fit:
        res = x  # type: ignore[assignment]
        assert res is not None
        lvl = res.level if level is None else level_to_alpha(level)
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
        slab_l: list[Any] = list(res.slab)
        if slab is None:
            slab_null = all(s == str(i + 1) for s, i in zip(res.slab, np.flatnonzero(keep), strict=True))
            if slab_null:
                slab_l = [f'Study {i + 1}' for i in np.flatnonzero(keep)]
        else:
            slab_l = _per_study(list(slab) if not isinstance(slab, float) else [slab], keep, k, k_all, 'slab')
            slab_null = False
        ilab_m = _ilab_matrix(ilab)
        if ilab_m is not None:
            ilab_m = _per_study(ilab_m, keep, k, k_all, 'ilab')
        pch_l = _pervec(pch, keep, k, k_all, 'pch')
        psize_l = None if psize is None else _pervec(psize, keep, k, k_all, 'psize')
        colout_l = _pervec(d.FG if colout is None else colout, keep, k, k_all, 'colout')
        colci_l = colout_l if colci is None else _pervec(colci, keep, k, k_all, 'colci')
        pr = None if int_only else predict(res, level=level)
        wts = res.weights_()
        if int_only:
            pred = np.full(k, float(res.beta[0]))
            pred_ci_lb = np.full(k, np.nan)
            pred_ci_ub = np.full(k, np.nan)
        else:
            assert pr is not None
            pred = pr.pred
            use_pi = bool(addpred)
            pred_ci_lb = pr.pi_lb if use_pi else pr.ci_lb
            pred_ci_ub = pr.pi_ub if use_pi else pr.ci_ub
        resid_z = None
        if order is not None and isinstance(order, str) and order in ('rstandard', 'abs.rstandard'):
            resid_z = rstandard(res).z
        if resid_z is not None:
            sort_extra['z'] = resid_z
        shade_a = shade
        sh_k_all = k_all
        crit0 = float(stats.norm.isf(lvl / 2))
        ci_lb_a = yi - crit0 * np.sqrt(vi_a)
        ci_ub_a = yi + crit0 * np.sqrt(vi_a)
    else:
        yi = np.atleast_1d(np.asarray(x, dtype=np.float64)).copy()
        k = int(yi.shape[0])
        lvl = 0.05 if level is None else level_to_alpha(level)
        crit = float(stats.norm.isf(lvl / 2))
        if ci_lb is not None and ci_ub is not None:
            lb0 = np.atleast_1d(np.asarray(ci_lb, dtype=np.float64))
            ub0 = np.atleast_1d(np.asarray(ci_ub, dtype=np.float64))
            if lb0.shape != ub0.shape:
                raise ValueError("Length of 'ci_lb' and 'ci_ub' are not the same.")
            if vi is None and sei is None:
                vi_a = ((ub0 - lb0) / (2 * crit)) ** 2
            else:
                vi_a = np.asarray(vi, dtype=np.float64) if vi is not None else np.asarray(sei, dtype=np.float64) ** 2
            if lb0.shape[0] != vi_a.shape[0]:
                raise ValueError("Length of 'vi' (or 'sei') does not match the length of ('ci_lb','ci_ub').")
            ci_lb_a, ci_ub_a = lb0, ub0
        else:
            if vi is None:
                if sei is None:
                    raise ValueError("Must specify either 'vi', 'sei', or ('ci_lb','ci_ub').")
                vi_a = np.asarray(sei, dtype=np.float64) ** 2
            else:
                vi_a = np.asarray(vi, dtype=np.float64)
            if yi.shape[0] != vi_a.shape[0]:
                raise ValueError("Length of 'vi' (or 'sei') does not match the length of 'yi'.")
            ci_lb_a = yi - crit * np.sqrt(vi_a)
            ci_ub_a = yi + crit * np.sqrt(vi_a)
        vi_a = np.atleast_1d(vi_a).astype(np.float64)
        attrs = getattr(x, 'attrs', {})
        if measure is None:
            measure = str(attrs['measure']) if isinstance(attrs, dict) and attrs.get('measure') else 'GEN'
        slab_null = False
        if slab is None:
            slab_l = [f'Study {i + 1}' for i in range(k)]
            slab_null = True
        else:
            if np.ndim(slab) == 0 and not isinstance(slab, str) and math.isnan(float(slab)):  # type: ignore[arg-type]
                slab_l = [''] * k  # slab=NA in R: no study labels
                slab_null = True
            else:
                slab_l = list(slab)  # type: ignore[arg-type]
        if len(slab_l) != k:
            raise ValueError(f"Length of the 'slab' argument ({len(slab_l)}) does not correspond to the number of outcomes ({k}).")
        ilab_m = _ilab_matrix(ilab)
        if ilab_m is not None and ilab_m.shape[0] != k:
            raise ValueError(f"Length of the 'ilab' argument ({ilab_m.shape[0]}) does not correspond to the number of outcomes ({k}).")
        pch_l = _expand(pch, k, 'pch')
        psize_l = None if psize is None else _expand(psize, k, 'psize')
        col_l = _expand(d.FG if col is None else col, k, 'col')
        colci_l = col_l if colci is None else _expand(colci, k, 'colci')
        colout_l = col_l
        wts = np.full(k, np.nan)
        pred = np.full(k, np.nan)
        pred_ci_lb = np.full(k, np.nan)
        pred_ci_ub = np.full(k, np.nan)
        shade_a = shade
        sh_k_all = k
        keep = np.ones(k, dtype=np.bool_)
        k_all = k
    if subset is not None and is_fit:
        raise ValueError('Function does not have a subset argument.')

    # -- shade type ------------------------------------------------------------------------------------------
    shade_type = 'none'
    shade_chr = ''
    shade_arr: Any = None
    if isinstance(shade_a, str):
        shade_type, shade_chr = 'character', shade_a
        if shade_chr not in ('zebra', 'zebra1', 'zebra2', 'all'):
            raise ValueError("Unknown option specified for 'shade' argument.")
    elif isinstance(shade_a, bool | np.bool_):
        shade_type, shade_chr = 'character', 'zebra'
    elif shade_a is not None:
        sa = np.asarray(shade_a)
        if sa.dtype == np.bool_:
            shade_type = 'logical'
            shade_arr = np.asarray(_per_study(list(sa), keep, k, sh_k_all, 'shade'), dtype=bool) if is_fit else sa
        else:
            shade_type = 'numeric'
            shade_arr = sa.astype(float)

    # -- subset / order --------------------------------------------------------------------------------------
    def reorder(idx: NDArray[np.intp]) -> None:
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

    sub_mask: NDArray[np.bool_] | None = None
    if subset is not None:
        sa_ = np.asarray(subset)
        sub_mask = sa_.astype(bool) if sa_.dtype == np.bool_ else np.isin(np.arange(k), sa_.astype(np.intp))
    if not is_fit:
        if order is not None:
            if isinstance(order, str):
                if order not in ('est', 'obs', 'yi', 'prec', 'vi'):
                    raise ValueError("'order' should be one of 'est', 'obs', 'yi', 'prec', 'vi'.")
                idx = np.argsort(yi, kind='stable') if order in ('est', 'obs', 'yi') else _lexorder(vi_a, yi)
            else:
                idx = _order_by(order, decreasing, k)
            reorder(idx)
            if sub_mask is not None:
                sub_mask = sub_mask[idx]
        if sub_mask is not None:
            reorder(np.flatnonzero(sub_mask).astype(np.intp))
    elif order is not None:
        assert res is not None
        if isinstance(order, str):
            if order in ('est', 'obs', 'yi'):
                idx = np.argsort(yi, kind='stable')
            elif order == 'fit':
                idx = np.argsort(pred, kind='stable')
            elif order in ('prec', 'vi'):
                idx = _lexorder(vi_a, yi)
            elif order == 'resid':
                idx = _lexorder(yi - pred, yi)
            elif order == 'rstandard':
                idx = _lexorder(sort_extra['z'], yi)
            elif order == 'abs.resid':
                idx = _lexorder(np.abs(yi - pred), yi)
            elif order == 'abs.rstandard':
                idx = _lexorder(np.abs(sort_extra['z']), yi)
            else:
                raise ValueError(f'unknown order {order!r}')
        else:
            idx = _order_by(_per_study(list(np.asarray(order)), keep, k, k_all, 'order'), decreasing, k)
        reorder(idx)
    k = int(yi.shape[0])

    # -- rows ----------------------------------------------------------------------------------------------
    if rows is None:
        rows_a = np.arange(k, 0, -1, dtype=float)
    elif np.ndim(rows) == 0:
        r0 = float(rows)  # type: ignore[arg-type]
        rows_a = np.arange(r0, r0 - k, -1, dtype=float)
    else:
        rows_a = np.asarray(rows, dtype=float)
    if rows_a.shape[0] != k:
        raise ValueError(f"Length of the 'rows' argument ({rows_a.shape[0]}) does not correspond to the number of outcomes ({k}).")
    rev = np.arange(k)[::-1]
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
    na_mask = np.isnan(yi) | np.isnan(vi_a)
    if X is not None:
        na_mask |= np.isnan(X).any(axis=1)
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
        rows_a = _compress_rows(rows_a, ok)
    k = int(yi.shape[0])

    crit = float(stats.norm.isf(lvl / 2))
    # -- transformation --------------------------------------------------------------------------------------
    if transf is not None:
        yi = apply_each(yi, transf, targs)
        ci_lb_a = apply_each(ci_lb_a, transf, targs)
        ci_ub_a = apply_each(ci_ub_a, transf, targs)
        pred = apply_each(pred, transf, targs)
        pred_ci_lb = apply_each(pred_ci_lb, transf, targs)
        pred_ci_ub = apply_each(pred_ci_ub, transf, targs)
    ci_lb_a, ci_ub_a = psort(ci_lb_a, ci_ub_a)
    pred_ci_lb, pred_ci_ub = psort(pred_ci_lb, pred_ci_ub)
    if ol is not None:
        yi, ci_lb_a, ci_ub_a = (_apply_olim(v, ol) for v in (yi, ci_lb_a, ci_ub_a))
        pred, pred_ci_lb, pred_ci_ub = (_apply_olim(v, ol) for v in (pred, pred_ci_lb, pred_ci_ub))

    if showweights and not is_fit:
        wts = 1 / vi_a
        wts = 100 * wts / np.nansum(wts)

    # -- point sizes ------------------------------------------------------------------------------------------
    if psize_l is None:
        if is_fit:
            psz = _point_sizes(np.sqrt(wts), plim, k)
        elif np.any(vi_a <= 0):
            psz = np.ones(k)
        else:
            psz = _point_sizes(1 / np.sqrt(vi_a), plim, k)
    else:
        psz = np.asarray(psize_l, dtype=float)

    # -- alim / at -----------------------------------------------------------------------------------------------
    at_in: FloatArray | None = None
    if at is not None:
        at_in = np.asarray(at, dtype=float)
        if np.isnan(at_in).any():
            raise ValueError("Argument 'at' cannot contain NAs.")
        if np.isinf(at_in).any():
            raise ValueError("Argument 'at' cannot contain +-Inf values.")
    alim_spec = True
    lbmin, ubmax = float(np.nanmin(ci_lb_a)), float(np.nanmax(ci_ub_a))
    if alim is None:
        if at_in is None:
            pr_ = pretty([lbmin, ubmax], steps - 1)
            alim_t = (float(pr_.min()), float(pr_.max()))
            alim_spec = False
        else:
            alim_t = (float(at_in.min()), float(at_in.max()))
    else:
        if len(alim) != 2:
            raise ValueError("Argument 'alim' must be of length 2.")
        alim_t = (float(min(alim)), float(max(alim)))
    if math.isnan(alim_t[0]) or math.isnan(alim_t[1]):
        raise ValueError("Argument 'alim' cannot contain NAs.")
    if at_in is None:
        at_a = np.linspace(alim_t[0], alim_t[1], steps) if alim_spec else pretty([lbmin, ubmax], steps - 1)
    else:
        at_a = np.array(list(dict.fromkeys(np.clip(at_in, alim_t[0], alim_t[1]).tolist())), dtype=float)
    if at_lab is None:
        if atransf is not None:
            at_lab_l = [fmtx1(float(v), dig[1], drop0) for v in apply_each(at_a, atransf, targs)]
        else:
            at_lab_l = [fmtx1(float(v), dig[1], drop0) for v in at_a]
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
        xl = (rround(f1 - rng * mult_l, dig[1]), rround(f2 + rng * mult_r, dig[1]))
    else:
        if len(xlim) != 2:
            raise ValueError("Argument 'xlim' must be of length 2.")
        xl = (float(xlim[0]), float(xlim[1]))
    xl = (min(xl), max(xl))
    tp = [xl[0], xl[1]] if textpos is None else [float(textpos[0]), float(textpos[1])]
    if len(tp) != 2:
        raise ValueError("Argument 'textpos' must be of length 2.")
    if math.isnan(tp[0]):
        tp[0] = xl[0]
    if math.isnan(tp[1]):
        tp[1] = xl[1]

    if predstyle is None:
        pstyle = 'line'
        if addpred is None:
            addpred = False
    else:
        _check_pred(predstyle)
        pstyle = predstyle
        if addpred is None:
            addpred = True
    fit_row = is_fit and int_only and addfit
    mrk = max(rows_a) if rows_a.size else 0.0
    if ylim is None:
        yl = (-2.0 - (0.0 if pstyle == 'line' else 1.0), mrk + top) if fit_row else (0.0, mrk + top)
    elif len(ylim) == 1:
        yl = (float(ylim[0]), mrk + top)
    else:
        yl = (float(min(ylim)), float(max(ylim)))

    # -- the plot ----------------------------------------------------------------------------------------------------
    d.only_xaxis(axx)
    axx.set_xlim(xl)
    axx.set_ylim(yl)
    axx.set_autoscale_on(False)
    colshade_c = coladj(-0.1) if colshade is None else colshade
    if shade_type == 'character':
        if shade_chr in ('zebra', 'zebra1'):
            sh = np.resize(np.array([True, False]), k)
        elif shade_chr == 'zebra2':
            sh = np.resize(np.array([False, True]), k)
        else:
            sh = np.ones(k, dtype=bool)
        shade_arr = sh
    if shade_type in ('character', 'logical'):
        for i in range(k):
            if shade_arr[i]:
                d.rect(axx, xl[0], rows_a[i] - 0.5, xl[1], rows_a[i] + 0.5, col=colshade_c, border=colshade_c, zorder=0.5)
    if shade_type == 'numeric':
        for s in shade_arr:
            d.rect(axx, xl[0], float(s) - 0.5, xl[1], float(s) + 0.5, col=colshade_c, border=colshade_c, zorder=0.5)
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
    d.hline(axx, yl[1] - (top - 1), lty=hl_lty)
    if refline is not None:
        d.seg(axx, refline, yl[0], refline, yl[1] - (top - 1), lty='dotted')
    height = yl[1] - yl[0]
    cex_adj = 1.0
    if cex is None:
        lheight = _strheight(axx, yl)
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
        col_mr = coladj(-0.2) if col is None else (col if isinstance(col, str) else col[0])
        bord_mr = coladj(-0.3) if border is None else (border if isinstance(border, str) else border[0])
        for i in range(k):
            if math.isnan(pred[i]):
                continue
            xs, ys = _diamond(pred_ci_lb[i], pred[i], pred_ci_ub[i], rows_a[i], polheight)
            sel = (xs > alim_t[0]) & (xs < alim_t[1])
            d.poly(axx, xs[sel], ys[sel], col=col_mr, border=bord_mr)
    ciendheight = height / 150 * cx * efv[0]
    arrowwidth = 1.4 / 100 * cx * (xl[1] - xl[0])
    arrowheight = height / 150 * cx * efv[1]
    barheight = min(0.25, height / 150 * cx * efv[3])
    pipolheight = (height / 100) * cx * efv[3]

    beta = beta_lb = beta_ub = pi_lb = pi_ub = math.nan
    if fit_row:
        assert res is not None
        pr2 = predict(res, level=100 * (1 - lvl))
        beta, beta_lb, beta_ub = float(pr2.pred[0]), float(pr2.ci_lb[0]), float(pr2.ci_ub[0])
        pi_lb, pi_ub = float(pr2.pi_lb[0]), float(pr2.pi_ub[0])
        if transf is not None:
            beta, beta_lb, beta_ub, pi_lb, pi_ub = (float(apply_each(np.array([v]), transf, targs)[0]) for v in (beta, beta_lb, beta_ub, pi_lb, pi_ub))
        beta_lb, beta_ub = (float(v[0]) for v in psort(np.array([beta_lb]), np.array([beta_ub])))
        pi_lb, pi_ub = (float(v[0]) for v in psort(np.array([pi_lb]), np.array([pi_ub])))
        if ol is not None:
            beta, beta_lb, beta_ub, pi_lb, pi_ub = (float(_apply_olim(np.array([v]), ol)[0]) for v in (beta, beta_lb, beta_ub, pi_lb, pi_ub))
        colv = _summary_cols(col, border)
        pcol, pborder = colv
        if res.method not in ('FE', 'EE', 'CE') and addpred:
            _draw_pred(
                axx,
                pstyle,
                pi_lb,
                pi_ub,
                beta,
                -1.0 if pstyle == 'line' else -2.0,
                alim_t,
                ciendheight,
                arrowwidth,
                arrowheight,
                barheight,
                pipolheight,
                lt[1],
                pcol[1],
                pborder[1],
            )
        xs, ys = _diamond(beta_lb, beta, beta_ub, -1.0, polheight)
        sel = (xs > alim_t[0]) & (xs < alim_t[1])
        d.poly(axx, xs[sel], ys[sel], col=pcol[0], border=pborder[0])
        ml = [_default_mlab(res.method)] if mlab is None else ([mlab] if isinstance(mlab, str) else list(mlab))
        if len(ml) == 1 and pstyle in ('polygon', 'bar'):
            ml.append(f'Prediction Interval{asym[0]}{rround(100 * (1 - lvl), dig[0]):g}% PI{asym[2]}')
        d.txt(axx, tp[0], -1 + radj[0], ml[0], pos=4, cex=cx)
        if pstyle in ('polygon', 'bar'):
            d.txt(axx, tp[0], -2 + radj[0], ml[1], pos=4, cex=cx)

    d.xaxis(axx, at_a, at_lab_l, cex=cex_a)
    measure_r = measure
    if xlab is None:
        xlab_l = [setlab(measure_r, transf, atransf, 1)]
    else:
        xlab_l = [xlab] if isinstance(xlab, str) else list(xlab)
    if len(xlab_l) not in (1, 2, 3):
        raise ValueError("Argument 'xlab' argument must be of length 1, 2, or 3.")
    amin, amax = float(at_a.min()), float(at_a.max())
    amid = amin + (amax - amin) / 2
    ha_l = 'center' if xladj[0] is None else ('left' if xladj[0] == 0 else ('right' if xladj[0] == 1 else 'center'))
    ha_r = 'center' if xladj[1] is None else ('left' if xladj[1] == 0 else ('right' if xladj[1] == 1 else 'center'))
    bold_x = xlabfont in (2, 4)
    if len(xlab_l) == 1:
        d.xlab_below(axx, amid, xlab_l[0], cex=cex_l, bold=bold_x)
    elif len(xlab_l) == 2:
        d.xlab_below(axx, amin, xlab_l[0], cex=cex_l, ha=ha_l, bold=bold_x)
        d.xlab_below(axx, amax, xlab_l[1], cex=cex_l, ha=ha_r, bold=bold_x)
    else:
        d.xlab_below(axx, amin, xlab_l[0], cex=cex_l, ha=ha_l, bold=bold_x)
        d.xlab_below(axx, amid, xlab_l[1], cex=cex_l, bold=bold_x)
        d.xlab_below(axx, amax, xlab_l[2], cex=cex_l, ha=ha_r, bold=bold_x)

    for i in range(k):
        ci_c = colci_l[i]
        if np.isnan(yi[i]) or np.isnan(ci_lb_a[i]) or np.isnan(ci_ub_a[i]) or ci_c is None:
            continue
        lo_i, hi_i = float(ci_lb_a[i]), float(ci_ub_a[i])
        if lo_i >= alim_t[1]:
            _arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
            continue
        if hi_i <= alim_t[0]:
            _arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
            continue
        d.seg(axx, max(lo_i, alim_t[0]), rows_a[i], min(hi_i, alim_t[1]), rows_a[i], lty=lt[0], col=ci_c)
        if lo_i >= alim_t[0]:
            d.seg(axx, lo_i, rows_a[i] - ciendheight, lo_i, rows_a[i] + ciendheight, col=ci_c)
        else:
            _arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)
        if hi_i <= alim_t[1]:
            d.seg(axx, hi_i, rows_a[i] - ciendheight, hi_i, rows_a[i] + ciendheight, col=ci_c)
        else:
            _arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, ci_c)

    for i in range(k):
        d.txt(axx, tp[0], rows_a[i] + radj[0], _astr(slab_l[i]), pos=4, cex=cx, col=colout_l[i] if not is_fit else d.FG)

    ilab_xpos_out: FloatArray | None = None
    ilab_pos_out: list[int] | None = None
    if ilab_m is not None:
        if ilab_xpos is None:
            dist = float(np.nanmin(ci_lb_a)) - xl[0]
            if ncol_ilab == 1:
                ixp = np.array([xl[0] + dist * 0.75])
            elif ncol_ilab == 2:
                ixp = xl[0] + dist * np.array([0.65, 0.85])
            elif ncol_ilab == 3:
                ixp = xl[0] + dist * np.array([0.6, 0.75, 0.9])
            else:
                ixp = np.linspace(xl[0] + dist * 0.5, xl[0] + dist * 0.9, ncol_ilab)
        else:
            ixp = np.atleast_1d(np.asarray(ilab_xpos, dtype=float))
        if ixp.shape[0] != ncol_ilab:
            raise ValueError(f"Number of 'ilab' columns ({ncol_ilab}) does not match the length of the 'ilab_xpos' argument ({ixp.shape[0]}).")
        ilab_xpos_out = ixp
        ipos: list[int | None]
        if ilab_pos is None:
            ipos = [None] * ncol_ilab
        else:
            ipos = [int(ilab_pos)] * ncol_ilab if np.ndim(ilab_pos) == 0 else [int(v) for v in np.atleast_1d(ilab_pos)]  # type: ignore[arg-type]
            ilab_pos_out = [int(v) for v in ipos if v is not None]
        if ilab_lab is not None and len(ilab_lab) != ncol_ilab:
            raise ValueError(f"Number of 'ilab' columns ({ncol_ilab}) does not match the length of the 'ilab_lab' argument ({len(ilab_lab)}).")
        for lcol in range(ncol_ilab):
            for i in range(k):
                d.txt(axx, float(ixp[lcol]), rows_a[i] + radj[2], _astr(ilab_m[i, lcol]), pos=ipos[lcol], cex=cx)
            if ilab_lab is not None:
                d.txt(axx, float(ixp[lcol]), yl[1] - (top - 1) + 1 + radj[2], ilab_lab[lcol], pos=ipos[lcol], cex=cx, bold=True)

    width_out: list[int] | None = None
    if annotate:
        est_all, lb_all, ub_all = yi, ci_lb_a, ci_ub_a
        pi_row = fit_row and pstyle in ('polygon', 'bar')
        wts_all: FloatArray | None = None
        pad: tuple[int, ...] = ()
        if fit_row:
            est_all = np.concatenate([yi, [beta], [math.nan] if pi_row else []])
            lb_all = np.concatenate([ci_lb_a, [beta_lb], [pi_lb] if pi_row else []])
            ub_all = np.concatenate([ci_ub_a, [beta_ub], [pi_ub] if pi_row else []])
        if showweights:
            if fit_row:
                wts_all = np.concatenate([wts, [100.0], [math.nan] if pi_row else []])
                pad = (k,)
            else:
                wts_all = wts
        raw, width_out = _build_annotext(est_all, lb_all, ub_all, dig, asym, width, wts_all, pi_row, atransf, targs, pad)
        annotext = _finish_annotext(raw, asym, False, pi_row)
        ypos = list(rows_a + radj[1])
        if fit_row:
            ypos += [-1 + radj[1]] + ([-2 + radj[1]] if pi_row else [])
        for yy, s in zip(ypos, annotext, strict=True):
            d.txt(axx, tp[1], float(yy), s, pos=2, cex=cx)

    for i in range(k):
        if np.isnan(yi[i]):
            continue
        if alim_t[0] <= yi[i] <= alim_t[1]:
            d.pts(axx, yi[i], rows_a[i], pch=int(pch_l[i]), cex=cx * float(psz[i]), col=colout_l[i])
    if fit_row:
        d.hline(axx, 0, lty=lt[2])

    # headers
    est_lab = setlab(measure_r, transf, atransf, 3, short=True)
    ci_pct = f'{rround(100 * (1 - lvl), dig[0]):g}'
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
        d.txt(axx, tp[0], yl[1] - (top - 1) + 1 + radj[0], hdr_left, pos=4, cex=cx, bold=True)
    if hdr_right is not None:
        d.txt(axx, tp[1], yl[1] - (top - 1) + 1 + radj[1], hdr_right, pos=2, cex=cx, bold=True)

    _FOREST_STATE[axx] = {
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


def _summary_cols(col: Any, border: Any) -> tuple[list[str | None], list[str | None]]:
    fg = d.FG
    c = [fg, fg, coladj(-0.2), None]
    if col is not None:
        cl = [col] if isinstance(col, str) else list(col)
        if len(cl) == 1:
            cl = [cl[0], fg, coladj(-0.2), None]
        if len(cl) == 2:
            cl = [*cl, coladj(-0.2), None]
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


def _draw_pred(
    ax: Axes,
    style: str,
    lb: float,
    ub: float,
    mid: float,
    row: float,
    alim: tuple[float, float],
    ciendheight: float,
    arrowwidth: float,
    arrowheight: float,
    barheight: float,
    pipolheight: float,
    lty: str,
    col: str | None,
    border: str | None,
) -> None:
    """The prediction interval of a summary (``line``, ``polygon`` or ``bar``)."""
    c = col if col is not None else d.FG
    if style == 'line':
        d.seg(ax, max(lb, alim[0]), row, min(ub, alim[1]), row, lty=lty, col=c)
        if lb >= alim[0]:
            d.seg(ax, lb, row - ciendheight, lb, row + ciendheight, col=c)
        else:
            _arrow(ax, 'l', alim, row, arrowwidth, arrowheight, c)
        if ub <= alim[1]:
            d.seg(ax, ub, row - ciendheight, ub, row + ciendheight, col=c)
        else:
            _arrow(ax, 'r', alim, row, arrowwidth, arrowheight, c)
    elif style == 'polygon':
        xs, ys = _diamond(lb, mid, ub, row, pipolheight)
        sel = (xs > alim[0]) & (xs < alim[1])
        d.poly(ax, xs[sel], ys[sel], col=col, border=border)
    elif style == 'bar':
        if lb >= alim[0]:
            d.rect(ax, lb, row - barheight, mid, row + barheight, col=col, border=border)
        else:
            d.rect(ax, alim[0] + arrowwidth, row - barheight, mid, row + barheight, col=col, border=border)
            d.poly(ax, [alim[0], alim[0] + arrowwidth, alim[0] + arrowwidth, alim[0]], [row, row + barheight, row - barheight, row], col=col, border=col)
        if ub <= alim[1]:
            d.rect(ax, ub, row - barheight, mid, row + barheight, col=col, border=border)
        else:
            d.rect(ax, alim[1] - arrowwidth, row - barheight, mid, row + barheight, col=col, border=border)
            d.poly(ax, [alim[1], alim[1] - arrowwidth, alim[1] - arrowwidth, alim[1]], [row, row + barheight, row - barheight, row], col=col, border=col)


def _per_study(values: Any, keep: NDArray[np.bool_], k: int, k_all: int, name: str) -> Any:
    """Select the entries of a per-study argument that correspond to the studies used in the fit."""
    n = len(values) if not isinstance(values, np.ndarray) else values.shape[0]
    if n == k:
        return values
    if n == k_all:
        if isinstance(values, np.ndarray):
            return values[keep]
        return [v for v, kp in zip(values, keep, strict=True) if kp]
    raise ValueError(f"Length of the '{name}' argument ({n}) does not correspond to the size of the original dataset ({k_all}).")


def addpoly(
    x: RMAResult | ArrayLike,
    vi: ArrayLike | None = None,
    sei: ArrayLike | None = None,
    ci_lb: ArrayLike | None = None,
    ci_ub: ArrayLike | None = None,
    pi_lb: ArrayLike | None = None,
    pi_ub: ArrayLike | None = None,
    rows: ArrayLike | float | None = -1,
    row: float = -2,
    level: float | None = None,
    annotate: bool | None = None,
    addpred: bool = False,
    predstyle: str | None = None,
    digits: int | None = None,
    width: int | Sequence[int] | None = None,
    mlab: str | Sequence[str] | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: Any = None,
    efac: float | Sequence[float] | None = None,
    col: str | Sequence[str] | None = None,
    border: str | Sequence[str] | None = None,
    lty: str | Sequence[str] | None = None,
    cex: float | None = None,
    constarea: bool = False,
    *,
    ax: Axes | None = None,
    alim: Sequence[float] | None = None,
    olim: Sequence[float] | None = None,
    annosym: Sequence[str] | None = None,
    textpos: Sequence[float] | None = None,
    rowadj: float | Sequence[float] | None = None,
    lcol: str | Sequence[str] | None = None,
) -> Axes:
    """Add polygons (summary estimates) to a forest plot (``addpoly.rma`` and ``addpoly.default``).

    For a fitted model without moderators the pooled estimate and its CI are drawn at ``row``
    (default ``-2``; the prediction interval with ``addpred=True`` or ``predstyle``). For numbers
    give the estimates ``x`` with ``vi``/``sei`` or ``ci_lb``/``ci_ub`` (and ``pi_lb``/``pi_ub``)
    and the ``rows`` to draw them in. Unspecified options (``alim``, ``digits``, ``annotate``,
    ``transf``, ``atransf``, ``efac``, ``cex``, ``textpos``, ...) are taken from the last
    :func:`forest` call on the same axes. Returns the axes.
    """
    axx = get_ax(ax)
    st = _FOREST_STATE.get(axx, {})
    ps = 'line' if predstyle is None else predstyle
    if predstyle is not None:
        _check_pred(predstyle)
        addpred = True
    if isinstance(x, RMAResult):
        if not x.int_only:
            raise ValueError('Fitted model should not contain moderators.')
        lvl_pct = None if level is None else level
        pr = predict(x, level=lvl_pct)
        yi = np.asarray(pr.pred, dtype=float)
        cl, cu = np.asarray(pr.ci_lb, dtype=float), np.asarray(pr.ci_ub, dtype=float)
        if addpred:
            pl, pu = np.asarray(pr.pi_lb, dtype=float), np.asarray(pr.pi_ub, dtype=float)
        else:
            pl = pu = np.array([math.nan])
        mlab_use: Any = _default_mlab(x.method) if mlab is None else mlab
        rows_use: Any = row
        lvl = x.level if level is None else level_to_alpha(level)
    else:
        yi = np.atleast_1d(np.asarray(x, dtype=float))
        lvl = st.get('level', 0.05) if level is None else level_to_alpha(level)
        crit = float(stats.norm.isf(lvl / 2))
        if ci_lb is not None and ci_ub is not None:
            cl, cu = np.atleast_1d(np.asarray(ci_lb, dtype=float)), np.atleast_1d(np.asarray(ci_ub, dtype=float))
        else:
            v = np.asarray(vi, dtype=float) if vi is not None else (np.asarray(sei, dtype=float) ** 2 if sei is not None else None)
            if v is None:
                raise ValueError("Must specify either 'vi', 'sei', or ('ci_lb','ci_ub').")
            cl, cu = yi - crit * np.sqrt(v), yi + crit * np.sqrt(v)
        if pi_lb is not None and pi_ub is not None:
            pl, pu = np.atleast_1d(np.asarray(pi_lb, dtype=float)), np.atleast_1d(np.asarray(pi_ub, dtype=float))
        else:
            if ps != 'line':
                raise ValueError('Cannot draw prediction interval if pi_lb and pi_ub are unspecified.')
            pl = pu = np.full(yi.shape[0], math.nan)
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
        raise ValueError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    asym = _annosym(st.get('annosym') if annosym is None else annosym)
    if efac is None:
        e0 = st.get('efac', [1.0] * 4)
        ef = [float(v) for v in (np.atleast_1d(e0) if np.ndim(e0) else [e0])]
        ef = ef * 4 if len(ef) == 1 else ef
    else:
        ef = _floats(efac)
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
        radj = _floats(rowadj)
        if len(radj) == 1:
            radj = [radj[0], radj[0]]
    fg = d.FG
    if k == 1:
        c0, b0 = _summary_cols(col, border)
        col_l: list[Any] = c0
        bord_l: list[Any] = b0
    else:
        if ps != 'line':
            raise ValueError(f"Can only use predstyle='{ps}' when plotting a single polygon.")
        col_l = _expand(fg if col is None else col, k, 'col')
        bord_l = _expand(fg if border is None else border, k, 'border')
    lcol_l = _expand(fg if lcol is None else lcol, k, 'lcol')
    lt = ['dotted', 'solid'] if lty is None else ([lty, 'solid'] if isinstance(lty, str) else list(lty))
    if ps == 'line':
        ml: list[str] | None
        if mlab_use is None:
            ml = [''] * k
        else:
            ml = _expand(mlab_use, k, 'mlab')
    else:
        ml = None
        if mlab_use is None:
            ml = ['', f'Prediction Interval{asym[0]}{rround(100 * (1 - lvl), dg):g}% PI{asym[2]}']
        else:
            ml = [mlab_use] if isinstance(mlab_use, str) else list(mlab_use)
            if len(ml) == 1:
                ml.append(f'Prediction Interval{asym[0]}{rround(100 * (1 - lvl), dg):g}% PI{asym[2]}')
    if rows_use is None:
        rows_a = -np.arange(1, k + 1, dtype=float)
    elif np.ndim(rows_use) == 0:
        r0 = float(rows_use)
        rows_a = np.arange(r0, r0 - k, -1, dtype=float)
    else:
        rows_a = np.asarray(rows_use, dtype=float)
    if ps == 'line':
        if rows_a.shape[0] != k:
            raise ValueError(f"Length of the 'rows' argument ({rows_a.shape[0]}) does not correspond to the number of polygons to be plotted ({k}).")
    elif rows_a.shape[0] == 1:
        rows_a = np.array([rows_a[0], rows_a[0] - 1])
    na = np.isnan(yi) | np.isnan(cl) | np.isnan(cu)
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
        raise ValueError('Processing terminated since k = 0.')
    if tr is not None:
        yi, cl, cu, pl, pu = (apply_each(v, tr, targs_) for v in (yi, cl, cu, pl, pu))
    cl, cu = psort(cl, cu)
    pl, pu = psort(pl, pu)
    if ol is not None:
        yi, cl, cu, pl, pu = (_apply_olim(v, ol) for v in (yi, cl, cu, pl, pu))
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
    pstyle_nonline = ps in ('polygon', 'bar')
    if do_annotate:
        est_a, lb_a, ub_a = (np.concatenate([yi, [math.nan]]), np.concatenate([cl, pl]), np.concatenate([cu, pu])) if pstyle_nonline else (yi, cl, cu)
        raw, _ = _build_annotext(est_a, lb_a, ub_a, [dg, dg, dg], asym, wd, None, False, atr, targs_)
        if pstyle_nonline:
            raw[1] = raw[1].replace('NA', '')
        annotext = [s.replace('-', asym[3]).replace(' ', asym[4]) for s in raw]
        if pstyle_nonline:
            for yy, s in zip([rows_a[0], rows_a[1]], annotext, strict=True):
                d.txt(axx, tp[1], float(yy + radj[1]), s, pos=2, cex=cx)
        else:
            for yy, s in zip(rows_a, annotext, strict=True):
                d.txt(axx, tp[1], float(yy + radj[1]), s, pos=2, cex=cx)
    if constarea:
        area = (cu - cl) * (height / 100) * cx * efv[0]
        area = area / np.nanmin(area)
        polheight = (height / 100) * cx * efv[2] * (1 / area)
    else:
        polheight = np.full(k, (height / 100) * cx * efv[2])
    ciendheight = height / 150 * cx * efv[0]
    arrowwidth = 1.4 / 100 * cx * (xlim[1] - xlim[0])
    arrowheight = height / 150 * cx * efv[1]
    barheight = min(0.25, height / 150 * cx * efv[3])
    pipolheight = (height / 100) * cx * efv[3]
    for i in range(k):
        if ps == 'line':
            lo_x = math.nan if math.isnan(pl[i]) else max(float(pl[i]), alim_t[0])
            hi_x = math.nan if math.isnan(pu[i]) else min(float(pu[i]), alim_t[1])
            d.seg(axx, lo_x, rows_a[i], hi_x, rows_a[i], lty=lt[0], col=lcol_l[i])
            if not math.isnan(pl[i]):
                if pl[i] >= alim_t[0]:
                    d.seg(axx, pl[i], rows_a[i] - ciendheight, pl[i], rows_a[i] + ciendheight, lty=lt[1], col=lcol_l[i])
                else:
                    _arrow(axx, 'l', alim_t, rows_a[i], arrowwidth, arrowheight, lcol_l[i])
            if not math.isnan(pu[i]):
                if pu[i] <= alim_t[1]:
                    d.seg(axx, pu[i], rows_a[i] - ciendheight, pu[i], rows_a[i] + ciendheight, lty=lt[1], col=lcol_l[i])
                else:
                    _arrow(axx, 'r', alim_t, rows_a[i], arrowwidth, arrowheight, lcol_l[i])
        else:
            _draw_pred(
                axx,
                ps,
                float(pl[i]),
                float(pu[i]),
                float(yi[i]),
                float(rows_a[1]),
                alim_t,
                ciendheight,
                arrowwidth,
                arrowheight,
                barheight,
                pipolheight,
                lt[1],
                col_l[1],
                bord_l[1],
            )
        xs, ys = _diamond(cl[i], yi[i], cu[i], rows_a[i], float(polheight[i]))
        sel = (xs > alim_t[0]) & (xs < alim_t[1])
        d.poly(axx, xs[sel], ys[sel], col=col_l[i], border=bord_l[i])
        if ml is not None:
            d.txt(axx, tp[0], float(rows_a[i] + radj[0]), ml[i], pos=4, cex=cx)
            if pstyle_nonline:
                d.txt(axx, tp[0], float(rows_a[1] + radj[0]), ml[1], pos=4, cex=cx)
    return axx
