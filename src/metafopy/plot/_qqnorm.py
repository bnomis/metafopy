"""Normal quantile-quantile plot of the (standardised or studentised) residuals (metafor's ``qqnorm.rma.uni``).

The plot can show a simulated pseudo confidence envelope around the line of equality, obtained as in Atkinson (1985)
(``Plots, Transformations and Regression``) by simulating normal data and applying the model's hat matrix; see
https://wviechtb.github.io/metafor/reference/qqnorm.rma.html.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from metafopy.core.rng import RRandom, get_rng, set_seed
from metafopy.methods.influence import hatvalues, rstandard, rstudent
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import coladj, get_ax, level_to_alpha, qqnorm_xy, supsmu

__all__ = ['QQNormResult', 'qqnorm']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class QQNormResult:
    """Theoretical (``x``) and sample (``y``) quantiles (what ``qqnorm()`` returns), the envelope (if drawn) and the axes."""

    ax: Axes
    x: FloatArray
    y: FloatArray
    lb: FloatArray | None = None
    ub: FloatArray | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'x': self.x, 'y': self.y})


def qqnorm(
    y: RMAResult,
    type: str = 'rstandard',
    pch: int = 21,
    col: str | None = None,
    bg: str | None = None,
    grid: bool | str = False,
    envelope: bool | str = True,
    level: float | None = None,
    bonferroni: bool = False,
    reps: int = 1000,
    smooth: bool = True,
    bass: float = 0,
    label: bool | int | str = False,
    offset: float = 0.3,
    pos: int = 13,
    lty: str | Sequence[str] | None = None,
    *,
    seed: int | None = None,
    rng: RRandom | None = None,
    ax: Axes | None = None,
) -> QQNormResult:
    """Normal Q-Q plot of the residuals of a model (``qqnorm.rma.uni``).

    ``type`` selects ``'rstandard'`` or ``'rstudent'`` residuals. ``envelope`` draws the simulated pseudo CI
    (a colour string sets its colour; ``level`` its confidence level, ``bonferroni`` applies a Bonferroni
    correction, ``reps`` the number of simulations, ``smooth``/``bass`` the Friedman super smoother applied to its
    bounds); ``seed`` seeds the R-compatible generator so that the envelope reproduces R's ``set.seed(seed)``.
    ``label`` (``True``/``'all'``, a number of the most deviating points or ``'out'`` for the points outside the
    envelope) labels points with the study labels at position ``pos`` (1-4, or 13/24 to place them depending on
    the side of the line). ``grid`` adds a grid, ``lty`` the line types of the line of equality and the envelope.

    Returns a :class:`QQNormResult`.
    """
    if type not in ('rstandard', 'rstudent'):
        raise ValueError("type must be 'rstandard' or 'rstudent'")
    if y.k == 1:
        raise ValueError('Stopped because k = 1.')
    axx = get_ax(ax)
    k = y.k
    env_col = coladj(-0.15)
    if label == 'out' and isinstance(envelope, bool):
        envelope = True
    draw_envelope = bool(envelope)
    if isinstance(envelope, str):
        env_col = envelope
    col_c = d.FG if col is None else col
    bg_c = coladj(-0.35) if bg is None else bg
    grid_col = grid if isinstance(grid, str) else coladj(-0.2)
    ltys = ['solid', 'dotted'] if lty is None else ([lty, lty] if isinstance(lty, str | int) else list(lty))

    res = rstandard(y) if type == 'rstandard' else rstudent(y)
    not_na = ~np.isnan(res.z)
    zi = res.z[not_na]
    slab = np.asarray(res.slab, dtype=object)[not_na]
    ord_ = np.argsort(zi, kind='stable')
    slab = slab[ord_]

    xq, yq = qqnorm_xy(zi)
    xt, yt = plain_plot(
        axx,
        (float(xq.min()), float(xq.max())),
        (float(yq.min()), float(yq.max())),
        'Theoretical Quantiles',
        'Sample Quantiles',
        'Normal Q-Q Plot',
        bty='l',
    )
    lb_y = ub_y = None
    temp_lb = temp_ub = None
    if draw_envelope:
        alpha = y.level if level is None else level_to_alpha(level)
        gen = rng if rng is not None else (set_seed(seed) if seed is not None else get_rng())
        dat = gen.rnorm(k * reps).reshape((k, reps), order='F')
        h = hatvalues(y, type='matrix')
        ei = (np.eye(k) - h) @ dat
        ei = np.sort(ei, axis=0)
        p_lo, p_hi = (alpha / 2 / k, 1 - alpha / 2 / k) if bonferroni else (alpha / 2, 1 - alpha / 2)
        lb = np.quantile(ei, p_lo, axis=1)
        ub = np.quantile(ei, p_hi, axis=1)
        temp_lb = qqnorm_xy(lb)
        temp_ub = qqnorm_xy(ub)
        if smooth:
            temp_lb = supsmu(temp_lb[0], temp_lb[1], bass)
            temp_ub = supsmu(temp_ub[0], temp_ub[1], bass)
        lb_y, ub_y = temp_lb[1], temp_ub[1]
        d.poly(
            axx,
            np.concatenate([temp_lb[0], temp_ub[0][::-1]]),
            np.concatenate([temp_lb[1], temp_ub[1][::-1]]),
            col=env_col,
            border=None,
            zorder=0.6,
        )
        d.line(axx, temp_lb[0], temp_lb[1], lty=ltys[1])
        d.line(axx, temp_ub[0], temp_ub[1], lty=ltys[1])
    if grid is not False:
        assert xt is not None
        assert yt is not None
        d.grid(axx, xt, yt, grid_col)
    d.abline(axx, 0.0, 1.0, lty=ltys[0])
    for a_, b_ in zip(xq, yq, strict=True):
        d.pts(axx, float(a_), float(b_), pch=pch, col=col_c, bg=bg_c)

    out = QQNormResult(axx, xq, yq, lb_y, ub_y)
    if label is False or label == 'none':
        return out
    if label is True or label == 'all':
        label = k
    pos_x, pos_y = xq[ord_], yq[ord_]

    def place(i: int) -> None:
        dx = pos_x[i] - pos_y[i]
        if pos <= 4:
            p = pos
        elif pos == 13:
            p = 1 if dx >= 0 else 3
        else:
            p = 2 if dx <= 0 else 4
        d.txt(axx, float(pos_x[i]), float(pos_y[i]), str(slab[i]), pos=p, offset=offset)

    if isinstance(label, int | float | np.integer | np.floating) and not isinstance(label, bool):
        nlab = round(float(label))
        if nlab < 1 or nlab > k:
            raise ValueError("Out of range value for 'label' argument.")
        dev = np.abs(pos_x - pos_y)
        for i in range(k):
            if int(np.sum(dev > dev[i])) < nlab:
                place(i)
    else:
        assert temp_lb is not None
        assert temp_ub is not None
        for i in range(k):
            if pos_y[i] < temp_lb[1][i] or pos_y[i] > temp_ub[1][i]:
                place(i)
    return out
