"""Normal quantile-quantile plot of the (standardised or studentised) residuals (metafor's ``qqnorm.rma.uni``).

The plot can show a simulated pseudo confidence envelope around the line of equality, obtained as in Atkinson (1985)
(``Plots, Transformations and Regression``) by simulating normal data and applying the model's hat matrix; see
https://wviechtb.github.io/metafor/reference/qqnorm.rma.html.
"""

import collections.abc
import dataclasses

import matplotlib.axes
import numpy
import numpy.typing
import pandas

import metafopy.core.rng
import metafopy.errors
import metafopy.methods.influence
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['QQNormResult', 'qqnorm']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class QQNormResult:
    """Theoretical (``x``) and sample (``y``) quantiles (what ``qqnorm()`` returns), the envelope (if drawn) and the axes."""

    ax: matplotlib.axes.Axes
    x: FloatArray
    y: FloatArray
    lb: FloatArray | None = None
    ub: FloatArray | None = None

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'x': self.x, 'y': self.y})


def qqnorm(
    y: metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult,
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
    lty: str | collections.abc.Sequence[str] | None = None,
    *,
    seed: int | None = None,
    rng: metafopy.core.rng.RRandom | None = None,
    ax: matplotlib.axes.Axes | None = None,
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

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: ``y`` is an ``rma.mv`` fit (``qqnorm.rma.mv`` only stops in metafor).
    """
    if isinstance(y, metafopy.models.mv_result.RMAMVResult):
        raise metafopy.errors.NotSupportedError('Method not available for objects of class "rma.mv".')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if type not in ('rstandard', 'rstudent'):
        raise metafopy.errors.InputError("type must be 'rstandard' or 'rstudent'")
    if y.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    axx = metafopy.plot.rutils.get_ax(ax)
    k = y.k
    env_col = metafopy.plot.rutils.coladj(-0.15)
    if label == 'out' and isinstance(envelope, bool):
        envelope = True
    draw_envelope = bool(envelope)
    if isinstance(envelope, str):
        env_col = envelope
    col_c = metafopy.plot.draw.FG if col is None else col
    bg_c = metafopy.plot.rutils.coladj(-0.35) if bg is None else bg
    grid_col = grid if isinstance(grid, str) else metafopy.plot.rutils.coladj(-0.2)
    ltys = ['solid', 'dotted'] if lty is None else ([lty, lty] if isinstance(lty, str | int) else list(lty))

    res = metafopy.methods.influence.rstandard(y) if type == 'rstandard' else metafopy.methods.influence.rstudent(y)
    not_na = ~numpy.isnan(res.z)
    zi = res.z[not_na]
    slab = numpy.asarray(res.slab, dtype=object)[not_na]
    ord_ = numpy.argsort(zi, kind='stable')
    slab = slab[ord_]

    xq, yq = metafopy.plot.rutils.qqnorm_xy(zi)
    xt, yt = metafopy.plot.plain.plain_plot(
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
        alpha = y.level if level is None else metafopy.plot.rutils.level_to_alpha(level)
        gen = rng if rng is not None else (metafopy.core.rng.set_seed(seed) if seed is not None else metafopy.core.rng.get_rng())
        dat = gen.rnorm(k * reps).reshape((k, reps), order='F')
        h = metafopy.methods.influence.hatvalues(y, type='matrix')
        ei = (numpy.eye(k) - h) @ dat
        ei = numpy.sort(ei, axis=0)
        p_lo, p_hi = (alpha / 2 / k, 1 - alpha / 2 / k) if bonferroni else (alpha / 2, 1 - alpha / 2)
        lb = numpy.quantile(ei, p_lo, axis=1)
        ub = numpy.quantile(ei, p_hi, axis=1)
        temp_lb = metafopy.plot.rutils.qqnorm_xy(lb)
        temp_ub = metafopy.plot.rutils.qqnorm_xy(ub)
        if smooth:
            temp_lb = metafopy.plot.rutils.supsmu(temp_lb[0], temp_lb[1], bass)
            temp_ub = metafopy.plot.rutils.supsmu(temp_ub[0], temp_ub[1], bass)
        lb_y, ub_y = temp_lb[1], temp_ub[1]
        metafopy.plot.draw.poly(
            axx,
            numpy.concatenate([temp_lb[0], temp_ub[0][::-1]]),
            numpy.concatenate([temp_lb[1], temp_ub[1][::-1]]),
            col=env_col,
            border=None,
            zorder=0.6,
        )
        metafopy.plot.draw.line(axx, temp_lb[0], temp_lb[1], lty=ltys[1])
        metafopy.plot.draw.line(axx, temp_ub[0], temp_ub[1], lty=ltys[1])
    if grid is not False:
        assert xt is not None
        assert yt is not None
        metafopy.plot.draw.grid(axx, xt, yt, grid_col)
    metafopy.plot.draw.abline(axx, 0.0, 1.0, lty=ltys[0])
    for a_, b_ in zip(xq, yq, strict=True):
        metafopy.plot.draw.pts(axx, float(a_), float(b_), pch=pch, col=col_c, bg=bg_c)

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
        metafopy.plot.draw.txt(axx, float(pos_x[i]), float(pos_y[i]), str(slab[i]), pos=p, offset=offset)

    if isinstance(label, int | float | numpy.integer | numpy.floating) and not isinstance(label, bool):
        nlab = round(float(label))
        if nlab < 1 or nlab > k:
            raise metafopy.errors.InputError("Out of range value for 'label' argument.")
        dev = numpy.abs(pos_x - pos_y)
        for i in range(k):
            if int(numpy.sum(dev > dev[i])) < nlab:
                place(i)
    else:
        assert temp_lb is not None
        assert temp_ub is not None
        for i in range(k):
            if pos_y[i] < temp_lb[1][i] or pos_y[i] > temp_ub[1][i]:
                place(i)
    return out
