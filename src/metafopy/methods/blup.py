"""Best linear unbiased predictions of the true effects (metafor's ``blup.rma.uni``).

For study ``i`` with shrinkage factor ``lambda_i = tau^2 / (tau^2 + v_i)`` the
BLUP of its true effect is ``lambda_i y_i + (1 - lambda_i) x_i' beta``, with
prediction variance ``lambda_i v_i + (1 - lambda_i)^2 x_i' Var(beta) x_i``
(Raudenbush & Bryk 1985, appendix; doi:10.3102/10769986010002075). The
computations follow https://wviechtb.github.io/metafor/reference/blup.html.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from metafopy._print import fmt_num
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult, level_to_alpha

__all__ = ['BlupResult', 'blup']

type FloatArray = NDArray[np.float64]
type NAAction = Literal['omit', 'exclude', 'pass', 'fail']
type Transf = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class BlupResult:
    """Predicted values with standard errors and prediction intervals (metafor's ``list.rma``).

    ``se`` is NaN when a transformation was applied, as in metafor.
    """

    pred: FloatArray
    se: FloatArray
    pi_lb: FloatArray
    pi_ub: FloatArray
    slab: tuple[str, ...]
    digits: int = 4
    transf: bool = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {'pred': self.pred, 'se': self.se, 'pi.lb': self.pi_lb, 'pi.ub': self.pi_ub},
            index=list(self.slab),
        )

    def __str__(self) -> str:
        # print.list.rma: the se column is left out after a transformation
        cols: list[tuple[str, FloatArray]] = [('pred', self.pred)]
        if not self.transf:
            cols.append(('se', self.se))
        cols += [('pi.lb', self.pi_lb), ('pi.ub', self.pi_ub)]
        table = format_r_frame([(name, [fmt_num(float(v), self.digits) for v in vals]) for name, vals in cols], self.slab)
        return '\n'.join(['', *table, ''])


def format_r_frame(columns: Sequence[tuple[str, Sequence[str]]], row_names: Sequence[str], gap: int = 1, width: int = 79) -> list[str]:
    """Lines of an R ``print.data.frame(..., right=TRUE)`` of already formatted columns.

    Like R, columns that do not fit into ``width`` characters continue in a
    second block of lines below the first.
    """
    name_w = max((len(n) for n in row_names), default=0)
    sep = ' ' * gap
    widths = [max(len(head), *(len(c) for c in cells)) if cells else len(head) for head, cells in columns]
    blocks: list[list[int]] = []
    used = name_w
    for j, w in enumerate(widths):
        if not blocks or used + gap + w > width:
            blocks.append([])
            used = name_w
        blocks[-1].append(j)
        used += gap + w
    lines: list[str] = []
    for block in blocks:
        lines.append((' ' * name_w + ''.join(sep + columns[j][0].rjust(widths[j]) for j in block)).rstrip())
        for i, name in enumerate(row_names):
            lines.append((name.ljust(name_w) + ''.join(sep + columns[j][1][i].rjust(widths[j]) for j in block)).rstrip())
    return lines


@singledispatch
def blup(res: Any, *args: Any, **kwargs: Any) -> BlupResult:
    """BLUPs of the true effects for a fitted model; see :func:`_blup_rma_uni`."""
    raise TypeError(f'blup() not implemented for {type(res).__name__}')


@blup.register
def _blup_rma_uni(
    res: RMAResult,
    level: float | None = None,
    digits: int | None = None,
    transf: Transf | None = None,
    targs: Any = None,
    *,
    na_action: NAAction = 'omit',
) -> BlupResult:
    """BLUPs of the true effects of an ``rma.uni`` fit.

    ``transf`` is applied (as ``transf(x)`` or ``transf(x, targs)``) to the
    predicted values and interval bounds. ``na_action`` plays the role of R's
    ``options(na.action=)``: with ``'exclude'``/``'pass'`` the result has one
    row per supplied study, NaN for those left out of the fit.
    """
    return shrinkage_predictions(res, 'blup', level, digits, transf, targs, na_action)


def shrinkage_predictions(
    res: RMAResult,
    kind: Literal['blup', 'ranef'],
    level: float | None,
    digits: int | None,
    transf: Transf | None,
    targs: Any,
    na_action: NAAction,
) -> BlupResult:
    """Shared body of :func:`blup` and :func:`metafopy.methods.ranef.ranef`."""
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise ValueError(f'unknown na_action {na_action!r}')
    if res.weights is not None or not res.weighted:
        raise ValueError('Extraction of random effects not available for models with non-standard weights.')

    alpha = res.level if level is None else level_to_alpha(level)
    if res.ddf is not None:
        crit = float(stats.t.isf(alpha / 2, res.ddf)) if res.ddf > 0 else np.nan
    else:
        crit = float(stats.norm.isf(alpha / 2))

    # shrinkage factors; an infinite tau^2 means no shrinkage at all
    tau2 = res.tau2
    with np.errstate(invalid='ignore', divide='ignore'):
        li = np.ones(res.k) if np.isinf(tau2) else tau2 / (tau2 + res.vi)
    xb = res.X @ res.beta
    xvx = np.einsum('ij,jk,ik->i', res.X, res.vb, res.X)

    if kind == 'blup':
        pred = li * res.yi + (1 - li) * xb
        vpred = np.where(li == 1, li * res.vi, li * res.vi + (1 - li) ** 2 * xvx)
    elif res.method in FIXED_EFFECTS:
        pred = np.zeros(res.k)
        vpred = np.zeros(res.k)
    else:
        pred = li * (res.yi - xb)
        vpred = li * res.vi + li**2 * xvx

    with np.errstate(invalid='ignore'):
        se = np.sqrt(vpred)
    pi_lb = pred - crit * se
    pi_ub = pred + crit * se

    if transf is not None:
        pred, pi_lb, pi_ub = (np.asarray(transf(a) if targs is None else transf(a, targs), dtype=np.float64) for a in (pred, pi_lb, pi_ub))
        se = np.full(res.k, np.nan)

    # make sure the interval bounds are in increasing order (NaNs stay put)
    flip = pi_lb > pi_ub
    pi_lb, pi_ub = np.where(flip, pi_ub, pi_lb), np.where(flip, pi_lb, pi_ub)

    slab = res.slab
    if na_action in ('exclude', 'pass'):
        # pad back to the supplied rows; dropped rows are labelled by position
        keep = res.data.keep
        idx = np.flatnonzero(keep)
        full = [np.full(keep.shape[0], np.nan) for _ in range(4)]
        for out, vals in zip(full, (pred, se, pi_lb, pi_ub), strict=True):
            out[idx] = vals
        pred, se, pi_lb, pi_ub = full
        labels = [str(i + 1) for i in range(keep.shape[0])]
        for pos, i in enumerate(idx):
            labels[i] = res.slab[pos]
        slab = tuple(labels)
    elif na_action == 'fail' and not res.data.keep.all():
        raise ValueError('Missing values in results.')

    return BlupResult(
        pred=pred,
        se=se,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        digits=4 if digits is None else digits,
        transf=transf is not None,
    )
