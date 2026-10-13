"""Best linear unbiased predictions of the true effects (metafor's ``blup.rma.uni``).

For study ``i`` with shrinkage factor ``lambda_i = tau^2 / (tau^2 + v_i)`` the
BLUP of its true effect is ``lambda_i y_i + (1 - lambda_i) x_i' beta``, with
prediction variance ``lambda_i v_i + (1 - lambda_i)^2 x_i' Var(beta) x_i``
(Raudenbush & Bryk 1985, appendix; doi:10.3102/10769986010002075). The
computations follow https://wviechtb.github.io/metafor/reference/blup.html.
"""

import collections.abc
import dataclasses
import functools
import typing

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing

__all__ = ['BlupResult', 'blup']

type FloatArray = numpy.typing.NDArray[numpy.float64]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type NAAction = typing.Literal['omit', 'exclude', 'pass', 'fail']
type Transf = collections.abc.Callable[..., typing.Any]


@dataclasses.dataclass(frozen=True, slots=True)
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

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {'pred': self.pred, 'se': self.se, 'pi.lb': self.pi_lb, 'pi.ub': self.pi_ub},
            index=list(self.slab),
        )

    def __str__(self) -> str:
        # print.list.rma: the se column is left out after a transformation
        cols: list[tuple[str, FloatArray]] = [('pred', self.pred)]
        if not self.transf:
            cols.append(('se', self.se))
        cols += [('pi.lb', self.pi_lb), ('pi.ub', self.pi_ub)]
        table = format_r_frame([(name, [metafopy.printing.fmt_num(float(v), self.digits) for v in vals]) for name, vals in cols], self.slab)
        return '\n'.join(['', *table, ''])


def format_r_frame(
    columns: collections.abc.Sequence[tuple[str, collections.abc.Sequence[str]]],
    row_names: collections.abc.Sequence[str],
    gap: int = 1,
    width: int = 79,
) -> list[str]:
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


@functools.singledispatch
def blup(res: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> BlupResult:
    """BLUPs of the true effects for a fitted model; see :func:`_blup_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'blup() not implemented for {type(res).__name__}')


@blup.register
def blup_rma_uni(
    res: metafopy.models.rma_uni.RMAResult,
    level: float | None = None,
    digits: int | None = None,
    transf: Transf | None = None,
    targs: typing.Any = None,
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
    res: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult,
    kind: typing.Literal['blup', 'ranef'],
    level: float | None,
    digits: int | None,
    transf: Transf | None,
    targs: typing.Any,
    na_action: NAAction,
) -> BlupResult:
    """Shared body of :func:`blup` and :func:`metafopy.methods.ranef.ranef`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise metafopy.errors.InputError(f'unknown na_action {na_action!r}')
    if res.weights is not None or not res.weighted:
        raise metafopy.errors.InputError('Extraction of random effects not available for models with non-standard weights.')

    alpha = res.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    if res.ddf is not None:
        crit = float(scipy.stats.t.isf(alpha / 2, res.ddf)) if res.ddf > 0 else numpy.nan
    else:
        crit = float(scipy.stats.norm.isf(alpha / 2))

    # shrinkage factors; an infinite tau^2 means no shrinkage at all
    tau2 = res.tau2
    with numpy.errstate(invalid='ignore', divide='ignore'):
        li = numpy.where(numpy.isinf(tau2), 1.0, tau2 / (tau2 + res.vi))
    xb = res.X @ res.beta
    xvx = numpy.einsum('ij,jk,ik->i', res.X, res.vb, res.X)

    if kind == 'blup':
        pred = li * res.yi + (1 - li) * xb
        vpred = numpy.where(li == 1, li * res.vi, li * res.vi + (1 - li) ** 2 * xvx)
    elif res.method in metafopy.models.tau2.FIXED_EFFECTS:
        pred = numpy.zeros(res.k)
        vpred = numpy.zeros(res.k)
    else:
        pred = li * (res.yi - xb)
        vpred = li * res.vi + li**2 * xvx

    with numpy.errstate(invalid='ignore'):
        se = numpy.sqrt(vpred)
    pi_lb = pred - crit * se
    pi_ub = pred + crit * se

    if transf is not None:
        pred, pi_lb, pi_ub = (numpy.asarray(transf(a) if targs is None else transf(a, targs), dtype=numpy.float64) for a in (pred, pi_lb, pi_ub))
        se = numpy.full(res.k, numpy.nan)

    # make sure the interval bounds are in increasing order (NaNs stay put)
    flip = pi_lb > pi_ub
    pi_lb, pi_ub = numpy.where(flip, pi_ub, pi_lb), numpy.where(flip, pi_lb, pi_ub)

    slab = res.slab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action in ('exclude', 'pass'):
        # pad back to the supplied rows; dropped rows are labelled by position
        keep = res.data.keep
        idx = numpy.flatnonzero(keep)
        full = [numpy.full(keep.shape[0], numpy.nan) for _ in range(4)]
        for out, vals in zip(full, (pred, se, pi_lb, pi_ub), strict=True):
            out[idx] = vals
        pred, se, pi_lb, pi_ub = full
        labels = [str(i + 1) for i in range(keep.shape[0])]
        for pos, i in enumerate(idx):
            labels[i] = res.slab[pos]
        slab = tuple(labels)
    elif na_action == 'fail' and not res.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')

    return BlupResult(
        pred=pred,
        se=se,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        digits=4 if digits is None else digits,
        transf=transf is not None,
    )
