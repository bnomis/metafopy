"""The frames the interactive forest and funnel charts draw (:class:`ChartData`, :func:`chart_data`).

All arithmetic is here and uses only public fields of the fit (``yi``, ``vi``, ``slab``, ``weights_()`` and ``predict()``), so the R fixtures that pin those
pin every number a chart shows. The intervals follow metafor's ``forest.rma``: ``yi +/- z * sqrt(vi)``, then ``transf``. See
https://wviechtb.github.io/metafor/reference/forest.rma.html.
"""

import collections.abc
import dataclasses

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.methods.predict
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.plot.rutils

__all__ = ['POOLED_COLUMNS', 'STUDY_COLUMNS', 'ChartData', 'chart_data', 'check_chart_data', 'zcrit']

type Fit = metafopy.models.rma_uni.RMAResult | metafopy.models.mv_result.RMAMVResult
type Transf = collections.abc.Callable[[numpy.typing.ArrayLike], numpy.ndarray]

STUDY_COLUMNS = ('slab', 'yi', 'vi', 'sei', 'ci_lb', 'ci_ub', 'weight', 'order', 'href', 'group', 'fill')
POOLED_COLUMNS = ('label', 'estimate', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub', 'k', 'tau2', 'I2')


@dataclasses.dataclass(frozen=True)
class ChartData:
    """The frames a chart draws: one row per study, and one row per pooled estimate.

    ``studies`` has the columns ``slab, yi, vi, sei, ci_lb, ci_ub, weight, order, href, group, fill`` (``ci_lb`` and ``ci_ub`` on the transformed
    scale when ``transformed``; ``weight`` in percent; ``order`` the position of the study in the input); ``pooled`` has ``label, estimate, ci_lb, ci_ub,
    pi_lb, pi_ub, k, tau2, I2``, one row per fit.
    """

    studies: pandas.DataFrame
    pooled: pandas.DataFrame
    measure: str
    transformed: bool


def zcrit(level: float | None, fit: Fit | None = None) -> float:
    """The normal critical value of a confidence ``level`` (a percentage, or 0.95), by default that of ``fit``."""
    alpha = fit.level if level is None and fit is not None else metafopy.plot.rutils.level_to_alpha(95 if level is None else level)
    return float(scipy.stats.norm.isf(alpha / 2))


def check_chart_data(data: ChartData) -> None:
    """Check that both frames of a hand-built :class:`ChartData` have every column.

    Raises:
        metafopy.errors.InputError: a column is missing.
    """
    for name, frame, columns in (('studies', data.studies, STUDY_COLUMNS), ('pooled', data.pooled, POOLED_COLUMNS)):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise metafopy.errors.InputError(f'ChartData.{name} is missing the column(s) {", ".join(repr(c) for c in missing)}.')


def full_labels(x: Fit, k_f: int) -> list[str]:
    """The study labels of every row of ``x`` before the omission of missing values."""
    slab_f = getattr(x, 'slab_f', None)
    if slab_f is not None and len(slab_f) == k_f:
        return [str(s) for s in slab_f]
    labels = [str(i + 1) for i in range(k_f)]
    for i, s in zip(numpy.flatnonzero(x.not_na), x.slab, strict=True):
        labels[int(i)] = str(s)
    return labels


def per_row(name: str, values: collections.abc.Sequence[object] | None, k_f: int) -> list[object]:
    """``values`` checked to have one entry per study of the fit before omission, ``None`` entries without values."""
    if values is None:
        return [None] * k_f
    if len(values) != k_f:
        raise metafopy.errors.InputError(f"Length of the '{name}' argument ({len(values)}) does not correspond to the number of studies ({k_f}).")
    return list(values)


def pooled_row(label: str, fit: Fit, transf: Transf | None, level: float | None) -> dict[str, object]:
    """One row of the ``pooled`` frame: the prediction of an intercept-only fit."""
    pred = metafopy.methods.predict.predict(fit, transf=transf, level=level)
    tau2 = fit.tau2
    return {
        'label': label,
        'estimate': float(pred.pred[0]),
        'ci_lb': float(pred.ci_lb[0]),
        'ci_ub': float(pred.ci_ub[0]),
        'pi_lb': float(pred.pi_lb[0]),
        'pi_ub': float(pred.pi_ub[0]),
        'k': int(fit.k),
        'tau2': float(tau2) if isinstance(tau2, int | float) else numpy.nan,
        'I2': float(getattr(fit, 'I2', numpy.nan)),
    }


def chart_data(
    x: Fit,
    *,
    transf: Transf | None = None,
    slab: collections.abc.Sequence[str] | None = None,
    href: collections.abc.Sequence[str | None] | None = None,
    groups: collections.abc.Sequence[str] | None = None,
    group_fits: collections.abc.Mapping[str, Fit] | None = None,
    level: float | None = None,
) -> ChartData:
    """Return the per-study and pooled frames a forest or funnel chart draws.

    Per-study intervals are ``yi +/- z(level) * sqrt(vi)``, as metafor's ``forest()`` draws them, then passed through ``transf``. Weights are
    ``weights_()`` of ``x``, as percentages. The pooled rows come from ``predict(x, transf=transf)`` for ``x``, then one row per entry of ``group_fits``,
    labelled by its key, in the mapping's order. ``groups`` gives each study's group label; it is required when ``group_fits`` is given and must have one
    entry per study of ``x``. ``slab``, ``href`` and ``groups`` index the studies of ``x`` before the omission of studies with a missing ``yi`` or ``vi``.
    ``href`` gives each study a link (any URL, including an in-page ``#anchor``); ``None`` leaves that study unlinked.

    Args:
        x: An intercept-only [rma_uni][metafopy.rma_uni] or [rma_mv][metafopy.rma_mv] fit.
        transf: Transformation of the intervals and the pooled rows (for example ``numpy.exp``).
        slab: Study labels, default those of ``x``.
        href: A link per study.
        groups: A group label per study.
        group_fits: A fit per group label, drawn as an extra pooled row.
        level: Confidence level in percent, default that of ``x``.

    Returns:
        The frames, a [ChartData][metafopy.interactive.data.ChartData].

    Raises:
        metafopy.errors.InputError: ``slab``, ``href`` or ``groups`` has the wrong length, or ``group_fits`` is given without ``groups``, or ``x`` has
            moderators (a forest of a meta-regression is not a pooled estimate).
    """
    if not x.int_only:
        raise metafopy.errors.InputError('chart_data() needs an intercept-only model; a forest of a meta-regression is not a pooled estimate.')
    if group_fits and groups is None:
        raise metafopy.errors.InputError("Argument 'groups' is required when 'group_fits' is given.")
    keep = numpy.asarray(x.not_na, dtype=bool)
    k_f = len(keep)
    labels = [str(s) for s in slab] if slab is not None else full_labels(x, k_f)
    if len(labels) != k_f:
        raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(labels)}) does not correspond to the number of studies ({k_f}).")
    links = per_row('href', href, k_f)
    group = per_row('groups', groups, k_f)
    fill_all = numpy.zeros(k_f, dtype=bool)
    fill = getattr(x, 'fill', None)
    if fill is not None and len(fill) == k_f:
        fill_all = numpy.asarray(fill, dtype=bool)
    rows = numpy.flatnonzero(keep)
    yi = numpy.asarray(x.yi_f, dtype=float)[rows]
    vi = numpy.asarray(x.vi_f, dtype=float)[rows]
    sei = numpy.sqrt(vi)
    z = zcrit(level, x)
    lb, ub = yi - z * sei, yi + z * sei
    if transf is not None:
        lb, ub = numpy.asarray(transf(lb), dtype=float), numpy.asarray(transf(ub), dtype=float)
    studies = pandas.DataFrame(
        {
            'slab': [labels[i] for i in rows],
            'yi': yi,
            'vi': vi,
            'sei': sei,
            'ci_lb': lb,
            'ci_ub': ub,
            'weight': numpy.asarray(x.weights_(), dtype=float),
            'order': rows,
            'href': pandas.Series([links[i] for i in rows], dtype=object),
            'group': pandas.Series([group[i] for i in rows], dtype=object),
            'fill': fill_all[rows],
        }
    )
    pooled = [pooled_row('Overall', x, transf, level)]
    for label, fit in (group_fits or {}).items():
        pooled.append(pooled_row(str(label), fit, transf, level))
    return ChartData(studies=studies, pooled=pandas.DataFrame(pooled, columns=list(POOLED_COLUMNS)), measure=str(x.measure), transformed=transf is not None)
