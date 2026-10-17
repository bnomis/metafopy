"""Contrast matrices for all pairwise differences between coefficients (metafor's ``pairmat``)."""

import dataclasses
import typing

import numpy
import pandas

import metafopy.errors
import metafopy.models.rma_uni

__all__ = ['PairmatSpec', 'pairmat', 'resolve']


@dataclasses.dataclass(frozen=True, slots=True)
class PairmatSpec:
    """``pairmat(btt=...)`` without the model, which :func:`metafopy.anova` and :func:`metafopy.predict` complete with the model they are called on.

    metafor's ``pairmat()`` finds the model in the call of ``anova()`` or ``predict()`` it is an argument of; the arguments are evaluated before the call
    here, so the call is kept until the model is known.
    """

    btt: metafopy.models.rma_uni.Btt
    btt2: metafopy.models.rma_uni.Btt = None
    fixed: bool = False


@typing.overload
def pairmat(
    x: metafopy.models.rma_uni.RMAResult, btt: metafopy.models.rma_uni.Btt, btt2: metafopy.models.rma_uni.Btt = None, *, fixed: bool = False
) -> pandas.DataFrame: ...


@typing.overload
def pairmat(
    x: metafopy.models.rma_uni.Btt = None, btt: metafopy.models.rma_uni.Btt = None, btt2: metafopy.models.rma_uni.Btt = None, *, fixed: bool = False
) -> PairmatSpec: ...


def pairmat(
    x: typing.Any = None, btt: metafopy.models.rma_uni.Btt = None, btt2: metafopy.models.rma_uni.Btt = None, *, fixed: bool = False
) -> pandas.DataFrame | PairmatSpec:
    """Rows contrast each pair of the coefficients picked by ``btt`` (0-based, or a regex).

    For use as ``anova(res, X=pairmat(res, btt='alloc'))``, or without the model, ``anova(res, X=pairmat(btt='alloc'))``, as in metafor, where the
    model is that of the ``anova()`` or ``predict()`` call. Rows for the coefficients picked by ``btt2`` themselves are placed first.
    ``fixed=True`` matches a string ``btt`` or ``btt2`` literally instead of as a regular expression (metafor's ``...`` argument).
    See https://wviechtb.github.io/metafor/reference/pairmat.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if not hasattr(x, 'coef_names'):  # metafor: an atomic first argument is btt and the model comes from the call this is an argument of
        if x is not None:
            btt = x
        if btt is None:
            raise metafopy.errors.InputError("Need to specify the 'btt' argument.")
        return PairmatSpec(btt, btt2, fixed)
    return contrasts(x, btt, btt2, fixed)


def resolve[T](value: T | PairmatSpec, model: typing.Any) -> T | pandas.DataFrame:
    """``value`` with a :class:`PairmatSpec` completed with ``model``."""
    if isinstance(value, PairmatSpec):
        return contrasts(model, value.btt, value.btt2, value.fixed)
    return value


def contrasts(
    x: metafopy.models.rma_uni.RMAResult, btt: metafopy.models.rma_uni.Btt, btt2: metafopy.models.rma_uni.Btt, fixed: bool = False
) -> pandas.DataFrame:
    """The pairwise contrasts of the coefficients ``btt`` of the fit ``x``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if x.int_only:
        raise metafopy.errors.InputError('Cannot construct contrast matrices for intercept-only models.')
    if btt is None:
        raise metafopy.errors.InputError("Need to specify the 'btt' argument.")
    names = list(x.coef_names)
    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, names, fixed)
    p = len(idx)
    if p == 1:
        raise metafopy.errors.InputError("Need to specify multiple coefficients via argument 'btt' for pairwise comparisons.")
    rows: list[numpy.ndarray] = []
    labels: list[str] = []
    for i in range(p - 1):
        for j in range(i + 1, p):
            row = numpy.zeros(x.p)
            row[idx[i]] = -1.0
            row[idx[j]] = 1.0
            rows.append(row)
            labels.append(f'{names[idx[i]]}-{names[idx[j]]}')
    if btt2 is not None:
        idx2 = metafopy.models.rma_uni.set_btt(btt2, x.p, x.int_incl, names, fixed)
        add = [numpy.eye(x.p)[b] for b in idx2]
        rows = add + rows
        labels = [names[b] for b in idx2] + labels
    return pandas.DataFrame(numpy.vstack(rows), index=labels, columns=names)
