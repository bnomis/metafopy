"""Contrast matrices for all pairwise differences between coefficients (metafor's ``pairmat``)."""

import numpy
import pandas

import metafopy.errors
import metafopy.models.rma_uni

__all__ = ['pairmat']


def pairmat(x: metafopy.models.rma_uni.RMAResult, btt: metafopy.models.rma_uni.Btt, btt2: metafopy.models.rma_uni.Btt = None) -> pandas.DataFrame:
    """Rows contrast each pair of the coefficients picked by ``btt`` (0-based, or a regex).

    For use as ``anova(res, X=pairmat(res, btt='alloc'))``. Rows for the
    coefficients picked by ``btt2`` themselves are placed first. metafor's
    ``pairmat()`` can omit ``x`` inside ``anova()``; here the model is always
    passed explicitly. See https://wviechtb.github.io/metafor/reference/pairmat.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if x.int_only:
        raise metafopy.errors.InputError('Cannot construct contrast matrices for intercept-only models.')
    if btt is None:
        raise metafopy.errors.InputError("Need to specify the 'btt' argument.")
    names = list(x.coef_names)
    idx = metafopy.models.rma_uni.set_btt(btt, x.p, x.int_incl, names)
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
        idx2 = metafopy.models.rma_uni.set_btt(btt2, x.p, x.int_incl, names)
        add = [numpy.eye(x.p)[b] for b in idx2]
        rows = add + rows
        labels = [names[b] for b in idx2] + labels
    # as in metafor, the rows are named but the columns are not, so that predict() takes them by position
    return pandas.DataFrame(numpy.vstack(rows), index=labels)
