"""Contrast matrices for all pairwise differences between coefficients (metafor's ``pairmat``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metafopy.models.rma_uni import Btt, RMAResult, set_btt

__all__ = ['pairmat']


def pairmat(x: RMAResult, btt: Btt, btt2: Btt = None) -> pd.DataFrame:
    """Rows contrast each pair of the coefficients picked by ``btt`` (0-based, or a regex).

    For use as ``anova(res, X=pairmat(res, btt='alloc'))``. Rows for the
    coefficients picked by ``btt2`` themselves are placed first. metafor's
    ``pairmat()`` can omit ``x`` inside ``anova()``; here the model is always
    passed explicitly. See https://wviechtb.github.io/metafor/reference/pairmat.html.
    """
    if x.int_only:
        raise ValueError('Cannot construct contrast matrices for intercept-only models.')
    if btt is None:
        raise ValueError("Need to specify the 'btt' argument.")
    names = list(x.coef_names)
    idx = set_btt(btt, x.p, x.int_incl, names)
    p = len(idx)
    if p == 1:
        raise ValueError("Need to specify multiple coefficients via argument 'btt' for pairwise comparisons.")
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for i in range(p - 1):
        for j in range(i + 1, p):
            row = np.zeros(x.p)
            row[idx[i]] = -1.0
            row[idx[j]] = 1.0
            rows.append(row)
            labels.append(f'{names[idx[i]]}-{names[idx[j]]}')
    if btt2 is not None:
        idx2 = set_btt(btt2, x.p, x.int_incl, names)
        add = [np.eye(x.p)[b] for b in idx2]
        rows = add + rows
        labels = [names[b] for b in idx2] + labels
    return pd.DataFrame(np.vstack(rows), index=labels, columns=names)
