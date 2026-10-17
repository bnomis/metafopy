"""Row subsetting of the results that metafor classes as ``list.rma`` (``predict``, ``rstandard``, ``rstudent``, ``blup``, ``ranef``, ``leave1out``, ``cumul``).

Ports ``[.list.rma`` of https://github.com/wviechtb/metafor/blob/master/R/methods.list.rma.r: ``res[i]`` keeps the rows ``i`` of every vector or matrix
of the result and its study labels, and the settings of the result (``digits``, ``transf``, ...) unchanged. ``i`` is a NumPy index (0-based).
"""

import dataclasses
import typing

import numpy
import pandas

__all__ = ['rows']


def rows[R](result: R, index: typing.Any, size: int) -> R:
    """``result`` with the rows ``index`` of every field that has ``size`` rows.

    Raises:
        IndexError: ``index`` does not select rows of the result.
    """
    take = numpy.arange(size)[index]
    take = numpy.atleast_1d(take)
    changes: dict[str, typing.Any] = {}
    for field in dataclasses.fields(typing.cast(typing.Any, result)):
        value = getattr(result, field.name)
        if field.name in ('order', 'ids'):
            continue  # metafor's `[.list.rma` subsets only the fields before `slab`: the order and the ids stay as they were
        if isinstance(value, numpy.ndarray) and value.ndim >= 1 and value.shape[0] == size:
            changes[field.name] = value[take]
        elif isinstance(value, pandas.DataFrame) and value.shape[0] == size:
            changes[field.name] = value.iloc[take]
        elif isinstance(value, tuple) and len(value) == size and field.name == 'slab':
            changes[field.name] = tuple(value[int(j)] for j in take)
    return typing.cast(R, dataclasses.replace(typing.cast(typing.Any, result), **changes))
