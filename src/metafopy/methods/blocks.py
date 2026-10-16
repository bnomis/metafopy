"""Block diagonal matrices: building one from sub-matrices (``bldiag``) and splitting one into sub-matrices (``blsplit``).

R sources: https://github.com/wviechtb/metafor/blob/master/R/bldiag.r and
https://github.com/wviechtb/metafor/blob/master/R/blsplit.r. Documentation:
https://wviechtb.github.io/metafor/reference/bldiag.html and
https://wviechtb.github.io/metafor/reference/blsplit.html.
"""

import collections.abc
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.core.linalg
import metafopy.errors

__all__ = ['bldiag', 'blsplit']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Block = numpy.typing.ArrayLike | pandas.DataFrame | scipy.sparse.sparray | scipy.sparse.spmatrix


def is_number(value: object) -> bool:
    return isinstance(value, int | float | numpy.number) and not isinstance(value, bool)


def to_matrix(value: Block) -> FloatArray:
    """A matrix as it enters ``bldiag``: sparse and data frames become dense, scalars and vectors become diagonal matrices.

    Raises:
        metafopy.errors.ArgumentError: the value has more than two dimensions.
    """
    if isinstance(value, scipy.sparse.sparray | scipy.sparse.spmatrix):
        return numpy.asarray(scipy.sparse.csr_array(typing.cast(typing.Any, value)).toarray(), dtype=numpy.float64)
    if isinstance(value, pandas.DataFrame):
        return value.to_numpy(dtype=numpy.float64)
    arr = numpy.asarray(value, dtype=numpy.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        raise metafopy.errors.ArgumentError(f'bldiag() needs matrices, vectors or scalars, got an array with {arr.ndim} dimensions')
    return numpy.diag(numpy.atleast_1d(arr))


def bldiag(*mlist: Block, order: numpy.typing.ArrayLike | None = None, order_is_permutation: bool = False) -> FloatArray:
    """Block diagonal matrix from matrices (metafor's ``bldiag``).

    Accepts matrices (arrays, data frames or scipy sparse arrays), vectors and scalars (turned into
    diagonal matrices, so ``bldiag(numpy.ones((3, 3)), 2)`` works) or a single list or tuple of these.
    Matrices with zero rows or columns are dropped; if none are left the result is 0 x 0.

    ``order`` is the clustering variable according to which the rows and columns of the result are
    reordered back to the original order of the data (R: ``order=cluster``). With
    ``order_is_permutation=True`` it is instead the permutation itself (R: ``order=order(cluster)``).

    Raises:
        metafopy.errors.InputError: ``order`` is used with a non-square result or has the wrong length.
        metafopy.errors.ArgumentError: an element is neither a matrix, a vector nor a scalar.
    """
    items: list[Block] = list(mlist)
    if len(items) == 1:
        only = items[0]
        if isinstance(only, list | tuple) and not all(is_number(v) for v in only):
            items = list(only)
    mats = [to_matrix(m) for m in items]
    mats = [m for m in mats if m.shape[0] != 0 and m.shape[1] != 0]
    if not mats:
        return numpy.zeros((0, 0))
    row_ends = numpy.cumsum([m.shape[0] for m in mats])
    col_ends = numpy.cumsum([m.shape[1] for m in mats])
    out = numpy.zeros((int(row_ends[-1]), int(col_ends[-1])))
    for m, r1, c1 in zip(mats, row_ends, col_ends, strict=True):
        out[r1 - m.shape[0] : r1, c1 - m.shape[1] : c1] = m
    if order is None:
        return out
    if out.shape[0] != out.shape[1]:
        raise metafopy.errors.InputError("Can only use 'order' argument for square matrices.")
    ordering = numpy.asarray(order)
    if ordering.shape[0] != out.shape[0]:
        raise metafopy.errors.InputError(
            f"Length of the 'order' argument ({ordering.shape[0]}) does not correspond to the dimensions of the matrix ({out.shape[0]}x{out.shape[1]})."
        )
    sort_vec = ordering.astype(numpy.intp) if order_is_permutation else numpy.argsort(ordering, kind='stable')
    placed = numpy.empty_like(out)
    placed[numpy.ix_(sort_vec, sort_vec)] = out
    return placed


def python_scalar(value: object) -> object:
    return value.item() if isinstance(value, numpy.generic) else value


def blsplit(
    x: FloatArray | scipy.sparse.sparray | scipy.sparse.spmatrix,
    cluster: numpy.typing.ArrayLike,
    fun: collections.abc.Callable[..., typing.Any] | None = None,
    args: typing.Any = None,
    sort: bool = False,
) -> dict[typing.Any, typing.Any]:
    """Split a block diagonal matrix into a dict of sub-matrices, one per unique ``cluster`` value (metafor's ``blsplit``).

    The keys are the unique cluster values in order of appearance (sorted if ``sort``). ``fun``, if
    given, is applied to each sub-matrix with ``args`` as additional positional arguments, i.e.
    ``fun(block, *args)`` (R: ``blsplit(V, cluster, round, 4)``; a single value is the one argument). A scipy sparse ``x`` gives sparse
    sub-matrices.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a matrix.
        metafopy.errors.InputError: ``x`` is not symmetric or ``cluster`` does not match its dimensions.
    """
    if cluster is None:
        raise metafopy.errors.InputError("Must specify the 'cluster' variable.")
    dense: FloatArray | None = None
    sparse_x: scipy.sparse.csr_array | None = None
    if isinstance(x, scipy.sparse.sparray | scipy.sparse.spmatrix):
        sparse_x = scipy.sparse.csr_array(typing.cast(typing.Any, x))
        diff = float(abs(sparse_x - sparse_x.T).sum())
        total = float(abs(sparse_x).sum())
        symmetric = sparse_x.shape[0] == sparse_x.shape[1] and (diff == 0 or (total > 0 and diff / total < 100 * float(numpy.finfo(float).eps)))
    elif isinstance(x, numpy.ndarray) and x.ndim == 2:
        dense = numpy.asarray(x, dtype=numpy.float64)
        symmetric = metafopy.core.linalg.is_symmetric(dense)
    else:
        raise metafopy.errors.ArgumentError("Argument 'x' must be a matrix.")
    if not symmetric:
        raise metafopy.errors.InputError("Argument 'x' must be a symmetric matrix.")
    k = int(typing.cast(typing.Any, x).shape[0])
    groups = numpy.asarray(cluster)
    if groups.ndim == 0:
        groups = groups[None]
    if groups.shape[0] != k:
        raise metafopy.errors.InputError(
            f"Length of the variable specified via 'cluster' ({groups.shape[0]}) does not correspond to the dimensions of the matrix ({k}x{k})."
        )
    unique = list(pandas.unique(groups))
    if sort:
        unique = sorted(unique)
    res: dict[typing.Any, typing.Any] = {}
    for value in unique:
        idx = numpy.flatnonzero(groups == value)
        if sparse_x is not None:
            block: typing.Any = sparse_x[idx][:, idx]
        elif dense is not None:
            block = dense[numpy.ix_(idx, idx)]
        else:
            raise metafopy.errors.InternalError('blsplit() has neither a dense nor a sparse matrix')
        if fun is not None:
            block = fun(block) if args is None else fun(block, *(args if isinstance(args, list | tuple) else numpy.atleast_1d(args).tolist()))
        res[python_scalar(value)] = block
    return res
