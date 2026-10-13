"""Argument handling of :func:`metafopy.rma_mv`: ``yi``, ``V``, ``W``, the model matrix and the ``random`` formulas.

Follows the first part of ``rma.mv()`` in metafor
(https://github.com/wviechtb/metafor/blob/master/R/rma.mv.r).
"""

import collections.abc
import dataclasses
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse

import metafopy.errors
import metafopy.formula
import metafopy.metadata
import metafopy.models.mv_factor
import metafopy.models.mv_structs

__all__ = ['RandomSpec', 'build_x', 'extract_random', 'prepare_v', 'prepare_w', 'prepare_yi', 'take_square']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Mat = FloatArray | scipy.sparse.csc_array
type VLike = str | float | numpy.typing.ArrayLike | pandas.DataFrame | scipy.sparse.sparray | collections.abc.Sequence[numpy.typing.ArrayLike]

SYMMETRY_TOL = 100 * float(numpy.finfo(float).eps)


@dataclasses.dataclass(slots=True)
class RandomSpec:
    """The ``random`` argument split into ``~ 1 | id`` terms and up to two ``~ inner | outer`` terms."""

    s_names: list[str]
    s_factors: list[metafopy.models.mv_factor.Factor]
    g_inner: pandas.DataFrame | None
    g_outer: pandas.Series | None
    g_formula: metafopy.formula.RandomTerm | None
    h_inner: pandas.DataFrame | None
    h_outer: pandas.Series | None
    h_formula: metafopy.formula.RandomTerm | None
    struct: tuple[str, str]
    raw: list[str]

    @property
    def with_s(self) -> bool:
        return bool(self.s_factors)

    @property
    def with_g(self) -> bool:
        return self.g_inner is not None

    @property
    def with_h(self) -> bool:
        return self.h_inner is not None


def column_values(value: object, data: pandas.DataFrame | None, name: str) -> FloatArray:
    """A numeric vector from a column name, a Series or an array-like.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(value, str):
        if data is None:
            raise metafopy.errors.InputError(f'{name}={value!r} names a column, but no data was given')
        if value in data.columns:
            value = data[value].to_numpy()
        else:
            try:
                value = numpy.asarray(data.eval(value))
            except (NameError, SyntaxError, KeyError, ValueError, TypeError, pandas.errors.UndefinedVariableError) as exc:
                raise metafopy.errors.NotFoundError(f'cannot find {value!r} (for {name}) in the data') from exc
    try:
        arr = numpy.asarray(value, dtype=numpy.float64)
    except (TypeError, ValueError) as exc:
        raise metafopy.errors.InputError(f'The object/variable specified for the {name!r} argument is not numeric.') from exc
    return arr


def prepare_yi(
    yi: object,
    data: pandas.DataFrame | None,
    mods: object,
    intercept: bool,
) -> tuple[FloatArray, object, bool]:
    """``yi`` as a vector; a formula in ``yi`` (``'y ~ x1 + x2'``) supplies ``mods`` and sets ``intercept=False``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(yi, str) and '~' in yi:
        if data is None:
            raise metafopy.errors.InputError('a model formula in yi needs data')
        lhs, rhs = (part.strip() for part in yi.split('~', 1))
        if not lhs:
            raise metafopy.errors.InputError('the model formula in yi must have a left-hand side')
        vec = column_values(lhs, data, 'yi')
        return vec, '~ ' + rhs, False
    if isinstance(yi, pandas.DataFrame):
        if yi.shape[1] != 1:
            raise metafopy.errors.InputError("The object/variable specified for the 'yi' argument is a data frame with multiple columns.")
        yi = yi.iloc[:, 0]
    arr = column_values(yi, data, 'yi')
    if arr.ndim == 2:
        if min(arr.shape) != 1:
            raise metafopy.errors.InputError("The object/variable specified for the 'yi' argument is a matrix with multiple rows/columns.")
        arr = arr.reshape(-1)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise metafopy.errors.InputError("The object/variable specified for the 'yi' argument is an array.")
    return arr, mods, intercept


def is_symmetric(M: Mat) -> bool:
    """``isSymmetric()``: the mean relative difference between ``M`` and its transpose is within ``100 * eps``."""
    if scipy.sparse.issparse(M):
        diff = abs(M - M.T).sum()
        scale = abs(M).sum()
    else:
        diff = float(numpy.abs(M - M.T).sum())
        scale = float(numpy.abs(M).sum())
    return bool(diff == 0 or (scale > 0 and diff / scale <= SYMMETRY_TOL))


def block_diagonal(blocks: collections.abc.Sequence[object], sparse: bool) -> Mat:
    mats = [numpy.atleast_2d(numpy.asarray(b, dtype=numpy.float64)) for b in blocks]
    if any(m.ndim != 2 or m.shape[0] != m.shape[1] for m in mats):
        raise metafopy.errors.InputError("All list elements in 'V' must be square matrices.")
    if sparse:
        return scipy.sparse.csc_array(scipy.sparse.block_diag([scipy.sparse.csc_array(m) for m in mats], format='csc'))
    return numpy.asarray(scipy.linalg.block_diag(*mats))


def prepare_v(V: VLike, data: pandas.DataFrame | None, k: int, sparse: bool) -> tuple[Mat, bool]:
    """``V`` as a ``k x k`` matrix (dense or sparse) and whether the user constrained it to 0.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if V is None:
        raise metafopy.errors.InputError("Must specify the 'V' argument.")
    if isinstance(V, list | tuple) and V and not numpy.isscalar(V[0]):
        mat: Mat = block_diagonal(V, sparse)
        return finish_v(mat, k, sparse, v0=False)
    if scipy.sparse.issparse(V):
        mat = scipy.sparse.csc_array(typing.cast(typing.Any, V))
        return finish_v(mat, k, sparse, v0=False)
    if isinstance(V, str):
        vec: FloatArray = column_values(V, data, 'V')
    elif isinstance(V, pandas.DataFrame):
        vec = numpy.asarray(V.to_numpy(dtype=numpy.float64))
    else:
        vec = column_values(V, data, 'V')
    v0 = (vec.ndim <= 1 and vec.size == 1 and bool(vec.reshape(-1)[0] == 0)) or (
        vec.ndim <= 1 and vec.size == k and bool(numpy.all(vec[~numpy.isnan(vec)] == 0))
    )
    if vec.ndim == 2 and (vec.shape[0] == 1 or vec.shape[1] == 1):
        vec = vec.reshape(-1)
    if vec.ndim <= 1:
        flat = vec.reshape(-1)
        if flat.size == 1:
            flat = numpy.full(k, flat[0])
        if flat.size != k:
            raise metafopy.errors.InputError(f"Length of 'yi' ({k}) and the length/dimensions of 'V' ({flat.size}) are not the same.")
        diag: Mat = scipy.sparse.diags_array(flat, format='csc') if sparse else numpy.diag(flat)
        return diag, v0
    return finish_v(vec, k, sparse, v0=False)


def finish_v(mat: Mat, k: int, sparse: bool, v0: bool) -> tuple[Mat, bool]:
    if mat.shape[0] != mat.shape[1]:
        raise metafopy.errors.InputError("'V' must be a square matrix.")
    if not is_symmetric(mat):
        raise metafopy.errors.InputError("'V' must be a symmetric matrix.")
    if mat.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of 'yi' ({k}) and the length/dimensions of 'V' ({mat.shape[0]}) are not the same.")
    if sparse and not scipy.sparse.issparse(mat):
        mat = scipy.sparse.csc_array(mat)
    if not sparse and scipy.sparse.issparse(mat):
        mat = numpy.asarray(mat.toarray())
    return mat, v0


def prepare_w(W: object, data: pandas.DataFrame | None, k: int, sparse: bool) -> Mat | None:
    """The weight matrix ``A`` (a vector is turned into a diagonal matrix).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if W is None:
        return None
    if scipy.sparse.issparse(W):
        mat: Mat = scipy.sparse.csc_array(typing.cast(typing.Any, W))
    else:
        arr = column_values(W.to_numpy(dtype=numpy.float64) if isinstance(W, pandas.DataFrame) else W, data, 'W')
        if arr.ndim <= 1 or arr.shape[0] == 1 or arr.shape[1] == 1:
            flat = arr.reshape(-1)
            if flat.size == 1:
                flat = numpy.full(k, flat[0])
            mat = numpy.diag(flat)
        else:
            mat = arr
    if mat.shape[0] != mat.shape[1]:
        raise metafopy.errors.InputError("'W' must be a square matrix.")
    if not is_symmetric(mat):
        raise metafopy.errors.InputError("'W' must be a symmetric matrix.")
    if mat.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of 'yi' ({k}) and length/dimensions of 'W' ({mat.shape[0]}) are not the same.")
    if sparse and not scipy.sparse.issparse(mat):
        mat = scipy.sparse.csc_array(mat)
    if not sparse and scipy.sparse.issparse(mat):
        mat = numpy.asarray(mat.toarray())
    return mat


def take_square(M: Mat, idx: numpy.typing.NDArray[numpy.intp]) -> Mat:
    if scipy.sparse.issparse(M):
        return scipy.sparse.csc_array(M.tocsr()[idx][:, idx])
    return numpy.asarray(M[numpy.ix_(idx, idx)])


def build_x(
    mods: typing.Any,
    intercept: bool,
    data: pandas.DataFrame | None,
    k: int,
) -> tuple[FloatArray, tuple[str, ...], numpy.typing.NDArray[numpy.bool_], bool]:
    """The model matrix before missing values are dropped, its column names, rows with NAs and the (possibly coerced) ``intercept``."""
    if mods is None:
        if not intercept:
            warnings.warn('Must either include an intercept and/or moderators in the model.\nCoerced an intercept into the model.', stacklevel=4)
        return numpy.ones((k, 1)), ('intrcpt',), numpy.zeros(k, dtype=numpy.bool_), True
    if isinstance(mods, str):
        X, names, missing, _ = metafopy.metadata.MetaData.design(mods, intercept, data, k)
        if X.shape[0] != k:
            raise metafopy.errors.InputError(f'Number of rows in the model matrix ({X.shape[0]}) does not match the length of the outcome vector ({k}).')
        return X, names, missing, False
    X, names, missing, _ = metafopy.metadata.MetaData.design(mods, intercept, data, k)
    return X, names, missing, intercept


def formula_variables(text: str) -> list[str]:
    """The variables of the inner part of an ``~ inner | outer`` term (``a + b`` -> ``['a', 'b']``)."""
    return [v for v in metafopy.models.mv_factor.split_top_level(text, '+') if v]


def split_nested(
    term: metafopy.formula.RandomTerm,
    data: pandas.DataFrame,
) -> tuple[list[str], list[metafopy.models.mv_factor.Factor]]:
    """Expand ``~ 1 | a/b/c`` into the interactions ``a``, ``a/b`` and ``a/b/c`` (``interaction(..., drop=TRUE, lex.order=TRUE, sep="/")``)."""
    base = [metafopy.models.mv_factor.to_factor(metafopy.models.mv_factor.evaluate_variable(v, data)) for v in term.outer]
    names: list[str] = []
    out: list[metafopy.models.mv_factor.Factor] = []
    for p in range(1, len(base) + 1):
        names.append('/'.join(term.outer[:p]))
        out.append(metafopy.models.mv_factor.interaction(base[:p], sep='/', lex_order=True))
    return names, out


def extract_random(
    random: str | collections.abc.Sequence[str] | None,
    struct: str | collections.abc.Sequence[str],
    data: pandas.DataFrame,
) -> RandomSpec:
    """Evaluate the variables of the ``random`` formulas.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    structs = [struct] * 2 if isinstance(struct, str) else list(struct)
    if len(structs) == 1:
        structs = structs * 2
    if len(structs) != 2 or any(s not in metafopy.models.mv_structs.STRUCTS for s in structs):
        raise metafopy.errors.InputError("Unknown 'struct' specified.")
    texts = [random] if isinstance(random, str) else list(random or [])
    terms = [metafopy.formula.parse_random(t) for t in texts]

    inner_outer = [t for t in terms if t.inner is not None]
    if len(inner_outer) > 2:
        raise metafopy.errors.InputError("Only up to two '~ inner | outer' formulas allowed in the 'random' argument.")

    s_names: list[str] = []
    s_factors: list[metafopy.models.mv_factor.Factor] = []
    io_parts: list[tuple[pandas.DataFrame, pandas.Series, metafopy.formula.RandomTerm]] = []
    struct_out = list(structs)
    io = 0
    for term in terms:
        if term.inner is None:
            names, facs = split_nested(term, data)
            s_names.extend(names)
            s_factors.extend(facs)
            continue
        outer_name = term.outer[0]
        outer = metafopy.models.mv_factor.evaluate_variable(outer_name, data).rename(outer_name)
        if structs[io] in metafopy.models.mv_structs.GENERAL:
            X_inner, names_inner, _, _ = metafopy.metadata.MetaData.design('~ ' + term.inner, True, data, len(data))
            inner = pandas.DataFrame(X_inner, columns=list(names_inner))
            if term.double_bar:
                struct_out[io] = 'GDIAG'
        else:
            variables = formula_variables(term.inner)
            inner = pandas.concat([metafopy.models.mv_factor.evaluate_variable(v, data).rename(v) for v in variables], axis=1)
        io_parts.append((inner.reset_index(drop=True), outer.reset_index(drop=True), term))
        io += 1

    g = io_parts[0] if io_parts else None
    h = io_parts[1] if len(io_parts) > 1 else None
    return RandomSpec(
        s_names=s_names,
        s_factors=s_factors,
        g_inner=None if g is None else g[0],
        g_outer=None if g is None else g[1],
        g_formula=None if g is None else g[2],
        h_inner=None if h is None else h[0],
        h_outer=None if h is None else h[1],
        h_formula=None if h is None else h[2],
        struct=typing.cast(tuple[str, str], tuple(struct_out)),
        raw=texts,
    )
