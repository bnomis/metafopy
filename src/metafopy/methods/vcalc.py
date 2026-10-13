"""Construction of the variance-covariance matrix of dependent effect sizes (metafor's ``vcalc``).

R source: https://github.com/wviechtb/metafor/blob/master/R/vcalc.r. Documentation:
https://wviechtb.github.io/metafor/reference/vcalc.html. The correlation structure for
estimates of different outcome types and for estimates that share a group follows
Pustejovsky and Tipton (2022), https://doi.org/10.1007/s11121-021-01246-3 and the
discussion at https://www.metafor-project.org/doku.php/analyses:berkey1998. The nearest
positive definite matrix is that of Higham (2002), https://doi.org/10.1093/imanum/22.3.329.
"""

import collections.abc
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.core.linalg
import metafopy.errors

__all__ = ['vcalc']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Variable = numpy.typing.ArrayLike | str | None
type Rho = float | numpy.typing.ArrayLike | pandas.DataFrame | collections.abc.Sequence[typing.Any] | None
type RVars = collections.abc.Sequence[str | int] | slice | None


def variable(value: Variable, data: pandas.DataFrame | None, name: str) -> numpy.ndarray | None:
    """A variable given as an array or as the name of a column in ``data`` (metafor's ``.getx``).

    Raises:
        metafopy.errors.InputError: a column is named without ``data``, or the variable is not one-dimensional.
        metafopy.errors.NotFoundError: the column is not in ``data``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if data is None:
            raise metafopy.errors.InputError(f'{name}={value!r} names a column, but no data was given')
        if value not in data.columns:
            raise metafopy.errors.NotFoundError(f"Cannot find the object/variable ('{value}') specified for the '{name}' argument.")
        return numpy.asarray(data[value].to_numpy())
    arr = numpy.asarray(value)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise metafopy.errors.InputError(f"The object/variable specified for the '{name}' argument must be one-dimensional, got shape {arr.shape}")
    return arr


def numeric_variable(value: Variable, data: pandas.DataFrame | None, name: str) -> FloatArray | None:
    """Like :func:`variable`, but the variable must be numeric (all-missing is accepted, as in ``.getx``).

    Raises:
        metafopy.errors.InputError: the variable is not numeric.
    """
    arr = variable(value, data, name)
    if arr is None:
        return None
    if arr.dtype.kind not in 'iuf' and not bool(numpy.all(pandas.isna(arr))):
        raise metafopy.errors.InputError(f"The object/variable specified for the '{name}' argument is not numeric.")
    return arr.astype(numpy.float64)


def has_na(values: numpy.ndarray) -> bool:
    return bool(numpy.any(pandas.isna(values)))


def as_character(values: numpy.ndarray) -> numpy.ndarray:
    """R's ``as.character()`` for the kinds of variables used as identifiers (``1.0`` becomes ``'1'``)."""
    out: list[str] = []
    for v in values.tolist():
        if isinstance(v, bool):
            out.append('TRUE' if v else 'FALSE')
        elif isinstance(v, float):
            out.append(str(int(v)) if v.is_integer() and abs(v) < 1e15 else f'{v:.15g}')
        else:
            out.append(str(v))
    return numpy.array(out, dtype=object)


class NamedMatrix(typing.NamedTuple):
    values: FloatArray
    rownames: list[str]
    colnames: list[str]


def classify_rho(rho: Rho) -> tuple[str, typing.Any]:
    """Tell apart the three kinds of ``rho`` that R distinguishes: a vector, a matrix and a list.

    Raises:
        metafopy.errors.ArgumentError: ``rho`` is of another type.
    """
    if isinstance(rho, pandas.DataFrame):
        return 'matrix', rho
    if isinstance(rho, list | tuple):
        if all(isinstance(v, int | float | numpy.number) and not isinstance(v, bool) for v in rho):
            return 'vector', [float(v) for v in rho]
        return 'list', list(rho)
    if isinstance(rho, int | float | numpy.number):
        return 'vector', [float(rho)]
    if isinstance(rho, numpy.ndarray):
        if rho.ndim == 2:
            return 'matrix', rho
        return 'vector', [float(v) for v in rho.ravel()]
    raise metafopy.errors.ArgumentError("Argument 'rho' must either be a vector, a matrix, or a list.")


def index_names(index: typing.Any) -> list[str] | None:
    """The labels of a data frame axis as strings; a default ``RangeIndex`` counts as having none, like an R matrix without dimnames."""
    if isinstance(index, pandas.RangeIndex):
        return None
    return [str(v) for v in as_character(numpy.asarray(index.to_numpy()))]


def rho_element(element: typing.Any, labels: numpy.ndarray) -> NamedMatrix | FloatArray:
    """One element of ``rho``: a scalar becomes a matrix with 1s on the diagonal and the unique ``labels`` as names.

    Raises:
        metafopy.errors.InputError: a matrix has no dimension names.
    """
    if isinstance(element, pandas.DataFrame):
        values = element.to_numpy(dtype=numpy.float64)
    else:
        values = numpy.asarray(element, dtype=numpy.float64)
    if values.size == 1:
        n = len({str(v) for v in labels})
        mat = numpy.full((n, n), float(values.ravel()[0]))
        numpy.fill_diagonal(mat, 1.0)
        names = list(dict.fromkeys(str(v) for v in labels))
        return NamedMatrix(mat, names, list(names))
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return values
    rownames: list[str] | None = None
    colnames: list[str] | None = None
    if isinstance(element, pandas.DataFrame):
        rownames = index_names(element.index)
        colnames = index_names(element.columns)
    if rownames is None and colnames is None:
        raise metafopy.errors.InputError("Any matrices specified via 'rho' must have dimension names.")
    return NamedMatrix(
        values, rownames if rownames is not None else typing.cast(list[str], colnames), colnames if colnames is not None else typing.cast(list[str], rownames)
    )


def process_rho(rho: Rho, type_spec: bool, obs_spec: bool, obs: numpy.ndarray, type_: numpy.ndarray) -> tuple[NamedMatrix, NamedMatrix]:
    """The two correlation matrices (between ``obs`` within a type and between types), as in the ``rho`` handling of R's ``vcalc``.

    Raises:
        metafopy.errors.InputError: ``rho`` does not match which of ``type`` and ``obs`` are specified, is not square, or is out of range.
    """
    parts: list[typing.Any]
    if rho is not None:
        kind, parsed = classify_rho(rho)
    else:
        kind, parsed = 'vector', []
    if type_spec and obs_spec:
        if kind == 'vector':
            if len(parsed) != 2:
                raise metafopy.errors.InputError(
                    "When 'type' and 'obs' are both specified, 'rho' must specify both the within- and between-construct correlations."
                )
            parts = list(parsed)
        elif kind == 'matrix':
            raise metafopy.errors.InputError(
                "When 'type' and 'obs' are both specified, 'rho' must specify both the within- and between-construct correlations."
            )
        else:
            if len(parsed) != 2:
                raise metafopy.errors.InputError("When 'type' and 'obs' are both specified and 'rho' is a list, then it must have two elements.")
            parts = parsed
    elif type_spec:
        if kind == 'vector':
            if len(parsed) != 1:
                raise metafopy.errors.InputError("When only 'type' is specified, 'rho' must be a scalar.")
            parts = [0.0, parsed[0]]
        elif kind == 'matrix':
            parts = [0.0, parsed]
        else:
            if len(parsed) != 1:
                raise metafopy.errors.InputError("When only 'type' is specified, 'rho' must have a single list element.")
            parts = [0.0, parsed[0]]
    elif obs_spec:
        if kind == 'vector':
            if len(parsed) != 1:
                raise metafopy.errors.InputError("When only 'obs' is specified, 'rho' must be a scalar.")
            parts = [parsed[0], 0.0]
        elif kind == 'matrix':
            parts = [parsed, 0.0]
        else:
            if len(parsed) != 1:
                raise metafopy.errors.InputError("When only 'obs' is specified, 'rho' must have a single list element.")
            parts = [parsed[0], 0.0]
    else:
        parts = [0.0, 0.0]
    first = rho_element(parts[0], obs)
    second = rho_element(parts[1], type_)
    if not isinstance(first, NamedMatrix) or not isinstance(second, NamedMatrix):
        raise metafopy.errors.InputError("All matrices specified via 'rho' argument must be square matrices.")
    if bool(numpy.any(numpy.abs(first.values) > 1)) or bool(numpy.any(numpy.abs(second.values) > 1)):
        raise metafopy.errors.InputError("All correlations specified via 'rho' must be in [-1,1].")
    return first, second


def lookup(matrix: NamedMatrix, labels: numpy.ndarray, what: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Row and column positions in a named ``rho`` matrix for each label; the first match wins, as R's name indexing does.

    Raises:
        metafopy.errors.InputError: a label has no row or column in the matrix.
    """
    rows = {name: i for i, name in reversed(list(enumerate(matrix.rownames)))}
    cols = {name: i for i, name in reversed(list(enumerate(matrix.colnames)))}
    if not all(label in rows and label in cols for label in labels):
        raise metafopy.errors.InputError(f"There are '{what}' values with no corresponding row/column in the correlation matrix.")
    return numpy.array([rows[v] for v in labels], dtype=numpy.intp), numpy.array([cols[v] for v in labels], dtype=numpy.intp)


def same_or_decay(g_a: numpy.ndarray, t_a: FloatArray, g_b: numpy.ndarray, t_b: FloatArray, phi: float) -> FloatArray:
    """1 for the same group and time, ``phi^|dt|`` for the same group at different times, 0 for different groups."""
    dt = numpy.abs(t_a[:, None] - t_b[None, :])
    decay = numpy.where(dt == 0, 1.0, phi**dt)
    return numpy.where(g_a[:, None] == g_b[None, :], decay, 0.0)


def block_corr(
    idx: numpy.ndarray,
    obs: numpy.ndarray,
    type_: numpy.ndarray,
    rho1: tuple[NamedMatrix, numpy.ndarray, numpy.ndarray],
    rho2: tuple[NamedMatrix, numpy.ndarray, numpy.ndarray],
    grp1: numpy.ndarray,
    grp2: numpy.ndarray,
    time1: FloatArray,
    time2: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    phi: float,
) -> FloatArray:
    """The R matrix of one cluster (the nested ``ifelse()`` expression of R's ``vcalc``, evaluated for all pairs at once)."""
    o, t = obs[idx], type_[idx]
    g1, g2, t1, t2, a, b = grp1[idx], grp2[idx], time1[idx], time2[idx], w1[idx], w2[idx]
    with numpy.errstate(all='ignore'):
        inv_a, inv_b = 1 / a, 1 / b
        numerator = (
            same_or_decay(g1, t1, g1, t1, phi) * numpy.sqrt(inv_a[:, None] / a[None, :])
            - same_or_decay(g1, t1, g2, t2, phi) * numpy.sqrt(inv_a[:, None] / b[None, :])
            - same_or_decay(g2, t2, g1, t1, phi) * numpy.sqrt(inv_b[:, None] / a[None, :])
            + same_or_decay(g2, t2, g2, t2, phi) * numpy.sqrt(inv_b[:, None] / b[None, :])
        )
        same_self = numpy.where(g1 == g2, numpy.where(t1 == t2, 1.0, phi ** numpy.abs(t1 - t2)), 0.0)
        spread = numpy.sqrt(inv_a + inv_b - 2 * same_self * numpy.sqrt(inv_a / b))
        denominator = spread[:, None] * spread[None, :]
        within = rho1[0].values[rho1[1][idx][:, None], rho1[2][idx][None, :]]
        between = rho2[0].values[rho2[1][idx][:, None], rho2[2][idx][None, :]]
        r = numpy.where(t[:, None] == t[None, :], numpy.where(o[:, None] == o[None, :], 1.0, within), between)
        m = r * numerator / denominator
    block: FloatArray = numpy.tril(m) + numpy.tril(m, -1).T
    block[0, 0] = 1.0
    return block


def select_rvars(rvars: RVars, data: pandas.DataFrame) -> pandas.DataFrame:
    """The columns of ``data`` named (or positioned, 0-based) by ``rvars``; a ``slice`` of names runs from the first to the last, inclusive.

    Raises:
        metafopy.errors.InputError: a column cannot be found.
    """
    names = list(data.columns)
    try:
        if isinstance(rvars, slice):
            start = names.index(rvars.start) if isinstance(rvars.start, str) else int(rvars.start)
            stop = names.index(rvars.stop) if isinstance(rvars.stop, str) else int(rvars.stop)
            return data.iloc[:, start : stop + 1]
        cols = [names.index(v) if isinstance(v, str) else int(v) for v in typing.cast(collections.abc.Sequence[str | int], rvars)]
        return data.iloc[:, cols]
    except (ValueError, IndexError, TypeError) as exc:
        raise metafopy.errors.InputError("Could not find all variables specified via 'rvars' in 'data'.") from exc


def vcalc(
    vi: Variable = None,
    cluster: Variable = None,
    subgroup: Variable = None,
    obs: Variable = None,
    type: Variable = None,
    time1: Variable = None,
    time2: Variable = None,
    grp1: Variable = None,
    grp2: Variable = None,
    w1: Variable = None,
    w2: Variable = None,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    rho: Rho = None,
    phi: float | None = None,
    rvars: RVars = None,
    checkpd: bool = True,
    nearpd: bool = False,
    sparse: bool = False,
    retdat: bool = False,
) -> FloatArray | scipy.sparse.csr_array | pandas.DataFrame:
    """Variance-covariance matrix of dependent estimates, block diagonal over ``cluster`` (metafor's ``vcalc``).

    Every variable argument is an array or the name of a column of ``data``. Within a cluster the
    correlation of two estimates is built from ``rho`` (a scalar, or a pair of scalars or of
    matrices as pandas data frames with row and column labels: the first for the ``obs`` values
    within a ``type``, the second for the ``type`` values), from ``phi`` and the time points
    ``time1``/``time2`` (an autoregressive structure), and from the groups ``grp1``/``grp2`` with
    sizes ``w1``/``w2`` for estimates that are contrasts between two groups. With ``rvars`` (names or
    0-based positions of columns of ``data``, or a ``slice`` of names for ``a:b``) the
    variance-covariance matrices of the clusters are read from those columns instead, and ``vi`` is
    typically 1.

    The result is a dense ``numpy.ndarray``; ``sparse=True`` gives a ``scipy.sparse.csr_array`` and
    ``retdat=True`` a data frame of the variables used with the matrix in columns ``V.1``, ``V.2``, ....
    ``checkpd`` warns about clusters whose correlation matrix is not positive definite and ``nearpd``
    replaces such a matrix by the nearest correlation matrix (``Matrix::nearPD(corr=TRUE)``).

    Differences from R: metafor's undocumented ``nearPD`` alias of ``nearpd`` is not accepted; in
    ``rvars`` mode the matrix of a cluster is placed at that cluster's rows even when the rows of a
    cluster are not contiguous (R places the blocks in order of first appearance, which misaligns
    the matrix in that case).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid (R stops with an error here).
        metafopy.errors.NotFoundError: a named column is not in ``data``.
        metafopy.errors.ArgumentError: ``rho`` has an unsupported type.
    """
    if vi is None:
        raise metafopy.errors.InputError("Must specify the 'vi' variable.")
    if cluster is None:
        raise metafopy.errors.InputError("Must specify the 'cluster' variable.")
    frame = None if data is None else pandas.DataFrame(data)
    if frame is None and rvars is not None:
        raise metafopy.errors.InputError("Must specify the 'data' argument when using 'rvars'.")

    subgroup_spec, type_spec, obs_spec = subgroup is not None, type is not None, obs is not None
    grp1_spec, grp2_spec, time1_spec, time2_spec = grp1 is not None, grp2 is not None, time1 is not None, time2 is not None
    w1_spec, w2_spec = w1 is not None, w2 is not None

    vi_ = numeric_variable(vi, frame, 'vi')
    cluster_ = variable(cluster, frame, 'cluster')
    subgroup_ = variable(subgroup, frame, 'subgroup')
    type_ = variable(type, frame, 'type')
    obs_ = variable(obs, frame, 'obs')
    grp1_ = variable(grp1, frame, 'grp1')
    grp2_ = variable(grp2, frame, 'grp2')
    time1_ = numeric_variable(time1, frame, 'time1')
    time2_ = numeric_variable(time2, frame, 'time2')
    w1_ = numeric_variable(w1, frame, 'w1')
    w2_ = numeric_variable(w2, frame, 'w2')
    if vi_ is None or cluster_ is None:
        raise metafopy.errors.InternalError('vi and cluster were checked above')

    if len(vi_) == 1 and len(cluster_) > 1:
        vi_ = numpy.repeat(vi_, len(cluster_))
    k = len(vi_)
    if k == 1:
        raise metafopy.errors.InputError('Processing terminated since k = 1.')

    def check_length(values: numpy.ndarray, name: str) -> None:
        if len(values) != k:
            raise metafopy.errors.InputError(f"Length of the variable specified via '{name}' ({len(values)}) does not match the length of 'vi' ({k}).")

    def check_na(values: numpy.ndarray, name: str) -> None:
        if has_na(values):
            raise metafopy.errors.InputError(f"No missing values allowed in '{name}' variable.")

    check_na(cluster_, 'cluster')
    check_length(cluster_, 'cluster')
    cluster_chr = as_character(cluster_)
    if subgroup_spec:
        if subgroup_ is None:
            raise metafopy.errors.InternalError('subgroup was checked above')
        check_na(subgroup_, 'subgroup')
        check_length(subgroup_, 'subgroup')
        cluster_chr = numpy.array([f'{c}.{s}' for c, s in zip(cluster_chr, as_character(subgroup_), strict=True)], dtype=object)
        cluster_ = cluster_chr
    codes, uniques = pandas.factorize(cluster_chr)
    n = len(uniques)
    groups = [numpy.flatnonzero(codes == i) for i in range(n)]

    blocks: list[tuple[numpy.ndarray, FloatArray]] = []
    type_out: numpy.ndarray | None = None
    obs_out: numpy.ndarray | None = None
    grp1_out: numpy.ndarray | None = None
    grp2_out: numpy.ndarray | None = None
    time1_out: FloatArray | None = None
    time2_out: FloatArray | None = None
    w1_out: FloatArray | None = None
    w2_out: FloatArray | None = None

    if rvars is None:
        if type_spec and rho is None:
            raise metafopy.errors.InputError("Must specify 'rho' when 'type' is specified.")
        if type_ is None:
            type_ = numpy.ones(k)
        check_na(type_, 'type')
        check_length(type_, 'type')
        if obs_spec and rho is None:
            raise metafopy.errors.InputError("Must specify 'rho' when 'obs' is specified.")
        if obs_ is None:
            obs_ = numpy.ones(k)
        check_na(obs_, 'obs')
        check_length(obs_, 'obs')

        if grp2_spec and not grp1_spec:
            raise metafopy.errors.InputError("Either specify only 'grp1', both 'grp1' and 'grp2', or neither.")
        if grp1_ is None:
            grp1_ = numpy.ones(k)
        if grp2_ is None:
            grp2_ = numpy.full(k, 2.0)
        check_na(grp1_, 'grp1')
        check_na(grp2_, 'grp2')
        check_length(grp1_, 'grp1')
        check_length(grp2_, 'grp2')

        if time2_spec and not time1_spec:
            raise metafopy.errors.InputError("Either specify only 'time1', both 'time1' and 'time2', or neither.")
        if time2_spec and not grp2_spec:
            raise metafopy.errors.InputError("Must specify 'grp2' when 'time2' is specified.")
        if time1_ is None:
            time1_ = numpy.ones(k)
        if time2_ is None:
            time2_ = time1_
        if time1_spec or time2_spec:
            if phi is None:
                raise metafopy.errors.InputError("Must specify 'phi' when 'time1' and/or 'time2' is specified.")
        else:
            phi = 1.0
        if abs(phi) > 1:
            raise metafopy.errors.InputError("Value of argument 'phi' must be in [-1,1].")
        check_na(time1_, 'time1')
        check_na(time2_, 'time2')
        check_length(time1_, 'time1')
        check_length(time2_, 'time2')

        if w2_spec and not w1_spec:
            raise metafopy.errors.InputError("Either specify only 'w1', both 'w1' and 'w2', or neither.")
        if w2_spec and not grp2_spec:
            raise metafopy.errors.InputError("Must specify 'grp2' when 'w2' is specified.")
        if w1_ is None:
            w1_ = numpy.ones(k)
        if w2_ is None:
            w2_ = w1_
        check_na(w1_, 'w1')
        check_na(w2_, 'w2')
        check_length(w1_, 'w1')
        check_length(w2_, 'w2')

        obs_chr, type_chr = as_character(obs_), as_character(type_)
        rho1, rho2 = process_rho(rho, type_spec, obs_spec, obs_chr, type_chr)
        obs_rows, obs_cols = lookup(rho1, obs_chr, 'obs')
        type_rows, type_cols = lookup(rho2, type_chr, 'type')
        both_groups = pandas.factorize(numpy.concatenate([as_character(grp1_), as_character(grp2_)]))[0]
        g1, g2 = both_groups[:k], both_groups[k:]
        obs_id, type_id = pandas.factorize(obs_chr)[0], pandas.factorize(type_chr)[0]
        for idx in groups:
            block = block_corr(idx, obs_id, type_id, (rho1, obs_rows, obs_cols), (rho2, type_rows, type_cols), g1, g2, time1_, time2_, w1_, w2_, phi)
            blocks.append((idx, block))
        type_out, obs_out, grp1_out, grp2_out = type_chr, obs_chr, grp1_, grp2_
        time1_out, time2_out, w1_out, w2_out = time1_, time2_, w1_, w2_
    else:
        if frame is None:
            raise metafopy.errors.InternalError('frame was checked above')
        given = [type_spec, obs_spec, grp1_spec, grp2_spec, time1_spec, time2_spec, w1_spec, w2_spec, rho is not None, phi is not None]
        if any(given):
            names = ['type', 'obs', 'grp1', 'grp2', 'time1', 'time2', 'w1', 'w2', 'rho', 'phi']
            ignored = ','.join(name for name, flag in zip(names, given, strict=True) if flag)
            warnings.warn(f"Argument{'s' if sum(given) > 1 else ''} '{ignored}' ignored for when 'rvars' is specified.", stacklevel=2)
        rv = select_rvars(rvars, frame)
        largest = max(len(g) for g in groups)
        count = 'is 1 variable ' if rv.shape[1] == 1 else f'are {rv.shape[1]} variables '
        if largest > rv.shape[1]:
            raise metafopy.errors.InputError(f"There {count}specified via 'rvars', but there are clusters with more rows.")
        if largest != rv.shape[1]:
            raise metafopy.errors.InputError(f"There {count}specified via 'rvars', but no cluster with this many rows.")
        for label, idx in zip(uniques, groups, strict=True):
            m = rv.iloc[idx, : len(idx)].to_numpy(dtype=numpy.float64)
            if bool(numpy.any(numpy.isnan(m[numpy.tril_indices_from(m)]))):
                warnings.warn(f"There are missing values in 'rvals' for cluster {label}.", stacklevel=2)
            blocks.append((idx, numpy.tril(m) + numpy.tril(m, -1).T))

    if checkpd or nearpd:
        for position, (idx, block) in enumerate(blocks):
            if has_na(block) or metafopy.core.linalg.is_positive_definite(block):
                continue
            label = uniques[position]
            if not nearpd:
                warnings.warn(f'The var-cov matrix appears to be not positive definite in cluster {label}.', stacklevel=2)
                continue
            try:
                block = metafopy.core.linalg.near_pd(block, corr=True).mat
            except metafopy.errors.EstimationError:
                warnings.warn(f'Using nearPD() failed in cluster {label}.', stacklevel=2)
                continue
            if not has_na(block) and not metafopy.core.linalg.is_positive_definite(block):
                warnings.warn(f'The var-cov matrix still appears to be not positive definite in cluster {label} even after nearPD().', stacklevel=2)
            blocks[position] = (idx, block)

    with numpy.errstate(invalid='ignore'):
        s = numpy.sqrt(vi_)
    if bool(numpy.any(vi_ < 0)):
        warnings.warn('NaNs produced', stacklevel=2)
    scaled = [(idx, block * numpy.outer(s[idx], s[idx])) for idx, block in blocks]
    result: FloatArray | scipy.sparse.csr_array
    if sparse:
        rows = numpy.concatenate([numpy.repeat(idx, len(idx)) for idx, _ in scaled])
        cols = numpy.concatenate([numpy.tile(idx, len(idx)) for idx, _ in scaled])
        vals = numpy.concatenate([block.ravel() for _, block in scaled])
        result = scipy.sparse.csr_array((vals, (rows, cols)), shape=(k, k))
        result.eliminate_zeros()
    else:
        result = numpy.zeros((k, k))
        for idx, block in scaled:
            result[numpy.ix_(idx, idx)] = block
        # R multiplies dense matrices, so 0 * NA or 0 * Inf makes the whole row and column of such an estimate NA
        for i in numpy.flatnonzero(~numpy.isfinite(s)):
            other = codes != codes[i]
            result[i, other] = numpy.nan
            result[other, i] = numpy.nan
    if not retdat:
        return result
    dense = result.toarray() if isinstance(result, scipy.sparse.csr_array) else result
    columns: dict[str, typing.Any] = {'cluster': cluster_}
    for name, column in (
        ('type', type_out),
        ('obs', obs_out),
        ('grp1', grp1_out),
        ('grp2', grp2_out),
        ('time1', time1_out),
        ('time2', time2_out),
        ('w1', w1_out),
        ('w2', w2_out),
    ):
        if column is not None:
            columns[name] = column
    columns['vi'] = vi_
    for j in range(k):
        columns[f'V.{j + 1}'] = dense[:, j]
    return pandas.DataFrame(columns)
