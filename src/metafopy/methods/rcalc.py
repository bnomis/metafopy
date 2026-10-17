"""Variance-covariance matrix of dependent correlation coefficients (metafor's ``rcalc``).

R source: https://github.com/wviechtb/metafor/blob/master/R/rcalc.r. Documentation:
https://wviechtb.github.io/metafor/reference/rcalc.html. The sampling covariances of
correlations from one sample are those of Olkin and Finn (1995), equation 5,
https://doi.org/10.1037/0033-2909.118.1.155 (equivalent to Steiger, 1980, equation 2,
https://doi.org/10.1037/0033-2909.87.2.245), and for Fisher r-to-z transformed correlations
Steiger (1980), equation 11.
"""

import collections.abc
import dataclasses
import re
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.errors
import metafopy.methods.blocks
import metafopy.methods.common
import metafopy.methods.vcalc

__all__ = ['RCalcResult', 'rcalc']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Matrix = FloatArray | pandas.DataFrame
type Nfun = str | collections.abc.Callable[[FloatArray], float]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful (it is shown in the error message)
NFUNS = ('min', 'harmonic', 'mean')


@dataclasses.dataclass(frozen=True)
class RCalcResult:
    """The pair of ``dat`` (one row per correlation) and ``V`` (their variance-covariance matrix) that ``rcalc`` returns.

    Both are ``None`` for a single matrix of dimension 1 x 1 or smaller. ``V`` is dense, except that
    ``sparse=True`` with several matrices gives a ``scipy.sparse.csr_array``.
    """

    dat: pandas.DataFrame | None
    V: FloatArray | scipy.sparse.csr_array | None


def match_nfun(nfun: str) -> str:
    """R's ``match.arg()``: an unambiguous prefix of one of the choices.

    Raises:
        metafopy.errors.InputError: ``nfun`` is not (a prefix of) one of the choices.
    """
    hits = [choice for choice in NFUNS if choice.startswith(nfun)] if nfun else []
    if nfun in NFUNS:
        return nfun
    if len(hits) != 1:
        raise metafopy.errors.InputError(f"'arg' should be one of {', '.join(f'"{c}"' for c in NFUNS)}")
    return hits[0]


def matrix_names(x: Matrix, n: int) -> tuple[FloatArray, list[str]]:
    """The values of a correlation matrix and its dimension names (R: column names, else row names, else ``x1``, ``x2``, ...)."""
    if isinstance(x, pandas.DataFrame):
        values = x.to_numpy(dtype=numpy.float64)
        names = metafopy.methods.vcalc.index_names(x.columns)
        if names is None:
            names = metafopy.methods.vcalc.index_names(x.index)
        return values, names if names is not None else [f'x{i + 1}' for i in range(n)]
    return numpy.array(x, dtype=numpy.float64), [f'x{i + 1}' for i in range(n)]


def rcalc_matrix(x: Matrix, ni: float, upper: bool, rtoz: bool) -> RCalcResult:
    """The matrix case of ``rcalc``: Olkin and Finn's (1995) covariances of the lower (or upper) triangle of ``x``.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a matrix.
        metafopy.errors.InputError: ``x`` is not square, or a correlation or the sample size is out of range.
    """
    if not isinstance(x, numpy.ndarray | pandas.DataFrame) or numpy.ndim(x) != 2:
        raise metafopy.errors.ArgumentError("Argument 'x' must be either a formula, a matrix, or a list of matrices.")
    if x.shape[0] != x.shape[1]:
        raise metafopy.errors.InputError("Argument 'x' must be a square matrix.")
    n = x.shape[0]
    values, dnames = matrix_names(x, n)
    if n <= 1:
        return RCalcResult(None, None)
    if upper:
        values = numpy.triu(values) + numpy.triu(values, 1).T
        pairs = [(i, j) for j in range(n) for i in range(j)]
    else:
        values = numpy.tril(values) + numpy.tril(values, -1).T
        pairs = [(i, j) for j in range(n) for i in range(j + 1, n)]
    pair_i = numpy.array([p[0] for p in pairs], dtype=numpy.intp)
    pair_j = numpy.array([p[1] for p in pairs], dtype=numpy.intp)
    ri = values[pair_i, pair_j]
    if bool(numpy.any(numpy.abs(ri[~numpy.isnan(ri)]) > 1)):
        raise metafopy.errors.InputError('One or more correlations are > 1 or < -1.')
    if ni <= 0:
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    r = numpy.arctanh(ri) if rtoz else ri
    i, j, k, ll = pair_i[:, None], pair_j[:, None], pair_i[None, :], pair_j[None, :]
    with numpy.errstate(all='ignore'):
        v = (
            0.5 * values[i, j] * values[k, ll] * (values[i, k] ** 2 + values[i, ll] ** 2 + values[j, k] ** 2 + values[j, ll] ** 2)
            + values[i, k] * values[j, ll]
            + values[i, ll] * values[j, k]
            - (
                values[i, j] * values[i, k] * values[i, ll]
                + values[j, i] * values[j, k] * values[j, ll]
                + values[k, i] * values[k, j] * values[k, ll]
                + values[ll, i] * values[ll, j] * values[ll, k]
            )
        )
        if rtoz:
            v = v / ((1 - values[i, j] ** 2) * (1 - values[k, ll] ** 2))
        if ni >= 5:
            v = v / (ni - 3 if rtoz else ni - 1)
        else:
            v = numpy.full_like(v, numpy.nan)
    var1 = numpy.array([sorted((dnames[a], dnames[b]))[0] for a, b in pairs], dtype=object)
    var2 = numpy.array([sorted((dnames[a], dnames[b]))[1] for a, b in pairs], dtype=object)
    dat = pandas.DataFrame(
        {
            'var1': var1,
            'var2': var2,
            'var1.var2': [f'{a}.{b}' for a, b in zip(var1, var2, strict=True)],
            'yi': r,
            'ni': float(ni),
        }
    )
    return RCalcResult(dat, v)


def rcalc_list(
    x: collections.abc.Mapping[str, Matrix],
    ni: FloatArray,
    rtoz: bool,
    sparse: bool,
    upper: bool,
    simplify: bool,
    rowid: collections.abc.Sequence[str] | None = None,
    vnames: collections.abc.Sequence[str] | None = None,
    noid: bool = False,
) -> RCalcResult | dict[str, RCalcResult]:
    """The list case of ``rcalc``: one result per matrix, stacked into a single ``dat`` and block diagonal ``V`` when ``simplify``.

    Raises:
        metafopy.errors.InputError: ``ni`` does not have one value per matrix.
    """
    if len(x) != len(ni):
        raise metafopy.errors.InputError("Argument 'ni' must be of the same length as there are elements in 'x'.")
    res = [rcalc_matrix(m, float(ni[i]), upper, rtoz) for i, m in enumerate(x.values())]
    if not simplify:
        return dict(zip(x.keys(), res, strict=True))
    kept = [(name, r) for name, r in zip(x.keys(), res, strict=True) if r.dat is not None and r.V is not None]
    if not kept:
        return RCalcResult(None, None)
    frames = [typing.cast(pandas.DataFrame, r.dat) for _, r in kept]
    mats = [typing.cast(FloatArray, r.V) for _, r in kept]
    dat = pandas.concat(frames, ignore_index=True)
    dat.insert(0, 'id', numpy.repeat(numpy.array([name for name, _ in kept], dtype=object), [len(f) for f in frames]))
    v: FloatArray | scipy.sparse.csr_array
    if sparse:
        v = scipy.sparse.csr_array(scipy.sparse.block_diag([scipy.sparse.csr_array(m) for m in mats], format='csr'))
    else:
        v = metafopy.methods.blocks.bldiag(mats)
    if rowid is not None:
        keys = [f'{a}.{b}.{a}.{c}' for a, b, c in zip(dat['id'], dat['var1'], dat['var2'], strict=True)]
        lookup: dict[str, int] = {}
        for position, key in enumerate(keys):
            lookup.setdefault(key, position)
        pos = numpy.array([lookup[r] for r in rowid], dtype=numpy.intp)
        dat = dat.iloc[pos]
        v = v[pos][:, pos] if isinstance(v, scipy.sparse.csr_array) else v[numpy.ix_(pos, pos)]
    if vnames is not None:
        columns = list(dat.columns)
        columns[0], columns[1], columns[2] = vnames[3], vnames[1], vnames[2]
        columns[3] = f'{vnames[1]}.{vnames[2]}'
        dat.columns = pandas.Index(columns)
    if noid:
        dat = dat.iloc[:, 1:]
    dat = dat.reset_index(drop=True)
    return RCalcResult(dat, v)


def formula_variables(formula: str) -> list[str]:
    """The variables of a formula in order of appearance, without duplicates (R: ``all.vars()``).

    Raises:
        metafopy.errors.InputError: ``formula`` has no ``~``.
    """
    if '~' not in formula:
        raise metafopy.errors.InputError(f'{formula!r} is not a formula')
    names = re.findall(r'[A-Za-z.][A-Za-z0-9_.]*(?![A-Za-z0-9_.]*\()', formula)
    return list(dict.fromkeys(names))


def rcalc_formula(
    x: str,
    ni: metafopy.methods.vcalc.Variable,
    data: pandas.DataFrame,
    rtoz: bool,
    nfun: Nfun,
    sparse: bool,
    upper: bool,
    simplify: bool,
) -> RCalcResult | dict[str, RCalcResult]:
    """The formula case of ``rcalc``: build a correlation matrix per study from rows of ``var1``, ``var2`` and ``ri``.

    Raises:
        metafopy.errors.InputError: the formula or the data of a study is invalid.
        metafopy.errors.NotFoundError: a variable of the formula is not in ``data``.
    """
    ni_ = metafopy.methods.vcalc.numeric_variable(ni, data, 'ni')
    if ni_ is None:
        raise metafopy.errors.InputError("Argument 'ni' must be specified.")
    names = formula_variables(x)
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise metafopy.errors.NotFoundError(f"object '{missing[0]}' not found in 'data'")
    noid = len(names) == 3
    if noid:
        dat = data[names].copy()
        dat['V4'] = 1
        names = [*names, 'V4']
    else:
        dat = data[names].copy()
    vnames = list(dat.columns)
    if dat.shape[1] != 4:
        raise metafopy.errors.InputError(f'Formula should contain 4 variables, but contains {dat.shape[1]} variables.')
    if bool(numpy.any(pandas.isna(dat.iloc[:, 1]))) or bool(numpy.any(pandas.isna(dat.iloc[:, 2]))):
        raise metafopy.errors.InputError('No missing values allowed in variable identifiers.')
    ids = dat.iloc[:, 3].to_numpy()
    if len(ni_) != len(data):
        raise metafopy.errors.InputError("Argument 'ni' must be of the same length as the data frame specified via 'data'.")
    if bool(numpy.any(pandas.isna(ids))):
        raise metafopy.errors.InputError('No missing values allowed in study identifier.')
    ri_all = metafopy.methods.vcalc.numeric_variable(dat.iloc[:, 0].to_numpy(), None, 'ri')
    if ri_all is None:
        raise metafopy.errors.InternalError('ri was checked above')
    id_chr = metafopy.methods.vcalc.as_character(ids)
    v1_chr = metafopy.methods.vcalc.as_character(dat.iloc[:, 1].to_numpy())
    v2_chr = metafopy.methods.vcalc.as_character(dat.iloc[:, 2].to_numpy())
    swap = [f'{i}.{a}' > f'{i}.{b}' for i, a, b in zip(id_chr, v1_chr, v2_chr, strict=True)]
    rowid = [f'{i}.{b}.{i}.{a}' if s else f'{i}.{a}.{i}.{b}' for i, a, b, s in zip(id_chr, v1_chr, v2_chr, swap, strict=True)]
    levels = pandas.Series(ids).drop_duplicates().sort_values().to_numpy()
    rlist: dict[str, Matrix] = {}
    nmi: list[float] = []
    for level in levels:
        idx = numpy.flatnonzero(ids == level)
        label = metafopy.methods.vcalc.as_character(numpy.array([level]))[0]
        ni_i = ni_[idx]
        if bool(numpy.any(ni_i[~numpy.isnan(ni_i)] <= 0)):
            raise metafopy.errors.InputError(f'One or more sample sizes are <= 0 in study {label}.')
        valid = ni_i[~numpy.isnan(ni_i)]
        if callable(nfun):
            value = nfun(ni_i)
            if numpy.ndim(value) != 0:
                raise metafopy.errors.InputError("Function specified via 'nfun' does not return a single value.")
            nmi.append(float(value))
        else:
            choice = match_nfun(nfun)
            if choice == 'min':
                nmi.append(float(valid.min()) if valid.size else float('inf'))
            elif choice == 'harmonic':
                nmi.append(float(1 / numpy.mean(1 / valid)) if valid.size else float('nan'))
            else:
                nmi.append(float(valid.mean()) if valid.size else float('nan'))
        var1, var2 = v1_chr[idx], v2_chr[idx]
        var1_var2 = [f'{a}.{b}' for a, b in zip(var1, var2, strict=True)]
        identical = [p for p, a, b in zip(var1_var2, var1, var2, strict=True) if a == b]
        if identical:
            raise metafopy.errors.InputError(f'Identical var1-var2 pair{"s" if len(identical) >= 2 else ""} ({", ".join(identical)}) in study {label}.')
        seen: set[str] = set()
        duplicated = []
        for p in var1_var2:
            if p in seen:
                duplicated.append(p)
            seen.add(p)
        if duplicated:
            raise metafopy.errors.InputError(f'Duplicated var1-var2 pair{"s" if len(duplicated) >= 2 else ""} ({", ".join(duplicated)}) in study {label}.')
        ri = ri_all[idx]
        if bool(numpy.any(numpy.abs(ri[~numpy.isnan(ri)]) > 1)):
            raise metafopy.errors.InputError(f'One or more correlations are > 1 or < -1 in study {label}.')
        variables = sorted(set(var1) | set(var2))
        position = {name: i for i, name in enumerate(variables)}
        matrix = numpy.full((len(variables), len(variables)), numpy.nan)
        numpy.fill_diagonal(matrix, 1.0)
        for a, b, r in zip(var1, var2, ri, strict=True):
            matrix[position[a], position[b]] = matrix[position[b], position[a]] = r
        rlist[label] = pandas.DataFrame(matrix, index=variables, columns=variables)
    return rcalc_list(rlist, numpy.array(nmi), rtoz, sparse, upper, simplify, rowid=rowid, vnames=vnames, noid=noid)


def rcalc(
    x: str | Matrix | collections.abc.Sequence[Matrix] | collections.abc.Mapping[str, Matrix],
    ni: metafopy.methods.vcalc.Variable | float = None,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    rtoz: bool = False,
    nfun: Nfun = 'min',
    sparse: bool = False,
    upper: bool = False,
    simplify: bool = True,
    *,
    rowid: collections.abc.Sequence[str] | None = None,
    vnames: collections.abc.Sequence[str] | None = None,
    noid: bool | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> RCalcResult | dict[str, RCalcResult]:
    """Variance-covariance matrix of correlation coefficients from the same samples (metafor's ``rcalc``).

    ``na_action`` stands for R's ``options(na.action=)``: an unknown value raises, the others
    give the same result (metafor sets ``na.pass`` while it computes).

    ``x`` is a formula string such as ``'ri ~ var1 + var2 | study'`` (the study part may be left
    out for a single study; ``ni`` is a column of ``data`` or an array), a correlation matrix, or a
    list (or dict) of correlation matrices, each an array or a data frame whose column labels name
    the variables. Only the lower triangle is used (the upper one if ``upper``).

    Returns an :class:`RCalcResult` with the data ``dat`` (one row per correlation, or per row of
    ``data`` when a formula is used) and the block diagonal ``V``. With ``simplify=False`` and a
    list of matrices, a dict of one result per matrix. As in R, the study identifier in ``dat``
    consists of strings (the names of the list, or the ids converted to character). ``nfun`` is
    ``'min'``, ``'harmonic'``, ``'mean'`` or a function of the sample sizes of a study.

    For a list of matrices with ``simplify=True``, ``rowid`` (values of ``'id.var1.var2'``) puts the rows of ``dat`` and ``V`` in that
    order, ``vnames`` (four names) renames the ``dat`` columns and ``noid=True`` drops the study identifier, as in metafor, which
    passes these on internally for a formula. The row and column names of ``V`` are in ``dat['var1.var2']``.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a formula, a matrix or a list of matrices.
        metafopy.errors.InputError: an argument or the data is invalid (R stops with an error here).
        metafopy.errors.NotFoundError: a variable of the formula is not in ``data``.
    """
    metafopy.methods.common.check_na_action(na_action)
    if not isinstance(x, str | numpy.ndarray | pandas.DataFrame | list | tuple | dict):
        raise metafopy.errors.ArgumentError("Argument 'x' must be either a formula, a matrix, or a list of matrices.")
    if ni is None:
        raise metafopy.errors.InputError("Argument 'ni' must be specified.")
    if isinstance(nfun, str):
        nfun = match_nfun(nfun)
    if isinstance(x, str):
        if data is None:
            raise metafopy.errors.InputError("Must specify the 'data' argument when 'x' is a formula.")
        return rcalc_formula(x, ni, pandas.DataFrame(data), rtoz, nfun, sparse, upper, simplify)
    if isinstance(x, list | tuple | dict):
        mapping = dict(x) if isinstance(x, dict) else {str(i + 1): m for i, m in enumerate(x)}
        ni_arr = numpy.atleast_1d(numpy.asarray(ni, dtype=numpy.float64))
        return rcalc_list({str(k): m for k, m in mapping.items()}, ni_arr, rtoz, sparse, upper, simplify, rowid, vnames, bool(noid))
    scalar = numpy.atleast_1d(numpy.asarray(ni, dtype=numpy.float64))
    if scalar.size != 1:
        raise metafopy.errors.ArgumentError("Argument 'ni' must be a single sample size when 'x' is a matrix.")
    return rcalc_matrix(x, float(scalar[0]), upper, rtoz)
