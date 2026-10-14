"""Aggregate effect sizes within clusters (metafor's ``aggregate.escalc``).

R source: https://github.com/wviechtb/metafor/blob/master/R/aggregate.escalc.r. Documentation:
https://wviechtb.github.io/metafor/reference/aggregate.escalc.html. Within each cluster the estimates
are combined by inverse-variance weighting with their var-cov matrix ``V`` (the weighted average
``1'W y / 1'W 1`` with variance ``1 / 1'W 1``, ``W = V^-1``), or by the unweighted mean with variance
``1'V 1 / k^2``. ``V`` is given or built from the sampling variances and a compound-symmetric
(``CS``), continuous-time autoregressive (``CAR``) or combined (``CS+CAR``, ``CS*CAR``) correlation
structure.

Differences from R that follow Python conventions:

- ``x`` is a :class:`pandas.DataFrame` (typically :func:`metafopy.escalc` output); the escalc
  attributes ``yi_names``, ``vi_names``, ``measure`` and ``digits`` are read from and written to
  ``DataFrame.attrs``;
- ``cluster``, ``time``, ``obs`` and ``V`` (as a vector) may be column names of ``x``; ``subset`` and
  ``select`` positions are 0-based;
- ``fun`` is a sequence of three callables, applied to the values of the numeric, the boolean and the
  other columns of each cluster.
"""

import collections.abc
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg

import metafopy.effects.datatools
import metafopy.effects.measures
import metafopy.effects.summary
import metafopy.errors

__all__ = ['aggregate_escalc']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Fun = collections.abc.Callable[[pandas.Series], typing.Any]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
STRUCTS = ('ID', 'CS', 'CAR', 'CS+CAR', 'CS*CAR')


def get_values(value: typing.Any, x: pandas.DataFrame, name: str) -> numpy.typing.NDArray[typing.Any] | None:
    """An argument given as values or as the name of a column of ``x``.

    Raises:
        metafopy.errors.NotFoundError: the column does not exist.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value not in x.columns:
            raise metafopy.errors.NotFoundError(f'column {value!r} (for {name}) not found in the data')
        return x[value].to_numpy()
    if isinstance(value, pandas.DataFrame):
        return value.to_numpy()
    return numpy.asarray(value)


def expand(value: typing.Any, n: int, name: str, lo: float, hi: float) -> FloatArray:
    """``rho``/``phi`` with one value per cluster, checked to lie in ``[lo, hi]``.

    Raises:
        metafopy.errors.InputError: the length or a value is invalid.
    """
    arr = numpy.atleast_1d(numpy.asarray(value, dtype=numpy.float64)).reshape(-1)
    if arr.shape[0] == 1:
        arr = numpy.repeat(arr, n)
    if arr.shape[0] != n:
        raise metafopy.errors.InputError(f"Length of '{name}' ({arr.shape[0]}) does not match the number of clusters ({n}).")
    if numpy.any(arr > hi) or numpy.any(arr < lo):
        raise metafopy.errors.InputError(f"Value(s) of '{name}' must be in [{lo:g},{hi:g}].")
    return arr


def build_V(
    struct: str,
    vi: FloatArray,
    cluster: numpy.typing.NDArray[typing.Any],
    ucluster: list[typing.Any],
    time: FloatArray | None,
    obs: numpy.typing.NDArray[typing.Any] | None,
    rho: FloatArray | None,
    phi: FloatArray | None,
) -> FloatArray:
    k = vi.shape[0]
    R = numpy.eye(k) if struct == 'ID' else numpy.zeros((k, k))
    if struct != 'ID':
        for i, cl in enumerate(ucluster):
            idx = numpy.flatnonzero(cluster == cl)
            block = numpy.ix_(idx, idx)
            if struct == 'CS' and rho is not None:
                R[block] = rho[i]
                continue
            if time is None or phi is None:
                continue
            ti = time[idx]
            car = phi[i] ** numpy.abs(numpy.subtract.outer(ti, ti))
            if struct == 'CAR':
                R[block] = car
            elif struct == 'CS+CAR' and rho is not None:
                R[block] = rho[i] + (1 - rho[i]) * car
            elif struct == 'CS*CAR' and rho is not None and obs is not None:
                oi = obs[idx]
                R[block] = numpy.where(numpy.equal.outer(oi, oi), 1.0, rho[i]) * car
    numpy.fill_diagonal(R, 1.0)
    S = numpy.sqrt(vi)
    out: FloatArray = S[:, None] * R * S[None, :]
    return out


def check_given_V(V: typing.Any, k: int) -> FloatArray:
    """A user-supplied ``V``: a vector gives a diagonal matrix.

    Raises:
        metafopy.errors.InputError: ``V`` is not a symmetric ``k x k`` matrix (or a vector of length ``k``).
    """
    arr = numpy.asarray(V, dtype=numpy.float64)
    if arr.ndim <= 1:
        arr = numpy.atleast_1d(arr).reshape(-1)
        if arr.shape[0] == 1:
            arr = numpy.repeat(arr, k)
        if arr.shape[0] != k:
            raise metafopy.errors.InputError(f"Length of 'V' ({arr.shape[0]}) does not match the length of the data frame ({k}).")
        return numpy.diag(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise metafopy.errors.InputError("'V' must be a square matrix.")
    finite = ~numpy.isnan(arr) & ~numpy.isnan(arr.T)
    if not numpy.allclose(arr[finite], arr.T[finite], rtol=1.5e-8, atol=0):
        raise metafopy.errors.InputError("'V' must be a symmetric matrix.")
    if arr.shape[0] != k:
        raise metafopy.errors.InputError(f"Dimensions of 'V' ({arr.shape[0]}x{arr.shape[1]}) do not match the length of the data frame ({k}).")
    return arr


def any_na_v(V: FloatArray) -> numpy.typing.NDArray[numpy.bool_]:
    """metafor's ``.anyNAv``: rows of ``V`` with a missing diagonal or a missing value among the columns with a non-missing diagonal."""
    ok_diag = ~numpy.isnan(numpy.diag(V))
    ok = ok_diag.copy()
    for i in numpy.flatnonzero(ok_diag):
        ok[i] = not numpy.isnan(V[i, ok_diag]).any()
    out: numpy.typing.NDArray[numpy.bool_] = ~ok
    return out


def default_funs(na_rm: bool) -> tuple[Fun, Fun, Fun]:
    """metafor's default ``fun``: the mean of numeric and logical variables, the most frequent value of the others."""

    def mean(values: pandas.Series) -> float:
        arr = values.to_numpy(dtype=numpy.float64)
        if na_rm:
            arr = arr[~numpy.isnan(arr)]
        return math.nan if arr.shape[0] == 0 else float(arr.mean())

    def mode(values: pandas.Series) -> typing.Any:
        vals = values.tolist()
        present = [v for v in vals if not metafopy.effects.datatools.is_missing(v)]
        levels = list(values.cat.categories) if isinstance(values.dtype, pandas.CategoricalDtype) else sorted(set(present))
        items: list[tuple[typing.Any, int]] = [(lvl, present.count(lvl)) for lvl in levels]
        n_missing = len(vals) - len(present)
        if not na_rm and n_missing > 0:
            items.append((None, n_missing))
        if not items:
            return None
        top = max(count for _, count in items)
        # tail(names(sort(table(x))), 1): sort() is stable, so among ties the last level wins
        return [lvl for lvl, count in items if count == top][-1]

    return mean, mean, mode


def aggregate_escalc(
    x: pandas.DataFrame,
    cluster: numpy.typing.ArrayLike | str,
    time: numpy.typing.ArrayLike | str | None = None,
    obs: numpy.typing.ArrayLike | str | None = None,
    V: numpy.typing.ArrayLike | pandas.DataFrame | str | None = None,
    struct: str = 'CS',
    rho: numpy.typing.ArrayLike | None = None,
    phi: numpy.typing.ArrayLike | None = None,
    weighted: bool = True,
    checkpd: bool = True,
    fun: collections.abc.Sequence[Fun] | None = None,
    na_rm: bool | collections.abc.Sequence[bool] = True,
    addk: bool = False,
    subset: numpy.typing.ArrayLike | str | None = None,
    select: collections.abc.Sequence[str | int] | None = None,
    digits: int | collections.abc.Sequence[int] | None = None,
    var_names: collections.abc.Sequence[str] | None = None,
) -> pandas.DataFrame:
    """Aggregate the estimates (and the other variables) of ``x`` to one row per cluster.

    ``cluster`` identifies the clusters (in order of first appearance). Without ``V`` the var-cov matrix
    of the estimates within a cluster has the correlation structure ``struct`` with correlation(s)
    ``rho`` and autocorrelation(s) ``phi`` (one value or one per cluster) over ``time`` (``obs``
    identifies the outcomes for ``CS*CAR``). ``weighted=False`` gives unweighted means. The other
    variables are aggregated with ``fun`` (default: means, and the most frequent value for non-numeric
    variables); ``na_rm`` (one value or one for the estimates and one for the other variables) drops
    missing values, ``addk`` adds the number of estimates per cluster as ``ki``, ``subset`` selects
    rows first and ``select`` the columns of the result.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: ``V`` is not positive definite or cannot be inverted in a cluster.
    """
    if struct not in STRUCTS:
        raise metafopy.errors.InputError("Unknown 'struct' specified.")
    nar = [bool(na_rm)] * 2 if isinstance(na_rm, bool | numpy.bool_) else [bool(v) for v in na_rm]
    if len(nar) == 1:
        nar = nar * 2
    x = x.reset_index(drop=True)
    attrs = dict(x.attrs)
    k = x.shape[0]

    cl_values = get_values(cluster, x, 'cluster')
    if cl_values is None:
        raise metafopy.errors.InputError("Must specify the 'cluster' variable.")
    cl = numpy.asarray(cl_values, dtype=object).reshape(-1)
    if any(metafopy.effects.datatools.is_missing(v) for v in cl):
        raise metafopy.errors.InputError("No missing values allowed in 'cluster' variable.")
    if cl.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of the variable specified via 'cluster' ({cl.shape[0]}) does not match the length of the data ({k}).")
    ucluster = list(pandas.unique(cl))
    n = len(ucluster)

    if var_names is None:
        yi_names = metafopy.effects.summary.names(attrs.get('yi_names'))
        vi_names = metafopy.effects.summary.names(attrs.get('vi_names'))
        if yi_names:
            yi_name = yi_names[0]
        elif 'yi' in x.columns:
            yi_name = 'yi'
        else:
            raise metafopy.errors.InputError("Cannot determine the name of the 'yi' variable.")
        if vi_names:
            vi_name = vi_names[0]
        elif 'vi' in x.columns:
            vi_name = 'vi'
        else:
            raise metafopy.errors.InputError("Cannot determine the name of the 'vi' variable.")
    else:
        if len(var_names) != 2:
            raise metafopy.errors.InputError("Argument 'var_names' must be of length 2.")
        yi_name, vi_name = var_names[0], var_names[1]
    for name in (yi_name, vi_name):
        if name not in x.columns:
            raise metafopy.errors.InputError(f"Cannot find variable '{name}' in the data frame.")
        if not pandas.api.types.is_numeric_dtype(x[name]):
            raise metafopy.errors.InputError(f"Variable '{name}' is not numeric.")
    yi = x[yi_name].to_numpy(dtype=numpy.float64, copy=True)
    vi = x[vi_name].to_numpy(dtype=numpy.float64, copy=True)

    V_values = get_values(V, x, 'V')
    if V_values is None:
        rho_v = phi_v = None
        time_v = obs_v = None
        if struct in ('CS', 'CS+CAR', 'CS*CAR'):
            if rho is None:
                raise metafopy.errors.InputError("Must specify 'rho' for this var-cov structure.")
            rho_v = expand(rho, n, 'rho', -1, 1)
        if struct in ('CAR', 'CS+CAR', 'CS*CAR'):
            if phi is None:
                raise metafopy.errors.InputError("Must specify 'phi' for this var-cov structure.")
            phi_v = expand(phi, n, 'phi', 0, 1)
            t_values = get_values(time, x, 'time')
            if t_values is None:
                raise metafopy.errors.InputError("Must specify a 'time' variable for this var-cov structure.")
            time_v = numpy.asarray(t_values, dtype=numpy.float64).reshape(-1)
            if time_v.shape[0] != k:
                raise metafopy.errors.InputError(
                    f"Length of the variable specified via 'time' ({time_v.shape[0]}) does not match the length of the data ({k})."
                )
            if struct == 'CS*CAR':
                o_values = get_values(obs, x, 'obs')
                if o_values is None:
                    raise metafopy.errors.InputError("Must specify an 'obs' variable for this var-cov structure.")
                obs_v = numpy.asarray(o_values, dtype=object).reshape(-1)
                if obs_v.shape[0] != k:
                    raise metafopy.errors.InputError(
                        f"Length of the variable specified via 'obs' ({obs_v.shape[0]}) does not match the length of the data ({k})."
                    )
        Vm = build_V(struct, vi, cl, ucluster, time_v, obs_v, rho_v, phi_v)
    else:
        Vm = check_given_V(V_values, k)
        eps = numpy.finfo(float).eps
        for c in ucluster:
            inside = cl == c
            block = Vm[numpy.ix_(inside, ~inside)]
            if numpy.any(numpy.abs(block[~numpy.isnan(block)]) >= eps):
                warnings.warn(f"Estimates in cluster '{c}' appear to have non-zero covariances with estimates belonging to different clusters.", stacklevel=2)

    sub = metafopy.effects.measures.chksubset(subset, k, x, 'subset')
    if sub is not None:
        x, yi, Vm, cl = x.loc[sub].reset_index(drop=True), yi[sub], Vm[numpy.ix_(sub, sub)], cl[sub]
        if x.shape[0] == 0:
            raise metafopy.errors.InputError('Processing terminated since k == 0.')

    if nar[0]:
        not_na = ~(numpy.isnan(yi) | any_na_v(Vm))
        if not not_na.all():
            x, yi, Vm, cl = x.loc[not_na].reset_index(drop=True), yi[not_na], Vm[numpy.ix_(not_na, not_na)], cl[not_na]
        if x.shape[0] == 0:
            raise metafopy.errors.InputError('Processing terminated since k == 0.')
    ucluster = list(pandas.unique(cl))
    n = len(ucluster)
    members = [numpy.flatnonzero(cl == c) for c in ucluster]

    if checkpd:
        all_pd = True
        eps = numpy.finfo(float).eps
        for c, idx in zip(ucluster, members, strict=True):
            Vi = Vm[numpy.ix_(idx, idx)]
            if not numpy.isnan(Vi).any() and numpy.any(numpy.linalg.eigvalsh(Vi) <= eps):
                all_pd = False
                warnings.warn(f"'V' appears to be not positive definite in cluster {c}.", stacklevel=2)
        if not all_pd:
            raise metafopy.errors.EstimationError("Cannot aggregate estimates with a non-positive-definite 'V' matrix.")

    yi_agg = numpy.full(n, numpy.nan)
    vi_agg = numpy.full(n, numpy.nan)
    for i, (c, idx) in enumerate(zip(ucluster, members, strict=True)):
        Vi = Vm[numpy.ix_(idx, idx)]
        if weighted:
            try:
                Wi = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Vi, lower=False), numpy.eye(idx.shape[0]))
            except (numpy.linalg.LinAlgError, ValueError) as err:
                raise metafopy.errors.EstimationError(f"Cannot take inverse of 'V' in cluster {c}.") from err
            sum_w = float(Wi.sum())
            yi_agg[i] = float((Wi @ yi[idx]).sum()) / sum_w
            vi_agg[i] = 1 / sum_w
        else:
            ki = idx.shape[0]
            yi_agg[i] = float(yi[idx].sum()) / ki
            vi_agg[i] = float(Vi.sum()) / ki**2

    if fun is not None:
        if len(fun) != 3 or not all(callable(f) for f in fun):
            raise metafopy.errors.InputError("Argument 'fun' must be a list of functions of length 3.")
        fun1, fun2, fun3 = fun
    else:
        fun1, fun2, fun3 = default_funs(nar[1])

    out: dict[str, list[typing.Any]] = {}
    for col in x.columns:
        series = x[col]
        if pandas.api.types.is_bool_dtype(series):
            f = fun2
        elif pandas.api.types.is_numeric_dtype(series):
            f = fun1
        else:
            f = fun3
        out[col] = [f(series.iloc[idx]) for idx in members]
    xagg = pandas.DataFrame(out, columns=list(x.columns))
    for col in x.columns:
        if isinstance(x[col].dtype, pandas.CategoricalDtype):
            xagg[col] = pandas.Categorical(xagg[col])
    xagg[yi_name] = yi_agg
    xagg[vi_name] = vi_agg
    if addk:
        xagg = pandas.concat([xagg, pandas.DataFrame({'ki': [idx.shape[0] for idx in members]})], axis=1)

    measure = attrs.get('measure') or 'GEN'
    new_digits = digits if digits is not None else attrs.get('digits')
    if new_digits is None:
        new_digits = 4
    if select is not None:
        cols = [str(c) for c in x.columns]
        picks = [cols[s] if isinstance(s, int | numpy.integer) else s for s in select]
        xagg = xagg.loc[:, picks]
    xagg = xagg.reset_index(drop=True)
    xagg.attrs = {'measure': measure, 'yi_names': [yi_name], 'vi_names': [vi_name], 'digits': new_digits}
    return xagg
