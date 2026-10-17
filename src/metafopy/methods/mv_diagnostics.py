"""Outlier and influence diagnostics for ``rma.mv`` fits (metafor's ``cooks.distance``, ``dfbetas``, ``rstudent`` and ``rstandard`` methods).

The model is refitted without each cluster (each observation unless ``cluster`` is given); with ``reestimate=True`` the
variance components are re-estimated (starting from the full-model estimates), otherwise they are kept at the full-model
values. Cook's distance, DFBETAS and the externally standardised (deleted) residuals follow Viechtbauer & Cheung (2010),
doi:10.1002/jrsm.11, extended to clusters as in metafor; the internally standardised residuals use the hat matrix of the
full fit. Ports https://github.com/wviechtb/metafor/blob/master/R/cooks.distance.rma.mv.r,
https://github.com/wviechtb/metafor/blob/master/R/dfbetas.rma.mv.r, https://github.com/wviechtb/metafor/blob/master/R/rstudent.rma.mv.r,
https://github.com/wviechtb/metafor/blob/master/R/rstandard.rma.mv.r and the helpers in ``misc.func.hidden.rma.mv.r``; see
https://wviechtb.github.io/metafor/reference/influence.rma.uni.html.
"""

import concurrent.futures
import dataclasses
import functools
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse

import metafopy.core.linalg
import metafopy.core.parallel
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.influence
import metafopy.models.mv_factor
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_mv
import metafopy.models.rma_uni
import metafopy.printing

__all__ = [
    'ClusterResidualResult',
    'cooks_rma_mv',
    'dfbetas_rma_mv',
    'refit_mv',
    'rstandard_rma_mv',
    'rstudent_rma_mv',
]

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type ObjArray = numpy.typing.NDArray[numpy.object_]
type MV = metafopy.models.mv_result.RMAMVResult
type Cluster = str | numpy.typing.ArrayLike | None

EPS = float(numpy.finfo(float).eps)
# what a failed refit looks like (metafor wraps the refits in try() and records NA)
REFIT_ERRORS = (metafopy.errors.MetaforError, ValueError, ArithmeticError, numpy.linalg.LinAlgError, scipy.linalg.LinAlgError)
# the arguments metafor does not pass on to the refits of the diagnostics
DROPPED = ('beta', 'btt', 'cvvc', 'lambda1', 'lambda2', 'restart')


@dataclasses.dataclass(frozen=True, slots=True)
class ClusterResidualResult:
    """Residuals per observation (``obs``, with the cluster of each in ``cluster``) and per cluster (metafor's ``list(obs, cluster)``).

    ``X2`` is the cluster's (deleted) residuals' quadratic form ``e' Var(e)^-1 e``, ``k`` the number of observations in each
    cluster and ``cluster_slab`` the cluster ids (sorted).
    """

    obs: metafopy.methods.influence.ResidualResult
    cluster: tuple[str, ...]
    X2: FloatArray
    k: FloatArray
    cluster_slab: tuple[str, ...]
    digits: int = 4

    def obs_frame(self) -> pandas.DataFrame:
        out = self.obs.to_frame()
        out['cluster'] = list(self.cluster)
        return out

    def cluster_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'X2': self.X2, 'k': self.k}, index=list(self.cluster_slab))

    def __str__(self) -> str:
        res = self.obs
        cols = [(n, [metafopy.printing.fmt_num(float(v), self.digits) for v in a]) for n, a in (('resid', res.resid), ('se', res.se), ('z', res.z))]
        cols.append(('cluster', list(self.cluster)))
        second = [
            ('X2', [metafopy.printing.fmt_num(float(v), self.digits) for v in self.X2]),
            ('k', ['NA' if numpy.isnan(v) else str(int(v)) for v in self.k]),
        ]
        return '\n'.join(
            [
                '$obs',
                '',
                *metafopy.methods.blup.format_r_frame(cols, res.slab),
                '',
                '$cluster',
                '',
                *metafopy.methods.blup.format_r_frame(second, self.cluster_slab),
                '',
            ]
        )


def id_label(v: typing.Any) -> str:
    if isinstance(v, int | float | numpy.number) and not isinstance(v, bool):
        return metafopy.models.mv_factor.format_number(float(v))
    return str(v)


@dataclasses.dataclass(frozen=True, slots=True)
class Clusters:
    """The cluster variable of the rows used in fitting (``values``), of the rows after subsetting (``values_f``) and the cluster ids."""

    values: ObjArray
    values_f: ObjArray
    ids: list[typing.Any]
    given: bool


def get_clusters(x: MV, cluster: Cluster) -> Clusters:
    """metafor's processing of ``cluster``: given for the ``k_all`` rows of the data, then subsetted and stripped of the rows with missing values.

    Raises:
        metafopy.errors.InputError: the variable cannot be found, has the wrong length or has missing values.
    """
    if cluster is None:
        values_all = numpy.arange(1, x.k_all + 1).astype(object)
    elif isinstance(cluster, str):
        data = x.spec.get('data')
        if not isinstance(data, pandas.DataFrame) or cluster not in data.columns:
            raise metafopy.errors.InputError(f"Cannot find the 'cluster' variable {cluster!r}.")
        values_all = data[cluster].to_numpy(dtype=object)
    else:
        values_all = numpy.asarray(cluster, dtype=object).reshape(-1)
    if values_all.shape[0] != x.k_all:
        raise metafopy.errors.InputError(
            f"Length of the variable specified via 'cluster' ({values_all.shape[0]}) does not match the length of the data ({x.k_all})."
        )
    values_f = values_all[numpy.array(x.ids, dtype=numpy.intp)]
    if pandas.isna(values_f).any():
        raise metafopy.errors.InputError("No missing values allowed in 'cluster' variable.")
    if values_f.shape[0] == 0:
        raise metafopy.errors.InputError("Cannot find 'cluster' variable (or it has zero length).")
    values = values_f[x.not_na]
    return Clusters(values=values, values_f=values_f, ids=list(pandas.unique(pandas.Series(values))), given=cluster is not None)


def sort_order(ids: list[typing.Any]) -> list[int]:
    """R's ``order(ids)``."""
    return sorted(range(len(ids)), key=lambda i: ids[i])


def vc_values(res: MV) -> dict[str, FloatArray | None]:
    return {name: None if res.vc_fix[name] is None else numpy.asarray(getattr(res, name), dtype=numpy.float64) for name in metafopy.models.mv_structs.VC_NAMES}


def refit_mv(
    x: MV,
    *,
    drop: BoolArray | None = None,
    fixed: dict[str, FloatArray | None] | None = None,
    init: bool = False,
) -> MV | None:
    """``x`` refitted as metafor's diagnostics do; ``None`` if the refit fails or a coefficient becomes inestimable.

    ``drop`` flags the rows (of the ``k`` used in fitting) to leave out. With ``fixed`` the variance components are fixed at
    the given values; otherwise the estimated components are re-estimated (with ``init=True`` starting from the estimates
    of ``x``) and the fixed ones kept.
    """
    spec = dict(x.spec)
    for name in DROPPED:
        spec.pop(name, None)
    if drop is not None:
        used = numpy.array(x.ids, dtype=numpy.intp)[x.not_na]
        mask = numpy.zeros(x.k_all, dtype=numpy.bool_)
        mask[used[~drop]] = True
        spec['subset'] = mask
    current = vc_values(x)
    for name in metafopy.models.mv_structs.VC_NAMES:
        flags = x.vc_fix[name]
        value = current[name]
        if flags is None or value is None:
            continue
        spec[name] = fixed[name] if fixed is not None else numpy.where(flags, value, numpy.nan)
    if init and fixed is None:
        control = metafopy.models.rma_mv.make_control(spec.get('control'))
        inits: dict[str, typing.Any] = {f'{name}_init': current[name] for name in metafopy.models.mv_structs.VC_NAMES if current[name] is not None}
        spec['control'] = dataclasses.replace(control, **inits)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = metafopy.models.rma_mv.rma_mv(**spec)
    except REFIT_ERRORS:
        return None
    if any(out.coef_na):
        return None
    return out


def refit_without(x: MV, incl: BoolArray, reestimate: bool) -> MV | None:
    """The fit without the rows ``incl``, re-estimating the variance components or keeping them at those of ``x``."""
    if reestimate:
        return refit_mv(x, drop=incl, init=True)
    return refit_mv(x, drop=incl, fixed=vc_values(x))


def cooks_one(x: MV, svb: FloatArray, btt: tuple[int, ...], clusters: Clusters, reestimate: bool, i: int) -> float:
    """``.cooks.distance.rma.mv``."""
    res = refit_without(x, numpy.asarray(clusters.values == clusters.ids[i], dtype=numpy.bool_), reestimate)
    if res is None:
        return float('nan')
    sub = numpy.array(btt, dtype=numpy.intp)
    dfb = x.beta[sub] - res.beta[sub]
    return float(dfb @ svb @ dfb)


def dfbetas_one(x: MV, clusters: Clusters, reestimate: bool, i: int) -> FloatArray:
    """``.dfbetas.rma.mv``."""
    res = refit_without(x, numpy.asarray(clusters.values == clusters.ids[i], dtype=numpy.bool_), reestimate)
    if res is None:
        return numpy.full(x.p, numpy.nan)
    tmp = refit_mv(x, fixed=vc_values(res))
    if tmp is None:
        return numpy.full(x.p, numpy.nan)
    return numpy.asarray((x.beta - res.beta) / numpy.sqrt(numpy.diag(tmp.vb)), dtype=numpy.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class Deleted:
    """What ``.rstudent.rma.mv`` returns for one cluster."""

    delresid: FloatArray
    sedelresid: FloatArray
    X2: float
    k_id: float
    pos: numpy.typing.NDArray[numpy.intp]


def dense(m: metafopy.models.mv_result.Mat) -> FloatArray:
    if isinstance(m, numpy.ndarray):
        return numpy.asarray(m, dtype=numpy.float64)
    return numpy.asarray(m.toarray(), dtype=numpy.float64)


def rstudent_one(x: MV, clusters: Clusters, reestimate: bool, i: int) -> Deleted:
    """``.rstudent.rma.mv``."""
    incl = numpy.asarray(clusters.values == clusters.ids[i], dtype=numpy.bool_)
    pos = numpy.flatnonzero(incl)
    k_id = int(incl.sum())
    nan = numpy.full(k_id, numpy.nan)
    res = refit_without(x, incl, reestimate)
    if res is None:
        return Deleted(delresid=nan, sedelresid=nan, X2=float('nan'), k_id=float('nan'), pos=pos)
    tmp = refit_mv(x, fixed=vc_values(res))
    if tmp is None:
        return Deleted(delresid=nan, sedelresid=nan, X2=float('nan'), k_id=float('nan'), pos=pos)
    Xi = x.X[incl]
    delresid = x.yi[incl] - Xi @ res.beta
    S = dense(tmp.M)[numpy.ix_(incl, incl)] + Xi @ res.vb @ Xi.T
    sedelresid = numpy.sqrt(numpy.diag(S))
    try:
        sve = scipy.linalg.cho_solve(scipy.linalg.cho_factor(S), numpy.eye(k_id))
    except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError:
        return Deleted(delresid=delresid, sedelresid=sedelresid, X2=float('nan'), k_id=float(k_id), pos=pos)
    return Deleted(delresid=delresid, sedelresid=sedelresid, X2=float(delresid @ sve @ delresid), k_id=float(k_id), pos=pos)


def fail_on_missing(x: MV, na_action: str) -> None:
    if na_action == 'fail' and not x.not_na.all():
        raise metafopy.errors.InputError('Missing values in results.')


def not_robust(x: MV) -> None:
    if x.robust is not None:
        raise metafopy.errors.NotSupportedError('Method not available for objects of class "robust.rma".')


def per_cluster(
    x: MV,
    clusters: Clusters,
    values: FloatArray,
    na_action: str,
) -> tuple[FloatArray, list[str]]:
    """Values per cluster with their labels, as metafor arranges them (study labels without ``cluster``, else the sorted ids)."""
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    pad = na_action in ('exclude', 'pass')
    if not pad:
        if not clusters.given:
            return values, [x.slab[j] for j in range(x.k)]
        order = sort_order(clusters.ids)
        return values[order], [id_label(clusters.ids[j]) for j in order]
    ids_f = list(pandas.unique(pandas.Series(clusters.values_f)))
    full = numpy.full((len(ids_f), *values.shape[1:]), numpy.nan)
    where = {id_label(v): j for j, v in enumerate(ids_f)}
    for row, v in enumerate(clusters.ids):
        full[where[id_label(v)]] = values[row]
    if not clusters.given:
        return full, list(x.slab_f)
    order = sort_order(ids_f)
    return full[order], [id_label(ids_f[j]) for j in order]


@metafopy.core.timing.timed
def cooks_rma_mv(
    model: MV,
    progbar: bool = False,
    cluster: Cluster = None,
    reestimate: bool = True,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    LB: bool = False,
    *,
    btt: metafopy.models.rma_uni.Btt = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.Series:
    """Cook's distance of each observation, or of each cluster (metafor's ``cooks.distance.rma.mv``).

    ``cluster`` (a column of the data or a vector for the rows of the original data) deletes whole clusters; the result is
    then indexed by the sorted cluster ids. ``btt`` (0-based) selects the coefficients the distance refers to. With
    ``reestimate=False`` the variance components are not re-estimated in the refits. ``parallel``, ``ncpus`` and ``cl``
    run the refits in a process pool (see :mod:`metafopy.core.parallel`); ``progbar=True`` prints a progress bar. ``LB`` is accepted as in
    metafor (load balancing of a cluster) and changes nothing.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    metafopy.methods.common.check_na_action(na_action)
    mode = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    clusters = get_clusters(x, cluster)
    btt_t = metafopy.models.rma_uni.set_btt(btt, x.p, False, x.coef_names)
    sub = numpy.array(btt_t, dtype=numpy.intp)
    svb = numpy.asarray(scipy.linalg.cho_solve(scipy.linalg.cho_factor(x.vb[numpy.ix_(sub, sub)]), numpy.eye(len(sub))), dtype=numpy.float64)
    one = functools.partial(cooks_one, x, svb, tuple(btt_t), clusters, reestimate)
    cook = numpy.array(metafopy.core.parallel.map_sims(one, list(range(len(clusters.ids))), parallel=mode, ncpus=ncpus, cl=cl, progbar=progbar))
    fail_on_missing(x, na_action)
    values, labels = per_cluster(x, clusters, cook, na_action)
    return pandas.Series(values, index=labels, name='cook.d')


@metafopy.core.timing.timed
def dfbetas_rma_mv(
    model: MV,
    progbar: bool = False,
    cluster: Cluster = None,
    reestimate: bool = True,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    LB: bool = False,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.DataFrame:
    """DFBETAS of each observation, or of each cluster (metafor's ``dfbetas.rma.mv``): one column per coefficient.

    The change in the coefficients when a cluster is deleted is scaled by the standard errors of the full-data fit with
    the variance components fixed at the values estimated without the cluster. The arguments are those of
    :func:`cooks_rma_mv`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model was passed through :func:`metafopy.robust`.
    """
    x = model
    not_robust(x)
    metafopy.methods.common.check_na_action(na_action)
    mode = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    clusters = get_clusters(x, cluster)
    one = functools.partial(dfbetas_one, x, clusters, reestimate)
    rows = metafopy.core.parallel.map_sims(one, list(range(len(clusters.ids))), parallel=mode, ncpus=ncpus, cl=cl, progbar=progbar)
    fail_on_missing(x, na_action)
    values, labels = per_cluster(x, clusters, numpy.vstack(rows), na_action)
    return pandas.DataFrame(values, index=labels, columns=list(x.coef_names))


def cluster_result(
    x: MV,
    clusters: Clusters,
    resid: FloatArray,
    se: FloatArray,
    X2: FloatArray,
    k_id: FloatArray,
    na_action: str,
    digits: int,
) -> metafopy.methods.influence.ResidualResult | ClusterResidualResult:
    """metafor's arrangement of (deleted) residuals: per observation, and per cluster when ``cluster`` was given."""
    k_f = x.not_na.shape[0]
    resid_f = numpy.full(k_f, numpy.nan)
    se_f = numpy.full(k_f, numpy.nan)
    resid_f[x.not_na] = resid
    se_f[x.not_na] = se
    z_f = resid_f / se_f
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    pad = na_action in ('exclude', 'pass')
    keep = numpy.ones(k_f, dtype=numpy.bool_) if pad else x.not_na
    slab = x.slab_f if pad else x.slab
    obs = metafopy.methods.influence.ResidualResult(resid=resid_f[keep], se=se_f[keep], z=z_f[keep], slab=tuple(slab), digits=digits)
    if not clusters.given:
        return obs
    cluster_obs = tuple(id_label(v) for v in (clusters.values_f if pad else clusters.values))
    if not pad:
        order = sort_order(clusters.ids)
        return ClusterResidualResult(
            obs=obs, cluster=cluster_obs, X2=X2[order], k=k_id[order], cluster_slab=tuple(id_label(clusters.ids[j]) for j in order), digits=digits
        )
    ids_f = list(pandas.unique(pandas.Series(clusters.values_f)))
    where = {id_label(v): j for j, v in enumerate(ids_f)}
    X2_f = numpy.full(len(ids_f), numpy.nan)
    for row, v in enumerate(clusters.ids):
        X2_f[where[id_label(v)]] = X2[row]
    k_f_ids = numpy.array([float(numpy.sum((clusters.values_f == v) & x.not_na)) for v in ids_f])
    order = sort_order(ids_f)
    return ClusterResidualResult(
        obs=obs, cluster=cluster_obs, X2=X2_f[order], k=k_f_ids[order], cluster_slab=tuple(id_label(ids_f[j]) for j in order), digits=digits
    )


@metafopy.core.timing.timed
def rstudent_rma_mv(
    model: MV,
    digits: int = 4,
    progbar: bool = False,
    cluster: Cluster = None,
    reestimate: bool = True,
    parallel: metafopy.core.parallel.Parallel = 'no',
    ncpus: int = 1,
    cl: concurrent.futures.Executor | None = None,
    LB: bool = False,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.influence.ResidualResult | ClusterResidualResult:
    """Externally standardised (deleted) residuals of an ``rma.mv`` fit (metafor's ``rstudent.rma.mv``).

    Each observation's residual is computed from the fit without it (without its whole cluster if ``cluster`` is given);
    its variance adds the marginal variance under the full-data fit with the variance components of that fit. With
    ``cluster`` a :class:`ClusterResidualResult` is returned, which also holds each cluster's ``X2`` statistic. The other
    arguments are those of :func:`cooks_rma_mv`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model was passed through :func:`metafopy.robust`.
    """
    x = model
    not_robust(x)
    metafopy.methods.common.check_na_action(na_action)
    mode = metafopy.core.parallel.resolve_parallel(parallel, ncpus, cl)
    clusters = get_clusters(x, cluster)
    one = functools.partial(rstudent_one, x, clusters, reestimate)
    parts = metafopy.core.parallel.map_sims(one, list(range(len(clusters.ids))), parallel=mode, ncpus=ncpus, cl=cl, progbar=progbar)
    delresid = numpy.full(x.k, numpy.nan)
    sedelresid = numpy.full(x.k, numpy.nan)
    for part in parts:
        delresid[part.pos] = part.delresid
        sedelresid[part.pos] = part.sedelresid
    delresid = metafopy.methods.common.tzero(delresid)
    X2 = numpy.array([part.X2 for part in parts])
    k_id = numpy.array([part.k_id for part in parts])
    fail_on_missing(x, na_action)
    return cluster_result(x, clusters, delresid, sedelresid, X2, k_id, na_action, digits)


def rstandard_rma_mv(
    model: MV,
    digits: int = 4,
    cluster: Cluster = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> metafopy.methods.influence.ResidualResult | ClusterResidualResult:
    """Internally standardised residuals of an ``rma.mv`` fit (metafor's ``rstandard.rma.mv``).

    The marginal residuals ``y - X b`` are divided by the square roots of the diagonal of ``(I - H) M (I - H)'``. With
    ``cluster`` (as for :func:`cooks_rma_mv`) a :class:`ClusterResidualResult` is returned, which also holds the ``X2``
    statistic ``e' Var(e)^-1 e`` of each cluster's residuals. metafor has no ``type`` argument for ``rma.mv`` fits (only
    marginal residuals).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model was passed through :func:`metafopy.robust`.
    """
    x = model
    not_robust(x)
    metafopy.methods.common.check_na_action(na_action)
    clusters = get_clusters(x, cluster)
    H = x.hat_matrix()
    ImH = numpy.eye(x.k) - H
    ei = metafopy.methods.common.tzero(x.yi - x.X @ x.beta)
    ve = ImH @ dense(x.M) @ ImH.T
    sei = numpy.sqrt(numpy.diag(ve))
    n = len(clusters.ids)
    X2 = numpy.full(n, numpy.nan)
    k_id = numpy.full(n, numpy.nan)
    if clusters.given:
        for i, cid in enumerate(clusters.ids):
            incl = numpy.asarray(clusters.values == cid, dtype=numpy.bool_)
            k_id[i] = float(incl.sum())
            vei = ve[numpy.ix_(incl, incl)]
            if not metafopy.core.linalg.is_positive_definite(vei.T @ vei):
                continue
            try:
                sve = scipy.linalg.cho_solve(scipy.linalg.cho_factor(vei), numpy.eye(vei.shape[0]))
            except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError:
                continue
            X2[i] = float(ei[incl] @ sve @ ei[incl])
    fail_on_missing(x, na_action)
    return cluster_result(x, clusters, ei, sei, X2, k_id, na_action, digits)


def influence_rma_mv(model: MV, *args: typing.Any, **kwargs: typing.Any) -> metafopy.methods.influence.InfluenceResult:
    """Not available: metafor has no ``influence`` method for ``rma.mv`` fits (use :func:`metafopy.cooks_distance` and relatives).

    Raises:
        metafopy.errors.NotSupportedError: always, as in metafor.
    """
    raise metafopy.errors.NotSupportedError("no applicable method for 'influence' applied to an object of class \"c('rma.mv', 'rma')\"")


metafopy.methods.influence.cooks_distance.register(metafopy.models.mv_result.RMAMVResult, cooks_rma_mv)
metafopy.methods.influence.dfbetas.register(metafopy.models.mv_result.RMAMVResult, dfbetas_rma_mv)
metafopy.methods.influence.rstudent.register(metafopy.models.mv_result.RMAMVResult, rstudent_rma_mv)  # type: ignore[arg-type]
metafopy.methods.influence.rstandard.register(metafopy.models.mv_result.RMAMVResult, rstandard_rma_mv)  # type: ignore[arg-type]
metafopy.methods.influence.influence.register(metafopy.models.mv_result.RMAMVResult, influence_rma_mv)
