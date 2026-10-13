"""Variance-covariance structures of the ``~ inner | outer`` terms of :func:`metafopy.rma_mv`.

Ports ``.process.G.aftersub()``, ``.process.G.afterrmna()``, ``.con.E()`` and the
``.con.vcov.*`` helpers of metafor's ``misc.func.hidden.rma.mv.r``
(https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.rma.mv.r).
A term has the form ``G = Z1 E Z1'`` (elementwise ``* Z2 Z2'``), where ``E`` is the
variance-covariance matrix of the inner levels implied by the structure
(Konstantopoulos, 2011, doi:10.1002/jrsm.35; Viechtbauer, 2010, doi:10.18637/jss.v036.i03).
"""

import dataclasses
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse
import scipy.spatial.distance
import scipy.special

import metafopy.errors
import metafopy.formula
import metafopy.models.mv_factor

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
__all__ = [
    'GENERAL',
    'SPATIAL',
    'STRUCTS',
    'Group',
    'Term',
    'con_e',
    'con_vcov_un',
    'con_vcov_un_chol',
    'process_after_rmna',
    'process_after_sub',
]

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Mat = FloatArray | scipy.sparse.csc_array

SPATIAL = ('SPEXP', 'SPGAU', 'SPLIN', 'SPRAT', 'SPSPH')
GENERAL = ('GDIAG', 'GEN')
# metafor 5.2-1 also has UNR, PHYBM, PHYPL and PHYPD in the code but disables them in the argument check
STRUCTS = ('AR', 'CAR', 'CS', 'DIAG', 'GDIAG', 'GEN', 'HAR', 'HCS', 'ID', 'SPEXP', 'SPGAU', 'SPLIN', 'SPRAT', 'SPSPH', 'UN')
FACTOR_STRUCTS = ('AR', 'CAR', 'CS', 'DIAG', 'HAR', 'HCS', 'ID', 'UN')
EPS10 = 10 * float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class Term:
    """What the likelihood needs to rebuild one ``G`` or ``H`` matrix from its parameters."""

    struct: str
    Z1: Mat
    Z2: Mat
    levels_r: BoolArray
    values: FloatArray | None
    Dmat: FloatArray | None
    v_arg: FloatArray
    r_arg: FloatArray
    cholesky: bool


@dataclasses.dataclass(slots=True)
class Group:
    """A processed ``~ inner | outer`` term (metafor's ``g.*`` or ``h.*`` objects)."""

    struct: str
    formula: metafopy.formula.RandomTerm
    is_g: bool
    names: tuple[str, ...]
    inner: pandas.DataFrame
    inner_factor: metafopy.models.mv_factor.Factor | None
    outer: metafopy.models.mv_factor.Factor
    nlevels: tuple[int, int]
    nlevels_f: tuple[int, int]
    levels: tuple[list[str], list[str]]
    levels_f: tuple[list[str], list[str]]
    values: FloatArray | None
    tau2s: int
    rhos: int
    tau2: FloatArray
    rho: FloatArray
    Z1: Mat
    Z2: Mat
    levels_r: BoolArray = dataclasses.field(default_factory=lambda: numpy.zeros(0, dtype=numpy.bool_))
    levels_k: FloatArray | None = None
    levels_comb_k: FloatArray | None = None
    Dmat: FloatArray | None = None
    rho_init: FloatArray | None = None
    inner_labels: list[str] = dataclasses.field(default_factory=list)


def label_vc(is_g: bool) -> tuple[str, str]:
    return ('tau2', 'rho') if is_g else ('gamma2', 'phi')


def expand1(x: FloatArray, n: int) -> FloatArray:
    return numpy.full(n, x[0]) if x.shape[0] == 1 else x


def inner_labels_of(inner: pandas.DataFrame) -> list[str]:
    """One label per row: the inner values pasted with `` + ``."""
    cols = [inner[c] for c in inner.columns]
    out: list[str] = []
    for i in range(len(inner)):
        parts = []
        for c in cols:
            v = c.iloc[i]
            parts.append(metafopy.models.mv_factor.format_number(float(v)) if isinstance(v, int | float | numpy.number) and not isinstance(v, bool) else str(v))
        out.append(' + '.join(parts))
    return out


def process_after_sub(
    inner: pandas.DataFrame,
    outer: pandas.Series,
    struct: str,
    formula: metafopy.formula.RandomTerm,
    tau2: FloatArray | None,
    rho: FloatArray | None,
    is_g: bool,
    sparse: bool,
) -> Group:
    """``.process.G.aftersub()``: factors, the number of parameters and the model matrices, before missing values are dropped.

    ``inner`` holds the inner variables (the expanded model matrix for ``GEN``/``GDIAG``), ``outer`` the grouping variable.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    k = len(outer)
    nvars = inner.shape[1] + 1
    tname, rname = label_vc(is_g)
    if struct in FACTOR_STRUCTS and inner.shape[1] != 1:
        raise metafopy.errors.InputError(f"Only a single inner variable allowed for an '~ inner | outer' term when 'struct=\"{struct}\"'.")
    names = (*(str(c) for c in inner.columns), str(outer.name))

    values: FloatArray | None = None
    if struct == 'CAR':
        if not pandas.api.types.is_numeric_dtype(inner.iloc[:, 0]):
            raise metafopy.errors.InputError("Inner variable in '~ inner | outer' term must be numeric for 'struct=\"CAR\"'.")
        values = numpy.sort(numpy.unique(numpy.round(inner.iloc[:, 0].to_numpy(dtype=float), 8)))

    outer_f = metafopy.models.mv_factor.to_factor(outer)
    if outer_f.has_na:
        raise metafopy.errors.InputError("No NAs allowed in variables specified via the 'random' argument.")
    inner_factor: metafopy.models.mv_factor.Factor | None
    if struct in SPATIAL or struct in GENERAL:
        if inner.isna().to_numpy().any():
            raise metafopy.errors.InputError("No NAs allowed in variables specified via the 'random' argument.")
        inner_factor = None
        labels = inner_labels_of(inner)
        ulabels = sorted(set(labels))
        nlevels = (len(ulabels), outer_f.nlevels)
        levels = (ulabels, list(outer_f.levels))
    else:
        inner_factor = metafopy.models.mv_factor.to_factor(inner.iloc[:, 0])
        if inner_factor.has_na:
            raise metafopy.errors.InputError("No NAs allowed in variables specified via the 'random' argument.")
        nlevels = (inner_factor.nlevels, outer_f.nlevels)
        levels = (list(inner_factor.levels), list(outer_f.levels))
        labels = []

    if struct in ('CS', 'ID', 'AR', 'CAR', *SPATIAL):
        tau2s, rhos = 1, 1
    elif struct in ('HCS', 'DIAG', 'HAR'):
        tau2s, rhos = nlevels[0], 1
    elif struct == 'UN':
        tau2s = nlevels[0]
        rhos = nlevels[0] * (nlevels[0] - 1) // 2 if nlevels[0] > 1 else 1
    elif struct == 'GEN':
        p = nvars - 1
        tau2s = p
        rhos = p * (p - 1) // 2 if p > 1 else 1
    else:  # GDIAG
        p = nvars - 1
        tau2s, rhos = p, 1

    tau2_v = numpy.full(tau2s, numpy.nan) if tau2 is None else expand1(numpy.asarray(tau2, dtype=numpy.float64).reshape(-1), tau2s)
    rho_v = numpy.full(rhos, numpy.nan) if rho is None else expand1(numpy.asarray(rho, dtype=numpy.float64).reshape(-1), rhos)
    if tau2_v.shape[0] != tau2s:
        raise metafopy.errors.InputError(
            f'Length of the {tname} argument ({tau2_v.shape[0]}) does not match the actual number of variance components ({tau2s}).'
        )
    if rho_v.shape[0] != rhos:
        raise metafopy.errors.InputError(f'Length of the {rname} argument ({rho_v.shape[0]}) does not match the actual number of correlations ({rhos}).')
    if (tau2_v[~numpy.isnan(tau2_v)] < 0).any():
        raise metafopy.errors.InputError(f'Specified value(s) of {tname} must be >= 0.')
    known = rho_v[~numpy.isnan(rho_v)]
    if struct == 'CAR' and ((known > 1) | (known < 0)).any():
        raise metafopy.errors.InputError(f'Specified value(s) of {rname} must be in [0,1].')
    if struct in SPATIAL and (known < 0).any():
        raise metafopy.errors.InputError(f'Specified value(s) of {rname} must be >= 0.')
    if struct != 'CAR' and struct not in SPATIAL and ((known > 1) | (known < -1)).any():
        raise metafopy.errors.InputError(f'Specified value(s) of {rname} must be in [-1,1].')

    Z1: Mat
    if struct in FACTOR_STRUCTS:
        if inner_factor is None:
            raise metafopy.errors.InternalError('inner factor missing')
        Z1 = metafopy.models.mv_factor.indicator(inner_factor, sparse) if nlevels[0] > 1 else ones_col(k, sparse)
    elif struct in SPATIAL:
        Z1 = identity(k, sparse)
    else:
        arr = inner.to_numpy(dtype=float)
        Z1 = scipy.sparse.csc_array(arr) if sparse else arr
    Z2 = metafopy.models.mv_factor.indicator(outer_f, sparse) if nlevels[1] > 1 else ones_col(k, sparse)

    return Group(
        struct=struct,
        formula=formula,
        is_g=is_g,
        names=names,
        inner=inner,
        inner_factor=inner_factor,
        outer=outer_f,
        nlevels=nlevels,
        nlevels_f=nlevels,
        levels=levels,
        levels_f=(list(levels[0]), list(levels[1])),
        values=values,
        tau2s=tau2s,
        rhos=rhos,
        tau2=tau2_v,
        rho=rho_v,
        Z1=Z1,
        Z2=Z2,
        inner_labels=labels,
    )


def identity(k: int, sparse: bool) -> Mat:
    return scipy.sparse.eye_array(k, format='csc') if sparse else numpy.eye(k)


def ones_col(k: int, sparse: bool) -> Mat:
    ones = numpy.ones((k, 1))
    return scipy.sparse.csc_array(ones) if sparse else ones


def take_rows(mat: Mat, rows: numpy.typing.NDArray[numpy.intp]) -> Mat:
    return mat.tocsr()[rows].tocsc() if scipy.sparse.issparse(mat) else mat[rows]


def con_vcov_un(vars_: FloatArray, cors: FloatArray, vccov: bool = False) -> FloatArray:
    """``.con.vcov.UN()``: covariance matrix from variances and correlations (or covariances when ``vccov``) in lower-triangle (column-major) order."""
    dims = vars_.shape[0]
    low = numpy.tril_indices(dims, -1)
    order = numpy.lexsort((low[0], low[1]))
    rows, cols = low[0][order], low[1][order]
    if vccov:
        G = numpy.zeros((dims, dims))
        G[rows, cols] = cors
        G[cols, rows] = cors
        numpy.fill_diagonal(G, vars_)
        return G
    R = numpy.ones((dims, dims))
    R[rows, cols] = cors
    R[cols, rows] = cors
    s = numpy.sqrt(vars_)
    return s[:, None] * R * s[None, :]


def con_vcov_un_chol(vars_: FloatArray, covs: FloatArray) -> FloatArray:
    """``.con.vcov.UN.chol()``: ``L L'`` for the lower-triangular ``L`` with diagonal ``vars_`` and below-diagonal ``covs``."""
    dims = vars_.shape[0]
    low = numpy.tril_indices(dims, -1)
    order = numpy.lexsort((low[0], low[1]))
    L = numpy.zeros((dims, dims))
    L[low[0][order], low[1][order]] = covs
    numpy.fill_diagonal(L, vars_)
    return L @ L.T


def lower_tri(G: FloatArray) -> FloatArray:
    """``G[lower.tri(G)]`` in R's column-major order."""
    low = numpy.tril_indices(G.shape[0], -1)
    order = numpy.lexsort((low[0], low[1]))
    return G[low[0][order], low[1][order]]


def cov2cor(E: FloatArray) -> FloatArray:
    d = numpy.sqrt(numpy.diag(E))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        out = E / d[:, None] / d[None, :]
    numpy.fill_diagonal(out, 1.0)
    return out


def ar1_corr(r: float, n: int) -> FloatArray:
    """``toeplitz(ARMAacf(ar=r, lag.max=n-1))``."""
    if n <= 1:
        return numpy.ones((1, 1))
    lags = numpy.abs(numpy.subtract.outer(numpy.arange(n), numpy.arange(n)))
    return numpy.power(float(r), lags)


def distance_matrix(group_inner: pandas.DataFrame, dist: object) -> FloatArray:
    """Distances between the rows of the inner variables (``dist(model.matrix(~ x + y - 1))``).

    Raises:
        metafopy.errors.InternalError: a distance matrix reached this function unresolved.
    """
    if isinstance(dist, pandas.DataFrame):
        raise metafopy.errors.InternalError('distance matrices are resolved by the caller')
    cols = []
    for c in group_inner.columns:
        s = group_inner[c]
        if pandas.api.types.is_numeric_dtype(s):
            cols.append(s.to_numpy(dtype=float)[:, None])
        else:
            dummies = pandas.get_dummies(s, drop_first=False).to_numpy(dtype=float)
            cols.append(dummies)
    cmat = numpy.hstack(cols)
    if callable(dist):
        out = numpy.asarray(dist(cmat), dtype=numpy.float64)
        return out
    metric: typing.Any = {'euclidean': 'euclidean', 'maximum': 'chebyshev', 'manhattan': 'cityblock'}.get(str(dist))
    if metric is None:
        raise metafopy.errors.NotSupportedError(f'dist={dist!r} is not supported (use "euclidean", "maximum", "manhattan", a function or a matrix)')
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cmat, metric=metric))


def process_after_rmna(
    g: Group,
    keep: BoolArray,
    dist: object,
    check_k_gtr_1: bool,
    sparse: bool,
) -> None:
    """``.process.G.afterrmna()``: re-evaluate levels after rows with missing values were dropped (updates ``g`` in place).

    ``keep`` flags the rows of the data given to :func:`process_after_sub` that remain.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    struct = g.struct
    tname, rname = label_vc(g.is_g)
    rows = numpy.flatnonzero(keep)
    inner = g.inner.iloc[rows].reset_index(drop=True)
    outer_old = g.outer.take(rows)
    g.Z1 = take_rows(g.Z1, rows) if struct not in SPATIAL else identity(rows.shape[0], sparse)
    g.Z2 = take_rows(g.Z2, rows)
    g.inner = inner
    nlev_f = g.nlevels_f
    levels_f = g.levels_f
    g.outer = outer_old.drop_unused()
    n = len(rows)
    if struct in SPATIAL or struct in GENERAL:
        labels = [g.inner_labels[i] for i in rows] if g.inner_labels else inner_labels_of(inner)
        g.inner_labels = labels
        ulabels = sorted(set(labels))
        g.nlevels = (len(ulabels), g.outer.nlevels)
        g.levels = (ulabels, list(g.outer.levels))
        # levels_f for these structures is the pre-removal set of labels
    else:
        if g.inner_factor is None:
            raise metafopy.errors.InternalError('inner factor missing')
        old_inner = g.inner_factor.take(rows)
        g.inner_factor = old_inner.drop_unused()
        g.nlevels = (g.inner_factor.nlevels, g.outer.nlevels)
        g.levels = (list(g.inner_factor.levels), list(g.outer.levels))

    g.levels_r = numpy.array([lv not in set(g.levels[0]) for lv in levels_f[0]], dtype=numpy.bool_)
    if g.levels_r.any() and struct not in ('AR', 'CAR', *SPATIAL, *GENERAL):
        warnings.warn(f'One or more levels of inner factor (i.e., {", ".join(numpy.array(levels_f[0])[g.levels_r])}) removed due to NAs.', stacklevel=3)

    if struct in ('ID', 'DIAG', 'GDIAG'):
        g.rho = numpy.zeros(1)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if g.nlevels[0] == 1 and struct in ('CS', 'HCS', 'AR', 'HAR', 'CAR') and numpy.isnan(g.rho[0]):
        g.rho = numpy.zeros(1)
        warnings.warn(f'Inner factor has only a single level, so fixed value of {rname} to 0.', stacklevel=3)
    if g.nlevels[0] == 1 and (struct in SPATIAL or struct in GENERAL):
        raise metafopy.errors.InputError('Cannot fit model since inner term only has a single level.')

    # k per level of the inner factor
    if struct in SPATIAL or struct in GENERAL:
        counts = pandas.Series(g.inner_labels).value_counts()
        g.levels_k = numpy.array([float(counts.get(lv, 0)) for lv in levels_f[0]])
    else:
        inner_codes = recode_to_levels(g.inner_factor, levels_f[0]) if g.inner_factor is not None else numpy.zeros(0, dtype=numpy.intp)
        outer_codes = g.outer.codes
        g.levels_k = numpy.array([float(len(set(outer_codes[inner_codes == i].tolist()))) for i in range(len(levels_f[0]))])

    if struct in ('HCS', 'UN', 'DIAG', 'HAR') and check_k_gtr_1:
        single = numpy.isnan(g.tau2) & (g.levels_k == 1)
        if single.any():
            g.tau2 = g.tau2.copy()
            g.tau2[single] = 0.0
            warnings.warn("Inner factor has k=1 for one or more levels. Corresponding 'tau2' value(s) fixed to 0.", stacklevel=3)

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if g.nlevels[1] == n and struct in ('CS', 'HCS', 'AR', 'HAR', 'CAR') and numpy.isnan(g.rho[0]):
        g.rho = numpy.zeros(1)
        warnings.warn(f'Each level of the outer factor contains only a single level of the inner factor, so fixed value of {rname} to 0.', stacklevel=3)
    if g.nlevels[1] == n and struct in SPATIAL:
        raise metafopy.errors.InputError('Cannot fit model since each level of the outer factor contains only a single level of the inner term.')

    g.levels_comb_k = None
    if struct not in SPATIAL and struct not in GENERAL:
        Z1d = g.Z1.toarray() if scipy.sparse.issparse(g.Z1) else g.Z1
        Z2d = g.Z2.toarray() if scipy.sparse.issparse(g.Z2) else g.Z2
        comb = Z2d.T @ Z1d  # outer x inner counts
        total = numpy.zeros((comb.shape[1], comb.shape[1]))
        for row in comb:
            nz = row != 0
            total += numpy.outer(nz, nz)
        g.levels_comb_k = lower_tri(total)
        if struct == 'UN' and ((g.levels_comb_k == 0) & numpy.isnan(g.rho)).any():
            g.rho = g.rho.copy()
            g.rho[g.levels_comb_k == 0] = 0.0
            warnings.warn(f'Some combinations of the levels of the inner factor never occurred. Corresponding {rname} value(s) fixed to 0.', stacklevel=3)
        if struct == 'UN' and nlev_f[0] == 1 and numpy.isnan(g.rho[0]):
            g.rho = numpy.zeros(1)
            warnings.warn(f'Inner factor has only a single level, so fixed value of {rname} to 0.', stacklevel=3)

    g.Dmat = None
    g.rho_init = None
    if struct in SPATIAL:
        if isinstance(dist, pandas.DataFrame):
            g.Dmat = distance_from_matrix(dist, g)
        else:
            g.Dmat = distance_matrix(g.inner, dist)
        lt = lower_tri(g.Dmat)
        g.rho_init = numpy.array([numpy.quantile(lt, 0.25)])

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if g.levels_r.any() and struct in ('HCS', 'HAR', 'DIAG'):
        g.tau2 = g.tau2.copy()
        g.tau2[g.levels_r] = 0.0
        warnings.warn(f'Fixed {tname} to 0 for removed level(s).', stacklevel=3)
    if g.levels_r.any() and struct == 'UN':
        G = con_vcov_un(g.tau2, g.rho)
        G[g.levels_r, :] = 0.0
        G[:, g.levels_r] = 0.0
        g.tau2 = g.tau2.copy()
        g.tau2[g.levels_r] = 0.0
        g.rho = lower_tri(G)
        warnings.warn(f'Fixed {tname} and corresponding {rname} value(s) to 0 for removed level(s).', stacklevel=3)

    if g.nlevels_f[0] == 2:
        if struct in ('CS', 'AR', 'CAR') and not numpy.isnan(g.tau2[0]) and g.tau2[0] == 0:
            g.rho = numpy.zeros(1)
        if struct in ('HCS', 'UN', 'HAR') and ((not numpy.isnan(g.tau2[0]) and g.tau2[0] == 0) or (not numpy.isnan(g.tau2[1]) and g.tau2[1] == 0)):
            g.rho = numpy.zeros(1)
    g.nlevels_f = nlev_f
    g.levels_f = levels_f


def recode_to_levels(f: metafopy.models.mv_factor.Factor, levels_f: list[str]) -> numpy.typing.NDArray[numpy.intp]:
    lookup = {lv: i for i, lv in enumerate(levels_f)}
    return numpy.array([lookup[f.levels[c]] if c >= 0 else -1 for c in f.codes], dtype=numpy.intp)


def distance_from_matrix(dist: pandas.DataFrame, g: Group) -> FloatArray:
    if dist.isna().to_numpy().any():
        raise metafopy.errors.InputError("No missing values allowed in matrices specified via 'dist'.")
    if dist.shape[0] != dist.shape[1]:
        raise metafopy.errors.InputError("Distance matrices specified via 'dist' must be square matrices.")
    arr = dist.to_numpy(dtype=float)
    if not numpy.allclose(arr, arr.T):
        raise metafopy.errors.InputError("Distance matrices specified via 'dist' must be symmetric matrices.")
    cols = [str(c) for c in dist.columns]
    idx = [str(c) for c in dist.index]
    if cols != idx and set(cols) == set(idx):
        arr = dist.loc[cols, cols].to_numpy(dtype=float)
    if len(set(cols)) != len(cols):
        raise metafopy.errors.InputError("Distance matrices specified via 'dist' must have unique dimension names.")
    lab = [str(v) for v in g.inner.iloc[:, 0].tolist()]
    missing = [v for v in lab if v not in cols]
    if missing:
        raise metafopy.errors.InputError(f"There are levels in '{g.names[0]}' for which there are no matching rows/columns in the corresponding 'dist' matrix.")
    pos = [cols.index(v) for v in lab]
    return arr[numpy.ix_(pos, pos)]


def con_e(v: FloatArray, r: FloatArray, term: Term, vctransf: bool, vccov: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``.con.E()``: the structure matrix ``E`` and the values of the variances and correlations it was built from.

    ``v`` and ``r`` are in the optimisation space (``log`` variances, ``atanh``/``qlogis``/``log`` correlations, or Cholesky
    factors) when ``vctransf``; fixed values (``term.v_arg``, ``term.r_arg``) replace them.


    Raises:
        metafopy.errors.InternalError: the distance matrix of a spatial structure is missing.
    """
    struct = term.struct
    v_arg, r_arg = term.v_arg, term.r_arg
    car = struct == 'CAR'
    spatial = struct in SPATIAL
    if not term.cholesky:
        if vctransf:
            v = numpy.where(numpy.isnan(v_arg), numpy.exp(v), v_arg)
            if car:
                r = numpy.where(numpy.isnan(r_arg), scipy.special.expit(r), r_arg)
            elif spatial:
                r = numpy.where(numpy.isnan(r_arg), numpy.exp(r), r_arg)
            else:
                r = numpy.where(numpy.isnan(r_arg), numpy.tanh(r), r_arg)
        else:
            v = numpy.where(numpy.isnan(v_arg), v, v_arg)
            r = numpy.where(numpy.isnan(r_arg), r, r_arg)
            v = numpy.where(v < 0, 0.0, v)
            if car:
                r = numpy.clip(r, 0.0, 1.0)
            elif spatial:
                r = numpy.where(r < 0, 0.0, r)
            elif not vccov:
                r = numpy.clip(r, -1.0, 1.0)
        v = numpy.where(v <= EPS10, 0.0, v)

    n1 = term.Z1.shape[1]
    E: FloatArray
    if struct == 'CS':
        E = numpy.full((n1, n1), r[0] * v[0])
        numpy.fill_diagonal(E, v[0])
    elif struct == 'HCS':
        E = numpy.full((n1, n1), r[0])
        numpy.fill_diagonal(E, 1.0)
        s = numpy.sqrt(v)
        E = s[:, None] * E * s[None, :]
        numpy.fill_diagonal(E, v)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif struct in ('UN', 'GEN'):
        if term.cholesky:
            E = con_vcov_un_chol(v, r)
            v = numpy.diag(E).copy()
            r = lower_tri(cov2cor(E))
            v = numpy.where(numpy.isnan(v_arg), v, v_arg)
            r = numpy.where(numpy.isnan(r_arg), r, r_arg)
        E = con_vcov_un(v, r, vccov)
    elif struct == 'GDIAG':
        E = numpy.diag(v)
    elif struct in ('ID', 'DIAG'):
        E = numpy.diag(numpy.broadcast_to(v, (n1,)).copy()) if v.shape[0] == 1 else numpy.diag(v)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif struct in ('AR', 'HAR', 'CAR'):
        if struct == 'CAR':
            if n1 > 1:
                vals = term.values if term.values is not None else numpy.arange(n1, dtype=float)
                E = numpy.power(float(r[0]), numpy.abs(numpy.subtract.outer(vals, vals)))
            else:
                E = numpy.ones((1, 1))
        else:
            E = ar1_corr(float(r[0]), n1)
        s = numpy.sqrt(numpy.broadcast_to(v, (n1,))) if struct != 'HAR' else numpy.sqrt(v)
        E = s[:, None] * E * s[None, :]
        numpy.fill_diagonal(E, numpy.broadcast_to(v, (n1,)) if struct != 'HAR' else v)
    else:  # spatial: Z1 is the identity, E is k x k
        D = term.Dmat
        if D is None:
            raise metafopy.errors.InternalError('distance matrix missing for a spatial structure')
        Z2 = term.Z2.toarray() if scipy.sparse.issparse(term.Z2) else term.Z2
        same = Z2 @ Z2.T
        rr = float(r[0])
        vv = float(v[0])
        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        if struct == 'SPEXP':
            E = vv * numpy.exp(-D / rr) * same
        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        elif struct == 'SPGAU':
            E = vv * numpy.exp(-(D**2) / rr**2) * same
        elif struct == 'SPLIN':
            E = vv * ((1 - D / rr) * (rr > D)) * same
        elif struct == 'SPRAT':
            E = vv * (1 - (D / rr) ** 2 / (1 + (D / rr) ** 2)) * same
        else:  # SPSPH
            E = vv * ((1 - 1.5 * D / rr + 0.5 * (D / rr) ** 3) * (rr > D)) * same

    if struct not in SPATIAL and struct not in GENERAL and term.levels_r.any():
        E[term.levels_r, :] = 0.0
        E[:, term.levels_r] = 0.0
    return v, r, E
