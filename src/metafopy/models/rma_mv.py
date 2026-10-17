"""Multivariate/multilevel meta-analytic models (metafor's ``rma.mv``).

The model is ``y = X beta + Z u + e`` with ``e ~ N(0, V)`` for a known sampling variance-covariance matrix ``V`` and random
effects whose structure is given by the ``random`` formulas, so that marginally ``y ~ N(X beta, M)`` (Konstantopoulos, 2011,
doi:10.1002/jrsm.35; Viechtbauer, 2010, doi:10.18637/jss.v036.i03). Ports
https://github.com/wviechtb/metafor/blob/master/R/rma.mv.r.
"""

import collections.abc
import dataclasses
import math
import re
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.sparse
import scipy.stats

import metafopy.core.linalg
import metafopy.core.numderiv
import metafopy.core.verbose
import metafopy.errors
import metafopy.formatters
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.mv_factor
import metafopy.models.mv_hessian
import metafopy.models.mv_inputs
import metafopy.models.mv_lik
import metafopy.models.mv_opt
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_uni
import metafopy.models.sel_opt

__all__ = ['MVControl', 'rma_mv']

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
CVVC_OPTIONS = ('varcor', 'varcov', 'transf')
OPTIMIZER_SETTINGS = ('abstol', 'maxit', 'ndeps', 'reltol', 'tol')

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Mat = FloatArray | scipy.sparse.csc_array

EPS = float(numpy.finfo(float).eps)
FIXED_METHODS = ('CE', 'EE', 'FE')
RSCALES = ('cor', 'cor0', 'cov0', 'none')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
INIT_NAMES = ('sigma2_init', 'tau2_init', 'rho_init', 'gamma2_init', 'phi_init')


@dataclasses.dataclass(frozen=True, slots=True)
class MVControl:
    """Control arguments of :func:`rma_mv` (metafor's ``control`` list).

    ``*_init`` give starting values (on the original scale), ``cholesky`` switches the Cholesky parametrisation of the
    ``UN`` and ``GEN`` structures on or off per term, ``gtol`` and ``maxiter`` control the default BFGS optimiser, and ``hessian_d``,
    ``hessian_eps`` and ``hessian_r`` are the arguments of ``numDeriv::hessian``'s Richardson method (metafor's
    ``hessianCtrl`` with its default ``hesspack='numDeriv'``) used for ``optbeta`` and ``cvvc``.

    ``optimizer`` (default ``'nlminb'``, here BFGS with the analytic score) may be any other optimiser of metafor's ``rma.mv``
    (``'optim'`` with ``optmethod``, ``'BFGS'``, ``'Nelder-Mead'``, ``'nlm'``, ``'bobyqa'``, ``'hjk'``, ...); they are run as
    :mod:`metafopy.models.sel_opt` describes (``optim`` with BFGS or Nelder-Mead is R's routine, the others are SciPy stand-ins), on
    the log-likelihood alone. ``opt_ctrl`` holds optimiser settings given in ``control`` (``maxit``, ``reltol``, ``abstol``, ``ndeps``,
    ``tol``); ``ncpus`` above 1 selects ``'optimParallel'`` but the likelihood is evaluated serially (``parallel`` and ``cl`` are
    accepted and ignored). ``nearpd`` forces ``G``, ``H`` (``UN`` and ``GEN`` structures) and ``M`` to be positive definite with
    ``nearPD`` (:func:`metafopy.core.linalg.near_pd`); the score is then taken by finite differences. ``hesspack``
    (``'numDeriv'``, ``'pracma'`` or ``'calculus'``) and ``hessian_ctrl`` (metafor's ``hessianCtrl``: ``d``, ``eps``, ``r``, ``v`` and
    ``zero.tol`` of numDeriv, ``h`` of pracma, ``accuracy`` and ``stepsize`` of calculus) choose the Hessian. ``mfmaxit`` is the
    number of likelihood evaluations after which the optimisation stops with an error.
    """

    sigma2_init: numpy.typing.ArrayLike | None = None
    tau2_init: numpy.typing.ArrayLike | None = None
    rho_init: numpy.typing.ArrayLike | None = None
    gamma2_init: numpy.typing.ArrayLike | None = None
    phi_init: numpy.typing.ArrayLike | None = None
    REMLf: bool = True
    evtol: float = 1e-7
    cholesky: bool | tuple[bool, bool] | None = None
    check_k_gtr_1: bool = True
    hesstol: float = math.sqrt(EPS)
    gtol: float = 1e-7
    maxiter: int = 5000
    hessian_d: float = 0.1
    hessian_eps: float = 1e-4
    hessian_r: int = 8
    optimizer: str = 'nlminb'
    optmethod: str = 'BFGS'
    opt_ctrl: collections.abc.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    ncpus: int = 1
    cl: typing.Any = None
    parallel: collections.abc.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    nearpd: bool = False
    hesspack: str = 'numDeriv'
    hessian_ctrl: collections.abc.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    mfmaxit: float = math.inf


def make_control(control: MVControl | collections.abc.Mapping[str, typing.Any] | None) -> MVControl:
    if control is None:
        return MVControl()
    if isinstance(control, MVControl):
        return control
    names = {f.name for f in dataclasses.fields(MVControl)}
    kwargs: dict[str, typing.Any] = {}
    opt_ctrl: dict[str, typing.Any] = {}
    for key, value in control.items():
        name = key.replace('.', '_').replace('hessianCtrl', 'hessian_ctrl')
        if name in OPTIMIZER_SETTINGS:
            opt_ctrl[name] = value
        elif name not in names:
            raise metafopy.errors.ArgumentError(f'unknown control argument {key!r}')
        else:
            kwargs[name] = value
    if opt_ctrl:
        kwargs['opt_ctrl'] = {**dict(kwargs.get('opt_ctrl', {})), **opt_ctrl}
    con = MVControl(**kwargs)
    return dataclasses.replace(con, hesspack=metafopy.models.sel_opt.partial_match(con.hesspack, metafopy.core.numderiv.HESSPACKS, 'hesspack'))


def hessian_settings(con: MVControl) -> tuple[str, dict[str, typing.Any]]:
    """The package and the ``hessianCtrl`` list of the Hessians: numDeriv uses ``r = 8`` unless the control gives ``r``."""
    if con.hesspack == 'numDeriv':
        return 'numDeriv', {'d': con.hessian_d, 'eps': con.hessian_eps, 'r': con.hessian_r, **con.hessian_ctrl}
    return con.hesspack, dict(con.hessian_ctrl)


def resolve_optimizer(con: MVControl) -> metafopy.models.sel_opt.OptimizerChoice | None:
    """metafor's optimiser bookkeeping for ``rma.mv``; ``None`` for the default ``nlminb`` (BFGS with the analytic score here).

    Raises:
        metafopy.errors.InputError: the optimiser is unknown.
    """
    optimizer = metafopy.models.sel_opt.partial_match(con.optimizer, metafopy.models.sel_opt.OPTIMIZERS_MV, 'optimizer')
    optmethod = metafopy.models.sel_opt.partial_match(con.optmethod, metafopy.models.sel_opt.OPTMETHODS, 'optmethod')
    if optimizer in metafopy.models.sel_opt.OPTMETHODS:
        optmethod, optimizer = optimizer, 'optim'
    if con.ncpus > 1:
        if int(con.ncpus) < 1:
            raise metafopy.errors.InputError("Control argument 'ncpus' must be >= 1.")
        optimizer = 'optimParallel'
    return None if optimizer == 'nlminb' else metafopy.models.sel_opt.OptimizerChoice(optimizer, optmethod)


@dataclasses.dataclass(slots=True)
class SInfo:
    """The ``~ 1 | id`` terms."""

    names: list[str]
    factors: list[metafopy.models.mv_factor.Factor]
    sigma2: FloatArray
    R: list[pandas.DataFrame | None]
    Rfix: list[bool]
    with_r: bool


def as_vector(x: numpy.typing.ArrayLike | None) -> FloatArray | None:
    return None if x is None else numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64)).reshape(-1)


def replace_call_notation(name: str) -> str:
    """Nicer names for variables given by ``interaction()``, ``paste()`` or ``paste0()``."""
    if 'interaction(' in name or 'paste(' in name or 'paste0(' in name:
        name = re.sub(r'interaction\((.*)\)', r'(\1)', name)
        name = re.sub(r'paste0?\((.*)\)', r'(\1)', name)
        name = name.replace(',', ':').replace(' ', '')
        name = re.sub(r'^\((.*)\)$', r'\1', name)
    return name


def clean_correlation_matrix(mat: object) -> pandas.DataFrame:
    """Validate one element of ``R``: square, symmetric, with unique dimension names and no missing values.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if not isinstance(mat, pandas.DataFrame):
        arr = numpy.asarray(mat, dtype=numpy.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise metafopy.errors.InputError("Elements of 'R' must be square matrices.")
        raise metafopy.errors.InputError("Elements of 'R' must have dimension names (pass a pandas DataFrame).")
    if mat.shape[0] != mat.shape[1]:
        raise metafopy.errors.InputError("Elements of 'R' must be square matrices.")
    arr = mat.to_numpy(dtype=numpy.float64)
    if not metafopy.models.mv_inputs.is_symmetric(arr):
        raise metafopy.errors.InputError("Elements of 'R' must be symmetric matrices.")
    arr = (arr + arr.T) / 2
    rows = [str(i) for i in mat.index]
    cols = [str(c) for c in mat.columns]
    no_rows = isinstance(mat.index, pandas.RangeIndex)
    no_cols = isinstance(mat.columns, pandas.RangeIndex)
    if no_rows and no_cols:
        raise metafopy.errors.InputError("Elements of 'R' must have dimension names.")
    if no_rows:
        rows = cols
    if no_cols:
        cols = rows
    frame = pandas.DataFrame(arr, index=rows, columns=cols)
    frame = frame.loc[~frame.index.duplicated(), ~frame.columns.duplicated()]
    if frame.isna().to_numpy().any():
        raise metafopy.errors.InputError("No missing values allowed in matrices specified via 'R'.")
    return frame


def process_s_before(
    random: metafopy.models.mv_inputs.RandomSpec,
    idx: numpy.typing.NDArray[numpy.intp],
    sigma2: numpy.typing.ArrayLike | None,
    R: collections.abc.Mapping[str, object] | None,
) -> SInfo:
    """Subset the ``~ 1 | id`` terms, check ``sigma2`` and ``R`` (the part of ``rma.mv`` before missing values are dropped).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    factors = [f.take(idx).drop_unused() for f in random.s_factors]
    if any(f.has_na for f in factors):
        raise metafopy.errors.InputError("No NAs allowed in variables specified in the 'random' argument.")
    n = len(factors)
    sig = (
        numpy.full(n, numpy.nan)
        if sigma2 is None
        else metafopy.models.mv_structs.expand1(numpy.atleast_1d(numpy.asarray(sigma2, dtype=numpy.float64)).reshape(-1), n)
    )
    if sig.shape[0] != n:
        raise metafopy.errors.InputError(f"Length of the 'sigma2' argument ({sig.shape[0]}) does not match the actual number of variance components ({n}).")
    if (sig[~numpy.isnan(sig)] < 0).any():
        raise metafopy.errors.InputError("Specified value(s) of 'sigma2' must be non-negative.")

    mats: list[pandas.DataFrame | None] = [None] * n
    with_r = False
    if R is not None:
        if not isinstance(R, collections.abc.Mapping):
            raise metafopy.errors.InputError("Argument 'R' must be a *named* list (a dict of DataFrames).")
        for j, name in enumerate(random.s_names):
            value = R.get(name)
            if value is not None:
                mats[j] = clean_correlation_matrix(value)
        if any(m is not None for m in mats):
            with_r = True
            for j, mat in enumerate(mats):
                if mat is None:
                    continue
                labels = set(mat.columns)
                if any(lv not in labels for lv in factors[j].levels):
                    raise metafopy.errors.InputError(
                        f"There are levels in '{random.s_names[j]}' for which there are no matching rows/columns in the corresponding 'R' matrix."
                    )
                if any(c not in set(factors[j].levels) for c in mat.columns):
                    warnings.warn(f"There are rows/columns in the 'R' matrix for '{random.s_names[j]}' for which there are no data.", stacklevel=4)
        else:
            warnings.warn("Argument 'R' specified, but list name(s) not in 'random'.", stacklevel=4)
    return SInfo(names=list(random.s_names), factors=factors, sigma2=sig, R=mats, Rfix=[m is not None for m in mats], with_r=with_r)


def scale_correlation(mat: pandas.DataFrame, rscale: str) -> pandas.DataFrame:
    arr = mat.to_numpy(dtype=numpy.float64)
    if rscale in ('cor', 'cor0'):
        if (numpy.diag(arr) <= 0).any():
            raise metafopy.errors.InputError('Cannot use Rscale="cor" or Rscale="cor0" with non-positive values on the diagonal of an \'R\' matrix.')
        arr = metafopy.models.mv_structs.cov2cor(arr)
        if (numpy.abs(arr) > 1).any():
            warnings.warn("Some values are larger than +-1 in an 'R' matrix after cov2cor() (see 'Rscale' argument).", stacklevel=5)
    if rscale == 'cor0':
        lo = float(arr.min())
        arr = (arr - lo) / (1 - lo)
    if rscale == 'cov0':
        arr = arr - float(arr.min())
    return pandas.DataFrame(arr, index=mat.index, columns=mat.columns)


def ddf_contain(
    X: FloatArray,
    k: int,
    s_factors: list[metafopy.models.mv_factor.Factor],
    g: metafopy.models.mv_structs.Group | None,
    h: metafopy.models.mv_structs.Group | None,
) -> FloatArray:
    """``.ddf.calc(dfs="contain")``: the number of levels of the smallest grouping variable within which each coefficient is constant, minus ``p``."""
    p = X.shape[1]

    def constant_within(col: FloatArray, codes: numpy.typing.NDArray[numpy.intp]) -> bool:
        return all(numpy.unique(col[codes == level]).shape[0] == 1 for level in numpy.unique(codes))

    def group_codes(grp: metafopy.models.mv_structs.Group) -> list[numpy.typing.NDArray[numpy.intp]]:
        cols = [grp.inner[c].astype(str).to_numpy() for c in grp.inner.columns]
        combo = numpy.array([' + '.join([*(str(c[i]) for c in cols), str(grp.outer.labels()[i])]) for i in range(grp.inner.shape[0])], dtype=object)
        out: list[numpy.typing.NDArray[numpy.intp]] = []
        for arr in (combo, numpy.asarray(grp.outer.labels(), dtype=object)):
            uniq, inverse = numpy.unique(arr.astype(str), return_inverse=True)
            out.append(inverse.astype(numpy.intp))
            del uniq
        return out

    sets: list[list[numpy.typing.NDArray[numpy.intp]]] = [[f.codes] for f in s_factors]
    for grp in (g, h):
        if grp is not None:
            sets.append(group_codes(grp))
    ddf = numpy.full(p, float(k))
    for j in range(p):
        best = float(k)
        for codes_list in sets:
            for codes in codes_list:
                if constant_within(X[:, j], codes):
                    best = min(best, float(numpy.unique(codes).shape[0]))
        ddf[j] = best - p
    ddf[ddf < 1] = 1
    return ddf


def make_term(
    g: metafopy.models.mv_structs.Group,
    cholesky: bool,
    nearpd: bool = False,
) -> metafopy.models.mv_structs.Term:
    return metafopy.models.mv_structs.Term(
        struct=g.struct,
        Z1=g.Z1,
        Z2=g.Z2,
        levels_r=g.levels_r,
        values=g.values,
        Dmat=g.Dmat,
        v_arg=g.tau2,
        r_arg=g.rho,
        cholesky=cholesky,
        nearpd=nearpd,
    )


def transform_start(
    g: metafopy.models.mv_structs.Group,
    tau2_init: FloatArray,
    rho_init: FloatArray,
    cholesky: bool,
) -> tuple[FloatArray, FloatArray]:
    """Starting values on the scale the optimiser works on (log variances, atanh/logit/log correlations, or Cholesky factors).

    Raises:
        metafopy.errors.EstimationError: the initial variance-covariance matrix is not positive definite.
    """
    if cholesky:
        E = metafopy.models.mv_structs.con_vcov_un(tau2_init, rho_init)
        try:
            L = numpy.linalg.cholesky(E)
        except numpy.linalg.LinAlgError as exc:
            raise metafopy.errors.EstimationError("Cannot take Choleski decomposition of initial 'G' matrix.") from exc
        low = metafopy.models.mv_structs.lower_tri(L)
        return numpy.diag(L).copy(), low if low.shape[0] else numpy.zeros(1)
    tau = numpy.log(tau2_init)
    if g.struct == 'CAR':
        rho = numpy.log(rho_init / (1 - rho_init))
    elif g.struct in metafopy.models.mv_structs.SPATIAL:
        rho = numpy.log(rho_init)
    else:
        rho = numpy.arctanh(rho_init)
    return tau, rho


def chk_vccon(ids: numpy.typing.ArrayLike | None, values: numpy.typing.ArrayLike | None, name: str) -> numpy.typing.ArrayLike | None:
    """``.chkvccon()``: equate the user-supplied values of constrained components (the mean of those given within each group).

    Raises:
        metafopy.errors.InputError: ``vccon`` does not match the length of the argument.
    """
    if ids is None or values is None:
        return values
    id_arr = numpy.atleast_1d(numpy.asarray(ids)).reshape(-1)
    vals = numpy.atleast_1d(numpy.asarray(values, dtype=numpy.float64)).reshape(-1).copy()
    if id_arr.shape[0] != vals.shape[0]:
        raise metafopy.errors.InputError(f"Length of 'vccon${name}' ({id_arr.shape[0]}) does not match the length of {name} ({vals.shape[0]}).")
    original = vals.copy()
    for ident in numpy.unique(id_arr):
        sel = id_arr == ident
        given = original[sel][~numpy.isnan(original[sel])]
        vals[sel] = given.mean() if given.shape[0] else numpy.nan
    return vals


def prepare_vccon(vccon: collections.abc.Mapping[str, numpy.typing.ArrayLike] | None) -> dict[str, numpy.typing.NDArray[typing.Any]]:
    """The ``vccon`` argument as a dict of arrays of group ids, keyed by ``sigma2``, ``tau2``, ``rho``, ``gamma2`` and ``phi``.

    Raises:
        metafopy.errors.InputError: an unknown component is named.
    """
    if vccon is None:
        return {}
    out: dict[str, numpy.typing.NDArray[typing.Any]] = {}
    for key, ids in vccon.items():
        if key not in metafopy.models.mv_structs.VC_NAMES:
            raise metafopy.errors.InputError(f"Unknown component {key!r} in 'vccon' (must be one of {', '.join(metafopy.models.mv_structs.VC_NAMES)}).")
        if ids is not None:
            out[key] = numpy.atleast_1d(numpy.asarray(ids)).reshape(-1)
    return out


def restart_controls(
    con: MVControl,
    previous: metafopy.models.mv_result.RMAMVResult,
    with_s: bool,
    with_g: bool,
    with_h: bool,
    sizes: tuple[int, int, int, int, int],
) -> MVControl:
    """``restart=TRUE``: take the starting values from the variance components of a previous fit.

    Raises:
        metafopy.errors.InputError: the previous fit has a different number of variance components.
    """
    present = (with_s, with_g, with_g, with_h, with_h)
    values = (previous.sigma2, previous.tau2, previous.rho, previous.gamma2, previous.phi)
    for flag, vals, size in zip(present, values, sizes, strict=True):
        if flag and (vals is None or vals.shape[0] != size):
            raise metafopy.errors.InputError('Restarting for a different model than the initial one.')
    return dataclasses.replace(
        con,
        sigma2_init=previous.sigma2 if with_s else con.sigma2_init,
        tau2_init=previous.tau2 if with_g else con.tau2_init,
        rho_init=previous.rho if with_g else con.rho_init,
        gamma2_init=previous.gamma2 if with_h else con.gamma2_init,
        phi_init=previous.phi if with_h else con.phi_init,
    )


def resolve_rscale(value: str | bool | int) -> str:
    """metafor's handling of ``Rscale``: a name, a logical or an integer 0 to 3.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(value, bool):
        return 'cor' if value else 'none'
    if isinstance(value, int):
        if not 0 <= value <= 3:
            raise metafopy.errors.InputError("Unknown 'Rscale' value specified.")
        return ('none', 'cor', 'cor0', 'cov0')[value]
    if value not in RSCALES:
        raise metafopy.errors.InputError("Unknown 'Rscale' value specified.")
    return value


@metafopy.models.call.records_call
def rma_mv(
    yi: object,
    V: object,
    *,
    W: object = None,
    mods: object = None,
    data: pandas.DataFrame | None = None,
    slab: object = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    random: str | collections.abc.Sequence[str] | None = None,
    struct: str | collections.abc.Sequence[str] = 'CS',
    intercept: bool = True,
    method: str = 'REML',
    test: str = 'z',
    dfs: str | float | collections.abc.Sequence[float] = 'residual',
    level: float = 95,
    digits: int | collections.abc.Mapping[str, int] | None = None,
    btt: metafopy.models.rma_uni.Btt = None,
    R: collections.abc.Mapping[str, object] | None = None,
    Rscale: str | bool | int = 'cor',
    sigma2: numpy.typing.ArrayLike | None = None,
    tau2: numpy.typing.ArrayLike | None = None,
    rho: numpy.typing.ArrayLike | None = None,
    gamma2: numpy.typing.ArrayLike | None = None,
    phi: numpy.typing.ArrayLike | None = None,
    sparse: bool = False,
    dist: object = 'euclidean',
    cvvc: bool | str = False,
    optbeta: bool = False,
    beta: numpy.typing.ArrayLike | None = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    vccon: collections.abc.Mapping[str, numpy.typing.ArrayLike] | None = None,
    restart: metafopy.models.mv_result.RMAMVResult | None = None,
    control: MVControl | collections.abc.Mapping[str, typing.Any] | None = None,
    verbose: bool | int = False,
    na_action: metafopy.metadata.NAAction = 'omit',
) -> metafopy.models.mv_result.RMAMVResult:
    """Fit a multivariate/multilevel meta-analytic model.

    Mirrors metafor's ``rma.mv()`` (https://wviechtb.github.io/metafor/reference/rma.mv.html). ``yi`` is a vector, a column
    of ``data`` or a model formula (``'yi ~ x1 + x2'``); ``V`` is a vector of sampling variances, a ``k x k`` matrix, a list of
    block matrices (see :func:`metafopy.bldiag`), a column name or ``0``. ``random`` is a formula such as ``'~ 1 | study'``,
    ``'~ 1 | district/school'`` or ``'~ outcome | study'`` (or a list of them), ``struct`` the variance-covariance structure of
    the ``~ inner | outer`` terms and ``R`` a dict of known correlation matrices (DataFrames with dimension names) keyed by the
    grouping variable. ``sigma2``, ``tau2``, ``rho``, ``gamma2`` and ``phi`` fix the corresponding parameters (``nan`` entries are
    estimated). ``dfs`` is ``'residual'``, ``'contain'`` or numeric. ``digits`` (a number of decimals, or a mapping with the keys ``est``, ``se``,
    ... of which ``est`` is used) is stored in the result and used by its printout unless ``digits`` is passed to ``summary()`` or set with
    :func:`metafopy.setmfopt`.

    ``cvvc`` (``False``, ``True``, ``'varcor'``, ``'varcov'`` or ``'transf'``) requests the Hessian of the variance components
    and its inverse (``hessian`` and ``vvc`` of the result): variances and correlations for ``True``/``'varcor'``, variances
    and covariances for ``'varcov'`` (``UN`` and ``GEN`` structures only), the transformed parameters for ``'transf'``. With
    ``optbeta=True`` the coefficients are optimised together with the variance components, and their covariance matrix is
    taken from the inverse Hessian (``lambda1`` and ``lambda2`` add a lasso and ridge penalty on the coefficients other than
    the intercept and imply ``optbeta=True``; see https://github.com/wviechtb/metafor/blob/master/R/rma.mv.r). ``beta`` fixes
    coefficients (``nan`` entries are estimated). ``vccon`` is a dict with the keys ``sigma2``, ``tau2``, ``rho``, ``gamma2``
    and ``phi`` whose entries are group ids: components with the same id are constrained to be equal. ``restart`` is a
    previous fit of the same model whose variance components are the starting values (metafor restarts from the last
    estimates in the session). The Hessians are the Richardson extrapolation of ``numDeriv`` (``hesspack='numDeriv'``).
    ``verbose`` prints metafor's table of the variance components and a log line per evaluation of the log-likelihood to
    stdout (the optimiser differs from metafor's ``nlminb``, so the sequence of evaluations differs too) and, with
    ``verbose=2`` or more, its progress messages to stderr.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: the model could not be fitted.
    """
    spec = {
        'yi': yi,
        'V': V,
        'W': W,
        'mods': mods,
        'data': data,
        'slab': slab,
        'subset': subset,
        'random': random,
        'struct': struct,
        'intercept': intercept,
        'method': method,
        'test': test,
        'dfs': dfs,
        'digits': digits,
        'level': level,
        'btt': btt,
        'R': R,
        'Rscale': Rscale,
        'sigma2': sigma2,
        'tau2': tau2,
        'rho': rho,
        'gamma2': gamma2,
        'phi': phi,
        'sparse': sparse,
        'dist': dist,
        'cvvc': cvvc,
        'optbeta': optbeta,
        'beta': beta,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'vccon': vccon,
        'control': control,
        'na_action': na_action,
    }
    if method not in (*FIXED_METHODS, 'ML', 'REML'):
        raise metafopy.errors.InputError("Unknown 'method' specified.")
    con = make_control(control)
    vlevel = int(verbose)
    if vlevel > 1:
        metafopy.core.verbose.cat('\n')
        metafopy.core.verbose.message('Extracting yi/V values ...')
        metafopy.core.verbose.message('Creating model matrix ...')
        if random is not None and method not in FIXED_METHODS:
            metafopy.core.verbose.message("Processing 'random' argument ...")
        metafopy.core.verbose.message('Generating/extracting study labels ...')
        if subset is not None:
            metafopy.core.verbose.message('Subsetting ...')
    if isinstance(cvvc, str) and cvvc not in CVVC_OPTIONS:
        raise metafopy.errors.InputError("Argument 'cvvc' must be TRUE, FALSE, 'varcor', 'varcov' or 'transf'.")
    cvvc_mode = 'varcor' if cvvc is True else (cvvc if isinstance(cvvc, str) else None)
    if lambda1 < 0 or lambda2 < 0:
        raise metafopy.errors.InputError("Arguments 'lambda1' and 'lambda2' must be non-negative.")
    if W is not None and (optbeta or lambda1 > 0 or lambda2 > 0):
        raise metafopy.errors.InputError("Cannot use custom weights when 'optbeta=True'.")
    optbeta = optbeta or lambda1 > 0 or lambda2 > 0
    vccon_ids = prepare_vccon(vccon)
    sigma2 = chk_vccon(vccon_ids.get('sigma2'), sigma2, 'sigma2')
    tau2 = chk_vccon(vccon_ids.get('tau2'), tau2, 'tau2')
    rho = chk_vccon(vccon_ids.get('rho'), rho, 'rho')
    gamma2 = chk_vccon(vccon_ids.get('gamma2'), gamma2, 'gamma2')
    phi = chk_vccon(vccon_ids.get('phi'), phi, 'phi')
    test_l = test.lower()
    if test_l == 'hksj':
        test_l = 'knha'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test_l not in ('z', 't', 'knha', 'adhoc'):
        raise metafopy.errors.InputError("Unknown option specified for the 'test' argument.")
    if isinstance(dfs, str) and dfs not in ('residual', 'contain'):
        raise metafopy.errors.InputError("Argument 'dfs' must be 'residual', 'contain' or numeric.")
    dfs_numeric = not isinstance(dfs, str)
    if test_l == 'z' and (dfs_numeric or dfs == 'contain'):
        test_l = 't'
    rscale = resolve_rscale(Rscale)
    dists = list(dist) if isinstance(dist, list | tuple) else [dist, dist]
    if len(dists) == 1:
        dists = dists * 2

    data_frame = None if data is None else data.reset_index(drop=True)
    yi_vec, mods_in, intercept_in = metafopy.models.mv_inputs.prepare_yi(yi, data_frame, mods, intercept)
    if yi_vec.dtype.kind not in 'fi':
        raise metafopy.errors.InputError("The object/variable specified for the 'yi' argument is not numeric.")
    k_all = int(yi_vec.shape[0])
    V_mat, v0 = metafopy.models.mv_inputs.prepare_v(V, data_frame, k_all, sparse)  # type: ignore[arg-type]
    A_mat = metafopy.models.mv_inputs.prepare_w(W, data_frame, k_all, sparse)
    X_all, x_names, x_missing, intercept_flag = metafopy.models.mv_inputs.build_x(mods_in, intercept_in, data_frame, k_all)

    stored_slab = metafopy.metadata.escalc_attr(yi if isinstance(yi, str) else None, data, k_all, 'slab') if slab is None else None
    if slab is None:
        slab_all = tuple(str(i + 1) for i in range(k_all)) if stored_slab is None else tuple(str(s) for s in stored_slab.tolist())
    elif isinstance(slab, str):
        if data_frame is None or slab not in data_frame.columns:
            raise metafopy.errors.InputError(f'slab={slab!r} names a column, but it is not in the data')
        slab_all = tuple(str(s) for s in data_frame[slab].tolist())
    else:
        slab_all = tuple(str(s) for s in numpy.asarray(slab, dtype=object).tolist())
    if len(slab_all) != k_all:
        raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(slab_all)}) does not correspond to the size of the dataset ({k_all}).")

    use_random = method not in FIXED_METHODS and random is not None
    if method in FIXED_METHODS and random is not None:
        warnings.warn(f'The \'random\' argument is disregaded when method="{method}".', stacklevel=2)
    if use_random and data_frame is None:
        raise metafopy.errors.InputError("The 'random' argument needs 'data' to evaluate its variables in.")
    rand = metafopy.models.mv_inputs.extract_random(random if use_random else None, struct, data_frame if data_frame is not None else pandas.DataFrame())

    mask = metafopy.metadata.subset_mask(subset, k_all, data_frame)
    idx = numpy.flatnonzero(mask)
    ids = tuple(int(i) for i in idx)
    yi_s = yi_vec[idx]
    V_s = metafopy.models.mv_inputs.take_square(V_mat, idx)
    A_s = None if A_mat is None else metafopy.models.mv_inputs.take_square(A_mat, idx)
    X_s = X_all[idx]
    x_missing_s = x_missing[idx]
    slab_s = metafopy.metadata.make_unique([slab_all[i] for i in idx])
    k = int(idx.shape[0])
    vi = numpy.asarray(V_s.diagonal() if scipy.sparse.issparse(V_s) else numpy.diag(V_s), dtype=numpy.float64).copy()
    k_f = k

    with_s, with_g, with_h = rand.with_s, rand.with_g, rand.with_h
    struct_t = rand.struct
    sinfo = process_s_before(rand, idx, sigma2, R)
    if not with_s:
        sinfo = SInfo(names=[], factors=[], sigma2=numpy.zeros(0), R=[], Rfix=[], with_r=False)
    s_levels_f = tuple(tuple(f.levels) for f in sinfo.factors)
    sigma2s = len(sinfo.factors) if with_s else 1

    group_g = group_h = None
    if with_g and rand.g_inner is not None and rand.g_outer is not None and rand.g_formula is not None:
        group_g = metafopy.models.mv_structs.process_after_sub(
            rand.g_inner.iloc[idx].reset_index(drop=True),
            rand.g_outer.iloc[idx].reset_index(drop=True),
            struct_t[0],
            rand.g_formula,
            as_vector(tau2),
            as_vector(rho),
            True,
            sparse,
        )
    if with_h and rand.h_inner is not None and rand.h_outer is not None and rand.h_formula is not None:
        group_h = metafopy.models.mv_structs.process_after_sub(
            rand.h_inner.iloc[idx].reset_index(drop=True),
            rand.h_outer.iloc[idx].reset_index(drop=True),
            struct_t[1],
            rand.h_formula,
            as_vector(gamma2),
            as_vector(phi),
            False,
            sparse,
        )

    # missing values
    has_na = numpy.isnan(yi_s) | x_missing_s
    if not v0:
        has_na |= numpy.isnan(vi)
        has_na |= metafopy.models.mv_inputs.any_na_v(V_s)
    if A_s is not None:
        has_na |= numpy.asarray(numpy.isnan(A_s.toarray() if scipy.sparse.issparse(A_s) else A_s).any(axis=1))
    not_na = ~has_na
    X_sub = X_s
    if has_na.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('Missing values in data.')
        keep = numpy.flatnonzero(not_na)
        yi_s = yi_s[keep]
        V_s = metafopy.models.mv_inputs.take_square(V_s, keep)
        A_s = None if A_s is None else metafopy.models.mv_inputs.take_square(A_s, keep)
        vi = vi[keep]
        X_s = X_s[keep]
        sinfo.factors = [f.take(keep) for f in sinfo.factors]
        k = int(keep.shape[0])
        warnings.warn(f'{int(has_na.sum())} {"rows" if has_na.sum() > 1 else "row"} with NAs omitted from model fitting.', stacklevel=2)
    if k <= 1:
        raise metafopy.errors.InputError('Processing terminated since k <= 1.')

    allvipos = True
    if (vi <= 0).any():
        allvipos = False
        if not v0:
            warnings.warn('There are outcomes with non-positive sampling variances.', stacklevel=2)
        neg = vi < 0
        if neg.any():
            dense = V_s.toarray() if scipy.sparse.issparse(V_s) else numpy.array(V_s)
            dense[neg, :] = 0.0
            dense[:, neg] = 0.0
            V_s = scipy.sparse.csc_array(dense) if sparse else dense
            vi[neg] = 0.0
            warnings.warn('Negative sampling variances constrained to 0.', stacklevel=2)
    if not v0:
        dense_v = V_s.toarray() if scipy.sparse.issparse(V_s) else V_s
        if not metafopy.core.linalg.is_positive_definite(numpy.asarray(dense_v)):
            warnings.warn("'V' appears to be not positive definite.", stacklevel=2)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        ratio = float(vi.max() / vi.min())
    if math.isfinite(ratio) and ratio >= 1e7:
        warnings.warn('Ratio of largest to smallest sampling variance extremely large. May not be able to obtain stable results.', stacklevel=2)

    # model matrix
    names = list(x_names)
    X = X_s
    coef_na = numpy.zeros(X.shape[1], dtype=numpy.bool_)
    if X.shape[1] == 0:
        warnings.warn('Cannot fit model with an empty model matrix. Coerced an intercept into the model.', stacklevel=2)
        X = numpy.ones((k, 1))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        names = ['intrcpt']
        coef_na = numpy.zeros(1, dtype=numpy.bool_)
    else:
        coef_na = metafopy.core.linalg.drop_redundant(X)
        if coef_na.any():
            warnings.warn('Redundant predictors dropped from the model.', stacklevel=2)
            X = X[:, ~coef_na]
            names = [n for n, d in zip(names, coef_na, strict=True) if not d]
    is_int = [metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])]
    int_incl = any(is_int)
    if int_incl:
        keep_cols = [j for j, flag in enumerate(is_int) if not flag]
        X = numpy.column_stack([numpy.ones(k), X[:, keep_cols]])
        names = ['intrcpt', *(names[j] for j in keep_cols)]
        intercept_flag = True
    X_f = model_matrix_all_rows(X_sub, coef_na, is_int, int_incl)
    if not metafopy.core.linalg.is_positive_definite(numpy.asarray(X.T @ X), tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')
    p = int(X.shape[1])
    coef_names = metafopy.metadata.make_unique(names)
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])
    btt_t = metafopy.models.rma_uni.set_btt(btt, p, int_incl, coef_names)
    m = len(btt_t)
    if beta is None:
        beta_arg = numpy.full(p, numpy.nan)
    else:
        beta_arg = numpy.atleast_1d(numpy.asarray(beta, dtype=numpy.float64)).reshape(-1)
        if beta_arg.shape[0] != p:
            raise metafopy.errors.InputError(f"Length of the 'beta' argument ({beta_arg.shape[0]}) does not match the actual number of fixed effects ({p}).")
    beta_est = numpy.isnan(beta_arg)

    # ~ 1 | id terms after missing values were dropped
    sinfo.factors = [f.drop_unused() for f in sinfo.factors]
    s_nlevels = tuple(f.nlevels for f in sinfo.factors)
    s_levels = tuple(tuple(f.levels) for f in sinfo.factors)
    sigma2_v = sinfo.sigma2.copy()
    Z_S: list[Mat] = []
    D_S: list[Mat] = []
    if with_s:
        single = numpy.isnan(sigma2_v) & (numpy.array(s_nlevels) == 1)
        if single.any() and con.check_k_gtr_1:
            sigma2_v[single] = 0.0
            warnings.warn("Single-level factor(s) found in 'random' argument. Corresponding 'sigma2' value(s) fixed to 0.", stacklevel=2)
        for fac in sinfo.factors:
            z = metafopy.models.mv_factor.indicator(fac, sparse) if fac.nlevels > 1 else metafopy.models.mv_structs.ones_col(k, sparse)
            Z_S.append(z)
        for j, z in enumerate(Z_S):
            frame = sinfo.R[j]
            if sinfo.Rfix[j] and frame is not None:
                scaled = (
                    scale_correlation(frame.loc[list(s_levels[j]), list(s_levels[j])], rscale)
                    if rscale != 'none'
                    else frame.loc[list(s_levels[j]), list(s_levels[j])]
                )
                r_arr = scipy.sparse.csc_array(scaled.to_numpy()) if sparse else scaled.to_numpy()
                D_S.append(typing.cast(Mat, z @ r_arr @ z.T))
            else:
                D_S.append(typing.cast(Mat, z @ z.T))
        D_S = [scipy.sparse.csc_array(d) if sparse else numpy.asarray(d) for d in D_S]

    if group_g is not None:
        metafopy.models.mv_structs.process_after_rmna(group_g, not_na, dists[0], con.check_k_gtr_1, sparse)
    if group_h is not None:
        metafopy.models.mv_structs.process_after_rmna(group_h, not_na, dists[1], con.check_k_gtr_1, sparse)

    Y = yi_s
    if_fixed = method in FIXED_METHODS or not (with_s or with_g or with_h)
    tau2s = group_g.tau2s if group_g is not None else 1
    rhos = group_g.rhos if group_g is not None else 1
    gamma2s = group_h.tau2s if group_h is not None else 1
    phis = group_h.rhos if group_h is not None else 1

    # initial values and QE from the equal-effects fit
    fac_v = None if v0 else metafopy.models.mv_lik.Factorization(V_s)
    QE = math.nan
    total: float
    if v0 or fac_v is None or not fac_v.ok:
        resid_ls = Y - X @ numpy.linalg.lstsq(X, Y, rcond=None)[0]
        total = float(resid_ls @ resid_ls) / (k - p) if k > p else math.nan
        if math.isnan(total):
            total = float(numpy.var(Y, ddof=1)) / 100
    else:
        beta_fe = numpy.linalg.solve(X.T @ fac_v.solve(X), X.T @ fac_v.solve(Y[:, None])[:, 0])
        res_fe = Y - X @ beta_fe
        with numpy.errstate(divide='ignore'):
            hm = 1.0 / float(numpy.mean(1.0 / vi))
        total = max(0.001 * (sigma2s + tau2s + gamma2s), float(numpy.var(res_fe, ddof=1)) - hm)
        QE = float(res_fe @ fac_v.solve(res_fe[:, None])[:, 0])
    if restart is not None:
        con = restart_controls(con, restart, with_s, group_g is not None, group_h is not None, (sigma2s, tau2s, rhos, gamma2s, phis))
    denom = sigma2s + tau2s + gamma2s
    sigma2_init = as_vector(con.sigma2_init)
    sigma2_init = numpy.full(sigma2s, total / denom) if sigma2_init is None else metafopy.models.mv_structs.expand1(sigma2_init, sigma2s)
    tau2_init = as_vector(con.tau2_init)
    tau2_init = numpy.full(tau2s, total / denom) if tau2_init is None else metafopy.models.mv_structs.expand1(tau2_init, tau2s)
    gamma2_init = as_vector(con.gamma2_init)
    gamma2_init = numpy.full(gamma2s, total / denom) if gamma2_init is None else metafopy.models.mv_structs.expand1(gamma2_init, gamma2s)
    rho_init = as_vector(con.rho_init)
    rho_init = (
        (group_g.rho_init if group_g is not None and group_g.rho_init is not None else numpy.full(rhos, 0.5))
        if rho_init is None
        else metafopy.models.mv_structs.expand1(rho_init, rhos)
    )
    phi_init = as_vector(con.phi_init)
    phi_init = (
        (group_h.rho_init if group_h is not None and group_h.rho_init is not None else numpy.full(phis, 0.5))
        if phi_init is None
        else metafopy.models.mv_structs.expand1(phi_init, phis)
    )

    if with_s and (sigma2_init <= 0).any():
        raise metafopy.errors.InputError("Value(s) of 'sigma2.init' must be > 0.")
    for grp, t_init, r_init, tn, rn in ((group_g, tau2_init, rho_init, 'tau2', 'rho'), (group_h, gamma2_init, phi_init, 'gamma2', 'phi')):
        if grp is None:
            continue
        if (t_init <= 0).any():
            raise metafopy.errors.InputError(f"Value(s) of '{tn}.init' must be > 0.")
        if grp.struct == 'CAR' and ((r_init <= 0) | (r_init >= 1)).any():
            raise metafopy.errors.InputError(f"Value(s) of '{rn}.init' must be in (0,1).")
        if grp.struct in metafopy.models.mv_structs.SPATIAL and (r_init <= 0).any():
            raise metafopy.errors.InputError(f"Value(s) of '{rn}.init' must be > 0.")
        if grp.struct != 'CAR' and grp.struct not in metafopy.models.mv_structs.SPATIAL and ((r_init <= -1) | (r_init >= 1)).any():
            raise metafopy.errors.InputError(f"Value(s) of '{rn}.init' must be in (-1,1).")

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    chol_default = [struct_t[0] in metafopy.models.mv_structs.UNSTRUCTURED, struct_t[1] in metafopy.models.mv_structs.UNSTRUCTURED]
    if con.cholesky is not None:
        chol_user = [bool(con.cholesky)] * 2 if isinstance(con.cholesky, bool) else [bool(c) for c in con.cholesky]
        chol_default = chol_user
    cholesky = (
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        bool(chol_default[0] and group_g is not None and struct_t[0] in metafopy.models.mv_structs.UNSTRUCTURED),
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        bool(chol_default[1] and group_h is not None and struct_t[1] in metafopy.models.mv_structs.UNSTRUCTURED),
    )

    # auditme: accept[redundant-sink-caller-plus-callee] the direct call logs the sigma2 start values, transform_start the tau2 and rho ones
    s_start = numpy.log(sigma2_init) if with_s else numpy.zeros(1)
    if group_g is not None:
        tau_start, rho_start = transform_start(group_g, tau2_init, rho_init, cholesky[0])
    else:
        tau_start, rho_start = numpy.zeros(1), numpy.zeros(1)
    if group_h is not None:
        gam_start, phi_start = transform_start(group_h, gamma2_init, phi_init, cholesky[1])
    else:
        gam_start, phi_start = numpy.zeros(1), numpy.zeros(1)
    par0 = numpy.concatenate([s_start, tau_start, rho_start, gam_start, phi_start])
    if optbeta:
        par0 = numpy.concatenate([numpy.zeros(p), par0])
    layout = metafopy.models.mv_lik.Layout(sigma2s=sigma2s, tau2s=tau2s, rhos=rhos, gamma2s=gamma2s, phis=phis)

    term_g = make_term(group_g, cholesky[0], con.nearpd) if group_g is not None else None
    term_h = make_term(group_h, cholesky[1], con.nearpd) if group_h is not None else None
    vccon_codes: dict[str, numpy.typing.NDArray[numpy.intp]] = {}
    for name, size, used in (
        ('sigma2', sigma2s, with_s),
        ('tau2', tau2s, group_g is not None),
        ('rho', rhos, group_g is not None),
        ('gamma2', gamma2s, group_h is not None),
        ('phi', phis, group_h is not None),
    ):
        if name in vccon_ids and used:
            if vccon_ids[name].shape[0] != size:
                raise metafopy.errors.InputError(
                    f"Length of 'vccon${name}' ({vccon_ids[name].shape[0]}) does not match the actual number of variance components ({size})."
                )
            vccon_codes[name] = numpy.unique(vccon_ids[name], return_inverse=True)[1].astype(numpy.intp).reshape(-1)
    problem = metafopy.models.mv_lik.Problem(
        Y=Y,
        V=V_s,
        A=A_s,
        X=X,
        D_S=tuple(D_S),
        sigma2_arg=sigma2_v if with_s else numpy.zeros(1),
        g=term_g,
        h=term_h,
        ZZ_g=None if term_g is None else metafopy.models.mv_lik.outer_cross(term_g.Z2),
        ZZ_h=None if term_h is None else metafopy.models.mv_lik.outer_cross(term_h.Z2),
        layout=layout,
        with_s=with_s,
        reml_full=con.REMLf,
        sparse=sparse,
        vccon=vccon_codes if vccon is not None else None,
        optbeta=optbeta,
        beta_arg=beta_arg,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        intercept=intercept_flag,
        nearpd=con.nearpd,
    )

    off = problem.offset
    free_list: list[int] = [j for j in range(p) if beta_est[j]] if optbeta else []
    if with_s:
        free_list += [off + j for j in range(sigma2s) if math.isnan(sigma2_v[j])]
    offset = off + sigma2s
    for grp, chol, n_v, n_r in ((group_g, cholesky[0], tau2s, rhos), (group_h, cholesky[1], gamma2s, phis)):
        if grp is not None:
            if chol:
                free_list += list(range(offset, offset + n_v + n_r)) if (numpy.isnan(grp.tau2).any() or numpy.isnan(grp.rho).any()) else []
            else:
                free_list += [offset + j for j in range(n_v) if math.isnan(grp.tau2[j])]
                free_list += [offset + n_v + j for j in range(n_r) if math.isnan(grp.rho[j])]
        offset += n_v + n_r
    free = numpy.array(free_list, dtype=numpy.intp)
    # which components the user fixed (taken before the saturated-model reset below, as metafor does)
    vc_fix: dict[str, BoolArray | None] = {
        'sigma2': ~numpy.isnan(sigma2_v) if with_s else None,
        'tau2': ~numpy.isnan(group_g.tau2) if group_g is not None else None,
        'rho': ~numpy.isnan(group_g.rho) if group_g is not None else None,
        'gamma2': ~numpy.isnan(group_h.tau2) if group_h is not None else None,
        'phi': ~numpy.isnan(group_h.rho) if group_h is not None else None,
    }
    reml = method == 'REML'
    if vlevel:
        specified = {
            'sigma2': sigma2_v if with_s else None,
            'tau2': group_g.tau2 if group_g is not None else None,
            'rho': group_g.rho if group_g is not None else None,
            'gamma2': group_h.tau2 if group_h is not None else None,
            'phi': group_h.rho if group_h is not None else None,
        }
        initial = {'sigma2': sigma2_init, 'tau2': tau2_init, 'rho': rho_init, 'gamma2': gamma2_init, 'phi': phi_init}
        metafopy.core.verbose.cat(vc_table(vc_fix, initial, specified, metafopy.mfopt.option_digits(digits)))
    if vlevel > 1:
        metafopy.core.verbose.message('Model fitting ...\n')

    iterations = 0
    par = par0
    if (optbeta or not if_fixed) and free.shape[0] > 0:
        opt = metafopy.models.mv_opt.maximise(
            problem,
            par0,
            free,
            reml,
            gtol=con.gtol,
            maxiter=con.maxiter,
            nonsmooth=lambda1 > 0,
            log=iteration_logger(problem, reml, metafopy.mfopt.option_digits(None)) if vlevel else None,
            choice=resolve_optimizer(con),
            opt_ctrl={'maxit': con.maxiter, **con.opt_ctrl},
            mfmaxit=con.mfmaxit,
        )
        par = opt.par
        iterations = opt.iterations
        if p == k:
            if with_s:
                sigma2_v = numpy.where(numpy.isnan(sigma2_v), 0.0, sigma2_v)
                problem = dataclasses.replace(problem, sigma2_arg=sigma2_v)
            for grp in (group_g, group_h):
                if grp is not None:
                    grp.tau2 = numpy.where(numpy.isnan(grp.tau2), 0.0, grp.tau2)
                    grp.rho = numpy.where(numpy.isnan(grp.rho), 0.0, grp.rho)
            term_g = make_term(group_g, cholesky[0], con.nearpd) if group_g is not None else None
            term_h = make_term(group_h, cholesky[1], con.nearpd) if group_h is not None else None
            problem = dataclasses.replace(problem, g=term_g, h=term_h)

    fit = metafopy.models.mv_lik.final_fit(problem, par)
    coef = fit.beta
    vb = numpy.asarray(fit.vb, dtype=numpy.float64).copy()
    hessian: FloatArray | None = None
    vvc: FloatArray | None = None
    vc_names: tuple[str, ...] = ()
    hesspack, hessian_ctrl = hessian_settings(con)
    hess_args = metafopy.models.mv_hessian.HessianArgs(hesspack, hessian_ctrl, con.hesstol)
    if optbeta:
        vb = numpy.full((p, p), numpy.nan)
        hres = metafopy.models.mv_hessian.coefficient_hessian(problem, par, reml, hess_args, int_incl)
        hessian, vc_names = hres.hessian, hres.names
        if hres.keep.any() and hres.hessian is not None:
            inv = metafopy.models.mv_hessian.chol2inv(hres.hessian)
            if inv is None:
                warnings.warn('Error when trying to invert the Hessian.', stacklevel=2)
            else:
                vvc = inv
                sel = hres.keep[:p]
                vb[numpy.ix_(sel, sel)] = inv[: int(sel.sum()), : int(sel.sum())]
    else:
        vb[~beta_est, :] = numpy.nan
        vb[:, ~beta_est] = numpy.nan
    M = fit.M
    sigma2_out = fit.sigma2 if with_s else numpy.zeros(1)  # metafor stores a 0 for a component the model does not have
    tau2_out, rho_out, G = (fit.tau2, fit.rho, fit.G) if group_g is not None else (numpy.zeros(1), numpy.zeros(1), None)
    gamma2_out, phi_out, H = (fit.gamma2, fit.phi, fit.H) if group_h is not None else (numpy.zeros(1), numpy.zeros(1), None)

    # tests of the fixed effects
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test_l in ('knha', 'adhoc', 't'):
        if isinstance(dfs, str):
            if dfs == 'residual':
                ddf = numpy.full(p, float(k - p))
            else:
                ddf = ddf_contain(X, k, sinfo.factors, group_g, group_h)
        else:
            ddf = metafopy.models.mv_structs.expand1(numpy.atleast_1d(numpy.asarray(dfs, dtype=numpy.float64)).reshape(-1), p)
            if ddf.shape[0] != p:
                raise metafopy.errors.InputError(f"Length of the 'dfs' argument ({ddf.shape[0]}) does not match the number of model coefficient ({p}).")
            ddf = numpy.where(ddf < 1, 1.0, ddf)
    else:
        ddf = numpy.full(p, numpy.nan)

    s2w = 1.0
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test_l in ('knha', 'adhoc'):
        dense_m = M.toarray() if scipy.sparse.issparse(M) else M
        fac_m = metafopy.models.mv_lik.Factorization(M)
        if not fac_m.ok:
            raise metafopy.errors.EstimationError('Failure when trying to compute adjustment factor for Knapp and Hartung method.')
        resid = Y - X @ coef
        rss = float(resid @ fac_m.solve(resid[:, None])[:, 0])
        s2w = 0.0 if rss <= EPS else rss / (k - p)
        del dense_m
        warnings.warn("Use of the Knapp and Hartung method for 'rma_mv()' models is experimental.", stacklevel=2)
    if test_l == 'adhoc':
        s2w = max(s2w, 1.0)
    vb = s2w * vb

    if vlevel > 1:
        metafopy.core.verbose.message(('' if vlevel > 2 else '\n') + 'Conducting tests of the fixed effects ...')
    idx_b = list(btt_t)
    try:
        QM = float(coef[idx_b] @ numpy.linalg.solve(vb[numpy.ix_(idx_b, idx_b)], coef[idx_b]))
    except numpy.linalg.LinAlgError:
        QM = math.nan
    if not numpy.all(numpy.isfinite(vb[numpy.ix_(idx_b, idx_b)])):
        QM = math.nan
    se = numpy.sqrt(numpy.diag(vb))
    zval = coef / se
    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if test_l in ('knha', 'adhoc', 't'):
        QM = QM / m
        df2 = float(numpy.min(ddf[idx_b]))
        qmdf: tuple[int, float | None] = (m, df2)
        QMp = float(scipy.stats.f.sf(QM, m, df2)) if df2 > 0 else math.nan
        pval = numpy.array([2 * scipy.stats.t.sf(abs(z), d) if d > 0 else math.nan for z, d in zip(zval, ddf, strict=True)])
        crit = numpy.array([scipy.stats.t.isf(alpha / 2, d) if d > 0 else math.nan for d in ddf])
    else:
        qmdf = (m, None)
        QMp = float(scipy.stats.chi2.sf(QM, m))
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = numpy.full(p, scipy.stats.norm.isf(alpha / 2))
    ci_lb = coef - crit * se
    ci_ub = coef + crit * se

    qedf = k - p
    if qedf > 0:
        QEp = float(scipy.stats.chi2.sf(QE, qedf)) if not math.isnan(QE) else math.nan
    else:
        QE, QEp = 0.0, 1.0

    # Hessian of the variance components
    if cvvc_mode is not None and not optbeta and use_random:
        any_fixed = any(flags is not None and bool(flags.any()) for flags in vc_fix.values())
        vres = metafopy.models.mv_hessian.variance_hessian(problem, reml, cvvc_mode, par, fit, hess_args, struct_t, any_fixed)
        hessian, vvc, vc_names = vres.hessian, vres.vvc, vres.names
        if vres.hessian is not None and vres.vvc is None:
            warnings.warn('Error when trying to invert the Hessian.', stacklevel=2)

    # fit statistics (only the estimated components count towards the number of parameters)
    if vlevel > 1:
        metafopy.core.verbose.message('Computing fit statistics and log-likelihood ...')
    p_eff = int(beta_est.sum())
    fixed_counts = {name: 0 if flags is None else int(flags.sum()) for name, flags in vc_fix.items()}
    present_vc = {'sigma2': with_s, 'tau2': group_g is not None, 'rho': group_g is not None, 'gamma2': group_h is not None, 'phi': group_h is not None}
    parms = p_eff
    if vccon is None:
        parms += sum(
            size - fixed_counts[name]
            for name, size in zip(metafopy.models.mv_structs.VC_NAMES, (sigma2s, tau2s, rhos, gamma2s, phis), strict=True)
            if present_vc[name]
        )
    else:
        # metafor counts only the components named in 'vccon' (one parameter per group of equal components)
        parms += sum(
            int(numpy.unique(vccon_ids[name]).shape[0]) - fixed_counts[name]
            for name in metafopy.models.mv_structs.VC_NAMES
            if present_vc[name] and name in vccon_ids
        )
    ll_ml, ll_reml = fit.ll_ml, fit.ll_reml
    if v0 or fac_v is None or not fac_v.ok:
        ll_qe = math.nan
    else:
        ll_qe = -0.5 * k * math.log(2 * math.pi) - 0.5 * fac_v.logdet
    dev_ml = -2 * (ll_ml - ll_qe) if allvipos else -2 * ll_ml
    parms_f = float(parms)
    ml = metafopy.models.rma_uni.FitStats(
        ll=ll_ml,
        dev=dev_ml,
        AIC=-2 * ll_ml + 2 * parms_f,
        BIC=-2 * ll_ml + parms_f * math.log(k),
        AICc=-2 * ll_ml + 2 * parms_f * max(k, parms_f + 2) / (max(k, parms_f + 2) - parms_f - 1),
    )
    kp = k - p_eff
    reml_stats = metafopy.models.rma_uni.FitStats(
        ll=ll_reml,
        dev=-2 * ll_reml,
        AIC=-2 * ll_reml + 2 * parms_f,
        BIC=-2 * ll_reml + parms_f * math.log(kp) if kp > 0 else math.nan,
        AICc=-2 * ll_reml + 2 * parms_f * max(kp, parms_f + 2) / (max(kp, parms_f + 2) - parms_f - 1),
    )

    s_names_out = tuple(replace_call_notation(n) for n in sinfo.names)
    g_names_out = tuple(replace_call_notation(n) for n in group_g.names) if group_g is not None else ()
    h_names_out = tuple(replace_call_notation(n) for n in group_h.names) if group_h is not None else ()

    if vlevel > 1:
        metafopy.core.verbose.message('Preparing output ...')
    out = metafopy.models.mv_result.RMAMVResult(
        beta=coef,
        se=se,
        zval=zval,
        pval=pval,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vb=vb,
        coef_names=coef_names,
        sigma2=sigma2_out,
        tau2=tau2_out,
        rho=rho_out,
        gamma2=gamma2_out,
        phi=phi_out,
        G=G,
        H=H,
        M=M,
        V=V_s,
        W=A_s,
        X=X,
        yi=Y,
        vi=vi,
        slab=coef_slab(slab_s, not_na),
        QE=QE,
        QEdf=qedf,
        QEp=QEp,
        QM=QM,
        QMdf=qmdf,
        QMp=QMp,
        k=k,
        k_f=k_f,
        k_all=k_all,
        p=p,
        m=m,
        btt=btt_t,
        parms=parms,
        fit_stats_ml=ml,
        fit_stats_reml=reml_stats,
        method=method,
        test=test_l,
        dfs=dfs if isinstance(dfs, str) else tuple(float(d) for d in numpy.atleast_1d(dfs)),
        ddf=ddf,
        s2w=s2w,
        level=alpha,
        int_only=int_only,
        int_incl=int_incl,
        intercept=intercept_flag,
        allvipos=allvipos,
        coef_na=tuple(bool(c) for c in coef_na),
        ids=ids,
        not_na=not_na,
        struct=struct_t,
        with_s=with_s,
        with_g=group_g is not None,
        with_h=group_h is not None,
        with_r=sinfo.with_r,
        sigma2s=sigma2s,
        tau2s=tau2s,
        rhos=rhos,
        gamma2s=gamma2s,
        phis=phis,
        vc_fix=vc_fix,
        s_names=s_names_out,
        g_names=g_names_out,
        h_names=h_names_out,
        s_nlevels=s_nlevels,
        s_levels=s_levels,
        s_levels_f=s_levels_f,
        slab_f=slab_s,
        X_f=X_f,
        Rfix=tuple(sinfo.Rfix) if sinfo.Rfix else (False,),
        Rscale=rscale,
        group_g=group_g,
        group_h=group_h,
        s_factors=tuple(sinfo.factors),
        Z_S=tuple(Z_S),
        D_S=tuple(D_S),
        problem=problem,
        hessian=hessian,
        vvc=vvc,
        vc_names=vc_names,
        spec=spec,
        digits=metafopy.mfopt.reduce_digits(digits) if digits is not None else 4,
        iterations=iterations,
        measure=str(data.attrs['measure']) if data is not None and data.attrs.get('measure') else 'GEN',
        p_eff=p_eff,
        optbeta=optbeta,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        beta_est=tuple(bool(b) for b in beta_est),
        vccon=None if vccon is None else {key: numpy.asarray(vccon_ids[key]) for key in vccon_ids},
        cvvc=cvvc_mode,
    )
    if vlevel:
        metafopy.core.verbose.cat('\n')
    return out


def vc_table(
    vc_fix: collections.abc.Mapping[str, BoolArray | None],
    initial: collections.abc.Mapping[str, FloatArray],
    specified: collections.abc.Mapping[str, FloatArray | None],
    digits: int,
) -> str:
    """metafor's ``Variance Components in Model:`` table of ``rma.mv(..., verbose=TRUE)``."""
    if all(flags is None for flags in vc_fix.values()):
        return '\nVariance Components in Model: none\n\n'
    header: list[str] = []
    rows: dict[str, list[str]] = {'included': [], 'fixed': [], 'initial': [], 'specified': []}
    for name in metafopy.models.mv_structs.VC_NAMES:
        flags = vc_fix[name]
        included = flags is not None
        n = 1 if flags is None else int(flags.shape[0])
        header += [name] if n == 1 else [f'{name}{i + 1}' for i in range(n)]
        spec = specified[name]
        for i in range(n):
            rows['included'].append('Yes' if included else 'No')
            rows['fixed'].append('---' if flags is None else ('TRUE' if bool(flags[i]) else 'FALSE'))
            init = float(initial[name][i]) if included and i < initial[name].shape[0] else math.nan
            rows['initial'].append('---' if math.isnan(init) else metafopy.formatters.r_as_character(round(init, digits)))
            val = math.nan if spec is None or i >= spec.shape[0] else float(spec[i])
            rows['specified'].append('---' if math.isnan(val) else metafopy.formatters.r_as_character(round(val, digits)))
    widths = [max(len(header[j]), *(len(r[j]) for r in rows.values())) for j in range(len(header))]
    name_w = max(len(n) for n in rows)
    lines = [' ' * name_w + ' ' + ' '.join(h.rjust(w) for h, w in zip(header, widths, strict=True))]
    lines += [n.ljust(name_w) + ' ' + ' '.join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for n, r in rows.items()]
    return '\nVariance Components in Model:\n\n' + '\n'.join(lines) + '\n\n'


def iteration_logger(problem: metafopy.models.mv_lik.Problem, reml: bool, digits: int) -> collections.abc.Callable[[FloatArray, float], None]:
    """A logger of each evaluation of the log-likelihood, printing metafor's ``Iteration ... ll = ... sigma2 = ...`` lines."""
    state = {'iteration': 0}

    def log(par: FloatArray, ll: float) -> None:
        fit = metafopy.models.mv_lik.final_fit(problem, par)
        fx = metafopy.formatters.fmtx
        parts = [f'Iteration {state["iteration"]:<5d} ', f'll = {fx(ll, digits, flag=" ")}   ']
        for label, values in (
            ('sigma2', fit.sigma2 if problem.with_s else None),
            ('tau2', fit.tau2 if problem.g is not None else None),
            ('rho', fit.rho if problem.g is not None else None),
            ('gamma2', fit.gamma2 if problem.h is not None else None),
            ('phi', fit.phi if problem.h is not None else None),
        ):
            if values is not None:
                parts.append(f'{label} =' + ' '.join(fx(list(numpy.atleast_1d(values)), digits, flag=' ')) + '  ')
        metafopy.core.verbose.cat(''.join(parts) + '\n')
        state['iteration'] += 1

    return log


def model_matrix_all_rows(X_sub: FloatArray, coef_na: BoolArray, is_int: list[bool], int_incl: bool) -> FloatArray:
    """``X.f``: the model matrix of every row of the subset (rows with missing values included), built as ``X`` is."""
    if X_sub.shape[1] == 0:
        return numpy.ones((X_sub.shape[0], 1))
    X_f = X_sub[:, ~coef_na] if coef_na.any() else X_sub
    if int_incl:
        keep_cols = [j for j, flag in enumerate(is_int) if not flag]
        return numpy.column_stack([numpy.ones(X_f.shape[0]), X_f[:, keep_cols]])
    return X_f


def coef_slab(slab: tuple[str, ...], not_na: BoolArray) -> tuple[str, ...]:
    return tuple(s for s, ok in zip(slab, not_na, strict=True) if ok)
