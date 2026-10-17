"""The fitted ``rma.mv`` model (:class:`RMAMVResult`)."""

import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse

import metafopy.core.labelled
import metafopy.errors
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.mv_lik
import metafopy.models.mv_structs
import metafopy.models.rma_uni
import metafopy.models.robust_info

__all__ = ['RMAMVResult']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Mat = FloatArray | scipy.sparse.csc_array


def formula_text(group: metafopy.models.mv_structs.Group | None) -> str | None:
    """The ``~ inner | outer`` formula of a random-effects term as text, ``None`` without a term."""
    if group is None:
        return None
    outer = '/'.join(group.formula.outer)
    return f'~ {group.formula.inner} | {outer}' if group.formula.inner is not None else f'~ 1 | {outer}'


def named_counts(group: metafopy.models.mv_structs.Group | None) -> pandas.Series[float] | None:
    """The observations per level of the inner factor as a series named by the levels."""
    if group is None or group.levels_k is None:
        return None
    levels = group.levels_f[0] if group.levels_f is not None and len(group.levels_f[0]) == len(group.levels_k) else None
    return pandas.Series(group.levels_k, index=levels)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RMAMVResult(metafopy.models.fit_methods.RobustFields, metafopy.models.fit_methods.FitMethods):
    """A fitted multivariate/multilevel meta-analytic model (metafor's ``rma.mv``).

    Names follow metafor with ``.`` replaced by ``_``; ``btt`` holds 0-based coefficient indices.
    ``spec`` keeps the arguments of the call, which :func:`metafopy.confint` and :func:`metafopy.profile` use to refit the model.
    ``hessian`` and ``vvc`` are the Hessian of ``-ll`` and its inverse (``optbeta=True``, where the leading rows are the
    coefficients, or ``cvvc``), with the row names ``vc_names``; ``p_eff`` is the number of estimated (not fixed) coefficients,
    ``beta_est`` flags them, and ``vccon`` holds the group ids of constrained variance components. ``slab_f`` and
    ``s_levels_f`` are the study labels and the levels of the ``~ 1 | id`` terms before rows with missing values were
    dropped (metafor's ``slab`` and ``s.levels.f``), used to pad results under ``na_action='exclude'``.
    ``X_f`` is the model matrix before rows with missing values were dropped (metafor's ``X.f``; NaN where a moderator value is missing),
    used by ``fitted()`` and ``predict()`` under ``na_action='pass'``.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    coef_names: tuple[str, ...]
    sigma2: FloatArray | None
    tau2: FloatArray | None
    rho: FloatArray | None
    gamma2: FloatArray | None
    phi: FloatArray | None
    G: FloatArray | None
    H: FloatArray | None
    M: Mat
    V: Mat
    W: Mat | None
    X: FloatArray
    yi: FloatArray
    vi: FloatArray
    slab: tuple[str, ...]
    QE: float
    QEdf: int
    QEp: float
    QM: float
    QMdf: tuple[int, float | None]
    QMp: float
    k: int
    k_f: int
    k_all: int
    p: int
    m: int
    btt: tuple[int, ...]
    parms: int
    fit_stats_ml: metafopy.models.rma_uni.FitStats
    fit_stats_reml: metafopy.models.rma_uni.FitStats
    method: str
    test: str
    dfs: str | float | tuple[float, ...]
    ddf: FloatArray
    s2w: float
    level: float
    int_only: bool
    int_incl: bool
    intercept: bool
    allvipos: bool
    coef_na: tuple[bool, ...]
    ids: tuple[int, ...]
    not_na: numpy.typing.NDArray[numpy.bool_]
    struct: tuple[str, str]
    with_s: bool
    with_g: bool
    with_h: bool
    with_r: bool
    sigma2s: int
    tau2s: int
    rhos: int
    gamma2s: int
    phis: int
    vc_fix: dict[str, numpy.typing.NDArray[numpy.bool_] | None]
    s_names: tuple[str, ...]
    g_names: tuple[str, ...]
    h_names: tuple[str, ...]
    s_nlevels: tuple[int, ...]
    s_levels: tuple[tuple[str, ...], ...]
    Rfix: tuple[bool, ...]
    Rscale: str
    group_g: metafopy.models.mv_structs.Group | None
    group_h: metafopy.models.mv_structs.Group | None
    s_factors: tuple[typing.Any, ...]
    Z_S: tuple[Mat, ...]
    D_S: tuple[Mat, ...]
    problem: metafopy.models.mv_lik.Problem
    hessian: FloatArray | None
    vvc: FloatArray | None
    vc_names: tuple[str, ...]
    spec: dict[str, typing.Any]
    iterations: int = 0
    measure: str = 'GEN'
    robust: metafopy.models.robust_info.RobustInfo | None = None
    p_eff: int = 0
    optbeta: bool = False
    lambda1: float = 0.0
    lambda2: float = 0.0
    beta_est: tuple[bool, ...] = ()
    vccon: dict[str, numpy.typing.NDArray[typing.Any]] | None = None
    cvvc: str | None = None
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)
    s_levels_f: tuple[tuple[str, ...], ...] = ()
    slab_f: tuple[str, ...] = ()
    X_f: FloatArray | None = None
    digits: int = 4
    yi_f: FloatArray = dataclasses.field(default_factory=lambda: numpy.empty(0), repr=False)
    vi_f: FloatArray = dataclasses.field(default_factory=lambda: numpy.empty(0), repr=False)
    V_f: Mat | None = dataclasses.field(default=None, repr=False)
    W_f: Mat | None = dataclasses.field(default=None, repr=False)
    ni: FloatArray | None = dataclasses.field(default=None, repr=False)
    ni_f: FloatArray | None = dataclasses.field(default=None, repr=False)
    subset: numpy.typing.NDArray[numpy.bool_] | None = dataclasses.field(default=None, repr=False)
    slab_null: bool = True
    control: typing.Any = dataclasses.field(default=None, repr=False)

    def __post_init__(self) -> None:
        # R names the rows of the model matrices by the rows of the data and their columns by the coefficients
        row_names = [str(int(i) + 1) for i in self.ids]
        if len(row_names) == self.not_na.shape[0]:
            used = [n for n, flag in zip(row_names, self.not_na, strict=True) if flag]
            object.__setattr__(self, 'X', metafopy.core.labelled.design_matrix(self.X, used, self.coef_names))
            if self.X_f is not None:
                object.__setattr__(self, 'X_f', metafopy.core.labelled.design_matrix(self.X_f, row_names, self.coef_names))
        # R sets the dimnames of G and H (levels of the inner factor) and of hessian and vvc (the names of the variance components)
        for name, names in (('G', self.G_dimnames), ('H', self.H_dimnames), ('hessian', self.vc_names), ('vvc', self.vc_names)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, metafopy.core.labelled.LabelledMatrix) and len(names) == value.shape[0]:
                object.__setattr__(self, name, metafopy.core.labelled.LabelledMatrix(value, names, names))

    # -- accessors -------------------------------------------------------

    def coef(self, type: str = 'beta') -> pandas.Series:
        """Estimated coefficients, indexed by name; ``type`` is accepted for metafor's ``coef(x, type=)`` and only location-scale fits use it."""
        return pandas.Series(self.beta, index=list(self.coef_names), name='estimate')

    @property
    def g_nlevels(self) -> tuple[int, int] | None:
        """Number of levels of the inner and outer factor of the ``tau2``/``rho`` term in the fit (metafor's ``g.nlevels``), ``None`` without one."""
        return None if self.group_g is None else self.group_g.nlevels

    @property
    def g_nlevels_f(self) -> tuple[int, int] | None:
        """As :attr:`g_nlevels` before the rows with missing values were dropped (metafor's ``g.nlevels.f``)."""
        return None if self.group_g is None else self.group_g.nlevels_f

    @property
    def g_levels_f(self) -> tuple[list[str], list[str]] | None:
        """The levels of the inner and outer factor of the ``tau2``/``rho`` term before the rows with missing values were dropped (``g.levels.f``)."""
        return None if self.group_g is None else self.group_g.levels_f

    @property
    def g_levels_k(self) -> pandas.Series[float] | None:
        """Number of observations at each level of the inner factor of the ``tau2``/``rho`` term, named by the levels (metafor's ``g.levels.k``)."""
        return named_counts(self.group_g)

    @property
    def g_levels_comb_k(self) -> FloatArray | None:
        """Number of observations at each combination of inner levels of the ``tau2``/``rho`` term (metafor's ``g.levels.comb.k``)."""
        return None if self.group_g is None else self.group_g.levels_comb_k

    @property
    def h_nlevels(self) -> tuple[int, int] | None:
        """As :attr:`g_nlevels` for the ``gamma2``/``phi`` term (metafor's ``h.nlevels``)."""
        return None if self.group_h is None else self.group_h.nlevels

    @property
    def h_nlevels_f(self) -> tuple[int, int] | None:
        """As :attr:`g_nlevels_f` for the ``gamma2``/``phi`` term (metafor's ``h.nlevels.f``)."""
        return None if self.group_h is None else self.group_h.nlevels_f

    @property
    def h_levels_f(self) -> tuple[list[str], list[str]] | None:
        """As :attr:`g_levels_f` for the ``gamma2``/``phi`` term (metafor's ``h.levels.f``)."""
        return None if self.group_h is None else self.group_h.levels_f

    @property
    def h_levels_k(self) -> pandas.Series[float] | None:
        """As :attr:`g_levels_k` for the ``gamma2``/``phi`` term (metafor's ``h.levels.k``)."""
        return named_counts(self.group_h)

    @property
    def h_levels_comb_k(self) -> FloatArray | None:
        """As :attr:`g_levels_comb_k` for the ``gamma2``/``phi`` term (metafor's ``h.levels.comb.k``)."""
        return None if self.group_h is None else self.group_h.levels_comb_k

    def dimnames_of(self, group: metafopy.models.mv_structs.Group | None, struct: str, names: tuple[str, ...], size: int) -> tuple[str, ...]:
        if group is None:
            return ()
        if struct in metafopy.models.mv_structs.SPATIAL:
            return tuple(str(i + 1) for i in range(size))
        if struct in metafopy.models.mv_structs.GENERAL:
            return names[:-1]
        return tuple(group.levels_f[0])

    @property
    def G_dimnames(self) -> tuple[str, ...]:
        """The row and column names of ``G`` (the levels of the inner factor, or the inner variables for ``GEN``/``GDIAG``), as metafor sets them."""
        return self.dimnames_of(self.group_g, self.struct[0], self.g_names, 0 if self.G is None else self.G.shape[0])

    @property
    def H_dimnames(self) -> tuple[str, ...]:
        """The row and column names of ``H``, as :attr:`G_dimnames`."""
        return self.dimnames_of(self.group_h, self.struct[1], self.h_names, 0 if self.H is None else self.H.shape[0])

    @property
    def G_frame(self) -> pandas.DataFrame | None:
        """``G`` as a data frame with its row and column names, ``None`` without a ``tau2``/``rho`` term."""
        if self.G is None:
            return None
        names = list(self.G_dimnames)
        return pandas.DataFrame(self.G, index=names, columns=names)

    @property
    def H_frame(self) -> pandas.DataFrame | None:
        """``H`` as a data frame with its row and column names, ``None`` without a ``gamma2``/``phi`` term."""
        if self.H is None:
            return None
        names = list(self.H_dimnames)
        return pandas.DataFrame(self.H, index=names, columns=names)

    @property
    def weighted(self) -> bool:
        """Whether the model is fitted by weighting (always ``True`` for ``rma_mv``), metafor's ``weighted``."""
        return True

    @property
    def data(self) -> pandas.DataFrame | None:
        """The data frame the model was fitted to (``None`` if the model was fitted to vectors), metafor's ``data``."""
        data = self.spec.get('data')
        return data if isinstance(data, pandas.DataFrame) else None

    @property
    def sparse(self) -> bool:
        """Whether sparse matrices were used, metafor's ``sparse``."""
        return bool(self.spec.get('sparse', False))

    @property
    def dist(self) -> tuple[typing.Any, typing.Any]:
        """The distance measures of the ``'SP'`` structures (the ``dist`` argument, two entries), metafor's ``dist``."""
        dist = self.spec.get('dist', 'euclidean')
        pair = (dist[0], dist[-1]) if isinstance(dist, list | tuple) else (dist, dist)
        methods = ('euclidean', 'maximum', 'manhattan', 'gcd')

        def full(value: typing.Any) -> typing.Any:
            """A name given in short is kept in full, as ``match.arg`` does in ``rma.mv``."""
            hits = [m for m in methods if isinstance(value, str) and m.startswith(value)] if isinstance(value, str) else []
            return hits[0] if len(hits) == 1 else value

        return (full(pair[0]), full(pair[1]))

    @property
    def random(self) -> list[str] | None:
        """The ``random`` formulas as given (``None`` without any), metafor's ``random``."""
        random = self.spec.get('random')
        if random is None:
            return None
        return [random] if isinstance(random, str) else [str(f) for f in random]

    @property
    def R(self) -> typing.Any:
        """The known correlation matrices given as ``R`` (``None`` without any), metafor's ``R``."""
        return self.spec.get('R')

    @property
    def formulas(self) -> tuple[str | None, str | None]:
        """The two ``~ inner | outer`` formulas of the model (``None`` for a missing one), metafor's ``formulas``."""
        return (formula_text(self.group_g), formula_text(self.group_h))

    @property
    def withS(self) -> bool:
        """Whether the model has ``~ 1 | id`` terms, metafor's ``withS``."""
        return self.with_s

    @property
    def withG(self) -> bool:
        """Whether the model has a first ``~ inner | outer`` term, metafor's ``withG``."""
        return self.with_g

    @property
    def withH(self) -> bool:
        """Whether the model has a second ``~ inner | outer`` term, metafor's ``withH``."""
        return self.with_h

    @property
    def withR(self) -> bool:
        """Whether known correlation matrices were given, metafor's ``withR``."""
        return self.with_r

    @property
    def s_nlevels_f(self) -> tuple[int, ...]:
        """The number of levels of each ``~ 1 | id`` factor before the removal of studies with missing values, metafor's ``s.nlevels.f``."""
        return tuple(len(levels) for levels in self.s_levels_f)

    @property
    def Z_G1(self) -> Mat | None:
        """The design matrix of the outer factor of the first ``~ inner | outer`` term, metafor's ``Z.G1``."""
        return None if self.group_g is None else self.group_g.Z1

    @property
    def Z_G2(self) -> Mat | None:
        """The design matrix of the inner factor of the first ``~ inner | outer`` term, metafor's ``Z.G2``."""
        return None if self.group_g is None else self.group_g.Z2

    @property
    def Z_H1(self) -> Mat | None:
        """The design matrix of the outer factor of the second ``~ inner | outer`` term, metafor's ``Z.H1``."""
        return None if self.group_h is None else self.group_h.Z1

    @property
    def Z_H2(self) -> Mat | None:
        """The design matrix of the inner factor of the second ``~ inner | outer`` term, metafor's ``Z.H2``."""
        return None if self.group_h is None else self.group_h.Z2

    @property
    def k_eff(self) -> int:
        """Number of effective observations (the rows used in the fit), metafor's ``k.eff``."""
        return self.k

    def vcov(self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> pandas.DataFrame:
        """Variance-covariance matrix of the coefficients, or with ``type`` of the observed effects; see :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov_frame(self, type, na_action)

    def fitted(self, *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Fitted values ``X beta``.

        ``na_action`` is ``'omit'`` (the rows used in fitting), ``'exclude'``, ``'pass'`` or ``'fail'``, as in :func:`metafopy.residuals`.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        return metafopy.methods.fitted.fitted_values(self, na_action)

    def residuals(self, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def fit_stats_for(self, REML: bool | None) -> metafopy.models.rma_uni.FitStats:
        if REML is None:
            REML = self.method == 'REML'
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def weight_matrix(self) -> FloatArray:
        """``W``: the user-supplied weight matrix or ``M^-1``."""
        if self.W is not None:
            return numpy.asarray(self.W.toarray() if scipy.sparse.issparse(self.W) else self.W)
        M = numpy.asarray(self.M.toarray() if scipy.sparse.issparse(self.M) else self.M)
        return numpy.asarray(scipy.linalg.cho_solve(scipy.linalg.cho_factor(M, lower=True), numpy.eye(M.shape[0])))

    def hat_matrix(self) -> FloatArray:
        """``X (X'WX)^-1 X'W`` (metafor ``hatvalues.rma.mv(type="matrix")``)."""
        W = self.weight_matrix()
        stXWX = numpy.linalg.inv(self.X.T @ W @ self.X)
        return numpy.asarray(self.X @ stXWX @ self.X.T @ W)

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    def hatvalues(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', na_action: str = 'omit') -> FloatArray:
        """Diagonal of the hat matrix (``type='diagonal'``) or the hat matrix itself (``'matrix'``), as metafor's ``hatvalues.rma.mv``.

        With ``na_action='exclude'`` or ``'pass'`` rows (and columns) left out because of missing values are NaN.
        See https://github.com/wviechtb/metafor/blob/master/R/hatvalues.rma.mv.r.

        Raises:
            metafopy.errors.InputError: an unknown ``type`` or ``na_action``, or missing values with ``na_action='fail'``.
        """
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if na_action not in ('omit', 'exclude', 'pass', 'fail'):
            raise metafopy.errors.InputError(f'unknown na_action {na_action!r}')
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if type not in ('diagonal', 'matrix'):
            raise metafopy.errors.InputError("type must be 'diagonal' or 'matrix'")
        if na_action == 'fail' and not self.not_na.all():
            raise metafopy.errors.InputError('Missing values in results.')
        H = self.hat_matrix()
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        pad = na_action in ('exclude', 'pass')
        if type == 'matrix':
            if not pad:
                return H
            full = numpy.full((self.not_na.shape[0], self.not_na.shape[0]), numpy.nan)
            full[numpy.ix_(self.not_na, self.not_na)] = H
            return full
        h = numpy.diag(H).copy()
        h[h > 1 - 10 * float(numpy.finfo(float).eps)] = 1.0
        if not pad:
            return h
        out = numpy.full(self.not_na.shape[0], numpy.nan)
        out[self.not_na] = h
        return out

    def weights_(
        self, type: typing.Literal['diagonal', 'matrix', 'rowsum'] = 'diagonal', na_action: metafopy.models.rma_uni.NAActionAll = 'omit'
    ) -> FloatArray:
        """Percentage weights (``'diagonal'``), the weight matrix ``W`` (``'matrix'``) or the percentage row sums of ``W`` (``'rowsum'``, intercept-only).

        Raises:
            metafopy.errors.InputError: ``type='rowsum'`` for a model with moderators, an invalid argument, or ``na_action='fail'`` with rows left out.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.weights

        return metafopy.methods.weights.weights_mv(self, type, na_action)

    # -- delegated methods ----------------------------------------------------

    def predict(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.predict

        return metafopy.methods.predict.predict(self, *args, **kwargs)

    def confint(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.confint

        return metafopy.methods.confint.confint(self, *args, **kwargs)

    def anova(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.anova

        return metafopy.methods.anova.anova(self, *args, **kwargs)

    def vif(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vif

        return metafopy.methods.vif.vif(self, *args, **kwargs)

    # -- printing -------------------------------------------------------------

    def summary(
        self,
        digits: int | collections.abc.Mapping[str, int] | None = None,
        showfit: bool = True,
        *,
        signif_stars: bool | None = None,
        signif_legend: bool | None = None,
    ) -> str:
        """The printout with the fit statistics table, as metafor's ``summary()``; ``str(res)`` leaves it out like ``print()``.

        ``digits`` defaults to the ``digits`` option, else to the ``digits`` the model was fitted with; ``signif_stars`` and ``signif_legend``
        as for :meth:`metafopy.models.rma_uni.RMAResult.summary`.
        """
        # auditme: accept[in-function-import] the printing module needs this class, so the import is deferred to avoid a cycle
        import metafopy.models.mv_print

        stars, legend = metafopy.mfopt.option_signif(signif_stars, signif_legend)
        return metafopy.models.fit_methods.SummaryText(
            metafopy.mfopt.option_space(
                metafopy.models.mv_print.format_result(self, metafopy.mfopt.option_digits(digits, self.digits), showfit, stars, legend)
            ),
            self,
        )

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<RMAMVResult {self.method} k={self.k} p={self.p}>'
