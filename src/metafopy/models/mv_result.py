"""The fitted ``rma.mv`` model (:class:`RMAMVResult`)."""

import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.sparse

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


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RMAMVResult(metafopy.models.fit_methods.FitMethods):
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
    def g_levels_k(self) -> FloatArray | None:
        """Number of observations at each level of the inner factor of the ``tau2``/``rho`` term (metafor's ``g.levels.k``)."""
        return None if self.group_g is None else self.group_g.levels_k

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
    def h_levels_k(self) -> FloatArray | None:
        """As :attr:`g_levels_k` for the ``gamma2``/``phi`` term (metafor's ``h.levels.k``)."""
        return None if self.group_h is None else self.group_h.levels_k

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

    def fit_stats(self, REML: bool | None) -> metafopy.models.rma_uni.FitStats:
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
        return metafopy.mfopt.option_space(
            metafopy.models.mv_print.format_result(self, metafopy.mfopt.option_digits(digits, self.digits), showfit, stars, legend)
        )

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<RMAMVResult {self.method} k={self.k} p={self.p}>'
