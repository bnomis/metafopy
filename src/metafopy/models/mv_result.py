"""The fitted ``rma.mv`` model (:class:`RMAMVResult`)."""

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
    ``beta_est`` flags them, and ``vccon`` holds the group ids of constrained variance components.
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

    # -- accessors -------------------------------------------------------

    def coef(self) -> pandas.Series:
        return pandas.Series(self.beta, index=list(self.coef_names), name='estimate')

    def vcov(self) -> pandas.DataFrame:
        names = list(self.coef_names)
        return pandas.DataFrame(self.vb, index=names, columns=names)

    def fitted(self) -> FloatArray:
        return numpy.asarray(self.X @ self.beta)

    def residuals(self) -> FloatArray:
        return numpy.asarray(self.yi - self.fitted())

    def fit_stats(self, REML: bool | None) -> metafopy.models.rma_uni.FitStats:
        if REML is None:
            REML = self.method == 'REML'
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def logLik(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).ll

    def deviance(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).dev

    def AIC(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).AIC

    def BIC(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).BIC

    def AICc(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).AICc

    def fitstats(self, REML: bool | None = None) -> pandas.Series:
        st = self.fit_stats(REML)
        label = 'REML' if (REML if REML is not None else self.method == 'REML') else 'ML'
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        return pandas.Series(st.as_tuple(), index=['logLik', 'deviance', 'AIC', 'BIC', 'AICc'], name=label)

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

    def hatvalues(self) -> FloatArray:
        """Diagonal of the hat matrix."""
        h = numpy.diag(self.hat_matrix()).copy()
        h[h > 1 - 10 * float(numpy.finfo(float).eps)] = 1.0
        return h

    def weights_(self, type: typing.Literal['diagonal', 'rowsum'] = 'diagonal') -> FloatArray:
        """Percentage weights: the diagonal of ``W`` (or its row sums for intercept-only models) scaled to sum to 100.

        Raises:
            metafopy.errors.InputError: ``type='rowsum'`` for a model with moderators.
        """
        W = self.weight_matrix()
        if type == 'rowsum':
            if not self.int_only:
                raise metafopy.errors.InputError('Row-sum weights are only meaningful for intercept-only models.')
            wi = W.sum(axis=1)
        else:
            wi = numpy.diag(W)
        return numpy.asarray(wi / wi.sum() * 100)

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

    # -- printing -------------------------------------------------------------

    def summary(self, digits: int | None = None, showfit: bool = False) -> str:
        # auditme: accept[in-function-import] the printing module needs this class, so the import is deferred to avoid a cycle
        import metafopy.models.mv_print

        return metafopy.mfopt.option_space(metafopy.models.mv_print.format_result(self, metafopy.mfopt.option_digits(digits), showfit))

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f'<RMAMVResult {self.method} k={self.k} p={self.p}>'
