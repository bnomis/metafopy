"""Result of a location-scale model (metafor's class ``rma.ls``, fitted by ``rma.uni(..., scale=)``)."""

import dataclasses
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.linalg
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.fit_methods
import metafopy.models.rma_uni

__all__ = ['LSControl', 'RMALSResult']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

DBL_EPS = numpy.finfo(float).eps


@dataclasses.dataclass(frozen=True, slots=True)
class LSControl:
    """Control settings for location-scale models (names follow metafor with ``.`` -> ``_``).

    ``optimizer`` is ``'nlminb'`` (the default, mapped to SciPy's BFGS or, for the
    identity link, SLSQP), ``'BFGS'``, ``'L-BFGS-B'``, ``'Nelder-Mead'``,
    ``'constrOptim'`` or ``'SLSQP'``; ``hessian_d``, ``hessian_eps`` and
    ``hessian_r`` are the arguments of ``numDeriv::hessian``'s Richardson method
    (metafor sets ``r = 8``).
    """

    alpha_init: numpy.typing.ArrayLike | None = None
    alpha_max: numpy.typing.ArrayLike = numpy.inf
    alpha_min: numpy.typing.ArrayLike = -numpy.inf
    beta_init: numpy.typing.ArrayLike | None = None
    evtol: float = 1e-7
    hes_alpha_fix: bool = False
    hes_beta_fix: bool = False
    hessian_d: float = 0.1
    hessian_eps: float = 1e-4
    hessian_r: int = 8
    maxiter: int = 5000
    optimizer: str = 'nlminb'
    REMLf: bool = True
    scaleZ: bool = True  # noqa: N815  mirrors the metafor control argument name
    tau2_max: float = numpy.inf
    tau2_min: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RMALSResult(metafopy.models.fit_methods.FitMethods):
    """A fitted location-scale model.

    The location part (``beta``, ``se``, ..., ``vb``) and the scale part
    (``alpha``, ``se_alpha``, ..., ``va``) follow metafor's names with ``.`` replaced
    by ``_``. ``tau2`` holds the fitted heterogeneity of every study used in the fit.
    ``btt`` and ``att`` are 0-based coefficient indices.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    coef_names: tuple[str, ...]
    alpha: FloatArray
    va: FloatArray
    se_alpha: FloatArray
    zval_alpha: FloatArray
    pval_alpha: FloatArray
    ci_lb_alpha: FloatArray
    ci_ub_alpha: FloatArray
    alpha_names: tuple[str, ...]
    alpha_fix: BoolArray
    tau2: FloatArray
    k: int
    p: int
    q: int
    m: int
    m_alpha: int
    btt: tuple[int, ...]
    att: tuple[int, ...]
    QE: float
    QEp: float
    QM: float
    QMp: float
    QMdf: tuple[int, float | None]
    QS: float
    QSp: float
    QSdf: tuple[int, float | None]
    I2: FloatArray
    H2: FloatArray
    vt: float
    fit_stats_ml: metafopy.models.rma_uni.FitStats
    fit_stats_reml: metafopy.models.rma_uni.FitStats
    parms: int
    method: str
    test: metafopy.models.rma_uni.Test
    level: float
    ddf: int | None
    ddf_alpha: int | None
    s2w: float
    weighted: bool
    weights: FloatArray | None
    link: str
    optbeta: bool
    beta_fix: BoolArray
    int_only: bool
    int_incl: bool
    Z_int_only: bool
    Z_int_incl: bool
    allvipos: bool
    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    Z: FloatArray
    slab: tuple[str, ...]
    data: metafopy.metadata.MetaData = dataclasses.field(repr=False)
    control: LSControl = dataclasses.field(repr=False)
    vcovmat: FloatArray | None = None
    H: FloatArray | None = None
    tau2_fix: bool = False
    measure: str = 'GEN'
    formula_scale: str | None = None
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)

    @property
    def alphas(self) -> int:
        """Number of scale coefficients (metafor's ``alphas``)."""
        return self.q

    # -- accessors ---------------------------------------------------------

    def coef(self, type: typing.Literal['alpha', 'beta'] | None = None) -> pandas.Series | dict[str, pandas.Series]:
        """Location (``'beta'``) and/or scale (``'alpha'``) coefficients, indexed by name."""
        beta = pandas.Series(self.beta, index=list(self.coef_names), name='estimate')
        if type == 'beta':
            return beta
        alpha = pandas.Series(self.alpha, index=list(self.alpha_names), name='estimate')
        if type == 'alpha':
            return alpha
        return {'beta': beta, 'alpha': alpha}

    def vcov(self, type: typing.Literal['alpha', 'beta', 'fixed'] = 'fixed') -> pandas.DataFrame | dict[str, pandas.DataFrame]:
        """Variance-covariance matrix of the location and/or scale coefficients."""
        names = list(self.coef_names)
        anames = list(self.alpha_names)
        vb = pandas.DataFrame(self.vb, index=names, columns=names)
        va = pandas.DataFrame(self.va, index=anames, columns=anames)
        if type == 'beta':
            return vb
        if type == 'alpha':
            return va
        return {'beta': vb, 'alpha': va}

    def fitted(self) -> dict[str, FloatArray]:
        """Fitted location values ``X beta`` and scale values ``Z alpha`` (the latter on the link scale)."""
        return {'location': self.X @ self.beta, 'scale': self.Z @ self.alpha}

    def residuals(self, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def logLik(self, REML: bool | None = None) -> float:
        """Log-likelihood; REML unless the model was fitted with ML (as in metafor)."""
        return self.fit_stats(REML).ll

    def deviance(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).dev

    def AIC(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).AIC

    def BIC(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).BIC

    def AICc(self, REML: bool | None = None) -> float:
        return self.fit_stats(REML).AICc

    def fit_stats(self, REML: bool | None) -> metafopy.models.rma_uni.FitStats:
        if REML is None:
            REML = self.method == 'REML'
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def fitstats(self, REML: bool | None = None) -> pandas.Series:
        """``logLik``, ``deviance``, ``AIC``, ``BIC``, ``AICc`` as a Series."""
        st = self.fit_stats(REML)
        return pandas.Series(
            st.as_tuple(),
            # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
            index=['logLik', 'deviance', 'AIC', 'BIC', 'AICc'],
            name='REML' if (REML if REML is not None else self.method == 'REML') else 'ML',
        )

    def weights_(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Percentage weights of the estimates in the model fit (``type='diagonal'``) or the weight matrix (``'matrix'``), metafor's ``weights()``.

        ``na_action`` is ``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'`` as in :func:`metafopy.residuals`.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.weights

        return metafopy.methods.weights.weights_uni(self, type, na_action)

    def hatvalues(self) -> FloatArray:
        """Diagonal of the hat matrix."""
        X = self.X
        if self.weighted:
            w = 1.0 / (self.vi + self.tau2) if self.weights is None else self.weights
            H = X @ metafopy.core.linalg.invcalc(X, w) @ (X.T * w)
        else:
            H = X @ metafopy.core.linalg.invcalc(X, numpy.ones(self.k)) @ X.T
        h = numpy.diag(H).copy()
        h[h > 1 - 10 * DBL_EPS] = 1.0
        return h

    # -- delegated methods (defined in metafopy.methods) ----------------------

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

    def blup(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.blup

        return metafopy.methods.blup.blup(self, *args, **kwargs)

    def ranef(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.ranef

        return metafopy.methods.ranef.ranef(self, *args, **kwargs)

    def vif(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vif

        return metafopy.methods.vif.vif(self, *args, **kwargs)

    def permutest(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.permutest

        return metafopy.methods.permutest.permutest(self, *args, **kwargs)

    def profile(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.profile

        return metafopy.methods.profile.profile(self, *args, **kwargs)

    # -- printing ----------------------------------------------------------

    def summary(self, digits: int | None = None, showfit: bool = False) -> str:
        # auditme: accept[in-function-import] the printer imports this module for its type
        import metafopy.models.rma_ls_print

        return metafopy.mfopt.option_space(metafopy.models.rma_ls_print.format_result(self, metafopy.mfopt.option_digits(digits), showfit))

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f'<RMALSResult {self.method} k={self.k} p={self.p} q={self.q}>'
