"""Result of a location-scale model (metafor's class ``rma.ls``, fitted by ``rma.uni(..., scale=)``)."""

import collections.abc
import dataclasses
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.labelled
import metafopy.core.linalg
import metafopy.errors
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

    ``polish_converged=False`` skips the Nelder-Mead polish after a converged BFGS run (used by the permutation test).
    ``optimizer`` is ``'nlminb'`` (the default, mapped to SciPy's BFGS or, for the identity link, ``constrOptim``), any other
    optimiser of metafor's ``rma.uni`` (``'optim'`` with ``optmethod``, ``'BFGS'``, ``'CG'``, ``'L-BFGS-B'``, ``'Nelder-Mead'``,
    ``'nlm'``, ``'uobyqa'``, ``'newuoa'``, ``'bobyqa'``, ``'nloptr'``, ``'hjk'``, ``'nmk'``, ``'mads'``, ``'ucminf'``, ``'lbfgsb3c'``,
    ``'subplex'``, ``'BBoptim'``, ``'optimParallel'``, ``'Rcgmin'``, ``'Rvmmin'``, and for the identity link ``'constrOptim'``,
    ``'solnp'``, ``'constrOptim.nl'``/``'alabama'`` and ``'auglag'``), mapped as :mod:`metafopy.models.sel_opt` maps them. ``opt_ctrl``
    holds the optimiser settings given in ``control`` (``maxit``, ``reltol``, ``abstol``, ``ndeps``, ``tol``). ``ncpus`` larger than 1
    selects ``'optimParallel'`` as in metafor, but the likelihood is evaluated serially (``parallel`` and ``cl`` are accepted and
    ignored). ``hessian_d``, ``hessian_eps`` and
    ``hessian_r`` are the arguments of ``numDeriv::hessian``'s Richardson method
    (metafor sets ``r = 8``, whose eight Richardson steps amplify round-off in the log-likelihood to a relative 1e-4 or more in the standard errors; metafopy
    defaults to ``r = 4``, which agrees with metafor's values to about 1e-5, see ``docs/differences.md``).
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
    hessian_r: int = 4
    maxiter: int = 5000
    optimizer: str = 'nlminb'
    optmethod: str = 'BFGS'
    opt_ctrl: collections.abc.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    ncpus: int = 1
    cl: typing.Any = None
    parallel: collections.abc.Mapping[str, typing.Any] = dataclasses.field(default_factory=dict)
    polish_converged: bool = True
    REMLf: bool = True
    scaleZ: bool = True  # noqa: N815  mirrors the metafor control argument name
    tau2_max: float = numpy.inf
    tau2_min: float = 0.0
    ignored: dict[str, typing.Any] = dataclasses.field(default_factory=dict, compare=False, repr=False)
    """The entries other functions of metafor read (``vc.min``, ``stepadj``, ...) and those given under another name (``tol`` is in ``opt_ctrl``),
    kept as R keeps them in ``res$control``; read them as attributes."""

    def __getattr__(self, name: str) -> typing.Any:
        ignored = object.__getattribute__(self, 'ignored')
        for key, value in ignored.items():
            if key.replace('.', '_') == name:
                return value
        raise AttributeError(name)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class RMALSResult(metafopy.models.fit_methods.DataFields, metafopy.models.fit_methods.FitMethods):
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
    Z_f: FloatArray | None = dataclasses.field(default=None, repr=False)
    digits: int = 4
    call: metafopy.models.call.Call | None = dataclasses.field(default=None, kw_only=True, repr=False, compare=False)
    coef_na: tuple[bool, ...] = ()
    coef_na_Z: tuple[bool, ...] = ()  # noqa: N815  mirrors metafor's name
    X_f: FloatArray = dataclasses.field(default_factory=lambda: numpy.empty((0, 0)), repr=False)
    add: float = 0.5
    to: str = 'only0'
    drop00: bool = False

    def __post_init__(self) -> None:
        # R names the rows of the model matrices by the rows of the data, their columns by the coefficients and the Hessian by both sets
        rows, rows_f = self.data.rownames, self.data.rownames_f
        design = metafopy.core.labelled.design_matrix
        object.__setattr__(self, 'X', design(self.X, rows, self.coef_names))
        object.__setattr__(self, 'Z', design(self.Z, rows, self.alpha_names))
        object.__setattr__(self, 'X_f', design(self.X_f, rows_f, self.coef_names))
        if self.Z_f is not None:
            object.__setattr__(self, 'Z_f', design(self.Z_f, rows_f, self.alpha_names))
        if self.H is not None:
            both = [*('X' + n for n in self.coef_names), *('Z' + n for n in self.alpha_names)]
            names = both if self.H.shape[0] == len(both) else list(self.alpha_names)
            object.__setattr__(self, 'H', design(self.H, names, names))

    @property
    def alphas(self) -> int:
        """Number of scale coefficients (metafor's ``alphas``)."""
        return self.q

    @property
    def model(self) -> str:
        """The kind of model, ``'rma.ls'`` (metafor's ``model``)."""
        return 'rma.ls'

    @property
    def se_tau2(self) -> float:
        """Not available for location-scale models (``NaN``, as metafor stores ``NA``)."""
        return math.nan

    @property
    def R2(self) -> None:
        """Not available for location-scale models (``None``, as metafor stores ``NULL``)."""
        return None

    @property
    def randhet(self) -> bool:
        """Whether random heterogeneity is modelled (always ``False`` here, metafor's ``randhet``)."""
        return False

    @property
    def Z_intercept(self) -> bool:
        """Whether the scale model has an intercept, metafor's ``Z.intercept``."""
        return self.Z_int_incl

    @property
    def dfs(self) -> int | None:
        """The degrees of freedom of the tests of the location coefficients (``None`` for the z-test), metafor's ``dfs``."""
        return self.ddf

    @property
    def tau2_f(self) -> FloatArray:
        """The amount of residual heterogeneity of each study after subsetting, ``NaN`` for those left out of the fit, metafor's ``tau2.f``."""
        out = numpy.full(self.data.k_f, math.nan)
        out[self.data.not_na] = self.tau2
        return out

    # -- accessors ---------------------------------------------------------

    def coef(self, type: typing.Literal['alpha', 'beta'] | None = None) -> pandas.Series | dict[str, pandas.Series]:
        """Location (``'beta'``) and/or scale (``'alpha'``) coefficients, indexed by name."""
        beta = pandas.Series(self.beta, index=list(self.coef_names), name='estimate')
        if type == 'beta':
            return beta
        alpha = pandas.Series(self.alpha, index=list(self.alpha_names), name='estimate')
        if type == 'alpha':
            return alpha
        return metafopy.core.labelled.NamedList({'beta': beta, 'alpha': alpha})

    @typing.overload
    def vcov(
        self,
        type: typing.Literal['beta', 'alpha', 'obs', 'marginal', 'fitted', 'resid'],
        *,
        na_action: metafopy.models.rma_uni.NAActionAll = 'omit',
    ) -> pandas.DataFrame: ...

    @typing.overload
    def vcov(
        self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit'
    ) -> pandas.DataFrame | dict[str, pandas.DataFrame]: ...

    def vcov(
        self, type: metafopy.models.rma_uni.VcovType = 'fixed', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit'
    ) -> pandas.DataFrame | dict[str, pandas.DataFrame]:
        """Variance-covariance matrices; ``'fixed'`` (the default) gives ``{'beta': ..., 'alpha': ...}``, the other types follow :func:`metafopy.vcov`."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.vcov

        return metafopy.methods.vcov.vcov(self, type, na_action=na_action)

    def fitted(self, *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> dict[str, FloatArray]:
        """Fitted location values ``X beta`` and scale values ``Z alpha`` (the latter on the link scale); ``na_action`` is as for ``rma_uni`` fits."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        return metafopy.core.labelled.NamedList(
            {'location': metafopy.methods.fitted.fitted_values(self, na_action), 'scale': metafopy.methods.fitted.scale_values(self, na_action)}
        )

    def residuals(self, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Residuals; see :func:`metafopy.residuals` (``type`` is ``'response'``, ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``)."""
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.residuals

        return metafopy.methods.residuals.residuals(self, type, na_action=na_action)

    def fit_stats_for(self, REML: bool | None) -> metafopy.models.rma_uni.FitStats:
        if REML is None:
            REML = self.method == 'REML'
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def weights_(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Percentage weights of the estimates in the model fit (``type='diagonal'``) or the weight matrix (``'matrix'``), metafor's ``weights()``.

        ``na_action`` is ``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'`` as in :func:`metafopy.residuals`.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.weights

        return metafopy.methods.weights.weights_uni(self, type, na_action)

    def hat_matrix(self) -> FloatArray:
        """The full hat matrix ``X (X'WX)^-1 X'W``."""
        X = self.X
        if self.weighted:
            w = 1.0 / (self.vi + self.tau2) if self.weights is None else self.weights
            return X @ metafopy.core.linalg.invcalc(X, w) @ (X.T * w)
        return X @ metafopy.core.linalg.invcalc(X, numpy.ones(self.k)) @ X.T

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    def hatvalues(self, type: typing.Literal['diagonal', 'matrix'] = 'diagonal', *, na_action: metafopy.models.rma_uni.NAActionAll = 'omit') -> FloatArray:
        """Diagonal of the hat matrix (``type='diagonal'``) or the hat matrix itself (``'matrix'``), as metafor's ``hatvalues()``.

        With ``na_action='exclude'`` or ``'pass'`` the studies left out because of missing values are NaN rows (and columns); ``'fail'`` stops if any were.

        Raises:
            metafopy.errors.InputError: ``type`` is not ``'diagonal'`` or ``'matrix'``, ``na_action`` is unknown, or studies are missing with ``'fail'``.
        """
        # auditme: accept[in-function-import] metafopy.methods imports this module, so the import is deferred to avoid a cycle
        import metafopy.methods.fitted

        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if type not in ('diagonal', 'matrix'):
            raise metafopy.errors.InputError("type must be 'diagonal' or 'matrix'")
        H = self.hat_matrix()
        if type == 'matrix':
            return metafopy.methods.fitted.hat_values(self, H, na_action)
        h = numpy.diag(H).copy()
        h[h > 1 - 10 * DBL_EPS] = 1.0
        return metafopy.methods.fitted.hat_values(self, h, na_action)

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
        # auditme: accept[in-function-import] the printer imports this module for its type
        import metafopy.models.rma_ls_print

        stars, legend = metafopy.mfopt.option_signif(signif_stars, signif_legend)
        return metafopy.models.fit_methods.SummaryText(
            metafopy.mfopt.option_space(
                metafopy.models.rma_ls_print.format_result(self, metafopy.mfopt.option_digits(digits, self.digits), showfit, stars, legend)
            ),
            self,
        )

    def __str__(self) -> str:
        return self.summary(showfit=False)

    def __repr__(self) -> str:
        return f'<RMALSResult {self.method} k={self.k} p={self.p} q={self.q}>'
