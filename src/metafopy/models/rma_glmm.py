"""Meta-analysis with generalized linear mixed-effects models (metafor's ``rma.glmm``).

Odds ratios (``measure='OR'``) of 2x2 tables and incidence rate ratios
(``'IRR'``) of two-group rate data are modelled with unconditional or
conditional models, single-group log odds (``'PLO'``) and log incidence rates
(``'IRLN'``) with logistic and Poisson mixed-effects models, following
Stijnen, Hamza and Ozdemir (2010), doi:10.1002/sim.3767. The measures that
metafor documents as experimental (risk ratios ``'RR'`` and risk differences
``'RD'`` of 2x2 tables, proportions ``'PR'`` and log proportions ``'PLN'``) use
binomial models with a log or identity link and give metafor's warning. See
https://wviechtb.github.io/metafor/reference/rma.glmm.html and the source at
https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r.

metafor calls ``lme4::glmer()`` for the mixed models, or with
``control=list(package='GLMMadaptive')`` or ``'glmmTMB'`` the functions
``GLMMadaptive::mixed_model()`` (adaptive Gauss-Hermite quadrature, Rizopoulos
2023, https://drizopoulos.github.io/GLMMadaptive/) and ``glmmTMB::glmmTMB()``
(Laplace approximation, Brooks et al. 2017, doi:10.32614/RJ-2017-066). Here
:mod:`metafopy.core.glmm` maximises the same likelihood (adaptive
Gauss-Hermite quadrature or the Laplace approximation) for all three, so
estimates agree with metafor's to the (loose) convergence tolerance of the
package's optimiser.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.glm
import metafopy.core.linalg
import metafopy.core.nchg
import metafopy.core.verbose
import metafopy.errors
import metafopy.metadata
import metafopy.mfopt
import metafopy.models.call
import metafopy.models.control_names
import metafopy.models.glmm_fit
import metafopy.models.glmm_result
import metafopy.models.rma_uni
import metafopy.models.tables

__all__ = ['rma_glmm']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
SUPPORTED = ('OR', 'IRR', 'PLO', 'IRLN')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
EXPERIMENTAL = ('PR', 'RR', 'RD', 'PLN')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
UNCONDITIONAL_ONLY = ('RR', 'RD')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
MODELS = ('UM.FS', 'UM.RS', 'CM.EL', 'CM.AL')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
METHODS = ('FE', 'EE', 'CE', 'ML')


@dataclasses.dataclass(frozen=True, slots=True)
class Design:
    """Model matrices at the three stages of the NA handling (all tables, tables used in fitting, tables with valid ``yi``/``vi``)."""

    X: FloatArray
    X_f: FloatArray
    X_yi: FloatArray
    names: tuple[str, ...]
    int_incl: bool


def make_control(control: metafopy.models.glmm_fit.GLMMControl | collections.abc.Mapping[str, typing.Any] | None) -> metafopy.models.glmm_fit.GLMMControl:
    """Build the control settings; a mapping uses metafor's names (``.`` may be written as ``_``).

    Raises:
        metafopy.errors.InputError: an unknown control setting is given.
    """
    if control is None:
        return metafopy.models.glmm_fit.GLMMControl()
    if isinstance(control, metafopy.models.glmm_fit.GLMMControl):
        return control
    names = {f.name for f in dataclasses.fields(metafopy.models.glmm_fit.GLMMControl)}
    clean = {
        str(key)
        .replace('.', '_')
        .replace('scaleX', 'scale_x')
        .replace('clogitCtrl', 'clogit_ctrl')
        .replace('clogisticCtrl', 'clogistic_ctrl')
        .replace('optCtrl', 'opt_ctrl')
        .replace('hessianCtrl', 'hessian_ctrl'): value
        for key, value in control.items()
    }
    unknown = {name for name in set(clean) - names if not metafopy.models.control_names.read_elsewhere(name)}
    if unknown:
        raise metafopy.errors.InputError(f'unknown control settings: {sorted(unknown)}')
    clean = {name: value for name, value in clean.items() if name in names}
    if 'package' in clean:
        clean['package'] = match_package(clean['package'])
    if 'hesspack' in clean:
        clean['hesspack'] = match_choice(clean['hesspack'], ('numDeriv', 'pracma', 'calculus'), 'hesspack')
    if 'dnchgcalc' in clean and clean['dnchgcalc'] not in metafopy.core.nchg.DNCHG_CALCS:
        raise metafopy.errors.InputError('Unknown dnchgcalc method specified.')
    if 'hessian_ctrl' in clean:
        clean['hessian_ctrl'] = dict(clean['hessian_ctrl'] or {})
    if 'optmethod' in clean:
        clean['optmethod'] = match_optmethod(clean['optmethod'])
    if 'optimizer' in clean:
        optimizer = match_optimizer(clean['optimizer'])
        if optimizer in metafopy.models.glmm_fit.OPTMETHODS:
            # as metafor: optimizer='BFGS' stands for optimizer='optim' with optmethod='BFGS'
            clean['optmethod'] = optimizer
            optimizer = 'optim'
        clean['optimizer'] = optimizer
    return metafopy.models.glmm_fit.GLMMControl(**clean)


def match_package(value: typing.Any) -> str:
    """The package named by ``control$package``; like ``match.arg()`` an unambiguous prefix is enough.

    Raises:
        metafopy.errors.InputError: the value names no package.
    """
    hits = [name for name in metafopy.models.glmm_fit.PACKAGES if isinstance(value, str) and name.startswith(value)] if value else []
    exact = [name for name in hits if name == value]
    if len(exact) == 1:
        return exact[0]
    if len(hits) != 1:
        raise metafopy.errors.InputError('\'arg\' should be one of "lme4", "GLMMadaptive", "glmmTMB"')
    return hits[0]


def match_optimizer(value: typing.Any) -> str:
    """The optimizer named by ``control$optimizer`` (an exact name, or an unambiguous prefix as in ``match.arg()``).

    Raises:
        metafopy.errors.InputError: the value names none of the optimisers in :data:`metafopy.models.glmm_fit.OPTIMIZERS`.
    """
    return match_choice(value, metafopy.models.glmm_fit.OPTIMIZERS, 'optimizer')


def match_optmethod(value: typing.Any) -> str:
    """The ``optim()`` method named by ``control$optmethod`` (like ``match.arg()``).

    Raises:
        metafopy.errors.InputError: the value names none of the methods in :data:`metafopy.models.glmm_fit.OPTMETHODS`.
    """
    return match_choice(value, metafopy.models.glmm_fit.OPTMETHODS, 'optmethod')


def match_choice(value: typing.Any, names: tuple[str, ...], what: str) -> str:
    if isinstance(value, str) and value in names:
        return value
    hits = [name for name in names if isinstance(value, str) and value and name.startswith(value)]
    if len(hits) != 1:
        raise metafopy.errors.InputError(f"'{what}' should be one of {', '.join(repr(n) for n in names)}")
    return hits[0]


def resolve_family(measure: str, link: str | None, family: metafopy.core.glm.GLMFamily | str | None) -> metafopy.core.glm.GLMFamily | None:
    """The family override of ``rma.glmm``'s ``family`` and ``link`` arguments (``None`` for the measure's own).

    As in metafor a ``family`` replaces the family and link of the model (a name stands for the family with its default
    link, like R's ``family = binomial``) and ``link`` changes only the link of the measure's family.

    Raises:
        metafopy.errors.InputError: the family or the link is unknown.
    """
    if family is not None:
        if isinstance(family, metafopy.core.glm.GLMFamily):
            return family
        if family == 'binomial':
            return metafopy.core.glm.binomial()
        if family == 'poisson':
            return metafopy.core.glm.poisson()
        raise metafopy.errors.InputError("Unknown 'family' specified; use 'binomial', 'poisson' or a metafopy.core.glm.GLMFamily.")
    if link is None:
        return None
    return metafopy.core.glm.GLMFamily(metafopy.models.glmm_fit.glm_family(measure), metafopy.core.glm.check_link(link))


def first_of(value: typing.Any) -> typing.Any:
    """metafor uses only the first of several ``add``/``to`` values in ``rma.glmm``."""
    if isinstance(value, str) or not isinstance(value, collections.abc.Sequence):
        return value
    return value[0]


def check_arguments(measure: str | None, model: str | None, method: str, coding: float, test: str) -> str:
    """Validate the argument values and return the model after the ``CM.AL``/``IRR`` and one-group adjustments.

    Raises:
        metafopy.errors.InputError: an argument has an unknown value or the measure cannot be used with the model.
    """
    if measure is None:
        raise metafopy.errors.InputError("Must specify the 'measure' argument.")
    if measure not in SUPPORTED and measure not in EXPERIMENTAL:
        raise metafopy.errors.InputError("Unknown 'measure' specified.")
    if method not in METHODS:
        raise metafopy.errors.InputError("Unknown 'method' specified.")
    if coding not in (0.5, 1, 0):
        raise metafopy.errors.InputError("Unknown 'coding' option specified.")
    if test not in ('z', 't'):
        raise metafopy.errors.InputError("Unknown option specified for the 'test' argument.")
    if model is not None and model not in MODELS:
        raise metafopy.errors.InputError("Unknown 'model' specified.")
    resolved = model or 'UM.FS'
    if measure in metafopy.models.glmm_fit.ONE_GROUP and model is not None:
        warnings.warn("Argument 'model' not relevant for this outcome measure.", stacklevel=3)
    if measure in EXPERIMENTAL:
        warnings.warn("The use of this 'measure' is experimental - treat results with caution.", stacklevel=3)
    if resolved in ('CM.EL', 'CM.AL') and measure in UNCONDITIONAL_ONLY:
        raise metafopy.errors.InputError("Cannot use this measure with model='CM.EL' or model='CM.AL'.")
    return 'CM.EL' if (resolved == 'CM.AL' and measure == 'IRR') else resolved


def build_design(tab: metafopy.models.tables.TableSet, mods_all: FloatArray, names_all: tuple[str, ...]) -> Design:
    """Model matrices for the three NA stages, with redundant columns dropped and the intercept moved first (as ``rma.glmm`` does).

    Raises:
        metafopy.errors.InputError: no columns remain.
    """
    sub = mods_all if tab.subset is None else mods_all[tab.subset]
    X_f = sub
    X = X_f[tab.not_na]
    X_yi = X_f[tab.not_na_yivi]
    names = list(names_all)
    drop = metafopy.core.linalg.drop_redundant(X)
    if drop.any():
        warnings.warn('Redundant predictors dropped from the model.', stacklevel=3)
        X, X_f = X[:, ~drop], X_f[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]
    drop_yi = metafopy.core.linalg.drop_redundant(X_yi) if X_yi.shape[0] > 0 else numpy.zeros(X_yi.shape[1], dtype=numpy.bool_)
    X_yi = X_yi[:, ~drop_yi]
    names, X, X_f, int_incl = move_intercept(names, X, X_f)
    if X_yi.shape[0] > 0:
        _, X_yi, _, _ = move_intercept([f'x{j}' for j in range(X_yi.shape[1])], X_yi, X_yi.copy())
    return Design(X=X, X_f=X_f, X_yi=X_yi, names=tuple(names), int_incl=int_incl)


def move_intercept(names: list[str], X: FloatArray, X_f: FloatArray) -> tuple[list[str], FloatArray, FloatArray, bool]:
    """Put a single intercept column first, removing duplicates (``cbind(intrcpt=1, X[, -int.indx])``)."""
    is_int = numpy.array([metafopy.core.linalg.is_intercept(X[:, j]) for j in range(X.shape[1])], dtype=numpy.bool_)
    if not is_int.any():
        return names, X, X_f, False
    k, k_f = X.shape[0], X_f.shape[0]
    new_names = ['intrcpt', *[n for n, i in zip(names, is_int, strict=True) if not i]]
    return new_names, numpy.column_stack([numpy.ones(k), X[:, ~is_int]]), numpy.column_stack([numpy.ones(k_f), X_f[:, ~is_int]]), True


def heterogeneity_tests(fit: metafopy.models.glmm_fit.ModelFit, k: int, p: int) -> tuple[float, float, float, float, int | None]:
    """Wald-type and likelihood ratio tests of the saturated model's extra coefficients: ``(Wld, p, LRT, p, df)``."""
    nan = math.nan
    if not (k > 1 and fit.qe_converged):
        return nan, nan, nan, nan, None
    if fit.qe_wld is not None:
        wld = fit.qe_wld
    elif fit.vb2_qe.shape[0] > 0:
        wld = wald_statistic(fit.b2_qe, fit.vb2_qe)
    else:
        wld = 0.0
    lrt = -2 * (fit.ll_fe - fit.ll_qe)
    wld = 0.0 if wld <= 0 else wld
    lrt = 0.0 if lrt <= 0 else lrt
    df = k - p
    if df > 0:
        return wld, float(scipy.stats.chi2.sf(wld, df)), lrt, float(scipy.stats.chi2.sf(lrt, df)), df
    return wld, 1.0, lrt, 1.0, df


def wald_statistic(b2: FloatArray, vb2: FloatArray) -> float:
    try:
        chol = numpy.linalg.cholesky(vb2)
    except numpy.linalg.LinAlgError:
        warnings.warn('Cannot invert the Hessian for the saturated model.', stacklevel=4)
        return math.nan
    z = numpy.linalg.solve(chol, b2)
    return float(z @ z)


def i2_h2(tab: metafopy.models.tables.TableSet, X_yi: FloatArray, tau2: float, p: int, i2def: str) -> tuple[float, float, float]:
    """``I^2``, ``H^2`` and the typical sampling variance ``vt`` from the ``yi``/``vi`` of the tables."""
    k_yi = tab.k_yi
    with numpy.errstate(all='ignore'):
        wi = 1.0 / tab.vi
        if i2def == '2':
            vt = float(1.0 / wi.mean())
        else:
            WX = wi[:, None] * X_yi
            P = numpy.diag(wi) - WX @ metafopy.core.linalg.invcalc(X_yi, wi) @ WX.T
            vt = (k_yi - p) / float(numpy.trace(P))
    return 100 * tau2 / (vt + tau2), tau2 / vt + 1, vt


def scale_design(X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, numpy.typing.NDArray[numpy.bool_]]:
    """Rescale the non-dummy columns (``scale()``); returns the new matrix, column means, standard deviations and the dummy flags."""
    is_d = numpy.array([bool(numpy.all((numpy.abs(X[:, j]) < 1e-8) | (numpy.abs(X[:, j] - 1) < 1e-8))) for j in range(X.shape[1])], dtype=numpy.bool_)
    mean_x = X[:, 1:].mean(axis=0)
    sd_x = X[:, 1:].std(axis=0, ddof=1)
    out = X.copy()
    cols = ~is_d
    out[:, cols] = (X[:, cols] - X[:, cols].mean(axis=0)) / X[:, cols].std(axis=0, ddof=1)
    return out, mean_x, sd_x, is_d


def back_transform(
    beta: FloatArray, vb: FloatArray, mean_x: FloatArray, sd_x: FloatArray, is_d: numpy.typing.NDArray[numpy.bool_]
) -> tuple[FloatArray, FloatArray]:
    """Undo the rescaling of the model matrix on the coefficients and their covariance matrix."""
    p = beta.shape[0]
    dummy = is_d[1:]
    mX = numpy.zeros((p, p))
    mX[0, 0] = 1.0
    mX[0, 1:] = -numpy.where(dummy, 0.0, mean_x / sd_x)
    mX[1:, 1:] = numpy.diag(numpy.where(dummy, 1.0, 1.0 / sd_x))
    return mX @ beta, mX @ vb @ mX.T


def omnibus_test(beta: FloatArray, vb: FloatArray, btt: tuple[int, ...]) -> float:
    idx = numpy.array(btt)
    try:
        chol = numpy.linalg.cholesky(vb[numpy.ix_(idx, idx)])
    except numpy.linalg.LinAlgError:
        warnings.warn('Cannot invert the Hessian for the QM-test.', stacklevel=3)
        return math.nan
    z = numpy.linalg.solve(chol, beta[idx])
    return float(z @ z)


def fit_statistics(ll: float, ll_qe: float, parms: int, k_eff: int) -> metafopy.models.rma_uni.FitStats:
    dev = -2 * (ll - ll_qe)
    big = max(k_eff, parms + 2)
    return metafopy.models.rma_uni.FitStats(
        ll=ll,
        dev=dev,
        AIC=-2 * ll + 2 * parms,
        BIC=-2 * ll + parms * math.log(k_eff),
        AICc=-2 * ll + 2 * parms * big / (big - parms - 1),
    )


@metafopy.models.call.records_call
def rma_glmm(
    *,
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    xi: metafopy.metadata.Column | None = None,
    mi: metafopy.metadata.Column | None = None,
    ti: metafopy.metadata.Column | None = None,
    ni: metafopy.metadata.Column | None = None,
    mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None = None,
    measure: str | None = None,
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float | collections.abc.Sequence[float] = 0.5,
    to: str | collections.abc.Sequence[str] = 'only0',
    drop00: bool | collections.abc.Sequence[bool] = True,
    intercept: bool = True,
    model: str | None = None,
    method: str = 'ML',
    coding: float = 0.5,
    cor: bool = False,
    test: str = 'z',
    level: float = 95,
    btt: metafopy.models.rma_uni.Btt = None,
    nAGQ: int = 7,
    verbose: bool | int = False,
    digits: int = 4,
    control: metafopy.models.glmm_fit.GLMMControl | collections.abc.Mapping[str, typing.Any] | None = None,
    na_action: metafopy.metadata.NAAction = 'omit',
    i2def: str = '1',
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
    link: str | None = None,
    family: metafopy.core.glm.GLMFamily | str | None = None,
) -> metafopy.models.glmm_result.GLMMResult:
    """Fit a generalized linear mixed-effects model to binary or count data.

    Mirrors metafor's ``rma.glmm()``; see
    https://wviechtb.github.io/metafor/reference/rma.glmm.html. ``measure`` is
    ``'OR'`` (2x2 tables: ``ai``, ``bi``, ``ci``, ``di`` or ``ai``, ``ci``,
    ``n1i``, ``n2i``), ``'IRR'`` (``x1i``, ``x2i``, ``t1i``, ``t2i``), ``'PLO'``
    (``xi`` and ``mi`` or ``ni``) or ``'IRLN'`` (``xi``, ``ti``). ``model`` is
    ``'UM.FS'`` (the default), ``'UM.RS'``, ``'CM.AL'`` or ``'CM.EL'`` and
    ``method`` ``'ML'`` or ``'FE'`` (``'EE'``, ``'CE'``). Only the first of
    several ``add`` and ``to`` values is used.

    The experimental measures ``'RR'`` and ``'RD'`` (arguments as for ``'OR'``,
    unconditional models only) and ``'PR'`` and ``'PLN'`` (arguments as for
    ``'PLO'``) are fitted as metafor does and give its warning that their use is
    experimental; as in metafor, the log and identity links can put fitted
    probabilities outside (0, 1), in which case fitting fails (``EstimationError``)
    or the estimates are unreliable.

    ``link`` (one of ``'logit'``, ``'probit'``, ``'cauchit'``, ``'cloglog'``, ``'log'``, ``'identity'``, ``'sqrt'``) replaces the
    link of the model and ``family`` (``'binomial'``, ``'poisson'`` or :func:`metafopy.core.glm.binomial` /
    :func:`metafopy.core.glm.poisson`, which carry a link) replaces family and link, like metafor's ``link`` and ``family``
    arguments passed through ``...``. They apply to the unconditional models (``'UM.FS'``, ``'UM.RS'``) and the one-group
    measures, the models that metafor fits with ``glm()`` and ``glmer()``; the conditional models ignore them as in metafor.
    The family must match the data (binomial for ``'OR'``, ``'RR'``, ``'RD'``, ``'PLO'``, ``'PR'``, ``'PLN'``, Poisson for
    ``'IRR'`` and ``'IRLN'``).

    ``control={'optimizer': 'clogit'}`` or ``'clogistic'`` fits the fixed-effects (``method='FE'``) conditional model
    ``'CM.EL'`` for ``'OR'`` with ``survival::clogit()`` or ``Epi::clogistic()`` (implemented in :mod:`metafopy.core.clogit`,
    without building the individual patient data); ``clogitCtrl`` and ``clogisticCtrl`` entries of ``control`` are passed on
    as in metafor (see :func:`metafopy.core.clogit.clogit_settings` and :func:`metafopy.core.clogit.clogistic_settings`).
    The default ``'nlminb'`` maximises the exact likelihood numerically. The other optimisers of metafor's ``control$optimizer``
    (``'optim'`` with ``optmethod``, ``'Nelder-Mead'``, ``'BFGS'``, ``'CG'``, ``'L-BFGS-B'``, ``'nlm'``, ``'uobyqa'``, ``'newuoa'``,
    ``'bobyqa'``, ...) maximise it as :mod:`metafopy.models.sel_opt` minimises: ``optim`` with BFGS or Nelder-Mead is R's routine
    (``optCtrl`` takes ``maxit``, ``reltol``, ``abstol`` and ``ndeps``), the others stand in with the closest SciPy method. The Wald test
    of the saturated model (``QE_Wld``) of ``'CM.EL'`` is ill-conditioned and differs between optimisers in R as well.
    The Hessian of ``'CM.EL'`` is computed as ``control={'hesspack': ..., 'hessianCtrl': {...}}`` asks (``'numDeriv'`` with ``r = 16``
    unless ``hessianCtrl`` gives ``r``, ``'pracma'`` with ``h``, ``'calculus'`` with ``accuracy`` and ``stepsize``); it uses the density of
    ``BiasedUrn::dFNCHypergeo`` with precision ``dnchgprec`` (1e-10), a port of its algorithm including the truncation of the tails
    that makes R's finite-difference standard errors noisy, or with ``dnchgcalc='dnoncenhypergeom'`` the untruncated density.

    ``control={'package': ...}`` chooses the package that fits the mixed models, as in metafor:
    ``'lme4'`` (the default), ``'GLMMadaptive'`` (adaptive quadrature with ``nAGQ`` points in every
    dimension, so that ``'UM.RS'`` with ``method='ML'`` is not reduced to ``nAGQ = 1``; it fails, like
    metafor, when a coefficient other than the first exceeds 12 in absolute value) or ``'glmmTMB'``
    (Laplace approximation, ``nAGQ`` is ignored). They maximise the same likelihood and differ in
    the integration, in what log-likelihood they report and in the convergence tolerances; see
    ``docs/differences.md``.

    ``verbose`` prints metafor's progress messages on stderr (``1``: the model fits; ``2``: every stage); the
    trace output of ``glm()``/``glmer()`` and of the optimisers has no counterpart here.

    Examples
    --------
    >>> from metafopy import datasets, rma_glmm
    >>> dat = datasets.load("bcg")  # doctest: +SKIP
    >>> rma_glmm(ai="tpos", bi="tneg", ci="cpos", di="cneg", data=dat, measure="OR")  # doctest: +SKIP

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: a model cannot be fitted.
    """
    model_name = check_arguments(measure, model, method, coding, test)
    vlevel = int(verbose)
    if vlevel:
        metafopy.core.verbose.cat('\n')
    if vlevel > 1:
        metafopy.core.verbose.message('Extracting the data and computing yi/vi values ...')
    if measure is None:
        raise metafopy.errors.InputError("Must specify the 'measure' argument.")
    con = make_control(control)
    add1, to1, drop1 = float(first_of(add)), str(first_of(to)), bool(first_of(drop00))
    if to1 not in metafopy.models.tables.TO_CHOICES:
        raise metafopy.errors.InputError("Unknown 'to' argument specified.")
    nagq = nAGQ
    if (
        con.package == 'lme4'
        and (measure in metafopy.models.glmm_fit.TWO_GROUP_BINOMIAL or measure == 'IRR')
        and model_name == 'UM.RS'
        and method == 'ML'
        and nagq > 1
    ):
        warnings.warn("Not possible to fit RE/ME model='UM.RS' with nAGQ > 1 with glmer(). nAGQ automatically set to 1.", stacklevel=2)
        nagq = 1

    # -- model matrix and table data --------------------------------------
    k_all = length_of(measure, ai, x1i, xi, data)
    if mods is None and not intercept:
        warnings.warn('Must either include an intercept and/or moderators in the model.\nCoerced an intercept into the model.', stacklevel=2)
    mods_all, names_all, mods_missing, _ = metafopy.metadata.MetaData.design(mods, intercept, data, k_all)
    tab = metafopy.models.tables.prepare_tables(
        measure,
        ai=ai,
        bi=bi,
        ci=ci,
        di=di,
        n1i=n1i,
        n2i=n2i,
        x1i=x1i,
        x2i=x2i,
        t1i=t1i,
        t2i=t2i,
        xi=xi,
        mi=mi,
        ni=ni,
        ti=ti,
        extra_missing=mods_missing,
        data=data,
        slab=slab,
        subset=subset,
        add=(add1, 0.0),
        to=(typing.cast(metafopy.models.tables.To, to1), 'none'),
        drop00=(drop1, drop1),
        onlyo1=onlyo1,
        addyi=addyi,
        addvi=addvi,
        na_action=na_action,
    )
    if vlevel > 1:
        metafopy.core.verbose.message('Creating the model matrix ...')
        metafopy.core.verbose.message('Generating/extracting the study labels ...')
        if subset is not None:
            metafopy.core.verbose.message('Subsetting ...')
        if not bool(numpy.all(tab.not_na)):
            metafopy.core.verbose.message('Handling NAs in the table data ...')
        if not bool(numpy.all(tab.not_na_yivi)):
            metafopy.core.verbose.message('Handling NAs in yi/vi ...')
    design = build_design(tab, mods_all, names_all)
    X, X_f, X_yi = design.X, design.X_f, design.X_yi
    k, p = X.shape
    names = metafopy.metadata.make_unique(design.names)
    int_incl = design.int_incl
    int_only = p == 1 and metafopy.core.linalg.is_intercept(X[:, 0])
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    if (method in ('FE', 'EE', 'CE') and p > k) or (method == 'ML' and p + 1 > k):
        raise metafopy.errors.InputError('Number of parameters to be estimated is larger than the number of observations.')
    btt_idx = metafopy.models.rma_uni.set_btt(btt, p, int_incl, names)
    m = len(btt_idx)
    if con.optimizer in ('clogit', 'clogistic') and method == 'ML':
        raise metafopy.errors.InputError("Cannot use 'clogit' or 'clogistic' with method='ML'.")
    if not metafopy.core.linalg.is_positive_definite(X.T @ X, tol=con.evtol):
        raise metafopy.errors.InputError('Model matrix not of full rank. Cannot fit model.')

    scaled = not int_only and int_incl and con.scale_x
    mean_x = sd_x = numpy.zeros(0)
    is_d = numpy.zeros(0, dtype=numpy.bool_)
    X_fit = X
    if scaled:
        X_fit, mean_x, sd_x, is_d = scale_design(X)

    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    spec = metafopy.models.glmm_fit.FitSpec(measure, model_name, method, X_fit, tab.cells, coding, cor, nagq, alpha, con, resolve_family(measure, link, family))
    if vlevel:
        metafopy.core.verbose.message('Fitting the FE model ...')
        if k > 1:
            metafopy.core.verbose.message('Fitting the saturated model ...')
        if method == 'ML':
            metafopy.core.verbose.message('Fitting the ML model ...')
    fit = metafopy.models.glmm_fit.fit_model(spec)
    used_method = 'ML' if fit.method == 'ML' else method

    # -- tests and heterogeneity ----------------------------------------------
    if vlevel > 1:
        metafopy.core.verbose.message('Conducting the heterogeneity tests ...')
        metafopy.core.verbose.message('Conducting the tests of the fixed effects ...')
    qe_wld, qep_wld, qe_lrt, qep_lrt, qe_df = heterogeneity_tests(fit, k, p)
    i2, h2, vt = i2_h2(tab, X_yi, fit.tau2, p, i2def)
    beta, vb = fit.beta, fit.vb
    qm = omnibus_test(beta, vb, btt_idx)
    if scaled:
        beta, vb = back_transform(beta, vb, mean_x, sd_x, is_d)
    ddf = k - p if test == 't' else None
    ve = numpy.diag(vb)
    se = numpy.where(ve >= 0, numpy.sqrt(numpy.where(ve >= 0, ve, 0.0)), math.nan)
    zval = beta / se
    if ddf is not None:
        qm = qm / m
        qmdf: tuple[int, int | None] = (m, ddf)
        qmp = float(scipy.stats.f.sf(qm, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * scipy.stats.t.sf(numpy.abs(zval), ddf) if ddf > 0 else numpy.full(p, math.nan)
        crit = float(scipy.stats.t.isf(alpha / 2, ddf)) if ddf > 0 else math.nan
    else:
        qmdf = (m, None)
        qmp = float(scipy.stats.chi2.sf(qm, m))
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(alpha / 2))

    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    ll_ml = fit.ll_fe if method in ('FE', 'EE', 'CE') else fit.ll_ml
    stats_ml = fit_statistics(ll_ml, fit.ll_qe, fit.parms, fit.k_eff)
    if vlevel > 1:
        metafopy.core.verbose.message('Computing fit statistics and log-likelihood ...')
        metafopy.core.verbose.message('Preparing the output ...')
    nan_stats = metafopy.models.rma_uni.FitStats(math.nan, math.nan, math.nan, math.nan, math.nan)
    out = metafopy.models.glmm_result.GLMMResult(
        beta=beta,
        se=se,
        zval=zval,
        pval=numpy.asarray(pval, dtype=numpy.float64),
        ci_lb=beta - crit * se,
        ci_ub=beta + crit * se,
        vb=vb,
        tau2=fit.tau2,
        se_tau2=fit.se_tau2,
        sigma2=fit.sigma2,
        rho=fit.rho,
        ci_lb_tau2=fit.ci_lb_tau2,
        ci_ub_tau2=fit.ci_ub_tau2,
        I2=i2,
        H2=h2,
        vt=vt,
        QE_Wld=qe_wld,
        QEp_Wld=qep_wld,
        QE_LRT=qe_lrt,
        QEp_LRT=qep_lrt,
        QE_df=qe_df,
        QM=qm,
        QMdf=qmdf,
        QMp=qmp,
        k=k,
        k_f=tab.k_f,
        k_yi=tab.k_yi,
        k_eff=fit.k_eff,
        k_all=tab.k_all,
        p=p,
        p_eff=fit.p_eff,
        parms=fit.parms,
        int_only=int_only,
        int_incl=int_incl,
        intercept=int_incl,
        X=X,
        X_f=X_f,
        coef_names=names,
        tables=tab,
        measure=measure,
        method=used_method,
        model=model_name,
        weighted=True,
        test=test,
        ddf=ddf,
        btt=btt_idx,
        m=m,
        digits=metafopy.mfopt.reduce_digits(digits),
        level=alpha,
        add=add1,
        to=to1,
        drop00=drop1,
        fit_stats_ml=stats_ml,
        fit_stats_reml=nan_stats,
        se_warn=fit.se_warn,
        data=data,
    )
    if vlevel:
        metafopy.core.verbose.cat('\n')
    return out


def length_of(
    measure: str,
    ai: metafopy.metadata.Column | None,
    x1i: metafopy.metadata.Column | None,
    xi: metafopy.metadata.Column | None,
    data: pandas.DataFrame | None,
) -> int:
    """Number of tables supplied (the length of the first count argument), needed to build the model matrix.

    Raises:
        metafopy.errors.InputError: the count arguments for the measure are missing.
    """
    two_group = measure in metafopy.models.glmm_fit.TWO_GROUP_BINOMIAL
    first = ai if two_group else (x1i if measure == 'IRR' else xi)
    values = metafopy.metadata.column(first, data, 'ai' if two_group else ('x1i' if measure == 'IRR' else 'xi'))
    if values is None:
        raise metafopy.errors.InputError('Must specify the table or rate data arguments for this measure.')
    return int(values.shape[0])
