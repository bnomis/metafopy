"""Selection models for publication bias (metafor's ``selmodel()``).

A selection model describes how the chance that a study enters the meta-analysis depends on its p-value (or, for
``type="trunc"``, on its effect size). The fixed effects, ``tau^2`` and the selection parameters ``delta`` are estimated
jointly by maximum likelihood. The implementation follows ``selmodel.rma.uni()`` in metafor 5.2-1,
https://github.com/wviechtb/metafor/blob/master/R/selmodel.rma.uni.r, see
https://wviechtb.github.io/metafor/reference/selmodel.html for the models and the references: Iyengar and
Greenhouse (1988, doi:10.1214/ss/1177013012), Hedges (1992), Vevea and Hedges (1995, doi:10.1007/BF02294384), Vevea and
Woods (2005, doi:10.1037/1082-989X.10.4.428), Preston et al. (2004, doi:10.1002/sim.1722), Citkowicz and Vevea (2017,
doi:10.1037/met0000092) and Pustejovsky (2024).

Differences from metafor: ``subset`` refers to the studies in the fitted model (after missing values were removed),
only the ``numDeriv`` Hessian is available, and optimisers other than ``optim`` (BFGS, Nelder-Mead) are mapped to
SciPy methods, see :mod:`metafopy.models.sel_opt`.
"""

import collections.abc
import dataclasses
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.linalg
import scipy.special
import scipy.stats

import metafopy.core.integrate
import metafopy.core.linalg
import metafopy.core.numderiv
import metafopy.errors
import metafopy.metadata
import metafopy.models.rma_uni
import metafopy.models.sel_lik
import metafopy.models.sel_opt
import metafopy.models.sel_result
import metafopy.models.sel_spec
import metafopy.models.tau2
import metafopy.printing

__all__ = ['selmodel']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Model = metafopy.models.rma_uni.RMAResult | metafopy.models.sel_result.SelmodelResult

EPS = sys.float_info.epsilon
# auditme: accept[non-alphabetical-literal] the order is metafor's documented order
ALTERNATIVES = ('two.sided', 'greater', 'less')  # auditme: accept[literal-duplication] metafor's option lists recur in the other functions
# auditme: accept[non-alphabetical-literal] the order is metafor's documented order
PRECISIONS = ('sei', 'vi', 'ninv', 'sqrtninv')  # auditme: accept[literal-duplication] metafor's option lists recur in the other functions
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented order
EXTRA_ARGUMENTS = ('time', 'tau2', 'beta', 'skiphes', 'skiphet', 'skipintcheck', 'scaleprec', 'defmap', 'mapfun', 'mapinvfun', 'pval', 'ptable', 'retopt')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented order
CONTROL_NAMES = (
    'verbose',
    'delta.init',
    'beta.init',
    'tau2.init',
    'delta.min',
    'delta.max',
    'tau2.max',
    'tau2tol',
    'deltatol',
    'pval.min',
    'optimizer',
    'optmethod',
    'parallel',
    'cl',
    'ncpus',
    'hes.beta.fix',
    'hes.tau2.fix',
    'hes.delta.fix',
    'htransf',
    'hessianCtrl',
    'hesspack',
    'scaleX',
)


def squash(name: str) -> str:
    return name.lower().replace('.', '').replace('_', '')


def vector(value: typing.Any, n: int | None = None) -> FloatArray:
    arr = numpy.atleast_1d(numpy.array([math.nan if v is None else v for v in numpy.atleast_1d(value)], dtype=numpy.float64))
    if n is not None and arr.shape[0] == 1 and n > 1:
        arr = numpy.repeat(arr, n)
    return arr


def parse_control(control: collections.abc.Mapping[str, typing.Any] | None) -> tuple[dict[str, typing.Any], dict[str, typing.Any], dict[str, typing.Any]]:
    """Split ``control`` into the model settings, the optimiser settings and the integration settings (metafor's ``con``, ``optcontrol`` and ``intCtrl``)."""
    con: dict[str, typing.Any] = {}
    optcontrol: dict[str, typing.Any] = {}
    int_ctrl: dict[str, typing.Any] = {}
    squashed = {squash(n): n for n in CONTROL_NAMES}
    for key, value in (control or {}).items():
        sk = squash(key)
        if sk == 'intctrl':
            int_ctrl = dict(value)
            continue
        hit = squashed.get(sk)
        if hit is None:
            cands = [n for s, n in squashed.items() if s.startswith(sk)] if sk else []
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            optcontrol[key] = value
        else:
            con[hit] = value
    ic: dict[str, typing.Any] = {'lower': -math.inf, 'upper': math.inf, 'subdivisions': 100, 'rel.tol': metafopy.core.integrate.DEFAULT_REL_TOL}
    for key, value in int_ctrl.items():
        sk = squash(key)
        for name in ic:
            if squash(name).startswith(sk):
                ic[name] = value
                break
    return con, optcontrol, ic


def pvalues(yi: FloatArray, vi: FloatArray, alternative: str) -> FloatArray:
    """metafor's ``.selmodel.pval()``."""
    zi = yi / numpy.sqrt(vi)
    if alternative == 'two.sided':
        return numpy.asarray(2 * scipy.special.ndtr(-numpy.abs(zi)))
    if alternative == 'less':
        return numpy.asarray(scipy.special.ndtr(zi))
    return numpy.asarray(scipy.special.ndtr(-zi))


def r_as_character(x: float) -> str:
    """R's ``as.character()`` of a double: at most 15 significant digits, fixed or scientific notation, whichever is shorter."""
    if x == 0:
        return '0'
    if math.isnan(x):
        return 'NA'
    target = float(f'{x:.14e}')
    sig = 15
    for d in range(1, 16):
        if float(f'{x:.{d - 1}e}') == target:
            sig = d
            break
    mant, _, exp = f'{x:.{sig - 1}e}'.partition('e')
    e = int(exp)
    sci = f'{mant}e{"-" if e < 0 else "+"}{abs(e):02d}'
    if e >= 0:
        fixed = f'{x:.{max(sig - 1 - e, 0)}f}'
    else:
        fixed = f'{x:.{sig - 1 - e}f}'
    return fixed if len(fixed) <= len(sci) else sci


def make_ptable(pvals: FloatArray, steps: FloatArray, subset: BoolArray) -> tuple[numpy.typing.NDArray[numpy.intp], pandas.DataFrame]:
    """metafor's ``.ptable()``: the interval (0-based, -1 if none) of each p-value and the number of p-values per interval."""
    pgrp = numpy.full(pvals.shape[0], -1, dtype=numpy.intp)
    for i, pv in enumerate(pvals):
        if subset[i]:
            hit = numpy.flatnonzero(pv <= steps)
            if hit.size:
                pgrp[i] = hit[0]
    left = [r_as_character(v) for v in numpy.concatenate([[0.0], steps[:-1]])]
    right = [r_as_character(v) for v in steps]
    width = max(len(s) for s in left)
    labels = [f'{lft.ljust(width)} < p <= {rgt}' for lft, rgt in zip(left, right, strict=True)]
    counts = [int((pgrp == j).sum()) for j in range(steps.shape[0])]
    return pgrp, pandas.DataFrame({'k': counts}, index=labels)


def normalise_steps(type_: str, steps: typing.Any) -> tuple[FloatArray, bool]:
    """The ``steps`` handling at the top of metafor's ``selmodel.rma.uni()``.

    Raises:
        metafopy.errors.InputError: ``steps`` has missing values or values outside of (0, 1).
    """
    if steps is None:
        return numpy.array([math.nan]), False
    arr = vector(steps)
    if arr.shape[0] == 1 and math.isnan(arr[0]):
        return arr, False
    if numpy.isnan(arr).any():
        raise metafopy.errors.InputError("No missing values allowed in 'steps' argument.")
    if type_ != 'trunc' and numpy.any((arr < 0) | (arr > 1)):
        raise metafopy.errors.InputError("Value(s) specified for 'steps' argument must be between 0 and 1.")
    arr = numpy.unique(arr)
    if type_ not in ('trunc', 'beta'):
        if arr[0] == 0:
            raise metafopy.errors.InputError("Lowest 'steps' value must be > 0.")
        if arr[-1] != 1:
            arr = numpy.append(arr, 1.0)
    return arr, True


def refit_ml(base: metafopy.models.rma_uni.RMAResult) -> metafopy.models.rma_uni.RMAResult:
    """Refit a random/mixed-effects model with ML estimation (the first step of ``selmodel.rma.uni()``).

    Raises:
        metafopy.errors.InputError: the refit failed.
    """
    md = dataclasses.replace(base.data, yi=base.yi, vi=base.vi, X=base.X, X_names=base.coef_names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return metafopy.models.rma_uni.fit_rma_uni(
                md,
                method='ML',
                weighted=base.weighted,
                test=base.test,
                level=1 - base.level,
                tau2=base.tau2 if base.tau2_fix else None,
                control=base.control,
                skip_r2=True,
            )
    except metafopy.errors.MetaforError as exc:
        raise metafopy.errors.InputError("Could not refit input model using method='ML'.") from exc


def mapping(value: typing.Any, deltas: int) -> tuple[str | None, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,) * deltas
    seq = tuple(None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in value)
    return seq * deltas if len(seq) == 1 else seq


def check_bounds(name: str, vals: FloatArray, spec: metafopy.models.sel_spec.SelSpec) -> None:
    """The checks of ``delta`` (``delta.min``, ``delta.max``, ``delta.init``) against the parameter space, as metafor does them.

    Raises:
        metafopy.errors.InputError: a value lies outside of the parameter space.
    """
    for j in range(spec.deltas):
        v = vals[j]
        if math.isnan(v):
            continue
        lb, ub = spec.delta_lb[j], spec.delta_ub[j]
        if spec.delta_lb_excl[j] and v <= lb:
            raise metafopy.errors.InputError(f"Value of '{name}[{j + 1}]' must be > {lb:g} for this type of selection model.")
        if not spec.delta_lb_excl[j] and v < lb:
            raise metafopy.errors.InputError(f"Value of '{name}[{j + 1}]' must be >= {lb:g} for this type of selection model.")
        if spec.delta_ub_excl[j] and v >= ub:
            raise metafopy.errors.InputError(f"Value of '{name}[{j + 1}]' must be < {ub:g} for this type of selection model.")
        if not spec.delta_ub_excl[j] and v > ub:
            raise metafopy.errors.InputError(f"Value of '{name}[{j + 1}]' must be <= {ub:g} for this type of selection model.")


def invert_hessian(h: FloatArray) -> FloatArray | None:
    """``chol2inv(chol(h))``, or ``None`` if ``h`` is not positive definite."""
    try:
        chol = scipy.linalg.cholesky(h, lower=True)
        inv = scipy.linalg.cho_solve((chol, True), numpy.eye(h.shape[0]))
    except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError:
        return None
    if not numpy.all(numpy.isfinite(inv)):
        return None
    return numpy.asarray(inv, dtype=numpy.float64)


def log_progress(prob: metafopy.models.sel_lik.Problem, par: FloatArray, digits: int, delta_transf: bool, tau2_transf: bool) -> None:
    ev = metafopy.models.sel_lik.evaluate(prob, par, delta_transf=delta_transf, tau2_transf=tau2_transf)
    fmt = metafopy.printing.fmt_num
    beta = ' '.join(fmt(b, digits) for b in ev.beta)
    delta = ' '.join(fmt(d, digits) for d in ev.delta)
    sys.stderr.write(f'll = {fmt(ev.ll, digits)}  beta = {beta}  tau2 = {fmt(ev.tau2, digits)}  delta = {delta}\n')


def selmodel(
    x: Model,
    type: str,
    alternative: str = 'greater',
    prec: str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    delta: typing.Any = None,
    steps: typing.Any = None,
    decreasing: bool = False,
    verbose: bool | int = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> metafopy.models.sel_result.SelmodelResult | pandas.DataFrame | FloatArray:
    """Fit a selection model to a fitted ``rma_uni`` model (metafor's ``selmodel.rma.uni``).

    ``type`` is one of ``'beta'``, ``'halfnorm'``, ``'negexp'``, ``'logistic'``, ``'power'``, ``'negexppow'``,
    ``'stepfun'``, ``'trunc'``, ``'truncest'`` (plus the variants ``'halfnorm1'``, ``'halfnorm2'``, ... and
    ``'stepcon'`` metafor documents in its source); partial names work. ``alternative`` is ``'greater'``,
    ``'less'`` or ``'two.sided'``; ``prec`` one of ``'sei'``, ``'vi'``, ``'ninv'`` and ``'sqrtninv'``. ``delta`` fixes
    selection parameters (``None``/``nan`` elements are estimated), ``steps`` holds the cutpoints, ``subset`` a logical
    (or index) vector over the studies of ``x`` to which the selection function applies. ``control`` takes metafor's
    entries (``delta.init``, ``optimizer``, ``optmethod``, ``intCtrl``, ... with ``.`` or ``_``) and the settings of the optimiser
    (``maxit``, ``reltol``, ``ndeps``). Further keyword arguments as in metafor: ``tau2``, ``beta`` (fixed values, ``nan``
    = free), ``scaleprec``, ``pval``, ``ptable`` (return the p-value table only), ``skiphes``, ``skiphet``,
    ``defmap``, ``mapfun``, ``mapinvfun`` and ``retopt`` (return the optimised parameter vector).

    Models that were not fitted with ML (or as an equal/fixed-effects model) are refitted with ML first.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a model this can be applied to.
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: the model could not be fitted.
        metafopy.errors.NotSupportedError: a requested option is not available.
    """
    prev: metafopy.models.sel_result.SelmodelResult | None = None
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        prev = x
        base = x.base
    elif isinstance(x, metafopy.models.rma_uni.RMAResult):
        base = x
        if base.robust is not None:
            raise metafopy.errors.ArgumentError('selmodel() cannot be applied to objects of class robust.rma.')
    else:
        raise metafopy.errors.ArgumentError(f'selmodel() not implemented for {x.__class__.__name__}')
    if not base.allvipos:
        raise metafopy.errors.InputError('Cannot fit selection model when one or more sampling variances are non-positive.')
    if not base.weighted or base.weights is not None:
        raise metafopy.errors.InputError('Cannot fit selection model for unweighted models or models with custom weights.')

    alt = metafopy.models.sel_opt.partial_match(alternative, ALTERNATIVES, 'alternative')
    type_ = metafopy.models.sel_spec.match_type(type)
    if type_ in metafopy.models.sel_spec.TRUNCATED and alt == 'two.sided':
        raise metafopy.errors.InputError("Cannot use alternative='two-sided' with this type of selection model.")
    decreasing = bool(decreasing)
    if type_ != 'stepfun' and decreasing:
        warnings.warn("Argument 'decreasing' ignored (not applicable to this type of selection model).", stacklevel=2)
        decreasing = False

    fixed_effects = base.method in metafopy.models.tau2.FIXED_EFFECTS
    if not fixed_effects and base.method != 'ML':
        base = refit_ml(base)
        prev = None
    ll_input = prev.ll if prev is not None else base.fit_stats_ml.ll
    tau2_fix_in = prev.tau2_fix if prev is not None else base.tau2_fix
    beta_in = prev.beta if prev is not None else base.beta
    tau2_in = prev.tau2 if prev is not None else base.tau2

    for name in kwargs:
        if name not in EXTRA_ARGUMENTS:
            warnings.warn(f'Extra argument ({name!r}) disregarded.', stacklevel=2)

    tau2_fix = tau2_fix_in
    tau2_kw = kwargs.get('tau2')
    if tau2_kw is None:
        if fixed_effects:
            tau2_arg = 0.0
        elif tau2_fix_in:
            tau2_arg = tau2_in
        else:
            tau2_arg = math.nan
    else:
        tau2_arg = math.nan if tau2_kw is None else float(tau2_kw)
        if not math.isnan(tau2_arg):
            tau2_fix = True

    p = base.p
    k = base.k
    beta_kw = kwargs.get('beta')
    if beta_kw is None:
        beta_arg = numpy.full(p, math.nan)
        betaspec = False
    else:
        beta_arg = vector(beta_kw, p)
        betaspec = True

    yi = base.yi
    vi = base.vi
    X = base.X.copy()

    # -- precision measure --------------------------------------------------------------
    scaleprec = kwargs.get('scaleprec')
    precspec = prec is not None
    prec_name: str | None = None
    if prec is not None:
        prec_name = metafopy.models.sel_opt.partial_match(prec, PRECISIONS, 'prec')
        if prec_name == 'sei':
            preci = numpy.sqrt(vi)
        elif prec_name == 'vi':
            preci = vi.copy()
        else:
            ni = base.data.ni
            if ni is None or numpy.isnan(ni).any():
                raise metafopy.errors.InputError(
                    'No sample size information stored in model object (or sample size information stored in model object contains NAs).'
                )
            preci = 1 / ni if prec_name == 'ninv' else 1 / numpy.sqrt(ni)
        if scaleprec is None or scaleprec:
            preci = preci / preci.max()
    else:
        preci = numpy.ones(k)
    precis = {'min': float(preci.min()), 'max': float(preci.max()), 'mean': float(preci.mean()), 'median': float(numpy.median(preci))}

    # -- p-values --------------------------------------------------------------------------
    if kwargs.get('pval') is None:
        pvals = pvalues(yi, vi, alt)
    else:
        pvals = vector(kwargs['pval'])
        if pvals.shape[0] != k:
            raise metafopy.errors.InputError(f"Length of the 'pval' argument ({pvals.shape[0]}) does not correspond to the number of studies ({k}).")
        if numpy.isnan(pvals).any():
            raise metafopy.errors.InputError("No missing values in 'pval' argument allowed.")
        if numpy.any(pvals <= 0) or numpy.any(pvals > 1):
            raise metafopy.errors.InputError("One or more 'pval' values are <= 0 or > 1.")
        pvals = pvals.copy()

    steps_arr, stepsspec = normalise_steps(type_, steps)
    if type_ == 'trunc' and not stepsspec:
        stepsspec = True
        steps_arr = numpy.array([0.0])

    if subset is None:
        subset_mask = numpy.ones(k, dtype=numpy.bool_)
        subset_spec = False
    else:
        sub = numpy.asarray(subset)
        if sub.dtype == numpy.bool_:
            if sub.shape != (k,):
                raise metafopy.errors.InputError(f"A logical 'subset' must have length {k} (the number of studies in the model).")
            subset_mask = sub.copy()
        else:
            subset_mask = numpy.zeros(k, dtype=numpy.bool_)
            subset_mask[sub.astype(numpy.intp)] = True
        subset_spec = True

    # -- control ----------------------------------------------------------------------------
    con, optcontrol, int_ctrl = parse_control(control)
    con.setdefault('verbose', False)
    if verbose:
        con['verbose'] = verbose
    verbose_level = int(con['verbose'])
    con.setdefault('tau2.max', math.inf)
    con.setdefault('tau2tol', min(float(numpy.min(vi / 10)), 1e-04))
    con.setdefault('deltatol', 1e-04)
    con.setdefault('optimizer', 'optim')
    con.setdefault('optmethod', 'BFGS')
    con.setdefault('ncpus', 1)
    con.setdefault('hes.beta.fix', False)
    con.setdefault('hes.tau2.fix', False)
    con.setdefault('hes.delta.fix', False)
    con.setdefault('htransf', False)
    con.setdefault('hesspack', 'numDeriv')
    con.setdefault('scaleX', not betaspec)
    if con['hesspack'] != 'numDeriv':
        raise metafopy.errors.NotSupportedError("Only hesspack='numDeriv' is available.")
    hessian_ctrl = dict(con.get('hessianCtrl') or {})
    hessian_ctrl.setdefault('r', 6)
    choice = metafopy.models.sel_opt.choose_optimizer(str(con['optimizer']), str(con['optmethod']), type_, int(con['ncpus']))

    scale_x = (not base.int_only) and base.int_incl and bool(con['scaleX'])
    mX = numpy.eye(p)
    if scale_x:
        Xsave = X
        Xrest = X[:, 1:]
        meanX = Xrest.mean(axis=0)
        sdX = Xrest.std(axis=0, ddof=1)
        is_d = numpy.array([metafopy.core.linalg.is_dummy(X[:, j]) for j in range(p)])
        dummy = is_d[1:]
        mX = numpy.zeros((p, p))
        mX[0, 0] = 1.0
        mX[0, 1:] = -1 * numpy.where(dummy, 0.0, meanX / sdX)
        mX[1:, 1:] = numpy.diag(numpy.where(dummy, 1.0, 1 / sdX))
        X = X.copy()
        for j in range(p):
            if not is_d[j]:
                X[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std(ddof=1)
    else:
        Xsave = X

    if con.get('beta.init') is None:
        beta_init = numpy.array(beta_in, dtype=numpy.float64)
    else:
        beta_init = vector(con['beta.init'])
        if beta_init.shape[0] != p:
            raise metafopy.errors.InputError(f"Length of the 'beta.init' argument ({beta_init.shape[0]}) does not match the actual number of parameters ({p}).")
    if scale_x:
        try:
            beta_init = numpy.linalg.solve(mX, beta_init)
        except numpy.linalg.LinAlgError as exc:
            raise metafopy.errors.InputError('Unable to rescale starting values for the fixed effects.') from exc

    tau2_max = float(con['tau2.max'])
    if tau2_in >= tau2_max:
        raise metafopy.errors.InputError("Value of 'tau2.max' must be > tau^2 value.")
    if con.get('tau2.init') is None:
        tau2_init = math.log(tau2_in + 1e-3)
    else:
        ti = vector(con['tau2.init'])
        if ti.shape[0] != 1:
            raise metafopy.errors.InputError("Argument 'tau2.init' should specify a single value.")
        if ti[0] <= 0:
            raise metafopy.errors.InputError("Value of 'tau2.init' must be > 0.")
        if ti[0] >= tau2_max:
            raise metafopy.errors.InputError("Value of 'tau2.init' must be < 'tau2.max'.")
        tau2_init = math.log(float(ti[0]))

    # -- selection model definition --------------------------------------------------------
    ddefmap = kwargs.get('defmap')
    spec = metafopy.models.sel_spec.build_spec(
        type_,
        alternative=alt,
        steps=steps_arr,
        stepsspec=stepsspec,
        precspec=precspec,
        decreasing=decreasing,
        preci=preci,
        yi=yi,
        defmap=bool(ddefmap),
    )
    preci = spec.preci
    steps_arr = spec.steps
    stepsspec = spec.stepsspec
    deltas = spec.deltas

    if delta is None:
        delta_arg = numpy.full(deltas, math.nan)
    else:
        delta_arg = vector(delta, deltas)
        if delta_arg.shape[0] != deltas:
            raise metafopy.errors.InputError(f"Argument 'delta' should be of length {deltas} for this type of selection model.")
        check_bounds('delta', delta_arg, spec)
    if type_ == 'stepfun' and (decreasing or math.isnan(delta_arg[0])):
        delta_arg[0] = 1.0
    if type_ == 'stepcon':
        delta_arg[0] = 1.0

    delta_min = spec.delta_min.copy() if con.get('delta.min') is None else vector(con['delta.min'], deltas)
    if delta_min.shape[0] != deltas:
        raise metafopy.errors.InputError(f"Argument 'delta.min' should be of length {deltas} for this type of selection model.")
    if numpy.isnan(delta_min).any():
        raise metafopy.errors.InputError("No missing values allowed in 'delta.min'.")
    check_bounds('delta.min', delta_min, spec)
    delta_min = numpy.where(~numpy.isnan(delta_arg) & (delta_min > delta_arg), delta_arg - EPS**0.2, delta_min)

    delta_max = spec.delta_max.copy() if con.get('delta.max') is None else vector(con['delta.max'], deltas)
    if delta_max.shape[0] != deltas:
        raise metafopy.errors.InputError(f"Argument 'delta.max' should be of length {deltas} for this type of selection model.")
    if numpy.isnan(delta_max).any():
        raise metafopy.errors.InputError("No missing values allowed in 'delta.max'.")
    check_bounds('delta.max', delta_max, spec)
    if numpy.any(delta_max < delta_min):
        raise metafopy.errors.InputError("Value(s) of 'delta.max' must be >= value(s) of 'delta.min'.")
    delta_max = numpy.where(~numpy.isnan(delta_arg) & (delta_max < delta_arg), delta_arg + EPS**0.2, delta_max)

    delta_init = spec.delta_init.copy() if con.get('delta.init') is None else vector(con['delta.init'], deltas)
    if delta_init.shape[0] != deltas:
        raise metafopy.errors.InputError(f"Argument 'delta.init' should be of length {deltas} for this type of selection model.")
    if numpy.isnan(delta_init).any():
        raise metafopy.errors.InputError("No missing values allowed in 'delta.init'.")
    check_bounds('delta.init', delta_init, spec)

    mapfun_kw = kwargs.get('mapfun')
    mapinvfun_kw = kwargs.get('mapinvfun')
    if bool(ddefmap) or spec.defmap or numpy.any(numpy.isinf(delta_max)):
        mapfun_kw = spec.transf_fun
        mapinvfun_kw = spec.transf_inv
        defmap_out: bool | None = True if (bool(ddefmap) or spec.defmap) else ddefmap
    else:
        defmap_out = ddefmap
    mapfun = list(mapping(mapfun_kw, deltas) or (None,) * deltas)
    mapinvfun = list(mapping(mapinvfun_kw, deltas) or (None,) * deltas)
    if type_ == 'truncest':
        mapfun[1] = 'I'
        mapinvfun[1] = 'I'

    delta_init = numpy.array(
        [
            metafopy.models.sel_lik.map_inverse(float(d), float(lo), float(hi), f)
            for d, lo, hi, f in zip(delta_init, delta_min, delta_max, mapinvfun, strict=True)
        ]
    )
    delta_init = numpy.where(numpy.isnan(delta_arg), delta_init, delta_arg)

    pval_min = spec.pval_min if con.get('pval.min') is None else float(con['pval.min'])

    n_sub = int(subset_mask.sum())
    n_free = int(numpy.isnan(delta_arg).sum())
    n_par = p + (0 if (fixed_effects or tau2_fix) else 1) + n_free
    if (n_sub if subset_spec else k) < n_par:
        what = f'k_subset={n_sub}' if subset_spec else f'k={k}'
        raise metafopy.errors.InputError(f'Number of studies ({what}) is too small to fit the selection model.')

    pvals = pvals.copy()
    pvals[pvals < pval_min] = pval_min
    pvals[pvals > 1 - pval_min] = 1 - pval_min

    pgrp = numpy.full(k, -1, dtype=numpy.intp)
    ptable: pandas.DataFrame | None = None
    if type_ != 'trunc' and stepsspec:
        pgrp, ptable = make_ptable(pvals, steps_arr, subset_mask)
        if kwargs.get('ptable'):
            return ptable
        if (ptable['k'] == 0).any():
            if not kwargs.get('skipintcheck') and type_ == 'stepfun' and numpy.isnan(delta_arg[1:]).any():
                warnings.warn('One or more intervals do not contain any observed p-values.', stacklevel=2)
            if not kwargs.get('skipintcheck') and type_ != 'stepfun':
                warnings.warn('One of the intervals does not contain any observed p-values.', stacklevel=2)

    # -- model fitting ------------------------------------------------------------------------
    prob = metafopy.models.sel_lik.Problem(
        spec=spec,
        yi=yi,
        vi=vi,
        X=X,
        preci=preci,
        subset=subset_mask,
        pvals=pvals,
        pgrp=pgrp,
        delta_arg=delta_arg,
        tau2_arg=tau2_arg,
        beta_arg=beta_arg,
        mapfun=tuple(mapfun),
        delta_min=delta_min,
        delta_max=delta_max,
        decreasing=decreasing,
        tau2_max=tau2_max,
        alternative=alt,
        int_lower=float(int_ctrl['lower']),
        int_upper=float(int_ctrl['upper']),
        int_subdivisions=int(int_ctrl['subdivisions']),
        int_rel_tol=float(int_ctrl['rel.tol']),
    )
    digits_out = 4 if digits is None else int(digits)
    par0 = numpy.concatenate([beta_init, [tau2_init], delta_init])

    def fn(par: FloatArray) -> float:
        if verbose_level > 0:
            log_progress(prob, par, digits_out, True, True)
        return metafopy.models.sel_lik.negative_loglik(prob, par, delta_transf=True, tau2_transf=True)

    ineq = None
    if type_ == 'stepcon':

        def ineq(par: FloatArray) -> FloatArray:
            d = metafopy.models.sel_lik.transform_delta(prob, par[p + 1 :], True)
            return numpy.asarray(-numpy.diff(d), dtype=numpy.float64)

    opt_par = metafopy.models.sel_opt.minimize(fn, par0, choice, optcontrol, ineq)
    if kwargs.get('retopt'):
        return opt_par
    fit = metafopy.models.sel_lik.evaluate(prob, opt_par, delta_transf=True, tau2_transf=True)
    ll = fit.ll
    beta = fit.beta
    tau2 = fit.tau2
    delta_hat = fit.delta
    tau2_transf = float(opt_par[p])
    delta_transf_par = opt_par[p + 1 :]

    if numpy.any(numpy.isnan(delta_arg) & (delta_hat <= delta_min + EPS**0.25)) or (
        spec.delta_max_check and numpy.any(numpy.isnan(delta_arg) & (delta_hat >= delta_max - 100 * EPS**0.25))
    ):
        warnings.warn(
            "One or more 'delta' estimates are (almost) equal to their lower or upper bound.\n"
            "Treat results with caution (or consider adjusting 'delta.min' and/or 'delta.max').",
            stacklevel=2,
        )

    # -- Hessian -------------------------------------------------------------------------------
    H: FloatArray | None = None
    vb = numpy.full((p, p), math.nan)
    se_tau2 = math.nan
    vd = numpy.full((deltas, deltas), math.nan)
    beta_hes = beta.copy() if con['hes.beta.fix'] else beta_arg
    tau2_hes = tau2 if (con['hes.tau2.fix'] or tau2 < float(con['tau2tol'])) else tau2_arg
    if con['hes.delta.fix']:
        delta_hes = delta_hat.copy()
    elif type_ == 'stepfun':
        delta_hes = numpy.where(delta_hat < float(con['deltatol']), delta_hat, delta_arg)
    else:
        delta_hes = delta_arg
    hest = numpy.concatenate([numpy.isnan(beta_hes), [math.isnan(tau2_hes)], numpy.isnan(delta_hes)])

    if hest.any() and not kwargs.get('skiphes'):
        if con['htransf']:
            hx = opt_par.copy()
            dt, tt = True, True
        else:
            hx = numpy.concatenate([beta, [tau2], delta_hat])
            dt, tt = False, False
        prob_h = dataclasses.replace(prob, delta_arg=numpy.asarray(delta_hes, dtype=numpy.float64), tau2_arg=float(tau2_hes), beta_arg=numpy.asarray(beta_hes))

        def hess_fn(par: FloatArray) -> float:
            return metafopy.models.sel_lik.negative_loglik(prob_h, par, delta_transf=dt, tau2_transf=tt)

        try:
            H = metafopy.core.numderiv.hessian(
                hess_fn, hx, d=float(hessian_ctrl.get('d', 0.1)), eps=float(hessian_ctrl.get('eps', 1e-4)), r=int(hessian_ctrl['r'])
            )
        except metafopy.errors.MetaforError, ArithmeticError, ValueError:
            H = None
            warnings.warn('Error when trying to compute the Hessian.', stacklevel=2)
        if H is not None:
            iH_hest = invert_hessian(H[numpy.ix_(hest, hest)])
            if iH_hest is None:
                warnings.warn('Error when trying to invert the Hessian.', stacklevel=2)
            else:
                iH = numpy.zeros((hest.shape[0], hest.shape[0]))
                iH[numpy.ix_(hest, hest)] = iH_hest
                nb = numpy.isnan(beta_hes)
                if nb.any():
                    vb[numpy.ix_(nb, nb)] = iH[:p, :p][numpy.ix_(nb, nb)]
                if math.isnan(tau2_hes):
                    se_tau2 = math.sqrt(iH[p, p]) if iH[p, p] >= 0 else math.nan
                nd = numpy.isnan(delta_hes)
                if nd.any():
                    vd[numpy.ix_(nd, nd)] = iH[p + 1 :, p + 1 :][numpy.ix_(nd, nd)]

    # -- Wald-type tests -------------------------------------------------------------------------
    if scale_x:
        beta = mX @ beta
        vb = mX @ vb @ mX.T
        X = Xsave
    btt = list(base.btt)
    try:
        sub_vb = vb[numpy.ix_(btt, btt)]
        chol = scipy.linalg.cholesky(sub_vb, lower=True)
        z = scipy.linalg.solve_triangular(chol, beta[btt], lower=True)
        QM = float(z @ z)
        if not math.isfinite(QM):
            QM = math.nan
    except numpy.linalg.LinAlgError, scipy.linalg.LinAlgError, ValueError:
        QM = math.nan
    QMp = float(scipy.stats.chi2.sf(QM, base.m)) if not math.isnan(QM) else math.nan

    with numpy.errstate(all='ignore'):
        se = numpy.sqrt(numpy.diag(vb))
        zval = beta / se
        pval = 2 * scipy.stats.norm.sf(numpy.abs(zval))
        crit = float(scipy.stats.norm.isf(base.level / 2))
        ci_lb = beta - crit * se
        ci_ub = beta + crit * se

        se_delta = numpy.sqrt(numpy.diag(vd))
        if con['htransf']:
            zval_delta = numpy.full(deltas, math.nan)
            pval_delta = numpy.full(deltas, math.nan)
            lo = delta_transf_par - crit * se_delta
            hi = delta_transf_par + crit * se_delta
            ci_lb_delta = numpy.array(
                [metafopy.models.sel_lik.map_delta(float(v), float(a), float(b), f) for v, a, b, f in zip(lo, delta_min, delta_max, mapfun, strict=True)]
            )
            ci_ub_delta = numpy.array(
                [metafopy.models.sel_lik.map_delta(float(v), float(a), float(b), f) for v, a, b, f in zip(hi, delta_min, delta_max, mapfun, strict=True)]
            )
            vd = numpy.full((deltas, deltas), math.nan)
            se_delta = numpy.full(deltas, math.nan)
        else:
            zval_delta = (delta_hat - spec.h0_delta) / se_delta
            pval_delta = 2 * scipy.stats.norm.sf(numpy.abs(zval_delta))
            ci_lb_delta = delta_hat - crit * se_delta
            ci_ub_delta = delta_hat + crit * se_delta
        ci_lb_delta = numpy.where(ci_lb_delta < spec.delta_lb, spec.delta_lb, ci_lb_delta)
        ci_ub_delta = numpy.where(ci_ub_delta > spec.delta_ub, spec.delta_ub, ci_ub_delta)
        ci_lb_delta = numpy.where(ci_lb_delta < delta_min, delta_min, ci_lb_delta)
        ci_ub_delta = numpy.where(ci_ub_delta > delta_max, delta_max, ci_ub_delta)

        if con['htransf']:
            ci_lb_tau2 = math.exp(tau2_transf - crit * se_tau2)
            ci_ub_tau2 = math.exp(tau2_transf + crit * se_tau2)
            se_tau2 = se_tau2 * math.exp(tau2_transf)
        else:
            ci_lb_tau2 = tau2 - crit * se_tau2
            ci_ub_tau2 = tau2 + crit * se_tau2
    if ci_lb_tau2 < 0:
        ci_lb_tau2 = 0.0

    # -- LRT for tau^2 = 0 ---------------------------------------------------------------------------
    LRT_tau2 = math.nan
    LRTp_tau2 = math.nan
    if not tau2_fix and not fixed_effects and not kwargs.get('skiphet'):
        # metafor restores the unscaled design matrix before this fit but keeps the starting values of the scaled one
        prob0 = dataclasses.replace(prob, tau2_arg=0.0, X=X)

        def fn0(par: FloatArray) -> float:
            return metafopy.models.sel_lik.negative_loglik(prob0, par, delta_transf=True, tau2_transf=False)

        try:
            par0_fit = metafopy.models.sel_opt.minimize(fn0, par0, choice, optcontrol, ineq)
            ll0_tau2 = metafopy.models.sel_lik.evaluate(prob0, par0_fit, delta_transf=True, tau2_transf=False).ll
            LRT_tau2 = max(0.0, -2 * (ll0_tau2 - ll))
            LRTp_tau2 = float(scipy.stats.chi2.sf(LRT_tau2, 1))
        except metafopy.errors.MetaforError, ArithmeticError, ValueError:
            pass

    # -- LRT for the selection parameters and fit statistics ------------------------------------------
    ll0 = ll_input
    LRT = max(0.0, -2 * (ll0 - ll))
    LRTdf = int((numpy.isnan(delta_arg) & numpy.array(spec.delta_lrt)).sum())
    LRTp = float(scipy.stats.chi2.sf(LRT, LRTdf)) if LRTdf > 0 else math.nan

    parms = p + (0 if (fixed_effects or tau2_fix) else 1) + n_free
    big = max(k, parms + 2)
    fit_stats = metafopy.models.rma_uni.FitStats(
        ll=ll,
        dev=-2 * ll,
        AIC=-2 * ll + 2 * parms,
        BIC=-2 * ll + parms * math.log(k),
        AICc=-2 * ll + 2 * parms * big / (big - parms - 1),
    )

    ptable_out = ptable
    return metafopy.models.sel_result.SelmodelResult(
        base=base,
        beta=numpy.asarray(beta, dtype=numpy.float64),
        se=se,
        zval=zval,
        pval=pval,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vb=vb,
        betaspec=betaspec,
        tau2=tau2,
        se_tau2=se_tau2,
        ci_lb_tau2=ci_lb_tau2,
        ci_ub_tau2=ci_ub_tau2,
        tau2_fix=tau2_fix,
        QM=QM,
        QMp=QMp,
        delta=delta_hat,
        vd=vd,
        se_delta=se_delta,
        zval_delta=zval_delta,
        pval_delta=pval_delta,
        ci_lb_delta=ci_lb_delta,
        ci_ub_delta=ci_ub_delta,
        deltas=deltas,
        delta_fix=tuple(bool(v) for v in ~numpy.isnan(delta_arg)),
        hessian=H,
        hest=tuple(bool(v) for v in hest),
        ll=ll,
        ll0=ll0,
        LRT=LRT,
        LRTdf=LRTdf,
        LRTp=LRTp,
        LRT_tau2=LRT_tau2,
        LRTp_tau2=LRTp_tau2,
        parms=parms,
        fit_stats_ml=fit_stats,
        fit_stats_reml=metafopy.models.sel_result.NA_FIT,
        pvals=pvals,
        spec=spec,
        type=type_,
        steps=steps_arr,
        decreasing=decreasing,
        stepsspec=stepsspec,
        pgrp=pgrp,
        ptable=ptable_out,
        subset=subset_mask,
        k0=int((~subset_mask).sum()),
        k1=n_sub,
        alternative=alt,
        pval_min=pval_min,
        prec=prec_name,
        precspec=precspec,
        precis=precis,
        scaleprec=scaleprec,
        delta_lb=spec.delta_lb,
        delta_ub=spec.delta_ub,
        delta_lb_excl=spec.delta_lb_excl,
        delta_ub_excl=spec.delta_ub_excl,
        delta_min=delta_min,
        delta_max=delta_max,
        tau2_max=tau2_max,
        control=dict(control or {}),
        defmap=defmap_out,
        mapfun=None if mapfun_kw is None else tuple(mapping(mapfun_kw, deltas) or ()),
        mapinvfun=None if mapinvfun_kw is None else tuple(mapping(mapinvfun_kw, deltas) or ()),
        digits=digits_out,
    )
