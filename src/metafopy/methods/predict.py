"""Predicted values, confidence and prediction intervals (metafor's ``predict.rma``)."""

import ast
import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.special
import scipy.stats

import metafopy.core.clubsandwich
import metafopy.core.labelled
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.methods.listrma
import metafopy.methods.pairmat
import metafopy.mfopt
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_uni
import metafopy.models.robust_info
import metafopy.models.tau2
import metafopy.printing
import metafopy.transf

__all__ = ['PredictResult', 'predict']

FloatArray = numpy.typing.NDArray[numpy.float64]
type NewMods = numpy.typing.ArrayLike | collections.abc.Mapping[str, float] | pandas.Series | pandas.DataFrame


@dataclasses.dataclass(frozen=True, slots=True)
class PredictResult:
    """Predictions with confidence (``ci``) and prediction (``pi``) intervals.

    ``se`` is NaN when a transformation was applied, as in metafor. ``pi_se`` is the standard error of the
    prediction (before any transformation), ``pi_dist`` the distribution (``'norm'`` or ``'t'``) of the prediction
    and ``pi_ddf`` its degrees of freedom (one per prediction) (used by ``forest(predstyle='shade')`` and ``'dist'``).
    ``prob`` holds the probabilities requested via ``prob``, ``X`` the model matrix of the predictions (``addx=True``)
    and ``vcov`` their variance-covariance matrix (``vcov=True`` without ``transf``); these two are data frames for
    ``matreg`` predictions (:class:`~metafopy.methods.matreg.MatregPredictResult`).
    """

    pred: FloatArray
    se: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    pi_lb: FloatArray
    pi_ub: FloatArray
    slab: tuple[str, ...] | None = None
    pi_se: FloatArray | None = None
    pi_dist: str | None = None
    pi_ddf: float | FloatArray | None = None
    ddf: float | FloatArray | None = None
    has_pi: bool = True
    prob: FloatArray | None = None
    X: FloatArray | pandas.DataFrame | None = None
    vcov: FloatArray | pandas.DataFrame | None = None
    x_names: tuple[str, ...] | None = None
    digits: int = 4
    digit_map: collections.abc.Mapping[str, int] | None = None
    method: str | None = None
    transf: bool = False
    pred_type: str = 'location'

    def __post_init__(self) -> None:
        """Give the model matrix its column names and ``vcov`` the prediction labels as its dimnames, as metafor's ``predict`` does."""
        names = self.x_names
        if isinstance(self.X, numpy.ndarray) and names is not None and self.X.ndim == 2 and len(names) == self.X.shape[1]:
            object.__setattr__(self, 'X', metafopy.core.labelled.LabelledMatrix(self.X, None, names))
        slab = self.slab
        vcov = self.vcov
        if isinstance(vcov, numpy.ndarray) and slab is not None and vcov.ndim == 2 and vcov.shape == (len(slab), len(slab)) and all(slab):
            object.__setattr__(self, 'vcov', metafopy.core.labelled.VCovMat(vcov, slab, slab))

    @property
    def cr_lb(self) -> FloatArray:
        """Alias of ``pi_lb`` (metafor keeps ``cr.lb`` for old code)."""
        return self.pi_lb

    @property
    def cr_ub(self) -> FloatArray:
        """Alias of ``pi_ub`` (metafor keeps ``cr.ub`` for old code)."""
        return self.pi_ub

    def __getitem__(self, index: typing.Any) -> typing.Self:
        """The rows ``index`` (metafor's ``[.list.rma``; a NumPy index, 0-based)."""
        return metafopy.methods.listrma.rows(self, index, self.pred.shape[0])

    def to_frame(self) -> pandas.DataFrame:
        cols: dict[str, typing.Any] = {
            'pred': self.pred,
            'se': self.se,
            'ci.lb': self.ci_lb,
            'ci.ub': self.ci_ub,
            'pi.lb': self.pi_lb,
            'pi.ub': self.pi_ub,
        }
        if self.prob is not None:
            cols['prob'] = self.prob
        if self.X is not None:
            names = list(self.X.columns) if isinstance(self.X, pandas.DataFrame) else self.x_names
            mat = self.X.to_numpy(dtype=numpy.float64) if isinstance(self.X, pandas.DataFrame) else self.X
            if names is not None and mat.shape[0] == self.pred.shape[0]:
                for j, name in enumerate(names):
                    # a data frame names the column of a one-column matrix by its column name alone
                    cols[str(name) if len(names) == 1 else f'X.{name}'] = mat[:, j]
        z_values = getattr(self, 'Z', None)
        z_names = getattr(self, 'z_names', None)
        if isinstance(z_values, numpy.ndarray) and z_names is not None and z_values.ndim == 2 and z_values.shape[0] == self.pred.shape[0]:
            for j, name in enumerate(z_names):
                cols[f'Z.{name}'] = numpy.asarray(z_values)[:, j]
        return pandas.DataFrame(cols, index=list(self.slab) if self.slab is not None else None)

    def with_digits(self, digits: typing.Any) -> typing.Self:
        """The result printed with ``digits`` (a number, or a mapping with any of the keys of :data:`metafopy.mfopt.DIGIT_KEYS`)."""
        return with_digits(self, digits, self.digit_map or self.digits)  # type: ignore[arg-type]

    def __str__(self) -> str:
        """The predictions as ``print.list.rma`` shows them: fixed decimals (``digits``), one row per prediction."""
        digits = metafopy.mfopt.option_digits(None, self.digits)
        by_kind = metafopy.mfopt.digits_map(self.digit_map, digits)
        frame = self.to_frame()
        # print.list.rma leaves out `se` after a transformation and the prediction interval of the models that have none
        drop = (['se'] if self.transf else []) + ([] if self.has_pi else ['pi.lb', 'pi.ub'])
        frame = frame.drop(columns=drop)
        model_columns = [c for c in frame.columns if c not in PRINT_KINDS and c not in ('pred', 'prob')]
        # the model matrices (X.*, Z.*) are printed as the columns of a numeric data frame, the other columns with the digits
        columns = [
            r_numeric_column(frame[name].to_numpy(dtype=numpy.float64))
            if str(name).startswith(('X.', 'Z.')) or name in model_columns
            else [metafopy.printing.fmt_num(float(v), by_kind[PRINT_KINDS.get(str(name), 'est')]) for v in frame[name]]
            for name in frame.columns
        ]
        rows = [list(row) for row in zip(*columns, strict=True)]
        return metafopy.mfopt.option_space(
            '\n'
            + metafopy.printing.fmt_table([str(c) for c in frame.columns], rows, [str(i) for i in frame.index], gap=1, width=metafopy.printing.console_width())
            + '\n'
        )


# the kind of number (a key of ``digits``) print.list.rma formats each column as; the others are formatted as ``est``
PRINT_KINDS = {'se': 'se', 'ci.lb': 'ci', 'ci.ub': 'ci', 'pi.lb': 'ci', 'pi.ub': 'ci'}


def r_numeric_column(values: FloatArray) -> list[str]:
    """A numeric column as ``print.data.frame`` shows it: 7 significant digits and a common number of decimals."""
    decimals = 0
    for v in values:
        if math.isfinite(v):
            text = f'{v:.7g}'
            if 'e' not in text and '.' in text:
                decimals = max(decimals, len(text.split('.')[1]))
    return ['NA' if math.isnan(v) else f'{v:.{decimals}f}' for v in values]


def with_digits[R: PredictResult](result: R, digits: typing.Any, stored: int) -> R:
    """``result`` with the ``digits`` of its printout (R's ``out$digits``): ``digits``, else the ``digits`` option, else that of the model."""
    return metafopy.mfopt.with_digits(result, digits, stored)


CUTOFF_FUNCTIONS: dict[str, collections.abc.Callable[[float], float]] = {
    'abs': abs,
    'atanh': math.atanh,
    'cos': math.cos,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'log2': math.log2,
    'plogis': scipy.special.expit,
    'qlogis': scipy.special.logit,
    'sin': math.sin,
    'sqrt': math.sqrt,
    'tan': math.tan,
    'tanh': math.tanh,
}
CUTOFF_CONSTANTS = {'Inf': math.inf, 'pi': math.pi}


def evaluate_cutoff(text: str) -> float | None:
    """The number an R expression such as ``'log(0.8)'`` or ``'-0.5^2'`` evaluates to, ``None`` when it is not a number.

    metafor evaluates the cutoff of ``prob`` with ``eval(parse(text=))``; only arithmetic and a few common functions are supported here.
    """
    try:
        tree = ast.parse(text.strip().replace('^', '**'), mode='eval')
        return float(evaluate_node(tree.body))
    except SyntaxError, ValueError, TypeError, ArithmeticError, KeyError:
        return None


def evaluate_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        return CUTOFF_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = evaluate_node(node.operand)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return float(left**right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords:
        return float(CUTOFF_FUNCTIONS[node.func.id](evaluate_node(node.args[0])))
    raise ValueError('unsupported expression')


def prob_values(prob: str | float, pred: FloatArray, pi_se: FloatArray | None, pi_dist: str, pi_ddf: float | FloatArray | None) -> FloatArray | None:
    """Probabilities that a true outcome is below/above a value (``prob='>0.2'``, ``'<0'``, ``'>log(0.8)'``; a number means ``'>'``).

    ``None`` when the value cannot be parsed (metafor then drops ``prob``).
    """
    text = str(prob)
    lower_tail = '<' in text
    value = evaluate_cutoff(text.replace('<', '').replace('>', '').replace('=', ''))
    if value is None:
        return None
    if pi_se is None:
        return numpy.full(pred.shape, math.nan)
    z = (value - pred) / pi_se
    if pi_dist == 'norm':
        return numpy.asarray(scipy.stats.norm.cdf(z) if lower_tail else scipy.stats.norm.sf(z), dtype=numpy.float64)
    df = numpy.asarray(math.nan if pi_ddf is None else pi_ddf, dtype=numpy.float64)
    return numpy.asarray(scipy.stats.t.cdf(z, df) if lower_tail else scipy.stats.t.sf(z, df), dtype=numpy.float64)


def expand_hetvar(hetvar: numpy.typing.ArrayLike, k_new: int) -> FloatArray:
    """``hetvar`` recycled to ``k_new`` values (metafor's ``.expand1``).

    Raises:
        metafopy.errors.InputError: the length of ``hetvar`` does not match.
    """
    if numpy.asarray(hetvar).dtype == numpy.bool_:
        raise metafopy.errors.InputError("Argument 'hetvar' must be a numeric vector.")
    try:
        hv = numpy.atleast_1d(numpy.asarray(hetvar, dtype=numpy.float64))
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError("Argument 'hetvar' must be a numeric vector.") from err
    if hv.size == 1:
        hv = numpy.full(k_new, float(hv[0]))
    if hv.size != k_new:
        raise metafopy.errors.InputError(f"Length of the 'hetvar' argument ({hv.size}) does not match the number of predicted values ({k_new}).")
    return hv


def check_prob_adjust(prob: str | float | None, adjust: bool, vcov: bool = False) -> None:
    """Validate ``adjust`` and ``vcov`` (logicals) and the combination of ``adjust`` with ``prob``.

    Raises:
        metafopy.errors.InputError: ``adjust`` or ``vcov`` is not a logical, or ``prob`` is given with ``adjust=True``.
    """
    if not isinstance(adjust, bool | numpy.bool_):
        raise metafopy.errors.InputError("Argument 'adjust' must be a logical.")
    if not isinstance(vcov, bool | numpy.bool_):
        raise metafopy.errors.InputError("Argument 'vcov' must be a logical.")
    if prob is not None and adjust:
        raise metafopy.errors.InputError("Cannot use 'prob' argument when adjust=True.")


PREDTYPES = ('default', 'simple', 'riley', 't')
"""The values of ``predtype`` (partial matching is allowed, as in R)."""

ROW_FIELDS = ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub', 'prob')
SIDE_FIELDS = ('pi_se', 'X', 'Z')
NO_MODERATORS = 'Cannot specify new moderator values for models without moderators.'


def resolve_predtype(predtype: str | None, pi_type: str | None) -> str:
    """The prediction interval type: ``predtype``, else its old spelling ``pi_type``, else ``'default'`` (R: ``ddd$predtype``, ``ddd$pi.type``).

    Raises:
        metafopy.errors.InputError: the value is not one of ``'default'``, ``'simple'``, ``'riley'`` and ``'t'`` (or a prefix of one).
    """
    chosen = predtype if predtype is not None else pi_type if pi_type is not None else 'default'
    return metafopy.methods.common.match_arg(chosen.lower(), PREDTYPES, 'predtype')


def expand_newvi(newvi: numpy.typing.ArrayLike | None, k_new: int) -> FloatArray:
    """``newvi`` (sampling variances of new studies) recycled to ``k_new`` values; 0 when not given.

    Raises:
        metafopy.errors.InputError: the length of ``newvi`` does not match.
    """
    if newvi is None:
        return numpy.zeros(k_new)
    try:
        nv = numpy.atleast_1d(numpy.asarray(newvi, dtype=numpy.float64)).reshape(-1)
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError("Argument 'newvi' must be a numeric vector.") from err
    if nv.size == 1:
        nv = numpy.full(k_new, float(nv[0]))
    if nv.size != k_new:
        raise metafopy.errors.InputError(f"Length of the 'newvi' argument ({nv.size}) does not match the number of predicted values ({k_new}).")
    return nv


def interval_setup(
    predtype: str,
    vpred: FloatArray,
    crit: float | FloatArray,
    pi_dist: str,
    ddf: FloatArray,
    parms: int,
    p: int,
    alpha: float,
) -> tuple[FloatArray, float | FloatArray, str, FloatArray]:
    """The variance of the predicted values, critical value(s), distribution and degrees of freedom the prediction interval uses for ``predtype``.

    ``'simple'`` ignores the uncertainty of the predicted value and uses the normal distribution; ``'riley'`` uses a t distribution with
    ``ddf - parms + p`` degrees of freedom (Riley et al., 2011, doi:10.1111/j.1467-985X.2010.00696.x) and ``'t'`` one with ``ddf``; the
    degrees of freedom are at least 1 (``predict.rma``).
    """
    pi_ddf = ddf
    if predtype == 'simple':
        crit = float(scipy.stats.norm.isf(alpha / 2))
        vpred = numpy.zeros_like(vpred)
        pi_dist = 'norm'
    if predtype in ('riley', 't'):
        pi_ddf = ddf - parms + p if predtype == 'riley' else ddf
        pi_ddf = numpy.where(pi_ddf < 1, 1.0, pi_ddf)
        crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else math.nan for d in pi_ddf])
        pi_dist = 't'
    return vpred, crit, pi_dist, pi_ddf


def restrict_rows(res: PredictResult, keep: numpy.typing.NDArray[numpy.bool_], blank: numpy.typing.NDArray[numpy.bool_] | None) -> PredictResult:
    """Keep the rows ``keep`` of a prediction and set the rows ``blank`` to NaN: the ``na.action`` step of ``predict.rma``.

    ``blank`` flags (before ``keep`` is applied) the studies left out of the fit under ``na.exclude``; the variance-covariance matrix of the
    predictions loses the same rows and columns.
    """
    k_new = keep.shape[0]
    changes: dict[str, typing.Any] = {}
    for name in ROW_FIELDS + SIDE_FIELDS:
        value = getattr(res, name, None)
        if isinstance(value, numpy.ndarray) and value.ndim >= 1 and value.shape[0] == k_new:
            new = value[keep]
            if blank is not None and name in ROW_FIELDS:
                new = numpy.where(blank[keep], math.nan, new)
            changes[name] = new
    pi_ddf = res.pi_ddf
    if isinstance(pi_ddf, numpy.ndarray) and pi_ddf.ndim >= 1 and pi_ddf.shape[0] == k_new:
        changes['pi_ddf'] = pi_ddf[keep]
    ddf = res.ddf
    if isinstance(ddf, numpy.ndarray) and ddf.ndim >= 1 and ddf.shape[0] == k_new:
        changes['ddf'] = ddf[keep]
    if res.slab is not None and len(res.slab) == k_new:
        changes['slab'] = tuple(s for s, ok in zip(res.slab, keep, strict=True) if ok)
    vcov = res.vcov
    if isinstance(vcov, numpy.ndarray) and vcov.shape == (k_new, k_new):
        new_vcov = vcov[numpy.ix_(keep, keep)]
        if blank is not None:
            drop = blank[keep]
            new_vcov = new_vcov.copy()
            new_vcov[drop, :] = math.nan
            new_vcov[:, drop] = math.nan
        changes['vcov'] = new_vcov
    return dataclasses.replace(res, **changes)


def rows_to_keep(
    pred: FloatArray, not_na: numpy.typing.NDArray[numpy.bool_] | None, fitted_rows: bool, na_action: metafopy.methods.common.NAAction
) -> tuple[numpy.typing.NDArray[numpy.bool_], numpy.typing.NDArray[numpy.bool_] | None]:
    """The rows ``predict.rma`` returns and those it blanks.

    Under ``'omit'`` the fitted values keep the studies used in fitting and the predictions for ``newmods`` the rows that are not NaN; under
    ``'exclude'`` and ``'pass'`` all rows are kept, and ``'exclude'`` blanks the fitted values of the studies left out of the fit.

    Raises:
        metafopy.errors.InputError: ``na_action='fail'`` and studies were left out of the fit.
    """
    k_new = pred.shape[0]
    if na_action == 'fail' and not_na is not None and not not_na.all():
        raise metafopy.errors.InputError('Missing values in results.')
    keep = numpy.ones(k_new, dtype=numpy.bool_)
    blank: numpy.typing.NDArray[numpy.bool_] | None = None
    if na_action == 'omit':
        keep = not_na if fitted_rows and not_na is not None else ~numpy.isnan(pred)
    elif na_action == 'exclude' and fitted_rows and not_na is not None:
        blank = ~not_na
    return keep, blank


@functools.singledispatch
def predict(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> PredictResult:
    """Predicted values for a fitted model; see :func:`_predict_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'predict() not implemented for {type(model).__name__}')


def uni_location_model(x: metafopy.models.rma_uni.RMAResult, rows: metafopy.methods.fitted.Rows) -> LocationModel:
    """What ``predict.rma`` needs from an ``rma.uni`` fit, with the rows of its subset ``rows``."""
    return LocationModel(
        k=x.k,
        p=x.p,
        parms=x.parms,
        int_only=x.int_only,
        int_incl=x.int_incl,
        X=x.X,
        slab=x.slab,
        beta=x.beta,
        vb=x.vb,
        ddf=x.ddf,
        alpha=x.level,
        tau2=x.tau2,
        fixed=x.method in metafopy.models.tau2.FIXED_EFFECTS,
        method=x.method,
        sandwich=club_sandwich_of(x.robust),
        coef_test=None if x.robust is None else x.robust.coef_test,
        X_f=rows.X_f,
        not_na=rows.not_na,
        slab_f=rows.slab,
        coef_names=x.coef_names,
    )


@predict.register
def predict_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    newmods: NewMods | metafopy.methods.pairmat.PairmatSpec | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
    verbose: bool = False,
) -> PredictResult:
    """Predict from an ``rma.uni`` fit.

    Without ``newmods`` an intercept-only model gives the pooled estimate and a
    meta-regression gives the fitted value of each study. ``newmods`` holds
    moderator values without the intercept column (it is added unless
    ``intercept=False``) or the full set of ``p`` columns. ``transf`` is
    applied to each value (with ``targs`` if given; the integral and mode
    transformations in :mod:`metafopy.transf` default to ``targs={'tau2': tau2}``).

    ``hetvar`` replaces ``tau2`` in the prediction interval; ``newvi`` adds the sampling variances of new studies to it; ``prob`` (such as
    ``'>0.2'`` or ``'<0'``) adds the probability that a true outcome is above or
    below the value; ``addx`` adds the model matrix; ``adjust`` applies a
    Bonferroni correction to the intervals; ``vcov`` adds the
    variance-covariance matrix of the predictions. ``predtype`` (``pi_type`` is its old spelling) is ``'default'``, ``'simple'`` (the
    interval ignores the uncertainty of the predicted value, normal distribution), ``'riley'`` (t distribution with ``k - parms`` degrees of
    freedom, Riley et al., 2011, doi:10.1111/j.1467-985X.2010.00696.x) or ``'t'`` (t distribution with the degrees of freedom of the test).
    ``na_action`` stands for R's ``na.action`` option: for the fitted values of a meta-regression ``'omit'`` returns the studies used in
    fitting, ``'exclude'`` one row per study with NaN for those left out and ``'pass'`` one row per study computed from the moderator values it
    has; predictions for ``newmods`` drop (``'omit'``) or keep (the others) the rows that are NaN. ``tau2_levels`` and
    ``gamma2_levels`` are accepted, as in metafor, and have no effect. ``verbose`` is accepted for metafor's printing of the details of a failing
    ``clubSandwich`` contrast and has no effect here.

    R source: https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/predict.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    newmods = metafopy.methods.pairmat.resolve(newmods, model)
    return with_digits(
        location_prediction(
            uni_location_model(model, metafopy.methods.fitted.rows_of(model)),
            newmods,
            intercept,
            transf,
            targs,
            level,
            hetvar=hetvar,
            prob=prob,
            addx=addx,
            adjust=adjust,
            vcov=vcov,
            predtype=predtype,
            pi_type=pi_type,
            newvi=newvi,
            na_action=na_action,
        ),
        digits,
        model.digits,
    )


def club_sandwich_of(info: metafopy.models.robust_info.RobustInfo | None) -> metafopy.core.clubsandwich.Sandwich | None:
    """The ``clubSandwich`` variance estimator of a fit made robust with ``club_sandwich=True``, else ``None``.

    Raises:
        metafopy.errors.InputError: the fit used the saddlepoint test, which has no confidence intervals.
    """
    if info is None or info.robumethod != 'clubSandwich' or info.sandwich is None:
        return None
    if info.coef_test == 'saddlepoint':
        raise metafopy.errors.InputError('Cannot use method when saddlepoint correction was used.')
    return info.sandwich


def contrast_ddf(sandwich: metafopy.core.clubsandwich.Sandwich, X_new: FloatArray, coef_test: str | None) -> FloatArray:
    """Degrees of freedom of the predicted values of a ``clubSandwich`` fit (``clubSandwich::linear_contrast``); NaN for rows with missing values."""
    out = numpy.full(X_new.shape[0], math.nan)
    ok = numpy.isfinite(X_new).all(axis=1)
    if ok.any():
        out[ok] = metafopy.core.clubsandwich.contrast_df(sandwich, X_new[ok], coef_test or 'Satterthwaite')
    return out


@predict.register
def predict_rma_glmm(
    model: metafopy.models.glmm_result.GLMMResult,
    newmods: NewMods | metafopy.methods.pairmat.PairmatSpec | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
    verbose: bool = False,
) -> PredictResult:
    """Predict from an ``rma.glmm`` fit (``predict.rma``; arguments as in :func:`predict_rma_uni`).

    The coefficients are on the scale of the model's link (log odds ratios, log risk ratios, risk
    differences, log odds, proportions, log proportions or log rates). Without ``newmods`` a
    meta-regression gives the fitted value of each study that entered the model. Prediction
    intervals add ``tau2`` and are absent for the fixed-effects methods.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    newmods = metafopy.methods.pairmat.resolve(newmods, model)
    x = model
    rows = metafopy.methods.fitted.rows_of(x)
    return with_digits(
        location_prediction(
            LocationModel(
                k=x.k,
                p=x.p,
                parms=x.parms,
                int_only=x.int_only,
                int_incl=x.int_incl,
                X=x.X,
                slab=tuple(s for s, ok in zip(x.slab, x.tables.not_na, strict=True) if ok),
                beta=x.beta,
                vb=x.vb,
                ddf=x.ddf,
                alpha=x.level,
                tau2=x.tau2,
                fixed=x.method in metafopy.models.tau2.FIXED_EFFECTS,
                X_f=rows.X_f,
                not_na=rows.not_na,
                slab_f=rows.slab,
                coef_names=x.coef_names,
                method=x.method,
            ),
            newmods,
            intercept,
            transf,
            targs,
            level,
            hetvar=hetvar,
            prob=prob,
            addx=addx,
            adjust=adjust,
            vcov=vcov,
            predtype=predtype,
            pi_type=pi_type,
            newvi=newvi,
            na_action=na_action,
        ),
        digits,
        x.digits,
    )


@predict.register
def predict_fixed_binary(
    model: metafopy.models.binary.FixedBinaryResult,
    newmods: NewMods | metafopy.methods.pairmat.PairmatSpec | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
    verbose: bool = False,
) -> PredictResult:
    """Predict from an ``rma.mh`` or ``rma.peto`` fit (``predict.rma``; arguments as in :func:`predict_rma_uni`).

    These are equal-effects models without moderators, so the prediction is the pooled estimate with its
    confidence interval (and no prediction interval); ``transf`` (for example ``numpy.exp``) gives the
    odds ratio, risk ratio, ... instead of its logarithm. See
    https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r.

    Raises:
        metafopy.errors.InputError: ``newmods`` was given (the models have no moderators).
    """
    newmods = metafopy.methods.pairmat.resolve(newmods, model)
    x = model
    return with_digits(
        dataclasses.replace(
            location_prediction(
                LocationModel(
                    k=x.k,
                    p=1,
                    parms=x.parms,
                    int_only=True,
                    int_incl=True,
                    X=numpy.ones((x.k, 1)),
                    slab=(),
                    beta=x.beta,
                    vb=x.vb,
                    ddf=None,
                    alpha=x.level,
                    tau2=0.0,
                    fixed=True,
                    not_na=x.tables.not_na,
                    coef_names=('X',),  # the matrix of the new moderator values has no column names, so a data frame calls the column X
                ),
                newmods,
                intercept,
                transf,
                targs,
                level,
                hetvar=hetvar,
                prob=prob,
                addx=addx,
                adjust=adjust,
                vcov=vcov,
                predtype=predtype,
                pi_type=pi_type,
                newvi=newvi,
                na_action=na_action,
            ),
            method='FE',
            has_pi=False,
        ),
        digits,
        x.digits,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class LocationModel:
    """What ``predict.rma`` needs from a model with a single ``tau2`` (``rma.uni``, ``rma.glmm``, ``rma.mh`` or ``rma.peto``).

    ``X_f``, ``not_na`` and ``slab_f`` are the model matrix, the flags of the studies used in fitting and the labels over all rows of the subset
    (metafor's ``X.f``, ``not.na`` and ``slab``); without them the rows used in fitting are all there is.
    """

    k: int
    p: int
    parms: int
    int_only: bool
    int_incl: bool
    X: FloatArray
    slab: tuple[str, ...]
    beta: FloatArray
    vb: FloatArray
    ddf: int | None
    alpha: float
    tau2: float
    fixed: bool
    sandwich: metafopy.core.clubsandwich.Sandwich | None = None
    coef_test: str | None = None
    X_f: FloatArray | None = None
    not_na: numpy.typing.NDArray[numpy.bool_] | None = None
    slab_f: tuple[str, ...] | None = None
    coef_names: tuple[str, ...] | None = None
    method: str | None = None


def string_names(labels: list[typing.Any]) -> list[str] | None:
    """The labels as names, or ``None`` when they are not all strings (an index of row numbers carries no names)."""
    return labels if all(isinstance(label, str) for label in labels) else None


def named_newmods(newmods: typing.Any) -> tuple[numpy.typing.ArrayLike, list[str] | None]:
    """The values of ``newmods`` and the names of its entries (a dict, a ``pandas.Series`` or a ``pandas.DataFrame``) or columns, if it has any."""
    if isinstance(newmods, collections.abc.Mapping):
        return list(newmods.values()), [str(name) for name in newmods]
    if isinstance(newmods, pandas.Series):
        return newmods.to_numpy(), string_names(list(newmods.index))
    if isinstance(newmods, pandas.DataFrame):
        return newmods.to_numpy(), string_names(list(newmods.columns))
    return newmods, None


def prediction_slab(newmods: typing.Any, k_new: int, x_p: int, int_incl: bool) -> tuple[str, ...]:
    """The row labels of predictions for ``newmods`` (metafor's ``rnames``): the names of a vector holding one value per prediction (a model
    with one moderator), the row names of a data frame, else the positions 1, 2, ...; a single unnamed prediction has the empty label."""
    labels: list[str] | None = None
    one_mod = x_p == (2 if int_incl else 1)
    if isinstance(newmods, pandas.DataFrame):
        labels = string_names(list(newmods.index))
    elif isinstance(newmods, pandas.Series | collections.abc.Mapping) and one_mod:
        labels = string_names(list(newmods.index if isinstance(newmods, pandas.Series) else newmods))
    if labels is not None and len(labels) == k_new:
        return tuple(labels)
    return ('',) if k_new == 1 else tuple(str(i + 1) for i in range(k_new))


def match_names(names: list[str], coef_names: collections.abc.Sequence[str]) -> list[int]:
    """The position in ``coef_names`` of each name; a name may be shortened to any subsequence of the name it stands for (as in metafor).

    Raises:
        metafopy.errors.InputError: a name is not found, matches several variables or the same variable is given twice, or a variable is missing.
    """
    pos: list[int] = []
    for name in names:
        dists = [len(target) - len(name) if is_subsequence(name, target) else math.inf for target in coef_names]
        best = min(dists, default=math.inf)
        if math.isinf(best):
            raise metafopy.errors.InputError(f"Could not find variable '{name}' in the model.")
        if dists.count(best) > 1:
            raise metafopy.errors.InputError(f"Could not match up variable '{name}' uniquely to a variable in the model.")
        pos.append(dists.index(best))
    if len(set(pos)) != len(pos):
        dups = ', '.join(dict.fromkeys(name for i, name in enumerate(names) if pos[i] in pos[:i]))
        raise metafopy.errors.InputError(f'Found multiple matches for the same variable name ({dups}).')
    if len(pos) != len(coef_names):
        missing = [name for i, name in enumerate(coef_names) if i not in pos]
        listed = ', '.join(missing[:3]) + (', ...' if len(missing) > 3 else '')
        what = 'this variable' if len(missing) == 1 else 'these variables'
        raise metafopy.errors.InputError(f"Argument 'newmods' does not specify values for {what}: {listed}")
    return pos


def is_subsequence(short: str, long: str) -> bool:
    chars = iter(long)
    return all(ch in chars for ch in short)


def new_moderators(
    x_p: int,
    int_incl: bool,
    newmods: typing.Any,
    intercept: bool | None,
    coef_names: collections.abc.Sequence[str] | None = None,
) -> FloatArray:
    """The model matrix for new moderator values (with the intercept column added unless ``intercept=False``).

    A ``pandas.Series``, ``dict`` or ``pandas.DataFrame`` with names is matched to the moderators of the model by name when the model has
    more than one moderator (``coef_names``), also by partial names (any subsequence of a name stands for it), as in metafor's ``predict()``.

    Raises:
        metafopy.errors.InputError: the dimensions do not match the model or the names cannot be matched to the moderators.
    """
    values, names = named_newmods(newmods)
    arr = numpy.asarray(values, dtype=numpy.float64)
    n_mods = x_p - 1 if int_incl else x_p
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None] if n_mods == 1 else arr[None, :]
    if names is not None and coef_names is not None and n_mods > 1 and len(names) == arr.shape[1] and all(names):
        mod_names = list(coef_names)[1:] if int_incl else list(coef_names)
        if int_incl and len(names) == x_p and list(names) == list(coef_names):
            mod_names = list(coef_names)
        arr = arr[:, numpy.argsort(match_names(names, mod_names), kind='stable')]
    if int_incl and arr.shape[1] == x_p and intercept is not None:
        warnings.warn("Arguments 'intercept' ignored when 'newmods' includes 'p' columns.", stacklevel=4)
    if int_incl and arr.shape[1] == x_p - 1:
        col = 1.0 if intercept is None or intercept else 0.0
        arr = numpy.column_stack([numpy.full(arr.shape[0], col), arr])
    if arr.shape[1] != x_p:
        raise metafopy.errors.InputError(f'dimensions of newmods ({arr.shape[1]} columns) do not match the model ({x_p})')
    return arr


def location_prediction(
    x: LocationModel,
    newmods: NewMods | None,
    intercept: bool | None,
    transf: collections.abc.Callable[..., typing.Any] | None,
    targs: typing.Any,
    level: float | None,
    *,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> PredictResult:
    """``predict.rma`` for models with a single ``tau2`` (``rma.uni``, ``rma.glmm``, ``rma.mh``, ``rma.peto`` and selection models).

    Raises:
        metafopy.errors.InputError: ``newmods`` or another argument does not match the model.
    """
    metafopy.methods.common.check_na_action(na_action)
    ptype = resolve_predtype(predtype, pi_type)
    p = x.p
    slab: tuple[str, ...] | None = None
    if x.int_only and newmods is not None:
        raise metafopy.errors.InputError(NO_MODERATORS)
    fitted_rows = newmods is None and not x.int_only
    if newmods is None:
        X_new = numpy.ones((1, 1)) if x.int_only else (x.X if x.X_f is None else x.X_f)
        if not x.int_only:
            slab = x.slab if x.slab_f is None else x.slab_f
    else:
        X_new = new_moderators(p, x.int_incl, newmods, intercept, x.coef_names)
        slab = prediction_slab(newmods, X_new.shape[0], p, x.int_incl)

    if slab is None and X_new.shape[0] == 1:
        slab = ('',)

    check_prob_adjust(prob, adjust, vcov)
    alpha = x.alpha if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    k_new = X_new.shape[0]
    if adjust:
        alpha = alpha / k_new

    pred = X_new @ x.beta
    vpred = numpy.einsum('ij,jk,ik->i', X_new, x.vb, X_new)
    vpred = numpy.where(vpred < 0, numpy.nan, vpred)
    se = numpy.sqrt(vpred)
    crit: float | FloatArray
    ddf_rows: FloatArray | None = None
    if x.sandwich is not None:
        ddf_rows = contrast_ddf(x.sandwich, X_new, x.coef_test)
        crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else numpy.nan for d in ddf_rows])
    elif x.ddf is not None:
        crit = float(scipy.stats.t.isf(alpha / 2, x.ddf)) if x.ddf > 0 else numpy.nan
    else:
        crit = float(scipy.stats.norm.isf(alpha / 2))
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se
    ddf_base = ddf_rows if ddf_rows is not None else numpy.full(pred.shape, float(x.k - p if x.ddf is None else x.ddf))
    pi_dist = 'norm' if x.ddf is None and ddf_rows is None else 't'
    vcov_pred = X_new @ x.vb @ X_new.T if vcov else None
    if vcov_pred is not None:
        vcov_pred = (vcov_pred + vcov_pred.T) / 2
    vpred_pi, crit_pi, pi_dist, pi_ddf = interval_setup(ptype, vpred, crit, pi_dist, ddf_base, x.parms, p, alpha)
    het = x.tau2 if hetvar is None else expand_hetvar(hetvar, k_new)
    pi_se = numpy.sqrt(vpred_pi + het + expand_newvi(newvi, k_new))
    prob_out = None if prob is None else prob_values(prob, pred, pi_se, pi_dist, pi_ddf)
    pi_lb = pred - crit_pi * pi_se
    pi_ub = pred + crit_pi * pi_se
    if x.fixed:
        # metafor drops prediction intervals for equal/fixed/common-effects models
        pi_lb = numpy.full_like(pred, numpy.nan)
        pi_ub = numpy.full_like(pred, numpy.nan)

    if transf is not None:
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (metafopy.transf.apply_transf(a, transf, targs, x.tau2) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = numpy.full_like(pred, numpy.nan)
        ci_lb, ci_ub = numpy.minimum(ci_lb, ci_ub), numpy.maximum(ci_lb, ci_ub)
        pi_lb, pi_ub = numpy.minimum(pi_lb, pi_ub), numpy.maximum(pi_lb, pi_ub)
        if metafopy.transf.is_tau2_transf(transf):
            # metafor drops the prediction interval after an integral/mode transformation
            pi_lb = numpy.full_like(pred, numpy.nan)
            pi_ub = numpy.full_like(pred, numpy.nan)

    out = PredictResult(
        pred=pred,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        pi_se=pi_se,
        pi_dist=pi_dist,
        pi_ddf=pi_ddf if pi_dist == 't' else None,
        ddf=(ddf_rows if ddf_rows is not None else None if x.ddf is None else numpy.full(pred.shape, float(x.ddf))),
        prob=prob_out,
        X=X_new.copy() if addx else None,
        x_names=tuple(x.coef_names) if addx and x.coef_names is not None else None,
        vcov=vcov_pred if transf is None else None,
        method=x.method,
        transf=transf is not None,
        has_pi=not (x.fixed or (transf is not None and metafopy.transf.is_tau2_transf(transf))),
    )
    keep, blank = rows_to_keep(pred, x.not_na, fitted_rows, na_action)
    return out if keep.all() and blank is None else restrict_rows(out, keep, blank)


def level_indices(
    levels: collections.abc.Sequence[str | int] | str | int,
    labels: collections.abc.Sequence[str],
    k_new: int,
    what: str,
) -> list[int]:
    """0-based variance-component positions from labels or 0-based integers, recycled to ``k_new``.

    Raises:
        metafopy.errors.InputError: a level does not exist or the number of levels does not match.
    """
    seq = [levels] if isinstance(levels, str | int) else list(levels)
    out: list[int] = []
    for v in seq:
        if isinstance(v, str):
            if v not in labels:
                raise metafopy.errors.InputError(f"Non-existing levels specified via '{what}' argument.")
            out.append(list(labels).index(v))
        else:
            if not 0 <= int(v) < len(labels):
                raise metafopy.errors.InputError(f"Non-existing values specified via '{what}' argument.")
            out.append(int(v))
    if len(out) == 1:
        out = out * k_new
    if len(out) != k_new:
        raise metafopy.errors.InputError(f"Length of the '{what}' argument ({len(out)}) does not match the number of predicted values ({k_new}).")
    return out


def mv_ddf_rows(x: metafopy.models.mv_result.RMAMVResult, X_new: FloatArray) -> FloatArray:
    """The degrees of freedom of each predicted value of an ``rma.mv`` fit: the smallest of the coefficients it involves (``k - p`` where unknown)."""
    d = numpy.atleast_1d(numpy.asarray(x.ddf, dtype=numpy.float64))
    k_new = X_new.shape[0]
    if d.size == 1:
        rows = numpy.full(k_new, float(d[0]))
    else:
        rows = numpy.array([float(numpy.min(d[X_new[i] != 0])) if numpy.any(X_new[i] != 0) else math.inf for i in range(k_new)])
    return numpy.where(numpy.isnan(rows), float(x.k - x.p), rows)


def component_variances(values: FloatArray | None, idx: numpy.typing.NDArray[numpy.intp] | None) -> FloatArray | None:
    """The variance components ``values`` picked by the 0-based positions ``idx`` (NaN where the position is unknown, -1)."""
    if idx is None or values is None:
        return None
    vals = numpy.asarray(values, dtype=numpy.float64)
    return numpy.where(idx >= 0, vals[numpy.maximum(idx, 0)], math.nan)


def fitted_levels(group: metafopy.models.mv_structs.Group | None, n_comp: int, not_na: numpy.typing.NDArray[numpy.bool_]) -> numpy.typing.NDArray[numpy.intp]:
    """The positions of the variance components of the studies of a fit, over all rows of its subset (-1 for rows left out whose level is lost)."""
    k_f = not_na.shape[0]
    if n_comp == 1 or group is None or group.inner_factor is None:
        return numpy.zeros(k_f, dtype=numpy.intp)
    out = numpy.full(k_f, -1, dtype=numpy.intp)
    out[not_na] = [int(c) for c in group.inner_factor.codes]
    return out


@predict.register
def predict_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    newmods: NewMods | metafopy.methods.pairmat.PairmatSpec | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
    verbose: bool = False,
) -> PredictResult:
    """Predict from an ``rma.mv`` fit (``predict.rma`` for ``rma.mv`` objects).

    Prediction intervals add the sum of the ``sigma2`` values and, when present, the ``tau2``
    and ``gamma2`` values selected via ``tau2_levels`` and ``gamma2_levels`` (labels of the
    inner factor or 0-based positions). They are NaN when several ``tau2``/``gamma2`` values exist
    and no levels are given. ``hetvar`` replaces the sum of the variance components (the levels
    are then ignored, with a warning); ``prob``, ``addx``, ``vcov``, ``predtype``, ``newvi`` and ``na_action`` are as in :func:`predict_rma_uni`.
    A model without moderators cannot be given ``newmods``. For the fitted values of studies left out of the fit (``na_action='pass'``), the
    ``tau2`` and ``gamma2`` value of a study whose inner level is lost is NaN when the model has several of them.

    R source: https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/predict.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    newmods = metafopy.methods.pairmat.resolve(newmods, model)
    x = model
    metafopy.methods.common.check_na_action(na_action)
    ptype = resolve_predtype(predtype, pi_type)
    check_prob_adjust(prob, adjust, vcov)
    if hetvar is not None and (tau2_levels is not None or gamma2_levels is not None):
        tau2_levels = gamma2_levels = None
        warnings.warn("Arguments 'tau2.levels' and 'gamma2.levels' ignored when specifying the 'hetvar' argument.", stacklevel=2)
    if x.int_only and newmods is not None:
        raise metafopy.errors.InputError(NO_MODERATORS)
    p = x.p
    slab: tuple[str, ...] | None = None
    general = any(v in ('GEN', 'GDIAG') for v in x.struct)
    g_labels = list(x.group_g.levels_f[0]) if x.group_g is not None else []
    h_labels = list(x.group_h.levels_f[0]) if x.group_h is not None else []
    tau2_idx: numpy.typing.NDArray[numpy.intp] | None = None
    gamma2_idx: numpy.typing.NDArray[numpy.intp] | None = None
    fitted_rows = newmods is None and not x.int_only
    not_na: numpy.typing.NDArray[numpy.bool_] | None = x.not_na
    if newmods is None:
        if x.int_only and not general:
            counts = [
                len(v) if isinstance(v, collections.abc.Sequence) and not isinstance(v, str) else 1 for v in (tau2_levels, gamma2_levels) if v is not None
            ]
            X_new = numpy.ones((max(counts, default=1), 1))
            k_new = X_new.shape[0]
            if x.with_g:
                tau2_idx = numpy.zeros(k_new, dtype=numpy.intp) if x.tau2s == 1 else level_array(tau2_levels, g_labels, k_new, 'tau2_levels')
            if x.with_h:
                gamma2_idx = numpy.zeros(k_new, dtype=numpy.intp) if x.gamma2s == 1 else level_array(gamma2_levels, h_labels, k_new, 'gamma2_levels')
        elif x.int_only:
            X_new = numpy.ones((1, 1))
            k_new = 1
            if x.with_g:
                tau2_idx = numpy.zeros(1, dtype=numpy.intp) if x.tau2s == 1 else level_array(tau2_levels, g_labels, 1, 'tau2_levels')
            if x.with_h:
                gamma2_idx = numpy.zeros(1, dtype=numpy.intp) if x.gamma2s == 1 else level_array(gamma2_levels, h_labels, 1, 'gamma2_levels')
        else:
            if tau2_levels is not None or gamma2_levels is not None:
                warnings.warn("Arguments 'tau2.levels' and 'gamma2.levels' ignored when obtaining fitted values.", stacklevel=2)
            rows = metafopy.methods.fitted.rows_of(x)
            X_new = rows.X_f
            slab = rows.slab
            k_new = X_new.shape[0]
            if x.with_g:
                tau2_idx = fitted_levels(x.group_g, x.tau2s, rows.not_na)
            if x.with_h:
                gamma2_idx = fitted_levels(x.group_h, x.gamma2s, rows.not_na)
    else:
        X_new = new_moderators(p, x.int_incl, newmods, intercept, x.coef_names)
        k_new = X_new.shape[0]
        slab = prediction_slab(newmods, k_new, p, x.int_incl)
        if x.with_g:
            tau2_idx = numpy.zeros(k_new, dtype=numpy.intp) if x.tau2s == 1 else level_array(tau2_levels, g_labels, k_new, 'tau2_levels')
        if x.with_h:
            gamma2_idx = numpy.zeros(k_new, dtype=numpy.intp) if x.gamma2s == 1 else level_array(gamma2_levels, h_labels, k_new, 'gamma2_levels')

    k_new = X_new.shape[0]
    if slab is None and k_new == 1:
        slab = ('',)
    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    if adjust:
        alpha = alpha / k_new
    pred = X_new @ x.beta
    vpred = numpy.einsum('ij,jk,ik->i', X_new, x.vb, X_new)
    vpred = numpy.where(vpred < 0, numpy.nan, vpred)
    se = numpy.sqrt(vpred)
    sandwich = club_sandwich_of(x.robust)
    ddf_row: FloatArray
    if sandwich is not None:
        ddf_row = contrast_ddf(sandwich, X_new, x.robust.coef_test if x.robust is not None else None)
        crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else math.nan for d in ddf_row])
        pi_dist = 't'
    else:
        ddf_row = mv_ddf_rows(x, X_new)
        if x.test == 'z':
            crit = numpy.full(k_new, float(scipy.stats.norm.isf(alpha / 2)))
            pi_dist = 'norm'
        else:
            crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else math.nan for d in ddf_row])
            pi_dist = 't'
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se
    vcov_pred = X_new @ x.vb @ X_new.T if vcov else None
    if vcov_pred is not None:
        vcov_pred = (vcov_pred + vcov_pred.T) / 2
    vpred_pi, crit_pi, pi_dist, pi_ddf = interval_setup(ptype, vpred, crit, pi_dist, ddf_row, x.parms, p, alpha)
    nv = expand_newvi(newvi, k_new)

    pi_ok = True
    if hetvar is not None:
        extra: FloatArray = expand_hetvar(hetvar, k_new)
    else:
        extra = numpy.full(k_new, float(numpy.sum(x.sigma2)) if x.sigma2 is not None else 0.0)
        if x.with_g:
            tau2_vals = component_variances(x.tau2, tau2_idx)
            if tau2_vals is None:
                pi_ok = False
            else:
                extra = extra + tau2_vals
        if x.with_h:
            gamma2_vals = component_variances(x.gamma2, gamma2_idx)
            if gamma2_vals is None:
                pi_ok = False
            else:
                extra = extra + gamma2_vals
    pi_se_all: FloatArray | None = None
    if pi_ok:
        pi_se_all = numpy.sqrt(vpred_pi + extra + nv)
        pi_lb = pred - crit_pi * pi_se_all
        pi_ub = pred + crit_pi * pi_se_all
    else:
        pi_lb = numpy.full(k_new, numpy.nan)
        pi_ub = numpy.full(k_new, numpy.nan)
    prob_out = None if prob is None else prob_values(prob, pred, pi_se_all, pi_dist, pi_ddf)

    if transf is not None:
        tau2_value = float(numpy.sum(x.tau2)) if x.tau2 is not None else 0.0
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (metafopy.transf.apply_transf(a, transf, targs, tau2_value) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = numpy.full_like(pred, numpy.nan)
        ci_lb, ci_ub = numpy.minimum(ci_lb, ci_ub), numpy.maximum(ci_lb, ci_ub)
        pi_lb, pi_ub = numpy.minimum(pi_lb, pi_ub), numpy.maximum(pi_lb, pi_ub)

    out = PredictResult(
        pred=pred,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        pi_se=pi_se_all,
        pi_dist=pi_dist,
        pi_ddf=pi_ddf if pi_dist == 't' else None,
        ddf=ddf_row if pi_dist == 't' else None,
        prob=prob_out,
        X=X_new.copy() if addx else None,
        x_names=tuple(x.coef_names) if addx and x.coef_names is not None else None,
        vcov=vcov_pred if transf is None else None,
        method=x.method,
        transf=transf is not None,
        has_pi=not (
            x.method in metafopy.models.tau2.FIXED_EFFECTS
            or any(g is not None and g.struct in ('GEN', 'GDIAG') for g in (x.group_g, x.group_h))
            or (transf is not None and metafopy.transf.is_tau2_transf(transf))
        ),
    )
    keep, blank = rows_to_keep(pred, not_na, fitted_rows, na_action)
    return with_digits(out if keep.all() and blank is None else restrict_rows(out, keep, blank), digits, model.digits)


def level_array(
    levels: collections.abc.Sequence[str | int] | str | int | None, labels: collections.abc.Sequence[str], k_new: int, what: str
) -> numpy.typing.NDArray[numpy.intp] | None:
    """:func:`level_indices` as an array, ``None`` when no levels were given."""
    if levels is None:
        return None
    return numpy.asarray(level_indices(levels, labels, k_new, what), dtype=numpy.intp)
