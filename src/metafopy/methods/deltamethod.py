"""The (multivariate) delta method for functions of model coefficients (metafor's ``deltamethod``).

R sources: https://github.com/wviechtb/metafor/blob/master/R/deltamethod.r with ``coef.deltamethod.r``,
``vcov.deltamethod.r`` and ``print.deltamethod.r``. Documentation:
https://wviechtb.github.io/metafor/reference/deltamethod.html. The first-order method is the usual
``J V J'``; the second- and third-order approximations add the terms of the Taylor expansion with the
Hessians and the third-derivative tensor of the function (for normally distributed estimates). As in
metafor, the derivatives are numerical (``calculus::derivative()`` and ``calculus::hessian()``, ported in
:mod:`metafopy.core.calculus`).

Differences from R that follow Python conventions:

- ``fun`` with a single parameter is given the whole coefficient vector; with several parameters it is
  called with one coefficient per parameter (R matches ``formalArgs(fun)``);
- the names of the transformed coefficients come from a returned :class:`pandas.Series` or mapping;
  other return values are numbered ``'1'``, ``'2'``, ... as in R;
- ``digits`` is an ``int`` or a mapping with the keys ``est``, ``se``, ``test``, ``pval`` and ``ci``.
"""

import collections.abc
import dataclasses
import inspect
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.special
import scipy.stats

import metafopy.core.calculus
import metafopy.effects.summary
import metafopy.errors
import metafopy.methods.ls_common
import metafopy.methods.vif
import metafopy.mfopt
import metafopy.printing

__all__ = ['DeltaMethodResult', 'deltamethod']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
DIGIT_KEYS = ('est', 'se', 'test', 'pval', 'ci')


@dataclasses.dataclass(frozen=True)
class DeltaMethodResult:
    """Transformed coefficients with their standard errors, Wald-type tests and confidence intervals.

    ``tab`` holds ``coef``, ``se``, ``zval``, ``pval``, ``ci_lb`` and ``ci_ub`` (one row per transformed
    coefficient), ``vb`` their variance-covariance matrix, ``level`` the alpha level of the intervals and
    ``test`` is always ``'z'``.
    """

    tab: pandas.DataFrame
    vb: FloatArray
    level: float
    digits: dict[str, int]
    test: str = 'z'

    def coef(self) -> pandas.Series:
        """The transformed coefficients (``coef.deltamethod``)."""
        return pandas.Series(self.tab['coef'].to_numpy(), index=list(self.tab.index), name='coef')

    def vcov(self) -> pandas.DataFrame:
        """Their variance-covariance matrix (``vcov.deltamethod``)."""
        names = list(self.tab.index)
        return pandas.DataFrame(self.vb, index=names, columns=names)

    def summary(
        self,
        digits: int | collections.abc.Sequence[float] | collections.abc.Mapping[str, int] | None = None,
        signif_stars: bool = True,
        signif_legend: bool | None = None,
    ) -> str:
        """The table as ``print.deltamethod`` shows it."""
        return format_result(
            self, self.digits if digits is None else get_digits(digits), signif_stars, signif_stars if signif_legend is None else signif_legend
        )

    def __str__(self) -> str:
        return self.summary()


def get_digits(
    digits: int | collections.abc.Sequence[float] | collections.abc.Mapping[str, int] | None, default: collections.abc.Mapping[str, int] | None = None
) -> dict[str, int]:
    """metafor's ``.get.digits()`` for the keys used here (missing keys keep their defaults)."""
    out = dict(default) if default is not None else dict.fromkeys(DIGIT_KEYS, 4)
    if digits is None:
        return out
    if isinstance(digits, collections.abc.Mapping):
        out.update({key: int(val) for key, val in digits.items() if key in out})
        return out
    return dict.fromkeys(DIGIT_KEYS, metafopy.mfopt.reduce_digits(digits))


def arity(fun: collections.abc.Callable[..., typing.Any]) -> int:
    """The number of parameters of ``fun`` (R's ``length(formalArgs(fun))``); a NumPy ufunc counts its inputs, and 1 when it cannot be inspected."""
    if isinstance(fun, numpy.ufunc):
        return int(fun.nin)
    try:
        params = inspect.signature(fun).parameters.values()
    except TypeError, ValueError:
        return 1
    return sum(1 for par in params if par.kind in (par.POSITIONAL_ONLY, par.POSITIONAL_OR_KEYWORD, par.KEYWORD_ONLY))


def as_vector(value: typing.Any) -> tuple[FloatArray, list[str] | None]:
    """The values returned by ``fun`` and their names (``None`` when unnamed).

    Raises:
        metafopy.errors.InputError: the function does not return numbers.
    """
    names: list[str] | None = None
    if isinstance(value, pandas.Series):
        names = [str(n) for n in value.index]
        value = value.to_numpy()
    elif isinstance(value, collections.abc.Mapping):
        names = [str(n) for n in value]
        value = list(value.values())
    try:
        arr = numpy.asarray(value, dtype=numpy.float64)
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError('Specified function does not return an atomic vector.') from err
    if arr.ndim > 1:
        raise metafopy.errors.InputError('Specified function does not return an atomic vector.')
    return arr.reshape(-1), names


def is_vector(x: typing.Any) -> bool:
    return isinstance(x, (int, float, numpy.number, numpy.ndarray, pandas.Series, list, tuple))


def coef_vcov(x: typing.Any, vcov: numpy.typing.ArrayLike | pandas.DataFrame | None) -> tuple[FloatArray, typing.Any]:
    """The coefficients and their var-cov matrix: ``x`` itself and ``vcov``, or ``coef(x)`` and ``vcov(x)`` of a model.

    Raises:
        metafopy.errors.InputError: ``vcov`` is missing for a vector, or ``x`` has no ``coef()``/``vcov()``.
    """
    if is_vector(x):
        if vcov is None:
            raise metafopy.errors.InputError("Must specify the 'vcov' argument when 'x' is a vector.")
        return numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64)).reshape(-1), vcov
    if not callable(getattr(x, 'coef', None)):
        raise metafopy.errors.InputError("Cannot extract coefficients via coef() from 'x'.")
    if not callable(getattr(x, 'vcov', None)):
        raise metafopy.errors.InputError("Cannot extract var-cov matrix via vcov() from 'x'.")
    if vcov is not None:
        warnings.warn("Argument 'vcov' ignored when 'x' is a model object.", stacklevel=3)
    coef = x.coef()
    vb = x.vcov()
    # location-scale models: the first component is beta
    if isinstance(coef, collections.abc.Mapping):
        coef = coef['beta']
    if isinstance(vb, collections.abc.Mapping):
        vb = vb['beta']
    return numpy.asarray(coef, dtype=numpy.float64).reshape(-1), vb


def check_vcov(vcov: typing.Any, p: int) -> FloatArray:
    """``vcov`` as a symmetric ``p x p`` matrix (a vector gives a diagonal matrix).

    Raises:
        metafopy.errors.InputError: ``vcov`` is not a square symmetric matrix of the right size.
    """
    vb = numpy.asarray(vcov.to_numpy() if isinstance(vcov, pandas.DataFrame) else vcov, dtype=numpy.float64)
    if vb.ndim <= 1 or vb.shape[0] == 1 or vb.shape[1] == 1:
        vb = numpy.diag(vb.reshape(-1))
    if vb.ndim != 2 or vb.shape[0] != vb.shape[1]:
        raise metafopy.errors.InputError("Argument 'vcov' must be a square matrix.")
    # R's isSymmetric(): all.equal(x, t(x)), a mean relative difference below 1.5e-8
    scale = float(numpy.abs(vb).mean()) if vb.size else 0.0
    if scale > 0 and float(numpy.abs(vb - vb.T).mean()) / scale >= 1.5e-8:
        raise metafopy.errors.InputError("Argument 'vcov' must be a symmetric matrix.")
    if vb.shape[0] != p:
        raise metafopy.errors.InputError(f"Length of the 'coef' vector ({p}) does not match the dimensions of 'vcov' ({vb.shape[0]}x{vb.shape[0]}).")
    return vb


def deltamethod(
    x: typing.Any,
    vcov: numpy.typing.ArrayLike | pandas.DataFrame | None = None,
    fun: collections.abc.Callable[..., typing.Any] | None = None,
    order: int = 1,
    level: float | None = None,
    H0: numpy.typing.ArrayLike = 0,
    digits: int | collections.abc.Sequence[float] | collections.abc.Mapping[str, int] | None = None,
) -> DeltaMethodResult:
    """Apply the (multivariate) delta method to a function of a set of coefficients.

    ``x`` is a vector of coefficients (``vcov`` then gives their var-cov matrix, or their variances as a
    vector) or a fitted model with ``coef()`` and ``vcov()`` methods (for a location-scale model the
    ``beta`` coefficients are used). ``fun`` maps the coefficients to one or more transformed values;
    ``order`` (1, 2 or 3) is the order of the Taylor approximation. ``level`` defaults to the model's
    level (95 otherwise) and ``H0`` is the value(s) under the null hypothesis of the Wald-type tests.

    See https://wviechtb.github.io/metafor/reference/deltamethod.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.EstimationError: the function or its derivatives cannot be evaluated.
    """
    if fun is None or not callable(fun):
        raise metafopy.errors.InputError("Argument 'fun' must be a function.")
    if order not in (1, 2, 3):
        raise metafopy.errors.InputError("Argument 'order' must be equal to 1, 2, or 3.")
    coef, vcov_in = coef_vcov(x, vcov)
    if level is None:
        model_level = None if is_vector(x) else getattr(x, 'level', None)
        level = 95.0 if model_level is None else float(model_level)
    digit_map = get_digits(digits)
    p = coef.shape[0]
    vb = check_vcov(vcov_in, p)

    nargs = arity(fun)
    if nargs != 1 and nargs != p:
        raise metafopy.errors.InputError(f'Number of function arguments ({nargs}) does not match the number of coefficients ({p}).')

    def call(b: FloatArray) -> typing.Any:
        return fun(b.copy()) if nargs == 1 else fun(*(float(v) for v in b))

    def vector_fun(b: FloatArray) -> FloatArray:
        return as_vector(call(b))[0]

    try:
        value = call(coef)
    except Exception as err:
        raise metafopy.errors.EstimationError('Error when applying the function to the coefficient(s).') from err
    est, names = as_vector(value)
    q = est.shape[0]
    labels = names if names is not None else [str(i + 1) for i in range(q)]

    with numpy.errstate(all='ignore'):
        try:
            grad = metafopy.core.calculus.derivative_each(vector_fun, coef, 1)
        except Exception as err:
            raise metafopy.errors.EstimationError('Error when computing the gradient.') from err
        grad = grad.reshape(q, p)
        if order == 1:
            vt = grad @ vb @ grad.T
        else:
            try:
                hess = metafopy.core.calculus.hessian(vector_fun, coef).reshape(q, p, p)
            except Exception as err:
                raise metafopy.errors.EstimationError('Error when computing the Hessian.') from err
            if order == 2:
                vt = grad @ vb @ grad.T
                for a in range(q):
                    for b in range(a, q):
                        val = vt[a, b] + 0.5 * numpy.trace(hess[a] @ vb @ hess[b] @ vb)
                        vt[a, b] = vt[b, a] = val
            else:
                vt = third_order(vector_fun, coef, vb, grad, hess)

    h0 = numpy.atleast_1d(numpy.asarray(H0, dtype=numpy.float64)).reshape(-1)
    if h0.shape[0] == 1:
        h0 = numpy.repeat(h0, q)
    if h0.shape[0] != q:
        raise metafopy.errors.InputError(f"Length of the 'H0' argument ({h0.shape[0]}) does not match the number of transformed coefficients ({q}).")

    alpha = metafopy.effects.summary.level_to_alpha(level)
    crit = float(scipy.stats.norm.isf(alpha / 2))
    with numpy.errstate(invalid='ignore', divide='ignore'):
        se = numpy.sqrt(numpy.diag(vt))
        zval = (est - h0) / se
    pval = 2 * scipy.special.ndtr(-numpy.abs(zval))
    tab = pandas.DataFrame({'coef': est, 'se': se, 'zval': zval, 'pval': pval, 'ci_lb': est - crit * se, 'ci_ub': est + crit * se}, index=labels)
    return DeltaMethodResult(tab=tab, vb=vt, level=alpha, digits=digit_map)


def third_order(f: metafopy.core.calculus.VectorFunction, coef: FloatArray, vb: FloatArray, grad: FloatArray, hess: FloatArray) -> FloatArray:
    """The third-order delta-method var-cov matrix (metafor's ``order=3`` branch).

    Raises:
        metafopy.errors.EstimationError: the third derivatives cannot be computed.
    """
    q, p = grad.shape
    third = numpy.full((q, p, p, p), numpy.nan)
    for i in range(p):
        for j in range(i, p):
            for k in range(j, p):
                ords = numpy.bincount([i, j, k], minlength=p).tolist()
                try:
                    val = metafopy.core.calculus.derivative_mixed(f, coef, ords)
                except Exception as err:
                    raise metafopy.errors.EstimationError('Error when computing the third-derivative tensor.') from err
                if val.shape[0] != q:
                    raise metafopy.errors.EstimationError('Incorrect length of the computed third derivatives.')
                for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
                    third[:, a, b, c] = val
    u = numpy.einsum('aijk,ij->ak', third, vb)
    k3 = numpy.kron(vb, numpy.kron(vb, vb))
    vt = numpy.full((q, q), numpy.nan)
    for a in range(q):
        # R's as.vector() of a p x p x p array runs over the first index fastest
        ta = third[a].reshape(-1, order='F')
        for b in range(a, q):
            tb = third[b].reshape(-1, order='F')
            term11 = grad[a] @ vb @ grad[b]
            term22 = 0.5 * numpy.trace(hess[a] @ vb @ hess[b] @ vb)
            term13 = 0.5 * (grad[a] @ vb @ u[b] + grad[b] @ vb @ u[a])
            term33 = (ta @ k3 @ tb) / 6 + (u[a] @ vb @ u[b]) / 4
            vt[a, b] = vt[b, a] = term11 + term22 + term13 + term33
    return vt


def format_result(x: DeltaMethodResult, digits: dict[str, int], signif_stars: bool, signif_legend: bool) -> str:
    tab = x.tab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    header = ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub']
    rows = [
        [
            metafopy.printing.fmt_num(float(tab['coef'].iloc[j]), digits['est']),
            metafopy.printing.fmt_num(float(tab['se'].iloc[j]), digits['se']),
            metafopy.printing.fmt_num(float(tab['zval'].iloc[j]), digits['test']),
            metafopy.printing.fmt_pval(float(tab['pval'].iloc[j]), digits['pval']),
            metafopy.printing.fmt_num(float(tab['ci_lb'].iloc[j]), digits['ci']),
            metafopy.printing.fmt_num(float(tab['ci_ub'].iloc[j]), digits['ci']),
        ]
        for j in range(tab.shape[0])
    ]
    if signif_stars:
        header.append('')
        stars = metafopy.methods.ls_common.center_stars(float(v) for v in tab['pval'])
        for row, star in zip(rows, stars, strict=True):
            row.append(star)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    out = ['']
    if len(rows) == 1:
        out.extend(line.rstrip() for line in metafopy.methods.vif.format_vector(header, rows[0]))
    else:
        out.append(metafopy.printing.fmt_table(header, rows, row_names=[str(n) for n in tab.index]))
    if signif_legend:
        out.extend(['', '---', "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"])
    out.append('')
    return '\n'.join(out)
