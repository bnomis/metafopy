"""The coefficient table of a fitted model as a data frame (metafor's ``coef(summary(res))``, i.e. ``coef.summary.rma``).

The frame has one row per coefficient and the columns ``estimate``, ``se``, ``zval`` (``tval`` and ``df`` for fits that use the t
distribution, which is the case for ``test='t'``, ``'knha'`` and ``'adhoc'`` and for robust fits), ``pval``, ``ci.lb`` and ``ci.ub``. Location-scale
models give a dict with the tables of the location (``'beta'``) and scale (``'alpha'``) coefficients, selection models one with the tables of
the coefficients (``'beta'``) and the selection parameters (``'delta'``, rows ``delta`` or ``delta.1``, ``delta.2``, ...); ``type`` picks one of them.

R sources: https://github.com/wviechtb/metafor/blob/master/R/coef.summary.rma.r and
https://github.com/wviechtb/metafor/blob/master/R/coef.rma.r; reference page: https://wviechtb.github.io/metafor/reference/coef.summary.rma.html.
"""

import typing

import numpy
import numpy.typing
import pandas

import metafopy.methods.fitview
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['coef_summary']

type FloatArray = numpy.typing.NDArray[numpy.float64]


def table(
    names: typing.Sequence[str],
    estimate: FloatArray,
    se: FloatArray,
    stat: FloatArray,
    pval: FloatArray,
    ci_lb: FloatArray,
    ci_ub: FloatArray,
    df: FloatArray | None,
) -> pandas.DataFrame:
    """One coefficient table; ``df`` is ``None`` for the z-test tables."""
    cols: dict[str, FloatArray] = {'estimate': estimate, 'se': se, 'tval' if df is not None else 'zval': stat}
    if df is not None:
        cols['df'] = df
    cols.update({'pval': pval, 'ci.lb': ci_lb, 'ci.ub': ci_ub})
    return pandas.DataFrame({k: numpy.asarray(v, dtype=numpy.float64).reshape(-1) for k, v in cols.items()}, index=list(names))


def degrees_of_freedom(x: typing.Any, ddf: typing.Any, count: int) -> FloatArray | None:
    """The ``df`` column: ``None`` when the fit uses the z-test, else the (recycled) degrees of freedom."""
    robust = getattr(x, 'robust', None)
    if robust is not None and isinstance(x, metafopy.models.rma_uni.RMAResult):
        return numpy.asarray(robust.dfs, dtype=numpy.float64)
    if getattr(x, 'test', 'z') not in ('knha', 'adhoc', 't'):
        return None
    return numpy.broadcast_to(numpy.asarray(numpy.nan if ddf is None else ddf, dtype=numpy.float64), (count,)).copy()


def beta_table(x: typing.Any) -> pandas.DataFrame:
    df = degrees_of_freedom(x, getattr(x, 'ddf', None), len(x.beta))
    return table(x.coef_names, x.beta, x.se, x.zval, x.pval, x.ci_lb, x.ci_ub, df)


def alpha_table(x: metafopy.models.rma_ls_result.RMALSResult) -> pandas.DataFrame:
    df = degrees_of_freedom(x, x.ddf_alpha, len(x.alpha))
    return table(x.alpha_names, x.alpha, x.se_alpha, x.zval_alpha, x.pval_alpha, x.ci_lb_alpha, x.ci_ub_alpha, df)


def delta_table(x: metafopy.models.sel_result.SelmodelResult) -> pandas.DataFrame:
    return table(x.delta_names, x.delta, x.se_delta, x.zval_delta, x.pval_delta, x.ci_lb_delta, x.ci_ub_delta, None)


@typing.overload
def coef_summary(object: typing.Any, type: typing.Literal['beta', 'alpha', 'delta']) -> pandas.DataFrame: ...


@typing.overload
def coef_summary(object: typing.Any, type: None = None) -> pandas.DataFrame | dict[str, pandas.DataFrame]: ...


def coef_summary(object: typing.Any, type: typing.Literal['beta', 'alpha', 'delta'] | None = None) -> pandas.DataFrame | dict[str, pandas.DataFrame]:
    """The coefficient table of a fitted model as a data frame, as metafor's ``coef(summary(res))``.

    The columns are ``estimate``, ``se``, ``zval`` (``tval`` and ``df`` when the t distribution is used), ``pval``, ``ci.lb`` and ``ci.ub``. For a
    location-scale model the result is ``{'beta': ..., 'alpha': ...}``, for a selection model ``{'beta': ..., 'delta': ...}``. ``type='beta'`` gives
    the table of the (location) coefficients of any fit, ``type='alpha'`` that of the scale coefficients of a location-scale model and
    ``type='delta'`` that of the selection parameters of a selection model; as in R, ``type='alpha'`` or ``'delta'`` for a fit without that part
    gives the table of the coefficients.

    Raises:
        metafopy.errors.ArgumentError: ``object`` is not a fitted model.
    """
    if object.__class__.__name__ in ('PermutestResult', 'LSPermutestResult'):
        tables = object.to_frame()
        return typing.cast('pandas.DataFrame | dict[str, pandas.DataFrame]', tables[type] if isinstance(tables, dict) and type is not None else tables)
    metafopy.methods.fitview.check_class(object)
    beta = beta_table(object)
    if type == 'beta':
        return beta
    if isinstance(object, metafopy.models.rma_ls_result.RMALSResult):
        alpha = alpha_table(object)
        return alpha if type == 'alpha' else {'beta': beta, 'alpha': alpha}
    if isinstance(object, metafopy.models.sel_result.SelmodelResult):
        delta = delta_table(object)
        return delta if type == 'delta' else {'beta': beta, 'delta': delta}
    return beta
