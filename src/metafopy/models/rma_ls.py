"""Location-scale models (``rma.uni(..., scale = ~ ...)`` in metafor, class ``rma.ls``).

In a location-scale model both the size of the average true effect and the amount of heterogeneity
depend on moderators: ``y_i = x_i' beta + u_i + e_i`` with ``u_i ~ N(0, tau_i^2)`` and
``log(tau_i^2) = z_i' alpha`` (``link='log'``) or ``tau_i^2 = z_i' alpha`` (``link='identity'``).
Viechtbauer and Lopez-Lopez (2022), Location-scale models for meta-analysis, Research Synthesis Methods,
doi:10.1002/jrsm.1562. The model is fitted by :func:`metafopy.rma_uni` when its ``scale`` argument is given;
this module normalises the scale model matrix and calls :mod:`metafopy.models.rma_ls_fit`.
"""

import collections.abc
import dataclasses
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.formula
import metafopy.metadata
import metafopy.models.rma_ls_fit
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.tau2

__all__ = ['ScaleSpec', 'fit_scale', 'scale_matrix']

type ScaleSpec = str | numpy.typing.ArrayLike | pandas.DataFrame
type FloatArray = numpy.typing.NDArray[numpy.float64]


def scale_matrix(scale: ScaleSpec, md: metafopy.metadata.MetaData) -> tuple[FloatArray, tuple[str, ...]]:
    """The model matrix of the scale part for all ``k_all`` supplied rows and its column names.

    A string is a one-sided formula (``'~ ni100 + imag'``); other inputs are used as given (without
    adding an intercept), as in metafor.

    Raises:
        metafopy.errors.InputError: the formula needs data, or the matrix has the wrong number of rows.
    """
    k_all = md.k_all
    if isinstance(scale, str):
        if '~' not in scale:
            raise metafopy.errors.InputError("the 'scale' argument must be a one-sided formula such as '~ x1 + x2' or a model matrix")
        rhs = scale.partition('~')[2].strip()
        if rhs == '1':
            # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
            return numpy.ones((k_all, 1)), ('intrcpt',)
        if md.data is None:
            raise metafopy.errors.InputError('a formula for scale needs data')
        Z, names, z_missing = metafopy.formula.design_matrix(scale, md.data)
        Z[z_missing] = numpy.nan
    elif isinstance(scale, pandas.DataFrame):
        names = tuple(str(c) for c in scale.columns)
        Z = scale.to_numpy(dtype=float)
    else:
        Z = numpy.asarray(scale, dtype=numpy.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        names = tuple(f'Z{j + 1}' for j in range(Z.shape[1]))
    if Z.shape[0] != k_all:
        raise metafopy.errors.InputError(
            f"Number of rows in the model matrix specified via the 'scale' argument ({Z.shape[0]}) does not match the length of the outcome vector ({k_all})."
        )
    return Z.astype(numpy.float64), metafopy.metadata.make_unique(names)


def fit_scale(
    md: metafopy.metadata.MetaData,
    scale: ScaleSpec,
    *,
    method: str | collections.abc.Sequence[str],
    weighted: bool,
    test: metafopy.models.rma_uni.Test,
    level: float,
    btt: metafopy.models.rma_uni.Btt,
    att: metafopy.models.rma_uni.Btt,
    tau2: float | None,
    link: str,
    optbeta: bool,
    alpha: numpy.typing.ArrayLike | None,
    beta: numpy.typing.ArrayLike | None,
    control: metafopy.models.rma_ls_result.LSControl | collections.abc.Mapping[str, typing.Any] | None,
    na_action: metafopy.metadata.NAAction,
    measure: str,
    dfs: float | collections.abc.Sequence[float] | None = None,
    abbrev: bool = False,
    skiphes: bool = False,
) -> metafopy.models.rma_ls_result.RMALSResult:
    """Fit a location-scale model to the inputs ``md`` with scale model ``scale``.

    Studies with missing values in the scale model matrix are left out (or an error is raised for
    ``na_action='fail'``) in addition to those with missing values in the outcomes or location moderators.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    meth = method if isinstance(method, str) else method[0]
    if meth not in metafopy.models.tau2.ESTIMATORS:
        raise metafopy.errors.InputError("Unknown 'method' specified.")
    if tau2 is not None and not numpy.isnan(tau2):
        warnings.warn("Argument 'tau2' ignored for location-scale models.", stacklevel=3)
    Z_all, z_names = scale_matrix(scale, md)
    Z = Z_all[md.keep]
    Z_f = Z_all[md.keep if md.na_rows is None else md.keep | md.na_rows]
    missing = numpy.isnan(Z).any(axis=1)
    if missing.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('missing values in data')
        sel = ~missing
        keep = md.keep.copy()
        keep[numpy.flatnonzero(md.keep)[missing]] = False
        # the rows left out for a missing scale value are rows of the subset (R's k.f) too, so that na_action='pass' returns them
        na_rows = numpy.zeros_like(md.keep) if md.na_rows is None else md.na_rows.copy()
        na_rows[numpy.flatnonzero(md.keep)[missing]] = True
        md = dataclasses.replace(
            md,
            na_rows=na_rows,
            yi=md.yi[sel],
            vi=md.vi[sel],
            X=md.X[sel],
            slab=tuple(numpy.asarray(md.slab, dtype=object)[sel].tolist()),
            keep=keep,
            weights=None if md.weights is None else md.weights[sel],
            ni=None if md.ni is None else md.ni[sel],
        )
        Z = Z[sel]
    if md.na_rows is not None and md.na_rows.any():
        metafopy.models.rma_uni.warn_na_omitted(int(md.na_rows.sum()))
    res = metafopy.models.rma_ls_fit.fit_rma_ls(
        md,
        Z,
        z_names,
        method=meth,
        weighted=weighted,
        test=test,
        level=level,
        btt=btt,
        att=att,
        link=link,
        optbeta=optbeta,
        alpha=alpha,
        beta=beta,
        control=metafopy.models.rma_ls_fit.make_ls_control(control),  # type: ignore[arg-type]
        formula_scale=scale if isinstance(scale, str) else None,
        dfs=dfs,
        abbrev=abbrev,
        skiphes=skiphes,
    )
    return dataclasses.replace(res, measure=measure, Z_f=Z_f)
