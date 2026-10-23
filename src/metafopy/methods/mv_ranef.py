"""Best linear unbiased predictions of the random effects of an ``rma.mv`` fit (metafor's ``ranef.rma.mv``).

For a term with random effects ``u`` of covariance ``D`` and design ``Z`` the BLUPs are ``D Z' W (y - X b)`` with
``W = M^-1``, and their prediction variances ``D - D Z' W (I - H) Z D`` with ``H`` the hat matrix (Raudenbush & Bryk,
1985, doi:10.3102/10769986010002075). Ports https://github.com/wviechtb/metafor/blob/master/R/ranef.rma.mv.r; see
https://wviechtb.github.io/metafor/reference/ranef.html.
"""

import collections.abc
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse
import scipy.stats

import metafopy.core.labelled
import metafopy.effects.measures
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.mv_diagnostics
import metafopy.methods.ranef
import metafopy.models.mv_result
import metafopy.models.mv_structs
import metafopy.models.rma_uni

__all__ = ['ranef_rma_mv']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type MV = metafopy.models.mv_result.RMAMVResult
type Transf = collections.abc.Callable[..., typing.Any]

EPS = float(numpy.finfo(float).eps)
# auditme: accept[non-alphabetical-literal] the column order of metafor's data frames
COLUMNS = ['intrcpt', 'se', 'pi.lb', 'pi.ub']


class RanefList(metafopy.core.labelled.NamedList):
    """The list of data frames ``ranef()`` returns for an ``rma.mv`` fit (a dict by term); it prints as R prints a list: ``$term`` above each frame."""


def term_crit(x: MV, alpha: float, nlevels: int) -> float:
    """The critical value of the prediction intervals of one term (``.ddf.calc(..., beta=FALSE)`` for t-tests)."""
    if x.test == 'z':
        return float(scipy.stats.norm.isf(alpha / 2))
    p = x.p_eff or x.p
    ddf = float(x.k - p) if x.dfs == 'residual' else float(nlevels - p)
    return float(scipy.stats.t.isf(alpha / 2, max(ddf, 1.0)))


def frame(pred: FloatArray, vpred: FloatArray, crit: float, rows: collections.abc.Sequence[str] | None = None) -> pandas.DataFrame:
    pred = pred.copy()
    pred[numpy.abs(pred) < 100 * EPS] = 0.0
    se = numpy.sqrt(numpy.diag(vpred))
    return pandas.DataFrame({'intrcpt': pred, 'se': se, 'pi.lb': pred - crit * se, 'pi.ub': pred + crit * se}, index=None if rows is None else list(rows))


def recover_r(D_S: FloatArray, Z: FloatArray) -> FloatArray:
    """The (scaled) correlation matrix ``R`` of the levels from ``D_S = Z R Z'`` (``Z`` is an indicator matrix)."""
    counts = Z.sum(axis=0)
    return numpy.asarray((Z.T @ D_S @ Z) / numpy.outer(counts, counts), dtype=numpy.float64)


def g_term(
    x: MV,
    group: metafopy.models.mv_structs.Group,
    struct: str,
    Gmat: FloatArray,
    W: FloatArray,
    ImH: FloatArray,
    ei: FloatArray,
    alpha: float,
) -> tuple[pandas.DataFrame, FloatArray]:
    """The BLUPs of an ``~ inner | outer`` term, one row per observed combination of the inner and outer levels.

    Raises:
        metafopy.errors.InternalError: the term of a non-spatial structure has no inner factor.
    """
    Z1, Z2 = metafopy.methods.mv_diagnostics.dense(group.Z1), metafopy.methods.mv_diagnostics.dense(group.Z2)
    G = (Z1 @ Gmat @ Z1.T) * (Z2 @ Z2.T)
    GW = G @ W
    pred = GW @ ei
    vpred = G - GW @ ImH @ G
    out = frame(pred, vpred, term_crit(x, alpha, group.outer.nlevels))
    outer = [str(v) for v in group.outer.labels()]
    if struct in metafopy.models.mv_structs.SPATIAL:
        ids = [str(int(i) + 1) for i, ok in zip(x.ids, x.not_na, strict=True) if ok]
        width = max(len(s) for s in ids)
        names = [f'{i.rjust(width)} | {o}' for i, o in zip(ids, outer, strict=True)]
    else:
        if group.inner_factor is None:
            raise metafopy.errors.InternalError('inner factor missing')
        inner = [str(v) for v in group.inner_factor.labels()]
        width = max(len(s) for s in inner)
        names = [f'{i.rjust(width)} | {o}' for i, o in zip(inner, outer, strict=True)]
    keep = numpy.array([name not in set(names[:j]) for j, name in enumerate(names)], dtype=numpy.bool_)
    out = out.iloc[numpy.flatnonzero(keep)]
    out.index = [n for n, k in zip(names, keep, strict=True) if k]
    vpred = vpred[numpy.ix_(keep, keep)]
    if struct not in metafopy.models.mv_structs.SPATIAL and group.inner_factor is not None:
        order = numpy.lexsort((group.inner_factor.codes[keep], group.outer.codes[keep]))
        out = out.iloc[order]
        vpred = vpred[numpy.ix_(order, order)]
    return out, vpred


def term_name(group: metafopy.models.mv_structs.Group) -> str:
    """``paste0(formula, collapse="")`` of the term's formula: ``~inner | outer``."""
    return '~' + str(group.formula).removeprefix('~').lstrip()


def ranef_rma_mv(
    object: MV,
    level: float | None = None,
    digits: int | None = None,
    transf: Transf | None = None,
    targs: typing.Any = None,
    verbose: bool = False,
    *,
    expand: bool = False,
    vcov: bool = False,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> dict[str, typing.Any] | pandas.DataFrame | None:
    """BLUPs of the random effects of an ``rma.mv`` fit: a dict with one data frame (``intrcpt``, ``se``, ``pi.lb``, ``pi.ub``) per term.

    The ``~ 1 | id`` terms are keyed by their grouping variables (``s_names``), the ``~ inner | outer`` terms by their
    formulas (``'~inner | outer'``), as in metafor. ``level`` sets the level of the prediction intervals; ``transf`` (with
    ``targs``) is applied to the predictions and interval bounds and drops the ``se`` column. ``expand=True`` gives the
    ``~ 1 | id`` terms one row per observation; ``vcov=True`` returns ``{'pred': ..., 'vcov': ...}`` with the prediction
    variance-covariance matrices. For ``struct='GEN'``/``'GDIAG'`` metafor returns (only) the matrix of BLUPs of the first
    ``~ inner | outer`` term, without standard errors; so does this function. ``digits`` and ``verbose`` have no effect
    (``digits`` is not used by metafor either; the progress messages are not reproduced). ``None`` for models without
    random effects.

    Args:
        object: A model fitted by [rma_mv][metafopy.rma_mv].
        level: Level of the prediction intervals in percent (e.g. 95), or a value below 1 read as in metafor (0.95 and 0.05 both give 95 percent).
            ``None`` (the default) uses the level of the model.
        digits: Accepted for compatibility; it has no effect.
        transf: A function applied to the predictions and interval bounds (for example ``numpy.exp``); the ``se`` column is then dropped.
            ``None`` (the default) applies no transformation.
        targs: Extra argument passed to ``transf`` as ``transf(x, targs)``. ``None`` (the default) calls ``transf(x)``.
        verbose: Accepted for compatibility; it has no effect.
        expand: Give the ``~ 1 | id`` terms one row per observation instead of one per level (default ``False``).
        vcov: Also return the variance-covariance matrices of the predictions (default ``False``).
        na_action: How observations dropped for missing values are handled: ``'omit'`` (the default; the ``~ 1 | id`` terms have a row for each
            level that occurs in the observations used), ``'exclude'`` or ``'pass'`` (those terms also have a row of NaN for each level that
            only occurs in dropped observations) or ``'fail'`` (raise an error if observations were dropped; metafor's ``ranef.rma.mv``
            has no such check). It has no effect on the ``~ inner | outer`` terms.

    Returns:
        A dict (named list) with one data frame per random-effects term, with columns ``intrcpt`` (the predicted random effect), ``se``, ``pi.lb`` and
        ``pi.ub``. With ``vcov=True`` a dict ``{'pred': ..., 'vcov': ...}`` whose ``vcov`` holds the matrices by term. For ``struct='GEN'`` or
        ``'GDIAG'`` a single data frame of BLUPs without standard errors. ``None`` for a model without random effects.

    Raises:
        metafopy.errors.InputError: the model has user-defined weights, ``na_action`` is unknown or ``'fail'`` with missing values.

    Metafor reference:
        [ranef](https://wviechtb.github.io/metafor/reference/ranef.html)
    """
    x = object
    expand, vcov = metafopy.methods.common.is_true(expand), metafopy.methods.common.is_true(vcov)
    metafopy.methods.common.check_na_action(na_action)
    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    if x.W is not None:
        raise metafopy.errors.InputError('Extraction of random effects not available for models with non-standard weights.')
    W = x.weight_matrix()
    ImH = numpy.eye(x.k) - x.hat_matrix()
    ei = x.yi - x.X @ x.beta
    out: RanefList = RanefList()
    vc: dict[str, FloatArray] = {}

    if x.with_s and x.sigma2 is not None:
        for j, name in enumerate(x.s_names):
            Z = metafopy.methods.mv_diagnostics.dense(x.Z_S[j])
            nlev = x.s_nlevels[j]
            R = recover_r(metafopy.methods.mv_diagnostics.dense(x.D_S[j]), Z) if x.Rfix[j] else numpy.eye(nlev)
            D = float(x.sigma2[j]) * R
            DZtW = D @ Z.T @ W
            vpred = D - DZtW @ ImH @ Z @ D
            pred = frame(DZtW @ ei, vpred, term_crit(x, alpha, nlev), x.s_levels[j])
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            if na_action in ('exclude', 'pass'):
                levels_f = x.s_levels_f[j] if x.s_levels_f else x.s_levels[j]
                present = set(x.s_levels[j])
                full = pandas.DataFrame(numpy.nan, index=list(levels_f), columns=COLUMNS)
                full.iloc[[i for i, lv in enumerate(levels_f) if lv in present]] = pred.to_numpy()
                pred = full
            elif vcov:
                vc[name] = vpred
            if expand:
                rows = numpy.argmax(Z, axis=1)
                pred = pred.iloc[rows]
                pred.index = list(metafopy.effects.measures.make_unique_dot1([x.s_levels[j][r] for r in rows]))
            out[name] = pred

    if x.with_g and x.group_g is not None and x.G is not None:
        if x.struct[0] in metafopy.models.mv_structs.GENERAL:
            return gen_blups(x, x.group_g, x.G, W, ei)
        pred, vpred = g_term(x, x.group_g, x.struct[0], x.G, W, ImH, ei, alpha)
        out[term_name(x.group_g)] = pred
        vc[term_name(x.group_g)] = vpred

    if x.with_h and x.group_h is not None and x.H is not None and x.struct[1] not in metafopy.models.mv_structs.GENERAL:
        pred, vpred = g_term(x, x.group_h, x.struct[1], x.H, W, ImH, ei, alpha)
        out[term_name(x.group_h)] = pred
        vc[term_name(x.group_h)] = vpred

    if transf is not None:
        for name, pred in out.items():
            cols = {c: metafopy.methods.common.apply_each(pred[c].to_numpy(dtype=numpy.float64), transf, targs) for c in ('intrcpt', 'pi.lb', 'pi.ub')}
            out[name] = pandas.DataFrame(cols, index=pred.index)

    if na_action == 'fail' and not x.not_na.all():
        raise metafopy.errors.InputError('Missing values in results.')
    if not out:
        return None
    if vcov:
        return {'pred': out, 'vcov': vc}
    return out


def gen_blups(x: MV, group: metafopy.models.mv_structs.Group, Gmat: FloatArray, W: FloatArray, ei: FloatArray) -> pandas.DataFrame:
    """metafor's BLUPs for ``struct='GEN'``/``'GDIAG'``: ``G Z1' W e`` within each level of the outer factor (no standard errors)."""
    Z1 = metafopy.methods.mv_diagnostics.dense(group.Z1)
    codes = group.outer.codes
    pred = numpy.zeros((group.outer.nlevels, Z1.shape[1]))
    for j in range(group.outer.nlevels):
        incl = numpy.flatnonzero(codes == j)
        pred[j] = Gmat @ Z1[incl].T @ W[numpy.ix_(incl, incl)] @ ei[incl]
    pred[numpy.abs(pred) < 100 * EPS] = 0.0
    return pandas.DataFrame(pred, index=list(group.outer.levels), columns=[str(c) for c in group.inner.columns])


metafopy.methods.ranef.ranef.register(metafopy.models.mv_result.RMAMVResult, ranef_rma_mv)  # type: ignore[arg-type]
