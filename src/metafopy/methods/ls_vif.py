"""Variance inflation factors for location-scale models (metafor's ``vif.rma`` for ``rma.ls`` objects).

The (generalised) variance inflation factors of the location coefficients and, when the scale model has predictors, of the
scale coefficients (Viechtbauer & Lopez-Lopez, 2022, doi:10.1002/jrsm.1564; Fox & Monette, 1992,
doi:10.1080/01621459.1992.10475190), computed as for :func:`metafopy.vif`. ``sim`` is not available for location-scale models.
"""

import collections.abc
import dataclasses
import math
import typing

import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.methods.ls_common
import metafopy.methods.vif
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni

__all__ = ['LSVIFResult', 'vif_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class LSVIFResult:
    """(G)VIFs of the location (``beta``) and scale (``alpha``) coefficients of a location-scale model."""

    beta: metafopy.methods.vif.VIFResult
    alpha: metafopy.methods.vif.VIFResult

    def __str__(self) -> str:
        return f'\nLocation Coefficients:\n{self.beta}\nScale Coefficients:\n{self.alpha}'


def coef_table(estimate: FloatArray, se: FloatArray, zval: FloatArray, pval: FloatArray, ci_lb: FloatArray, ci_ub: FloatArray, names: collections.abc.Sequence[str], ddf: int | None) -> pandas.DataFrame:
    cols: dict[str, typing.Any] = {'estimate': estimate, 'se': se}
    if ddf is not None:
        cols['tval'] = zval
        cols['df'] = numpy.full(len(names), ddf)
    else:
        cols['zval'] = zval
    cols.update({'pval': pval, 'ci.lb': ci_lb, 'ci.ub': ci_ub})
    return pandas.DataFrame(cols, index=list(names))


def compvif_ls(
    group: collections.abc.Sequence[int],
    vcov: FloatArray,
    int_incl: bool,
    refit: collections.abc.Callable[[FloatArray], FloatArray | None] | None,
    X: FloatArray,
) -> tuple[float, list[int]]:
    """GVIF of the coefficients ``group`` (0-based) of one part of the model and the indices actually used.

    Without ``refit`` the GVIF comes from the correlation matrix of the estimates; with ``refit`` (a function returning the
    variance-covariance matrix of the part after refitting the model with a given model matrix) the model is refitted with
    only those coefficients (plus the intercept) and the variance-covariance blocks are compared.
    """
    if refit is None:
        if int_incl:
            V = vcov[1:, 1:]
            b = [i - 1 for i in group if i > 0]
        else:
            V = vcov
            b = list(group)
        rb = metafopy.methods.vif.cov2cor(V)
        rest = [i for i in range(V.shape[0]) if i not in b]
        gvif = float(numpy.linalg.det(rb[numpy.ix_(b, b)]) * numpy.linalg.det(rb[numpy.ix_(rest, rest)]) / numpy.linalg.det(rb))
        return gvif, [i + 1 for i in b] if int_incl else b

    b = [i for i in group if i > 0] if int_incl else list(group)
    Xb = X[:, b]
    if int_incl:
        Xb = numpy.column_stack([numpy.ones(X.shape[0]), Xb])
    v_sub = refit(Xb)
    if v_sub is None:
        return math.nan, b
    v_sub = v_sub[1:, 1:] if int_incl else v_sub
    return float(numpy.linalg.det(vcov[numpy.ix_(b, b)]) / numpy.linalg.det(v_sub)), b


def part_vif(
    res: metafopy.models.rma_ls_result.RMALSResult,
    *,
    which: str,
    spec_arg: metafopy.methods.vif.BttArg,
    table: bool,
    reestimate: bool,
) -> metafopy.methods.vif.VIFResult:
    """The (G)VIFs of the location (``which='beta'``) or the scale (``which='alpha'``) coefficients."""
    if which == 'beta':
        n, int_incl, names, vcov, M = res.p, res.int_incl, res.coef_names, res.vb, res.X
    else:
        n, int_incl, names, vcov, M = res.q, res.Z_int_incl, res.alpha_names, res.va, res.Z
    arg = 'btt' if which == 'beta' else 'att'
    missing = spec_arg is None
    specs: list[metafopy.methods.vif.BttSpec]
    if spec_arg is None:
        specs = [j for j in range(1 if int_incl else 0, n)]
    elif isinstance(spec_arg, str | int):
        specs = [spec_arg]
    elif any(isinstance(b, str | collections.abc.Sequence) for b in spec_arg):
        specs = list(spec_arg)  # type: ignore[arg-type]
    else:
        specs = [list(spec_arg)]  # type: ignore[arg-type]
    groups = [metafopy.models.rma_uni.set_btt(s, n, int_incl, names) for s in specs]
    if int_incl and any(g == (0,) for g in groups):
        raise metafopy.errors.InputError(f"Cannot compute VIF(s) for the specified '{arg}' argument.")

    refit: collections.abc.Callable[[FloatArray], FloatArray | None] | None = None
    if reestimate:
        mask = numpy.where(res.alpha_fix, res.alpha, numpy.nan)

        def refit_part(Mb: FloatArray) -> FloatArray | None:
            beta_arg = numpy.where(res.beta_fix, res.beta, numpy.nan) if res.optbeta and which == 'alpha' else None
            fit = metafopy.methods.ls_common.refit_ls(
                res, alpha=mask if which == 'beta' else None, beta=beta_arg, skiphes=False, X=Mb if which == 'beta' else None, Z=Mb if which == 'alpha' else None
            )
            if fit is None:
                return None
            return fit.vb if which == 'beta' else fit.va

        refit = refit_part

    entries: list[metafopy.methods.vif.VIFEntry] = []
    for spec, g in zip(specs, groups, strict=True):
        gvif, used = compvif_ls(g, vcov, int_incl, refit, M)
        m = len(used)
        entries.append(
            metafopy.methods.vif.VIFEntry(
                spec=metafopy.methods.vif.format_spec(spec),
                coefs=metafopy.methods.vif.format_btt(i + 1 for i in used),
                coefname=names[used[0]] if m == 1 else '',
                m=m,
                vif=gvif,
                sif=gvif ** (1 / (2 * m)),
            )
        )
    tab: pandas.DataFrame | None = None
    if table and missing:
        if which == 'beta':
            tab = coef_table(res.beta, res.se, res.zval, res.pval, res.ci_lb, res.ci_ub, res.coef_names, res.ddf)
        else:
            tab = coef_table(res.alpha, res.se_alpha, res.zval_alpha, res.pval_alpha, res.ci_lb_alpha, res.ci_ub_alpha, res.alpha_names, res.ddf_alpha)
    return metafopy.methods.vif.VIFResult(vif=tuple(entries), bttspec=not missing, table=tab, test=res.test if tab is not None else None)


@metafopy.methods.vif.vif.register
def vif_rma_ls(
    res: metafopy.models.rma_ls_result.RMALSResult,
    btt: metafopy.methods.vif.BttArg = None,
    att: metafopy.methods.vif.BttArg = None,
    table: bool = False,
    reestimate: bool = False,
    sim: bool | int = False,
    digits: int | None = None,
) -> metafopy.methods.vif.VIFResult | LSVIFResult:
    """(G)VIFs for the location and, if the scale model has predictors, the scale coefficients of a location-scale model.

    ``btt`` and ``att`` (0-based) select sets of location and scale coefficients as ``btt`` does in :func:`metafopy.vif`;
    ``table=True`` adds the coefficient table; ``reestimate=True`` refits the model with only the coefficients in each set.
    A single :class:`metafopy.VIFResult` is returned when only one of the two parts applies, otherwise an
    :class:`LSVIFResult` with both.

    Raises:
        metafopy.errors.InputError: an argument is invalid or the model has no predictors.
        metafopy.errors.NotSupportedError: ``sim`` was requested.
    """
    loc = not res.int_only
    scale = not res.Z_int_only
    if not loc and not scale:
        raise metafopy.errors.InputError('VIFs not applicable to intercept-only models.')
    if (isinstance(sim, bool) and sim) or (not isinstance(sim, bool) and sim != 0):
        raise metafopy.errors.NotSupportedError("Argument 'sim' is not supported for location-scale models.")
    out_beta = part_vif(res, which='beta', spec_arg=btt, table=table, reestimate=reestimate) if loc else None
    out_alpha = part_vif(res, which='alpha', spec_arg=att, table=table, reestimate=reestimate) if scale else None
    if out_beta is not None and out_alpha is not None:
        result = LSVIFResult(beta=out_beta, alpha=out_alpha)
    else:
        single = out_beta if out_beta is not None else out_alpha
        if single is None:
            raise metafopy.errors.InternalError('no VIFs were computed')
        result = single
    if digits is not None:
        result = apply_digits(result, digits)
    return result


def apply_digits(result: metafopy.methods.vif.VIFResult | LSVIFResult, digits: int) -> metafopy.methods.vif.VIFResult | LSVIFResult:
    if isinstance(result, LSVIFResult):
        return LSVIFResult(beta=dataclasses.replace(result.beta, digits=digits), alpha=dataclasses.replace(result.alpha, digits=digits))
    return dataclasses.replace(result, digits=digits)
