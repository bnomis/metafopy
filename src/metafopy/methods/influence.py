"""Outlier and influential-case diagnostics (metafor's ``influence.rma.uni`` and relatives).

For each study ``i`` the model is refitted without it; the externally
standardised residual (``rstudent``), ``DFFITS``, Cook's distance, the
covariance ratio, ``DFBETAS``, the leave-one-out ``tau^2`` and ``QE`` follow
Viechtbauer & Cheung (2010), doi:10.1002/jrsm.11. Also provided are the
internally standardised residuals (:func:`rstandard`), and the shortcuts
:func:`rstudent`, :func:`cooks_distance` and :func:`dfbetas`.
See https://wviechtb.github.io/metafor/reference/influence.rma.uni.html.
"""

import dataclasses
import functools
import typing

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.linalg
import metafopy.core.parallel
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['InfluenceResult', 'ResidualResult', 'cooks_distance', 'dfbetas', 'influence', 'rstandard', 'rstudent']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualResult:
    """Residuals with standard errors and z-values (metafor's ``list.rma`` of ``rstandard``/``rstudent``)."""

    resid: FloatArray
    se: FloatArray
    z: FloatArray
    slab: tuple[str, ...]
    digits: int = 4

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'resid': self.resid, 'se': self.se, 'z': self.z}, index=list(self.slab))

    def __str__(self) -> str:
        cols = [(n, [metafopy.printing.fmt_num(float(v), self.digits) for v in a]) for n, a in (('resid', self.resid), ('se', self.se), ('z', self.z))]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, self.slab), ''])


@dataclasses.dataclass(frozen=True, slots=True)
class InfluenceResult:
    """Case-deletion diagnostics: one entry per study.

    ``inf`` holds the studies' influence table (``rstudent``, ``dffits``,
    ``cook_d``, ``cov_r``, ``tau2_del``, ``QE_del``, ``hat``, ``weight``),
    ``dfbs`` the ``DFBETAS`` (one column per coefficient) and ``is_infl`` marks
    the studies metafor flags with an asterisk (|DFFITS| > 3 sqrt(p/(k-p)),
    Cook's D beyond the median of its chi-square reference distribution,
    hat > 3p/k or any |DFBETAS| > 1).
    """

    rstudent: FloatArray
    dffits: FloatArray
    cook_d: FloatArray
    cov_r: FloatArray
    tau2_del: FloatArray
    QE_del: FloatArray
    hat: FloatArray
    weight: FloatArray
    dfbs: FloatArray
    is_infl: numpy.typing.NDArray[numpy.bool_]
    slab: tuple[str, ...]
    coef_names: tuple[str, ...]
    tau2: float
    QE: float
    k: int
    p: int
    m: int
    digits: int = 4

    @property
    def inf(self) -> tuple[str, ...]:
        """``'*'`` for each influential study, ``''`` otherwise."""
        return tuple('*' if v else '' for v in self.is_infl)

    def to_frame(self) -> pandas.DataFrame:
        """The influence table, one row per study."""
        return pandas.DataFrame(
            {
                'rstudent': self.rstudent,
                'dffits': self.dffits,
                'cook.d': self.cook_d,
                'cov.r': self.cov_r,
                'tau2.del': self.tau2_del,
                'QE.del': self.QE_del,
                'hat': self.hat,
                'weight': self.weight,
                'inf': self.inf,
            },
            index=list(self.slab),
        )

    def dfbs_frame(self) -> pandas.DataFrame:
        """The DFBETAS, one column per coefficient."""
        return pandas.DataFrame(self.dfbs, columns=list(self.coef_names), index=list(self.slab))

    def summary(self, infonly: bool = False) -> str:
        return format_influence(self, infonly)

    def __str__(self) -> str:
        return format_influence(self, False)


def format_influence(res: InfluenceResult, infonly: bool) -> str:
    sel = res.is_infl if infonly else numpy.ones(res.slab.__len__(), dtype=numpy.bool_)
    if not sel.any():
        return '$inf\n\n$dfbs\n' if res.p > 1 else ''
    slab = [s for s, keep in zip(res.slab, sel, strict=True) if keep]

    def col(name: str, vals: FloatArray) -> tuple[str, list[str]]:
        return name, [metafopy.printing.fmt_num(float(v), res.digits) for v, keep in zip(vals, sel, strict=True) if keep]

    inf = [m for m, keep in zip(res.inf, sel, strict=True) if keep]
    base = [
        col('rstudent', res.rstudent),
        col('dffits', res.dffits),
        col('cook.d', res.cook_d),
        col('cov.r', res.cov_r),
        col('tau2.del', res.tau2_del),
        col('QE.del', res.QE_del),
        col('hat', res.hat),
        col('weight', res.weight),
    ]
    if res.p == 1:
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame([*base, col('dfbs', res.dfbs[:, 0]), ('inf', inf)], slab), ''])
    dfbs = [col(name, res.dfbs[:, j]) for j, name in enumerate(res.coef_names)]
    first = metafopy.methods.blup.format_r_frame([*base, ('inf', inf)], slab)
    second = metafopy.methods.blup.format_r_frame(dfbs, slab)
    return '\n'.join(['$inf', '', *first, '', '$dfbs', '', *second, ''])


@functools.singledispatch
def influence(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Case-deletion diagnostics for a fitted model; see :func:`_influence_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'influence() not implemented for {type(model).__name__}')


@influence.register
@metafopy.core.timing.timed
def influence_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    digits: int | None = None,
    *,
    btt: metafopy.models.rma_uni.Btt = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    measure: str = 'all',
    progbar: bool = False,
) -> InfluenceResult | pandas.Series | pandas.DataFrame | ResidualResult:
    """Leave-one-out diagnostics for an ``rma.uni`` fit (metafor's ``influence()``).

    ``btt`` (0-based coefficient indices, default all coefficients) selects the
    coefficients Cook's distance and the covariance ratio refer to. With
    ``na_action='exclude'`` or ``'pass'`` the result has one row per supplied
    study. Studies whose refit fails (or loses a coefficient) get NaN. ``digits`` defaults to the ``digits`` option, else to that of the model;
    ``progbar=True`` prints a progress bar to stderr while the studies are left out in turn.

    ``measure`` is metafor's undocumented argument that restricts the result: ``'cooks.distance'`` gives the Cook's distances (as
    :func:`cooks_distance`, also allowed for a model made robust), ``'dfbetas'`` the DFBETAS (as :func:`dfbetas`) and ``'rstudent'`` the
    deleted residuals with their standard errors and z-values (as :func:`rstudent`); any other value gives the full result, as in R.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if measure == 'cooks.distance':
        res = influence_fit(model, digits, btt=btt, na_action=na_action)
        return pandas.Series(res.cook_d, index=list(res.slab), name='cook.d')
    metafopy.methods.fitview.check_class(model, notav=metafopy.methods.fitview.NOT_ROBUST)
    if measure == 'rstudent':
        return rstudent_rma_uni(model, digits, na_action=na_action)
    res = influence_fit(model, digits, btt=btt, na_action=na_action, progbar=progbar)
    if measure == 'dfbetas':
        return res.dfbs_frame()
    return res


def influence_fit(
    model: metafopy.models.rma_uni.RMAResult,
    digits: int | None = None,
    *,
    btt: metafopy.models.rma_uni.Btt = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    progbar: bool = False,
) -> InfluenceResult:
    """The leave-one-out diagnostics of :func:`influence_rma_uni` without the check for fits made robust (which ``cooks.distance`` allows).

    Raises:
        metafopy.errors.InputError: ``na_action`` is unknown, there is a single study or values are missing with ``na_action='fail'``.
    """
    metafopy.methods.common.check_na_action(na_action)
    x = model
    if x.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    core = influence_core(x, btt, progbar)
    inf = core.inf
    if na_action == 'fail' and not x.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    slab = x.slab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action in ('exclude', 'pass'):
        (rs, df, ck, cr, td, qd, ht, wt, dfbs), slab = metafopy.methods.common.pad_to_supplied(
            x, inf['rstudent'], inf['dffits'], inf['cook.d'], inf['cov.r'], inf['tau2.del'], inf['QE.del'], inf['hat'], inf['weight'], core.dfbs
        )
        is_infl = flag(x, df, ck, ht, dfbs, core.m)
    else:
        rs, df, ck, cr, td, qd, ht, wt, dfbs = (
            inf['rstudent'],
            inf['dffits'],
            inf['cook.d'],
            inf['cov.r'],
            inf['tau2.del'],
            inf['QE.del'],
            inf['hat'],
            inf['weight'],
            core.dfbs,
        )
        is_infl = flag(x, df, ck, ht, dfbs, core.m)
    return InfluenceResult(
        rstudent=rs,
        dffits=df,
        cook_d=ck,
        cov_r=cr,
        tau2_del=td,
        QE_del=qd,
        hat=ht,
        weight=wt,
        dfbs=dfbs,
        is_infl=is_infl,
        slab=slab,
        coef_names=x.coef_names,
        tau2=x.tau2,
        QE=x.QE,
        k=x.k,
        p=x.p,
        m=core.m,
        digits=metafopy.mfopt.option_digits(digits, x.digits),
    )


def flag(
    x: metafopy.models.rma_uni.RMAResult, dffits: FloatArray, cook_d: FloatArray, hat: FloatArray, dfbs: FloatArray, m: int
) -> numpy.typing.NDArray[numpy.bool_]:
    with numpy.errstate(invalid='ignore'):
        return numpy.asarray(
            (numpy.abs(dffits) > 3 * numpy.sqrt(x.p / (x.k - x.p)))
            | (scipy.stats.chi2.cdf(cook_d, df=m) > 0.5)
            | (hat > 3 * x.p / x.k)
            | numpy.any(numpy.abs(dfbs) > 1, axis=1),
            dtype=numpy.bool_,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Core:
    inf: dict[str, FloatArray]
    dfbs: FloatArray
    m: int
    resid: FloatArray
    seresid: FloatArray


def influence_core(x: metafopy.models.rma_uni.RMAResult, btt: metafopy.models.rma_uni.Btt, progbar: bool = False) -> Core:
    k, p = x.k, x.p
    sub = numpy.array(metafopy.models.rma_uni.set_btt(btt, p, False, x.coef_names), dtype=numpy.intp)
    m = len(sub)
    X = x.X
    pred_full = X @ x.beta
    svb = numpy.linalg.inv(x.vb[numpy.ix_(sub, sub)])
    H: FloatArray | None = None
    A: FloatArray | None = None
    stXAX: FloatArray | None = None
    stXX: FloatArray | None = None
    if x.weighted:
        if x.weights is not None:
            A = numpy.diag(x.weights)
            stXAX = metafopy.core.linalg.invcalc(X, x.weights)
            H = X @ stXAX @ X.T @ A
    else:
        stXX = metafopy.core.linalg.invcalc(X, numpy.ones(k))
        H = X @ stXX @ X.T
    hat = x.hatvalues()

    tau2_del = numpy.full(k, numpy.nan)
    delpred = numpy.full(k, numpy.nan)
    vdelpred = numpy.full(k, numpy.nan)
    s2w_del = numpy.full(k, numpy.nan)
    QE_del = numpy.full(k, numpy.nan)
    dffits = numpy.full(k, numpy.nan)
    dfbs = numpy.full((k, p), numpy.nan)
    cook_d = numpy.full(k, numpy.nan)
    cov_r = numpy.full(k, numpy.nan)

    all_idx = numpy.arange(k)
    for i in metafopy.core.parallel.progress(range(k), k, progbar):
        res = metafopy.methods.common.refit(x, all_idx != i)
        if res is None:
            continue
        tau2_del[i] = res.tau2
        QE_del[i] = res.QE
        Xi = X[i]
        delpred[i] = Xi @ res.beta
        vdelpred[i] = Xi @ res.vb @ Xi
        s2w_del[i] = res.s2w
        if x.weighted and x.weights is None:
            dffits[i] = (pred_full[i] - delpred[i]) / numpy.sqrt(res.s2w * hat[i] * (tau2_del[i] + x.vi[i]))
        else:
            assert H is not None
            d = numpy.diag(H @ numpy.diag(tau2_del[i] + x.vi) @ H.T)
            dffits[i] = (pred_full[i] - delpred[i]) / numpy.sqrt(res.s2w * d)[i]
        if x.weighted:
            if x.weights is None:
                vb_del = metafopy.core.linalg.invcalc(X, 1.0 / (x.vi + tau2_del[i]))
            else:
                assert stXAX is not None
                assert A is not None
                vb_del = stXAX @ X.T @ A @ numpy.diag(x.vi + tau2_del[i]) @ A @ X @ stXAX
        else:
            assert stXX is not None
            vb_del = stXX @ X.T @ numpy.diag(x.vi + tau2_del[i]) @ X @ stXX
        dfb = x.beta - res.beta
        dfbs[i] = dfb / numpy.sqrt(res.s2w * numpy.diag(vb_del))
        dfb_s = dfb[sub]
        cook_d[i] = float(dfb_s @ svb @ dfb_s)
        cov_r[i] = numpy.linalg.det(res.vb[numpy.ix_(sub, sub)]) / numpy.linalg.det(x.vb[numpy.ix_(sub, sub)])

    resid = metafopy.methods.common.tzero(x.yi - delpred)
    seresid = numpy.sqrt(x.vi * s2w_del + vdelpred + tau2_del * s2w_del)
    stresid = resid / seresid
    inf = {
        'rstudent': stresid,
        'dffits': dffits,
        'cook.d': cook_d,
        'cov.r': cov_r,
        'tau2.del': tau2_del,
        'QE.del': QE_del,
        'hat': hat,
        'weight': x.weights_(),
    }
    return Core(inf=inf, dfbs=dfbs, m=m, resid=resid, seresid=seresid)


@functools.singledispatch
def rstudent(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> ResidualResult:
    """Externally standardised (studentised deleted) residuals; see :func:`_rstudent_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'rstudent() not implemented for {type(model).__name__}')


@rstudent.register
def rstudent_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    digits: int | None = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
    progbar: bool = False,
) -> ResidualResult:
    """Deleted residuals ``y_i - x_i' beta_(-i)`` with their standard errors and z-values.

    ``progbar=True`` prints a progress bar to stderr while the studies are left out in turn.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    metafopy.methods.fitview.check_class(model, notav=metafopy.methods.fitview.NOT_ROBUST)
    if model.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    core = influence_core(model, None, progbar)
    if na_action == 'fail' and not model.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    z = core.inf['rstudent']
    slab = model.slab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action in ('exclude', 'pass'):
        (resid, se, z), slab = metafopy.methods.common.pad_to_supplied(model, core.resid, core.seresid, z)
    else:
        resid, se = core.resid, core.seresid
    return ResidualResult(resid=resid, se=se, z=z, slab=slab, digits=metafopy.mfopt.option_digits(digits, model.digits))


@functools.singledispatch
def cooks_distance(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> pandas.Series:
    """Cook's distances; see :func:`_cooks_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'cooks_distance() not implemented for {type(model).__name__}')


@cooks_distance.register
def cooks_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    *,
    btt: metafopy.models.rma_uni.Btt = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    progbar: bool = False,
) -> pandas.Series:
    """Cook's distance of each study (metafor's ``cooks.distance``), indexed by study label; ``progbar=True`` prints a progress bar to stderr."""
    res = influence_fit(model, btt=btt, na_action=na_action, progbar=progbar)
    return pandas.Series(res.cook_d, index=list(res.slab), name='cook.d')


@functools.singledispatch
def dfbetas(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> pandas.DataFrame:
    """DFBETAS; see :func:`_dfbetas_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'dfbetas() not implemented for {type(model).__name__}')


@dfbetas.register
def dfbetas_rma_uni(
    model: metafopy.models.rma_uni.RMAResult, *, na_action: metafopy.methods.common.NAAction = 'omit', progbar: bool = False
) -> pandas.DataFrame:
    """Standardised change in each coefficient when a study is left out (one column per coefficient); ``progbar=True`` prints a progress bar to stderr."""
    metafopy.methods.fitview.check_class(model, notav=metafopy.methods.fitview.NOT_ROBUST)
    return influence_fit(model, na_action=na_action, progbar=progbar).dfbs_frame()


@functools.singledispatch
def rstandard(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> ResidualResult:
    """Internally standardised residuals; see :func:`_rstandard_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'rstandard() not implemented for {type(model).__name__}')


@rstandard.register
def rstandard_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    digits: int | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    type: typing.Literal['marginal', 'conditional'] = 'marginal',
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> ResidualResult:
    """Standardised residuals; ``type='marginal'`` uses the marginal residuals, ``'conditional'`` the conditional ones.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    x = model
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.NOT_ROBUST)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if type not in ('marginal', 'conditional'):
        raise metafopy.errors.InputError("type must be 'marginal' or 'conditional'")
    if type == 'conditional' and (x.weights is not None or not x.weighted):
        raise metafopy.errors.InputError('Extraction of conditional residuals not available for models with non-standard weights.')
    k = x.k
    H = x.hat_matrix()
    ImH = numpy.eye(k) - H
    if type == 'marginal':
        ei = metafopy.methods.common.tzero(x.yi - x.X @ x.beta)
        M = numpy.diag(x.vi + x.tau2)
        sei = numpy.sqrt(numpy.diag(ImH @ M @ ImH.T))
    else:
        li = x.tau2 / (x.tau2 + x.vi)
        pred = li * x.yi + (1 - li) * (x.X @ x.beta)
        ei = x.yi - pred
        sei = numpy.sqrt(x.vi**2 / (x.vi + x.tau2) * (1 - numpy.diag(H)))
    if na_action == 'fail' and not x.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    z = ei / sei
    slab = x.slab
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action in ('exclude', 'pass'):
        (ei, sei, z), slab = metafopy.methods.common.pad_to_supplied(x, ei, sei, z)
    return ResidualResult(resid=ei, se=sei, z=z, slab=slab, digits=metafopy.mfopt.option_digits(digits, x.digits))
