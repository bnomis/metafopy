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

import metafopy.core.labelled
import metafopy.core.linalg
import metafopy.core.parallel
import metafopy.core.timing
import metafopy.effects.datatools
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.listrma
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.printing

__all__ = ['InfluenceResult', 'ResidualResult', 'cooks_distance', 'dfbetas', 'influence', 'rstandard', 'rstudent']

type FloatArray = numpy.typing.NDArray[numpy.float64]

EPS = float(numpy.finfo(float).eps)


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualResult(metafopy.printing.HtmlRepr):
    """Residuals with standard errors and z-values (metafor's ``list.rma`` of ``rstandard``/``rstudent``).

    Attributes:
        resid: The residuals, one per study.
        se: Their standard errors.
        z: The residuals divided by their standard errors.
        slab: Study labels, in the order of the rows.
        digits: Number of decimals used when the result is printed.
    """

    resid: FloatArray
    se: FloatArray
    z: FloatArray
    slab: tuple[str, ...]
    digits: int = 4

    def __getitem__(self, index: typing.Any) -> typing.Self:
        """The rows ``index`` (metafor's ``[.list.rma``; a NumPy index, 0-based)."""
        return metafopy.methods.listrma.rows(self, index, self.resid.shape[0])

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'resid': self.resid, 'se': self.se, 'z': self.z}, index=list(self.slab))

    def __str__(self) -> str:
        cols = [(n, [metafopy.printing.fmt_num(float(v), self.digits) for v in a]) for n, a in (('resid', self.resid), ('se', self.se), ('z', self.z))]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, self.slab), ''])


@dataclasses.dataclass(frozen=True, slots=True)
class InfluenceResult(metafopy.printing.HtmlRepr):
    """Case-deletion diagnostics: one entry per study.

    The ``inf`` property is the studies' influence table (columns ``rstudent``, ``dffits``, ``cook.d``, ``cov.r``, ``tau2.del``, ``QE.del``,
    ``hat``, ``weight`` and ``inf``), ``dfbs`` the ``DFBETAS`` (one column per coefficient) and ``is_infl`` marks the studies metafor flags with
    an asterisk: |DFFITS| > 3 sqrt(p/(k-p)), the lower-tail area of a chi-square distribution with ``m`` degrees of freedom cut off by Cook's
    distance above 50%, hat > 3p/k or any |DFBETAS| > 1.

    Attributes:
        rstudent: Externally standardised (deleted) residuals.
        dffits: DFFITS values, the standardised change in the fitted value of each study when it is left out.
        cook_d: Cook's distances.
        cov_r: Covariance ratios.
        tau2_del: Estimates of ``tau^2`` from the fits without each study.
        QE_del: Test statistics for residual heterogeneity from the fits without each study.
        hat: Diagonal elements of the hat matrix.
        weight: Weights of the studies in the model fit, in percent.
        dfbs_array: DFBETAS as an array with one column per coefficient.
        is_infl: Boolean flags for the influential studies.
        slab: Study labels, in the order of the rows.
        coef_names: Names of the model coefficients.
        tau2: Estimate of ``tau^2`` of the full model.
        QE: Test statistic for residual heterogeneity of the full model.
        k: Number of studies in the full model.
        p: Number of coefficients in the full model.
        m: Number of coefficients the Cook's distances and covariance ratios refer to (see ``btt``).
        digits: Number of decimals used when the result is printed.
        ids: 0-based positions, among the supplied studies, of the studies in the result, if known.
        not_na: Boolean flags for the studies in the result that were used in the fit (all ``True`` with ``na_action='omit'``), if known.
    """

    rstudent: FloatArray
    dffits: FloatArray
    cook_d: FloatArray
    cov_r: FloatArray
    tau2_del: FloatArray
    QE_del: FloatArray
    hat: FloatArray
    weight: FloatArray
    dfbs_array: FloatArray
    is_infl: numpy.typing.NDArray[numpy.bool_]
    slab: tuple[str, ...]
    coef_names: tuple[str, ...]
    tau2: float
    QE: float
    k: int
    p: int
    m: int
    digits: int = 4
    ids: numpy.typing.NDArray[numpy.intp] | None = None
    not_na: numpy.typing.NDArray[numpy.bool_] | None = None

    @property
    def marks(self) -> tuple[str, ...]:
        """``'*'`` for each influential study, ``''`` otherwise."""
        return tuple('*' if v else '' for v in self.is_infl)

    @property
    def inf(self) -> pandas.DataFrame:
        """The influence table (metafor's ``inf``): ``rstudent``, ``dffits``, ``cook.d``, ``cov.r``, ``tau2.del``, ``QE.del``, ``hat``, ``weight`` and
        ``inf`` (``'*'`` for the influential studies), one row per study; ``attrs`` holds ``slab`` and ``digits``."""
        return self.to_frame()

    @property
    def dfbs(self) -> pandas.DataFrame:
        """The DFBETAS (metafor's ``dfbs``), one column per coefficient; ``attrs`` holds ``slab`` and ``digits``."""
        return self.dfbs_frame()

    def to_frame(self) -> pandas.DataFrame:
        """The influence table, one row per study."""
        out = pandas.DataFrame(
            {
                'rstudent': self.rstudent,
                'dffits': self.dffits,
                'cook.d': self.cook_d,
                'cov.r': self.cov_r,
                'tau2.del': self.tau2_del,
                'QE.del': self.QE_del,
                'hat': self.hat,
                'weight': self.weight,
                'inf': self.marks,
            },
            index=list(self.slab),
        )
        out.attrs.update(slab=list(self.slab), digits=self.digits)
        return out

    def dfbs_frame(self) -> pandas.DataFrame:
        """The DFBETAS, one column per coefficient."""
        # R builds a data frame, which turns the coefficient names into syntactic names (factor(alloc)random becomes factor.alloc.random)
        names = metafopy.effects.datatools.make_names([str(n) for n in self.coef_names], unique=True)
        out = metafopy.core.labelled.RFrame(self.dfbs_array, columns=names, index=list(self.slab))
        out.attrs.update(slab=list(self.slab), digits=self.digits)
        return out

    def summary(self, infonly: bool = False) -> str:
        return format_influence(self, infonly)

    def with_digits(self, digits: typing.Any) -> typing.Self:
        """The result printed with ``digits``; every column of the printout is an ``est`` column, so a mapping acts through its ``est`` element."""
        return dataclasses.replace(self, digits=metafopy.mfopt.digits_map(digits, self.digits)['est'])

    def __str__(self) -> str:
        return format_influence(self, False)


def format_influence(res: InfluenceResult, infonly: bool) -> str:
    sel = res.is_infl if infonly else numpy.ones(res.slab.__len__(), dtype=numpy.bool_)
    if not sel.any():
        return '$inf\n\n$dfbs\n' if res.p > 1 else ''
    slab = [s for s, keep in zip(res.slab, sel, strict=True) if keep]

    def col(name: str, vals: FloatArray) -> tuple[str, list[str]]:
        return name, [metafopy.printing.fmt_num(float(v), res.digits) for v, keep in zip(vals, sel, strict=True) if keep]

    inf = [m for m, keep in zip(res.marks, sel, strict=True) if keep]
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
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame([*base, col('dfbs', res.dfbs_array[:, 0]), ('inf', inf)], slab), ''])
    names = metafopy.effects.datatools.make_names([str(n) for n in res.coef_names], unique=True)
    dfbs = [col(name, res.dfbs_array[:, j]) for j, name in enumerate(names)]
    first = metafopy.methods.blup.format_r_frame([*base, ('inf', inf)], slab)
    second = metafopy.methods.blup.format_r_frame(dfbs, slab)
    return '\n'.join(['$inf', '', *first, '', '$dfbs', '', *second, ''])


@functools.singledispatch
def influence(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Case-deletion (leave-one-out) diagnostics for a fitted model.

    This is a single-dispatch function: the implementation is chosen from the class of ``model``. For a model fitted by
    [rma_uni][metafopy.rma_uni] see [influence_rma_uni][metafopy.methods.influence.influence_rma_uni]; for one fitted by
    [rma_mv][metafopy.rma_mv] see [influence_rma_mv][metafopy.methods.mv_diagnostics.influence_rma_mv], which is not available (as in metafor).

    Args:
        model: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``model``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``model``.

    Returns:
        What the implementation returns; for `rma_uni` fits an [InfluenceResult][metafopy.InfluenceResult].

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [influence.rma.uni](https://wviechtb.github.io/metafor/reference/influence.rma.uni.html)
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

    The model is refitted without each study in turn, which gives the externally standardised residuals, DFFITS, Cook's distances, covariance
    ratios, DFBETAS, leave-one-out ``tau^2`` and ``QE``, hat values and weights (Viechtbauer & Cheung, 2010,
    [doi:10.1002/jrsm.11](https://doi.org/10.1002/jrsm.11)).
    Studies whose refit fails (or loses a coefficient) get NaN.

    Args:
        model: A model fitted by [rma_uni][metafopy.rma_uni]; not allowed for a model passed through [robust][metafopy.robust] (except with
            ``measure='cooks.distance'``).
        digits: Number of decimals used when the result is printed. ``None`` uses the ``digits`` option, else the ``digits`` of the model.
        btt: The coefficients Cook's distance and the covariance ratio refer to, as 0-based indices (or a regular expression matched against the
            coefficient names). ``None`` (the default) uses all coefficients.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of missing values for each, so the result has one row per supplied study) or ``'fail'`` (raise an error).
        measure: Metafor's undocumented argument that restricts the result: ``'cooks.distance'`` gives only the Cook's distances (as
            [cooks_distance][metafopy.cooks_distance]), ``'dfbetas'`` only the DFBETAS (as [dfbetas][metafopy.dfbetas]) and ``'rstudent'`` only the
            deleted residuals (as [rstudent][metafopy.rstudent]). Any other value (the default is ``'all'``) gives the full result.
        progbar: Print a progress bar to stderr while the studies are left out in turn (default ``False``); no bar is shown with
            ``measure='cooks.distance'`` or ``'rstudent'``.

    Returns:
        By default an [InfluenceResult][metafopy.InfluenceResult] with one entry per study; its ``inf`` property is the influence table
        (``rstudent``, ``dffits``, ``cook.d``, ``cov.r``, ``tau2.del``, ``QE.del``, ``hat``, ``weight`` and ``inf``, which marks influential studies
        with ``'*'``) and ``dfbs`` the DFBETAS. With ``measure`` set: a Series of Cook's distances, a DataFrame of DFBETAS, or a
        [ResidualResult][metafopy.ResidualResult].

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model was passed through [robust][metafopy.robust].

    Metafor reference:
        [influence.rma.uni](https://wviechtb.github.io/metafor/reference/influence.rma.uni.html)
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
    # metafor trims ids and not.na to the studies used under na.omit (influence.rma.uni.r)
    ids, not_na = (x.ids[x.not_na], x.not_na[x.not_na]) if na_action == 'omit' else (x.ids, x.not_na)
    return InfluenceResult(
        rstudent=rs,
        dffits=df,
        cook_d=ck,
        cov_r=cr,
        tau2_del=td,
        QE_del=qd,
        hat=ht,
        weight=wt,
        dfbs_array=dfbs,
        is_infl=is_infl,
        slab=slab,
        coef_names=x.coef_names,
        tau2=x.tau2,
        QE=x.QE,
        k=x.k,
        p=x.p,
        m=core.m,
        digits=metafopy.mfopt.option_digits(digits, x.digits),
        ids=ids,
        not_na=not_na,
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
    """Externally standardised (studentised deleted) residuals of a fitted model (metafor's ``rstudent``).

    This is a single-dispatch function: the implementation is chosen by the class of ``model``. See
    [rstudent_rma_uni][metafopy.methods.influence.rstudent_rma_uni], [rstudent_rma_mv][metafopy.methods.mv_diagnostics.rstudent_rma_mv] and
    [rstudent_fixed_binary][metafopy.methods.binary.rstudent_fixed_binary].

    Args:
        model: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation (``digits`` first).
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        The residuals, a [ResidualResult][metafopy.ResidualResult] (a cluster result for ``rma_mv`` with ``cluster``).

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``model`` (there is none for location-scale models).
        TypeError: the implementation does not accept a keyword argument.
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
    """Externally standardised residuals of an [rma_uni][metafopy.rma_uni] fit (metafor's ``rstudent.rma.uni``).

    The deleted residual ``y_i - x_i' beta_(-i)`` compares each study with the prediction from the model fitted without it; its standard error adds the
    sampling variance, ``tau^2`` and the variance of the prediction.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/rstudent.rma.uni.r>

    Args:
        model: Model fitted by [rma_uni][metafopy.rma_uni].
        digits: Number of decimals of the printout. Default: the ``digits`` option, else that of the model.
        na_action: What to do with studies left out of the fit because of missing values: ``'omit'`` (default) drops them, ``'exclude'`` and ``'pass'``
            keep one row per study with NaN for those left out, ``'fail'`` raises an InputError if there are any. Corresponds to R's
            ``na.action`` option.
        progbar: Print a progress bar to stderr while the studies are left out in turn (default False, as in metafor).

    Returns:
        A [ResidualResult][metafopy.ResidualResult] with the deleted residuals ``resid``, their standard errors ``se`` and the standardised values ``z``,
        labelled with the study labels.

    Raises:
        metafopy.errors.InputError: ``na_action`` is not valid, or is ``'fail'`` and studies were left out of the fit; the model has a single study.
        metafopy.errors.NotSupportedError: the model is a [robust][metafopy.robust] result.

    Metafor reference:
        [residuals.rma](https://wviechtb.github.io/metafor/reference/residuals.rma.html)
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
    """Cook's distances of the studies (or clusters) of a fitted model.

    This is a single-dispatch function: the implementation is chosen from the class of ``model`` (see
    [cooks_rma_mv][metafopy.methods.mv_diagnostics.cooks_rma_mv] for [rma_mv][metafopy.rma_mv] fits). The `rma_uni` implementation computes
    them as part of [influence][metafopy.influence].

    Args:
        model: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``model``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``model``. For an `rma_uni` fit these are ``btt``
            (0-based coefficient indices the distances refer to; default all), ``na_action`` (``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'``)
            and ``progbar`` (print a progress bar).

    Returns:
        A Series of Cook's distances indexed by study label (named ``cook.d``).

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [influence.rma.uni](https://wviechtb.github.io/metafor/reference/influence.rma.uni.html)
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
    """Cook's distance of each study (metafor's ``cooks.distance``), indexed by study label.

    Args:
        model: A model fitted by [rma_uni][metafopy.rma_uni].
        btt: The coefficients the distances refer to, as 0-based indices (or a regular expression matched against the coefficient names).
            ``None`` (the default) uses all coefficients.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no entry for them), ``'exclude'`` or ``'pass'``
            (an entry of NaN for each) or ``'fail'`` (raise an error).
        progbar: Print a progress bar to stderr while the studies are left out in turn (default ``False``).

    Returns:
        The Cook's distances, a Series named ``cook.d`` indexed by study label.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    res = influence_fit(model, btt=btt, na_action=na_action, progbar=progbar)
    return pandas.Series(res.cook_d, index=list(res.slab), name='cook.d')


@functools.singledispatch
def dfbetas(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> pandas.DataFrame:
    """DFBETAS of the studies (or clusters) of a fitted model.

    This is a single-dispatch function: the implementation is chosen from the class of ``model`` (see
    [dfbetas_rma_mv][metafopy.methods.mv_diagnostics.dfbetas_rma_mv] for [rma_mv][metafopy.rma_mv] fits). The `rma_uni` implementation computes
    them as part of [influence][metafopy.influence].

    Args:
        model: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``model``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``model``. For an `rma_uni` fit these are
            ``na_action`` (``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'``) and ``progbar`` (print a progress bar).

    Returns:
        A data frame of DFBETAS with one row per study (or cluster) and one column per coefficient.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [influence.rma.uni](https://wviechtb.github.io/metafor/reference/influence.rma.uni.html)
    """
    raise metafopy.errors.ArgumentError(f'dfbetas() not implemented for {type(model).__name__}')


@dfbetas.register
def dfbetas_rma_uni(
    model: metafopy.models.rma_uni.RMAResult, *, na_action: metafopy.methods.common.NAAction = 'omit', progbar: bool = False
) -> pandas.DataFrame:
    """Standardised change in each coefficient when a study is left out (metafor's ``dfbetas``).

    Args:
        model: A model fitted by [rma_uni][metafopy.rma_uni]; not allowed for a model passed through [robust][metafopy.robust].
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of missing values for each) or ``'fail'`` (raise an error).
        progbar: Print a progress bar to stderr while the studies are left out in turn (default ``False``).

    Returns:
        A data frame with one row per study (indexed by study label) and one column per model coefficient, holding the DFBETAS.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.NotSupportedError: the model was passed through [robust][metafopy.robust].
    """
    metafopy.methods.fitview.check_class(model, notav=metafopy.methods.fitview.NOT_ROBUST)
    return influence_fit(model, na_action=na_action, progbar=progbar).dfbs_frame()


@functools.singledispatch
def rstandard(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> ResidualResult:
    """Internally standardised residuals of a fitted model (metafor's ``rstandard``).

    This is a single-dispatch function: the implementation is chosen by the class of ``model``. See
    [rstandard_rma_uni][metafopy.methods.influence.rstandard_rma_uni], [rstandard_rma_mv][metafopy.methods.mv_diagnostics.rstandard_rma_mv],
    [rstandard_rma_ls][metafopy.methods.ls_resid.rstandard_rma_ls] and [rstandard_fixed_binary][metafopy.methods.binary.rstandard_fixed_binary].

    Args:
        model: The fitted model. Its class selects the implementation.
        *args: Positional arguments passed on to the implementation (``digits`` first).
        **kwargs: Keyword arguments passed on to the implementation.

    Returns:
        The residuals, a [ResidualResult][metafopy.ResidualResult] (a cluster result for ``rma_mv`` with ``cluster``).

    Raises:
        metafopy.errors.ArgumentError: there is no implementation for the class of ``model``.
        TypeError: the implementation does not accept a keyword argument.
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
    """Internally standardised residuals of an [rma_uni][metafopy.rma_uni] fit (metafor's ``rstandard.rma.uni``).

    With ``type='marginal'`` the residuals ``y - X b`` are divided by the square roots of the diagonal of ``(I - H) M (I - H)'``, where ``M`` holds the
    marginal variances; with ``type='conditional'`` the residuals are those from the BLUPs, divided by their conditional standard errors. Conditional
    residuals are not available for models with non-standard weights.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/rstandard.rma.uni.r>

    Args:
        model: Model fitted by [rma_uni][metafopy.rma_uni] (not a robust fit).
        digits: Number of decimals of the printout. Default: the ``digits`` option, else that of the model.
        type: ``'marginal'`` (default) for marginal residuals or ``'conditional'`` for conditional residuals (the deviations of the estimates from the
            BLUPs of the true effects). Exactly one of these two values (R also accepts a unique prefix).
        na_action: What to do with studies left out of the fit because of missing values: ``'omit'`` (default) drops them, ``'exclude'`` and ``'pass'``
            keep one row per study with NaN for those left out, ``'fail'`` raises an InputError if there are any. Corresponds to R's
            ``na.action`` option.

    Returns:
        A [ResidualResult][metafopy.ResidualResult] with the residuals ``resid``, their standard errors ``se`` and the standardised values ``z``, labelled
        with the study labels.

    Raises:
        metafopy.errors.InputError: ``type`` is not ``'marginal'`` or ``'conditional'``; ``type='conditional'`` for a model with weights given or
            fitted without weighting; ``na_action`` is not valid, or is ``'fail'`` and studies were left out of the fit.
        metafopy.errors.NotSupportedError: the model is a [robust][metafopy.robust] result.

    Metafor reference:
        [residuals.rma](https://wviechtb.github.io/metafor/reference/residuals.rma.html)
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
