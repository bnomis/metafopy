"""Back-calculate inputs for effect sizes: metafor's ``conv.2x2``, ``conv.delta`` and ``conv.wald``.

R sources: https://github.com/wviechtb/metafor/blob/master/R/conv.2x2.r (with
``R/misc.func.hidden.conv.2x2.r``), ``R/conv.delta.r`` and ``R/conv.wald.r``. Documentation:
https://wviechtb.github.io/metafor/reference/conv.2x2.html,
https://wviechtb.github.io/metafor/reference/conv.delta.html and
https://wviechtb.github.io/metafor/reference/conv.wald.html.

- ``conv_2x2`` reconstructs 2x2 tables from an odds ratio, a phi coefficient or a chi-square
  statistic with the marginal counts (Bonett 2007, doi:10.1016/j.csda.2006.07.016; Bonett and
  Price 2005, doi:10.3102/10769986030002213), or from three or four of sensitivity, specificity,
  positive and negative predictive value with the sample size (Kiss et al. 2023,
  doi:10.1002/jrsm.1694);
- ``conv_delta`` transforms estimates and their sampling variances with the delta method, using
  numerical derivatives as ``numDeriv::grad()`` computes them;
- ``conv_wald`` turns Wald-type confidence intervals, test statistics or p-values into sampling
  variances.

Differences from R that follow Python conventions: arguments are array-likes, scalars or column names
in ``data``; ``include`` positions are 0-based; ``conv_2x2``'s ``method`` (an R ``...`` option) and
``conv_wald``'s ``cifac`` are keyword arguments; extra arguments of ``conv_delta``'s ``transf`` are
passed as keyword arguments.
"""

import collections.abc
import inspect
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.core.bfgs
import metafopy.core.numderiv
import metafopy.effects.datatools
import metafopy.effects.measures
import metafopy.effects.summary
import metafopy.errors

__all__ = ['conv_2x2', 'conv_delta', 'conv_wald']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Arg = numpy.typing.ArrayLike | str | None
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type Replace = typing.Literal['ifna', 'all', 'complete'] | bool

EPS = float(numpy.finfo(float).eps)


def match_replace(replace: Replace) -> str:
    """Raises:
    metafopy.errors.InputError: ``replace`` is not one of ``'ifna'``, ``'all'`` and ``'complete'`` (or a prefix).
    """
    if isinstance(replace, bool | numpy.bool_):
        return 'all' if replace else 'ifna'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    hits = [opt for opt in ('ifna', 'all', 'complete') if opt.startswith(str(replace))] if replace else []
    if len(hits) != 1:
        raise metafopy.errors.InputError('\'arg\' should be one of "ifna", "all", "complete"')
    return hits[0]


def as_frame(data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None) -> pandas.DataFrame | None:
    if data is None:
        return None
    out = data.copy() if isinstance(data, pandas.DataFrame) else pandas.DataFrame(data)
    return out.reset_index(drop=True)


def fill(values: FloatArray | None, k: int) -> FloatArray:
    return numpy.full(k, numpy.nan) if values is None else values.astype(numpy.float64, copy=True)


def quiet(x: FloatArray) -> FloatArray:
    """R's NA as an ordinary NaN."""
    out = x.copy()
    out[numpy.isnan(out)] = math.nan
    return out


def replace_column(data: pandas.DataFrame, name: str, values: FloatArray, replace: str) -> None:
    """Put ``values`` into column ``name`` of ``data`` (``replace``: fill the missing values, overwrite where available, or overwrite all)."""
    values = quiet(values)
    if name not in data.columns:
        data[name] = values
        return
    if replace == 'ifna':
        current = data[name].to_numpy(dtype=numpy.float64, copy=True)
        data[name] = numpy.where(numpy.isnan(current), values, current)
    elif replace == 'all':
        current = data[name].to_numpy(dtype=numpy.float64, copy=True)
        data[name] = numpy.where(numpy.isnan(values), current, values)
    else:
        data[name] = values


def yi_vi_names(x: pandas.DataFrame | None, var_names: collections.abc.Sequence[str] | None) -> tuple[str, str]:
    """The names of the ``yi`` and ``vi`` variables: ``var_names``, else escalc's attributes, else ``yi``/``vi``.

    Raises:
        metafopy.errors.InputError: the names cannot be determined.
    """
    if var_names is not None:
        if len(var_names) != 2:
            raise metafopy.errors.InputError("Argument 'var_names' must be of length 2.")
        return var_names[0], var_names[1]
    if x is None or not ('yi_names' in x.attrs or 'vi_names' in x.attrs):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        return 'yi', 'vi'
    yi_names = metafopy.effects.summary.names(x.attrs.get('yi_names'))
    vi_names = metafopy.effects.summary.names(x.attrs.get('vi_names'))
    if yi_names:
        yi_name = yi_names[0]
    elif 'yi' in x.columns:
        yi_name = 'yi'
    else:
        raise metafopy.errors.InputError("Cannot determine the name of the 'yi' variable.")
    if vi_names:
        vi_name = vi_names[0]
    elif 'vi' in x.columns:
        vi_name = 'vi'
    else:
        raise metafopy.errors.InputError("Cannot determine the name of the 'vi' variable.")
    return yi_name, vi_name


def include_mask(include: numpy.typing.ArrayLike | str | None, k: int, data: pandas.DataFrame | None) -> BoolArray:
    mask = metafopy.effects.measures.chksubset(include, k, data, 'include', stoponk0=False)
    return numpy.ones(k, dtype=numpy.bool_) if mask is None else mask


def attr_ni(x: pandas.DataFrame | None, yi_name: str) -> FloatArray | None:
    """The ``ni`` attribute of variable ``yi_name`` (metafopy keeps the one of the first ``yi_names`` variable)."""
    if x is None:
        return None
    names = metafopy.effects.summary.names(x.attrs.get('yi_names'))
    ni = x.attrs.get('ni')
    if ni is None or not names or names[0] != yi_name:
        return None
    arr = numpy.asarray(ni, dtype=numpy.float64).reshape(-1)
    return arr if arr.shape[0] == x.shape[0] else None


def attr_measure(x: pandas.DataFrame | None, yi_name: str) -> str:
    if x is None:
        return 'GEN'
    names = metafopy.effects.summary.names(x.attrs.get('yi_names'))
    measure = x.attrs.get('measure')
    return str(measure) if measure and names and names[0] == yi_name else 'GEN'


def finish(
    x: pandas.DataFrame,
    yi_name: str,
    vi_name: str,
    ni: FloatArray | None,
    measure: str,
    append: bool,
    prev_attrs: dict[collections.abc.Hashable | None, typing.Any],
) -> pandas.DataFrame:
    """``escalc(measure, data=x, yi=x[[yi.name]], vi=x[[vi.name]], var.names=...)`` and the optional column selection."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = x.copy()
        base.attrs = dict(prev_attrs)
        out = metafopy.effects.measures.escalc(
            measure,
            yi=x[yi_name].to_numpy(dtype=numpy.float64),
            vi=x[vi_name].to_numpy(dtype=numpy.float64),
            ni=ni if ni is not None and not numpy.isnan(ni).all() else None,
            data=base,
            var_names=(yi_name, vi_name),
        )
    if not append:
        attrs = dict(out.attrs)
        out = out[[yi_name, vi_name]].copy()
        out.attrs = attrs
    return out


# --------------------------------------------------------------------------
# conv.2x2


def rec2x2diag(sens: float, spec: float, ppv: float, ni: float) -> FloatArray:
    """metafor's ``.rec2x2diag``: the (unrounded) cells from sensitivity, specificity, PPV and the sample size."""
    den = sens * (1 - ppv) + ppv * (1 - spec)
    return numpy.array(
        [
            ni * sens * ppv * (1 - spec) / den,
            ni * sens * (1 - ppv) * (1 - spec) / den,
            ni * (1 - sens) * ppv * (1 - spec) / den,
            ni * sens * spec * (1 - ppv) / den,
        ]
    )


def funconv2x2diag(par: FloatArray, obs: FloatArray, ni: float) -> float:
    """metafor's ``.funconv2x2diag``: squared discrepancies between the observed statistics and those of the reconstructed table."""
    if par[0] * (1 - par[2]) + par[2] * (1 - par[1]) <= EPS:
        return 10.0
    rec = rec2x2diag(float(par[0]), float(par[1]), float(par[2]), ni)
    with numpy.errstate(all='ignore'):
        sens = rec[0] / (rec[0] + rec[2])
        spec = rec[3] / (rec[1] + rec[3])
        ppv = rec[0] / (rec[0] + rec[1])
        npv = rec[3] / (rec[2] + rec[3])
    return float((obs[0] - sens) ** 2 + (obs[1] - spec) ** 2 + (obs[2] - ppv) ** 2 + (obs[3] - npv) ** 2 + (ni - rec.sum()) ** 2)


def largest_remainder(x: FloatArray, n: float) -> FloatArray:
    """metafor's ``.largestremaindermethod``: round down and give the remaining units to the largest fractional parts."""
    floor = numpy.floor(x)
    remainder = n - floor.sum()
    out = floor.copy()
    if remainder > 0:
        order = numpy.argsort(-(x - floor), kind='stable')
        out[order[: int(remainder)]] += 1
    return out


def diag_cells(sens: float, spec: float, ppv: float, npv: float, ni: float, method: str) -> FloatArray:
    if method == 'simple':
        return rec2x2diag(sens, spec, ppv, ni)
    obs = numpy.array([sens, spec, ppv, npv])
    if method == 'optim':
        try:
            res = metafopy.core.bfgs.optim_bfgs(lambda par: funconv2x2diag(par, obs, ni), [sens, spec, ppv])
        except metafopy.errors.EstimationError, ArithmeticError, ValueError:
            return numpy.full(4, numpy.nan)
        return rec2x2diag(float(res.par[0]), float(res.par[1]), float(res.par[2]), ni)
    grids = [numpy.linspace(max(0.0, v - 0.005), min(1.0, v + 0.005), 21) for v in (sens, spec, ppv)]
    loss = numpy.empty((21, 21, 21))
    for i1, a in enumerate(grids[0]):
        for i2, b in enumerate(grids[1]):
            for i3, c in enumerate(grids[2]):
                loss[i1, i2, i3] = funconv2x2diag(numpy.array([a, b, c]), obs, ni)
    # which(arr.ind=TRUE) lists the hits with the first index running fastest; metafor takes the first
    hits = numpy.argwhere(loss.transpose(2, 1, 0) == numpy.nanmin(loss))
    i3, i2, i1 = hits[0]
    return rec2x2diag(float(grids[0][i1]), float(grids[1][i2]), float(grids[2][i3]), ni)


def conv_2x2(
    ori: Arg = None,
    ri: Arg = None,
    x2i: Arg = None,
    ni: Arg = None,
    n1i: Arg = None,
    n2i: Arg = None,
    sens: Arg = None,
    spec: Arg = None,
    ppv: Arg = None,
    npv: Arg = None,
    correct: bool | numpy.typing.ArrayLike | str = True,
    drop01: bool = True,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    include: numpy.typing.ArrayLike | str | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    var_names: collections.abc.Sequence[str] = ('ai', 'bi', 'ci', 'di'),
    append: bool = True,
    replace: Replace = 'ifna',
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    method: typing.Literal['simple', 'optim', 'search'] = 'optim',
) -> pandas.DataFrame:
    """Reconstruct the cell frequencies of 2x2 tables.

    From the odds ratio ``ori``, the phi coefficient ``ri`` or the (signed) chi-square statistic ``x2i``
    with the total ``ni`` and the marginal counts ``n1i`` (first row) and ``n2i`` (first column); or from
    at least three of ``sens``, ``spec``, ``ppv`` and ``npv`` with ``ni``. ``correct`` says whether
    ``x2i`` includes Yates' continuity correction; ``drop01`` skips diagnostic statistics equal to 0 or 1.
    ``method`` is how the table is fitted to the diagnostic statistics.

    Returns the cells as columns ``var_names`` (added to ``data`` when given and ``append=True``).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    repl = match_replace(replace)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if method not in ('simple', 'optim', 'search'):
        raise metafopy.errors.InputError('\'arg\' should be one of "simple", "optim", "search"')
    if len(var_names) != 4:
        raise metafopy.errors.InputError("Argument 'var_names' must be of length 4.")
    x = as_frame(data)
    inp = metafopy.effects.measures.Inputs(x, dict(ori=ori, ri=ri, x2i=x2i, sens=sens, spec=spec, ppv=ppv, npv=npv, ni=ni, n1i=n1i, n2i=n2i))
    raw = {name: inp.get(name) for name in ('ori', 'ri', 'x2i', 'sens', 'spec', 'ppv', 'npv', 'ni', 'n1i', 'n2i')}
    if not metafopy.effects.measures.equal_length(*raw.values()):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    k = max(v.shape[0] for v in raw.values() if v is not None) if any(v is not None for v in raw.values()) else 0
    v = {name: fill(val, k) for name, val in raw.items()}

    if isinstance(correct, str):
        corr = metafopy.effects.datatools.column_or_values(correct, x, 'correct')
    else:
        corr = numpy.atleast_1d(numpy.asarray(correct, dtype=numpy.float64)).reshape(-1)
    if corr is None:
        corr = numpy.ones(1)
    if corr.shape[0] == 1:
        corr = numpy.repeat(corr, k)
    if corr.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of the 'correct' argument ({corr.shape[0]}) does not match the length of the data ({k}).")
    corr_b = numpy.where(numpy.isnan(corr), True, corr != 0)

    inc = include_mask(include, k, x)
    for name in v:
        v[name][~inc] = numpy.nan
    with numpy.errstate(invalid='ignore'):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        for name in ('ni', 'n1i', 'n2i'):
            v[name] = numpy.round(v[name])

    E = metafopy.errors.InputError
    with numpy.errstate(invalid='ignore'):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if any(numpy.any(v[n] < 0) for n in ('ni', 'n1i', 'n2i')):
            raise E('One or more sample sizes or marginal counts are negative.')
        if numpy.any(v['n1i'] > v['ni']) or numpy.any(v['n2i'] > v['ni']):
            raise E('One or more marginal counts are larger than the sample sizes.')
        if numpy.any(numpy.abs(v['ri']) > 1):
            raise E('One or more phi coefficients are > 1 or < -1.')
        for name, what in (('sens', 'sensitivity'), ('spec', 'specificity'), ('ppv', 'positive predictive'), ('npv', 'negative predictive')):
            if numpy.any((v[name] < 0) | (v[name] > 1)):
                raise E(f'One or more {what} values are < 0 or > 1.')

    ri_v, ori_v, x2i_v = v['ri'], v['ori'], v['x2i']
    sens_v, spec_v, ppv_v, npv_v = v['sens'], v['spec'], v['ppv'], v['npv']
    ni_v, n1_v, n2_v = v['ni'], v['n1i'], v['n2i']
    p1 = n1_v / ni_v
    p2 = n2_v / ni_v
    cells = numpy.full((k, 4), numpy.nan)

    for i in range(k):
        if math.isnan(ni_v[i]):
            continue
        diag = [sens_v[i], spec_v[i], ppv_v[i], npv_v[i]]
        if sum(not math.isnan(d) for d in diag) >= 3:
            if drop01 and any(d in (0.0, 1.0) for d in diag):
                continue
            se, sp, pp, npv_i = diag
            with numpy.errstate(all='ignore'):
                if math.isnan(se):
                    se = (1 - sp) * pp * npv_i / ((1 - sp) * pp * npv_i + sp * (1 - pp) * (1 - npv_i))
                if math.isnan(sp):
                    sp = (1 - se) * pp * npv_i / ((1 - se) * pp * npv_i + se * (1 - pp) * (1 - npv_i))
                if math.isnan(pp):
                    pp = se * sp * (1 - npv_i) / (se * sp * (1 - npv_i) + (1 - se) * (1 - sp) * npv_i)
                if math.isnan(npv_i):
                    npv_i = se * sp * (1 - pp) / (se * sp * (1 - pp) + (1 - se) * (1 - sp) * pp)
            if se * (1 - pp) + pp * (1 - sp) == 0:
                continue
            with numpy.errstate(all='ignore'):
                tmp = diag_cells(se, sp, pp, npv_i, float(ni_v[i]), method)
            cells[i] = largest_remainder(tmp, float(ni_v[i]))
        else:
            if math.isnan(n1_v[i]) or math.isnan(n2_v[i]):
                continue
            p11 = math.nan
            with numpy.errstate(all='ignore'):
                if not math.isnan(ori_v[i]):
                    if ori_v[i] == 1:
                        p11 = n1_v[i] * n2_v[i] / ni_v[i] ** 2
                    else:
                        p1_, p2_, q1_ = p1[i], 1 - p1[i], p2[i]
                        xx = ori_v[i] * (p1_ + q1_) + p2_ - q1_
                        yy = math.sqrt(xx**2 - 4 * p1_ * q1_ * ori_v[i] * (ori_v[i] - 1))
                        p11 = (xx - yy) / (2 * (ori_v[i] - 1))
                r = ri_v[i]
                if math.isnan(r) and not math.isnan(x2i_v[i]):
                    sign = float(numpy.sign(x2i_v[i]))
                    r = sign * math.sqrt(abs(x2i_v[i]) / ni_v[i])
                    if corr_b[i]:
                        r = sign * (
                            math.sqrt(abs(x2i_v[i]) / ni_v[i]) + ni_v[i] / (2 * math.sqrt(n1_v[i] * (ni_v[i] - n1_v[i]) * n2_v[i] * (ni_v[i] - n2_v[i])))
                        )
                if math.isnan(p11) and not math.isnan(r):
                    p11 = p1[i] * p2[i] + r * math.sqrt(p1[i] * (1 - p1[i]) * p2[i] * (1 - p2[i]))
            a = round(float(ni_v[i] * p11)) if not math.isnan(p11) else math.nan
            b = n1_v[i] - a
            c = n2_v[i] - a
            cells[i] = [a, b, c, ni_v[i] - a - b - c]

    with numpy.errstate(invalid='ignore'):
        hasneg = (cells < 0).any(axis=1)
    if hasneg.any():
        rows = ','.join(str(i) for i in numpy.flatnonzero(hasneg))
        warnings.warn(f'There are negative cell frequencies in table{"s " if hasneg.sum() > 1 else " "}{rows}.', stacklevel=2)
        cells[hasneg] = numpy.nan

    if x is not None and append:
        for j, name in enumerate(var_names):
            replace_column(x, name, cells[:, j], repl)
        return x
    return pandas.DataFrame({name: quiet(cells[:, j]) for j, name in enumerate(var_names)})


# --------------------------------------------------------------------------
# conv.delta


def transf_params(transf: collections.abc.Callable[..., typing.Any]) -> list[str]:
    """The names of the arguments of ``transf`` after the first (R: ``names(formals(args(transf)))[-1]`` without ``...``)."""
    try:
        params = list(inspect.signature(transf).parameters.values())
    except TypeError, ValueError:
        return []
    named = [p for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return [p.name for p in named[1:]]


def scalar(value: typing.Any) -> float:
    arr = numpy.asarray(value, dtype=numpy.float64).reshape(-1)
    return float(arr[0]) if arr.shape[0] >= 1 else math.nan


def conv_delta(
    yi: Arg,
    vi: Arg,
    ni: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    include: numpy.typing.ArrayLike | str | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    var_names: collections.abc.Sequence[str] | None = None,
    append: bool = True,
    replace: Replace = 'ifna',
    **kwargs: typing.Any,
) -> pandas.DataFrame:
    """Transform observed effect sizes and their sampling variances with the delta method.

    ``transf(yi[i], ...)`` gives the transformed estimate and ``vi[i] * transf'(yi[i])**2`` its sampling
    variance (the derivative is numerical). Keyword arguments that name further arguments of ``transf``
    are passed to it, one value per study (values or column names of ``data``). The results go into the
    columns ``var_names`` (default: the escalc variables of ``data``, else ``yi``/``vi``), where
    ``replace`` controls which existing values are overwritten; the result is escalc output.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if yi is None or vi is None:
        raise metafopy.errors.InputError("Must specify the 'yi' and 'vi' arguments.")
    if transf is None:
        raise metafopy.errors.InputError("Must specify the 'transf' argument.")
    repl = match_replace(replace)
    x = as_frame(data)
    prev_attrs: dict[collections.abc.Hashable | None, typing.Any] = dict(data.attrs) if isinstance(data, pandas.DataFrame) else {}
    if x is not None:
        x.attrs = dict(prev_attrs)
    yi_name, vi_name = yi_vi_names(x, var_names)

    inp = metafopy.effects.measures.Inputs(x, dict(yi=yi, vi=vi, ni=ni))
    yv, vv, nv = inp.get('yi'), inp.get('vi'), inp.get('ni')
    if yv is None or vv is None:
        raise metafopy.errors.InputError("Must specify the 'yi' and 'vi' arguments.")
    if yv.shape[0] != vv.shape[0]:
        raise metafopy.errors.InputError("Length of 'yi' and 'vi' are not the same.")
    if not metafopy.effects.measures.equal_length(yv, vv, nv):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    k = yv.shape[0]
    nv = fill(nv, k)
    yv, vv = yv.copy(), vv.copy()
    inc = include_mask(include, k, x)
    yv[~inc] = numpy.nan
    vv[~inc] = numpy.nan
    nv[~inc] = numpy.nan

    extra: dict[str, FloatArray] = {}
    params = transf_params(transf)
    for name, value in kwargs.items():
        if name not in params:
            continue
        arr = metafopy.effects.datatools.column_or_values(value, x, name)
        if arr is None:
            continue
        extra[name] = numpy.repeat(arr, k) if arr.shape[0] == 1 else arr

    yt = numpy.full(k, numpy.nan)
    vt = numpy.full(k, numpy.nan)
    for i in range(k):
        args_i = {name: float(arr[i]) for name, arr in extra.items()}

        def f(z: float, args_i: dict[str, float] = args_i) -> float:
            return scalar(transf(z, **args_i))

        with warnings.catch_warnings(), numpy.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                yt[i] = f(float(yv[i]))
            except Exception:  # auditme: accept[silent-broad-except] R wraps the call in try() and records NA
                yt[i] = math.nan
            try:
                vt[i] = vv[i] * metafopy.core.numderiv.grad_scalar(f, float(yv[i])) ** 2
            except Exception:  # auditme: accept[silent-broad-except] R wraps the call in try() and records NA
                vt[i] = math.nan

    if x is None:
        x = pandas.DataFrame({yi_name: numpy.full(k, numpy.nan), vi_name: numpy.full(k, numpy.nan)})
    replace_column(x, yi_name, yt, repl)
    src_ni = attr_ni(x, yi) if isinstance(yi, str) else None
    ni_out = metafopy.effects.measures.replmiss(nv, src_ni)
    ni_out = metafopy.effects.measures.replmiss(ni_out, attr_ni(x, yi_name))
    replace_column(x, vi_name, vt, repl)
    return finish(x, yi_name, vi_name, ni_out, 'GEN', append, prev_attrs)


# --------------------------------------------------------------------------
# conv.wald


def conv_wald(
    out: Arg = None,
    ci_lb: Arg = None,
    ci_ub: Arg = None,
    zval: Arg = None,
    pval: Arg = None,
    n: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any] | None = None,
    include: numpy.typing.ArrayLike | str | None = None,
    level: numpy.typing.ArrayLike | str = 95,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    check: bool = True,
    var_names: collections.abc.Sequence[str] | None = None,
    append: bool = True,
    replace: Replace = 'ifna',
    cifac: float = 0.1,
) -> pandas.DataFrame:
    """Convert Wald-type confidence intervals and tests into sampling variances.

    ``vi`` is ``((ci_ub - ci_lb) / (2 z))**2`` (``((ci_ub - out) / z)**2`` without a lower bound) for the
    ``level``-% interval, else ``(out / zval)**2`` with ``zval`` from the two-sided ``pval`` if needed.
    ``transf`` (e.g. ``numpy.log`` for odds ratios) is applied to ``out`` and the bounds first; ``n`` is
    kept as the ``ni`` attribute. ``check`` warns when ``out`` is not about halfway between the bounds
    (more than ``cifac`` of the interval width off).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if out is None and ci_lb is None and ci_ub is None and zval is None and pval is None:
        raise metafopy.errors.InputError("Must specify at least some of these arguments: 'out', 'ci.lb', 'ci.ub', 'zval', 'pval'.")
    repl = match_replace(replace)
    x = as_frame(data)
    prev_attrs: dict[collections.abc.Hashable | None, typing.Any] = dict(data.attrs) if isinstance(data, pandas.DataFrame) else {}
    if x is not None:
        x.attrs = dict(prev_attrs)
    yi_name, vi_name = yi_vi_names(x, var_names)

    inp = metafopy.effects.measures.Inputs(x, dict(out=out, ci_lb=ci_lb, ci_ub=ci_ub, zval=zval, pval=pval, n=n))
    raw = {name: inp.get(name) for name in ('out', 'ci_lb', 'ci_ub', 'zval', 'pval', 'n')}
    if not metafopy.effects.measures.equal_length(*raw.values()):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    k = max(v.shape[0] for v in raw.values() if v is not None)
    v = {name: fill(val, k) for name, val in raw.items()}
    inc = include_mask(include, k, x)
    for name in v:
        v[name][~inc] = numpy.nan
    with numpy.errstate(invalid='ignore'):
        if numpy.any(v['pval'] < 0) or numpy.any(v['pval'] > 1):
            raise metafopy.errors.InputError('One or more p-values are < 0 or > 1.')

    lvl = (
        metafopy.effects.datatools.column_or_values(level, x, 'level')
        if isinstance(level, str)
        else numpy.atleast_1d(numpy.asarray(level, dtype=numpy.float64))
    )
    if lvl is None:
        lvl = numpy.array([95.0])
    if lvl.shape[0] == 1:
        lvl = numpy.repeat(lvl, k)
    if lvl.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of the 'level' argument ({lvl.shape[0]}) does not correspond to the size of the dataset ({k}).")
    alpha = numpy.array([metafopy.effects.summary.level_to_alpha(float(a)) for a in lvl])
    crit = scipy.stats.norm.isf(alpha / 2)

    o, lb, ub = v['out'], v['ci_lb'], v['ci_ub']
    if transf is not None:
        with numpy.errstate(all='ignore'):
            o, lb, ub = (numpy.array([scalar(transf(val)) for val in arr]) for arr in (o, lb, ub))

    if x is None:
        x = pandas.DataFrame({yi_name: numpy.full(k, numpy.nan), vi_name: numpy.full(k, numpy.nan)})
    replace_column(x, yi_name, o, repl)
    ni_out = attr_ni(x, yi_name)
    ni_out = v['n'] if ni_out is None else metafopy.effects.measures.replmiss(ni_out, v['n'])

    yi_now = x[yi_name].to_numpy(dtype=numpy.float64)
    with numpy.errstate(all='ignore'):
        vi_new: FloatArray | None = numpy.where(numpy.isnan(lb), ((ub - o) / crit) ** 2, ((ub - lb) / (2 * crit)) ** 2)
        if check:
            diffs = numpy.abs((ub + lb) / 2 - yi_now) / (ub - lb)
            large = (diffs > cifac) & ~numpy.isnan(diffs)
            if vi_name in x.columns:
                large &= numpy.isnan(x[vi_name].to_numpy(dtype=numpy.float64))
            if large.any():
                rows = numpy.flatnonzero(large).tolist()
                txt = ', '.join(str(r) for r in rows[:5]) + (', ...' if len(rows) > 5 else '')
                warnings.warn(f"The observed outcome does not appear to be halfway between '(ci_lb, ci_ub)' in row(s): {txt}", stacklevel=2)
        z = metafopy.effects.measures.replmiss(v['zval'], scipy.stats.norm.isf(v['pval'] / 2))
        vi_new = metafopy.effects.measures.replmiss(vi_new, (yi_now / z) ** 2)
    if vi_new is None:
        raise metafopy.errors.InternalError('vi_new must not be None')
    replace_column(x, vi_name, vi_new, repl)
    measure = attr_measure(x, yi_name)
    return finish(x, yi_name, vi_name, ni_out, measure, append, prev_attrs)
