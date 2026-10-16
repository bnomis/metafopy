"""Effect sizes and sampling variances (metafor's ``escalc``).

A port of ``escalc()`` from metafor 5.2-1 (``R/escalc.r``). Every measure,
``vtype`` option, zero-cell adjustment (``add``/``to``/``drop00``) and the
``addyi``/``addvi``/``onlyo1`` switches follow the R code path, so values agree
with metafor to floating-point rounding. Formulas and their sources are
documented in metafor's help page, ``?escalc``
(https://wviechtb.github.io/metafor/reference/escalc.html), and in
Viechtbauer (2010), doi:10.18637/jss.v036.i03.

Differences from R that follow Python conventions:

- arguments are given by keyword (or positionally in metafor's order) and may
  be array-likes, scalars, or column names in ``data``;
- numeric ``subset``/``include`` values are 0-based row positions;
- the result is a :class:`pandas.DataFrame`; metafor's attributes (``measure``,
  ``slab``, ``ni``, ``digits``, ``yi.names``, ``vi.names``) live in
  ``DataFrame.attrs`` under the same names with ``.`` replaced by ``_``.
"""

import collections.abc
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.integrate
import scipy.special
import scipy.stats

import metafopy.effects.na
import metafopy.effects.rtet
import metafopy.errors
import metafopy.metadata

__all__ = ['MEASURES', 'cmicalc', 'escalc']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Arg = numpy.typing.ArrayLike | str | None
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type To = typing.Literal['all', 'only0', 'if0all', 'none']

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
MEASURES: tuple[str, ...] = (
    # 2x2 table measures
    'RR', 'OR', 'PETO', 'RD', 'AS', 'PHI', 'ZPHI', 'YUQ', 'YUY', 'RTET', 'ZTET',
    # 2x2 table transformations to SMDs
    'PBIT', 'OR2D', 'OR2DN', 'OR2DL',
    # 2x2 table measures for matched pairs / pre-post data
    'MPRD', 'MPRR', 'MPOR', 'MPORC', 'MPPETO', 'MPORM',
    # two-group person-time data
    'IRR', 'IRD', 'IRSD',
    # two-group mean/SD measures
    'MD', 'POMPMD', 'SMD', 'SMDH', 'SMD1', 'SMD1H', 'ROM',
    'VR', 'CVR',
    'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL',
    # correlations
    'COR', 'UCOR', 'ZCOR',
    'PCOR', 'ZPCOR', 'SPCOR', 'ZSPCOR',
    'R2', 'ZR2', 'R2F', 'ZR2F',
    # single proportions
    'PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT',
    # single-group person-time data
    'IR', 'IRLN', 'IRS', 'IRFT',
    # single-group means, SDs, CVs
    'MN', 'POMPMN', 'SMN', 'MNLN', 'SDLN', 'CVLN',
    # dependent-samples measures
    'MC', 'POMPMC', 'SMCC', 'SMCR', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC',
    # Cronbach's alpha
    'ARAW', 'AHW', 'ABT',
    # relative excess heterozygosity, common language effect size / AUC
    'REH', 'CLES', 'CLESN', 'AUC', 'AUCN',
    # generic outcomes
    'GEN',
)  # fmt: skip

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
VTYPES = ('UB', 'LS', 'LS2', 'LS3', 'HO', 'ST', 'CS', 'AV', 'AV2', 'AVHO', 'H0', 'H0a', 'H0b', 'MAX')

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
TWO_BY_TWO = (
    'RR',
    'OR',
    'RD',
    'AS',
    'PETO',
    'PHI',
    'ZPHI',
    'YUQ',
    'YUY',
    'RTET',
    'ZTET',
    'PBIT',
    'OR2D',
    'OR2DN',
    'OR2DL',
    'MPRD',
    'MPRR',
    'MPOR',
    'MPORC',
    'MPPETO',
    'MPORM',
)
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
MEANS_2G = ('MD', 'POMPMD', 'SMD', 'SMDH', 'SMD1', 'SMD1H', 'ROM', 'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL', 'VR', 'CVR')
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
CHANGE = ('MC', 'POMPMC', 'SMCC', 'SMCR', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC')

INCOMPLETE = 'Cannot compute outcomes. Check that all of the required information is specified\n  via the appropriate arguments'


# --------------------------------------------------------------------------
# helpers mirroring metafor's hidden utility functions


def cmicalc(mi: numpy.typing.ArrayLike, method: bool | str = 'exact') -> FloatArray:
    """Small-sample bias correction factor ``c(m)`` for SMDs (Hedges, 1981, doi:10.3102/10769986006002107).

    ``method`` is ``"exact"`` (gamma-function form, computed on the log scale),
    ``"approx"`` (``1 - 3/(4m - 1)``), or a logical (``True`` = exact, ``False`` = none).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    m = numpy.asarray(mi, dtype=numpy.float64)
    if isinstance(method, bool):
        kind = 'exact' if method else 'none'
    else:
        opts = [o for o in ('exact', 'approx') if o.startswith(method)] if method else []
        if not opts:
            raise metafopy.errors.InputError("Correction method must be either 'exact' or 'approx'.")
        kind = opts[0]
    with numpy.errstate(all='ignore'):
        if kind == 'exact':
            out = numpy.where(
                m <= 1, metafopy.effects.na.NA, numpy.exp(scipy.special.gammaln(m / 2) - numpy.log(numpy.sqrt(m / 2)) - scipy.special.gammaln((m - 1) / 2))
            )
        elif kind == 'approx':
            out = numpy.where(m <= 1, metafopy.effects.na.NA, 1 - 3 / (4 * m - 1))
        else:
            out = numpy.ones_like(m)
    # comparisons with NaN give False above, so NaN inputs propagate through the formula
    return numpy.asarray(out, dtype=numpy.float64)


def fcalc(a: float, b: float, g: FloatArray, x: FloatArray) -> FloatArray:
    """Gauss hypergeometric 2F1(a, b; g; x); NaN unless ``g > a + b`` (metafor's ``.Fcalc``)."""
    g, x = numpy.broadcast_arrays(numpy.asarray(g, dtype=numpy.float64), numpy.asarray(x, dtype=numpy.float64))
    out = numpy.full(g.shape, metafopy.effects.na.NA)
    ok = ~numpy.isnan(g) & ~numpy.isnan(x) & (g > a + b)
    out[ok] = scipy.special.hyp2f1(a, b, g[ok], x[ok])
    # SciPy's hyp2f1 can fail (NaN) for large g with x near 1, where GSL's does not;
    # fall back to Euler's integral representation (DLMF 15.6.1), valid for g > b > 0.
    bad = ok & ~numpy.isfinite(out)
    for i in numpy.flatnonzero(bad):
        out[i] = hyp2f1_euler(a, b, float(g[i]), float(x[i]))
    return out


def hyp2f1_euler(a: float, b: float, c: float, x: float) -> float:
    val, _ = scipy.integrate.quad(lambda t: (1 - x * t) ** (-a), 0, 1, weight='alg', wvar=(b - 1, c - b - 1), epsabs=0, epsrel=1.2e-14, limit=200)
    return math.exp(scipy.special.gammaln(c) - scipy.special.gammaln(b) - scipy.special.gammaln(c - b)) * val


def wmean(x: FloatArray, w: FloatArray) -> float:
    i = ~(numpy.isnan(x) | numpy.isnan(w))
    with numpy.errstate(all='ignore'):
        return float(numpy.sum(x[i] * w[i]) / numpy.sum(w[i]))


def convp2t(pval: FloatArray | None, df: FloatArray) -> FloatArray | None:
    if pval is None:
        return None
    df = numpy.where(df < 1, metafopy.effects.na.NA, df)
    pval = numpy.where(numpy.abs(pval) > 1, metafopy.effects.na.NA, pval)
    with numpy.errstate(all='ignore'):
        return numpy.asarray(numpy.sign(pval) * scipy.stats.t.isf(numpy.abs(pval) / 2, df), dtype=numpy.float64)


def convp2f(pval: FloatArray | None, df1: FloatArray, df2: FloatArray) -> FloatArray | None:
    if pval is None:
        return None
    df1 = numpy.where(df1 < 1, metafopy.effects.na.NA, df1)
    df2 = numpy.where(df2 < 1, metafopy.effects.na.NA, df2)
    pval = numpy.where((pval < 0) | (pval > 1), metafopy.effects.na.NA, pval)
    with numpy.errstate(all='ignore'):
        return numpy.asarray(scipy.stats.f.isf(pval, df1, df2), dtype=numpy.float64)


def replmiss(x: FloatArray | None, y: FloatArray | None) -> FloatArray | None:
    """Replace NaNs in ``x`` by ``y`` (metafor's ``replmiss``); ``None`` means R's ``NULL``.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if y is None or y.size == 0:
        return x
    if x is None or x.size == 0:
        x = numpy.full(y.shape[0], metafopy.effects.na.NA)
    if y.shape[0] == 1:
        y = numpy.repeat(y, x.shape[0])
    if x.shape[0] != y.shape[0]:
        raise metafopy.errors.InputError("Length of 'x' and 'y' are not the same.")
    out = x.copy()
    na = numpy.isnan(out)
    out[na] = y[na]
    return out


def rtoz(x: FloatArray) -> FloatArray:
    x = numpy.clip(x, -1, 1)
    with numpy.errstate(all='ignore'):
        return numpy.asarray(numpy.arctanh(x), dtype=numpy.float64)


def expand(x: FloatArray | None, k: int) -> FloatArray | None:
    if x is not None and x.shape[0] == 1:
        return numpy.repeat(x, k)
    return x


def equal_length(*xs: FloatArray | None) -> bool:
    ks = {x.shape[0] for x in xs if x is not None and x.shape[0] > 0}
    return len(ks) <= 1


def all_specified(*xs: FloatArray | None) -> bool:
    return all(x is not None and x.shape[0] > 0 for x in xs)


def anyna(cond: FloatArray | BoolArray, *operands: FloatArray | None) -> bool:
    """``any(cond, na.rm=TRUE)``: ignore positions where an operand is NaN."""
    mask = numpy.asarray(cond, dtype=numpy.bool_)
    for op in operands:
        if op is not None:
            mask = mask & ~numpy.isnan(op)
    return bool(mask.any())


def vtypes(vtype: str | collections.abc.Sequence[str], k: int, allowed: collections.abc.Sequence[str], msg: str) -> numpy.typing.NDArray[numpy.str_]:
    vt = numpy.array([vtype] if isinstance(vtype, str) else list(vtype), dtype=numpy.str_)
    if vt.shape[0] == 1:
        vt = numpy.repeat(vt, k)
    if not numpy.isin(vt, allowed).all():
        raise metafopy.errors.InputError(f"For this outcome measure, 'vtype' must be {msg}.")
    return vt


def subset_array(x: FloatArray | None, subset: BoolArray | None) -> FloatArray | None:
    if x is None or subset is None:
        return x
    if x.shape[0] != subset.shape[0]:
        raise metafopy.errors.InputError(f'Element is not of length {subset.shape[0]}.')
    return x[subset]


def chksubset(x: numpy.typing.ArrayLike | str | None, k: int, data: pandas.DataFrame | None, name: str, stoponk0: bool = True) -> BoolArray | None:
    if x is None:
        return None
    if isinstance(x, str):
        if data is None:
            raise metafopy.errors.InputError(f'a string {name} needs data to evaluate against')
        values = numpy.asarray(data.eval(x))
    else:
        values = numpy.asarray(x)
    if values.ndim == 0:
        values = values[None]
    if values.shape[0] == 0:
        raise metafopy.errors.InputError(f"Argument '{name}' is of length 0.")
    if values.dtype == numpy.bool_ or values.dtype == object:
        if values.shape[0] != k:
            raise metafopy.errors.InputError(f"Length of the '{name}' argument ({values.shape[0]}) is not of length k = {k}.")
        mask = numpy.array([bool(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else False for v in values], dtype=numpy.bool_)
    else:
        idx = values.astype(numpy.float64)
        idx = idx[~numpy.isnan(idx)].round().astype(numpy.intp)
        if numpy.any(idx >= k) or numpy.any(idx < -k):
            raise metafopy.errors.InputError(f"Argument '{name}' includes values outside 0..{k - 1}.")
        mask = numpy.zeros(k, dtype=numpy.bool_)
        mask[idx] = True
    if stoponk0 and not mask.any():
        raise metafopy.errors.InputError('Stopped because k = 0 after subsetting.')
    return mask


def chkyisize(yi: FloatArray, measure: str, cutoff: float | None) -> None:
    addtxt = None
    if cutoff is None:
        cut = math.nan
        if measure in ('SMD', 'SMDH', 'SMD1', 'SMD1H'):
            cut = 2.0
            addtxt = 'Maybe SEs were used in place of SDs to compute them?'
    else:
        cut = abs(float(cutoff))
    if not math.isnan(cut) and anyna(numpy.abs(yi) > cut, yi):
        txt = f'One or more {measure} values are unusually large (i.e., abs(yi) > {cut:g}).'
        if addtxt:
            txt += f'\n{addtxt}'
        warnings.warn(txt, stacklevel=3)


# --------------------------------------------------------------------------


class Inputs:
    """Resolves escalc arguments (arrays, scalars or column names) the way ``.getx`` does."""

    def __init__(self, data: pandas.DataFrame | None, values: dict[str, Arg]) -> None:
        self.data = data
        self.values = dict(values)

    def get(self, name: str) -> FloatArray | None:
        value = self.values.get(name)
        if value is None:
            return None
        if isinstance(value, str):
            if self.data is None:
                raise metafopy.errors.InputError(f'{name}={value!r} names a column, but no data was given')
            if value not in self.data.columns:
                raise metafopy.errors.NotFoundError(f'column {value!r} (for {name}) not found in data')
            value = self.data[value].to_numpy()
        try:
            arr = numpy.asarray(value, dtype=numpy.float64)
        except (TypeError, ValueError) as err:
            raise metafopy.errors.ArgumentError(f"The object/variable specified for the '{name}' argument is not numeric.") from err
        if arr.ndim == 0:
            arr = arr[None]
        if arr.ndim != 1:
            raise metafopy.errors.InputError(f'{name} must be one-dimensional, got shape {arr.shape}')
        # missing values supplied by the user are R's NA
        return numpy.where(numpy.isnan(arr), metafopy.effects.na.NA, arr)


def escalc(
    measure: str | None = None,
    ai: Arg = None,
    bi: Arg = None,
    ci: Arg = None,
    di: Arg = None,
    n1i: Arg = None,
    n2i: Arg = None,
    x1i: Arg = None,
    x2i: Arg = None,
    t1i: Arg = None,
    t2i: Arg = None,
    m1i: Arg = None,
    m2i: Arg = None,
    sd1i: Arg = None,
    sd2i: Arg = None,
    xi: Arg = None,
    mi: Arg = None,
    ri: Arg = None,
    ti: Arg = None,
    fi: Arg = None,
    pi: Arg = None,
    sdi: Arg = None,
    r2i: Arg = None,
    mini: Arg = None,
    maxi: Arg = None,
    ni: Arg = None,
    yi: Arg = None,
    vi: Arg = None,
    sei: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, numpy.typing.ArrayLike] | None = None,
    slab: numpy.typing.ArrayLike | str | None = None,
    flip: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    include: numpy.typing.ArrayLike | str | None = None,
    add: float | None = None,
    to: To = 'only0',
    drop00: bool = False,
    vtype: str | collections.abc.Sequence[str] = 'LS',
    correct: bool | str = True,
    cutoff: float | None = None,
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    var_names: collections.abc.Sequence[str] = ('yi', 'vi'),
    add_measure: bool = False,
    append: bool = True,
    replace: bool = True,
    digits: int | collections.abc.Sequence[int] | None = None,
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
) -> pandas.DataFrame:
    """Calculate effect sizes ``yi`` and sampling variances ``vi`` (metafor's ``escalc``).

    See https://wviechtb.github.io/metafor/reference/escalc.html for the
    measures, their required arguments and the ``vtype`` options. ``add``
    defaults to 1/2, except for measures where metafor uses 0 (AS, PHI, ZPHI,
    RTET, ZTET, IRSD, PAS, PFT, IRS, IRFT). ``onlyo1``, ``addyi`` and
    ``addvi`` correspond to metafor's ``...`` options of the same names.

    Returns a :class:`pandas.DataFrame` with columns ``var_names`` (appended
    to ``data`` when given and ``append=True``); ``attrs['measure']``,
    ``attrs['slab']`` and ``attrs['ni']`` carry metafor's attributes.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
        metafopy.errors.NotSupportedError: the requested combination is not implemented.
        metafopy.errors.InternalError: an internal invariant does not hold.
    """
    if measure is None and yi is None:
        raise metafopy.errors.InputError("Must specify an effect size or outcome measure via the 'measure' argument.")
    if yi is not None and measure is None:
        measure = 'GEN'
    if not isinstance(measure, str):
        raise metafopy.errors.ArgumentError("The 'measure' argument must be a character string.")
    # metafor 5.2-1 accepts 'HR' and 'HD' but has no code to compute them: they work only with yi and vi/sei
    # (as for measure='GEN') and otherwise fail with an R-internal error ('argument "vi" is missing')
    reserved = ('HR', 'HD')
    if measure in reserved and yi is None:
        raise metafopy.errors.NotSupportedError(f"measure={measure!r} is reserved in metafor but not computed there; supply 'yi' and 'vi' or 'sei'.")
    if measure not in MEASURES and measure not in reserved:
        raise metafopy.errors.InputError("Unknown 'measure' specified.")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to not in ('all', 'only0', 'if0all', 'none'):
        raise metafopy.errors.InputError("Unknown 'to' argument specified.")
    if not isinstance(drop00, bool):
        raise metafopy.errors.ArgumentError("Unknown 'drop00' argument specified.")
    vtype_list = [vtype] if isinstance(vtype, str) else list(vtype)
    if any(v not in VTYPES for v in vtype_list if v is not None):
        raise metafopy.errors.InputError("Unknown 'vtype' argument specified.")

    var_names = list(var_names)
    if add_measure:
        if len(var_names) == 2:
            var_names.append('measure')
        if len(var_names) != 3:
            raise metafopy.errors.InputError("Argument 'var_names' must be of length 2 or 3.")
    else:
        if len(var_names) == 3:
            var_names = var_names[:2]
        if len(var_names) != 2:
            raise metafopy.errors.InputError("Argument 'var_names' must be of length 2.")

    if add is None:
        add = 0.0 if measure in ('AS', 'PHI', 'ZPHI', 'RTET', 'ZTET', 'IRSD', 'PAS', 'PFT', 'IRS', 'IRFT') else 0.5

    if measure in ('ROM', 'VR', 'CVR', 'SDLN', 'CVLN', 'ROMC', 'CVRC') and not isinstance(correct, bool):
        raise metafopy.errors.ArgumentError("Argument 'correct' must be a logical for this measure.")

    has_data = data is not None
    if data is not None and not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)
    if not (data is None or isinstance(data, pandas.DataFrame)):
        raise metafopy.errors.InternalError('unexpected state: data is None or isinstance(data, pandas.DataFrame)')

    args: dict[str, Arg] = dict(
        ai=ai, bi=bi, ci=ci, di=di, n1i=n1i, n2i=n2i, x1i=x1i, x2i=x2i, t1i=t1i, t2i=t2i, m1i=m1i, m2i=m2i, sd1i=sd1i, sd2i=sd2i,
        xi=xi, mi=mi, ri=ri, ti=ti, fi=fi, pi=pi, sdi=sdi, r2i=r2i, mini=mini, maxi=maxi, ni=ni, yi=yi, vi=vi, sei=sei,
    )  # fmt: skip
    inp = Inputs(data, args)

    with numpy.errstate(all='ignore'):
        if yi is None:
            out = compute(measure, inp, data, subset, add, to, drop00, vtype, correct, onlyo1, addyi, addvi)
        else:
            out = generic(inp, data, subset, slab)
    yv, vv, ni_u, k_all, sub = out.yi, out.vi, out.ni_u, out.k_all, out.subset
    if out.slab is not None and slab is None:
        slab = out.slab

    yv = numpy.asarray(yv, dtype=numpy.float64).ravel().copy()
    vv = numpy.asarray(vv, dtype=numpy.float64).ravel().copy()
    k = yv.shape[0]

    is_inf = numpy.isinf(yv) | numpy.isinf(vv)
    if is_inf.any():
        warnings.warn("Some 'yi' and/or 'vi' values equal to +-Inf. Recoded to NAs.", stacklevel=2)
        yv[is_inf] = metafopy.effects.na.NA
        vv[is_inf] = metafopy.effects.na.NA
    is_nan = (numpy.isnan(yv) & ~metafopy.effects.na.is_na(yv)) | (numpy.isnan(vv) & ~metafopy.effects.na.is_na(vv))
    yv[is_nan] = metafopy.effects.na.NA
    vv[is_nan] = metafopy.effects.na.NA

    chkyisize(yv, measure, cutoff)

    with numpy.errstate(all='ignore'):
        vv[vv < 0] = metafopy.effects.na.NA

    labels: list[str] | None = None
    if slab is not None:
        if isinstance(slab, str):
            if data is None:
                raise metafopy.errors.InputError(f'slab={slab!r} names a column, but no data was given')
            raw = data[slab].tolist()
        else:
            raw = numpy.asarray(slab, dtype=object).ravel().tolist()
        if len(raw) != k_all:
            raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(raw)}) does not correspond to the size of the dataset ({k_all}).")
        if sub is not None:
            raw = [s for s, keep in zip(raw, sub, strict=True) if keep]
        if any(s is None or (isinstance(s, float) and math.isnan(s)) for s in raw):
            raise metafopy.errors.InputError('NAs in study labels.')
        labels = [str(s) for s in raw]
        if len(set(labels)) != len(labels):
            labels = list(make_unique_dot1(labels))

    inc = chksubset(include, k_all, data, 'include', stoponk0=False)
    if inc is None:
        inc = numpy.ones(k_all, dtype=numpy.bool_)
    if sub is None:
        sub = numpy.ones(k_all, dtype=numpy.bool_)
    inc = inc[sub]

    flip_arr = flip_array(flip, k_all, data)
    flip_arr = flip_arr[sub]
    with numpy.errstate(all='ignore'):
        yv[inc] = flip_arr[inc] * yv[inc]
        vv[inc] = flip_arr[inc] ** 2 * vv[inc]
    # R's NA is a signalling NaN; hand back ordinary quiet NaNs
    yv[numpy.isnan(yv)] = math.nan
    vv[numpy.isnan(vv)] = math.nan

    yi_repl = numpy.ones(k, dtype=numpy.bool_)
    if has_data and append:
        if data is None:
            raise metafopy.errors.InternalError('data must not be None')
        dat = data.loc[sub].reset_index(drop=True).copy() if not sub.all() else data.reset_index(drop=True).copy()
        cols: list[tuple[str, FloatArray | None]] = [(var_names[0], yv), (var_names[1], vv)]
        for name, values in cols:
            if values is None:
                raise metafopy.errors.InternalError('values must not be None')
            if replace or name not in dat.columns:
                repl = numpy.ones(k, dtype=numpy.bool_)
            else:
                repl = dat[name].isna().to_numpy()
            if name not in dat.columns:
                dat[name] = metafopy.effects.na.NA
            col = dat[name].to_numpy(dtype=numpy.float64, copy=True)
            col[inc & repl] = values[inc & repl]
            dat[name] = col
            if name == var_names[0]:
                yi_repl = repl
        if add_measure:
            mname = var_names[2]
            if replace or mname not in dat.columns:
                mrepl = numpy.ones(k, dtype=numpy.bool_)
            else:
                mvals = dat[mname]
                mrepl = (mvals.isna() | (mvals == '')).to_numpy()
            mcol = dat[mname].astype(object).to_numpy(copy=True) if mname in dat.columns else numpy.full(k, None, dtype=object)
            mcol[~numpy.isnan(yv) & inc & mrepl] = measure
            dat[mname] = mcol
    else:
        dat = pandas.DataFrame({var_names[0]: numpy.where(inc, yv, metafopy.effects.na.NA), var_names[1]: numpy.where(inc, vv, metafopy.effects.na.NA)})
        if add_measure:
            mcol = numpy.full(k, None, dtype=object)
            mcol[~numpy.isnan(yv) & inc] = measure
            dat[var_names[2]] = mcol

    if add_measure:
        dat[var_names[2]] = dat[var_names[2]].where(dat[var_names[2]].notna(), '')

    prev_attrs = dict(data.attrs) if has_data and data is not None else {}
    prev_ni = prev_slab = None
    if has_data and append and data is not None:
        prev_ni = metafopy.metadata.frame_attr(data, 'ni')
        prev_slab = metafopy.metadata.frame_attr(data, 'slab')
        prev_ni = None if prev_ni is None else prev_ni.astype(numpy.float64)[sub]
        prev_slab = None if prev_slab is None else prev_slab[sub]
    ni_attr: FloatArray | None
    if ni_u is None:
        ni_attr = prev_ni
    elif prev_ni is None:
        ni_attr = numpy.where(inc, ni_u, metafopy.effects.na.NA)
    else:
        ni_attr = prev_ni.copy()
        ni_attr[inc & yi_repl] = ni_u[inc & yi_repl]
    dat.attrs = {}
    dat.attrs['measure'] = measure
    dat.attrs['slab'] = labels if labels is not None or prev_slab is None else [str(s) for s in prev_slab.tolist()]
    dat.attrs['ni'] = ni_attr
    dat.attrs[metafopy.metadata.ROWS_KEY] = list(dat.index)
    dat.attrs['digits'] = digits
    yi_names = [var_names[0], *[n for n in prev_attrs.get('yi_names', []) if n != var_names[0]]]
    vi_names = [var_names[1], *[n for n in prev_attrs.get('vi_names', []) if n != var_names[1]]]
    dat.attrs['yi_names'] = [n for n in yi_names if n in dat.columns]
    dat.attrs['vi_names'] = [n for n in vi_names if n in dat.columns]
    return dat


def make_unique_dot1(labels: collections.abc.Sequence[str]) -> tuple[str, ...]:
    """metafor's ``.make.unique``: every repeated label gets ``.1``, ``.2``, ... (including the first)."""
    counts: dict[str, int] = {}
    for s in labels:
        counts[s] = counts.get(s, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for s in labels:
        if counts[s] == 1:
            out.append(s)
        else:
            seen[s] = seen.get(s, 0) + 1
            out.append(f'{s}.{seen[s]}')
    return metafopy.metadata.make_unique(out)


def flip_array(flip: numpy.typing.ArrayLike | str | None, k_all: int, data: pandas.DataFrame | None) -> FloatArray:
    if flip is None:
        return numpy.ones(k_all)
    if isinstance(flip, str):
        if data is None:
            raise metafopy.errors.InputError(f'flip={flip!r} names a column, but no data was given')
        values = data[flip].to_numpy()
    else:
        values = numpy.asarray(flip)
    if values.ndim == 0:
        values = values[None]
    if values.dtype == numpy.bool_ or values.dtype == object:
        if values.shape[0] == 1:
            values = numpy.repeat(values, k_all)
        out = numpy.where([v is True or v is numpy.True_ for v in values], -1.0, 1.0)
    else:
        out = values.astype(numpy.float64)
    if out.shape[0] != k_all:
        raise metafopy.errors.InputError(f"Length of the 'flip' argument ({out.shape[0]}) does not correspond to the size of the dataset ({k_all}).")
    return numpy.asarray(out, dtype=numpy.float64)


class Out:
    __slots__ = ('k_all', 'ni_u', 'slab', 'subset', 'vi', 'yi')

    def __init__(
        self, yi: FloatArray, vi: FloatArray, ni_u: FloatArray | None, k_all: int, subset: BoolArray | None, slab: numpy.typing.ArrayLike | None = None
    ) -> None:
        self.yi = yi
        self.vi = vi
        self.ni_u = ni_u
        self.k_all = k_all
        self.subset = subset
        self.slab = slab


def generic(inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, slab: numpy.typing.ArrayLike | str | None) -> Out:
    yi = inp.get('yi')
    if yi is None:
        raise metafopy.errors.InternalError('yi must not be None')
    vi = inp.get('vi')
    sei = inp.get('sei')
    ni = inp.get('ni')
    if vi is None:
        if sei is None:
            raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
        vi = sei**2
    if not equal_length(yi, vi, ni):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    k_all = yi.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    return Out(req(subset_array(yi, sub)), req(subset_array(vi, sub)), subset_array(ni, sub), k_all, sub)


def req(x: FloatArray | None) -> FloatArray:
    if x is None:
        raise metafopy.errors.InternalError('x must not be None')
    return x


def compute(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    add: float,
    to: To,
    drop00: bool,
    vtype: str | collections.abc.Sequence[str],
    correct: bool | str,
    onlyo1: bool,
    addyi: bool,
    addvi: bool,
) -> Out:
    if measure in TWO_BY_TWO:
        return two_by_two(measure, inp, data, subset, add, to, drop00, vtype, onlyo1, addyi, addvi)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('IRR', 'IRD', 'IRSD'):
        return two_group_rates(measure, inp, data, subset, add, to, drop00, addyi, addvi)
    if measure in MEANS_2G:
        return two_group_means(measure, inp, data, subset, vtype, correct)
    if measure in ('CLES', 'AUC'):
        return cles(inp, data, subset, vtype)
    if measure in ('CLESN', 'AUCN'):
        return clesn(inp, data, subset, vtype)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('COR', 'UCOR', 'ZCOR'):
        return cor(measure, inp, data, subset, vtype)
    if measure in ('PCOR', 'ZPCOR', 'SPCOR', 'ZSPCOR'):
        return pcor(measure, inp, data, subset, vtype)
    if measure in ('R2', 'ZR2', 'R2F', 'ZR2F'):
        return r2(measure, inp, data, subset, vtype)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT'):
        return proportions(measure, inp, data, subset, add, to, vtype, addyi, addvi)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('IR', 'IRLN', 'IRS', 'IRFT'):
        return rates(measure, inp, data, subset, add, to, addyi, addvi)
    if measure in ('MN', 'POMPMN', 'SMN', 'MNLN', 'SDLN', 'CVLN'):
        return means(measure, inp, data, subset, vtype, correct)
    if measure in CHANGE:
        return change(measure, inp, data, subset, vtype, correct)
    if measure in ('ARAW', 'AHW', 'ABT'):
        return alpha(measure, inp, data, subset)
    if measure == 'REH':
        return reh(inp, data, subset)
    raise metafopy.errors.InputError("Unknown 'measure' specified.")  # pragma: no cover


def two_by_two(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    add: float,
    to: To,
    drop00: bool,
    vtype: str | collections.abc.Sequence[str],
    onlyo1: bool,
    addyi: bool,
    addvi: bool,
) -> Out:
    ai = inp.get('ai')
    bi = inp.get('bi')
    ci = inp.get('ci')
    di = inp.get('di')
    n1i = inp.get('n1i')
    n2i = inp.get('n2i')
    ri = inp.get('ri')
    pi = inp.get('pi')

    if not equal_length(ai, bi, ci, di, n1i, n2i, ri, pi):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if n1i is not None and ai is not None and bi is not None and anyna(n1i != ai + bi, n1i, ai, bi):
        raise metafopy.errors.InputError("One or more 'n1i' values are not equal to 'ai + bi'.")
    if n2i is not None and ci is not None and di is not None and anyna(n2i != ci + di, n2i, ci, di):
        raise metafopy.errors.InputError("One or more 'n2i' values are not equal to 'ci + di'.")

    if n1i is not None and ai is not None:
        bi = replmiss(bi, n1i - ai)
    if n2i is not None and ci is not None:
        di = replmiss(di, n2i - ci)

    if not all_specified(ai, bi, ci, di):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ai, bi, ci, di or ai, n1i, ci, n2i).')
    if ai is None:
        raise metafopy.errors.InternalError('ai must not be None')
    if bi is None:
        raise metafopy.errors.InternalError('bi must not be None')
    if ci is None:
        raise metafopy.errors.InternalError('ci must not be None')
    if di is None:
        raise metafopy.errors.InternalError('di must not be None')

    if measure == 'MPORM' and not (all_specified(ri) or all_specified(pi)):
        raise metafopy.errors.InputError("Need to specify also argument 'ri' (and/or 'pi') for this measure.")

    k_all = ai.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    if sub is not None:
        ai, bi, ci, di = req(subset_array(ai, sub)), req(subset_array(bi, sub)), req(subset_array(ci, sub)), req(subset_array(di, sub))
        ri = subset_array(ri, sub)
        pi = subset_array(pi, sub)

    n1i = ai + bi
    n2i = ci + di

    if anyna(numpy.concatenate([ai > n1i, ci > n2i]), numpy.concatenate([ai, ci]), numpy.concatenate([n1i, n2i])):
        raise metafopy.errors.InputError('One or more event counts are larger than the corresponding group sizes.')
    allc = numpy.concatenate([ai, bi, ci, di])
    if anyna(allc < 0, allc):
        raise metafopy.errors.InputError('One or more counts are negative.')
    alln = numpy.concatenate([n1i, n2i])
    if anyna(alln < 0, alln):
        raise metafopy.errors.InputError('One or more group sizes are negative.')
    if measure == 'MPORM' and ri is not None and anyna(numpy.abs(ri) > 1, ri):
        raise metafopy.errors.InputError('One or more correlations are > 1 or < -1.')
    if measure == 'MPORM' and pi is not None and anyna((pi < 0) | (pi > 1), pi):
        raise metafopy.errors.InputError('One or more proportions are > 1 or < 0.')

    ni_u = ai + bi + ci + di
    if measure == 'MPORM':
        ni_u = numpy.round(ni_u / 2)

    k = ai.shape[0]

    ai = ai.copy()
    bi = bi.copy()
    ci = ci.copy()
    di = di.copy()

    if drop00:
        id00 = ((ai == 0) & (ci == 0)) | ((bi == 0) & (di == 0))
        ai[id00] = metafopy.effects.na.NA
        bi[id00] = metafopy.effects.na.NA
        ci[id00] = metafopy.effects.na.NA
        di[id00] = metafopy.effects.na.NA

    ai_u, bi_u, ci_u, di_u = ai.copy(), bi.copy(), ci.copy(), di.copy()
    n1i_u = ai + bi
    n2i_u = ci + di

    if to == 'all':
        ai = ai + add
        ci = ci + add
        if not onlyo1:
            bi = bi + add
            di = di + add

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to in ('only0', 'if0all'):
        id0 = (ai == 0) | (ci == 0) | (bi == 0) | (di == 0)

    if to == 'only0':
        ai[id0] += add
        ci[id0] += add
        if not onlyo1:
            bi[id0] += add
            di[id0] += add

    if to == 'if0all' and id0.any():
        ai = ai + add
        ci = ci + add
        if not onlyo1:
            bi = bi + add
            di = di + add

    n1i = ai + bi
    n2i = ci + di
    ni = n1i + n2i
    if measure == 'MPORM':
        ni = numpy.round(ni / 2)

    p1i_u = ai_u / n1i_u
    p2i_u = ci_u / n2i_u
    p1i = ai / n1i
    p2i = ci / n2i

    if addvi:
        mnwp1i = wmean(p1i, n1i)
        mnwp2i = wmean(p2i, n2i)
        ppi = (ai + ci) / (n1i + n2i)
    else:
        mnwp1i_u = wmean(p1i_u, n1i_u)
        mnwp2i_u = wmean(p2i_u, n2i_u)
        ppi_u = (ai_u + ci_u) / (n1i_u + n2i_u)

    yi = numpy.full(k, metafopy.effects.na.NA)
    vi = numpy.full(k, metafopy.effects.na.NA)

    if measure == 'RR':
        yi = numpy.log(p1i) - numpy.log(p2i) if addyi else numpy.log(p1i_u) - numpy.log(p2i_u)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV', 'H0'), "either 'LS' or 'AV'")
        if addvi:
            ls = 1 / ai - 1 / n1i + 1 / ci - 1 / n2i
            av = (1 - mnwp1i) / (mnwp1i * n1i) + (1 - mnwp2i) / (mnwp2i * n2i)
            h0 = (1 - ppi) / ppi * (1 / n1i + 1 / n2i)
        else:
            ls = 1 / ai_u - 1 / n1i_u + 1 / ci_u - 1 / n2i_u
            av = (1 - mnwp1i_u) / (mnwp1i_u * n1i_u) + (1 - mnwp2i_u) / (mnwp2i_u * n2i_u)
            h0 = (1 - ppi_u) / ppi_u * (1 / n1i_u + 1 / n2i_u)
        vi = pick(vt, {'LS': ls, 'AV': av, 'H0': h0})

    if measure in ('OR', 'OR2D', 'OR2DN', 'OR2DL', 'MPORM'):
        yi = numpy.log(p1i / (1 - p1i)) - numpy.log(p2i / (1 - p2i)) if addyi else numpy.log(p1i_u / (1 - p1i_u)) - numpy.log(p2i_u / (1 - p2i_u))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV', 'H0'), "either 'LS' or 'AV'")
        if addvi:
            ls = 1 / ai + 1 / bi + 1 / ci + 1 / di
            av = 1 / (mnwp1i * (1 - mnwp1i) * n1i) + 1 / (mnwp2i * (1 - mnwp2i) * n2i)
            h0 = 1 / (ppi * (1 - ppi)) * (1 / n1i + 1 / n2i)
        else:
            ls = 1 / ai_u + 1 / bi_u + 1 / ci_u + 1 / di_u
            # metafor uses the adjusted group sizes here (R/escalc.r, OR with vtype="AV" and addvi=FALSE)
            av = 1 / (mnwp1i_u * (1 - mnwp1i_u) * n1i) + 1 / (mnwp2i_u * (1 - mnwp2i_u) * n2i)
            h0 = 1 / (ppi_u * (1 - ppi_u)) * (1 / n1i_u + 1 / n2i_u)
        vi = pick(vt, {'LS': ls, 'AV': av, 'H0': h0})

    if measure == 'RD':
        yi = p1i - p2i if addyi else p1i_u - p2i_u
        vt = vtypes(vtype, k, ('UB', 'LS', 'AV', 'H0'), "either 'UB', 'LS', or 'AV'")
        if addvi:
            ub = p1i * (1 - p1i) / (n1i - 1) + p2i * (1 - p2i) / (n2i - 1)
            ls = p1i * (1 - p1i) / n1i + p2i * (1 - p2i) / n2i
            av = mnwp1i * (1 - mnwp1i) / n1i + mnwp2i * (1 - mnwp2i) / n2i
            h0 = (1 / n1i + 1 / n2i) * ppi * (1 - ppi)
        else:
            ub = p1i_u * (1 - p1i_u) / (n1i_u - 1) + p2i_u * (1 - p2i_u) / (n2i_u - 1)
            ls = p1i_u * (1 - p1i_u) / n1i_u + p2i_u * (1 - p2i_u) / n2i_u
            av = mnwp1i_u * (1 - mnwp1i_u) / n1i_u + mnwp2i_u * (1 - mnwp2i_u) / n2i_u
            h0 = (1 / n1i_u + 1 / n2i_u) * ppi_u * (1 - ppi_u)
        vi = pick(vt, {'UB': ub, 'LS': ls, 'AV': av, 'H0': h0})

    if measure == 'PETO':
        xt = ai + ci
        yt = bi + di
        Ei = xt * n1i / ni
        Vi = xt * yt * (n1i / ni) * (n2i / ni) / (ni - 1)
        yi = (ai - Ei) / Vi
        vi = 1 / Vi

    if measure == 'AS':
        yi = numpy.arcsin(numpy.sqrt(p1i)) - numpy.arcsin(numpy.sqrt(p2i))
        vi = 1 / (4 * n1i) + 1 / (4 * n2i)

    if measure in ('PHI', 'ZPHI'):
        yi = (ai * di - bi * ci) / numpy.sqrt((ai + bi) * (ci + di) * (ai + ci) * (bi + di))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('ST', 'CS', 'LS'), "either 'ST', 'CS', or 'LS'")
        q1i = 1 - p1i
        q2i = 1 - p2i
        pi1_ = (ai + bi) / ni
        pi2_ = (ci + di) / ni
        pi_1 = (ai + ci) / ni
        pi_2 = (bi + di) / ni
        n1, n2, p1, p2 = n1i, n2i, p1i, p2i
        st = ((n1 + n2) ** 2 * (4 * n1**3 * p1**2 * p2 * q1i**2 * q2i
                                + 4 * n2**3 * p1 * p2**2 * q1i * q2i**2
                                + n1 * n2**2 * p2 * q2i * (p2 * q1i + p1 * q2i) * (p2 * q1i + p1 * (4 * q1i + q2i))
                                + n1**2 * n2 * p1 * q1i * (p2 * q1i + p1 * q2i) * (p1 * q2i + p2 * (q1i + 4 * q2i)))) / (
            4 * (ai + ci) ** 3 * (bi + di) ** 3
        )  # fmt: skip
        cs = (
            1
            / ni
            * (
                1
                - yi**2
                + yi * (1 + 1 / 2 * yi**2) * (pi1_ - pi2_) * (pi_1 - pi_2) / numpy.sqrt(pi1_ * pi2_ * pi_1 * pi_2)
                - 3 / 4 * yi**2 * ((pi1_ - pi2_) ** 2 / (pi1_ * pi2_) + (pi_1 - pi_2) ** 2 / (pi_1 * pi_2))
            )
        )
        vi = pick(vt, {'ST': st, 'CS': cs, 'LS': cs})

    if measure == 'YUQ':
        yi = (ai / bi) / (ci / di)
        yi = (yi - 1) / (yi + 1)
        vi = 1 / 4 * (1 - yi**2) ** 2 * (1 / ai + 1 / bi + 1 / ci + 1 / di)

    if measure == 'YUY':
        yi = (ai / bi) / (ci / di)
        yi = (numpy.sqrt(yi) - 1) / (numpy.sqrt(yi) + 1)
        vi = 1 / 16 * (1 - yi**2) ** 2 * (1 / ai + 1 / bi + 1 / ci + 1 / di)

    if measure in ('RTET', 'ZTET'):
        yi = numpy.full(k, metafopy.effects.na.NA)
        vi = numpy.full(k, metafopy.effects.na.NA)
        for i in range(k):
            if numpy.isnan(ai[i]) or numpy.isnan(bi[i]) or numpy.isnan(ci[i]) or numpy.isnan(di[i]):
                continue
            yi[i], vi[i] = metafopy.effects.rtet.rtet(float(ai[i]), float(bi[i]), float(ci[i]), float(di[i]))

    if measure in ('ZPHI', 'ZTET'):
        vi = vi / (1 - numpy.where(yi**2 > 1, 1, yi**2)) ** 2
        yi = rtoz(yi)

    if measure == 'PBIT':
        z1i = scipy.special.ndtri(p1i)
        z2i = scipy.special.ndtri(p2i)
        yi = z1i - z2i
        vi = 2 * math.pi * p1i * (1 - p1i) * numpy.exp(z1i**2) / n1i + 2 * math.pi * p2i * (1 - p2i) * numpy.exp(z2i**2) / n2i

    if measure in ('OR2D', 'OR2DL'):
        yi = math.sqrt(3) / math.pi * yi
        vi = 3 / math.pi**2 * vi

    if measure == 'OR2DN':
        yi = yi / 1.65
        vi = vi / 1.65**2

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('MPRD', 'MPRR', 'MPOR'):
        pi12 = bi / ni
        pi21 = ci / ni
        pi1_ = (ai + bi) / ni
        pi_1 = (ai + ci) / ni
        if measure == 'MPRD':
            yi = pi1_ - pi_1
            vi = pi12 * (1 - pi12) / ni + 2 * pi12 * pi21 / ni + pi21 * (1 - pi21) / ni
        if measure == 'MPRR':
            yi = numpy.log(pi1_) - numpy.log(pi_1)
            vi = (pi12 + pi21) / (ni * pi1_ * pi_1)
        if measure == 'MPOR':
            yi = numpy.log(pi1_ / (1 - pi1_)) - numpy.log(pi_1 / (1 - pi_1))
            vi = (pi12 * (1 - pi12) + pi21 * (1 - pi21) + 2 * pi12 * pi21) / (ni * pi1_ * (1 - pi1_) * pi_1 * (1 - pi_1))

    if measure == 'MPORM':
        if pi is not None:
            ai_p = pi * (ai_u + bi_u)
            bi_p = ai_u - ai_p
            ci_p = ci_u - ai_p
            di_p = bi_u - ci_u + ai_p
            ri_p = (ai_p * di_p - bi_p * ci_p) / numpy.sqrt((ai_p + bi_p) * (ci_p + di_p) * (ai_p + ci_p) * (bi_p + di_p))
            ri_p = numpy.where((ri_p < -1) | (ri_p > 1), metafopy.effects.na.NA, ri_p)
            ri = replmiss(ri, ri_p)
        if ri is None:
            raise metafopy.errors.InternalError('ri must not be None')
        if addvi:
            si = (ri * numpy.sqrt(ai * bi * ci * di) + (ai * bi)) / ni
            deltai = ni**2 * (ni * si - ai * bi) / (ai * bi * ci * di)
            vi = vi - 2 * deltai / ni
        else:
            si_u = (ri * numpy.sqrt(ai_u * bi_u * ci_u * di_u) + (ai_u * bi_u)) / ni_u
            # metafor uses the adjusted bi in this term (R/escalc.r, MPORM with addvi=FALSE)
            deltai_u = ni_u**2 * (ni_u * si_u - ai_u * bi) / (ai_u * bi_u * ci_u * di_u)
            vi = vi - 2 * deltai_u / ni_u

    if measure == 'MPORC':
        yi = numpy.log(bi) - numpy.log(ci)
        vi = 1 / bi + 1 / ci

    if measure == 'MPPETO':
        Ei = (bi + ci) / 2
        Vi = (bi + ci) / 4
        yi = (bi - Ei) / Vi
        vi = 1 / Vi

    return Out(yi, vi, ni_u, k_all, sub)


def pick(vt: numpy.typing.NDArray[numpy.str_], options: dict[str, FloatArray | float]) -> FloatArray:
    out = numpy.full(vt.shape[0], metafopy.effects.na.NA)
    for key, values in options.items():
        sel = vt == key
        if sel.any():
            out[sel] = numpy.broadcast_to(numpy.asarray(values, dtype=numpy.float64), vt.shape)[sel]
    return out


def two_group_rates(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    add: float,
    to: To,
    drop00: bool,
    addyi: bool,
    addvi: bool,
) -> Out:
    x1i, x2i, t1i, t2i = inp.get('x1i'), inp.get('x2i'), inp.get('t1i'), inp.get('t2i')
    if not all_specified(x1i, x2i, t1i, t2i):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., x1i, x2i, t1i, t2i).')
    if not equal_length(x1i, x2i, t1i, t2i):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if x1i is None:
        raise metafopy.errors.InternalError('x1i must not be None')
    if x2i is None:
        raise metafopy.errors.InternalError('x2i must not be None')
    if t1i is None:
        raise metafopy.errors.InternalError('t1i must not be None')
    if t2i is None:
        raise metafopy.errors.InternalError('t2i must not be None')
    k_all = x1i.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    x1i, x2i, t1i, t2i = (req(subset_array(v, sub)) for v in (x1i, x2i, t1i, t2i))

    xs = numpy.concatenate([x1i, x2i])
    if anyna(xs < 0, xs):
        raise metafopy.errors.InputError('One or more counts are negative.')
    ts = numpy.concatenate([t1i, t2i])
    if anyna(ts <= 0, ts):
        raise metafopy.errors.InputError('One or more person-times are <= 0.')

    ni_u = t1i + t2i
    x1i = x1i.copy()
    x2i = x2i.copy()

    if drop00:
        id00 = (x1i == 0) & (x2i == 0)
        x1i[id00] = metafopy.effects.na.NA
        x2i[id00] = metafopy.effects.na.NA

    x1i_u = x1i.copy()
    x2i_u = x2i.copy()

    if to == 'all':
        x1i = x1i + add
        x2i = x2i + add
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to in ('only0', 'if0all'):
        id0 = (x1i == 0) | (x2i == 0)
    if to == 'only0':
        x1i[id0] += add
        x2i[id0] += add
    if to == 'if0all' and id0.any():
        x1i = x1i + add
        x2i = x2i + add

    ir1i_u = x1i_u / t1i
    ir2i_u = x2i_u / t2i
    ir1i = x1i / t1i
    ir2i = x2i / t2i

    if measure == 'IRR':
        yi = numpy.log(ir1i) - numpy.log(ir2i) if addyi else numpy.log(ir1i_u) - numpy.log(ir2i_u)
        vi = 1 / x1i + 1 / x2i if addvi else 1 / x1i_u + 1 / x2i_u
    elif measure == 'IRD':
        yi = ir1i - ir2i if addyi else ir1i_u - ir2i_u
        vi = ir1i / t1i + ir2i / t2i if addvi else ir1i_u / t1i + ir2i_u / t2i
    else:
        yi = numpy.sqrt(ir1i) - numpy.sqrt(ir2i) if addyi else numpy.sqrt(ir1i_u) - numpy.sqrt(ir2i_u)
        vi = 1 / (4 * t1i) + 1 / (4 * t2i)
    return Out(yi, vi, ni_u, k_all, sub)


def two_group_means(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    vtype: str | collections.abc.Sequence[str],
    correct: bool | str,
) -> Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    n1i, n2i = inp.get('n1i'), inp.get('n2i')
    di, ti, pi, ri = inp.get('di'), inp.get('ti'), inp.get('pi'), inp.get('ri')
    mini, maxi = inp.get('mini'), inp.get('maxi')

    if measure in ('SMD', 'RPB', 'ZPB', 'RBIS', 'ZBIS', 'D2OR', 'D2ORN', 'D2ORL'):
        if not equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ri):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if n1i is not None and n2i is not None:
            ti = replmiss(ti, convp2t(pi, n1i + n2i - 2))
            if ti is not None:
                di = replmiss(di, ti * numpy.sqrt(1 / n1i + 1 / n2i))
            mi_ = n1i + n2i - 2
            hi = mi_ / n1i + mi_ / n2i
            if ri is not None:
                di = replmiss(di, numpy.sqrt(hi) * ri / numpy.sqrt(1 - ri**2))
        if di is not None:
            has = ~numpy.isnan(di)
            m1i = fill(m1i, has, di, di.shape[0])
            m2i = fill(m2i, has, 0.0, di.shape[0])
            sd1i = fill(sd1i, has, 1.0, di.shape[0])
            sd2i = fill(sd2i, has, 1.0, di.shape[0])
        if not all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i (and di, ti, pi)).')

    if measure in ('MD', 'SMDH', 'SMD1H', 'ROM', 'CVR'):
        if not all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i).')
        if not equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if measure == 'POMPMD':
        if not all_specified(m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi).')
        if not equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, mini, maxi):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if m1i is None:
            raise metafopy.errors.InternalError('m1i must not be None')
        if m2i is None:
            raise metafopy.errors.InternalError('m2i must not be None')
        if sd1i is None:
            raise metafopy.errors.InternalError('sd1i must not be None')
        if sd2i is None:
            raise metafopy.errors.InternalError('sd2i must not be None')
        if mini is None:
            raise metafopy.errors.InternalError('mini must not be None')
        if maxi is None:
            raise metafopy.errors.InternalError('maxi must not be None')
        m1i = 100 * (m1i - mini) / (maxi - mini)
        m2i = 100 * (m2i - mini) / (maxi - mini)
        sd1i = 100 * sd1i / (maxi - mini)
        sd2i = 100 * sd2i / (maxi - mini)

    if measure == 'VR':
        if not all_specified(sd1i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., sd1i, sd2i, n1i, n2i).')
        if not equal_length(sd1i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if measure == 'SMD1':
        if not all_specified(m1i, m2i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd2i, n1i, n2i).')
        if not equal_length(m1i, m2i, sd2i, n1i, n2i):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if n1i is None:
        raise metafopy.errors.InternalError('n1i must not be None')
    if n2i is None:
        raise metafopy.errors.InternalError('n2i must not be None')
    k_all = n1i.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    m1i, m2i, sd1i, sd2i = (subset_array(v, sub) for v in (m1i, m2i, sd1i, sd2i))
    n1i, n2i = req(subset_array(n1i, sub)), req(subset_array(n2i, sub))

    sds = numpy.concatenate([v for v in (sd1i, sd2i) if v is not None]) if (sd1i is not None or sd2i is not None) else numpy.empty(0)
    if anyna(sds < 0, sds):
        raise metafopy.errors.InputError('One or more standard deviations are negative.')
    ns = numpy.concatenate([n1i, n2i])
    if anyna(ns <= 0, ns):
        raise metafopy.errors.InputError('One or more group sizes are <= 0.')

    ni_u = n1i + n2i
    k = n1i.shape[0]
    ni = ni_u

    nan = numpy.full(k, metafopy.effects.na.NA)
    m1 = nan if m1i is None else m1i
    m2 = nan if m2i is None else m2i
    sd1 = nan if sd1i is None else sd1i
    sd2 = nan if sd2i is None else sd2i

    if measure in ('SMD1', 'SMD1H'):
        mi = n2i - 1
        sdpi = sd2
        npi = n2i
    else:
        mi = ni - 2
        sdpi = numpy.sqrt(((n1i - 1) * sd1**2 + (n2i - 1) * sd2**2) / mi)
        npi = ni

    d = (m1 - m2) / sdpi

    yi = nan
    vi = nan

    if measure in ('MD', 'POMPMD'):
        yi = m1 - m2
        vt = vtypes(vtype, k, ('LS', 'UB', 'HO'), "either 'LS', 'UB', or 'HO'")
        ub = sd1**2 / n1i + sd2**2 / n2i
        vi = pick(vt, {'UB': ub, 'LS': ub, 'HO': sdpi**2 * (1 / n1i + 1 / n2i)})

    if measure in ('SMD', 'SMD1'):
        cmi = cmicalc(mi, method=correct)
        yi = cmi * d
        vt = vtypes(vtype, k, ('LS', 'LS2', 'LS3', 'UB', 'AV', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        mnwyi = wmean(yi, ni)
        vi = pick(
            vt,
            {
                'LS': 1 / n1i + 1 / n2i + yi**2 / (2 * npi),
                'LS2': cmi**2 * (1 / n1i + 1 / n2i + d**2 / (2 * npi)),
                'LS3': 1 / n1i + 1 / n2i + yi**2 / (2 * (npi - 3.94)),
                'UB': 1 / n1i + 1 / n2i + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'AV': 1 / n1i + 1 / n2i + mnwyi**2 / (2 * npi),
                'H0': numpy.where(mi > 2, cmi**2 * (1 / n1i + 1 / n2i) * mi / (mi - 2), metafopy.effects.na.NA),
            },
        )

    if measure == 'SMDH':
        cmi = cmicalc(mi, method=correct)
        sdpi = numpy.sqrt((sd1**2 + sd2**2) / 2)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        vt = vtypes(vtype, k, ('LS', 'LS2', 'LS3'), "either 'LS' or 'LS2'")
        vi = pick(
            vt,
            {
                'LS': yi**2 * (sd1**4 / (n1i - 1) + sd2**4 / (n2i - 1)) / (8 * sdpi**4) + (sd1**2 / (n1i - 1) + sd2**2 / (n2i - 1)) / sdpi**2,
                'LS2': sd1**2 / (n1i * sdpi**2) + sd2**2 / (n2i * sdpi**2) + yi**2 / (8 * sdpi**4) * (sd1**4 / n1i + sd2**4 / n2i),
                'LS3': sd1**2 / (n1i * sdpi**2) + sd2**2 / (n2i * sdpi**2) + yi**2 / (8 * sdpi**4) * (sd1**4 / (n1i - 1) + sd2**4 / (n2i - 1)),
            },
        )

    if measure == 'SMD1H':
        cmi = cmicalc(mi, method=correct)
        yi = cmi * d
        vi = (sd1**2 / sd2**2) / (n1i - 1) + 1 / (n2i - 1) + yi**2 / (2 * (n2i - 1))

    if measure == 'ROM':
        if correct:
            yi = numpy.log(m1 / m2) + 1 / 2 * (sd1**2 / (n1i * m1**2) - sd2**2 / (n2i * m2**2))
        else:
            yi = numpy.log(m1 / m2)
        vt = vtypes(vtype, k, ('LS', 'HO', 'AV', 'AVHO', 'LS2'), "either 'LS', 'LS2', 'HO', 'AV', or 'AVHO'")
        cv1 = sd1 / m1
        cv2 = sd2 / m2
        mn1wcvi = wmean(cv1, n1i)
        mn2wcvi = wmean(cv2, n2i)
        not_na = ~(numpy.isnan(n1i) | numpy.isnan(n2i) | numpy.isnan(cv1) | numpy.isnan(cv2))
        mnwcvi = (numpy.sum(n1i[not_na] * cv1[not_na]) + numpy.sum(n2i[not_na] * cv2[not_na])) / numpy.sum((n1i + n2i)[not_na])
        vi = pick(
            vt,
            {
                'LS': sd1**2 / (n1i * m1**2) + sd2**2 / (n2i * m2**2),
                'HO': sdpi**2 / (n1i * m1**2) + sdpi**2 / (n2i * m2**2),
                'AV': mn1wcvi**2 / n1i + mn2wcvi**2 / n2i,
                'AVHO': mnwcvi**2 * (1 / n1i + 1 / n2i),
                'LS2': sd1**2 / (n1i * m1**2) + sd2**2 / (n2i * m2**2) + sd1**4 / (2 * n1i**2 * m1**4) + sd2**4 / (2 * n2i**2 * m2**4),
            },
        )

    if measure in ('RPB', 'ZPB'):
        hi = mi / n1i + mi / n2i
        yi = d / numpy.sqrt(d**2 + hi)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('ST', 'CS', 'LS'), "either 'ST', 'CS', or 'LS'")
        st = hi**2 / (hi + d**2) ** 3 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))
        cs = (1 - yi**2) ** 2 * (ni * yi**2 / (4 * n1i * n2i) + (2 - 3 * yi**2) / (2 * ni))
        vi = pick(vt, {'ST': st, 'LS': st, 'CS': cs})

    if measure in ('RBIS', 'ZBIS'):
        hi = mi / n1i + mi / n2i
        yi = d / numpy.sqrt(d**2 + hi)
        p1i = n1i / ni
        p2i = n2i / ni
        zi = scipy.stats.norm.isf(p1i)
        fzi = scipy.stats.norm.pdf(zi)
        yi = numpy.sqrt(p1i * p2i) / fzi * yi
        yi_t = numpy.where(numpy.abs(yi) > 1, numpy.sign(yi), yi)
        vi = 1 / (ni - 1) * (p1i * p2i / fzi**2 - (3 / 2 + (1 - p1i * zi / fzi) * (1 + p2i * zi / fzi)) * yi_t**2 + yi_t**4)

    if measure in ('ZPB', 'ZBIS'):
        vi = vi / (1 - numpy.where(yi**2 > 1, 1, yi**2)) ** 2
        yi = rtoz(yi)

    if measure in ('D2OR', 'D2ORL'):
        yi = math.pi / math.sqrt(3) * d
        vi = math.pi**2 / 3 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))

    if measure == 'D2ORN':
        yi = 1.65 * d
        vi = 1.65**2 * (1 / n1i + 1 / n2i + d**2 / (2 * ni))

    if measure == 'VR':
        if correct:
            yi = numpy.log(sd1 / sd2) + 1 / (2 * (n1i - 1)) - 1 / (2 * (n2i - 1))
        else:
            yi = numpy.log(sd1 / sd2)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(vt, {'LS': 1 / (2 * (n1i - 1)) + 1 / (2 * (n2i - 1)), 'LS2': 1 / 2 * (n1i / (n1i - 1) ** 2 + n2i / (n2i - 1) ** 2)})

    if measure == 'CVR':
        if correct:
            yi = (
                numpy.log(sd1 / m1)
                - numpy.log(sd2 / m2)
                + 1 / (2 * (n1i - 1))
                - 1 / (2 * (n2i - 1))
                + 1 / 2 * (sd2**2 / (n2i * m2**2) - sd1**2 / (n1i * m1**2))
            )
        else:
            yi = numpy.log(sd1 / m1) - numpy.log(sd2 / m2)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(
            vt,
            {
                'LS': 1 / (2 * (n1i - 1)) + sd1**2 / (n1i * m1**2) + 1 / (2 * (n2i - 1)) + sd2**2 / (n2i * m2**2),
                'LS2': sd1**2 / (n1i * m1**2)
                + sd2**2 / (n2i * m2**2)
                + sd1**4 / (2 * n1i**2 * m1**4)
                + sd2**4 / (2 * n2i**2 * m2**4)
                + 1 / 2 * (n1i / (n1i - 1) ** 2 + n2i / (n2i - 1) ** 2),
            },
        )

    return Out(yi, vi, ni_u, k_all, sub)


def fill(x: FloatArray | None, where: BoolArray, value: FloatArray | float, k: int) -> FloatArray:
    out = numpy.full(k, metafopy.effects.na.NA) if x is None else x.copy()
    if isinstance(value, numpy.ndarray):
        out[where] = value[where]
    else:
        out[where] = value
    return out


def cles(inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, vtype: str | collections.abc.Sequence[str]) -> Out:
    ai, n1i, n2i, mi = inp.get('ai'), inp.get('n1i'), inp.get('n2i'), inp.get('mi')
    if not all_specified(ai, n1i, n2i):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ai, n1i, n2i).')
    if not equal_length(ai, n1i, n2i, mi):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ai is None:
        raise metafopy.errors.InternalError('ai must not be None')
    if n1i is None:
        raise metafopy.errors.InternalError('n1i must not be None')
    if n2i is None:
        raise metafopy.errors.InternalError('n2i must not be None')
    mi = numpy.zeros(ai.shape[0]) if mi is None else numpy.where(numpy.isnan(mi), 0.0, mi)
    k_all = ai.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    ai, n1i, n2i, mi = (req(subset_array(v, sub)) for v in (ai, n1i, n2i, mi))

    ns = numpy.concatenate([n1i, n2i])
    if anyna(ns <= 0, ns):
        raise metafopy.errors.InputError('One or more group sizes are <= 0.')
    if anyna((ai < 0) | (ai > 1), ai):
        raise metafopy.errors.InputError('One or more AUC values are < 0 or > 1.')
    if anyna((mi < 0) | (mi > 1), mi):
        raise metafopy.errors.InputError("One or more 'mi' values are < 0 or > 1.")

    ni = n1i + n2i
    k = ai.shape[0]
    yi = ai
    vt = vtypes(vtype, k, ('LS', 'LS2', 'H0', 'MAX'), "'LS' or 'LS2'")
    navgi = (n1i + n2i) / 2
    q0 = ai * (1 - ai)
    q1 = ai / (2 - ai)
    q2 = 2 * ai**2 / (1 + ai)
    vi = pick(
        vt,
        {
            'LS': q0 / ((n1i - 1) * (n2i - 1)) * (2 * navgi - 1 - (3 * navgi - 3) / ((2 - ai) * (1 + ai))),
            'LS2': (q0 - mi / 4 + (n1i - 1) * (q1 - ai**2) + (n2i - 1) * (q2 - ai**2)) / ((n1i - 1) * (n2i - 1)),
            'H0': (n1i + n2i + 1) / (12 * n1i * n2i),
            'MAX': q0 / (numpy.minimum(n1i, n2i) - 1),
        },
    )
    return Out(yi, vi, ni, k_all, sub)


def clesn(inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, vtype: str | collections.abc.Sequence[str]) -> Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    n1i, n2i, di, ti, pi, ai = inp.get('n1i'), inp.get('n2i'), inp.get('di'), inp.get('ti'), inp.get('pi'), inp.get('ai')
    if not equal_length(m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ai):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if not all_specified(n1i, n2i):
        raise metafopy.errors.InputError(f'{INCOMPLETE}.')
    if n1i is None:
        raise metafopy.errors.InternalError('n1i must not be None')
    if n2i is None:
        raise metafopy.errors.InternalError('n2i must not be None')
    k_all = max(v.shape[0] for v in (m1i, m2i, sd1i, sd2i, n1i, n2i, di, ti, pi, ai) if v is not None)
    vt_list = [vtype] if isinstance(vtype, str) else list(vtype)
    vt = numpy.array(vt_list * k_all if len(vt_list) == 1 else vt_list, dtype=object)

    if sd1i is None or sd2i is None:
        sd1i = numpy.full(k_all, metafopy.effects.na.NA)
        sd2i = numpy.full(k_all, metafopy.effects.na.NA)

    ti = replmiss(ti, convp2t(pi, n1i + n2i - 2))
    if ti is not None:
        di = replmiss(di, ti * numpy.sqrt(1 / n1i + 1 / n2i))
    if di is not None:
        vt[~numpy.isnan(di)] = 'HO'

    ho = vt == 'HO'
    sdpi = numpy.where(ho, numpy.sqrt(((n1i - 1) * sd1i**2 + (n2i - 1) * sd2i**2) / (n1i + n2i - 2)), numpy.sqrt((sd1i**2 + sd2i**2) / 2))
    if m1i is not None and m2i is not None:
        di = replmiss(di, (m1i - m2i) / sdpi)
    if di is not None:
        ai = replmiss(ai, scipy.stats.norm.cdf(di / math.sqrt(2)))
    if ai is not None:
        di = replmiss(di, scipy.special.ndtri(ai) * math.sqrt(2))
    if ai is None or di is None:
        raise metafopy.errors.InputError(f'{INCOMPLETE}.')

    k_all = ai.shape[0]
    sdsmiss = numpy.isnan(sd1i) | numpy.isnan(sd2i)
    sd1i = numpy.where(sdsmiss, 1.0, sd1i)
    sd2i = numpy.where(sdsmiss, 1.0, sd2i)
    vt[sdsmiss] = 'HO'

    sub = chksubset(subset, k_all, data, 'subset')
    if sub is not None:
        vt = vt[sub]
    ai, di, sd1i, sd2i, n1i, n2i = (req(subset_array(v, sub)) for v in (ai, di, sd1i, sd2i, n1i, n2i))

    sds = numpy.concatenate([sd1i, sd2i])
    if anyna(sds < 0, sds):
        raise metafopy.errors.InputError('One or more standard deviations are negative.')
    ns = numpy.concatenate([n1i, n2i])
    if anyna(ns <= 0, ns):
        raise metafopy.errors.InputError('One or more group sizes are <= 0.')
    if anyna((ai < 0) | (ai > 1), ai):
        raise metafopy.errors.InputError('One or more AUC values are < 0 or > 1.')

    ni_u = n1i + n2i
    k = ai.shape[0]
    yi = ai
    vts = vtypes([str(v) for v in vt], k, ('LS', 'LS2', 'LS3', 'HO', 'H0'), "'LS' or 'HO'")
    vri = sd1i**2 / (sd1i**2 + sd2i**2)
    e = numpy.exp(-(di**2) / 2)
    vi = pick(
        vts,
        {
            'LS': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 1) + di**2 * (1 - vri) ** 2 / (n2i - 1) + 4 * vri / (n1i - 1) + 4 * (1 - vri) / (n2i - 1)),
            'LS2': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 0) + di**2 * (1 - vri) ** 2 / (n2i - 0) + 4 * vri / (n1i - 0) + 4 * (1 - vri) / (n2i - 0)),
            'LS3': e / (8 * math.pi) * (di**2 * vri**2 / (n1i - 1) + di**2 * (1 - vri) ** 2 / (n2i - 1) + 4 * vri / (n1i - 0) + 4 * (1 - vri) / (n2i - 0)),
            'HO': e / (4 * math.pi) * (1 / n1i + 1 / n2i + di**2 / (2 * (n1i + n2i))),
            'H0': 1 / (8 * math.pi) * (4 * vri / (n1i - 1) + 4 * (1 - vri) / (n2i - 1)),
        },
    )
    return Out(yi, vi, ni_u, k_all, sub)


def cor(
    measure: str, inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, vtype: str | collections.abc.Sequence[str]
) -> Out:
    ri, ni, ti, pi = inp.get('ri'), inp.get('ni'), inp.get('ti'), inp.get('pi')
    if not equal_length(ri, ni, ti, pi):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ni is not None:
        ti = replmiss(ti, convp2t(pi, ni - 2))
        if ti is not None:
            ri = replmiss(ri, ti / numpy.sqrt(ti**2 + ni - 2))
    if not all_specified(ri, ni):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ri, ni (and ti, pi)).')
    if ri is None:
        raise metafopy.errors.InternalError('ri must not be None')
    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    k_all = ri.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    ri, ni = req(subset_array(ri, sub)), req(subset_array(ni, sub))

    if anyna(numpy.abs(ri) > 1, ri):
        raise metafopy.errors.InputError('One or more correlations are > 1 or < -1.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    vtl = [vtype] if isinstance(vtype, str) else list(vtype)
    if measure != 'UCOR' and 'UB' in vtl:
        raise metafopy.errors.InputError("Use of vtype='UB' only permitted when measure='UCOR'.")
    if measure == 'UCOR' and anyna(ni <= 4, ni):
        warnings.warn('Cannot compute the bias-corrected correlation coefficient when ni <= 4.', stacklevel=4)
    if measure == 'ZCOR' and anyna(ni <= 3, ni):
        warnings.warn('Cannot estimate the sampling variance when ni <= 3.', stacklevel=4)

    k = ri.shape[0]
    yi = ri
    if measure == 'UCOR':
        yi = ri * fcalc(1 / 2, 1 / 2, (ni - 2) / 2, 1 - ri**2)

    if measure in ('COR', 'UCOR'):
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        allowed = ('LS', 'UB', 'AV', 'H0', 'H0a', 'H0b') if measure == 'COR' else ('LS', 'UB', 'AV')
        msg = "either 'LS', 'UB', 'AV', or 'H0'" if measure == 'COR' else "either 'LS', 'UB', or 'AV'"
        vt = vtypes(vtype, k, allowed, msg)
        mnwyi = wmean(yi, ni)
        vi = pick(
            vt,
            {
                'LS': (1 - yi**2) ** 2 / (ni - 1),
                'UB': yi**2 - (1 - (ni - 3) / (ni - 2) * (1 - ri**2) * fcalc(1, 1, ni / 2, 1 - ri**2)),
                'AV': (1 - mnwyi**2) ** 2 / (ni - 1),
                'H0': (1 - yi**2) / (ni - 2),
                'H0a': (1 - yi**2) / (ni - 2),
                'H0b': 1 / ni,
            },
        )
    else:
        yi = rtoz(ri)
        vi = 1 / (ni - 3)

    vi = numpy.where(ni <= 3, metafopy.effects.na.NA, vi)
    return Out(yi, vi, ni, k_all, sub)


def pcor(
    measure: str, inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, vtype: str | collections.abc.Sequence[str]
) -> Out:
    ri, ti, mi, ni, pi, r2i = inp.get('ri'), inp.get('ti'), inp.get('mi'), inp.get('ni'), inp.get('pi'), inp.get('r2i')
    if not equal_length(ri, ti, mi, ni, pi, r2i):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ni is not None and mi is not None:
        ti = replmiss(ti, convp2t(pi, ni - mi - 1))
        if ti is not None:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            if measure in ('PCOR', 'ZPCOR'):
                ri = replmiss(ri, ti / numpy.sqrt(ti**2 + ni - mi - 1))
            elif r2i is not None:
                ri = replmiss(ri, ti * numpy.sqrt(1 - r2i) / numpy.sqrt(ni - mi - 1))
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('PCOR', 'ZPCOR') and not all_specified(ri, mi, ni):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ri, ti, mi, ni (and pi)).')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('SPCOR', 'ZSPCOR') and not all_specified(ri, mi, ni, r2i):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ri, ti, mi, ni, r2i (and pi)).')
    if ri is None:
        raise metafopy.errors.InternalError('ri must not be None')
    if mi is None:
        raise metafopy.errors.InternalError('mi must not be None')
    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    k_all = ri.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    ri, mi, ni = req(subset_array(ri, sub)), req(subset_array(mi, sub)), req(subset_array(ni, sub))
    r2i = subset_array(r2i, sub)

    if anyna(numpy.abs(ri) > 1, ri):
        raise metafopy.errors.InputError('One or more (semi-)partial correlations are > 1 or < -1.')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('SPCOR', 'ZSPCOR') and r2i is not None and anyna((r2i > 1) | (r2i < 0), r2i):
        raise metafopy.errors.InputError('One or more R^2 values are > 1 or < 0.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    if anyna(mi <= 0, mi):
        raise metafopy.errors.InputError('One or more mi values are <= 0.')
    if anyna(ni - mi - 1 <= 0, ni, mi):
        raise metafopy.errors.InputError('One or more dfs are <= 0.')

    k = ri.shape[0]
    if measure == 'PCOR':
        yi = ri
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        mnwyi = wmean(yi, ni)
        vi = pick(vt, {'LS': (1 - yi**2) ** 2 / (ni - mi), 'AV': (1 - mnwyi**2) ** 2 / (ni - mi)})
    elif measure == 'ZPCOR':
        yi = rtoz(ri)
        vi = 1 / (ni - mi - 2)
    else:
        if r2i is None:
            raise metafopy.errors.InternalError('r2i must not be None')
        yi = ri
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        mnwyi = wmean(yi, ni)
        vi = pick(
            vt,
            {
                'LS': (r2i**2 - 2 * r2i + (r2i - yi**2) + 1 - (r2i - yi**2) ** 2) / ni,
                'AV': (r2i**2 - 2 * r2i + (r2i - mnwyi**2) + 1 - (r2i - mnwyi**2) ** 2) / ni,
            },
        )
        if measure == 'ZSPCOR':
            vi = vi / (1 - numpy.where(yi**2 > 1, 1, yi**2)) ** 2
            yi = rtoz(yi)
    return Out(yi, vi, ni, k_all, sub)


def r2(
    measure: str, inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, vtype: str | collections.abc.Sequence[str]
) -> Out:
    r2i, mi, ni, fi, pi = inp.get('r2i'), inp.get('mi'), inp.get('ni'), inp.get('fi'), inp.get('pi')
    if not equal_length(r2i, mi, ni):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if mi is not None and ni is not None:
        fi = replmiss(fi, convp2f(pi, mi, ni - mi - 1))
        if fi is not None:
            r2i = replmiss(r2i, mi * fi / (mi * fi + (ni - mi - 1)))
    if not all_specified(r2i, mi, ni):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., r2i, mi, ni (and fi, pi)).')
    if r2i is None:
        raise metafopy.errors.InternalError('r2i must not be None')
    if mi is None:
        raise metafopy.errors.InternalError('mi must not be None')
    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    k_all = r2i.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    r2i, mi, ni = req(subset_array(r2i, sub)), req(subset_array(mi, sub)), req(subset_array(ni, sub))

    if anyna((r2i > 1) | (r2i < 0), r2i):
        raise metafopy.errors.InputError('One or more R^2 values are > 1 or < 0.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    if anyna(mi <= 0, mi):
        raise metafopy.errors.InputError('One or more mi values are <= 0.')
    if anyna(ni - mi - 1 <= 0, ni, mi):
        raise metafopy.errors.InputError('One or more dfs are <= 0.')

    k = r2i.shape[0]
    if measure == 'R2':
        yi = r2i
        vt = vtypes(vtype, k, ('LS', 'AV', 'LS2', 'AV2'), "either 'LS' or 'AV'")
        mnwyi = wmean(yi, ni)
        vi = pick(
            vt,
            {
                'LS': 4 * yi * (1 - yi) ** 2 / ni,
                'AV': 4 * mnwyi * (1 - mnwyi) ** 2 / ni,
                'LS2': 4 * yi * (1 - yi) ** 2 * (ni - mi - 1) ** 2 / ((ni**2 - 1) * (ni + 3)),
                'AV2': 4 * mnwyi * (1 - mnwyi) ** 2 * (ni - mi - 1) ** 2 / ((ni**2 - 1) * (ni + 3)),
            },
        )
    elif measure == 'ZR2':
        vtl = [vtype] if isinstance(vtype, str) else list(vtype)
        if not all(v == 'LS' for v in vtl):
            raise metafopy.errors.InputError("For this outcome measure, 'vtype' must be 'LS'.")
        yi = rtoz(numpy.sqrt(r2i))
        vi = 1 / ni
    else:
        yi = r2i
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vtypes(vtype, k, ('LS',), "'LS'")
        vi = numpy.full(k, metafopy.effects.na.NA)
        R2s = numpy.linspace(0.0001, 0.9999, 10**5)
        dx = R2s[1] - R2s[0]
        for i in range(k):
            if numpy.isnan(r2i[i]) or numpy.isnan(mi[i]) or numpy.isnan(ni[i]):
                continue
            Fval = (ni[i] - mi[i] - 1) / mi[i] * r2i[i] / (1 - r2i[i])
            ncps = (mi[i] + ni[i] - mi[i]) * R2s / (1 - R2s)
            den = scipy.stats.ncf.pdf(Fval, mi[i], ni[i] - mi[i] - 1, ncps)
            den = den / (numpy.sum(numpy.diff(R2s) * (den[1:] + den[:-1])) / 2)
            vi[i] = numpy.sum(dx * R2s**2 * den) - numpy.sum(dx * R2s * den) ** 2
        if measure == 'ZR2F':
            vi = vi / (1 - yi**2) ** 2
            yi = rtoz(numpy.sqrt(r2i))
    return Out(yi, vi, ni, k_all, sub)


def proportions(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    add: float,
    to: To,
    vtype: str | collections.abc.Sequence[str],
    addyi: bool,
    addvi: bool,
) -> Out:
    xi, mi, ni = inp.get('xi'), inp.get('mi'), inp.get('ni')
    if not equal_length(xi, mi, ni):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ni is not None and xi is not None and mi is not None and anyna(ni != xi + mi, ni, xi, mi):
        raise metafopy.errors.InputError("One or more 'ni' values are not equal to 'xi + mi'.")
    if ni is not None and xi is not None:
        mi = replmiss(mi, ni - xi)
    if not all_specified(xi, mi):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., xi, mi or xi, ni).')
    if xi is None:
        raise metafopy.errors.InternalError('xi must not be None')
    if mi is None:
        raise metafopy.errors.InternalError('mi must not be None')
    k_all = xi.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    xi, mi = req(subset_array(xi, sub)).copy(), req(subset_array(mi, sub)).copy()

    ni = xi + mi
    if anyna(xi > ni, xi, ni):
        raise metafopy.errors.InputError('One or more event counts are larger than the corresponding group sizes.')
    c = numpy.concatenate([xi, mi])
    if anyna(c < 0, c):
        raise metafopy.errors.InputError('One or more counts are negative.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more group sizes are <= 0.')

    ni_u = ni
    k = xi.shape[0]
    xi_u = xi.copy()
    mi_u = mi.copy()

    if to == 'all':
        xi = xi + add
        mi = mi + add
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to in ('only0', 'if0all'):
        id0 = (xi == 0) | (mi == 0)
    if to == 'only0':
        xi[id0] += add
        mi[id0] += add
    if to == 'if0all' and id0.any():
        xi = xi + add
        mi = mi + add

    ni = xi + mi
    pri_u = xi_u / ni_u
    pri = xi / ni

    if measure in ('PR', 'PLN', 'PLO'):
        if addvi:
            mnwpri = wmean(pri, ni)
        else:
            mnwpri_u = wmean(pri_u, ni_u)

    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    if measure == 'PR':
        yi = pri if addyi else pri_u
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'UB', 'AV'), "either 'LS', 'UB', or 'AV'")
        if addvi:
            vi = pick(vt, {'LS': pri * (1 - pri) / ni, 'UB': pri * (1 - pri) / (ni - 1), 'AV': mnwpri * (1 - mnwpri) / ni})
        else:
            vi = pick(vt, {'LS': pri_u * (1 - pri_u) / ni_u, 'UB': pri_u * (1 - pri_u) / (ni_u - 1), 'AV': mnwpri_u * (1 - mnwpri_u) / ni_u})
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif measure == 'PLN':
        yi = numpy.log(pri) if addyi else numpy.log(pri_u)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        if addvi:
            vi = pick(vt, {'LS': 1 / xi - 1 / ni, 'AV': 1 / (mnwpri * ni) - 1 / ni})
        else:
            vi = pick(vt, {'LS': 1 / xi_u - 1 / ni_u, 'AV': 1 / (mnwpri_u * ni_u) - 1 / ni_u})
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif measure == 'PLO':
        yi = numpy.log(pri / (1 - pri)) if addyi else numpy.log(pri_u / (1 - pri_u))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'AV'), "either 'LS' or 'AV'")
        if addvi:
            vi = pick(vt, {'LS': 1 / xi + 1 / mi, 'AV': 1 / (mnwpri * ni) + 1 / ((1 - mnwpri) * ni)})
        else:
            vi = pick(vt, {'LS': 1 / xi_u + 1 / mi_u, 'AV': 1 / (mnwpri_u * ni_u) + 1 / ((1 - mnwpri_u) * ni_u)})
    elif measure == 'PRZ':
        yi = scipy.special.ndtri(pri)
        vi = 2 * math.pi * pri * (1 - pri) * numpy.exp(yi**2) / ni
    elif measure == 'PAS':
        yi = numpy.arcsin(numpy.sqrt(pri))
        vi = 1 / (4 * ni)
    else:
        yi = 1 / 2 * (numpy.arcsin(numpy.sqrt(xi / (ni + 1))) + numpy.arcsin(numpy.sqrt((xi + 1) / (ni + 1))))
        vi = 1 / (4 * ni + 2)
    return Out(yi, vi, ni_u, k_all, sub)


def rates(
    measure: str, inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None, add: float, to: To, addyi: bool, addvi: bool
) -> Out:
    xi, ti = inp.get('xi'), inp.get('ti')
    if not all_specified(xi, ti):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., xi, ti).')
    if not equal_length(xi, ti):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if xi is None:
        raise metafopy.errors.InternalError('xi must not be None')
    if ti is None:
        raise metafopy.errors.InternalError('ti must not be None')
    k_all = xi.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    xi, ti = req(subset_array(xi, sub)).copy(), req(subset_array(ti, sub))
    if anyna(xi < 0, xi):
        raise metafopy.errors.InputError('One or more counts are negative.')
    if anyna(ti <= 0, ti):
        raise metafopy.errors.InputError('One or more person-times are <= 0.')

    ni_u = ti
    xi_u = xi.copy()
    if to == 'all':
        xi = xi + add
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to in ('only0', 'if0all'):
        id0 = xi == 0
    if to == 'only0':
        xi[id0] += add
    if to == 'if0all' and id0.any():
        xi = xi + add

    iri_u = xi_u / ti
    iri = xi / ti
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    if measure == 'IR':
        yi = iri if addyi else iri_u
        vi = iri / ti if addvi else iri_u / ti
    elif measure == 'IRLN':
        yi = numpy.log(iri) if addyi else numpy.log(iri_u)
        vi = 1 / xi if addvi else 1 / xi_u
    elif measure == 'IRS':
        yi = numpy.sqrt(iri) if addyi else numpy.sqrt(iri_u)
        vi = 1 / (4 * ti)
    else:
        yi = 1 / 2 * (numpy.sqrt(iri) + numpy.sqrt(iri + 1 / ti))
        vi = 1 / (4 * ti)
    return Out(yi, vi, ni_u, k_all, sub)


def means(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    vtype: str | collections.abc.Sequence[str],
    correct: bool | str,
) -> Out:
    mi, sdi, ni, mini, maxi = inp.get('mi'), inp.get('sdi'), inp.get('ni'), inp.get('mini'), inp.get('maxi')
    if measure in ('MN', 'SMN', 'MNLN', 'CVLN'):
        if not all_specified(mi, sdi, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., mi, sdi, ni).')
        if not equal_length(mi, sdi, ni):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if measure == 'POMPMN':
        if not all_specified(mi, sdi, ni, mini, maxi):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., mi, sdi, ni, mini, maxi).')
        if not equal_length(mi, sdi, ni, mini, maxi):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if mi is None:
            raise metafopy.errors.InternalError('mi must not be None')
        if sdi is None:
            raise metafopy.errors.InternalError('sdi must not be None')
        if mini is None:
            raise metafopy.errors.InternalError('mini must not be None')
        if maxi is None:
            raise metafopy.errors.InternalError('maxi must not be None')
        mi = 100 * (mi - mini) / (maxi - mini)
        sdi = 100 * sdi / (maxi - mini)
    if measure == 'SDLN':
        if not all_specified(sdi, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., sdi, ni).')
        if not equal_length(sdi, ni):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if sdi is None:
        raise metafopy.errors.InternalError('sdi must not be None')
    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    k_all = ni.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    mi = subset_array(mi, sub)
    sdi, ni = req(subset_array(sdi, sub)), req(subset_array(ni, sub))

    if anyna(sdi < 0, sdi):
        raise metafopy.errors.InputError('One or more standard deviations are negative.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    if measure in ('MNLN', 'CVLN') and mi is not None and anyna(mi < 0, mi):
        raise metafopy.errors.InputError('One or more means are negative.')

    k = ni.shape[0]
    m = numpy.full(k, metafopy.effects.na.NA) if mi is None else mi
    if measure in ('MN', 'POMPMN'):
        yi = m
        sdpi = math.sqrt(wmean(sdi**2, ni - 1))
        vt = vtypes(vtype, k, ('LS', 'HO'), "either 'LS' or 'HO'")
        vi = pick(vt, {'LS': sdi**2 / ni, 'HO': sdpi**2 / ni})
    # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
    elif measure == 'SMN':
        cmi = cmicalc(ni - 1, method=correct)
        yi = cmi * m / sdi
        vi = 1 / ni + yi**2 / (2 * ni)
    elif measure == 'MNLN':
        yi = numpy.log(m)
        vi = sdi**2 / (ni * m**2)
    elif measure == 'SDLN':
        yi = numpy.log(sdi) + 1 / (2 * (ni - 1)) if correct else numpy.log(sdi)
        vi = 1 / (2 * (ni - 1))
    else:
        yi = numpy.log(sdi / m) + 1 / (2 * (ni - 1)) if correct else numpy.log(sdi / m)
        vi = 1 / (2 * (ni - 1)) + sdi**2 / (ni * m**2)
    return Out(yi, vi, ni, k_all, sub)


def change(
    measure: str,
    inp: Inputs,
    data: pandas.DataFrame | None,
    subset: numpy.typing.ArrayLike | str | None,
    vtype: str | collections.abc.Sequence[str],
    correct: bool | str,
) -> Out:
    m1i, m2i, sd1i, sd2i = inp.get('m1i'), inp.get('m2i'), inp.get('sd1i'), inp.get('sd2i')
    ri, ni, di, ti, pi = inp.get('ri'), inp.get('ni'), inp.get('di'), inp.get('ti'), inp.get('pi')
    mini, maxi = inp.get('mini'), inp.get('maxi')

    kmax = max((v.shape[0] for v in (m1i, m2i, sd1i, sd2i, ni, di, ti, pi) if v is not None), default=0)
    ri = expand(ri, kmax)

    if measure in ('MC', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'CVRC'):
        if not all_specified(m1i, m2i, sd1i, sd2i, ri, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, ri).')
        if not equal_length(m1i, m2i, sd1i, sd2i, ri, ni):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if measure == 'POMPMC':
        if not all_specified(m1i, m2i, sd1i, sd2i, ni, ri, mini, maxi):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, mini, maxi).')
        if not equal_length(m1i, m2i, sd1i, sd2i, ni, ri, mini, maxi):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if m1i is None:
            raise metafopy.errors.InternalError('m1i must not be None')
        if m2i is None:
            raise metafopy.errors.InternalError('m2i must not be None')
        if sd1i is None:
            raise metafopy.errors.InternalError('sd1i must not be None')
        if sd2i is None:
            raise metafopy.errors.InternalError('sd2i must not be None')
        if mini is None:
            raise metafopy.errors.InternalError('mini must not be None')
        if maxi is None:
            raise metafopy.errors.InternalError('maxi must not be None')
        m1i = 100 * (m1i - mini) / (maxi - mini)
        m2i = 100 * (m2i - mini) / (maxi - mini)
        sd1i = 100 * sd1i / (maxi - mini)
        sd2i = 100 * sd2i / (maxi - mini)

    if measure == 'SMCC':
        if not equal_length(m1i, m2i, sd1i, sd2i, ri, ni, di, ti, pi):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if ni is not None:
            ti = replmiss(ti, convp2t(pi, ni - 1))
            if ti is not None:
                di = replmiss(di, ti * numpy.sqrt(1 / ni))
        if di is not None:
            has = ~numpy.isnan(di)
            n = di.shape[0]
            m1i = fill(m1i, has, di, n)
            m2i = fill(m2i, has, 0.0, n)
            sd1i = fill(sd1i, has, 1.0, n)
            sd2i = fill(sd2i, has, 1.0, n)
            ri = fill(ri, has, 0.5, n)
        if not all_specified(m1i, m2i, sd1i, sd2i, ri, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, sd2i, ni, ri (and di, ti, pi)).')

    if measure == 'SMCR':
        if not all_specified(m1i, m2i, sd1i, ri, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., m1i, m2i, sd1i, ni, ri).')
        if not equal_length(m1i, m2i, sd1i, ri, ni):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if measure == 'VRC':
        if not all_specified(sd1i, sd2i, ri, ni):
            raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., sd1i, sd2i, ni, ri).')
        if not equal_length(sd1i, sd2i, ri, ni):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')

    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    if ri is None:
        raise metafopy.errors.InternalError('ri must not be None')
    k_all = ni.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    m1i, m2i, sd1i, sd2i = (subset_array(v, sub) for v in (m1i, m2i, sd1i, sd2i))
    ni, ri = req(subset_array(ni, sub)), req(subset_array(ri, sub))

    if measure in ('MC', 'POMPMC', 'SMCC', 'SMCRH', 'SMCRP', 'SMCRPH', 'CLESCN', 'AUCCN', 'ROMC', 'VRC', 'CVRC'):
        sds = numpy.concatenate([v for v in (sd1i, sd2i) if v is not None]) if (sd1i is not None or sd2i is not None) else numpy.empty(0)
        if anyna(sds < 0, sds):
            raise metafopy.errors.InputError('One or more standard deviations are negative.')
    if measure == 'SMCR' and sd1i is not None and anyna(sd1i < 0, sd1i):
        raise metafopy.errors.InputError('One or more standard deviations are negative.')
    if anyna(numpy.abs(ri) > 1, ri):
        raise metafopy.errors.InputError('One or more correlations are > 1 or < -1.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')

    k = ni.shape[0]
    mi = ni - 1
    nan = numpy.full(k, metafopy.effects.na.NA)
    m1 = nan if m1i is None else m1i
    m2 = nan if m2i is None else m2i
    sd1 = nan if sd1i is None else sd1i
    sd2 = nan if sd2i is None else sd2i

    sddiffi = numpy.sqrt(sd1**2 + sd2**2 - 2 * ri * sd1 * sd2)
    sdpi = numpy.sqrt((sd1**2 + sd2**2) / 2)

    yi = nan
    vi = nan

    if measure in ('MC', 'POMPMC'):
        yi = m1 - m2
        vi = sddiffi**2 / ni

    if measure == 'SMCC':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sddiffi
        yi = cmi * d
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2', 'UB', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        vi = pick(
            vt,
            {
                'LS': 1 / ni + yi**2 / (2 * ni),
                'LS2': cmi**2 * (1 / ni + d**2 / (2 * ni)),
                'UB': 1 / ni + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'H0': numpy.where(mi > 2, cmi**2 / ni * mi / (mi - 2), metafopy.effects.na.NA),
            },
        )

    if measure == 'SMCR':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sd1
        yi = cmi * d
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2', 'UB', 'H0'), "either 'LS', 'LS2', 'UB', or 'H0'")
        rui = ri * fcalc(1 / 2, 1 / 2, (ni - 2) / 2, 1 - ri**2)
        vi = pick(
            vt,
            {
                'LS': 2 * (1 - ri) / ni + yi**2 / (2 * ni),
                'LS2': cmi**2 * (2 * (1 - ri) / ni + d**2 / (2 * ni)),
                'UB': 2 * (1 - rui) / ni + (1 - (mi - 2) / (mi * cmi**2)) * yi**2,
                'H0': numpy.where(mi > 2, cmi**2 * 2 * (1 - rui) / ni * mi / (mi - 2), metafopy.effects.na.NA),
            },
        )

    if measure == 'SMCRH':
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sd1
        yi = cmi * d
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(vt, {'LS': sddiffi**2 / (sd1**2 * (ni - 1)) + yi**2 / (2 * (ni - 1)), 'LS2': sddiffi**2 / (sd1**2 * ni) + yi**2 / (2 * ni)})

    if measure == 'SMCRP':
        mi = 2 * (ni - 1) / (1 + ri**2)
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS',), "'LS'")
        vi = pick(vt, {'LS': 2 * (1 - ri) / ni + yi**2 * (1 + ri**2) / (4 * ni)})

    if measure == 'SMCRPH':
        mi = 2 * (ni - 1) / (1 + ri**2)
        cmi = cmicalc(mi, method=correct)
        d = (m1 - m2) / sdpi
        yi = cmi * d
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "'LS' or 'LS2'")
        num = sd1**4 + sd2**4 + 2 * ri**2 * sd1**2 * sd2**2
        vi = pick(
            vt,
            {
                'LS': sddiffi**2 / (sdpi**2 * (ni - 1)) + yi**2 * num / (8 * sdpi**4 * (ni - 1)),
                'LS2': sddiffi**2 / (sdpi**2 * ni) + yi**2 * num / (8 * sdpi**4 * ni),
            },
        )

    if measure in ('CLESCN', 'AUCCN'):
        d = (m1 - m2) / sdpi
        yi = scipy.stats.norm.cdf(d / math.sqrt(2))
        vi = (
            numpy.exp(-(d**2) / 2)
            / (4 * math.pi)
            * (sddiffi**2 / (sdpi**2 * (ni - 1)) + d**2 * (sd1**4 + sd2**4 + 2 * ri**2 * sd1**2 * sd2**2) / (8 * sdpi**4 * (ni - 1)))
        )

    ls_romc = sd1**2 / (ni * m1**2) + sd2**2 / (ni * m2**2) - 2 * ri * sd1 * sd2 / (m1 * m2 * ni)
    ls2_romc = ls_romc + sd1**4 / (2 * ni**2 * m1**4) + sd2**4 / (2 * ni**2 * m2**4) - ri**2 * sd1**2 * sd2**2 / (ni**2 * m1**2 * m2**2)

    if measure == 'ROMC':
        if correct:
            yi = numpy.log(m1 / m2) + 1 / 2 * (sd1**2 / (ni * m1**2) - sd2**2 / (ni * m2**2))
        else:
            yi = numpy.log(m1 / m2)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(vt, {'LS': ls_romc, 'LS2': ls2_romc})

    if measure == 'VRC':
        yi = numpy.log(sd1 / sd2)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(vt, {'LS': (1 - ri**2) / (ni - 1), 'LS2': (1 - ri**2) / (ni - 1) + (1 - ri**4) / (ni - 1) ** 2})

    if measure == 'CVRC':
        if correct:
            yi = numpy.log(sd1 / m1) - numpy.log(sd2 / m2) + 1 / 2 * (sd2**2 / (ni * m2**2) - sd1**2 / (ni * m1**2))
        else:
            yi = numpy.log(sd1 / m1) - numpy.log(sd2 / m2)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        vt = vtypes(vtype, k, ('LS', 'LS2'), "either 'LS' or 'LS2'")
        vi = pick(vt, {'LS': ls_romc + (1 - ri**2) / (ni - 1), 'LS2': ls2_romc + (1 - ri**2) / (ni - 1) + (1 - ri**4) / (ni - 1) ** 2})

    return Out(yi, vi, ni, k_all, sub)


def alpha(measure: str, inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None) -> Out:
    ai, mi, ni = inp.get('ai'), inp.get('mi'), inp.get('ni')
    if not all_specified(ai, mi, ni):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ai, mi, ni).')
    if not equal_length(ai, mi, ni):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ai is None:
        raise metafopy.errors.InternalError('ai must not be None')
    if mi is None:
        raise metafopy.errors.InternalError('mi must not be None')
    if ni is None:
        raise metafopy.errors.InternalError('ni must not be None')
    k_all = ai.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    ai, mi, ni = (req(subset_array(v, sub)) for v in (ai, mi, ni))
    if anyna(ai > 1, ai):
        raise metafopy.errors.InputError('One or more alpha values are > 1.')
    if anyna(mi < 2, mi):
        raise metafopy.errors.InputError('One or more mi values are < 2.')
    if anyna(ni <= 0, ni):
        raise metafopy.errors.InputError('One or more sample sizes are <= 0.')
    if measure == 'ARAW':
        yi = ai
        vi = 2 * mi * (1 - ai) ** 2 / ((mi - 1) * (ni - 2))
    elif measure == 'AHW':
        yi = 1 - (1 - ai) ** (1 / 3)
        vi = 18 * mi * (ni - 1) * (1 - ai) ** (2 / 3) / ((mi - 1) * (9 * ni - 11) ** 2)
    else:
        yi = -numpy.log(1 - ai)
        vi = 2 * mi / ((mi - 1) * (ni - 2))
    return Out(yi, vi, ni, k_all, sub)


def reh(inp: Inputs, data: pandas.DataFrame | None, subset: numpy.typing.ArrayLike | str | None) -> Out:
    ai, bi, ci = inp.get('ai'), inp.get('bi'), inp.get('ci')
    if not all_specified(ai, bi, ci):
        raise metafopy.errors.InputError(f'{INCOMPLETE} (i.e., ai, bi, ci).')
    if not equal_length(ai, bi, ci):
        raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
    if ai is None:
        raise metafopy.errors.InternalError('ai must not be None')
    if bi is None:
        raise metafopy.errors.InternalError('bi must not be None')
    if ci is None:
        raise metafopy.errors.InternalError('ci must not be None')
    k_all = ai.shape[0]
    sub = chksubset(subset, k_all, data, 'subset')
    ai, bi, ci = (req(subset_array(v, sub)) for v in (ai, bi, ci))
    if anyna(ai < 0, ai) or anyna(bi < 0, bi) or anyna(ci < 0, ci):
        raise metafopy.errors.InputError('One or more group sizes are negative.')
    ni = ai + bi + ci
    p0i = ai / ni
    p1i = bi / ni
    p2i = ci / ni
    yi = numpy.log(p1i) - numpy.log(2 * numpy.sqrt(p0i * p2i))
    vi = ((1 - p1i) / (4 * p0i * p2i) + 1 / p1i) / ni
    return Out(yi, vi, ni, k_all, sub)
