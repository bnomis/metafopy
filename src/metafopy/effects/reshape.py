"""Restructure summary data: metafor's ``to.long``, ``to.table`` and ``to.wide``.

R sources: https://github.com/wviechtb/metafor/blob/master/R/to.long.r, ``R/to.table.r`` and
``R/to.wide.r``. Documentation: https://wviechtb.github.io/metafor/reference/to.long.html,
https://wviechtb.github.io/metafor/reference/to.table.html and
https://wviechtb.github.io/metafor/reference/to.wide.html.

Differences from R that follow Python conventions:

- the data arguments are array-likes or column names in ``data``; numeric ``subset`` values and
  ``append`` positions are 0-based;
- R's ``options(na.action=)`` is the ``na_action`` argument (``'omit'``, ``'exclude'``, ``'fail'`` or
  ``'pass'``);
- the factors of the long format are :class:`pandas.Categorical` columns of strings (``study`` with the
  sorted labels as categories, ``group``/``time``/``outcome`` with the categories ``'2'``, ``'1'``);
- ``to_table`` returns a :class:`TableArray`, whose ``values`` have R's dimensions (rows, columns, studies).
"""

import collections.abc
import dataclasses
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.effects.datatools
import metafopy.effects.measures
import metafopy.errors
import metafopy.metadata

__all__ = ['TableArray', 'to_long', 'to_table', 'to_wide']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Arg = numpy.typing.ArrayLike | str | None
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type NAAction = typing.Literal['omit', 'exclude', 'fail', 'pass']

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
KNOWN = (
    'RR', 'OR', 'PETO', 'RD', 'AS', 'PHI', 'YUQ', 'YUY', 'RTET', 'PBIT', 'OR2D', 'OR2DN', 'OR2DL',
    'MPRD', 'MPRR', 'MPOR', 'MPORC', 'MPPETO', 'MPORM', 'IRR', 'IRD', 'IRSD', 'MD', 'SMD', 'SMDH', 'ROM',
    'VR', 'CVR', 'RPB', 'RBIS', 'D2OR', 'D2ORN', 'D2ORL', 'COR', 'UCOR', 'ZCOR', 'PCOR', 'ZPCOR', 'SPCOR',
    'R2', 'ZR2', 'R2F', 'ZR2F', 'PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT', 'IR', 'IRLN', 'IRS', 'IRFT',
    'MN', 'SMN', 'MNLN', 'SDLN', 'CVLN', 'MC', 'SMCC', 'SMCR', 'SMCRH', 'ROMC', 'VRC', 'CVRC', 'ARAW', 'AHW', 'ABT',
)  # fmt: skip
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
NOT_AVAILABLE = ('VR', 'CVR', 'PCOR', 'ZPCOR', 'SPCOR', 'R2', 'ZR2', 'R2F', 'ZR2F', 'SDLN', 'CVLN', 'VRC')
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
TABLE = ('RR', 'OR', 'RD', 'AS', 'PETO', 'PHI', 'YUQ', 'YUY', 'RTET', 'PBIT', 'OR2D', 'OR2DN', 'OR2DL', 'MPRD', 'MPRR', 'MPOR', 'MPORC', 'MPPETO', 'MPORM')
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
MEANS2 = ('MD', 'SMD', 'SMDH', 'ROM', 'RPB', 'RBIS', 'D2OR', 'D2ORN', 'D2ORL')
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
CHANGE2 = ('MC', 'SMCC', 'SMCRH', 'ROMC', 'CVRC')


@dataclasses.dataclass
class Collected:
    """The (subsetted, zero-cell adjusted) inputs, the study labels and the rows of ``data`` that go with them."""

    values: dict[str, FloatArray]
    slab: list[typing.Any]
    data: pandas.DataFrame | None


def family(measure: str) -> str:
    if measure in TABLE:
        return 'table'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('IRR', 'IRD', 'IRSD'):
        return 'irr'
    if measure in MEANS2:
        return 'means2'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('COR', 'UCOR', 'ZCOR'):
        return 'cor'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('PR', 'PLN', 'PLO', 'PRZ', 'PAS', 'PFT'):
        return 'pr'
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if measure in ('IR', 'IRLN', 'IRS', 'IRFT'):
        return 'ir'
    if measure in ('MN', 'SMN', 'MNLN'):
        return 'mn'
    if measure in (*CHANGE2, 'SMCR'):
        return 'mc'
    return 'alpha'


def anyv(cond: BoolArray) -> bool:
    """``any(cond, na.rm=TRUE)`` (comparisons with NaN are already False)."""
    return bool(numpy.any(cond))


def add_cells(cells: list[FloatArray], add: float, to: str) -> list[FloatArray]:
    """metafor's ``to='all'``/``'only0'``/``'if0all'`` adjustment of the cells of each study."""
    if to == 'all':
        return [c + add for c in cells]
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to in ('only0', 'if0all'):
        id0 = numpy.zeros(cells[0].shape[0], dtype=numpy.bool_)
        for c in cells:
            id0 |= c == 0
        if to == 'only0':
            return [numpy.where(id0, c + add, c) for c in cells]
        if id0.any():
            return [c + add for c in cells]
    return cells


def collect(
    measure: str,
    inp: metafopy.effects.measures.Inputs,
    data: pandas.DataFrame | None,
    slab: numpy.typing.ArrayLike | str | None,
    subset: numpy.typing.ArrayLike | str | None,
    add: float,
    to: str,
    drop00: bool,
    na_action: str,
) -> Collected:
    """The data part shared by ``to.long`` and ``to.table`` (everything before the output is laid out).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    E = metafopy.errors.InputError
    fam = family(measure)
    g = inp.get
    vals: dict[str, FloatArray]

    def need(names: str, *xs: FloatArray | None) -> list[FloatArray]:
        if not metafopy.effects.measures.all_specified(*xs):
            raise E(f'{metafopy.effects.measures.INCOMPLETE} (i.e., {names}).')
        return [x for x in xs if x is not None]

    def same_length(*xs: FloatArray | None) -> None:
        if not metafopy.effects.measures.equal_length(*xs):
            raise E('Supplied data vectors are not all of the same length.')

    with numpy.errstate(invalid='ignore'):
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        if fam == 'table':
            ai, bi, ci, di, n1i, n2i = (g(n) for n in ('ai', 'bi', 'ci', 'di', 'n1i', 'n2i'))
            same_length(ai, bi, ci, di, n1i, n2i)
            if n1i is not None and ai is not None and bi is not None and anyv(n1i != ai + bi):
                raise E("One or more 'n1i' values are not equal to 'ai + bi'.")
            if n2i is not None and ci is not None and di is not None and anyv(n2i != ci + di):
                raise E("One or more 'n2i' values are not equal to 'ci + di'.")
            bi = metafopy.effects.measures.replmiss(bi, None if n1i is None or ai is None else n1i - ai)
            di = metafopy.effects.measures.replmiss(di, None if n2i is None or ci is None else n2i - ci)
            ai, bi, ci, di = need('ai, bi, ci, di or ai, n1i, ci, n2i', ai, bi, ci, di)
            vals = {'ai': ai, 'bi': bi, 'ci': ci, 'di': di}
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'irr':
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            x1i, x2i, t1i, t2i = need('x1i, x2i, t1i, t2i', *(g(n) for n in ('x1i', 'x2i', 't1i', 't2i')))
            same_length(x1i, x2i, t1i, t2i)
            vals = {'x1i': x1i, 'x2i': x2i, 't1i': t1i, 't2i': t2i}
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'means2':
            names = ('m1i', 'm2i', 'sd1i', 'sd2i', 'n1i', 'n2i')
            arrs = need(', '.join(names), *(g(n) for n in names))
            same_length(*arrs)
            vals = dict(zip(names, arrs, strict=True))
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'cor':
            ri, ni, ti = g('ri'), g('ni'), g('ti')
            same_length(ri, ni, ti)
            ri = metafopy.effects.measures.replmiss(ri, None if ti is None or ni is None else ti / numpy.sqrt(ni - 2 + ti**2))
            ri, ni = need('ri, ni', ri, ni)
            vals = {'ri': ri, 'ni': ni}
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'pr':
            xi, mi, ni = g('xi'), g('mi'), g('ni')
            same_length(xi, mi, ni)
            if ni is not None and xi is not None and mi is not None and anyv(ni != xi + mi):
                raise E("One or more 'ni' values are not equal to 'xi + mi'.")
            mi = metafopy.effects.measures.replmiss(mi, None if ni is None or xi is None else ni - xi)
            xi, mi = need('xi, mi or xi, ni', xi, mi)
            vals = {'xi': xi, 'mi': mi}
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'ir':
            xi, ti = need('xi, ti', g('xi'), g('ti'))
            same_length(xi, ti)
            vals = {'xi': xi, 'ti': ti}
        elif fam == 'mn':
            mi, sdi, ni = need('mi, sdi, ni', g('mi'), g('sdi'), g('ni'))
            same_length(mi, sdi, ni)
            vals = {'mi': mi, 'sdi': sdi, 'ni': ni}
        elif fam == 'mc':
            mc_names = ('m1i', 'm2i', 'sd1i', 'sd2i', 'ni', 'ri') if measure in CHANGE2 else ('m1i', 'm2i', 'sd1i', 'ni', 'ri')
            arrs = need(', '.join(mc_names), *(g(n) for n in mc_names))
            same_length(*arrs)
            vals = dict(zip(mc_names, arrs, strict=True))
        else:
            ai, mi, ni = need('ai, mi, ni', g('ai'), g('mi'), g('ni'))
            same_length(ai, mi, ni)
            vals = {'ai': ai, 'mi': mi, 'ni': ni}

        k = next(iter(vals.values())).shape[0]
        sub = metafopy.effects.measures.chksubset(subset, k, data, 'subset')
        if sub is not None:
            vals = {key: v[sub] for key, v in vals.items()}

        v = vals
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        if fam == 'table':
            n1, n2 = v['ai'] + v['bi'], v['ci'] + v['di']
            if anyv(v['ai'] > n1) or anyv(v['ci'] > n2):
                raise E('One or more event counts are larger than the corresponding group sizes.')
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            if any(anyv(v[c] < 0) for c in ('ai', 'bi', 'ci', 'di')):
                raise E('One or more counts are negative.')
            if drop00:
                id00 = ((v['ai'] == 0) & (v['ci'] == 0)) | ((v['bi'] == 0) & (v['di'] == 0))
                v = {key: numpy.where(id00, numpy.nan, arr) for key, arr in v.items()}
            v = dict(zip(('ai', 'ci', 'bi', 'di'), add_cells([v['ai'], v['ci'], v['bi'], v['di']], add, to), strict=True))
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            v = {key: v[key] for key in ('ai', 'bi', 'ci', 'di')}
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'irr':
            if anyv(v['x1i'] < 0) or anyv(v['x2i'] < 0):
                raise E('One or more counts are negative.')
            if anyv(v['t1i'] <= 0) or anyv(v['t2i'] <= 0):
                raise E('One or more person-times are <= 0.')
            if drop00:
                id00 = (v['x1i'] == 0) & (v['x2i'] == 0)
                v['x1i'] = numpy.where(id00, numpy.nan, v['x1i'])
                v['x2i'] = numpy.where(id00, numpy.nan, v['x2i'])
            v['x1i'], v['x2i'] = add_cells([v['x1i'], v['x2i']], add, to)
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'means2':
            if anyv(v['sd1i'] < 0) or anyv(v['sd2i'] < 0):
                raise E('One or more standard deviations are negative.')
            if anyv(v['n1i'] <= 0) or anyv(v['n2i'] <= 0):
                raise E('One or more group sizes are <= 0.')
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'cor':
            if anyv(numpy.abs(v['ri']) > 1):
                raise E('One or more correlations are > 1 or < -1.')
            if anyv(v['ni'] <= 0):
                raise E('One or more sample sizes are <= 0.')
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'pr':
            ni = v['xi'] + v['mi']
            if anyv(v['xi'] > ni):
                raise E('One or more event counts are larger than the corresponding group sizes.')
            if anyv(v['xi'] < 0) or anyv(v['mi'] < 0):
                raise E('One or more counts are negative.')
            if anyv(ni <= 0):
                raise E('One or more group sizes are <= 0.')
            v['xi'], v['mi'] = add_cells([v['xi'], v['mi']], add, to)
        # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
        elif fam == 'ir':
            if anyv(v['xi'] < 0):
                raise E('One or more counts are negative.')
            if anyv(v['ti'] <= 0):
                raise E('One or more person-times are <= 0.')
            (v['xi'],) = add_cells([v['xi']], add, to)
        elif fam == 'mn':
            if anyv(v['sdi'] < 0):
                raise E('One or more standard deviations are negative.')
            if anyv(v['ni'] <= 0):
                raise E('One or more sample sizes are <= 0.')
            if measure == 'MNLN' and anyv(v['mi'] < 0):
                raise E('One or more means are negative.')
        elif fam == 'mc':
            if anyv(v['sd1i'] < 0) or ('sd2i' in v and anyv(v['sd2i'] < 0)):
                raise E('One or more standard deviations are negative.')
            if anyv(numpy.abs(v['ri']) > 1):
                raise E('One or more correlations are > 1 or < -1.')
            if anyv(v['ni'] <= 0):
                raise E('One or more sample sizes are <= 0.')
        else:
            if anyv(v['ai'] > 1):
                raise E('One or more alpha values are > 1.')
            if anyv(v['mi'] < 2):
                raise E('One or more mi values are < 2.')
            if anyv(v['ni'] <= 0):
                raise E('One or more sample sizes are <= 0.')

    labels = study_labels(slab, data, k)
    if sub is not None:
        labels = [s for s, keep in zip(labels, sub, strict=True) if keep]
        if data is not None:
            data = data.loc[sub]
    if len({str(s) for s in labels}) != len(labels):
        labels = list(metafopy.effects.measures.make_unique_dot1([str(s) for s in labels]))

    has_na = numpy.zeros(len(labels), dtype=numpy.bool_)
    for arr in v.values():
        has_na |= numpy.isnan(arr)
    if has_na.any():
        if na_action == 'omit':
            keep = ~has_na
            v = {key: arr[keep] for key, arr in v.items()}
            labels = [s for s, ok in zip(labels, keep, strict=True) if ok]
            if data is not None:
                data = data.loc[keep]
            n = int(has_na.sum())
            warnings.warn(f'{n} {"tables" if n > 1 else "table"} with NAs omitted.', stacklevel=3)
        elif na_action == 'fail':
            raise E('Missing values in tables.')
    if len(labels) < 1:
        raise E('Processing terminated since k = 0.')
    return Collected(v, labels, data)


def study_labels(slab: numpy.typing.ArrayLike | str | None, data: pandas.DataFrame | None, k: int) -> list[typing.Any]:
    """The study labels: ``slab`` (values or a column of ``data``), else ``1, ..., k``.

    Raises:
        metafopy.errors.InputError: the labels have missing values or the wrong length.
    """
    if slab is None:
        return list(range(1, k + 1))
    if isinstance(slab, str):
        if data is None:
            raise metafopy.errors.InputError(f'slab={slab!r} names a column, but no data was given')
        raw = data[slab].tolist()
    else:
        raw = numpy.asarray(slab, dtype=object).reshape(-1).tolist()
    if any(metafopy.effects.datatools.is_missing(s) for s in raw):
        raise metafopy.errors.InputError('NAs in study labels.')
    if len(raw) != k:
        raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(raw)}) does not correspond to the size of the dataset ({k}).")
    return raw


def check_common(measure: str | None, to: str, na_action: str) -> str:
    """Raises:
    metafopy.errors.InputError: the measure, ``to`` or ``na_action`` is not valid.
    """
    if measure is None:
        raise metafopy.errors.InputError("Must specify an effect size or outcome measure via the 'measure' argument.")
    if not isinstance(measure, str):
        raise metafopy.errors.InputError("The 'measure' argument must be a character string.")
    if measure not in KNOWN:
        raise metafopy.errors.InputError("Unknown 'measure' specified.")
    if measure in NOT_AVAILABLE:
        raise metafopy.errors.InputError('Function not available for this outcome measure.')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action not in ('omit', 'exclude', 'fail', 'pass'):
        raise metafopy.errors.InputError("Unknown 'na.action' specified under options().")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to not in ('all', 'only0', 'if0all', 'none'):
        raise metafopy.errors.InputError("Unknown 'to' argument specified.")
    return measure


def append_positions(append: bool | collections.abc.Sequence[int | str] | int | str, data: pandas.DataFrame | None) -> list[int] | None:
    """The columns of ``data`` to append: all (``True``), none, 0-based positions or (partially matched) names."""
    if data is None:
        return None
    if isinstance(append, bool | numpy.bool_):
        return list(range(data.shape[1])) if append else None
    items = [append] if isinstance(append, int | str) else list(append)
    columns = [str(c) for c in data.columns]
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            pos = metafopy.effects.datatools.charmatch(item, columns)
        else:
            pos = round(float(item))
            pos = pos if 0 <= pos < len(columns) else None
        if pos is not None and pos not in out:
            out.append(pos)
    return out


def study_factor(labels: collections.abc.Sequence[typing.Any], each: int) -> pandas.Categorical[str]:
    """``factor(rep(slab, each=each))`` with string labels (the categories in R's sorted order)."""
    cats = sorted(set(labels))
    values = [str(s) for s in labels for _ in range(each)]
    return pandas.Categorical(values, categories=[str(c) for c in cats])


def code_factor(codes: collections.abc.Sequence[int], k: int) -> pandas.Categorical[str]:
    """``factor(rep(codes, k), levels=c(2, 1))``."""
    return pandas.Categorical([str(c) for _ in range(k) for c in codes], categories=['2', '1'])


def interleave(*arrays: FloatArray) -> FloatArray:
    """``c(rbind(a, b, ...))``."""
    out: FloatArray = numpy.column_stack(arrays).reshape(-1)
    return out


def to_long(
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
    sdi: Arg = None,
    ni: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, numpy.typing.ArrayLike] | None = None,
    slab: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float = 1 / 2,
    to: str = 'none',
    drop00: bool = False,
    vlong: bool = False,
    append: bool | collections.abc.Sequence[int | str] | int | str = True,
    var_names: collections.abc.Sequence[str] | None = None,
    na_action: NAAction = 'omit',
) -> pandas.DataFrame:
    """Convert summary data in wide format (one row per study) into long format (one row per group or cell).

    The arguments are those of :func:`metafopy.escalc` for the given ``measure``. ``vlong=True`` gives one
    row per cell of 2x2 tables (and per outcome for proportions). ``append`` adds all (``True``) or the
    given columns of ``data``; ``var_names`` renames the created columns.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    measure = check_common(measure, to, na_action)
    has_data = data is not None
    if data is not None and not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)
    data = None if data is None else data.reset_index(drop=True)
    appendvars = append_positions(append, data)
    args: dict[str, Arg] = dict(ai=ai, bi=bi, ci=ci, di=di, n1i=n1i, n2i=n2i, x1i=x1i, x2i=x2i, t1i=t1i, t2i=t2i,
                                m1i=m1i, m2i=m2i, sd1i=sd1i, sd2i=sd2i, xi=xi, mi=mi, ri=ri, ti=ti, sdi=sdi, ni=ni)  # fmt: skip
    col = collect(measure, metafopy.effects.measures.Inputs(data, args), data, slab, subset, add, to, drop00, na_action)
    v, labels = col.values, col.slab
    k = len(labels)
    fam = family(measure)
    check_names = False
    columns: list[typing.Any]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    if fam == 'table':
        a, b, c, d = v['ai'], v['bi'], v['ci'], v['di']
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if measure in ('MPRD', 'MPRR', 'MPOR'):
            if vlong:
                names, each = ['study', 'time', 'outcome', 'freq'], 4
                columns = [code_factor([1, 1, 2, 2], k), code_factor([1, 2, 1, 2], k), interleave(a + b, c + d, a + c, b + d)]
                check_names = True
            else:
                names, each = ['study', 'time', 'out1', 'out2'], 2
                columns = [code_factor([1, 2], k), interleave(a + b, a + c), interleave(c + d, b + d)]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        elif measure in ('MPORC', 'MPPETO'):
            if vlong:
                names, each = ['study', 'out.time1', 'out.time2', 'freq'], 4
                columns = [code_factor([1, 1, 2, 2], k), code_factor([1, 2, 1, 2], k), interleave(a, b, c, d)]
            else:
                names, each = ['study', 'out.time1', 'out1.time2', 'out2.time2'], 2
                columns = [code_factor([1, 2], k), interleave(a, c), interleave(b, d)]
        elif vlong:
            names, each = ['study', 'group', 'outcome', 'freq'], 4
            columns = [code_factor([1, 1, 2, 2], k), code_factor([1, 2, 1, 2], k), interleave(a, b, c, d)]
        else:
            names, each = ['study', 'group', 'out1', 'out2'], 2
            columns = [code_factor([1, 2], k), interleave(a, c), interleave(b, d)]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'irr':
        names, each = ['study', 'group', 'events', 'ptime'], 2
        columns = [code_factor([1, 2], k), interleave(v['x1i'], v['x2i']), interleave(v['t1i'], v['t2i'])]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'means2':
        names, each = ['study', 'group', 'mean', 'sd', 'n'], 2
        columns = [code_factor([1, 2], k), interleave(v['m1i'], v['m2i']), interleave(v['sd1i'], v['sd2i']), interleave(v['n1i'], v['n2i'])]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'cor':
        names, each = ['study', 'r', 'n'], 1
        columns = [v['ri'], v['ni']]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'pr':
        if vlong:
            names, each = ['study', 'outcome', 'freq'], 2
            columns = [code_factor([1, 2], k), interleave(v['xi'], v['mi'])]
        else:
            names, each = ['study', 'out1', 'out2'], 1
            columns = [v['xi'], v['mi']]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'ir':
        names, each = ['study', 'events', 'ptime'], 1
        columns = [v['xi'], v['ti']]
    elif fam == 'mn':
        names, each = ['study', 'mean', 'sd', 'n'], 1
        columns = [v['mi'], v['sdi'], v['ni']]
    elif fam == 'mc':
        if measure in CHANGE2:
            names = ['study', 'mean1', 'mean2', 'sd1', 'sd2', 'n', 'r']
            columns = [v['m1i'], v['m2i'], v['sd1i'], v['sd2i'], v['ni'], v['ri']]
        else:
            names = ['study', 'mean1', 'mean2', 'sd1', 'n', 'r']
            columns = [v['m1i'], v['m2i'], v['sd1i'], v['ni'], v['ri']]
        each = 1
    else:
        names, each = ['study', 'alpha', 'm', 'n'], 1
        columns = [v['ai'], v['mi'], v['ni']]
        check_names = True
    if var_names is not None:
        if len(var_names) != len(names):
            raise metafopy.errors.InputError(f'Variable names not of length {len(names)}.')
        names = list(var_names)
    out = pandas.DataFrame(dict(enumerate([study_factor(labels, each), *columns])))
    out.columns = pandas.Index(names)
    if has_data and appendvars is not None and col.data is not None:
        rows = col.data.iloc[numpy.repeat(numpy.arange(k), each), appendvars].reset_index(drop=True)
        out = pandas.concat([rows, out], axis=1)
        if check_names:
            out.columns = pandas.Index(metafopy.effects.datatools.make_names([str(c) for c in out.columns], unique=True))
    return out.reset_index(drop=True)


@dataclasses.dataclass(frozen=True)
class TableArray:
    """The tables of ``to_table``: ``values`` has shape (rows, columns, studies), as R's array; ``table(i)`` gives one study as a data frame."""

    values: FloatArray
    rows: tuple[str, ...]
    cols: tuple[str, ...]
    slab: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.slab)

    def table(self, study: int | str) -> pandas.DataFrame:
        """The table of one study, by 0-based position or label.

        Raises:
            metafopy.errors.NotFoundError: there is no such study.
        """
        if isinstance(study, str):
            if study not in self.slab:
                raise metafopy.errors.NotFoundError(f'no study labelled {study!r}')
            study = self.slab.index(study)
        return pandas.DataFrame(self.values[:, :, study], index=list(self.rows), columns=list(self.cols))

    def __str__(self) -> str:
        return '\n\n'.join(f', , {s}\n\n{self.table(i).to_string()}' for i, s in enumerate(self.slab))


def to_table(
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
    sdi: Arg = None,
    ni: Arg = None,
    *,
    data: pandas.DataFrame | collections.abc.Mapping[str, numpy.typing.ArrayLike] | None = None,
    slab: numpy.typing.ArrayLike | str | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: float = 1 / 2,
    to: str = 'none',
    drop00: bool = False,
    rows: collections.abc.Sequence[str] | None = None,
    cols: collections.abc.Sequence[str] | None = None,
    na_action: NAAction = 'omit',
) -> TableArray:
    """Convert summary data into one table per study (2x2 tables, group means/SDs/sizes, ...).

    The arguments are those of :func:`metafopy.escalc` for the given ``measure``; ``rows`` and ``cols``
    name the rows and columns of the tables.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    measure = check_common(measure, to, na_action)
    if data is not None and not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)
    data = None if data is None else data.reset_index(drop=True)
    args: dict[str, Arg] = dict(ai=ai, bi=bi, ci=ci, di=di, n1i=n1i, n2i=n2i, x1i=x1i, x2i=x2i, t1i=t1i, t2i=t2i,
                                m1i=m1i, m2i=m2i, sd1i=sd1i, sd2i=sd2i, xi=xi, mi=mi, ri=ri, ti=ti, sdi=sdi, ni=ni)  # fmt: skip
    col = collect(measure, metafopy.effects.measures.Inputs(data, args), data, slab, subset, add, to, drop00, na_action)
    v = col.values
    fam = family(measure)
    cells: list[list[FloatArray]]
    dflt: tuple[tuple[str, ...], tuple[str, ...]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    if fam == 'table':
        a, b, c, d = v['ai'], v['bi'], v['ci'], v['di']
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if measure in ('MPRD', 'MPRR', 'MPOR'):
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            dflt = (('Time1', 'Time2'), ('Out1', 'Out2'))
            cells = [[a + b, c + d], [a + c, b + d]]
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        elif measure in ('MPORC', 'MPPETO'):
            dflt = (('Time1.Out1', 'Time1.Out2'), ('Time2.Out1', 'Time2.Out2'))
            cells = [[a, b], [c, d]]
        else:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            dflt = (('Grp1', 'Grp2'), ('Out1', 'Out2'))
            cells = [[a, b], [c, d]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'irr':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp1', 'Grp2'), ('Events', 'Person-Time'))
        cells = [[v['x1i'], v['t1i']], [v['x2i'], v['t2i']]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'means2':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp1', 'Grp2'), ('Mean', 'SD', 'n'))
        cells = [[v['m1i'], v['sd1i'], v['n1i']], [v['m2i'], v['sd2i'], v['n2i']]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'cor':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp',), ('r', 'n'))
        cells = [[v['ri'], v['ni']]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'pr':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp',), ('Out1', 'Out2'))
        cells = [[v['xi'], v['mi']]]
    # auditme: accept[assert-never-missing] fam is a plain str from family(); the final else covers the last family, as in R
    elif fam == 'ir':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp',), ('Events', 'Person-Time'))
        cells = [[v['xi'], v['ti']]]
    elif fam == 'mn':
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp',), ('Mean', 'SD', 'n'))
        cells = [[v['mi'], v['sdi'], v['ni']]]
    elif fam == 'mc':
        if measure in CHANGE2:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            dflt = (('Grp',), ('Mean1', 'Mean2', 'SD1', 'SD2', 'n', 'r'))
            cells = [[v['m1i'], v['m2i'], v['sd1i'], v['sd2i'], v['ni'], v['ri']]]
        else:
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            dflt = (('Grp',), ('Mean1', 'Mean2', 'SD1', 'n', 'r'))
            cells = [[v['m1i'], v['m2i'], v['sd1i'], v['ni'], v['ri']]]
    else:
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        dflt = (('Grp',), ('alpha', 'm', 'n'))
        cells = [[v['ai'], v['mi'], v['ni']]]
    row_names = tuple(dflt[0]) if rows is None else tuple(rows)
    col_names = tuple(dflt[1]) if cols is None else tuple(cols)
    if len(row_names) != len(dflt[0]):
        raise metafopy.errors.InputError(f'Group names not of length {len(dflt[0])}.')
    if len(col_names) != len(dflt[1]):
        raise metafopy.errors.InputError(f'Outcome names not of length {len(dflt[1])}.')
    values = numpy.array([[cell for cell in row] for row in cells], dtype=numpy.float64)
    return TableArray(values=values, rows=row_names, cols=col_names, slab=tuple(str(s) for s in col.slab))


def to_wide(
    data: pandas.DataFrame | collections.abc.Mapping[str, typing.Any],
    study: str | int,
    grp: str | int,
    ref: typing.Any = None,
    grpvars: collections.abc.Sequence[str | int] | str | int | None = None,
    postfix: collections.abc.Sequence[str] = ('.1', '.2'),
    addid: bool = True,
    addcomp: bool = True,
    adddesign: bool = True,
    minlen: int = 2,
    var_names: collections.abc.Sequence[str] = ('id', 'comp', 'design'),
) -> pandas.DataFrame:
    """Convert arm-level data (one row per group) into contrast-level data (one row per comparison with the reference group).

    ``study`` and ``grp`` name (or give the 0-based position of) the study and group variables, ``ref``
    is the reference group (default: the most common group) and ``grpvars`` the group-specific variables,
    which get ``postfix[0]`` for the group and ``postfix[1]`` for the reference (the last arm of each
    study, in the order of the group levels with ``ref`` last). ``addid``, ``addcomp`` and ``adddesign``
    add a row id (1, 2, ...), the comparison and the design (group names shortened to at least ``minlen``
    characters).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if not isinstance(data, pandas.DataFrame):
        data = pandas.DataFrame(data)
    data = data.reset_index(drop=True)
    if len(var_names) != 3:
        raise metafopy.errors.InputError("Argument 'var_names' must of length 3.")
    if grpvars is None:
        raise metafopy.errors.InputError("Argument 'grpvars' must be specified.")
    nvars = data.shape[1]
    study_pos = metafopy.effects.datatools.resolve_column(data, study, 'study')
    if any(metafopy.effects.datatools.is_missing(s) for s in data.iloc[:, study_pos]):
        raise metafopy.errors.InputError("Variable specified via 'study' argument should not contain missing values.")
    grp_pos = metafopy.effects.datatools.resolve_column(data, grp, 'grp')
    grp_col = data.iloc[:, grp_pos]
    if any(metafopy.effects.datatools.is_missing(s) for s in grp_col):
        raise metafopy.errors.InputError("Variable specified via 'grp' argument should not contain missing values.")
    lvls = metafopy.effects.datatools.levels_of(grp_col)
    if ref is None:
        ref = metafopy.effects.datatools.most_common(grp_col.tolist(), lvls)
    ref_pos = metafopy.effects.datatools.charmatch(str(ref), [str(lvl) for lvl in lvls])
    if ref_pos is None:
        raise metafopy.errors.InputError("Could not find or uniquely identify reference group specified via the 'ref' argument.")
    lvls = [*lvls[:ref_pos], *lvls[ref_pos + 1 :], lvls[ref_pos]]

    study_vals = data.iloc[:, study_pos].tolist()
    ustudy = sorted(set(study_vals))
    study_rank = numpy.array([ustudy.index(s) for s in study_vals])
    lvl_index = {lvl: j for j, lvl in enumerate(lvls)}
    grp_rank = numpy.array([lvl_index[g] for g in grp_col.tolist()])
    data = data.iloc[numpy.lexsort((grp_rank, study_rank))].reset_index(drop=True)
    study_rank = numpy.array([ustudy.index(s) for s in data.iloc[:, study_pos].tolist()])

    columns = [str(c) for c in data.columns]
    specs = [grpvars] if isinstance(grpvars, str | int) else list(grpvars)
    gpos: list[int] = []
    for spec in specs:
        if isinstance(spec, str):
            pos = metafopy.effects.datatools.charmatch(spec, columns)
            if pos is None:
                raise metafopy.errors.InputError("Could not find or uniquely identify variable(s) specified via the 'grpvars' argument.")
        else:
            pos = round(float(spec))
            if pos < 0 or pos >= nvars:
                raise metafopy.errors.InputError("Specified positions of 'grpvars' variables do not exist in the data frame.")
        if pos not in gpos:
            gpos.append(pos)
    gpos = [grp_pos, *[p for p in gpos if p != grp_pos]]

    pieces: list[pandas.DataFrame] = []
    for s in range(len(ustudy)):
        x = data.loc[study_rank == s]
        n = x.shape[0]
        left = x.iloc[: n - 1] if n > 1 else x
        right = x.iloc[[n - 1] * (n - 1), gpos] if n > 1 else pandas.DataFrame([[numpy.nan] * len(gpos)])
        right = right.reset_index(drop=True)
        right.columns = pandas.RangeIndex(nvars, nvars + len(gpos))
        left = left.reset_index(drop=True)
        left.columns = pandas.RangeIndex(nvars)
        pieces.append(pandas.concat([left, right], axis=1))
    dat = pandas.concat(pieces, axis=0, ignore_index=True)
    names = list(columns)
    for p in gpos:
        names[p] = f'{columns[p]}{postfix[0]}'
    names += [f'{columns[p]}{postfix[1]}' for p in gpos]
    dat.columns = pandas.Index(names)

    grps = metafopy.effects.datatools.shorten_names([str(g) for g in data.iloc[:, grp_pos].tolist()], minlen=minlen)
    comp: list[str | None] = []
    design: list[str | None] = []
    for s in range(len(ustudy)):
        g = [grps[i] for i in numpy.flatnonzero(study_rank == s)]
        if len(g) > 1:
            comp.extend(f'{a}-{g[-1]}' for a in g[:-1])
            design.extend(['-'.join(str(a) for a in g)] * (len(g) - 1))
        else:
            comp.append(None)
            design.append(None)
    if addid:
        dat[var_names[0]] = numpy.arange(1, dat.shape[0] + 1)
    if addcomp:
        dat[var_names[1]] = comp
    if adddesign:
        dat[var_names[2]] = design
    return dat
