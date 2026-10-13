"""Preparation of 2x2 table and two-group rate data for ``rma_mh``, ``rma_peto`` and ``rma_glmm``.

This follows the data-handling steps shared by metafor's ``rma.mh``
(https://github.com/wviechtb/metafor/blob/master/R/rma.mh.r), ``rma.peto``
and ``rma.glmm``: extract and subset the cell counts, compute ``yi``/``vi``
with ``escalc`` (using the first element of ``add``, ``to`` and ``drop00``),
optionally drop double-zero tables (second element of ``drop00``), set tables
with missing counts aside and apply the second ``add``/``to`` pair to the
counts that enter the actual model.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas

import metafopy.effects
import metafopy.errors
import metafopy.metadata

__all__ = ['TableSet', 'pair', 'prepare_tables']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type IntArray = numpy.typing.NDArray[numpy.intp]
type To = typing.Literal['all', 'only0', 'if0all', 'none']

TWO_BY_TWO = ('ai', 'bi', 'ci', 'di')
RATES = ('x1i', 'x2i', 't1i', 't2i')
TO_CHOICES = ('all', 'only0', 'if0all', 'none')


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TableSet:
    """Table data at each stage of the ``rma.mh`` / ``rma.peto`` data handling.

    ``*_f`` arrays hold all supplied (and subsetted) tables including those with
    missing counts; the arrays without the suffix only the ones used in model
    fitting. ``cells`` are the counts after the second ``add``/``to`` step.
    """

    measure: str
    cells_f: dict[str, FloatArray]
    cells: dict[str, FloatArray]
    yi_f: FloatArray
    vi_f: FloatArray
    ni_f: FloatArray
    yi: FloatArray
    vi: FloatArray
    ni: FloatArray
    not_na: BoolArray
    not_na_yivi: BoolArray
    ids: IntArray
    slab: tuple[str, ...]
    slab_null: bool
    subset: BoolArray | None
    k: int
    k_f: int
    k_yi: int
    k_all: int


def pair[T](value: T | collections.abc.Sequence[T], default: T, name: str) -> tuple[T, T]:
    """Expand a scalar or two-element ``add``/``to``/``drop00`` argument to a pair, as ``rma.mh`` does.

    Raises:
        metafopy.errors.InputError: the argument has more than two elements.
    """
    if isinstance(value, str) or not isinstance(value, collections.abc.Sequence):
        return (value, default)
    if len(value) == 1:
        return (value[0], default)
    if len(value) != 2:
        raise metafopy.errors.InputError(f"Argument '{name}' should specify one or two values.")
    return (value[0], value[1])


def check_to(to: tuple[str, str]) -> None:
    for t in to:
        if t not in TO_CHOICES:
            raise metafopy.errors.InputError("Unknown 'to' argument specified.")


def add_to_cells(cells: dict[str, FloatArray], names: tuple[str, ...], add: float, to: str, any0_names: tuple[str, ...]) -> None:
    """Add ``add`` to the counts as ``to`` requests (second ``add``/``to`` pair of ``rma.mh``)."""
    if to == 'none':
        return
    with numpy.errstate(invalid='ignore'):
        id0 = numpy.zeros(cells[names[0]].shape[0], dtype=numpy.bool_)
        for n in any0_names:
            id0 |= cells[n] == 0
    if to == 'all':
        sel = numpy.ones_like(id0)
    elif to == 'only0':
        sel = id0
    else:
        sel = numpy.ones_like(id0) if id0.any() else numpy.zeros_like(id0)
    for n in names:
        cells[n] = numpy.where(sel, cells[n] + add, cells[n])


def prepare_tables(
    measure: str,
    *,
    ai: metafopy.metadata.Column | None = None,
    bi: metafopy.metadata.Column | None = None,
    ci: metafopy.metadata.Column | None = None,
    di: metafopy.metadata.Column | None = None,
    n1i: metafopy.metadata.Column | None = None,
    n2i: metafopy.metadata.Column | None = None,
    x1i: metafopy.metadata.Column | None = None,
    x2i: metafopy.metadata.Column | None = None,
    t1i: metafopy.metadata.Column | None = None,
    t2i: metafopy.metadata.Column | None = None,
    data: pandas.DataFrame | None = None,
    slab: metafopy.metadata.Column | collections.abc.Sequence[str] | None = None,
    subset: numpy.typing.ArrayLike | str | None = None,
    add: tuple[float, float] = (0.5, 0.0),
    to: tuple[To, To] = ('only0', 'none'),
    drop00: tuple[bool, bool] = (True, False),
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
    na_action: metafopy.metadata.NAAction = 'omit',
) -> TableSet:
    """Extract, subset and screen the table data like the first half of ``rma.mh``.

    ``measure`` is ``'OR'``, ``'RR'``, ``'RD'``, ``'PETO'`` (2x2 tables) or ``'IRR'``, ``'IRD'`` (rates).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    rate = measure in ('IRR', 'IRD')
    names = RATES if rate else TWO_BY_TWO
    if rate:
        given = {n: metafopy.metadata.column(v, data, n) for n, v in zip(RATES, (x1i, x2i, t1i, t2i), strict=True)}
        if any(v is None for v in given.values()):
            raise metafopy.errors.InputError('Must specify arguments x1i, x2i, t1i, t2i for this measure.')
        raw = {n: typing.cast(FloatArray, v) for n, v in given.items()}
        ni = raw['t1i'] + raw['t2i']
    else:
        a, b, c, d = (metafopy.metadata.column(v, data, n) for n, v in zip(TWO_BY_TWO, (ai, bi, ci, di), strict=True))
        n1 = metafopy.metadata.column(n1i, data, 'n1i')
        n2 = metafopy.metadata.column(n2i, data, 'n2i')
        if b is None and n1 is not None and a is not None:
            b = n1 - a
        if d is None and n2 is not None and c is not None:
            d = n2 - c
        if a is None or b is None or c is None or d is None:
            raise metafopy.errors.InputError('Must specify arguments ai, bi, ci, di (or ai, ci, n1i, n2i) for this measure.')
        raw = {'ai': a, 'bi': b, 'ci': c, 'di': d}
        ni = a + b + c + d

    k_all = raw[names[0]].shape[0]
    if any(v.shape[0] != k_all for v in raw.values()):
        raise metafopy.errors.InputError('The table data arguments must all have the same length.')
    ids = numpy.arange(k_all, dtype=numpy.intp)

    if slab is None:
        labels: tuple[str, ...] = tuple(str(i + 1) for i in range(k_all))
        slab_null = True
    else:
        if isinstance(slab, str):
            if data is None:
                raise metafopy.errors.InputError(f'slab={slab!r} names a column, but no data was given')
            values = data[slab].tolist()
        else:
            values = numpy.asarray(slab, dtype=object).tolist()
        if any(s is None or (isinstance(s, float) and math.isnan(s)) for s in values):
            raise metafopy.errors.InputError('NAs in study labels.')
        if len(values) != k_all:
            raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(values)}) does not correspond to the size of the dataset ({k_all}).")
        labels = tuple(str(s) for s in values)
        slab_null = False

    mask: BoolArray | None = None
    if subset is not None:
        mask = metafopy.metadata.subset_mask(subset, k_all, data)
        idx = numpy.flatnonzero(mask)
        raw = {n: v[idx] for n, v in raw.items()}
        ni = ni[idx]
        labels = tuple(labels[i] for i in idx)
        ids = ids[idx]
    labels = metafopy.metadata.make_unique(labels)
    k = ids.shape[0]

    cell_args: dict[str, typing.Any] = dict(raw)
    es = metafopy.effects.escalc(
        measure,
        **cell_args,
        add=add[0],
        to=to[0],
        drop00=drop00[0],
        onlyo1=onlyo1,
        addyi=addyi,
        addvi=addvi,
        cutoff=math.nan,
    )
    yi_f = es['yi'].to_numpy(dtype=numpy.float64)
    vi_f = es['vi'].to_numpy(dtype=numpy.float64)

    cells_f = {n: v.copy() for n, v in raw.items()}
    if drop00[1]:
        if rate:
            id00 = (cells_f['x1i'] == 0) & (cells_f['x2i'] == 0)
        else:
            id00 = ((cells_f['ai'] == 0) & (cells_f['ci'] == 0)) | ((cells_f['bi'] == 0) & (cells_f['di'] == 0))
        for n in ('x1i', 'x2i') if rate else names:
            cells_f[n][id00] = numpy.nan

    has_na = numpy.zeros(k, dtype=numpy.bool_)
    for n in names:
        has_na |= numpy.isnan(cells_f[n])
    not_na = ~has_na
    cells = {n: v.copy() for n, v in cells_f.items()}
    if has_na.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('Missing values in tables.')
        cells = {n: v[not_na] for n, v in cells.items()}
        warnings.warn(f'{int(has_na.sum())} {"studies" if has_na.sum() > 1 else "study"} with NAs omitted from model fitting.', stacklevel=3)
    k_fit = int(not_na.sum())
    if k_fit < 1:
        raise metafopy.errors.InputError('Processing terminated since k = 0.')

    yivi_na = numpy.isnan(yi_f) | numpy.isnan(vi_f)
    not_na_yivi = ~yivi_na
    yi, vi, ni_fit = yi_f, vi_f, ni
    if yivi_na.any():
        if na_action == 'fail':
            raise metafopy.errors.InputError('Missing yi/vi values.')
        yi, vi, ni_fit = yi_f[not_na_yivi], vi_f[not_na_yivi], ni[not_na_yivi]
        warnings.warn('Some yi/vi values are NA.', stacklevel=3)

    if rate:
        add_to_cells(cells, ('x1i', 'x2i'), add[1], to[1], ('x1i', 'x2i'))
    else:
        add_to_cells(cells, TWO_BY_TWO, add[1], to[1], TWO_BY_TWO)

    return TableSet(
        measure=measure,
        cells_f=cells_f,
        cells=cells,
        yi_f=yi_f,
        vi_f=vi_f,
        ni_f=ni,
        yi=yi,
        vi=vi,
        ni=ni_fit,
        not_na=not_na,
        not_na_yivi=not_na_yivi,
        ids=ids,
        slab=labels,
        slab_null=slab_null,
        subset=mask,
        k=k_fit,
        k_f=k,
        k_yi=int(yi.shape[0]),
        k_all=k_all,
    )
