"""Normalisation of model inputs into a single :class:`MetaData` bundle.

This mirrors the argument handling at the top of metafor's ``rma.uni`` and
``rma.mv``: arguments may be arrays or column names in ``data``; ``subset`` is
applied first, then the ``na.action`` rule; ``mods`` may be a one-sided formula
or a matrix (in which case an intercept is added unless ``intercept=False``).
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

__all__ = ['MetaData', 'NAAction', 'escalc_ni', 'make_unique']

type NAAction = typing.Literal['omit', 'fail', 'pass']
type Column = numpy.typing.ArrayLike | str

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]


def make_unique(labels: collections.abc.Sequence[str]) -> tuple[str, ...]:
    """Make labels unique the way R's ``make.unique`` does (``a``, ``a.1``, ``a.2``)."""
    seen: set[str] = set()
    counts: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
            continue
        n = counts.get(label, 0)
        while True:
            n += 1
            candidate = f'{label}.{n}'
            if candidate not in seen:
                break
        counts[label] = n
        seen.add(candidate)
        out.append(candidate)
    return tuple(out)


def column(value: Column | None, data: pandas.DataFrame | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if data is None:
            raise metafopy.errors.InputError(f'{name}={value!r} names a column, but no data was given')
        if value not in data.columns:
            raise metafopy.errors.NotFoundError(f'column {value!r} (for {name}) not found in data')
        value = data[value].to_numpy()
    arr = numpy.asarray(value, dtype=numpy.float64)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise metafopy.errors.InputError(f'{name} must be one-dimensional, got shape {arr.shape}')
    return arr


def subset_mask(subset: numpy.typing.ArrayLike | str | None, k: int, data: pandas.DataFrame | None) -> BoolArray:
    if subset is None:
        return numpy.ones(k, dtype=numpy.bool_)
    if isinstance(subset, str):
        if data is None:
            raise metafopy.errors.InputError('a string subset needs data to evaluate against')
        values = numpy.asarray(data.eval(subset))
    else:
        values = numpy.asarray(subset)
    if values.dtype == numpy.bool_:
        if values.shape != (k,):
            raise metafopy.errors.InputError(f'logical subset must have length {k}, got {values.shape}')
        return values.copy()
    mask = numpy.zeros(k, dtype=numpy.bool_)
    mask[values.astype(numpy.intp)] = True
    return mask


ROWS_KEY = 'rows'


def frame_attr(frame: pandas.DataFrame | pandas.Series | None, key: str) -> numpy.typing.NDArray[typing.Any] | None:
    """The row-wise attribute ``key`` (``'ni'`` or ``'slab'``) of an ``escalc`` frame, restricted to the rows that are left in ``frame``.

    R subsets the ``ni`` and ``slab`` attributes of the ``yi`` variable with the rows (``[.escalc``). pandas copies ``attrs`` unchanged when
    rows are selected, so ``escalc`` also stores the row labels it was created with (``attrs['rows']``) and the attribute is matched on them.
    Without a match (including a frame renumbered from 0, whose rows cannot be told apart from the first ones) the attribute is used when it has one
    value per row and dropped otherwise.
    """
    if frame is None:
        return None
    value = frame.attrs.get(key)
    if value is None:
        return None
    arr = numpy.asarray(value).reshape(-1)
    rows = frame.attrs.get(ROWS_KEY)
    if rows is not None and len(rows) == arr.shape[0]:
        ref = pandas.Index(rows)
        index = frame.index
        default_index = isinstance(index, pandas.RangeIndex) and index.start == 0 and index.step == 1
        if not default_index and not index.equals(ref) and ref.is_unique and index.isin(ref).all():
            aligned: numpy.typing.NDArray[typing.Any] = arr[ref.get_indexer(index)]
            return aligned
    return arr if arr.shape[0] == frame.shape[0] else None


def escalc_attr(yi: Column | None, data: pandas.DataFrame | None, k_all: int, key: str) -> numpy.typing.NDArray[typing.Any] | None:
    """The ``ni`` or ``slab`` stored with ``yi`` by ``escalc``, which ``rma.uni`` picks up from ``attr(yi, "ni")`` and ``attr(yi, "slab")``."""
    if isinstance(yi, str) and data is not None and yi in data.columns:
        arr = frame_attr(data, key)
    elif isinstance(yi, pandas.Series):
        arr = frame_attr(yi, key)
    else:
        return None
    return arr if arr is not None and arr.shape[0] == k_all else None


def escalc_ni(yi: Column | None, data: pandas.DataFrame | None, k_all: int) -> FloatArray | None:
    """Sample sizes stored with ``yi`` by ``escalc``, which ``rma.uni`` picks up from ``attr(yi, "ni")``."""
    arr = escalc_attr(yi, data, k_all, 'ni')
    return None if arr is None else arr.astype(numpy.float64)


@dataclasses.dataclass(frozen=True, slots=True)
class MetaData:
    """Model inputs after subsetting and missing-value handling.

    Attributes hold only the rows used in fitting (``k`` of them). ``keep``
    maps them back to the ``k_all`` rows that were supplied, so that results
    can be reported on the original rows (metafor's ``na.action="na.pass"``).
    ``na_rows`` flags the subsetted rows dropped for missing values, ``formula_mods`` is the formula of ``mods`` (if it was one), and ``labels_all``,
    ``yi_all``, ``vi_all`` and ``X_all`` hold the label, estimate, variance and design row of every supplied row
    (``forest`` shows the flagged rows with ``na_action='pass'``). ``counts_all`` holds the 2x2 table or person-time counts of the ``k_all`` rows
    when the effects were computed from them (metafor's ``outdat.f``, which ``labbe`` needs).
    """

    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    X_names: tuple[str, ...]
    slab: tuple[str, ...]
    keep: BoolArray
    weights: FloatArray | None = None
    ni: FloatArray | None = None
    intercept: bool = True
    data: pandas.DataFrame | None = dataclasses.field(default=None, repr=False)
    labels_all: tuple[str, ...] | None = dataclasses.field(default=None, repr=False)
    na_rows: BoolArray | None = dataclasses.field(default=None, repr=False)
    formula_mods: str | None = dataclasses.field(default=None, repr=False)
    yi_all: FloatArray | None = dataclasses.field(default=None, repr=False)
    vi_all: FloatArray | None = dataclasses.field(default=None, repr=False)
    X_all: FloatArray | None = dataclasses.field(default=None, repr=False)
    counts_all: dict[str, FloatArray] | None = dataclasses.field(default=None, repr=False)
    supplied: int | None = dataclasses.field(default=None, repr=False)
    """Rows supplied when the effects came from counts after subsetting (R's ``k.all``); ``k_all`` stays the length of ``keep``."""

    @property
    def k(self) -> int:
        """Number of estimates used in fitting."""
        return int(self.yi.shape[0])

    @property
    def k_all(self) -> int:
        """Number of rows supplied before subsetting and NA removal."""
        return int(self.keep.shape[0])

    @property
    def p(self) -> int:
        """Number of fixed-effects coefficients."""
        return int(self.X.shape[1])

    @classmethod
    def from_inputs(
        cls,
        yi: Column | None = None,
        vi: Column | None = None,
        *,
        sei: Column | None = None,
        weights: Column | None = None,
        ni: Column | None = None,
        mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None = None,
        intercept: bool = True,
        data: pandas.DataFrame | None = None,
        slab: Column | collections.abc.Sequence[str] | None = None,
        subset: numpy.typing.ArrayLike | str | None = None,
        na_action: NAAction = 'omit',
    ) -> MetaData:
        """Build a :class:`MetaData` from ``rma``-style arguments.

        When ``yi`` is omitted and ``data`` has ``yi``/``vi`` columns (as
        returned by :func:`metafopy.escalc`), those columns are used.

        Raises:
            metafopy.errors.InputError: an argument or the data is invalid.
        """
        if yi is None and data is not None and 'yi' in data.columns:
            yi = 'yi'
            if vi is None and sei is None and 'vi' in data.columns:
                vi = 'vi'
        y = column(yi, data, 'yi')
        if y is None:
            raise metafopy.errors.InputError('yi must be given')
        k_all = y.shape[0]

        v = column(vi, data, 'vi')
        se = column(sei, data, 'sei')
        if v is None and se is not None:
            v = se**2
        if v is None:
            raise metafopy.errors.InputError('one of vi or sei must be given')

        w = column(weights, data, 'weights')
        n = column(ni, data, 'ni')
        if v.shape[0] == 1 and k_all != 1:
            v = numpy.repeat(v, k_all)
        if w is not None and w.shape[0] == 1 and k_all != 1:
            w = numpy.repeat(w, k_all)
        if n is None:
            n = escalc_ni(yi, data, k_all)
        if n is not None and n.shape[0] != k_all:
            n = None  # as metafor: sample sizes of the wrong length 'cannot be trusted' and are dropped
        for name, arr in (('vi', v), ('weights', w)):
            if arr is not None and arr.shape[0] != k_all:
                raise metafopy.errors.InputError(f'length of {name} ({arr.shape[0]}) differs from yi ({k_all})')

        X, X_names, x_missing, has_intercept = cls.design(mods, intercept, data, k_all)

        if slab is None:
            stored = escalc_attr(yi, data, k_all, 'slab')
            if stored is not None and any(s is None or (isinstance(s, float) and numpy.isnan(s)) for s in stored.tolist()):
                raise metafopy.errors.InputError('NAs in study labels.')
            labels: tuple[str, ...] = tuple(str(i + 1) for i in range(k_all)) if stored is None else tuple(str(s) for s in stored.tolist())
        elif isinstance(slab, str):
            if data is None:
                raise metafopy.errors.InputError(f'slab={slab!r} names a column, but no data was given')
            labels = tuple(str(s) for s in data[slab].tolist())
        else:
            labels = tuple(str(s) for s in numpy.asarray(slab, dtype=object).tolist())
        if len(labels) != k_all:
            raise metafopy.errors.InputError(f'length of slab ({len(labels)}) differs from yi ({k_all})')
        labels = make_unique(labels)

        keep = subset_mask(subset, k_all, data)
        missing = numpy.isnan(y) | numpy.isnan(v) | x_missing
        if w is not None:
            missing |= numpy.isnan(w)
        missing &= keep
        if missing.any():
            if na_action == 'fail':
                raise metafopy.errors.InputError('missing values in data')
            keep = keep & ~missing
        na_rows = missing.copy()

        if w is not None and numpy.any(w[keep] < 0):
            raise metafopy.errors.InputError('there are negative weights')

        idx = numpy.flatnonzero(keep)
        return cls(
            yi=y[idx],
            vi=v[idx],
            X=X[idx],
            X_names=X_names,
            slab=tuple(labels[i] for i in idx),
            keep=keep,
            weights=None if w is None else w[idx],
            ni=None if n is None else n[idx],
            intercept=has_intercept,
            data=data,
            labels_all=labels,
            na_rows=na_rows,
            formula_mods=mods if isinstance(mods, str) and data is not None else None,
            yi_all=y,
            vi_all=v,
            X_all=X,
        )

    @staticmethod
    def design(
        mods: str | numpy.typing.ArrayLike | pandas.DataFrame | None,
        intercept: bool,
        data: pandas.DataFrame | None,
        k: int,
    ) -> tuple[FloatArray, tuple[str, ...], BoolArray, bool]:
        if mods is None:
            if not intercept:
                warnings.warn('Must either include an intercept and/or moderators in the model.\nCoerced an intercept into the model.', stacklevel=4)
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            return numpy.ones((k, 1)), ('intrcpt',), numpy.zeros(k, dtype=numpy.bool_), True
        if isinstance(mods, str):
            if data is None:
                raise metafopy.errors.InputError('a formula for mods needs data')
            X, names, missing = metafopy.formula.design_matrix(mods, data)
            return X, names, missing, 'intrcpt' in names
        if isinstance(mods, pandas.DataFrame):
            names = tuple(str(c) for c in mods.columns)
            X = mods.to_numpy(dtype=float)
        else:
            X = numpy.asarray(mods, dtype=numpy.float64)
            if X.ndim == 1:
                X = X[:, None]
            names = tuple(f'X{j + 1}' for j in range(X.shape[1]))
        if X.shape[0] != k:
            raise metafopy.errors.InputError(f'mods has {X.shape[0]} rows, expected {k}')
        if intercept:
            X = numpy.column_stack([numpy.ones(k), X])
            names = ('intrcpt', *names)
        return X.astype(numpy.float64), names, numpy.isnan(X).any(axis=1), intercept
