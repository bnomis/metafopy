"""Matrices and data frames that print as R prints them.

R's matrices carry ``dimnames`` and its data frames print every column with seven significant digits; numpy and pandas do neither. The classes here are
subclasses of ``numpy.ndarray`` and ``pandas.DataFrame`` (so everything that works with those works with these) that add the names of the rows and columns
of R's matrices and R's layout:

* :class:`LabelledMatrix`: a matrix with ``rownames`` and ``colnames`` (metafor's ``G``, ``H``, ``vvc``, ``hessian``, ...), printed like ``print(matrix)``;
* :class:`VCovMat`: the result of ``vcalc()``, printed like metafor's ``print.vcovmat()``;
* :class:`EscalcFrame`: the result of ``escalc()``, printed like metafor's ``print.escalc()``.

See https://github.com/wviechtb/metafor/blob/master/R/print.vcovmat.r and https://github.com/wviechtb/metafor/blob/master/R/print.escalc.r.
"""

import collections.abc
import math
import re
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.mfopt
import metafopy.printing

__all__ = ['EscalcFrame', 'LabelledMatrix', 'NamedList', 'SparseVCovMat', 'VCovMat', 'design_matrix', 'frame_text', 'matrix_text', 'vector_text']

SIGNIF = 7


def console_width() -> int:
    return int(metafopy.printing.SETTINGS.get('width', 80))


def sig_digits(x: float) -> int:
    """The significant digits (at most seven) R needs to print ``x``."""
    text = f'{abs(x):.{SIGNIF - 1}e}'
    mantissa = text.split('e')[0].replace('.', '').rstrip('0')
    return max(len(mantissa), 1)


def decimals(values: collections.abc.Sequence[float]) -> int:
    """The number of decimals of the common fixed notation of a column of numbers (R's ``formatReal`` with seven significant digits)."""
    best = 0
    for x in values:
        if not math.isfinite(x) or x == 0:
            continue
        exponent = math.floor(math.log10(abs(float(f'{abs(x):.{SIGNIF - 1}e}'))))
        best = max(best, sig_digits(x) - 1 - exponent)
    return max(best, 0)


def cell(x: typing.Any, places: int) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)) or x is pandas.NA:
        return 'NA'
    if isinstance(x, (bool, numpy.bool_)):
        return 'TRUE' if x else 'FALSE'
    if isinstance(x, (float, numpy.floating)):
        if math.isinf(x):
            return 'Inf' if x > 0 else '-Inf'
        return f'{float(x):.{places}f}'
    return str(x)


def column(values: list[typing.Any]) -> list[str]:
    """One column of a data frame or matrix as R prints it: numbers share their decimals, everything else is shown as it is."""
    numbers = [float(v) for v in values if isinstance(v, (int, float, numpy.floating, numpy.integer)) and not isinstance(v, (bool, numpy.bool_))]
    floats = [v for v in values if isinstance(v, (float, numpy.floating))]
    places = decimals(numbers) if floats else 0
    return [
        cell(v if not isinstance(v, (int, numpy.integer)) or isinstance(v, (bool, numpy.bool_)) else float(v), places) if floats else cell(v, 0) for v in values
    ]


def row_labels(index: pandas.Index) -> list[str]:
    if isinstance(index, pandas.RangeIndex) and index.start == 0 and index.step == 1:
        return [str(i + 1) for i in range(len(index))]
    return [str(i) for i in index]


def frame_text(frame: pandas.DataFrame, formatted: collections.abc.Mapping[str, list[str]] | None = None, width: int | None = None) -> str:
    """``print(data.frame)``: right-aligned columns of seven significant digits, row names on the left (from 1), wrapped at the console width.

    ``formatted`` gives the cells of columns that are already text (``print.escalc`` formats some columns itself).
    """
    header = [str(c) for c in frame.columns]
    cols = [(formatted or {}).get(str(c)) or column(frame.iloc[:, j].tolist()) for j, c in enumerate(frame.columns)]
    rows = [[cols[j][i] for j in range(len(cols))] for i in range(frame.shape[0])]
    return metafopy.printing.fmt_wrapped_table(header, rows, row_labels(frame.index), width or console_width())


def matrix_text(
    values: numpy.typing.NDArray[typing.Any], rownames: collections.abc.Sequence[str] | None = None, colnames: collections.abc.Sequence[str] | None = None
) -> str:
    """``print(matrix)``: without names the rows are ``[1,]`` and the columns ``[,1]``; every column has its own decimals."""
    arr = numpy.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = list(rownames) if rownames else [f'[{i + 1},]' for i in range(arr.shape[0])]
    cols = list(colnames) if colnames else [f'[,{j + 1}]' for j in range(arr.shape[1])]
    cells = [column(arr[:, j].tolist()) for j in range(arr.shape[1])]
    body = [[cells[j][i] for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    return metafopy.printing.fmt_wrapped_table(cols, body, rows, console_width())


def vector_text(series: pandas.Series) -> str:
    """``print(named vector)``: the names above the values, right-aligned in common columns."""
    places = decimals([float(v) for v in series.tolist() if isinstance(v, (int, float, numpy.floating, numpy.integer))])
    cells = [cell(float(v) if isinstance(v, (int, float, numpy.floating, numpy.integer)) and not isinstance(v, bool) else v, places) for v in series.tolist()]
    names = [str(n) for n in series.index]
    w = max(max(len(c) for c in cells), max(len(n) for n in names))
    per_line = max(1, console_width() // (w + 1))
    lines = []
    for start in range(0, len(cells), per_line):
        lines.append(' '.join(n.rjust(w) for n in names[start : start + per_line]))
        lines.append(' '.join(c.rjust(w) for c in cells[start : start + per_line]))
    return '\n'.join(lines)


class LabelledMatrix(numpy.ndarray):
    """A numpy matrix with the names of its rows and columns (R's ``dimnames``); it prints as ``print(matrix)`` does.

    Slices and results of arithmetic are plain arrays unless they keep the shape.
    """

    rownames: tuple[str, ...] | None
    colnames: tuple[str, ...] | None

    def __new__(
        cls,
        data: numpy.typing.ArrayLike,
        rownames: collections.abc.Sequence[str] | None = None,
        colnames: collections.abc.Sequence[str] | None = None,
    ) -> typing.Self:
        obj = numpy.asarray(data, dtype=numpy.float64).view(cls)
        obj.rownames = None if rownames is None else tuple(str(n) for n in rownames)
        obj.colnames = None if colnames is None else tuple(str(n) for n in colnames)
        return obj

    def __array_finalize__(self, obj: typing.Any) -> None:
        rows = getattr(obj, 'rownames', None)
        cols = getattr(obj, 'colnames', None)
        shape = getattr(obj, 'shape', None)
        keep = shape is not None and shape == self.shape
        self.rownames = rows if keep else None
        self.colnames = cols if keep else None

    def __array_wrap__(self, array: numpy.ndarray, context: typing.Any = None, return_scalar: bool = False) -> typing.Any:
        # a reduction gives a number and a result of another shape a plain array, as for any array; only matrices stay matrices
        if return_scalar or array.ndim == 0:
            return array[()]
        if array.ndim != 2:
            return numpy.asarray(array)
        return super().__array_wrap__(array, context, return_scalar)

    def __getitem__(self, key: typing.Any) -> typing.Any:
        out = super().__getitem__(key)
        if isinstance(out, LabelledMatrix) and out.ndim != 2:
            return numpy.asarray(out)
        return out

    @property
    def dimnames(self) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
        """The row and column names, ``None`` for a dimension without names."""
        return self.rownames, self.colnames

    def transpose(self, *axes: typing.Any) -> typing.Self:
        """The transpose; the names of the rows and columns change places."""
        out: typing.Self = numpy.ndarray.transpose(self, *axes)
        if isinstance(out, LabelledMatrix) and self.ndim == 2 and not axes:
            out.rownames, out.colnames = self.colnames, self.rownames
        return out

    @property
    def T(self) -> typing.Self:
        """The transpose, see :meth:`transpose`."""
        return self.transpose()

    def to_frame(self) -> pandas.DataFrame:
        """The matrix as a data frame with its names."""
        return pandas.DataFrame(numpy.asarray(self), index=self.rownames, columns=self.colnames)

    def __reduce__(self) -> tuple[typing.Any, ...]:
        return (LabelledMatrix, (numpy.asarray(self), self.rownames, self.colnames))

    def __str__(self) -> str:
        if self.ndim != 2:
            return numpy.ndarray.__str__(self)
        return matrix_text(numpy.asarray(self), self.rownames, self.colnames)

    def __repr__(self) -> str:
        return self.__str__()


def design_matrix(
    values: numpy.typing.NDArray[typing.Any],
    rownames: collections.abc.Sequence[str] | None,
    colnames: collections.abc.Sequence[str] | None,
) -> numpy.typing.NDArray[typing.Any]:
    """``values`` with the row and column names of R's model matrix; the names that do not fit the shape are left out."""
    arr = numpy.asarray(values, dtype=numpy.float64)
    if arr.ndim != 2:
        return arr
    rows = rownames if rownames is not None and len(rownames) == arr.shape[0] else None
    cols = colnames if colnames is not None and len(colnames) == arr.shape[1] else None
    return LabelledMatrix(arr, rows, cols)


class VCovMat(LabelledMatrix):
    """The var-cov matrix of ``vcalc()``, printed like metafor's ``print.vcovmat()``: fixed decimals, zeros as ``.``, rows and columns numbered from 1."""

    def __new__(
        cls,
        data: numpy.typing.ArrayLike,
        rownames: collections.abc.Sequence[str] | None = None,
        colnames: collections.abc.Sequence[str] | None = None,
    ) -> typing.Self:
        return super().__new__(cls, data, rownames, colnames)

    def __reduce__(self) -> tuple[typing.Any, ...]:
        return (VCovMat, (numpy.asarray(self), self.rownames, self.colnames))

    def text(self, digits: int = 4, zero: str = '.', na: str = 'NA') -> str:
        """The matrix as ``print.vcovmat`` shows it."""
        x = numpy.asarray(self, dtype=numpy.float64)
        if x.ndim != 2 or 0 in x.shape:
            return f'< table of extent {" x ".join(str(d) for d in x.shape)} >'
        trimmed = [[(' ' if v >= 0 else '') + trim_zeros(f'{v:.{digits}f}', digits) if math.isfinite(v) else 'NA' for v in row] for row in x]
        longest = max((len(c) for row in trimmed for c in row), default=0)
        digits = max(0, min(digits, longest - 3))
        negative = bool(numpy.any(x[numpy.isfinite(x)] < 0))
        body = []
        for row in x:
            cells = []
            for v in row:
                if math.isnan(v):
                    cells.append(na)
                elif abs(v) <= 10 * numpy.finfo(float).eps:
                    cells.append(zero)
                else:
                    text = f'{v:.{digits}f}' if math.isfinite(v) else ('Inf' if v > 0 else '-Inf')
                    cells.append((' ' + text) if negative and v >= 0 else text)
            body.append(cells)
        cols = list(self.colnames) if self.colnames else [str(j + 1) for j in range(x.shape[1])]
        rows = list(self.rownames) if self.rownames else [str(i + 1) for i in range(x.shape[0])]
        return metafopy.printing.fmt_wrapped_table(cols, body, rows, console_width())

    def __str__(self) -> str:
        if self.ndim != 2:
            return numpy.ndarray.__str__(self)
        return '\n' + self.text() + '\n'


def trim_zeros(text: str, digits: int) -> str:
    """``formatC(..., drop0trailing=TRUE)``."""
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


FRAME_DIGITS = ('est', 'se', 'test', 'pval', 'ci', 'var', 'sevar', 'fit', 'het')


class EscalcFrame(pandas.DataFrame):
    """The data frame ``escalc()`` returns; it prints as metafor's ``print.escalc()`` does (columns of the effect sizes and variances with ``digits``
    decimals, the other columns as ``print.data.frame`` shows them, rows numbered from 1).

    Selecting rows or columns gives a plain data frame that keeps ``attrs``.
    """

    @property
    def _constructor(self) -> typing.Callable[..., pandas.DataFrame]:
        return pandas.DataFrame

    def digit_for(self, name: str) -> int:
        """The decimals of one kind of column (``est``, ``var``, ``se``, ``test``, ``ci``): the ``digits`` of the frame, else the ``digits`` option."""
        digits = self.attrs.get('digits')
        if isinstance(digits, collections.abc.Mapping) and name in digits:
            return metafopy.mfopt.reduce_digits(digits[name])
        return metafopy.mfopt.option_digits(digits)

    def text(self) -> str:
        """The frame as ``print.escalc`` shows it."""
        formatted: dict[str, list[str]] = {}
        for key, part in (('yi_names', 'est'), ('vi_names', 'var'), ('sei_names', 'se'), ('zi_names', 'test'), ('ci_lb_names', 'ci'), ('ci_ub_names', 'ci')):
            for name in self.attrs.get(key, []) or []:
                if name in self.columns:
                    places = self.digit_for(part)
                    formatted[str(name)] = [cell(float(v), places) if v is not None else 'NA' for v in self[name].tolist()]
        return frame_text(pandas.DataFrame(self), formatted)

    def __str__(self) -> str:
        if self.ndim != 2:
            return numpy.ndarray.__str__(self)
        return '\n' + self.text() + '\n'

    def __repr__(self) -> str:
        return self.__str__()


SYNTACTIC = re.compile(r'[A-Za-z.][A-Za-z0-9._]*')


def element_text(value: typing.Any) -> str:
    """One element of a list as R prints it: a data frame or matrix in its layout, a number as ``[1] x``."""
    if isinstance(value, pandas.DataFrame):
        return frame_text(value)
    if isinstance(value, LabelledMatrix):
        return str(value).strip('\n') if isinstance(value, VCovMat) else matrix_text(value, value.rownames, value.colnames)
    if isinstance(value, numpy.ndarray) and value.ndim == 2:
        return matrix_text(value, getattr(value, 'rownames', None), getattr(value, 'colnames', None))
    if isinstance(value, (int, float, numpy.integer, numpy.floating)) and not isinstance(value, bool):
        return '[1] ' + cell(float(value), decimals([float(value)]))
    return str(value)


class NamedList(dict[typing.Any, typing.Any]):
    """A dict that prints as R prints a named list: ``$name`` above each element (names that are not syntactic in backticks)."""

    def __str__(self) -> str:
        blocks = []
        for name, value in self.items():
            label = str(name) if SYNTACTIC.fullmatch(str(name)) else f'`{name}`'
            blocks.append(f'${label}\n{element_text(value)}\n')
        return '\n'.join(blocks)

    def __repr__(self) -> str:
        return self.__str__()


class SparseVCovMat(scipy.sparse.csr_array):
    """The sparse var-cov matrix of ``vcalc(sparse=True)``; it prints as R prints a ``dgCMatrix`` (every column with its own decimals, zeros as ``.``)."""

    def text(self) -> str:
        """The matrix as ``print`` of a ``Matrix::dgCMatrix`` shows it."""
        dense = self.toarray()
        n, m = dense.shape
        columns = []
        for j in range(m):
            nonzero = dense[:, j] != 0
            places = decimals([float(v) for v in dense[nonzero, j]])
            cells = [cell(float(v), places) if flag else '.' for v, flag in zip(dense[:, j], nonzero, strict=True)]
            width = max((len(c) for c in cells), default=0)
            columns.append([c.ljust(width) for c in cells])
        rows = [[columns[j][i] for j in range(m)] for i in range(n)]
        body = metafopy.printing.fmt_wrapped_table([''] * m, rows, [f'[{i + 1},]' for i in range(n)], console_width())
        return f'{n} x {m} sparse Matrix of class "dgCMatrix"\n{body}'

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return self.text()
