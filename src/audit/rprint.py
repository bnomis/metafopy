"""R's layout of data frames, matrices, named vectors and lists, so that the audit compares what R prints with what a Python user sees in the same place.

pandas truncates wide frames, numbers rows from 0 and gives every column its own precision; R prints all columns, numbers rows from 1 and gives each
column seven significant digits. The printed forms of metafopy's own result classes are compared as they are; only the plain pandas and numpy values
that stand for R's data frames, matrices and vectors are laid out here.
"""

import math
import typing

import numpy
import pandas

import metafopy.printing

SIGNIF = 7


def sig_digits(x: float) -> int:
    """The significant digits (at most seven) R needs to print ``x``."""
    text = f'{abs(x):.{SIGNIF - 1}e}'
    mantissa = text.split('e')[0].replace('.', '').rstrip('0')
    return max(len(mantissa), 1)


def decimals(values: list[float]) -> int:
    """The number of decimals of the common fixed notation of a column (R's ``formatReal``)."""
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


def frame_text(frame: pandas.DataFrame, width: int = 80) -> str:
    """``print(data.frame)``: right-aligned columns of seven significant digits, row names on the left, wrapped at ``width``."""
    header = [str(c) for c in frame.columns]
    cols = [column(frame.iloc[:, j].tolist()) for j in range(frame.shape[1])]
    rows = [[cols[j][i] for j in range(len(cols))] for i in range(frame.shape[0])]
    return metafopy.printing.fmt_table(header, rows, row_names=row_labels(frame.index), gap=1, width=width + 1)


def matrix_text(values: numpy.ndarray, width: int = 80) -> str:
    """``print(matrix)`` for a numeric array without dimnames: columns ``[,1]``, ``[,2]``, ... (R labels the columns of an unnamed matrix so)."""
    if values.ndim == 1:
        return frame_text(pandas.DataFrame({'': values}), width)
    frame = pandas.DataFrame(values, columns=[f'[,{j + 1}]' for j in range(values.shape[1])])
    frame.index = pandas.Index([f'[{i + 1},]' for i in range(values.shape[0])])
    return frame_text(frame, width)


def vector_text(series: pandas.Series) -> str:
    """``print(named vector)``: the names above the values, right-aligned in common columns."""
    places = decimals([float(v) for v in series.tolist() if isinstance(v, (int, float, numpy.floating, numpy.integer))])
    cells = [cell(float(v) if isinstance(v, (int, float, numpy.floating, numpy.integer)) and not isinstance(v, bool) else v, places) for v in series.tolist()]
    names = [str(n) for n in series.index]
    w = max(max(len(c) for c in cells), max(len(n) for n in names))
    per_line = max(1, 80 // (w + 1))
    lines = []
    for start in range(0, len(cells), per_line):
        lines.append(' '.join(n.rjust(w) for n in names[start : start + per_line]))
        lines.append(' '.join(c.rjust(w) for c in cells[start : start + per_line]))
    return '\n'.join(lines)
