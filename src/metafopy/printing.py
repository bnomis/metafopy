"""Text formatting helpers that reproduce metafor's printed output.

metafor formats numbers with fixed decimals (``digits``, default 4), prints
small p-values as ``<.0001`` and appends significance stars to coefficient
tables. These helpers keep that behaviour in one place so every result class
prints the way R users expect.
"""

import collections.abc
import math
import sys

import numpy
import numpy.typing

import metafopy.errors

__all__ = [
    'SIGNIF_LEGEND',
    'drop_stars',
    'fmt_df',
    'fmt_df_scalar',
    'fmt_named_vector',
    'fmt_num',
    'fmt_pval',
    'fmt_table',
    'fmt_wrapped_table',
    'signif_footer',
    'signif_stars',
    'zero_below',
]

SIGNIF_LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"


def fmt_num(x: float, digits: int = 4) -> str:
    """Fixed-point format as R's ``formatC(x, format="f", digits=digits)``."""
    if math.isnan(x):
        return 'NA'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    out = f'{x:.{digits}f}'
    # R never prints a negative zero
    if out.startswith('-') and float(out) == 0.0:
        out = out[1:]
    return out


def fmt_pval(p: float, digits: int = 4) -> str:
    """Format a p-value; values below ``10**-digits`` print as ``<.0001``."""
    if math.isnan(p):
        return 'NA'
    threshold = 10.0**-digits
    if p < threshold:
        return '<.' + '0' * (digits - 1) + '1'
    return fmt_num(p, digits)


def signif_stars(p: float) -> str:
    """R's default significance codes (``symnum`` with right-closed intervals): ``***`` <= .001 < ``**`` <= .01 < ``*`` <= .05 < ``.`` <= .1."""
    if math.isnan(p):
        return ''
    for cut, star in ((0.001, '***'), (0.01, '**'), (0.05, '*'), (0.1, '.')):
        if p <= cut:
            return star
    return ''


def drop_stars(header: list[str], rows: list[list[str]], signif_stars: bool) -> tuple[list[str], list[list[str]]]:
    """Without ``signif_stars`` the last column (the significance stars, header ``''``) of a coefficient table is left out."""
    if signif_stars:
        return header, rows
    return header[:-1], [row[:-1] for row in rows]


def signif_footer(signif_legend: bool, extra: collections.abc.Sequence[str] = ()) -> list[str]:
    """The lines after a coefficient table as ``print.rma.uni`` writes them: ``---`` and the ``Signif. codes`` legend when ``signif_legend``,
    then the further legend lines ``extra`` (the cluster-robust footnote); the table is followed by one blank line only when there is no legend."""
    out: list[str] = ['', '---'] if signif_legend or extra else []
    if signif_legend:
        out.append(SIGNIF_LEGEND)
        if extra:
            out.append('')
    return [*out, *extra, '']


def fmt_table(
    header: collections.abc.Sequence[str],
    rows: collections.abc.Sequence[collections.abc.Sequence[str]],
    row_names: collections.abc.Sequence[str] | None = None,
    gap: int = 2,
) -> str:
    """Lay out a right-aligned text table, with optional left-aligned row names; ``gap`` spaces separate the columns.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    columns = [list(header), *[list(r) for r in rows]]
    ncol = len(header)
    if any(len(r) != ncol for r in columns):
        raise metafopy.errors.InputError('every row must have as many cells as the header')
    widths = [max(len(r[j]) for r in columns) for j in range(ncol)]
    names = ['', *row_names] if row_names is not None else None
    name_w = max(len(n) for n in names) if names else 0
    lines: list[str] = []
    for i, row in enumerate(columns):
        cells = (' ' * gap).join(cell.rjust(widths[j]) for j, cell in enumerate(row))
        if names is not None:
            cells = names[i].ljust(name_w) + ' ' * gap + cells
        lines.append(cells.rstrip())
    return '\n'.join(lines)


def fmt_wrapped_table(
    header: collections.abc.Sequence[str],
    rows: collections.abc.Sequence[collections.abc.Sequence[str]],
    row_names: collections.abc.Sequence[str],
    width: int = 80,
) -> str:
    """A right-aligned character matrix as R prints it: one space between columns, left-aligned row names and columns that do not fit
    in ``width`` characters continued in a further block below."""
    ncol = len(header)
    widths = [max(len(header[j]), *(len(r[j]) for r in rows)) for j in range(ncol)]
    name_w = max(len(n) for n in row_names)
    blocks: list[list[int]] = []
    used = name_w
    for j in range(ncol):
        if not blocks or used + 1 + widths[j] > width:
            blocks.append([])
            used = name_w
        blocks[-1].append(j)
        used += 1 + widths[j]
    out: list[str] = []
    for block in blocks:
        out.append(' ' * name_w + ''.join(' ' + header[j].rjust(widths[j]) for j in block))
        out += [name.ljust(name_w) + ''.join(' ' + row[j].rjust(widths[j]) for j in block) for name, row in zip(row_names, rows, strict=True)]
    return '\n'.join(line.rstrip() for line in out)


def fmt_named_vector(names: collections.abc.Sequence[str], values: collections.abc.Sequence[str]) -> str:
    """A named character vector as R prints it with ``print(x, quote=FALSE, print.gap=2)``: every column has the same width."""
    width = max(len(s) for s in [*names, *values])
    return '  '.join(n.rjust(width) for n in names) + '\n' + '  '.join(v.rjust(width) for v in values)


def zero_below(x: float, digits: int = 4, thresh: float = 10 * sys.float_info.epsilon) -> str:
    """``fmt_num`` that prints values with ``|x| < thresh`` as ``0`` (metafor's ``fmtx(x, digits, thresh=)``)."""
    return '0' if abs(x) < thresh else fmt_num(x, digits)


def fmt_df(values: numpy.typing.NDArray[numpy.float64]) -> list[str]:
    """A numeric column as ``data.frame`` printing does (common number of decimals); degrees of freedom are rounded to 2 decimals.
    Infinite and missing values print as ``Inf``, ``-Inf`` and ``NA``."""
    finite = [round(float(v), 2) for v in values if math.isfinite(float(v))]
    decimals = 0
    for v in finite:
        text = f'{v:.2f}'.rstrip('0').rstrip('.')
        if '.' in text:
            decimals = max(decimals, len(text.split('.')[1]))
    out: list[str] = []
    for v in values:
        x = float(v)
        if math.isnan(x):
            out.append('NA')
        elif math.isinf(x):
            out.append('Inf' if x > 0 else '-Inf')
        else:
            out.append(f'{round(x, 2):.{decimals}f}')
    return out


def fmt_df_scalar(value: float | None) -> str:
    if value is None or math.isnan(value):
        return 'NA'
    if math.isinf(value):
        return 'Inf' if value > 0 else '-Inf'
    return str(int(value)) if float(value).is_integer() else f'{round(float(value), 2):g}'
