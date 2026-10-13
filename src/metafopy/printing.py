"""Text formatting helpers that reproduce metafor's printed output.

metafor formats numbers with fixed decimals (``digits``, default 4), prints
small p-values as ``<.0001`` and appends significance stars to coefficient
tables. These helpers keep that behaviour in one place so every result class
prints the way R users expect.
"""

import collections.abc
import math
import sys

import metafopy.errors

__all__ = ['fmt_named_vector', 'fmt_num', 'fmt_pval', 'fmt_table', 'signif_stars', 'zero_below']


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


def fmt_table(
    header: collections.abc.Sequence[str],
    rows: collections.abc.Sequence[collections.abc.Sequence[str]],
    row_names: collections.abc.Sequence[str] | None = None,
) -> str:
    """Lay out a right-aligned text table, with optional left-aligned row names.

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
        cells = '  '.join(cell.rjust(widths[j]) for j, cell in enumerate(row))
        if names is not None:
            cells = names[i].ljust(name_w) + '  ' + cells
        lines.append(cells.rstrip())
    return '\n'.join(lines)


def fmt_named_vector(names: collections.abc.Sequence[str], values: collections.abc.Sequence[str]) -> str:
    """A named character vector as R prints it with ``print(x, quote=FALSE, print.gap=2)``: every column has the same width."""
    width = max(len(s) for s in [*names, *values])
    return '  '.join(n.rjust(width) for n in names) + '\n' + '  '.join(v.rjust(width) for v in values)


def zero_below(x: float, digits: int = 4, thresh: float = 10 * sys.float_info.epsilon) -> str:
    """``fmt_num`` that prints values with ``|x| < thresh`` as ``0`` (metafor's ``fmtx(x, digits, thresh=)``)."""
    return '0' if abs(x) < thresh else fmt_num(x, digits)
