"""The extra lines and footnote marks of printed cluster-robust results (``print.rma.uni`` and ``print.rma.mv`` for ``robust.rma``)."""

import statistics

import numpy
import numpy.typing

import metafopy.models.robust_info
import metafopy.printing

__all__ = ['FOOT_LEGEND', 'FOOT_MARK', 'counts_lines', 'decorate', 'df_cells', 'legend_lines']

FOOT_MARK = '¹'
FOOT_LEGEND = '1)'


def counts_lines(info: metafopy.models.robust_info.RobustInfo, k: int) -> list[str]:
    """The ``Number of estimates / clusters / Estimates per cluster`` block."""
    tcl = info.tcl
    if bool(numpy.all(tcl == tcl[0])):
        per = str(int(tcl[0]))
    else:
        median = statistics.median(int(v) for v in tcl)
        per = f'{int(tcl.min())}-{int(tcl.max())} (mean: {metafopy.printing.fmt_num(float(tcl.mean()), 2)}, median: {median:g})'
    return [f'Number of estimates:   {k}', f'Number of clusters:    {info.n}', f'Estimates per cluster: {per}', '']


def decorate(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Mark every column but the first with the footnote symbol and pad the cells (``.addfootsym(res.table, 2:7, ...)``)."""
    head = [h if j == 0 or h == '' else h + FOOT_MARK for j, h in enumerate(header)]
    body = [[c if j == 0 or j == len(row) - 1 else c + ' ' for j, c in enumerate(row)] for row in rows]
    return head, body


def df_cells(dfs: numpy.typing.NDArray[numpy.float64]) -> list[str]:
    return [metafopy.printing.fmt_df_scalar(float(v)) for v in dfs]


def legend_lines(info: metafopy.models.robust_info.RobustInfo, int_only: bool) -> list[str]:
    """The footnote explaining the cluster-robust inference."""
    head = f'{FOOT_LEGEND} results based on cluster-robust inference (var-cov estimator: {info.vbest}'
    what = 't-test and confidence interval' if int_only else 't/F-tests and confidence intervals'
    if info.robumethod == 'default':
        return [head + ',', f'   approx {what}, df: residual method)']
    if info.coef_test == 'Satterthwaite' and info.conf_test == 'Satterthwaite' and info.wald_test == 'HTZ':
        return [head + ',', f'   approx {what}, df: Satterthwaite approx)']
    return [head + ')']
