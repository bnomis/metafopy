"""Progress output of ``verbose=`` arguments, written where R writes it: ``cat()`` to stdout and ``message()`` to stderr.

metafor prints iteration logs with ``cat(...)`` and progress notes with ``message(...)``; these helpers keep that split.
"""

import sys

import metafopy.printing

__all__ = ['cat', 'iteration_line', 'message']


def cat(text: str) -> None:
    """R's ``cat(text)``: ``text`` as is, on stdout."""
    sys.stdout.write(text)
    sys.stdout.flush()


def message(text: str) -> None:
    """R's ``message(text)``: ``text`` and a newline, on stderr."""
    sys.stderr.write(f'{text}\n')
    sys.stderr.flush()


def iteration_line(iteration: int, tau2: float, digits: int = 4) -> str:
    """metafor's ``paste("Iteration", formatC(iter, width=5, flag="-", format="f", digits=0), "tau^2 =", fmtx(tau2, digits), "\\n")``."""
    return f'Iteration {iteration:<5d} tau^2 = {metafopy.printing.fmt_num(tau2, digits)} \n'


def binary_start(verbose: int, subset: object) -> None:
    """The messages ``rma.mh()`` and ``rma.peto()`` print before the tables are processed."""
    if not verbose:
        return
    cat('\n')
    message('Extracting the data and computing yi/vi values ...')
    message('Generating/extracting the study labels ...')
    if subset is not None:
        message('Subsetting ...')


def binary_end(verbose: int, tables_na: bool, yivi_na: bool) -> None:
    """The messages ``rma.mh()`` and ``rma.peto()`` print from the missing-value handling to the output."""
    if not verbose:
        return
    if tables_na:
        message('Handling NAs in the table data ...')
    if yivi_na:
        message('Handling NAs in yi/vi ...')
    message('Model fitting ...')
    message('Heterogeneity testing ...')
    message('Computing fit statistics and log-likelihood ...')
    message('Preparing the output ...')
    cat('\n')
