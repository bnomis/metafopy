"""Serial or process-pool maps with metafor's ``parallel``, ``ncpus``, ``cl`` and ``progbar`` arguments.

metafor's ``vif()`` and ``profile()`` take ``parallel = "no" | "snow" | "multicore"``, ``ncpus`` and ``cl`` (a cluster) and show a progress bar
(``pbapply``). Here ``'snow'`` starts a pool of fresh worker processes and ``'multicore'`` forks them (where the platform allows it), both via
:class:`concurrent.futures.ProcessPoolExecutor`; ``cl`` is any :class:`concurrent.futures.Executor`. The results are returned in the order of the
inputs, so a deterministic function gives the same output as the serial run. See
https://github.com/wviechtb/metafor/blob/master/R/profile.rma.uni.selmodel.r and https://github.com/wviechtb/metafor/blob/master/R/vif.rma.r.
"""

import collections.abc
import concurrent.futures
import multiprocessing
import sys
import typing

import metafopy.errors

__all__ = ['Parallel', 'map_sims', 'progress', 'resolve_parallel', 'tick']

# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type Parallel = typing.Literal['no', 'snow', 'multicore']


def resolve_parallel(parallel: str, ncpus: int, cl: concurrent.futures.Executor | None) -> Parallel:
    """metafor's handling of ``parallel``, ``ncpus`` and ``cl``: ``'no'`` unless more than one worker (or an executor) is available.

    Raises:
        metafopy.errors.InputError: ``parallel`` is not ``'no'``, ``'snow'`` or ``'multicore'`` or ``ncpus`` is below 1.
    """
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if parallel not in ('no', 'snow', 'multicore'):
        raise metafopy.errors.InputError(f'unknown parallel option {parallel!r}')
    if ncpus < 1:
        raise metafopy.errors.InputError("Argument 'ncpus' must be >= 1.")
    if isinstance(cl, concurrent.futures.Executor):  # as R, which only takes a SOCKcluster for ``cl`` and ignores any other value
        return 'snow'
    if parallel == 'no' and ncpus > 1:
        return 'snow'
    if parallel != 'no' and ncpus < 2:
        return 'no'
    return typing.cast(Parallel, parallel)


def map_sims[T, V](
    one: collections.abc.Callable[[T], V],
    items: collections.abc.Sequence[T],
    *,
    parallel: Parallel,
    ncpus: int,
    cl: concurrent.futures.Executor | None,
    progbar: bool,
) -> list[V]:
    """``one`` applied to each item, serially or in a process pool (or on the executor ``cl``); the order of the results is that of ``items``.

    Shared by the functions with metafor's ``parallel``, ``ncpus``, ``cl`` and ``progbar`` arguments (``vif``, ``profile``).
    """
    if parallel == 'no':
        return list(progress(map(one, items), len(items), progbar))
    # 'multicore' forks the workers (as R's mclapply); 'snow' starts fresh ones
    ctx = multiprocessing.get_context('fork' if parallel == 'multicore' and 'fork' in multiprocessing.get_all_start_methods() else None)
    if isinstance(cl, concurrent.futures.Executor):
        return list(progress(cl.map(one, items), len(items), progbar))
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus, mp_context=ctx) as pool:
        return list(progress(pool.map(one, items), len(items), progbar))


def tick(done: int, total: int, show: bool) -> None:
    """The progress bar after ``done`` of ``total`` steps, written to stderr (for loops that :func:`progress` cannot wrap); a no-op unless ``show``."""
    if show and total > 0:
        print(f'\r  |{"+" * (50 * done // total):<50}| {100 * done // total:3d}%', end='' if done < total else '\n', file=sys.stderr)


def progress[V](rows: collections.abc.Iterable[V], total: int, show: bool) -> collections.abc.Iterator[V]:
    """``rows`` as they come, with a progress bar on stderr (``show``) that counts them against ``total``."""
    for i, row in enumerate(rows, start=1):
        tick(i, total, show)
        yield row
