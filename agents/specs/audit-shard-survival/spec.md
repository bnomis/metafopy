# Audit shard survival

**Version:** 1.0
**Status:** implemented
**Date:** 2026-11-02

The audit harness runs its cases in shards (slices of the case list): first
each shard in R, then each in Python against R's outcome. An R shard already
survives its process dying: the case it was running is recorded as an R
failure and the shard restarts on the cases left. A Python shard does not.
This spec makes Python shards survive the same way, and makes the harness
kill a shard of either language whose memory passes a limit, recording that
case as a finding and carrying on.

## Motivation

From the `TODO.md` item "A Python audit worker that dies stops the whole audit
step", found on 2026-11-02 while fixing the lost worker in the `options` step:

- `audit.engine.run_cases` runs the Python shards in a
  `concurrent.futures.ProcessPoolExecutor`. One worker killed by the system
  (out of memory, a crash in compiled code) raises `BrokenProcessPool`, ends
  the step, and no shard writes its results.
- On macOS 27, `setrlimit(RLIMIT_AS)` is refused (`audit.config.limit_memory`
  returns on `ValueError`), so neither Python nor R shards have a memory cap.
  A case that grows without bound kills its worker instead of raising
  `MemoryError`. The case that did so, `permutest(exact=True)`, grew past
  26 GB and is fixed (`9ee24d4`); nothing stops the next one.
- An R shard whose process dies has its case recorded with the error
  `R process died on this case`. `pyrun.classify` then reports that case as
  `accepts-where-R-refuses` whenever Python succeeded, which blames metafopy
  for R's crash.

Decisions taken with the operator on 2026-11-02:

1. The parent process watches memory, by polling each shard's resident size
   with `ps`, not a thread inside each worker. One watcher covers both
   languages and needs no new dependency; a thread could not interrupt
   compiled code that holds the GIL.
2. The limit applies to R shards as well as Python shards.
3. A killed or dead Python process gives the case a `memory` or
   `process-died` finding, reported under *Crashes*. An R process killed or
   dead gives an `r-killed` finding, in its own report section, never
   `accepts-where-R-refuses`.
4. The limit stays `config.MEMORY_LIMIT` (4 GiB), now read as resident size.

## Module location

- `src/audit/engine.py` - the shard loop, the watcher, the Python shard
  process
- `src/audit/pyrun.py` - the Python shard's entry point and streamed output;
  `classify` for `r-killed`
- `src/audit/config.py` - `MEMORY_LIMIT` docstring, `POLL_SECONDS`
- `src/audit/report.py` - the new kinds in `SECTIONS`
- `src/audit/README.md` - how a dead or oversized shard is reported
- `src/tests/test_harness_engine.py` - tests

## API

```python
# audit/engine.py

Stop = typing.Literal['died', 'memory', 'stalled']


def resident_bytes(pid: int) -> int | None:
    """The resident size of process ``pid`` in bytes (``ps -o rss= -p <pid>``, which reports KiB), or None when the process has gone."""


def watch(proc: subprocess.Popen[bytes], out: pathlib.Path, stall: float, limit: int) -> Stop | None:
    """Wait for a shard process. Kill it when its output has not moved for ``stall`` seconds ('stalled') or its resident size passes ``limit``
    bytes ('memory'). A process that exits non-zero by itself is 'died'. None when it finished cleanly."""


# auditme: accept[spec-api-signature-drift] this spec records the cases path as built; scripts-shard-survival adds `write` and the parts to `dead`
def run_shard(
    work: pathlib.Path,
    name: str,
    shard: list[audit.cases.Case],
    timeout: float,
    start: collections.abc.Callable[[pathlib.Path, pathlib.Path], subprocess.Popen[bytes]],
    dead: collections.abc.Callable[[str, Stop], Json],
) -> list[Json]:
    """Run ``shard`` in processes made by ``start(cases_path, out_path)``, restarting on the cases left whenever one stops early. The case running
    when a process stopped gets ``dead(case_id, stop)`` as its result. Returns one result per finished case, in shard order."""


def run_r_shard(work: pathlib.Path, i: int, fixtures_path: pathlib.Path, shard: list[audit.cases.Case], timeout: float) -> None:
    """R side of shard ``i``; writes ``r-{i}.json`` as today."""


def run_py_shard(work: pathlib.Path, i: int, fixtures_path: pathlib.Path, shard: list[audit.cases.Case], timeout: float) -> None:
    """Python side of shard ``i``; writes ``py-{i}.json`` in today's shape."""


# audit/pyrun.py


def run_stream(fixtures_path: pathlib.Path, cases_path: pathlib.Path, r_path: pathlib.Path, out_path: pathlib.Path, timeout: float) -> None:
    """Run the cases in ``cases_path`` and append to ``out_path`` a ``{"start": id}`` line before each case and its result line after it,
    flushed each time, as ``runcases.R`` does."""


def r_killed(error: str) -> bool:
    """Whether R's error records its process being killed for memory or dying (``engine.run_shard``), not R refusing the call."""
```

`pyrun` gains `python -m audit.pyrun <job.json>`, where the job file holds the
arguments of `run_stream`. `run_shard(job)` and its tuple job are removed with
the pool.

## Implementation

### One shard loop for both languages

`run_r_shard`'s restart loop becomes `run_shard`, shared by both languages:

1. Write the remaining cases to `<name>-rest.json`; `start` launches the
   process on it, writing JSON lines to `<name>.jsonl`.
2. `watch` waits; `read_outcomes` (unchanged) gives the finished cases and the
   case running when it stopped.
3. If a case was running, it gets `dead(case_id, stop)`. A process that
   stopped with no case running (it died before its first case, for example
   building fixtures) gives each remaining case `dead(case_id, stop)` and the
   loop ends, so a broken environment cannot loop.
4. Rename the output to `<name>-attempt<k>.jsonl` and go round again on the
   remaining cases, at most `len(shard)` times, as today.

The R reasons keep today's timeout text, so `pyrun.r_timed_out` and the ledger
are unchanged: `stalled` gives `R did not return within <s>s (timeout)`;
`memory` gives `R passed the memory limit of <n> GiB (killed)`; `died` gives
`R process died on this case (killed)`. `pyrun.r_killed` matches the
`(killed)` ending.

The Python `dead` result is `{"id", "r_status", "py_status": "error",
"py_error", "findings": [one finding]}` with kind `memory`, `process-died`
or, for `stalled`, `timeout` (compiled code ignores the in-process
`SIGALRM`). Its detail names the limit or the exit status.

### The Python shard process

`run_py_shard` starts `sys.executable -m audit.pyrun <job.json>`, with the
inherited environment, so `PYTHONPATH` from `bin/activate.sh` and
`--metafopy-src` reach it as they reach the pool's workers today. The process
calls `audit.config.limit_memory()`, builds the fixtures, and seeds 1234
before each case, as `run_shard` does today, so a restarted process gives each
case the result it would have had. Its stdout and stderr go to
`py-{i}.log`. When the loop ends, `py-{i}.json` is written as
`{"results": [...]}` in shard order, so `run_cases`' merge is unchanged.

`run_cases` runs the Python shards in a `ThreadPoolExecutor`, as it already
runs the R shards; each thread drives one process.

### The memory watch

`watch` replaces `wait_or_kill`. Every `POLL_SECONDS` it checks the process
has not exited, the output's size (stall, as today) and `resident_bytes(pid)`.
Past `limit` it sends `SIGKILL` and returns `'memory'`. The resident size of
the shard process alone is read: `Rscript` runs R in the same process, and the
Python shard does not fork. A spike between two polls can still reach the
system's out-of-memory killer; the process then dies, and the case is
`died`, not lost.

### Classification and report

- `pyrun.classify`: when R failed with `r_killed(error)`, the finding is
  `r-killed` with R's error as detail, whatever Python did. This goes before
  the `accepts-where-R-refuses` branch.
- `report.SECTIONS`: `memory` and `process-died` join *Crashes*; a new
  section `('R processes killed', ('r-killed',))` follows *Timeouts*.
- `README.md`: a section on how a dead or oversized shard is reported, beside
  "R calls slower than the case limit".

## Error handling

- A dead or killed process is a finding, never an exception: `run_cases` no
  longer raises `BrokenProcessPool`, and every case gets a result.
- `resident_bytes` returns None when `ps` fails or prints nothing (the
  process has gone); `watch` then relies on `proc.poll()`.
- A non-zero exit with no case running ends the shard as in step 3 above,
  with the exit status and the log path in each case's detail.
- An error in the harness itself (an unreadable job file) still raises in
  the parent: it is not a case's fault.

## Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `audit.config.MEMORY_LIMIT` | `int` | `4 * 1024**3` | Bytes. The `RLIMIT_AS` cap where the platform allows it, and now also the resident size past which `watch` kills a shard |
| `audit.config.POLL_SECONDS` | `float` | `2.0` | How often `watch` checks a shard (today's literal in `wait_or_kill`) |

The stall limit stays `2 * timeout + 30` seconds, now for both languages.

## Tests

In `src/tests/test_harness_engine.py`, with small `python -c` processes as
stand-ins for shards, as the existing `wait_or_kill` tests do:

- `resident_bytes` of the test process is positive; of a finished process,
  None.
- `watch` returns `'memory'` for a process that allocates past a small limit,
  `'died'` for one that exits 1, `'stalled'` and None as today.
- `run_shard` with a stand-in that dies on its second case: that case gets
  the `dead` result, every other case finishes, and an `-attempt1.jsonl` file
  is kept.
- `run_shard` with a stand-in that dies before its first case: every case
  gets the `dead` result and the loop runs once.
- `classify` gives `r-killed` for R errors ending `(killed)`, whether Python
  succeeded or failed; R's `(timeout)` errors still give `timeout`.
- `run_cases` end to end on two tiny cases whose Python side is made to exit
  (via a monkeypatched call): the step returns a result for both.

## Out of scope

The `scripts` step (`audit.scripts.run`) has the same hazard: a
`ProcessPoolExecutor` for its Python side and R processes with no restart.
Its shards have no per-unit start lines, so it needs its own change; it is
filed in `TODO.md` as "The audit's scripts step stops when one of its
processes dies" (review of 2026-11-02).

## External dependencies

None new. `ps` (present on macOS and Linux) is called through `subprocess`.
Windows, where the harness already runs uncapped, has no `ps`: there
`resident_bytes` returns None and only the stall and death checks apply.
