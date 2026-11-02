# Scripts shard survival

**Version:** 1.0
**Status:** approved
**Date:** 2026-11-02

The audit's `scripts` step runs metafor's own help examples and test files,
one top-level expression (a unit) at a time, in R and then, translated, in
Python. Its shards are lost when one of their processes dies. The cases path
already survives that
([Audit shard survival](../audit-shard-survival/spec.md), implemented
2026-11-02). This spec runs the scripts shards through the same restart loop
and memory watch, with a script as the unit of restart.

## Motivation

From the `TODO.md` item "The audit's scripts step stops when one of its
processes dies", found on 2026-11-02 and checked again on 2026-11-02 at
`6e54bc1`:

- `audit.scripts.run` starts one `Rscript runscripts.R` per shard, with no
  watch and no restart, and raises `RuntimeError` when a shard's `r-N.json`
  is missing. `runscripts.R` writes its whole result at the end with one
  `write_json`, so a dead R process loses its whole shard.
- The Python side runs `audit.scripts.run_shard` in a
  `concurrent.futures.ProcessPoolExecutor`, so one dead worker raises
  `BrokenProcessPool` and ends the step.
- On macOS 27 neither side has an address-space cap (`setrlimit(RLIMIT_AS)`
  is refused), so a unit that grows without bound kills its process.

The units of one script share one environment: an R environment per script,
and a Python `audit.rlib.Env` per script, with `stale` tracking of the
variables that differ between the two sides. A restarted process therefore
cannot pick up in the middle of a script.

Decisions taken with the operator on 2026-11-02:

1. **Restart at the next script.** A process that dies partway through a
   script is restarted on the scripts after it. The rest of the dead script
   is not run, and is counted in the coverage tally.
2. **Keep finished units.** Each unit is streamed as its own line, so the
   units of a dead script that finished are kept and compared as usual.
3. **Report the dead unit as a finding, as the cases path does.** R dying
   gives `r-killed`; Python dying gives `memory`, `process-died` or
   `timeout`, in the report sections that already exist.

## Module location

- `src/audit/engine.py` - `run_shard` and `read_outcomes`, generalised to any
  item with an `id`, with part lines
- `src/audit/r/runscripts.R` - the `list` and `run` modes, streaming
- `src/audit/scripts.py` - `Script`, the R and Python shard drivers, the
  Python shard process, assembly of `r-N.json` and `py-N.json`
- `src/audit/README.md` - the scripts step in "Shards that die or pass the
  memory limit"
- `src/tests/test_harness_engine.py`, `src/tests/test_harness_scripts.py`
  (new) - tests

## API

```python
# audit/engine.py


class Item(typing.Protocol):
    @property
    def id(self) -> str: ...


def read_outcomes(path: pathlib.Path) -> tuple[dict[str, Json], str | None, list[Json]]:
    """The finished items of a runner's JSON-lines file, the item running when it stopped (None when it finished cleanly), and the part lines
    that item wrote before it stopped. A line ``{"part": id, ...}`` belongs to item ``id``; a finished item's parts are in its record under
    ``"parts"``, in order."""


def run_shard[T: Item](
    work: pathlib.Path,
    name: str,
    shard: list[T],
    timeout: float,
    start: collections.abc.Callable[[pathlib.Path, pathlib.Path], subprocess.Popen[bytes]],
    dead: collections.abc.Callable[[str, Stop, list[Json]], Json],
    write: collections.abc.Callable[[pathlib.Path, list[T]], None],
) -> list[Json]:
    """As today, for any items with an ``id``: ``write(path, remaining)`` writes the items a process is started on, and ``dead`` also gets the
    part lines the running item wrote before its process stopped."""


# audit/scripts.py


@dataclasses.dataclass(frozen=True)
class Script:
    id: str  # 'rd/<topic>' or 'test/<topic>', as today
    kind: str  # 'rd' or 'test'
    topic: str
    file: str  # path inside the metafor source: 'man/<topic>.Rd' or 'tests/testthat/test_<topic>.R'


def list_scripts(source: pathlib.Path, work: pathlib.Path, only: list[str] | None) -> list[Script]:
    """Every script ``runscripts.R list`` finds in the metafor source, in R's order, filtered by ``only`` as today."""


def run_r_scripts(work: pathlib.Path, i: int, source: pathlib.Path, shard: list[Script], timeout: float) -> None:
    """R side of scripts shard ``i`` through ``engine.run_shard``; writes ``r-{i}.json`` as ``{"scripts": [...]}`` in shard order, today's shape."""


def run_py_scripts(work: pathlib.Path, i: int, shard: list[Script], nse: list[str], functions: list[str], timeout: float) -> None:
    """Python side of scripts shard ``i`` through ``engine.run_shard``; writes ``py-{i}.json`` as ``{"findings": [...], "tally": {...}}``."""


def stream_scripts(r_path: pathlib.Path, scripts_path: pathlib.Path, out_path: pathlib.Path, nse: list[str], timeout: float) -> None:
    """Run the scripts in ``scripts_path`` against R's records in ``r_path``, writing a start line, one part line per unit
    (``{"part": id, "findings": [...], "tally": {...}}``) and a closing ``{"id": id}`` line per script, flushed each time."""
```

`python -m audit.scripts <job.json>` runs `stream_scripts`; the job file holds
its arguments and the metafor function names (today's `.functions` file).
`run_shard(job)` and its tuple job go with the pool. `run(...)` keeps its
signature and return value, so `audit.steps.scripts_step` is unchanged.

## Implementation

### The R side

`runscripts.R` gets two modes, chosen by its first argument:

- `list <source> <out.json>` writes every script it would run today, as
  `[{id, kind, topic, file}]`: help pages whose `tools::Rd2ex` gives a file,
  then the `test_*.R` files, in today's order, filtered by `AUDIT_ONLY`.
- `run <source> <scripts.json> <out.jsonl> <timeout>` runs the listed
  scripts. For each one it writes `{"start": id}`, then one
  `{"part": id, "unit": <unit record>}` line per unit, then
  `{"id", "kind", "topic"}` (with `"error"` when the file does not parse),
  flushed after each line, as `runcases.R` does. The unit record is today's
  (`status`, `value` or `error`, `warnings`, `secs`, `drawn`, `src`, `ast`).
  An `Rd` page is turned into examples with `Rd2ex` when its script starts.

The parent shards the list with `audit.cases.shards`, so it now owns the
sharding that `runscripts.R` does today with its shard arguments.
`run_r_scripts` runs each shard through `run_shard` with `audit.rdriver.start`,
so the R process keeps its `preexec_fn` cap and its log `r-{i}.log`. A
finished script's record is its closing line, with `units` taken from its
parts. The `dead` result for a script whose process stopped is
`{"id", "kind", "topic", "lost": true, "units": <its parts> + [dead unit]}`.
The dead unit is `{"status": "error", "error": <reason>, "warnings": [],
"src": "", "ast": null}`, and the reasons are the cases path's: `R passed
the memory limit of <n> GiB (killed)`, `R process died on this case
(killed)`, and `R did not return within <s>s (timeout)` for a stall.

Units stream one line each, so the output moves at least once per unit, and
the stall limit stays `2 * timeout + 30` seconds, above the R unit's own
time limit.

### The Python side

`stream_scripts` loads R's records for its scripts from `r-{i}.json` and runs
each script as `run_script` does today, one unit at a time, writing each
unit's findings and tally as a part line. `run_script` becomes a generator
of per-unit `(findings, tally)`, so the stream and the tests share it. The
process calls `audit.config.limit_memory()` and seeds 1234 before each unit,
as today.

When R's unit is the dead unit of a lost script (an error for which
`pyrun.r_killed` holds), Python gives it an `r-killed` finding with R's error
as detail and does not translate it. A dead unit whose R error is the
`(timeout)` stall reason is counted as `R failed (skipped)`, as R's own
timeouts in scripts are today. A lost script also counts
`not run: R process died partway through the script` once in the tally.

`run_py_scripts` starts `sys.executable -m audit.scripts <job.json>` with the
inherited environment, output to `py-{i}.log`, through `run_shard`. Its
`dead` result keeps the parts' findings and tallies, and adds, for unit
`len(parts)` of that script, a finding of kind `memory`, `process-died` or
(for `stalled`) `timeout`, with id `scripts/<topic>/<kind>/<n>` and a detail
naming the limit or the exit status and the log. The units after it are
counted as `not run: Python process died partway through the script`.

`run` lists the scripts, runs the R shards in a `ThreadPoolExecutor` and
then the Python shards in another, as `engine.run_cases` does, and sums the
findings and tallies of every `py-{i}.json` as today.

### Shared loop

`read_outcomes` collects part lines into `parts` by item; a line with
neither `start`, `part` nor `id` is still skipped. `run_shard` passes the
running item's parts to `dead`, and takes `write` in place of its call to
`audit.cases.write_cases`. The cases path passes `audit.cases.write_cases`,
and its `dead` callbacks ignore the parts, so its results are unchanged.

## Error handling

- A dead or killed process, on either side, is a finding and a tally entry,
  never an exception: every listed script gets a record, and the step
  finishes.
- A process that stops before its first script gives every remaining script
  a lost record with a dead unit at index 0, as `run_shard` does for cases
  today (no loop).
- `runscripts.R list` failing, or writing nothing readable, raises
  `RuntimeError` naming its log: the harness cannot run the step, and that
  is not a unit's fault.
- An unreadable job file raises in the Python shard process, which then dies
  before its first script and is reported as above.

## Settings

None new. The step uses `audit.config.MEMORY_LIMIT` (4 GiB, resident size)
and `audit.config.POLL_SECONDS` (2.0) through `engine.watch`, and the stall
limit `2 * timeout + 30` seconds of `engine.run_shard`.

## Tests

- `src/tests/test_harness_engine.py`: `read_outcomes` gathers part lines into
  a finished item's `parts` and returns the running item's parts;
  `run_shard` with plain-dict items and its own `write`, where a stand-in
  dies after two parts of its second item, passes those parts to `dead`.
- `src/tests/test_harness_scripts.py` (new), with no R: a lost script record
  becomes an `r-killed` finding at the dead unit and one `not run` tally;
  `stream_scripts` on a stub `r-0.json` of two tiny scripts writes start, part
  and closing lines that `read_outcomes` reads back; `run` with
  `run_r_scripts` stubbed and the Python side made to exit on one unit
  returns findings for every script, one of them `process-died`, and
  `not run` for the units after it.
- A real `bin/metafor-audit.sh --steps scripts` run before the change and one
  after it, compared by tally and by cause. This checkout has no earlier
  scripts run under `.audit/runs/` (checked 2026-11-02), so the run before
  is part of the work. Every difference is explained; the shard order may
  move units between shards, but not change a unit's result.

## Out of scope

Replaying a dead script's earlier units to rebuild its environment and carry
on past the dead unit (rejected on 2026-11-02 in favour of the next script).

## External dependencies

None new. `ps` is used through `engine.watch`, as for the cases path.
