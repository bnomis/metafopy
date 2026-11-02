# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


## A Python audit worker that dies stops the whole audit step

Found on 2026-11-02 while fixing the lost worker in `options` (see `DONE.md`):
`audit.engine.run_cases` runs the Python side of the shards in a
`ProcessPoolExecutor`, so one worker killed by the system (out of memory, a
crash in compiled code) raises `BrokenProcessPool` and ends the step, and no
shard writes its results. On macOS 27 the Python workers run without a memory
cap, because `setrlimit(RLIMIT_AS)` is refused there, so any case that grows
without bound kills its worker instead of raising `MemoryError`. The case
that did so (`permutest(exact=True)`) is fixed; nothing stops the next one.

Fix direction: watch each Python worker's resident size between cases, or
from a thread, and record a case that passes a limit as an error; and make
`run_cases` record a dead worker's current case as a finding for its shard,
then carry on with the rest of that shard, as the R side already does
(`r-N-attempt1.jsonl`).

Spec: agents/specs/audit-shard-survival/spec.md; tasks: agents/tasks/audit-shard-survival/tasks-1.md.

## The audit's scripts step stops when one of its processes dies

Found on 2026-11-02 while writing `agents/specs/audit-shard-survival/spec.md`,
whose review chose to keep this step out of that spec:
`audit.scripts.run` runs the Python side of the scripts step in a
`concurrent.futures.ProcessPoolExecutor`, so one worker killed by the system
raises `BrokenProcessPool` and ends the step. Its R processes are started
once each with no restart, and a shard whose `r-N.json` is missing raises
`RuntimeError`. Neither side has a memory watch on macOS.

Cause: the scripts shards write no per-unit start lines (`runscripts.R`), so
the case-level restart of the cases path cannot be applied to them.

Fix direction: after the shard survival spec ships, make `runscripts.R` and
`audit.scripts.run_shard` write a start line before each unit, and run both
sides through that spec's `audit.engine.run_shard` and `watch`.

## Run the metafor audit to find the next gaps against R metafor

Found on 2026-11-01: `TODO.md` held no open port work, and this checkout
has no earlier run under `.audit/runs/` to draw from.
`bin/metafor-audit.sh` compares metafopy with R metafor across every exported
function and argument (see `src/audit/README.md`).

Cause: new port work is found by the audit harness, and it has not been run
since the auditme work.

Fix direction: run `bin/metafor-audit.sh` in the background, then file each
cause in its `wrong-or-missing.md` here with `/todo-add`. Run
`bin/metafor-audit.sh --recall` too, and fix any miss in `src/audit/data/`.

<!-- doc-counts: accept-start note of 2026-11-02 - a dated record, not a claim about the present -->
Note 2026-11-02: `bin/metafor-audit.sh --steps options` completed into
`.audit/runs/20261102-061046` at `2fc8c6d`; its `wrong-or-missing.md` lists
81 causes, none yet filed here. The `matrix` step last ran in
`.audit/runs/20261101-211517`, and no recall run was found. Still to do: file
those causes, run the whole audit in one run, and run `--recall`.
<!-- doc-counts: accept-end -->

## Remove the auditme accept markers once the shared rules stop the false positives

Found on 2026-11-01 while clearing the auditme hits
(`agents/specs/auditme-clean/spec.md`): some hits were the rules getting the
code wrong, not the code. Each site is accepted with a marker and listed by
rule in `agents/specs/auditme-clean/requests.md`.

Cause: the shared auditme rules decide from the shape of a call or a
`raise` alone; the fix is in the agents repository, filed in its `TODO.md` as
"auditme rules report hits the code does not have".

Fix direction: once that item ships, bump the agents rev in
`.pre-commit-config.yaml`, remove each marker listed in `requests.md` that
the new rules no longer need, and delete its entry there.

Note 2026-11-02: the agents item shipped on 2026-11-01 (its `DONE.md`,
`agents/specs/auditme-false-positives/spec.md`), merged on its `origin/main`
at `827a373`. `.pre-commit-config.yaml` still pins `8b43d9b`, which predates
it, so this item can start.
