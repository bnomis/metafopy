# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


## The metafor audit's options step loses a Python worker on macOS

Found on 2026-11-01 by the second run of `bin/metafor-audit.sh` on this
machine (macOS 27.2, R 4.6.2, run `.audit/runs/20261101-211517`), the first
after the memory cap stopped failing: `check-r`, `universe` and `matrix`
(2503 cases, 3298 findings) completed, and every R shard of `options`
finished, but `options` then stopped with
`concurrent.futures.process.BrokenProcessPool: A process in the process pool
was terminated abruptly`. No Python shard wrote its `py-*.json`, and macOS
wrote no crash report for it.

Cause: unconfirmed. `audit.engine.run_cases` runs the Python side of each
shard in a `ProcessPoolExecutor` (`src/audit/engine.py:96`), and a worker that
dies takes the whole step with it. Suspected: a case that grows without
bound, now that macOS runs workers without the 4 GiB address-space cap (see
the macOS item in `DONE.md`), so the system kills the worker where Linux
would raise `MemoryError` inside it.

Fix direction: run the step alone (`bin/metafor-audit.sh --steps options`)
and watch the workers' memory to find the case; then decide between a memory
guard macOS honours (for example checking the resident size between cases)
and making a lost worker a reported finding for its shard rather than the end
of the step.

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
