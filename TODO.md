# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.


## The metafor audit cannot start R on macOS 27.2

Found on 2026-11-01 by the first run of `bin/metafor-audit.sh` on this machine
(macOS 27.2, R 4.6.2, run `.audit/runs/20261101-204233`): `check-r` and
`universe` passed, then `matrix` stopped with `subprocess.SubprocessError:
Exception occurred in preexec_fn` when it started its first R shard, so no
case was compared.

Cause: `audit.rdriver.start` passes `preexec_fn=limit_memory`, which calls
`audit.config.limit_memory` (`src/audit/config.py:27`), and that sets
`resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))` with
`MEMORY_LIMIT` at 4 GiB. On macOS 27.2 the call raises `ValueError: current
limit exceeds maximum limit` for every finite limit tried (4, 8 and 64 GiB,
and a finite soft limit with an unlimited hard one), checked in a plain
Python process the same day.

Fix direction: make the cap best-effort where the platform refuses it (catch
the `ValueError` on macOS, as Windows already skips it), or cap with a limit
macOS honours, so the shard still starts; keep the cap on Linux, where it
works. The R shards are already killed after twice the case timeout, so a
runaway shard is still stopped. Then rerun the audit item below.

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
