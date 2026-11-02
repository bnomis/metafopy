# Review: Audit shard survival spec

**Review page:** https://claude.ai/artifact/CWQodUXjiX6oNTABHE7d3Q

## What is being reviewed and how

The spec `agents/specs/audit-shard-survival/spec.md` (proposed 2026-11-02)
makes the audit's Python shards survive a dead process the way R shards
already do, and has the parent kill a shard of either language whose resident
size passes `MEMORY_LIMIT`. Approve or flag each part below, and answer the
one open decision. Implementation waits until every item has an answer.

## Automatic checks

- Every template section present (Module location, API, Error handling,
  Implementation, Settings, External dependencies): pass.
- API has type-annotated signatures: pass.
- Every module path exists under `src/audit/` or `src/tests/`: pass.
- Size: 218 lines, a little over the ~200-line guide; it covers one change
  to one package, so it is not split.

## The items

1. **Decisions recorded** - the four answers of 2026-11-02: parent-side
   memory watch with `ps`; the limit applies to R too; `memory` and
   `process-died` under Crashes, `r-killed` in its own section; the limit
   stays 4 GiB, read as resident size.
2. **One shard loop for both languages** - `run_r_shard`'s restart loop
   becomes `run_shard`, used by R and Python; a process that dies before its
   first case ends the shard instead of looping.
3. **The Python shard process** - each Python shard runs as
   `python -m audit.pyrun <job.json>`, writing start and result lines like
   `runcases.R`; `py-{i}.json` keeps its shape. The pool goes.
4. **The memory watch** - `watch` replaces `wait_or_kill`, polling output size
   and resident size every `POLL_SECONDS`.
5. **Classification and report** - `r-killed` replaces
   `accepts-where-R-refuses` when R's process died or was killed; new report
   kinds and section; a README section.
6. **Error handling and settings** - no case is left without a result;
   `POLL_SECONDS` is new; Windows has no `ps`, so only stall and death apply
   there.
7. **Tests** - stand-in `python -c` processes for memory, death, stall and
   restart, plus `classify` and one end-to-end `run_cases`.

## Decisions

### D1: the `scripts` step

**What is being decided:** whether this spec also covers the `scripts` step.

**Background:** `audit.scripts.run` runs the Python side of the scripts step
in a `ProcessPoolExecutor` too, and its R processes have no restart. Its
shards write no per-unit start lines, so the shared loop cannot be reused
without changing `runscripts.R`.

**What it affects:** the size of this spec and its task list; whether the
scripts step can still be stopped by one dead worker after this ships.

**Options:**

- **File it separately (recommended):** this spec stays one change to the
  `options` and `matrix` path; the scripts step goes in `TODO.md` as its own
  item once this spec is approved, and reuses `watch` and `run_shard` later.
- **Include it here:** the spec grows by a section on `runscripts.R` start
  lines and the scripts loop, and goes past the size guide; one task list
  fixes both.

## Answers

Answered on 2026-11-02 on the review page:

- Items 1 to 7: approved, no comments.
- D1, the `scripts` step: file it separately. Filed in `TODO.md` as "The
  audit's scripts step stops when one of its processes dies".
