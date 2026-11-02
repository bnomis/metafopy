# Review: Scripts shard survival spec

**Review page:** https://claude.ai/artifact/BXYXpB2kCa4g3RsxARvspk

## What is being reviewed and how

The spec `agents/specs/scripts-shard-survival/spec.md` (proposed 2026-11-02)
makes the audit's `scripts` step survive a dead or oversized process on
either side, by running its shards through the restart loop and memory watch
the cases path got from the audit shard survival spec. A script is the unit
of restart. Approve or flag each part below. The three design choices were
answered on 2026-11-02 before the draft and are recorded in the spec, so
there is no open decision. Implementation waits until every part has an
answer.

## Automatic checks

- Every template section present (Module location, API, Error handling,
  Implementation, Settings, External dependencies): pass.
- API has type-annotated signatures: pass.
- Every existing module path named exists; `src/tests/test_harness_scripts.py`
  is marked new: pass.
- Size: over the ~200-line guide (run `wc -l` on the spec); it covers one
  change to one step, so it is not split.

## The items

1. **Decisions recorded** - the three answers of 2026-11-02: restart at the
   next script; keep a dead script's finished units; report the dead unit as
   a finding, with the cases path's kinds.
2. **Shared loop, generalised** - `engine.run_shard` takes any item with an
   `id` and a `write` callback; `read_outcomes` gathers `{"part": id}` lines,
   and `dead` gets the running item's parts. The cases path's results do not
   change.
3. **The R side** - `runscripts.R` gets a `list` mode (the parent now shards
   the scripts) and a `run` mode that streams a start line, one line per
   unit, and a closing line per script. A dead script's record keeps its
   finished units and ends with a dead unit whose error ends `(killed)` or
   `(timeout)`.
4. **The Python side** - `python -m audit.scripts <job.json>` streams one
   line per unit; the process pool goes. A dead unit gets `memory`,
   `process-died` or `timeout`; R's killed unit gets `r-killed`; the rest of
   a dead script is counted as `not run` in the tally.
5. **Error handling** - every listed script gets a record and the step
   finishes; a process that dies before its first script does not loop;
   `runscripts.R list` failing raises, as a harness error.
6. **Tests** - part lines in `read_outcomes` and `run_shard`; a new
   `test_harness_scripts.py` with no R; a real scripts run before and after
   the change, compared by tally and cause. No scripts run exists under
   `.audit/runs/` yet (checked 2026-11-02), so the run before is part of the
   work.

## Decisions

None open. The answers below were given on 2026-11-02 in the question
dialog, before the draft:

- Restart: at the next script (not replaying the dead script).
- A dead script's finished units: kept.
- The dead unit: a finding, with the cases path's kinds.

## Answers

Answered on 2026-11-02 on the review page:

- Items 1 to 6: approved, no comments.
