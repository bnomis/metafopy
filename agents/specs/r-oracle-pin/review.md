# Review: R oracle pin spec

**Review page:** https://claude.ai/artifact/MQCiHjWGs9yKYpwQoch8Qq

## What is being reviewed and how

The spec `agents/specs/r-oracle-pin/spec.md` (proposed 2026-11-03) pins the
whole R library the audit compares against, makes `check-r` refuse a run on a
different library unless `--allow-r-drift` is given, installs the pins on a
new machine, and counts recall items the pinned R no longer shows apart from
hits and misses. It then brings the known differences, the universe and the
recall corpus up to the pin. The four design choices were answered on
2026-11-03 before the draft and are recorded in the spec, so there is no open
decision. Approve or flag each part below; implementation waits until every
part has an answer.

## Automatic checks

- Every template section present (Module location, API, Error handling,
  Implementation, Settings, External dependencies): pass.
- API has type-annotated signatures: pass.
- Every existing module path named exists; `src/audit/data/r-versions.json`
  is marked new: pass.
- Size: under the ~200-line guide (`wc -l` on the spec): pass.

## The items

1. **Decisions recorded** - the four answers of 2026-11-03: pin the current
   library (R 4.6.2, lme4 2.1-0, as the 2026-11-02 run recorded it); refuse a
   different library with an override that the report records; mark recall
   items the oracle moved; install every package at its pinned version.
2. **The pin and `check-r`** - `data/r-versions.json` is the 2026-11-02
   run's library; `check-r` compares every package and R, refuses with each
   difference and three ways on, and refuses when the pin file is missing.
3. **`--allow-r-drift`** - runs anyway, puts the differences in the report's
   notes, and is passed on by `--recall`.
4. **Setup from the pin** - `setup.R` installs each package whose version
   differs at its pinned version; a failure is listed, and `check-r` then
   refuses.
5. **Recall items the oracle moved** - a new `oracle_moved` field, counted
   apart and listed in its own section; `a6-glmm-um-rs` is the first; an
   item reproduced again is flagged so the field can go.
6. **Bringing the data up to the pin** - remove or move the stale known
   differences, refresh the universe, rerun the audit and `--recall`, and
   update `docs/differences.md`; `TODO.md` items 2 to 8 were found with this
   library, so they stand unless the rerun shows otherwise.
7. **Tests** - `library_drift`, `check_r` with R stubbed, `recall.write`
   with an `oracle_moved` item, and the real audit and recall runs.

## Decisions

None open. The answers below were given on 2026-11-03 in the question
dialog, before the draft:

- The pin: the current library, not the older lme4 2.0.6.
- A different library: refused, with `--allow-r-drift` recorded in the report.
- A recall item the pinned R no longer shows: marked `oracle_moved`.
- Setup: installs every package at its pinned version.

## Answers

Answered on 2026-11-03 on the review page:

- Items 1 to 7: approved, no comments.
