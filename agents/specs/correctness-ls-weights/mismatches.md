# Correctness tests: user weights: mismatches

**Task list:** `agents/tasks/correctness-ls-weights/tasks-1.md`
**Spec:** `agents/specs/correctness-ls-weights/spec.md`
**Review page:** none (no mismatch case, so no review)

## What is reviewed and how

Each item would be a case whose reference, coded from the cited publication, disagrees with metafopy, committed with `status = "mismatch"`. For
each the operator would choose whether metafopy follows the reference, keeps metafor's behaviour, or the difference is investigated further. This
work produced no such case.

## Automatic checks

Checked 2026-11-20:

- **Pass:** no case added by this task list has `status = "mismatch"`: the `ls.weights.*` cases in
  `src/tests/correctness/cases/location_scale.toml` and the `uni.weights.*` cases in `src/tests/correctness/cases/inference.toml` carry none. The
  only `mismatch` case in `location_scale.toml`, `ls.profile.ML.identity.bcg`, belongs to the location-scale extensions list.
- **Pass:** `pytest src/tests/correctness -m 'not slow'` (see the task's commit for its result).

## Mismatches

None.

## Decisions

None.

## Answers

None: no answer needed.
