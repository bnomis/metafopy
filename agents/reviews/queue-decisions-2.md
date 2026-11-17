# Review: queue decisions of the CI platform failures order

**Date:** 2026-11-17
**Order:** `agents/tasks/order-ci-platform-failures.md`
**Review page:** https://claude.ai/artifact/TTuz6hMpVhDQhHYoJDGCGJ

## What is reviewed and how

While `/rro agents/tasks/order-ci-platform-failures.md` ran (2026-11-16 to 2026-11-17), three task-list stops that were not reserved for the operator
were decided by Claude, each committed with a `Queue-decision:` trailer (`git log --grep '^Queue-decision:'`). Approve or flag each. A flag means the
decision is revisited in a follow-up; the list is already merged. The stop reserved for the operator (Task 8's `rma_mv` REML score) was answered by the
operator on 2026-11-17 and is not reviewed here.

## Automatic checks

- `bin/rro.sh decisions agents/tasks/order-ci-platform-failures.md --since 388cc6d` lists exactly these three decisions: pass.
- Row 3 passed its merge gate (test log: 8578 passed, 37 skipped, 14 xfailed) and is merged as `0abbe0c`; CI run 44055240124 passed every job on its
  draft pull request #74 (closed unmerged): pass.

## The items

### 1. The halfnorm print asserts its observed range (`agents/tasks/cross-platform-numerics/tasks-1.md` Task 9, commit `f57ab45`)

- **The stop:** `test_selmodel.py::test_print[hack_halfnorm_less]` prints `ci.lb` values that move by up to 0.0075 between platforms (CI Mac), because
  `delta` sits at its bound (0.000138) on every environment and the Hessian moves with the platform. The spec's tolerance rule (10 times the largest
  deviation, 0.075) was refused: it is 72 per cent of the standard error, larger than the effect the test detects.
- **The decision:** option (a) of the three the list offered, as verdict 4 (platform-dependent; the spec's choice "Platform-dependent marks" asserts the
  observed range): `NOISY_ATOL` 0.01 for that case's printed `ci.lb` only, every other printed field compared as before, with an entry in
  `docs/differences-notes.md`. Implemented in `c069f04`.
- **Recommendation:** approve - it is the spec's own route for a quantity at a bound; the alternatives were a tolerance the spec refuses or skipping the
  comparison on two platforms.

### 2. Push the branch before the CI round (`agents/tasks/cross-platform-numerics/tasks-1.md` Task 8, commit `f57ab45`)

- **The stop:** Task 8 (the `rma_mv` REML score) needed one CI probe run of its commit for a verdict, but the push was planned for Task 11, which waits
  for Tasks 5 to 10.
- **The decision:** Task 8 pushes the list's branch at once to update draft pull request #74. The answer to D1 (*Draft pull request*, 2026-11-16) allows
  the list to push its branch and read CI; this moved one push earlier. Its run (probe run 44002318304) showed metafopy's own fit stopping short on Linux
  and Windows, the stop the operator then answered.
- **Recommendation:** approve - within D1, and the only way to get the figures.

### 3. Bound the test-time cost of the robust reference fit (`agents/tasks/cross-platform-numerics/tasks-1.md` Task 5, commit `6e31308`)

- **The stop:** CI round 1 showed the test jobs at 63 to 70 minutes (20 to 38 before Task 5) and Windows past two hours. Measured on the development Mac
  on 2026-11-17: the suite took 48 min 37 s against 19 min 52 s on `main`, mostly in three tests whose every permutation refit ran the full multi-start
  search (`test_random_permutation_p_values_follow_the_exact_ones` 843.7 s against 43.6 s; `ls.permutest.scale.exact.ML.z.bcglat` 433.5 s against
  24.9 s; `ls.permutest.location.exact.ML.z.bcglat` 414.8 s against 18.6 s).
- **The decision:** Claude stopped the queue and added to the reopened Task 5: run the extra starts only when the first fit fails its score check, and
  keep those three tests within twice `main`'s time. Implemented in `8bec6e1`; CI round 2 passed in 37 to 65 minutes, and the final local suite took
  19 min 53 s.
- **Recommendation:** approve - the spec set no bound on test time, and without one every later CI run would have cost about three times as much.

## Decisions

None beyond the items: each item's approve or flag is the decision.

## Answers

Answered on the review page on 2026-11-17 (12:02 UTC), read back from its `answers` store; no comments:

| Item | Answer |
|---|---|
| 1. The halfnorm print asserts its observed range (Task 9, `f57ab45`) | approve |
| 2. Push the branch before the CI round (Task 8, `f57ab45`) | approve |
| 3. Bound the test-time cost of the robust reference fit (Task 5, `6e31308`) | approve |

All three decisions stand; no follow-up is needed.
