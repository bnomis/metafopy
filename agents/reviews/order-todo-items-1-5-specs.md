# Review: specs of the TODO items 1 to 5 order

**Date:** 2026-11-18
**Order:** `agents/tasks/order-todo-items-1-5.md` (written after this review is answered)
**Review page:** https://claude.ai/artifact/VC2zj662ASUg6GBVvnR1Zc

## What is reviewed and how

The order covers the first five `TODO.md` items as they stood at `dcfb11c` (re-checked open by a TODO review at 08:06 UTC on 2026-11-18). Two became
specs: `agents/specs/ls-profile-starts/spec.md` (item 1) and `agents/specs/permutest-exact-limit/spec.md` (item 2). The other three become small task
lists with no spec. Approve or flag the plan and each spec, and answer each decision. Approving a spec that changes metafopy's output is the
confirmation `claude/rules/metafopy.md` asks for. The task lists are written only after every item and decision has an answer.

## Automatic checks

- Both specs have the specdev sections, quote their `TODO.md` item's dated findings, and pass the pre-commit hooks (including the spec-signature audit
  check): pass.
- The `ls-profile-starts` prototype ran every test file that profiles a location-scale fit (2026-11-18, `dcfb11c`): the only change is
  `ls.confint.scale.ML.bcg` turning from expected failure to pass; no R parity fixture moves: pass.
- The `permutest-exact-limit` bound was run over every test file that uses `permutest` (800 passed, with a recorder): no test reaches the new bound:
  pass.
- Item 4's data (probed 2026-11-18): the ER-visit counts are in Appendix III of the Bagos and Nikolopoulos PDF in `/Users/simonb/src/metafopy papers/`;
  the school-absence counts rebuild from Table 1 of Guevara, Berlin and Wolf (2004), BMC Med Res Methodol 4:17 (PMC481068, CC-BY); both reproduce the
  printed Table 5 in metafopy and R metafor 5.2-1 to 1e-4: pass. Hooper: not reproducible from the PDFs present (see D2).

## The items

### 1. The plan: three small task lists, no spec

| Row | Item | List | Waits for | Stops for you |
|---|---|---|---|---|
| 3 | 5. The rma_mv funnel test still skips when rstandard is missing | `mv-funnel-skip`: confirm the `mvm` case runs, remove `has_mv_residuals` and the skip in `src/tests/test_fits.py` | - | - |
| 4 | 4. Bagos and Nikolopoulos (2009) Table 5 | `bagos2009-table5`: published cases for the ER-visit and school-absence rows of Table 5 in `glmm_published.toml` (`UM.RS`, `coding=1`, `cor=True` for the random-study-effects row; the fixed-study-effects and fixed-effects rows as well); correct the "Sources needed" record of `agents/specs/correctness-glmm/mismatches.md`; Hooper by D2 | - | only if a printed row does not reproduce as the probe found |
| 5 | 3. Six ledger entries of known.toml match nothing in the audit | `audit-stale-entries`: run each entry's case in R and metafopy, remove or narrow it, then one full audit with `--recall` | 1 and 2 (their output changes reach the audit) | - |

- **Recommendation:** approve.

### 2. The spec `agents/specs/ls-profile-starts/spec.md` (item 1, row 1)

- **Cause, measured 2026-11-18:** at a = 1.36 every current start of the profile refit has `alpha.1` between -4.96 and -49.2, while the continuous
  branch's optimum is at -74.3; from each of them the refit falls onto the `tau2 = 0` plateau.
- **What it changes:** `LINE_WEIGHTS` in `src/metafopy/methods/ls_profile.py` from `(0.25, 0.5, 0.75, 1.0)` to `(0.25, 0.5, 0.75, 1.0, 1.5, 2.0)`, two
  regression tests, `docs/differences.md` and `CHANGELOG.md` entries; `ls.confint.scale.ML.bcg` moves to `pass`.
- **What output moves:** the ML `dat.bcg` log-link upper bound of `alpha.2`, 1.3598266 to 1.3812075 (the crossing; in every fit of that model);
  identity-link lower bounds that were a `<` at a point the profile does not reach become checked crossings (ML `dat.bcg` `alpha.2` -0.0082 to
  -0.0136; the `I(ablat/10)` form -0.0820 to -0.1356; REML `dat.bcg` `alpha.1` 0 to -0.2686 and `alpha.2` NaN to -0.0288; `bd_identity_ni`
  `alpha.2` 0 to -0.0049), and their NaN profile points gain values equal to a brute-force profile to 1e-6; two log-link profile points rise to a higher
  refit (`vl2022.table1.wald` by 0.155, `table3.ML.mixed` by 0.019) with no bound moving; optimiser noise of at most 2.6e-8 in two `bd_3_*` bounds.
  `test_rma_ls.py` takes 12 per cent longer (34.5 s to 38.8 s).
- **Choices made in the spec:** the two added weights (1.5, 2.0), not a denser set (+32 per cent run time, same values) or the reference's grid (four
  times the run time); the starts apply to the identity link too, since limiting them would keep bounds known to be wrong.
- **Found outside the item:** the test reference `formulas.ls_profile_ll` cannot fit the identity link at some feasible values (-1e10 where the
  brute-force profile is -8.045852); no case uses it. I chose to have the list file it in `TODO.md` rather than leave it in the spec's "Not in scope".
- **Recommendation:** approve.

### 3. The spec `agents/specs/permutest-exact-limit/spec.md` (item 2, row 2)

- **Cause, measured 2026-11-18:** R refuses an exact test only when allocating its arrays fails; `uni_mods` (389188800 permutations) passes that, then
  R builds the matrix of every permutation (37.7 GiB). metafopy's 16 GiB guard counts R's arrays but not that matrix, so it neither refuses nor
  finishes (about 75 h).
- **What it changes:** by decision D1 below.
- **Choice made in the spec:** the audit records the case in the `known.toml` ledger, not by giving R a heap limit, which could turn an allocation
  failure inside a refit anywhere in the audit into a failed refit.
- **Recommendation:** approve.

## Decisions

### D1. What `permutest(exact=True)` does on an exact test R cannot hold (item 2)

- **Background:** see item 3 above. No R output exists for `uni_mods` or `ls_bcg` under any option; no R parity fixture or correctness case changes.
- **What it affects:** `src/metafopy/methods/permutest.py`, `ls_permutest.py`, two test files, `src/audit/data/known.toml`, `docs/differences.md`.
- **Options:**
  - *(a) Count R's permutation matrix in the byte guard* (recommended): `uni_mods` and `ls_bcg` with `exact=True` raise R's "Number of iterations
    requested too large." in microseconds; the bound is `n_iter * (2 width + 1 + k) * 8` bytes against 16 GiB (119304647 iterations for `k = 13`, two
    coefficients); nothing else in the audit or the tests is newly refused.
  - *(b) Finish faster with batched refits*: about 3.2 h for `uni_mods` (prototype 29.8 us a permutation), but a second implementation of the refit to
    keep in step with the first, 15.6 GB of result arrays, and still nothing from R to compare.
  - *(c) A fixed cap of 10^7 iterations*: an arbitrary rule R does not have, refusing tests R completes.
  - *(d) Change nothing*: metafopy keeps running for about 75 h on the case.
- **Recommendation:** (a) - it fixes the root cause (the guard already means "refuse what R cannot hold"), refuses exactly the two cases R cannot run,
  keeps R's message, and is reversible.

### D2. Hooper in Bagos and Nikolopoulos (2009) Table 5 (item 4)

- **Background:** probed 2026-11-18. The ten trials of Table 1 of Hooper et al. (2001), the PDF in the papers folder, match Bagos's person-years but
  not his events (567 and 638 against 553 and 653), and a fixed-effects Poisson fit gives 0.858 against the printed 0.819; no swap of arms or rounding
  reproduces it. The data most likely comes from the full Cochrane review (Hooper et al. 2001, CD002137, DOI 10.1002/14651858.CD002137), which is not
  in the papers folder and may need a subscription (not confirmed); repository copies exist at UEA ePrints and LSHTM Research Online, not checked for
  the analysis tables.
- **What it affects:** the `bagos2009-table5` list and `TODO.md`.
- **Options:**
  - *(a) Narrow the item to Hooper* (recommended): the list writes the ER-visit and school-absence cases, and replaces the item with a smaller
    `TODO.md` item for Hooper that names the Cochrane review as the source to find.
  - *(b) You place the Cochrane review before the order*: put CD002137 (the 2001 issue) in `/Users/simonb/src/metafopy papers/`; the list then tries
    Hooper too and stops for you if it still does not reproduce.
  - *(c) Record Hooper as not reproducible*: a dated `mismatches.md` entry closes it with the probe's figures; no item stays open.
- **Recommendation:** (a) - two of three outcomes land now, and the third needs a document only a person can obtain; it stays visible as its own item.

## Prerequisites

None for you to provide under the recommended answers. R metafor 5.2-1 is in `/Users/simonb/src/metafopy/.audit/Rlib`. The school-absence rows come from
an open CC-BY paper, recorded in the case's citation with its PMC id. Under D2 (b), the Cochrane review must be placed before the order starts.

## Answers

Answered on the review page on 2026-11-18 (10:53 UTC), read back from its `answers` store; no comments:

| Item or decision | Answer |
|---|---|
| 1. The plan: three small task lists, no spec | approve |
| 2. The spec `agents/specs/ls-profile-starts/spec.md` | approve |
| 3. The spec `agents/specs/permutest-exact-limit/spec.md` | approve |
| D1. What `permutest(exact=True)` does on an exact test R cannot hold | (a) count R's permutation matrix in the byte guard |
| D2. Hooper in Bagos and Nikolopoulos (2009) Table 5 | (a) narrow the item to Hooper |

Applied: both specs are at version 1.1, status `approved`; `permutest-exact-limit` records D1 under its Decisions; D2 is carried into the
`bagos2009-table5` task list.
