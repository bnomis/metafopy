# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## metafopy's profile of the ls scale coefficient leaves the highest branch before the crossing

<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->
Found on 2026-11-18 by `agents/tasks/ls-confint-scale-branch/tasks-1.md` (Tasks 1 and 2) on `ls.confint.scale.ML.bcg` in
`src/tests/correctness/cases/location_scale.toml`: the profile of `alpha.2` (the `ablat` coefficient of the scale model) has a continuous branch and a
`tau2 = 0` plateau (3.868641), and the continuous branch is the maximum until it crosses the cut-off at 1.38120753375 (the fixed reference and R's best
refits agree). metafopy's own profile jumps to the plateau earlier: 3.789459 at a = 1.36 and the plateau 3.868641 from 1.37, so `confint` gives the
upper bound 1.3598 with no `>` sign, a jump in the profile and not the crossing 1.3812.
<!-- doc-counts: accept-end -->

Cause: `profile_starts` and `profile_fit` in `src/metafopy/methods/ls_profile.py` refit from too few starts, so the refit falls onto the plateau while a
higher continuous branch exists (the reference had the same fault before Task 2 widened its starts).

Fix direction: a spec is needed first, since the fix changes metafopy's `confint` and `profile` output and every `ls.confint.scale.*` case uses the same
starts. Then `ls.confint.scale.ML.bcg` moves to `pass`.

Spec: none yet.

Spec: `agents/specs/ls-profile-starts/spec.md`; tasks: `agents/tasks/ls-profile-starts/tasks-1.md`.

## The exact permutest of uni_mods neither refuses nor finishes in metafopy

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
Measured on 2026-11-18 by Task 2 of `agents/tasks/audit-ledger-gaps/tasks-1.md` (metafopy `dd58553`; full audit run
`.audit/runs/20261118-042436`, case `options/permutest/uni_mods/exact=true`, kind `r-killed`, "R passed the memory limit of 4 GiB (killed)"): for
`rma(yi, vi, mods = ~ ablat, data = es_bcg)`, `permutest(fit, exact='i')` reports 389188800 iterations and 15567552000 bytes of arrays, just under
`MAX_PERMUTATION_BYTES` (17179869184) in `src/metafopy/methods/permutest.py`, so metafopy does not refuse. `permutest(fit, exact=True)` was still running
after about 29 minutes (cut off by the background limit) at about 275 MB resident at 20 minutes.
<!-- doc-counts: accept-end -->

Cause: the guard bounds the size of the arrays, not the run time, so a case R cannot hold in 4 GiB passes the guard and runs for a very long time. The
case stays listed as an R kill in the audit report because no entry can say what metafopy does.

Fix direction: decide whether metafopy should refuse an exact permutest whose iteration count R cannot hold (a limit on iterations, as R's "Number of
iterations requested too large." is), or finish it faster; then add the case to the `r-killed` entry of `options/permutest/ls_bcg/exact=true` in
`src/audit/data/known.toml`. Needs a spec if the guard changes.

Spec: none yet.

Spec: `agents/specs/permutest-exact-limit/spec.md`; tasks: `agents/tasks/permutest-exact-limit/tasks-1.md`.

## Six ledger entries of known.toml match nothing in the audit

<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->
Found on 2026-11-18 by the full audit `.audit/runs/20261118-042436` (`wrong-or-missing.md`, "Not covered", "Stale ledger entries"), and already listed by
`.audit/runs/20261115-160207`: the entries of `src/audit/data/known.toml` matching `scripts/misc_selmodel/*` (`*sav <- selmodel(res, type = "beta")*`),
`scripts/misc_coef_se/*` (`*sel <- selmodel(res, type = "beta")*`), `scripts/misc_selmodel/test/41` (`*[[]out <- capture.output(print(sav))[]]*`),
`options/predict/ls_na/newscale=*`, the `ls_year`/`permutest_ls*` print-diff `*QM(df = 2) = 15.610[45]*` and the `print/sel_bcg` print entry
`98.1051`. None shows metafopy's output wrong (a ledger entry that matches nothing hides nothing).
<!-- doc-counts: accept-end -->

Cause: the difference each entry accepted no longer occurs in the pinned R 4.6.2 with metafor 5.2-1 (the same kind of cause as the stale `delta.1`
entry removed on 2026-11-03), or metafopy's output moved; not traced entry by entry.

Fix direction: for each entry, run its case in R and metafopy; remove the entry where the two now agree, narrow its `detail` where they differ in
another way, and rerun `bin/metafor-audit.sh` to see the report list no stale entry.

Spec: none (small item).

Spec: none (small item); tasks: `agents/tasks/audit-stale-entries/tasks-1.md`.

## Find the data behind Table 5 of Bagos and Nikolopoulos (2009)

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by Tasks 9 and 11 of `agents/tasks/correctness-glmm/tasks-1.md` (`agents/specs/correctness-glmm/mismatches.md`, decision D3); the
user answered *investigate* on 2026-11-14. Bagos and Nikolopoulos (2009) Table 5 prints random-study-effects incidence rate ratios (Hooper 0.746 (0.592,
0.940), school absences 0.721 (0.579, 0.897), ER visits 0.525 (0.361, 0.764)) but no per-study counts or person-times, so no Poisson case could be
written. The Table 1 trial rows of Hooper et al. (2001) give an IRR of 0.859 (`UM.FS`) against the printed 0.819, and Wolf et al. (2003) prints means
and standard deviations, not counts.
<!-- doc-counts: accept-end -->

Fix direction: find the original per-trial counts and person-times that reproduce Table 5 (the authors' supplementary data, or the primary trial
reports), then write the published cases in `src/tests/correctness/cases/glmm_published.toml`.

Spec: none (small item); tasks: `agents/tasks/bagos2009-table5/tasks-1.md`.

## The rma_mv funnel test still skips when rstandard is missing

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. `test_funnel_matches_metafor` in `src/tests/test_fits.py` skips the `mvm` case when `has_mv_residuals` finds no
`rstandard` for `rma_mv` fits ("rstandard() of rma_mv fits is not available yet"), but `rstandard_rma_mv` now exists in
`src/metafopy/methods/mv_diagnostics.py`, so the skip guards a gap that has closed.
<!-- doc-counts: accept-end -->

Fix direction: confirm the `mvm` case runs, then remove `has_mv_residuals` and the skip.

Spec: none (small item); tasks: `agents/tasks/mv-funnel-skip/tasks-1.md`.

## Correctness coverage the extension specs left for later

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. `agents/specs/correctness-location-scale/spec.md` (out of scope) leaves for a later spec `profile` beyond the
property, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha` and `beta`, and `alpha_min` and `alpha_max`;
`agents/specs/correctness-location-scale/mismatches.md` adds REML with `optbeta=True`, the deviance and `vb` of `optbeta` fits.
`agents/specs/correctness-mv-robust/spec.md` says a later spec can add the `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef`.
No spec, task list or item covers them; they are checked against R metafor only.
<!-- doc-counts: accept-end -->

Fix direction: deferred. When taken up, one spec per area with `/specdev`, listing the defining references and the coverage items in
`src/tests/correctness/test_coverage.py`.
