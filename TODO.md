# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

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

## The reference ls_profile_ll cannot fit the identity link at some feasible values

<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->
Found on 2026-11-18 by `agents/specs/ls-profile-starts/spec.md` ("Not in scope") and `agents/tasks/ls-profile-starts/tasks-1.md`:
`formulas.ls_profile_ll` in `src/tests/correctness/formulas.py` gives -1e10 at `alpha.2 = -0.00615` of the ML identity-link `dat.bcg` fit
(`ls.fit.ML.identity.bcg` in `src/tests/correctness/cases/location_scale.toml`), where the brute-force profile is -8.045852. No
`ls.confint.scale.*` case uses the identity link, so no case is affected.
<!-- doc-counts: accept-end -->

Cause: the SLSQP starts of `ls_profile_ll` (`start[free]`, the adjusted start and the line between them) are infeasible there (`Z alpha < 0`), so no
refit runs.

Fix direction: give `ls_profile_ll` a feasible start for the identity link (for example the starts of `profile_starts`, filtered as it filters them),
then add an identity-link `ls.confint.scale.*` case checked against the brute-force profile.
