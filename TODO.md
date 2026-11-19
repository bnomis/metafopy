# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Add the printed Baskerville estimates of Citkowicz and Vevea (2017) as published cases

<!-- doc-counts: accept-start found on 2026-11-19 - a dated record, not a claim about the present -->
Found on 2026-11-19, when the user added the article to the papers folder (`/Users/simonb/src/metafopy papers/A Parsimonious Weight Function for
Modeling Publication Bias paper.pdf`; Citkowicz M, Vevea JL (2017), Psychological Methods 22(1):28-41, doi:10.1037/met0000119). It was marked dropped
on 2026-11-15 (`agents/specs/correctness-publication-bias/mismatches.md`, sources table) because it needed APA access, so the two
`published.citkowicz2017.baskerville2012.*` cases in `src/tests/correctness/cases/selmodel_published.toml` check the authors' supplementary R code on
the supplement's 19 effects. The article's example (section "Example", Figure 3) uses all 23 effects of Baskerville et al. (2012), the rows of the
bundled `dat.baskerville2012` (`vi = se**2`). Fitted on 2026-11-19 with ML, metafopy and R metafor 5.2-1 agree to 4 decimals:

| Model | Printed | metafopy and R |
|---|---|---|
| mean only, unadjusted: mean, se, tau2 | 0.555, 0.069, 0.016 | 0.5552, 0.0633, 0.0162 |
| mean only, `type='beta'`: a, b, LRT, tau2, mean (se) | 0.473, 4.459, 7.847, 0.0, 0.115 (0.166) | 0.4731, 4.4613, 7.8469, 0.0000, 0.1147 (0.1664) |
| `blind` moderator, unadjusted: not blinded (se), blinded (se), tau2 | 0.574 (0.098), 0.538 (0.090), 0.017 | 0.5731 (0.0925), 0.5397 (0.0868), 0.0163 |
| `blind` moderator, `type='beta'`: a, b, LRT, tau2 | 0.420, 5.097, 9.045, 0.0 | 0.4201, 5.0960, 9.0432 (9.0443 with `~ blind` coding), 0.0000 |
| `blind` moderator, `type='beta'`: not blinded (se), blinded (se) | 0.134 (0.171), -0.002 (0.199) | 0.1342 (0.1708), -0.0021 (0.1991) |

Every adjusted mean and standard error, a, tau2 and the mean-only LRT match; b, the moderator LRT, the unadjusted standard errors and the whole
unadjusted moderator model do not. The probe scripts were in the session scratchpad, not kept.
<!-- doc-counts: accept-end -->

Cause of the misses: not traced. Suspected: the article's own `nlminb` fit (tolerance 1e-4 in the supplement code) and its Hessian standard errors;
the beta-model misses are in the last digit, where the likelihood is flat (`docs/differences.md`).

Fix direction: write the matching values as published cases on `dat.baskerville2012` in `selmodel_published.toml` (locator: the "Example" section);
record each miss as a strict expected failure marked "not yet reviewed" in `agents/specs/correctness-publication-bias/mismatches.md` for the user to
judge. Keep the supplement cases, and say in their locator that the supplement's data are 19 of the article's 23 effects. This reverses the
2026-11-15 "dropped" entry for the article: update it in `mismatches.md`, the note on line 124 of `agents/specs/correctness-publication-bias/spec.md`,
the header comment of `selmodel_published.toml` ("the 2017 article itself was not available") and the Citkowicz entry of `NOTES.md`.

## Find the Hooper data behind Table 5 of Bagos and Nikolopoulos (2009)

<!-- doc-counts: accept-start probed on 2026-11-18 - a dated record, not a claim about the present -->
Narrowed on 2026-11-18 from "Find the data behind Table 5 of Bagos and Nikolopoulos (2009)" (`agents/tasks/bagos2009-table5/tasks-1.md`; decision
D2 (a) of `agents/reviews/order-todo-items-1-5-specs.md`, answered 2026-11-18): the ER-visit and school-absence columns now have cases; the
Hooper column (dietary fat and combined cardiovascular events) does not. Probed on 2026-11-18: the ten trials of Table 1 of Hooper et al. (2001,
BMJ 322:757) match Bagos and Nikolopoulos's person-years but carry 567 intervention and 638 control events against their 553 and 653, and a
fixed-effects Poisson fit gives an IRR of 0.858 against the printed 0.819; no swap of arms or rounding of the London corn/olive control events
reproduces it.
<!-- doc-counts: accept-end -->

Fix direction: find the full Cochrane review, Hooper L et al. (2001), Dietary fat intake and prevention of cardiovascular disease, Cochrane
Database Syst Rev CD002137, DOI 10.1002/14651858.CD002137 (the 2001 issue), and its per-trial events and person-years; repository copies at UEA
ePrints and LSHTM Research Online were not checked. If they reproduce Table 5's Hooper column, write its cases in
`src/tests/correctness/cases/glmm_published.toml` beside the `glmm.published.bagos2009.table5.*` cases.

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
