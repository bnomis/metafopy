# Correctness tests for the location-scale extensions: mismatches

**Task list:** `agents/tasks/correctness-ls-extensions/tasks-1.md`
**Spec:** `agents/specs/correctness-ls-extensions/spec.md`
**Review page:** published by the operator session when it handles this review

## What is reviewed and how

Each item below is a `derived` case whose reference, coded from the cited publication, disagrees with metafopy. The case is committed with
`status = "mismatch"` (a strict expected failure) and a reason naming both values. For each item you choose whether metafopy follows the reference,
keeps metafor's behaviour, or the difference is investigated further. The two items below are the only mismatches of this work.

## Automatic checks

Checked 2026-11-20:

- **Pass:** `pytest src/tests/correctness -m 'not slow'` passes: the two cases below are strict expected failures and nothing else fails (exit status 0 at 12:22 UTC).
- **Pass:** every `status = "mismatch"` case added by this task list has an item below; the two are `ls.alpha.bounds.ML.bcg.wide` and
  `ls.profile.ML.identity.bcg`, both in `src/tests/correctness/cases/location_scale.toml`.

## Mismatches

### ls.alpha.bounds.ML.bcg.wide: a box that does not bind, and metafopy returns a non-maximum

- **Case:** `ls.alpha.bounds.ML.bcg.wide` (covers `ls.alpha.bounds`).
- **Source:** Viechtbauer and Lopez-Lopez (2022), Res Synth Methods 13:697-715, doi:10.1002/jrsm.1562, equation 7 or 9 maximised inside the box
  `alpha_min`, `alpha_max` of the control settings (metafor 5.2-1, help page `rma.uni`); the reference is `formulas.ls_constrained_fit`, the
  L-BFGS-B maximum within the box.
- **Data and model:** `dat.bcg` with `escalc` RR, `mods = ~ ablat`, `scale = ~ ablat`, ML, log link, box (-10, -1.2) to (0, 1.2).
- **Reference:** alpha (-4.95749, 0.03859), log-likelihood -7.5393, the unconstrained optimum, which lies inside the box.
- **metafopy:** alpha (0, -0.0795806), log-likelihood -8.6762, with only the "(almost) equal to their bound" warning: a non-maximum 1.14 below the
  reference.
- **metafor:** R metafor 5.2-1 stops with "Optimizer (nlminb) did not achieve convergence". It returns no fit to compare.
- **Cause:** the start (the heterogeneity-estimator-based start, clipped onto the upper bound) sits where the logistic map used for the box sends
  it to infinity. A start inside the box reaches the reference. Wider boxes take both programs to the `tau2 = 0` plateau (log-likelihood -9.4736).

### ls.profile.ML.identity.bcg: the identity-link profile refits end a hair below the reference

- **Case:** `ls.profile.ML.identity.bcg` (covers `ls.profile.identity`).
- **Source:** Viechtbauer and Lopez-Lopez (2022), sections 2.5.3 and 2.6, the profile likelihood of each scale coefficient (equations 17 and 18);
  the reference is `formulas.ls_profile_ll`.
- **Data and model:** `dat.bcg` with `escalc` RR, `mods = ~ ablat`, `scale = ~ ablat`, ML, identity link.
- **Reference:** profile log-likelihood -8.255095081966374 at alpha[0] = -0.133293.
- **metafopy:** -8.255095092551993, 1.06e-08 below the reference; the tolerance is 1e-09 of the largest absolute log-likelihood (8.3e-09 allowed).
- **metafor:** every alpha[0] refit fails ("All model fits failed") and its alpha[1] values lie 1.3e-08 to 1.2e-07 below the reference.
- **Cause:** the optimum of these refits lies on `tau2_i = 0`, where the identity link's feasible region ends; the polish that follows the
  identity-link fit stops a few 1e-9 short there.
- **Earlier answer, restated:** on 2026-11-15 the review of the location-scale work answered *follow the reference* for the identity-link fits
  stopping short of the maximum (item `ls_identity_opt` in `agents/specs/correctness-location-scale/mismatches.md`), since implemented as the
  identity-link polish. That is not asked again; only the remaining shortfall is.

## Decisions

### D1: ls.alpha.bounds.ML.bcg.wide, what does metafopy do when the start lies on the bound?

- **What is decided:** whether metafopy moves its start inside the box so that a non-binding box returns the maximum, or keeps metafor's behaviour.
- **Background:** metafopy's start is clipped onto the upper bound, where the logistic map used for the box sends it to infinity, so the optimiser
  stops at a non-maximum and warns only that coefficients are near their bound. metafor itself fails on this case, with an error, rather than
  return a wrong answer; metafopy returns a wrong answer with a warning. The other three box cases (binding boxes) agree to 6.8e-09.
- **What it affects:** the start code of the location-scale fit with `alpha_min` and `alpha_max`; the case's `status`; `docs/differences.md` (metafor
  errors here, metafopy would return the maximum).
- **Options:**
  - *Follow the reference (Recommended):* a `TODO.md` item moves the start strictly inside the box, flips the case to a passing one, marks the R
    parity fixtures if any exist for it, adds a `known.toml` entry for the case metafor cannot fit, and a `docs/differences.md` section.
  - *Keep metafor:* the case's `reason` gains "accepted as a difference from the reference on <date>", it stays a strict expected failure, and no
    code changes. metafopy keeps returning a non-maximum with only a warning.
  - *Investigate:* a `TODO.md` item records the figures above, and the case stays an expected failure.
- **Recommendation:** follow the reference. metafopy's answer is not the maximum whatever metafor does (the same reasoning as the 2026-11-15 answer
  for the identity link), and the cause is a start that can be moved without changing any result that is already correct.

### D2: ls.profile.ML.identity.bcg, does the remaining shortfall need a further fix?

- **What is decided:** whether the 1.06e-08 shortfall of metafopy's identity-link profile refits is fixed further, or accepted.
- **Background:** the identity-link polish answered on 2026-11-15 was implemented; one case still falls 2.2e-09 outside the tolerance, where the
  optimum lies on the boundary `tau2_i = 0`. R metafor is further from the reference than metafopy (up to 1.2e-07 below, and every alpha[0] refit
  fails), so metafopy is the closer of the two.
- **What it affects:** the identity-link fit code, or the case's `reason`; no other case is outside its tolerance.
- **Options:**
  - *Keep metafor:* the case's `reason` gains "accepted as a difference from the reference on <date>"; the case stays a strict expected failure and
    no code changes. metafopy stays closer to the reference than R metafor.
  - *Follow the reference:* a `TODO.md` item tightens the identity-link polish at the boundary, flips the case, and adds a `known.toml` entry for
    the alpha[0] refits metafor cannot fit.
  - *Investigate:* a `TODO.md` item records the figures above, and the case stays an expected failure.
- **Recommendation:** keep metafor. The shortfall is 1e-08 in a log-likelihood of about 8, on a boundary optimum, and metafopy is already closer to
  the reference than the program it ports; the case's reason records the difference either way.

## Answers

Left empty for the review: the operator session fills it in with the date.

## Sources needed

None: every source this list cites was present in the papers folder `/Users/simonb/src/metafopy papers/` (Task 5 of the task list).
