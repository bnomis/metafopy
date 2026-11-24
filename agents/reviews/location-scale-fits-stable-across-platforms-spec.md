# Review: the location-scale-fits-stable-across-platforms spec

**Review page:** https://claude.ai/artifact/CmGGzkSKzMSKFE7pnXqeVY
**Date:** 2026-11-24

**What is being reviewed and how.** A new spec, `agents/specs/location-scale-fits-stable-across-platforms/spec.md` (with `measurements.md`), for
the `TODO.md` item "CI fails three numerical tests that pass on a local macOS run". It makes the four rounding-dependent results behind the red
CI test jobs deterministic. Approve or flag the spec, and answer the three decisions; approving it confirms the library output changes D1 and D2
make.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-24 - a dated record, not a claim about the present -->
- Pass: CI run 45335133105 at `00f7fda` fails the test jobs on ubuntu, macOS and Windows.
- Pass: the local `.venv` matches `uv sync --locked --all-extras`, and the failing tests pass locally with CI's flags, so the cause is numeric.
- Pass: pre-commit passes on the spec, its measurements and the index row.
<!-- doc-counts: accept-end -->

## The items

### The spec: location-scale fits stable across platforms

- **File:** agents/specs/location-scale-fits-stable-across-platforms/spec.md (with measurements.md), version 1.0, proposed 2026-11-24
- **What it does:** For the TODO.md item "CI fails three numerical tests that pass on a local macOS run" (CI runs 45270718992 and 45335133105, 2026-11-23/24). Every failure is a result that depends on rounding on badly conditioned location-scale fits of dat.bcg, so the platform, and even the run, decides it. (1) The differences.md example: after the NA studies drop, study 4 has leverage 1; its HE residual is 0 in exact arithmetic; R computes -7.4e-15, snaps it to 0 and stops with 'No missing values allowed in alpha.init'; metafopy computes 2.3e-14, just over its snap, and fits locally but not on CI. (2) The vif.reestimate case and the REML case's Singular matrix: the test's own reference fit keeps degenerate points where tau2 underflows to 0 when the real optimum's score sits at the 1e-6 threshold (year uncentred, condition number about 1e11). (3) The REML vif value: metafopy's BFGS/Nelder-Mead stop moves the vif by up to 2.2e-6 relative under 1e-15 input noise; R itself is 2.6e-6 off. (4) The Windows exact permutest: not a tie; 77 of 1287 permuted models have no finite maximum and the reference counts a platform-dependent number of them as unfitted, which moves its bounds.
- **Other:** Choices made, not asked: the test's reference fit works in centred and scaled scale columns and keeps a point only when its information is positive definite (test-only, reversible); case 4 follows from that with no change of its own. Each case gives a local reproduction of the CI failure before the fix and the check after it (perturbing yi by 1e-15 noise, forcing the residual to 0, lowering the score threshold). A move of any other output past its tolerance, or a recall miss, stops the task. The closing step runs the affected tests; you merge and push with /ship; the task then checks the CI run and stops naming any still-failing test. Also found: on 8 permutations metafopy and R stop at the same lower local maximum than the reference; filed with /todo-add in the first task. Approving the spec is your confirmation of the library output changes D1 and D2 make.
- **Recommendation: approve.**

## Decisions

<!-- doc-counts: accept-start measured on 2026-11-24 - a dated record, not a claim about the present -->

### D1. A study of leverage 1 in the scale starting values: refuse with R's error, or fit?

**Background:** R stops because the zero residual of a leverage-1 study makes log(ei^2) -Inf in its start regression. On 2026-11-17 (D2 (a) of agents/reviews/order-todo-items-1-10-specs.md) you chose that metafopy does not reproduce a failure of R's optimiser setup, on the stated cause that the NA came from a dropped location column. That cause was wrong: metafopy fitted only because its residual is 2.3e-14 on the Mac, 1e-15 over the snap, and the fit it reached (log-likelihood 0.0858, Hessian not invertible) is one of several; another start gives -4.1456. So this reverses that decision on corrected facts.

**What it affects:** docs/differences.md (the section is removed), src/audit/data/known.toml (its entry removed), test_docs_examples.py; any log-link location-scale fit with a leverage-1 study.

**Why the recommendation:** The model carries no information on that study's scale, R refuses it, and any end point metafopy reached would depend on a start metafor does not have.

- **Refuse with R's error (Recommended)**: Residuals of studies with 1 - h < 1e-10 are set to 0, so metafopy raises R's error on every platform; the documented difference and its known.toml entry go.
- **Fit, leaving such studies out of the start**: A deterministic start metafor does not have; the end point (ll -4.1456) differs from the one documented today, and the differences entry is rewritten with the right cause.

### D2. metafopy's log-link optimiser stop: polish on the analytic score, or widen the test?

**Background:** BFGS with a numerical gradient and a Nelder-Mead polish stop at the objective's rounding. On the flat REML likelihood of ls.vif.REML.bcg2 that leaves the location vif 2e-6 to 4e-6 relative from the optimum on CI; R's nlminb is 2.6e-6 off. Newton steps on the analytic score from metafopy's stop reach 1e-10 (measured 2026-11-24). Precedent: the rma_mv REML stop spec (2026-11-17) added the same capped Newton polish.

**What it affects:** Every log-link location-scale output, by about 1e-6 relative (inside the existing R comparison tolerance); permutation and profile refits; each fit's run time by one analytic information per Newton step.

**Why the recommendation:** It fixes the cause, which also reaches the other flat location-scale fits recorded in known.toml (ls_year, ls_na), rather than hiding a platform-dependent output behind a wider tolerance.

- **Capped Newton polish in the library (Recommended)**: metafopy ends at the same point on every platform; outputs move by the stop's error, about 1e-6; ridge and boundary fits stay put, held by the cap.
- **Widen the test's location vif tolerance to 4e-5**: No library change; metafopy's output stays platform-dependent at about 1e-6.

### D3. How strict the exact permutation reference stays

**Background:** 77 of the 1287 permuted models of the case have no finite maximum, so their z depends on where an optimiser stops. After the reference-fit fix the bounds are 774 to 851 of 1287 (p 0.601 to 0.661); metafopy gives 794. On 8 regular permutations metafopy and R stop at a lower local maximum than the reference.

**What it affects:** ls_permutest_exact in src/tests/correctness/test_derived.py, both exact permutation cases.

**Why the recommendation:** Checking each permutation would tie this item to a separate question, which local maximum metafopy should report, that is filed on its own.

- **Bounds only (Recommended)**: A weaker but deterministic check; the local-maximum finding is filed separately with /todo-add.
- **Bounds plus per-permutation agreement**: Stronger; passes only once the 8 permutations are recorded as a mismatch in agents/specs/correctness-ls-extensions/mismatches.md.

<!-- doc-counts: accept-end -->

## Answers

Answered on 2026-11-24 on the review page (every answer the recommended option, no comments):

- The spec: approved, which confirms the library output changes of D1 and D2.
- D1: refuse with R's error: a study with 1 - h below 1e-10 has its residual set to 0 in the scale start; the `differences.md` section and its
  `known.toml` entry are removed. This replaces D2 (a) of `agents/reviews/order-todo-items-1-10-specs.md` (2026-11-17).
- D2: capped Newton polish on the analytic score after metafopy's log-link stop.
- D3: the exact permutation cases check the widened bounds only; the local-maximum finding is filed separately.
