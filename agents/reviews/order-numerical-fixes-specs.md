# Review: the specs of the numerical fixes order

**Date:** 2026-11-15
**Order:** `agents/tasks/order-numerical-fixes.md` (not written yet; it follows this review)
**Review page:** https://claude.ai/artifact/SKRD9gTc7XQk4iptCsiqYp

## What is reviewed and how

`/todo-run-order 1-8` takes the first eight `TODO.md` items (as of `c04fae2`) to a run order. Six are small and become task lists with the item as
their source, no spec: the audit entries for the no-intercept difference (item 1), the fail-safe N integer search (item 2), the truncest piecewise
search (item 3), the location-scale identity-link polish (item 5), the location-scale profile crossings (item 6) and the R SJIT convergence check
(item 8). Two are large and became specs on 2026-11-15: the CM.EL analytic information (item 4) and the precision measure on a step-function
selection model (item 7). Approve or flag the plan and each spec, and answer each decision. The task lists are written only after every answer is
in, because the answers change the specs.

## Automatic checks

- Each spec has the specdev sections (introduction, Module location, API, Error handling, Implementation, External dependencies) and an open-decisions
  section: pass.
- Each spec carries its `TODO.md` item's dated findings unchanged: pass (checked by reading both specs, 2026-11-15).
- R metafor 5.2-1 loads from `/Users/simonb/src/metafopy/.audit/Rlib` (checked 2026-11-15 13:48 UTC); both specs tell every task to set
  `METAFOPY_AUDIT_RLIB` to that path, because worktrees have no `.audit/Rlib`: pass.
- Size: `cm-el-information` is 206 lines, a little over the specdev guide of about 200; noted, not a failure.

## The items

### The plan: six small task lists, no spec

| TODO item | Task list | Waits for | Why small |
|---|---|---|---|
| 1. No audit entries record the no-intercept QE and tau2 difference | `audit-no-intercept` | - | one full audit run and `known.toml` entries; no code change |
| 2. The fail-safe N root search rounds up one too far | `fsn-integer-search` | - | local to `src/metafopy/bias/fsn.py`; the item's fix direction allows a small list |
| 3. The truncest estimate stops at a local maximum | `truncest-piecewise-search` | - | local to the `truncest` branch of `src/metafopy/models/selmodel.py` |
| 5. The location-scale identity-link fit stops short | `ls-identity-polish` | - | local to `src/metafopy/models/rma_ls_fit.py`; its first task confirms the suspected cause |
| 6. Profile-likelihood intervals of the scale coefficients | `ls-profile-crossings` | 5 | local to `src/metafopy/methods/ls_profile.py`; reads the fit that list 5 polishes |
| 8. Whether R's SJIT converges on the scaled data | `sjit-scaled-convergence` | - | one R check, then a fix in `src/metafopy/models/rma_uni.py` or a close |

Items 2, 3, 5 and 6 were answered *follow the reference* on 2026-11-15, so their output changes are already confirmed. Each list sets its cases to
`pass`, marks the R parity fixtures that differ, adds `known.toml` entries and a `docs/differences.md` entry. Recommendation: approve.

### `agents/specs/cm-el-information/spec.md` - the analytic CM.EL information

Replaces the numDeriv Hessian (`r = 16`) of `rma_glmm(model='CM.EL')` by the analytic information of the exact conditional likelihood for the `FE`
fit, the saturated fit behind `QE_Wld` and the `ML` fit (posterior moments). Coefficients, tau2 and log-likelihoods do not change. Measured on
2026-11-15: the analytic values equal the correctness formulas to about 1e-9, and both metafopy and R reproduce them with `hessianCtrl = list(r = 4)`;
the departures come from Richardson extrapolation over 16 halvings. The seven CM.EL cases pass; parity comparisons that fail are marked differing.
Recommendation: approve.

### `agents/specs/selmodel-prec-stepfun/spec.md` - a precision measure on a step-function selection model

Settles what `selmodel(type='stepfun', prec=...)` does. Found on 2026-11-15: metafor's estimates are the maximum of a consistent density (weight 1 in
the first interval, `delta[j] / preci` elsewhere), but its ll carries the constant `sum(-log(preci))` over the first-interval studies, which inflates
the LRT (28.32 against 7.59 for `sei` on `dat.bangertdrowns2004`), and no publication defines the model. A related finding (a fixed `delta[0]` other
than 1 only shifts ll) is filed as its own `TODO.md` item by the list. Recommendation: approve.

## Decisions

### D1. The CM.EL `ML` standard error

- **What is decided:** whether the `ML` standard error (and `se_tau2`, the tau2 interval) follows the analytic information, as the `FE` one does.
- **Background:** the user answered *investigate* on 2026-11-14 because metafopy, the formulas and R disagreed. The 2026-11-15 measurement shows the
  formulas' value is the observed information to about 1e-9; metafopy is off by numDeriv's `r = 16` (6.4e-4 relative on `dat.nielweise2007`).
- **What it affects:** the three `ML` cases, `marginal_information` and its tests, the `ML` parity comparisons.
- **Options:** *follow the reference* - the ML part is built and the three cases pass; *keep metafor* - the ML part is dropped, the cases become a
  recorded difference with the measurement as the reason.
- **Recommendation:** follow the reference; the measurement removes the doubt behind *investigate*, and it matches the `FE` answer.

### D2. The CM.EL Hessian control entries

- **What is decided:** what `hesspack`, `hessianCtrl`, `dnchgcalc` and `dnchgprec` do once the information is analytic.
- **Background:** metafor documents them as choosing the numerical Hessian; metafopy already departs from R's default density (`dnchgcalc`) and keeps
  R's behaviour behind a `control` entry.
- **What it affects:** the default of `hesspack` (`None` instead of `'numDeriv'`), two tests in `test_audit5_numerics.py`, `docs/guide/binary.md`.
- **Options:** *opt-in numerical* - analytic by default, today's numerical Hessian when any entry is given (two code paths, both tests stay valid);
  *always analytic* - the entries are accepted but change nothing, the numerical code goes, the two tests are rewritten.
- **Recommendation:** opt-in numerical, so a caller who asks for metafor's Hessian gets it, as with `dnchgcalc` today.

### D3. `selmodel(type='stepfun', prec=...)`

- **What is decided:** what the combination does (with `decreasing=False`).
- **Background:** the user answered *investigate* on 2026-11-15. No publication defines a precision-weighted step function; metafor warns that it is
  undocumented and experimental, and its ll and LRT carry the constant above. The audit never reaches the combination.
- **What it affects:** the `selmodel` API, the parity fit `hack_stepfun_sei`, the four `selmodel.prec.*` cases (the only cover of the
  `selmodel.prec.*` coverage items), `docs/differences.md`.
- **Options:** *(a) reject* - `InputError`; `hack_stepfun_sei` leaves the parity tests for a test that it raises; the four cases are replaced by
  `halfnorm` cases with each `prec` (Preston et al. 2004), which passed a probe on 2026-11-15. *(b) keep metafor* - no code change; the cases become
  exemptions, `halfnorm` cases carry the coverage, and metafopy keeps an LRT that overstates selection. *(c) define it as the consistent density* -
  `ll_stepfun` drops the constant, LRT becomes NaN (no nested null), a model metafopy invents without a source.
- **Recommendation:** (a); nothing defines the model, (b) keeps a likelihood that is not a density, (c) invents one; rejecting is reversible.

## Prerequisites

None for the operator to provide. R metafor 5.2-1 and its pinned library are already at `/Users/simonb/src/metafopy/.audit/Rlib`, and every list that
needs R (1, 8, the CM.EL audit slice, the `docs/differences.md` examples) points at it. No paper is needed: the specs cite sources already in
`/Users/simonb/src/metafopy papers`.

## Answers

Answered on the review page on 2026-11-15 (14:28 UTC), no comments:

| Item or decision | Answer |
|---|---|
| The plan: six small task lists, no spec | approve |
| `agents/specs/cm-el-information/spec.md` | approve |
| `agents/specs/selmodel-prec-stepfun/spec.md` | approve |
| D1. The CM.EL `ML` standard error | follow the reference |
| D2. The CM.EL Hessian control entries | opt-in numerical |
| D3. `selmodel(type='stepfun', prec=...)` | (a) reject with `InputError` |

Applied to both specs on 2026-11-15; each is at version 1.1.
