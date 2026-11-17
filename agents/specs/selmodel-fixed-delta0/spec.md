# selmodel: a fixed first weight other than 1 on a step-function selection model

**Version:** 1.1
**Status:** approved
**Date:** 2026-11-17

`selmodel(type='stepfun', delta=[d0, ...])` with a fixed `d0` other than 1 fits a model that is neither the one Vevea and Hedges (1995) define nor
the one the user asked for: the first interval's weight is 1 in the normalising constant and `d0` for the observed p-value. This spec settles what
metafopy does with such a `delta`, and every record that has to follow. It is the sibling of
[selmodel: stepfun with prec](../selmodel-prec-stepfun/spec.md), whose cause (the same hard-coded first weight) and decision it follows.

### Motivation (the `TODO.md` item, dates and figures as filed)

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
Found on 2026-11-16 while closing `agents/specs/selmodel-prec-stepfun/spec.md`. `selmodel(type='stepfun', steps=[0.025, 0.5, 1], delta=[0.5, nan,
nan])` on an ML fit of `dat.bangertdrowns2004` gives the estimates of `delta=[1, nan, nan]` (beta 0.10569, delta 0.40141 and 0.27643) and an ll of
-27.1257 against -17.4217, in metafopy and in R metafor 5.2-1 alike (`R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript`). The
ll differs by `n1 * log(0.5)`, with `n1` the number of observed p-values in the first interval.
<!-- doc-counts: accept-end -->

Cause (from the item): the first interval's weight is hard-coded to 1 in the normalising constant, but the numerator uses the user's `delta[0]`, so a
fixed `delta[0]` only adds `n1 * log(delta[0])` to the ll and never moves the estimates (`.selmodel.ll.stepfun` in `R/misc.func.hidden.selmodel.r`,
`ll_stepfun` in `src/metafopy/models/sel_lik.py`).

### What the investigation found (measured 2026-11-17 on the development Mac, R 4.6.2, metafor 5.2-1 from `.audit/Rlib`)

<!-- doc-counts: accept-start measured on 2026-11-17 - a dated record, not a claim about the present -->
All on the ML fit of `dat.bangertdrowns2004` with `steps=[0.025, 0.5, 1]` (14, 23 and 11 p-values in the three intervals); metafopy and R agree to
every printed digit in every row.

| `delta` | beta | tau2 | delta estimate | ll | LRT (df) | p |
|---|---|---|---|---|---|---|
| `[1, nan, nan]` | 0.10569 | 0.02738 | 1, 0.40141, 0.27643 | -17.4217 | 1.6810 (2) | 0.4315 |
| `[0.5, nan, nan]` | 0.10569 | 0.02738 | 0.5, 0.40141, 0.27643 | -27.1257 | 0 (2) | 1 |
| `[2, nan, nan]` | 0.10569 | 0.02738 | 2, 0.40141, 0.27643 | -7.7176 | 21.0891 (2) | 2.6e-5 |
| `[1, 0.5, 0.25]` | 0.09380 | 0.03774 | fixed | -17.6927 | 1.1389 (0) | NA |
| `[0.5, 0.25, 0.125]` | 0.05103 | 0.01867 | fixed | -27.3968 | 0 (0) | NA |
| `[2, 1, 0.5]` | 0.15506 | 0.06113 | fixed | -8.8520 | 18.8203 (0) | NA |

- **With the other weights free, the item holds**: the estimates are those of `d0 = 1` and ll, LRT, AIC, BIC and AICc move by `n1 * log(d0)`. With
  `d0 = 2` the LRT reports selection at p 2.6e-5 where the model with `d0 = 1` gives p 0.43.
- **With every weight fixed, the estimates move, which the item did not say.** `[1, 0.5, 0.25]`, `[0.5, 0.25, 0.125]` and `[2, 1, 0.5]` give the
  same relative weights, which is all a step function states (Vevea and Hedges 1995, p. 424: "the weights are relative, not absolute"), yet three
  different beta (0.0938, 0.0510, 0.1551). metafor fits the relative weights `1, d1/1, d2/1` with `d0` replaced by 1, so `[0.5, 0.25, 0.125]` is
  fitted as `1 : 0.25 : 0.125`, not as the `1 : 0.5 : 0.25` the user wrote. A sensitivity analysis with a priori weights (Vevea and Woods 2005) is
  silently given other weights.
- **The publication.** Vevea and Hedges (1995, Psychometrika 60:419-435, pp. 424-425) write the density as `w(Y_i) phi(...) / (eta_i A_i)` with
  `A_i = sum_j omega_j B_ij`, `omega_1` in both, and remove the indeterminacy "by constraining the weight for the first p-value interval to be
  1.0". `?selmodel` says the same: "To make the model identifiable, we set delta_1 = 1." The independent likelihood of the correctness harness
  (`formulas.selection_loglik` with `formulas.step_weight`) is that density: at metafor's estimate for `[0.5, 0.25, 0.125]` it gives -19.4360, at
  the estimate of `[1, 0.5, 0.25]` -17.6927, so metafor's fit is not the maximum of the defining density.
- **Normalising with `d0`** (a probe that multiplies the first term of `A` by `delta[0]` in `ll_stepfun`): `[0.5, nan, nan]` gives beta 0.10569,
  ll -17.4217 and LRT 1.6810 as `d0 = 1`, and delta 0.2007, 0.13821 (the `d0 = 1` estimates times 0.5); its Wald p-values of the free weights,
  tested against 1, become 0.0000, where against `d0` they would match the `d0 = 1` fit. All three all-fixed rows give beta 0.09380, ll -17.6927.
- **Other step forms discard `d0` silently.** `type='stepfun', decreasing=True` and `type='stepcon'` overwrite `delta[0]` with 1 without a
  warning (R lines `if (decreasing) delta[1] <- 1` and `if (type == "stepcon") delta[1] <- 1`; `selmodel.py` does the same): `delta=[0.5, nan,
  nan]` returns delta 1, 0.40142, 0.27644 and ll -17.4217 in both languages.
- **`d0 = 0`** stops in the optimiser: R "Error during the optimization", metafopy `EstimationError: initial value in 'vmmin' is not finite`.
- **Nothing in metafor's documentation, tests or the audit uses it.** Every `delta` of a step function in `man/selmodel.Rd` and
  `tests/testthat/test_misc_selmodel.r` (the Vevea and Woods 2005 tables) starts with 1. The audit's only `delta` value is `[1, 0.5]`
  (`src/audit/data/domains.toml`), and every `options/selmodel/*/delta=[1,0.5]` case of run `20261115-160207` stops in both languages for want of
  `type`. No R parity fit in `src/tests/r/generate_selmodel.R` (`hack_stepfun_fixed` is `c(1, 0.5, 0.2)`, `hack_stepfun_partial` `c(NA, NA, 0.3)`)
  and no correctness case (`selmodel.fixed_delta.cohen1981` is `[1.0, 0.9, 0.7, 0.5]`) has a first weight other than 1.
- **Internal refits are safe.** `refit` in `src/metafopy/methods/selmodel.py` (profiles and `confint`) passes the fit's fixed `delta`, whose first
  value is always 1 for a step function, so a check on `delta[0]` never fires there.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/models/selmodel.py`: the check, where `delta` is read and checked against its bounds; the `delta` docstring and `Raises`.
- Option (b) only: `src/metafopy/models/sel_lik.py` (`ll_stepfun`), `src/metafopy/models/selmodel.py` (the Wald null of the free weights).
- Tests and records: `src/tests/test_selmodel.py`, `docs/differences.md`; `src/tests/correctness/cases/selmodel.toml` under option (b) only;
  `src/audit/data/known.toml` only if the audit flags the change.

## API

No new function and no signature change under any option. Under the recommended option (a) one set of arguments becomes an error:

```python
def selmodel(
    x: Model,
    type: str,
    alternative: str = 'greater',
    prec: str | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    delta: typing.Any = None,
    steps: typing.Any = None,
    decreasing: bool = False,
    verbose: bool | int = False,
    digits: int | None = None,
    control: collections.abc.Mapping[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> metafopy.models.sel_result.SelmodelResult | pandas.DataFrame | FloatArray:
    """...
    Args:
        delta: Fixed values of the selection parameters, one per parameter of the chosen type (``None`` or ``nan`` elements are estimated); a wrong
            length or a value outside the parameter space raises ``InputError``. For ``'stepfun'`` and ``'stepcon'`` the first value is the weight
            of the first p-value interval, fixed at 1 to identify the model (Vevea & Hedges, 1995): it must be 1 or ``nan``, and the other values
            are weights relative to it (``docs/differences.md``, "Where metafor departs from its reference").
    Raises:
        metafopy.errors.InputError: ... a first ``delta`` other than 1 or ``nan`` for ``type='stepfun'`` or ``'stepcon'``.
    """
```

## Error handling

- Option (a): `metafopy.errors.InputError("The first value of 'delta' must be 1 or nan for this type of selection model: the weight of the first
  p-value interval is fixed at 1, so give the other weights relative to it (see docs/differences.md).")`, raised in `selmodel.py` right after
  `check_bounds('delta', delta_arg, spec)` and before the lines that set `delta_arg[0] = 1.0`, so before any fitting. It fires when `type_` is
  `'stepfun'` or `'stepcon'` (the scope of the choice "Which step-function forms the check covers" below) and `delta_arg[0]` is neither `nan` nor exactly 1.0. A value outside the parameter
  space keeps its existing bounds error, which comes first (for example `delta[0] = 2` with `decreasing=True`: "must be <= 1"). `d0 = 0` on
  `'stepfun'` now raises this `InputError` instead of the optimiser's `EstimationError`. The message names "the first value", not an index, so it
  is right under metafopy's 0-based convention and metafor's 1-based `delta[1]` alike.
- Option (b): no new error. Option (c): no new error.

## Implementation

**Option (a), decided 2026-11-17.** In `selmodel.py`, after the bounds check of `delta`:

```python
if type_ in metafopy.models.sel_spec.STEPWISE and not math.isnan(delta_arg[0]) and delta_arg[0] != 1.0:
    raise metafopy.errors.InputError(...)  # the message above
```

The two existing lines that set `delta_arg[0] = 1.0` stay (for a `nan` first value, and for `decreasing` and `'stepcon'`, where the value is now
1 or `nan` already). Update the `delta` docstring and `Raises` as in the API. `ll_stepfun` and `w_step` are left as they are: with `delta[0] = 1`
the numerator and the constant agree and the model is Vevea and Hedges's. Settled without asking: exact comparison with 1.0, not a tolerance,
because the value is the user's literal and `1` and `1.0` are both exact; raising, not normalising silently, because a silent rescale would
report weights the user did not write.

**Option (b).** In `ll_stepfun`, multiply the first-interval term of `A` by `delta[0]` for each `alternative` (`'greater'`: `A = delta[0] *
tail_up(...)`; `'less'` likewise; `'two.sided'`: both tails of the first term), which is equation 5 of Vevea and Hedges. In `selmodel.py`, test
the free weights of a step function against `delta[0]` instead of `spec.h0_delta` (the `zval_delta` line), so the Wald tests keep their meaning.
`decreasing` and `'stepcon'` keep overwriting `delta[0]` with 1 unless that choice says otherwise.

**Option (c).** No change to `src/metafopy/`.

## Decisions

### D1. What metafopy does with a fixed first weight other than 1 on `type='stepfun'`

**What is decided:** whether `selmodel(type='stepfun', delta=[d0, ...])` with `d0` neither `nan` nor 1 raises, fits Vevea and Hedges's density
with `d0`, or keeps metafor's fit.

**Background:** see the investigation. The rule since 2026-11-11 (`claude/rules/metafopy.md`) is that the defining publication is the first
oracle and metafor the second, and that metafopy's output changes only after the user confirms it; this review is that confirmation. The
publication defines the density for any `omega_1` but identifies it by fixing `omega_1 = 1`; metafor's documentation says the same, and no
example, test or audit case of metafor uses another value. The precedent: on 2026-11-15 the user chose to reject `type='stepfun'` with `prec`
with `InputError` (D3 in `agents/reviews/order-numerical-fixes-specs.md`), for the same hard-coded first weight, rather than keep metafor's
non-density or define a model of metafopy's own.

**What it affects:** `src/metafopy/models/selmodel.py` (all options but (c)), `sel_lik.py` (option (b)), `src/tests/test_selmodel.py`,
`docs/differences.md`. No R parity fixture and no correctness case uses a first weight other than 1, so none changes under any option; the audit
does not reach the combination, so no `known.toml` entry is expected.

| Option | Outputs that change | What happens next |
|---|---|---|
| **(a) Reject (recommended)** | `selmodel(type='stepfun', delta=[d0, ...])`, `d0` not 1 or `nan`: `InputError` where it now returns a fit (or, for `d0 = 0`, `EstimationError`). Nothing else. | The check above, two tests, a `docs/differences.md` entry. A user who wrote `[0.5, 0.25, 0.125]` is told to write `[1, 0.5, 0.25]`, which fits the weights they meant. |
| (b) Normalise with `d0` | For those calls: ll, LRT, LRTp, AIC, BIC, AICc (now those of `d0 = 1` when the others are free); the free delta estimates, their SEs and CIs (scaled by `d0`); their Wald tests (null `d0`); with every weight fixed, beta, tau2, SEs and everything after (the fit of the relative weights). Plots of the selection function show `d0` as written. | The changes under Implementation, a regression test against the scaled `d0 = 1` fit, a correctness case `selmodel.fixed_delta0.bangertdrowns2004` (`delta = [0.5, 0.25, 0.125]`, `test_derived.selmodel_fit`) and a `docs/differences.md` entry. metafopy reports a first weight other than 1, which neither the paper nor metafor shows, and R parity breaks for these calls. |
| (c) Keep metafor | None. | A `docs/differences.md` entry under "R's quirks copied on purpose" saying the first weight is taken as 1 in the constant, and a docstring note. metafopy keeps reporting an LRT that moves with `d0` and all-fixed fits of weights the user did not write. |

**Recommendation: (a).** It follows the precedent: the publication fixes the first weight at 1, so a model with another fixed value is not one
to fit; (c) keeps a log-likelihood that is not a density and silently changes a priori weights; (b) is faithful to equation 5 but makes the
output depart from metafor in many places (estimates, tests, plots) for a parameterisation nobody documents, when the user can get the same model
by dividing by `d0`. Rejecting removes the wrong answer, costs one line of user code, and is reversible: (b) can be added later.

**Decided: (a)**, by the operator on 2026-11-17 on the review page of `agents/reviews/order-todo-items-1-10-specs.md` (no comment). Version 1.1
records the answer; the options (b) and (c) stay above as the record of what was weighed.

## Choices made

Made while writing the spec and reported, not asked: it follows from D1 (a) and is reversible by amending this spec.

### Which step-function forms the check covers

**What it settles:** under (a), whether the error also covers `type='stepfun', decreasing=True` and `type='stepcon'`, where metafor (and
metafopy) silently replace `d0` with 1 and fit the defined model.

**Background:** for those forms the fit is Vevea and Hedges's with `omega_1 = 1`, so the numbers are right for the model fitted; what is wrong is
that the user's `d0` is dropped without a word. In the `prec` decision those two forms kept metafor's behaviour because metafor warns there; here
metafor says nothing.

**What it affects:** one condition in the check (`type_ in STEPWISE` against `type_ == 'stepfun' and not decreasing`), one parametrisation of the
regression test, one sentence of the `docs/differences.md` entry.

| Option | What happens next |
|---|---|
| **(i) Every step form (recommended)** | `'stepfun'` with or without `decreasing` and `'stepcon'` raise the same `InputError`. One rule for every step function: the first value is 1 or `nan`. |
| (ii) `'stepfun'` with `decreasing=False` only | `decreasing=True` and `'stepcon'` keep metafor's silent replacement; the docstring says so. |
| (iii) Warn on the other two | They warn "The first value of 'delta' is set to 1 for this type of selection model." and fit as now. |

**Chosen: (i).** A value the user wrote and the program throws away is a wrong answer they cannot see; metafopy already raises where R
passes on misuse (`docs/differences.md`, "Stricter than R"), and one rule for all step forms is simpler to document and to test than three.

## Tests and records

For option (a) with the choice (i); other answers adjust as the tables say.

1. `src/tests/test_selmodel.py`: `test_stepfun_first_delta_must_be_one`, parametrised over `(type, decreasing)` in `('stepfun', False)`,
   `('stepfun', True)`, `('stepcon', False)` and `d0` in `0.5` and `0.0`, plus `2.0` for `('stepfun', False)` (the other two forms bound
   `delta[0]` by 1, so `2.0` keeps its bounds error there), on `bangertdrowns_ml()` with `steps=[0.025, 0.5, 1]` and `delta=[d0, nan, nan]`,
   asserting `InputError` matching `"first value of 'delta' must be 1"`. Against the old code it fails for every
   parameter: a fit is returned, or `EstimationError` for `d0 = 0` on `'stepfun'`. Add `test_stepfun_first_delta_one_or_nan_fits`:
   `delta=[1, nan, nan]` and `[nan, nan, nan]` give the same fit (beta 0.10569 and ll -17.4217 to 1e-4, the measured values), and
   `[1, 0.5, 0.25]` fits (beta 0.09380). Existing parity fits stay as they are (`hack_stepfun_fixed`, `hack_stepfun_partial`).
2. `docs/differences.md`, "Where metafor departs from its reference": an entry `### selmodel(type="stepfun") with a fixed first weight`, after the
   `prec` entry and in its format (summary, R and Python side by side, `*Why:*` with the check date). The summary names the forms the choice covers.
   The R side shows `selmodel(res, type="stepfun", steps=c(0.025, 0.5, 1), delta=c(2, NA, NA))` on the ML fit of `dat.bangertdrowns2004` with its
   printed output, run on the day the entry is written, and the line "With `delta=c(1, NA, NA)` the log-likelihood is -17.4217 and LRT 1.6810."
   The Python side starts `# docs: skip` and shows the `InputError`. The `*Why:*` gives the constant `n1 * log(d0)`, the all-fixed case, Vevea and
   Hedges (1995, pp. 424-425) and the decision with its date.
3. The `TODO.md` item is moved to `DONE.md` with the decision and its date (`/todo-do` convention).
4. Run `pytest src/tests/test_selmodel.py src/tests/correctness src/tests/test_docs_examples.py`, then `ruff format`, `ruff check` and `mypy`.
5. Run `bin/metafor-audit.sh` and `bin/metafor-audit.sh --recall` with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (worktrees have
   no `.audit/Rlib`). If the audit flags `selmodel` with a first `delta` other than 1, add a `known.toml` entry with `ref = "docs/differences.md"`
   citing the decision and its date; otherwise none.

No R parity fixture changes and `src/tests/r/generate_selmodel.R` is not rerun. No step needs a stop for the operator once D1 is answered
in the review of this spec.

## Acceptance

- `selmodel` raises the `InputError` above for each form the choice covers and fits as before for a first value of 1 or `nan`; both new tests pass and the
  first fails on the commit before the change.
- Every existing test in `src/tests/test_selmodel.py` and `src/tests/correctness` passes unchanged, with no fixture edited.
- The `docs/differences.md` entry exists and `test_docs_examples.py` passes; the `delta` docstring and `Raises` say the same as the entry.
- `ruff format --check`, `ruff check`, `mypy` and the audit with `--recall` pass, with no new unaccepted finding.

## External dependencies

None new. R 4.6.2 with metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib` for the `docs/differences.md` example and the audit.
