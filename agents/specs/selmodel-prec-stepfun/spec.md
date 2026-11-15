# selmodel: a precision measure on a step-function selection model

**Version:** 1.1
**Status:** approved (2026-11-15, `agents/reviews/order-numerical-fixes-specs.md`: option (a), reject)
**Date:** 2026-11-15

`selmodel(type='stepfun', prec=...)` fits a step-function selection model whose weights also depend on each study's precision. No publication
defines that model, metafor warns that it is "undocumented and experimental", and its likelihood is not a density. This spec settles what
metafopy does with the combination, and every record that has to follow.

### Motivation (the `TODO.md` item, dates and figures as filed)

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the cases `selmodel.prec.{sei,vi,ninv,sqrtninv}.bangertdrowns2004` in `src/tests/correctness/cases/selmodel.toml`
(`agents/specs/correctness-publication-bias/mismatches.md`); the user answered *investigate* on 2026-11-15. With `type='stepfun'` and `prec` set,
metafopy gives beta 0.13780 (`sei`), 0.16908 (`vi`), 0.16769 (`ninv`) and 0.13711 (`sqrtninv`); the maximum of the density with weights `delta[j] /
preci` gives beta 0.10569 and ll -17.4217 for each. R metafor 5.2-1 gives metafopy's values (operator session, 2026-11-15), and warns that the
combination is undocumented and experimental.
<!-- doc-counts: accept-end -->

Cause (from the item): metafopy, like metafor, normalises the first interval with weight 1 but weights the observed p-value by `delta[0] / preci`, so
the likelihood is not a density and its maximum moves with the precision measure. The density with `delta[j] / preci` in every interval makes `prec` a
no-op (the per-study factor cancels between numerator and normalising constant), so neither is the intended model.

### What the investigation found (checked 2026-11-15, R 4.6.2, metafor 5.2-1 from `.audit/Rlib`)

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
- **Source.** metafor's `R/selmodel.rma.uni.r` gives the step weight `delta[j] / preci` in every interval (`wi.fun`), but `.selmodel.ll.stepfun`
  (`R/misc.func.hidden.selmodel.r`) starts the normalising constant `A_i` with weight 1 for the first interval and `delta[j] / preci[i]` for the
  others. `?selmodel` names `prec` only for the continuous types (Preston et al. 2004), and says of them that with `steps`, "studies with p-values
  below the chosen threshold then have w(p_i) = 1 regardless of their exact p-value or precision". Vevea and Hedges (1995, Psychometrika 60:419-435,
  pp. 424-425) fix the first weight at 1 to remove the indeterminacy and note that weights could depend on study characteristics, without defining such
  a model; Vevea and Woods (2005, Psychological Methods 10:428-443) cite Copas's standard-error model only, which is not a step function. Hedges (1992)
  and Iyengar and Greenhouse (1988) are scanned PDFs in `/Users/simonb/src/metafopy papers` without a text layer; the earlier review found no
  precision measure in either. No source defining a precision-weighted step function was found.
- **metafor's estimates are the maximum of a consistent density, its log-likelihood is not.** Call density C the one with weight 1 in the first
  interval and `delta[j] / preci` in the others (numerator and `A_i` alike). metafor's (and metafopy's) beta, tau2 and delta are C's maximum: an
  independent Nelder-Mead maximisation of `formulas.selection_loglik` with C's weight agrees to 1e-6 for each `prec` on `dat.bangertdrowns2004`. The
  reported ll exceeds C's by the parameter-free constant `sum(-log(preci_i))` over the studies in the first interval: ll -4.1029 against C's -14.4689
  (`sei`), 7.5714 against -13.1608 (`vi`), 7.1307 against -13.6829 (`ninv`), -4.3548 against -14.7616 (`sqrtninv`). Standard errors, confidence
  intervals, profiles and `LRT.tau2` are unaffected (the constant has no derivative and cancels at tau2 = 0); ll, `LRT`, `LRTp`, deviance, AIC, BIC
  and AICc are shifted by it.
- **The LRT is inflated, and under C it is not defined.** metafor reports LRT 28.3186 (p 7.1e-7, `sei`) and 51.6670 (`vi`) against `ll0` -18.2622;
  with C's ll they are 7.5864 (p 0.0225) and 10.2028. No value of delta makes C the model without selection (delta = 1 still leaves weight
  `1 / preci`), so the test is not of nested models: on `dat.hackshaw1998` (`hack_stepfun_sei`, steps 0.025) C's ll is -10.6408, below `ll0` -10.1486,
  while metafor reports ll -2.0493 and LRT 16.1986 (p 5.7e-5).
- **Related, outside this spec.** The same hard-coded first weight makes a user-fixed `delta[0]` other than 1 a constant: `delta=[0.5, nan, nan]` on
  `dat.bangertdrowns2004` gives the estimates of `delta=[1, nan, nan]` (beta 0.10569, delta 0.40141, 0.27643) and an ll lower by `n1 * log(0.5)`
  (-27.1257 against -17.4217), in R and metafopy alike.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/models/sel_spec.py` (where the `'stepfun'` branch warns "undocumented and experimental"), `src/metafopy/models/selmodel.py` (the
  `prec` docstring), `src/metafopy/models/sel_lik.py` (`ll_stepfun`, option (c) only).
- Tests and records: `src/tests/test_selmodel.py`, `src/tests/correctness/cases/selmodel.toml`, `src/tests/correctness/formulas.py` (option (c) only),
  `docs/differences.md`, `src/audit/data/known.toml` (only if the audit flags the change).

## API

Under the recommended option (a) the signature of `selmodel` is unchanged; one combination of arguments becomes an error.

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
        prec: Measure of precision the selection function may depend on: ``'sei'``, ``'vi'``, ``'ninv'`` or ``'sqrtninv'`` ... It is ignored with
            a warning for ``'beta'``, ``'trunc'``, ``'truncest'``, ``'stepcon'`` and ``'stepfun'`` with ``decreasing=True``, and raises ``InputError``
            for ``'stepfun'`` otherwise: no publication defines a step function with a precision measure, and metafor's version is not a density
            (``docs/differences.md``, "Where metafor departs from its reference").
    Raises:
        metafopy.errors.InputError: ... or ``prec`` with ``type='stepfun'`` and ``decreasing=False``.
    """
```

## Error handling

- Option (a): `metafopy.errors.InputError("Argument 'prec' cannot be used with type='stepfun': a step-function selection model with a precision
  measure is not defined (see docs/differences.md).")`, raised in the `'stepfun'` branch of `sel_spec.py` where metafor warns, before any fitting.
  `decreasing=True` and `'stepcon'` keep metafor's warning "Argument 'prec' ignored (not applicable to this type of selection model)." and fit
  without `prec`, as now: that path is metafor's documented one and is a density.
- Option (b): no new error; the existing warning stays.
- Option (c): no new error; the warning stays, its text extended to say the LRT is not available.

## Implementation

**Option (a), recommended.** In `sel_spec.py`, replace the `warnings.warn('Adding a precision measure to this selection model is undocumented and
experimental.', ...)` of the `'stepfun'` branch by the `InputError` above. Update the `prec` docstring and `Raises` of `selmodel` in `selmodel.py`.
`ll_stepfun` is left as it is (with `prec` unset `preci` is 1 and the constant is 0). Settled without asking: raising, not ignoring with a warning,
because ignoring would return the plain step-function fit labelled as a precision model, and metafopy already raises where R passes on misuse
(`docs/differences.md`, "Stricter than R").

**Option (b).** No change to `src/metafopy/`.

**Option (c).** In `ll_stepfun`, weight the numerator of a study in the first interval by `delta[0]` instead of `delta[0] / pr` (other intervals
unchanged), which drops the constant; in `w_step` (`sel_spec.py`) return `d[0]` for the first interval so the plotted selection function matches; set
`LRT` and `LRTp` to `nan` when `prec` is set on `'stepfun'` (no nested null). Estimates, standard errors and profiles do not move.

## Open decision

Answered on 2026-11-15 in `agents/reviews/order-numerical-fixes-specs.md` (D3): **(a) reject with `InputError`**. The task list builds option (a)
only; the options as they were put are kept below as the record.

**What is decided:** what `selmodel(type='stepfun', prec=...)` does (with `decreasing=False`).

**Background:** see the motivation. The rule since 2026-11-11 (`NOTES.md`, `claude/rules/metafopy.md`) is that the defining publication outranks
metafor and that metafopy's output changes only after the user confirms it; here no publication defines the model. metafor's estimates are the
maximum of density C; its ll, LRT and information criteria carry a constant that inflates the evidence for selection (LRT 28.3 against 7.6, `sei`,
`dat.bangertdrowns2004`, 2026-11-15).

**What it affects:** the `selmodel` API, the R parity fit `hack_stepfun_sei` in `src/tests/reference/phase5/selmodel.json` (generated by
`src/tests/r/generate_selmodel.R`), the cases `selmodel.prec.{sei,vi,ninv,sqrtninv}.bangertdrowns2004`, which are the only cases covering the
coverage items `selmodel.prec.*` of `src/tests/correctness/test_coverage.py`, and `docs/differences.md`. The audit does not reach the combination:
every `options/selmodel/*/prec=` case of run `20261104-210502` stops in both languages for want of `type`, and metafor's tests and examples use `prec`
only with the continuous types, so no `known.toml` entry is expected for any option.

| Option | What happens next |
|---|---|
| **(a) Reject (recommended)** | `InputError` as above. `hack_stepfun_sei` leaves the parity tests and gets a test that it raises; the four cases are replaced by `halfnorm` cases with each `prec` (Preston et al. 2004, the model `?selmodel` defines `prec` for), so the coverage items stay covered by a published model; a `docs/differences.md` entry. Users who relied on the experimental fit get an error naming the reason. |
| (b) Keep metafor | No code change. The four cases become `[[exempt]]` entries "defined only in metafor's code; R parity covers it" and `halfnorm` cases carry the `selmodel.prec.*` items; a `docs/differences.md` entry under "R's quirks copied on purpose" saying ll, LRT and AIC are shifted by the constant. metafopy keeps reporting an LRT that overstates selection. |
| (c) Define it as density C | The changes under Implementation. The four cases pass after `formulas.step_weight` gives weight `delta[0]` (not divided by `preci`) to the first interval and the derived check skips `LRT` when it is `nan`; `hack_stepfun_sei` keeps its estimate and inference tests but its `ll`, `LRT`, `LRTp`, fit statistics and printed output are excluded as differing. The model is one metafopy defines itself, from metafor's convention for the continuous types, with no source and no likelihood-ratio test. |

**Recommendation: (a).** No publication defines the model, so there is nothing to follow; (b) keeps a log-likelihood that is not a density and an
LRT inflated by a constant; (c) makes a consistent model up and still cannot give the LRT that every other `selmodel` fit reports. Rejecting removes a
wrong answer without inventing a method, follows metafor's own warning, and leaves the documented precision models (`halfnorm`, `negexp`, `logistic`,
`power`) untouched. It is reversible: should a source appear, (c) can be added later.

## Tests and records

For option (a); (b) and (c) adjust as the table says.

1. `src/tests/test_selmodel.py`: drop `hack_stepfun_sei` from `ALL_CASES` through a named mapping `DIFFERS = {'hack_stepfun_sei': '<reason citing
   docs/differences.md>'}`, and add `test_stepfun_prec_raises`, parametrised over `selmodel.PRECISIONS`, asserting `InputError` on the
   `hack_stepfun_sei` spec and on `dat.bangertdrowns2004`; add a test that `decreasing=True` with `prec` still warns and equals the fit without `prec`.
   The fixture and its R generator stay unchanged, as the record of metafor.
2. `src/tests/correctness/cases/selmodel.toml`: remove the four `selmodel.prec.*.bangertdrowns2004` cases; add `selmodel.prec.{sei,vi,ninv,sqrtninv}.
   halfnorm.bangertdrowns2004` (`kind = "derived"`, `type = "halfnorm"`, `prec` as named, `check = "test_derived.selmodel_fit"`, `tolerance = 2e-3` as for the other
   `halfnorm` cases, whose likelihood is flat in `delta`), each `covers = ["selmodel.prec.<p>"]`, and a comment naming this decision with its answer
   date. A probe of these cases on 2026-11-15 passed `test_derived.selmodel_fit` for each `prec`. A new case that fails is reported to the operator
   before it is set to `mismatch`, as the correctness specs require.
3. `docs/differences.md`, "Where metafor departs from its reference": an entry "selmodel(type='stepfun') with a precision measure", in the format of
   the entries there (summary, R and Python side by side, `*Why:*` with the check date). The R side shows `selmodel(res, type="stepfun",
   steps=c(0.025, 0.5, 1), prec="sei")` on `dat.bangertdrowns2004` and its ll and LRT; the Python side starts `# docs: skip` and shows the `InputError`.
   The `*Why:*` gives the density argument and the constant, citing Vevea and Hedges (1995).
4. The `TODO.md` item is moved to `DONE.md` with the decision and its date (`/todo-do` convention); the related fixed-`delta[0]` finding above is filed
   as its own item with `/todo-add`, after re-checking it.
5. Run `pytest src/tests/test_selmodel.py src/tests/correctness src/tests/test_docs_examples.py`, then `ruff format`, `ruff check` and `mypy`.
6. Run `bin/metafor-audit.sh` and `bin/metafor-audit.sh --recall` with `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (worktrees have no
   `.audit/Rlib`; every R call needs the same variable, or `R_LIBS` set to that path). If the audit flags `selmodel` with `prec` and `stepfun`, add a
   `known.toml` entry with `ref = "docs/differences.md"`, citing the decision and its date; otherwise none.

No step needs a stop for the operator once the decision is answered in the review of this spec: the output change is the one confirmed there.

## External dependencies

None new. R 4.6.2 with metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib` for the `docs/differences.md` example and the audit; SciPy
(already a dependency) for the derived cases.
