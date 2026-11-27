# Coefficient names and print tests follow R

**Version:** 1.0
**Status:** proposed
**Date:** 2026-11-27
**Review:** `agents/reviews/coefficient-names-and-print-tests-follow-r-spec.md` (page https://claude.ai/artifact/4Vfupvo5ADv27gJf2EqpvD); answered 2026-11-27: every item approved

`metafopy.formula.design_matrix` builds the model matrix of every `mods=` and `scale=` formula (`rma_uni`, `rma_mv`, `rma_ls` through
`rma_uni(..., scale=)`, and every method that refits them), and names its columns as R's `model.matrix` does so that coefficient labels match
metafor 5.2-1. One naming rule is missing: R orders the variables inside an interaction term by where each variable first appears in the formula,
and metafopy keeps the order the term was written in. This spec adds that rule, and clears the print-test skips that hide it, most of which turn out
to come from the tests rather than the engine.

## Motivation

The `TODO.md` item "`rma_mv` prints some coefficient labels unlike R, and the print tests skip them for a partly stale reason", as filed:

<!-- doc-counts: accept-start found on 2026-11-26 - a dated record, not a claim about the present -->
> Found on 2026-11-26 at `900698c` (macOS arm64) during a `TODO.md` review, by comparing metafopy's summaries with the R fixtures of the cases the
> print tests skip with "coefficient labels follow the formula engine" (`src/tests/test_rma_mv.py:185`, `NAMES_DIFFER`, and
> `src/tests/test_robust.py:159`, both since `88dbd52`, 2026-10-13):
>
> - `ass_mods`, `ass_mods_t`: R prints `factor(deltype)general`, `factor(deltype)overt`; metafopy prints `deltypegeneral`, `deltypeovert`;
> - `ber_un_mods`: R prints `outcomeAL:I(year - 1985)`; metafopy orders the interaction the other way, `I(year - 1985):outcomeAL`; its `QM` also
>   prints `76.7085` against R's `76.7084`;
> - `uni_cs_btt`: the label now matches R (`^`); the summary differs only in wrapping, since metafopy breaks the 81-character coefficient table at
>   `ci.ub` where R prints it on one line;
> - `kon_btt`: both summaries are identical, so its skip is stale.
<!-- doc-counts: accept-end -->

Measured on 2026-11-27 at `e9a5f1d` (macOS arm64, R 4.6.2 with base R's `model.matrix`; scripts in the session scratchpad, not kept; the figures are the
record):

<!-- doc-counts: accept-start measured on 2026-11-27 - a dated record, not a claim about the present -->
- **`factor()` already matches R.** `design_matrix('~ factor(deltype) + year')` names the columns `factor(deltype)general`, `factor(deltype)overt`,
  as R does. The `ass_mods` cases differ because `src/tests/test_rma_mv.py:112-113` writes `C(deltype)` where `src/tests/r/generate_rma_mv.R:110-111`
  writes `factor(deltype)`. R names `C(deltype)` columns `C(deltype)general`; metafopy strips `C()` and names them `deltypegeneral`
  (`formula.CWRAP` in `r_column_name`). The operator chose on 2026-11-27 to keep that (see Decisions).
- **Interaction order is the one engine difference.** Over sixteen formulas on one small data frame, metafopy's names equal R's except where an
  interaction is written in another order than its variables first appear: `~ 0 + outcome + I(year - 1985):outcome` (R `outcomeAL:I(year - 1985)`,
  metafopy `I(year - 1985):outcomeAL`), `~ x + z + z:x` (R `x:z`, metafopy `z:x`), `~ deltype + outcome + outcome:deltype` (R
  `deltypegeneral:outcomePD`, metafopy `outcomePD:deltypegeneral`). `~ z:x + x + z` gives `z:x` in both, since `z` appears first there. Of the
  formulas the repository's tests and docs pass as `mods=` or `scale=`, only `ber_un_mods`'s is affected.
- **The wrapping difference is the tests' width.** `src/tests/conftest.py` (`wide_console`) sets metafopy's print width to 200 for every test,
  with the docstring "The printouts stored from R were made at `width = 200`". Only 13 of the 58 scripts in `src/tests/r/` set
  `options(width = 200)`; `generate_rma_mv.R` is not one of them, so its printouts are at R's default of 80. `generate_robust.R` is, and at width
  200 `uni_cs_btt` matches R exactly; the 81-character wrap in the finding was measured outside pytest, at metafopy's default of 80. With
  `factor(deltype)` and width 80 both `ass_mods` printouts match R; at 200 they do not, because R wraps the significance stars into a second block.
  At width 80 every other `rma_mv` print case (not in `PRINT_UNSTABLE`) still matches, as it does at 200.
- **`ber_un_mods`'s `QM` digit is a flat likelihood, not a defect.** metafopy's REML log-likelihood is `2.0731736402789487` against R's
  `2.07317364027501` (a difference of 4e-12); `beta` differs by at most 1.0e-6 relative and `rho` by 2.4e-6. `QM` is `76.708512` against R's
  `76.708353`, on either side of the rounding point `76.70845`, so the fourth decimal shows a difference neither optimum is better for.
- **`kon_btt`** prints identically to R at both widths.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/formula.py` - the interaction-order rule (`design_matrix`, and a new `r_term_order`)
- `src/tests/test_formula_names.py` (new), `src/tests/r/generate_formula_names.R` (new), `src/tests/reference/formula_names.json` (new, generated)
- `src/tests/test_rma_mv.py`, `src/tests/test_robust.py`, `src/tests/conftest.py`
- `docs/differences.md`

## API

No public signature changes. One new module-level function in `metafopy.formula` (not in `__all__`):

```python
def r_term_order(rhs: str) -> str:
    """The formula ``rhs`` with the variables of each interaction term in the order R's ``terms.formula`` gives them.

    R lists a formula's variables in the order each first appears in the formula text, and writes every term with its variables in that order,
    so ``~ 0 + outcome + I(year - 1985):outcome`` has the term ``outcome:I(year - 1985)``, whose columns are named ``outcomeAL:I(year - 1985)``.
    Main-effect terms, the intercept (``0 +``, ``- 1``) and the order of the terms are left as they are.

    See https://stat.ethz.ch/R-manual/R-devel/library/stats/html/terms.formula.html.

    Raises:
        metafopy.errors.InternalError: a variable of a parsed term could not be found in the formula text.
    """
```

`design_matrix(formula, data)` keeps its signature and returns; it calls `r_term_order` on the right-hand side before `r_power` and
`formulaic.model_matrix`, so the column names and the column order of every interaction term follow R.

## Implementation

1. **Variable order.** Parse `rhs` with `formulaic.Formula` (formulaic 1.2.2, from `uv.lock`); each term's `factors` give its variables'
   expressions (`I(year - 1985)`, `outcome`). Find each expression's first position in the formula's token stream (`r_tokens`, so spacing does not
   matter) and order each interaction term's factors by it. A factor whose expression is not in the token stream is a bug in this function, and
   raises `InternalError` naming the term; never fall back to the written order.
2. **Rebuild.** Write the formula back with each reordered term joined by `:`, keeping the original intercept (`0 +` or `- 1` when the parsed
   formula has none) and the parsed term order. A formula with no interaction term is returned unchanged, so nothing but interactions can move.
3. **Columns follow.** formulaic varies a term's first variable fastest, as R does (measured 2026-11-27 on `~ deltype:outcome`), so once the
   factors are in R's order the column names and their order within the term follow R with no change to `r_column_name`.
4. **`C()` stays.** `r_column_name` keeps stripping `C(...)` (the operator's decision of 2026-11-27); its docstring says so and why.

## Tests

- **R names oracle.** `src/tests/r/generate_formula_names.R` writes, for a fixed list of formulas over one small data frame (numeric `x`, `z`,
  `year`; character `deltype` with three levels and `outcome` with two), each formula's `colnames(model.matrix(...))` and the matrix itself to
  `src/tests/reference/formula_names.json`. The list covers main effects, `factor()`, `I()`, `^`, `*`, interactions written in and against
  first-appearance order, a two-factor interaction with its main effects, and `0 +`. Generate it with `LC_ALL=C.UTF-8 Rscript`, as `AGENTS.md`
  says, with `R_LIBS` set to the audit's pinned library (`audit.config.R_LIB`, `.audit/Rlib` by default), which holds jsonlite.
  `src/tests/test_formula_names.py` checks `design_matrix`'s names equal R's and its matrix equals R's column for column, for every entry.
- **`r_term_order`** gets direct tests: a formula with no interaction comes back unchanged; `0 +` and `- 1` survive; a three-way interaction is
  reordered; an expression with spaces (`I(year - 1985)`) is found.
- **`src/tests/test_rma_mv.py`:**
  - `ass_mods` and `ass_mods_t` write `factor(deltype)`, as `generate_rma_mv.R` does;
  - `NAMES_DIFFER` is removed, so `test_fit` checks `coef_names` and `test_print` the printout of every case;
  - `ber_un_mods` joins `PRINT_UNSTABLE` with the measured reason (log-likelihood equal to R's within 1e-11, `QM` either side of a rounding
    point); its names stay checked by `test_fit`'s `coef_names`;
  - a module fixture sets the print width to 80 (`monkeypatch.setitem(metafopy.printing.SETTINGS, 'width', 80)`), because
    `generate_rma_mv.R` prints at R's default; its docstring says so.
- **`src/tests/test_robust.py`:** the `uni_cs_btt` skip in `test_print` is removed.
- **`src/tests/conftest.py`:** `wide_console`'s docstring says that the generators which print wide tables set `width = 200`, and that a module
  whose generator does not sets 80 for its print tests.

## Output changes

Approving this spec is the confirmation `claude/rules/metafopy.md` asks for before metafopy's output changes. What changes:

- the coefficient names of an interaction term whose variables are written out of their first-appearance order, everywhere they appear
  (`coef_names`, the printed model results, `coef()`, `confint()`, `predict()` and `anova()` labels, `btt=` given as a name pattern). On
  2026-11-27 this is `ber_un_mods` alone among the repository's tests and docs;
- for an interaction of two factors written out of that order, the order of the term's columns (and so of its coefficients), which then follows R.

Estimates, standard errors and every test statistic are unchanged: the columns of a term are the same products, renamed and, for two factors,
permuted together with their coefficients.

## Error handling

`r_term_order` raises `metafopy.errors.InternalError` when a parsed factor's expression is missing from the formula text (see Implementation 1).
Every other failure is the existing one: `design_matrix` raises `metafopy.errors.InputError` for an invalid formula or data, and formulaic's own
parse errors reach the caller as they do today.

## Documentation

`docs/differences.md`, under "By design", gains a section "`C()` names its columns like a plain factor": R's `model.matrix` names
`C(alloc)` columns `C(alloc)random`, metafopy `allocrandom`, because in metafopy `C()` is formulaic's categorical marker rather than R's
contrasts setter; `factor(alloc)` and plain `alloc` name their columns as R does. It follows the page's R-and-Python comparison layout, with
output from running each side. `docs/guide/models.md` is unchanged: its `'~ C(alloc)'` example and `allocrandom` output stay correct.

## Out of scope

An interaction of two factors with neither main effect (`~ deltype:outcome`) gets other columns in R than in metafopy: R codes both factors in
full (seven columns with the intercept, measured 2026-11-27), metafopy codes the second by contrasts (six). That is contrast coding, not naming; it
is filed as its own `TODO.md` item when this spec is approved, not fixed here.

## Settings

None. The print width the tests set is `metafopy.printing.SETTINGS['width']`, which exists; this spec adds no setting.

## External dependencies

- formulaic 1.2.2 (pinned in `uv.lock`): `formulaic.Formula` for the parsed terms, `formulaic.model_matrix` as now.
- R (4.6.2 here on 2026-11-27) with jsonlite from the audit's pinned library, for generating `formula_names.json` only; `model.matrix` is base
  R's `stats`, so metafor is not loaded.

## Decisions

- **2026-11-27, operator:** `C(x)` keeps naming its columns like a plain factor (`allocrandom`), as `docs/guide/models.md` shows; the difference
  from R is recorded in `docs/differences.md`.
- **2026-11-27, made in writing this spec (reversible):** the interaction order follows R with no compatibility switch, because metafopy is at
  version 0.0.1 and its contract is to name coefficients as metafor does.
- **2026-11-27, made in writing this spec (reversible):** `ber_un_mods`'s printout is treated as flat-likelihood unstable rather than tightened,
  because its log-likelihood already equals R's to 4e-12 and the differing digit is a rounding point between two equally good optima.
