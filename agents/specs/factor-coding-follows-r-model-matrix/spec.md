# Factor coding follows R's model matrix

**Version:** 1.0
**Status:** implemented
**Date:** 2026-11-27
**Review:** `agents/reviews/factor-coding-follows-r-model-matrix-spec.md` (page
https://claude.ai/artifact/L3xcqoWFqU6HJgDYE3ri2q); work stops until every item has an answer

`metafopy.formula.design_matrix` builds the model matrix of every `mods=` and `scale=` formula (`rma_uni`, `rma_mv`, `rma_glmm`, `rma_ls` through
`rma_uni(..., scale=)`, and every method that refits them). For each factor in each term it lets formulaic choose between contrasts and full
indicator coding, and formulaic's rule differs from R's `model.matrix` when a term appears without the terms it is marginal to: formulaic splits
such a term so that the matrix stays full rank, while R codes the factors in full, which leaves the matrix rank-deficient. metafor then drops the
redundant columns with a warning. The two parametrise the same model differently, so the coefficients, their names and every per-coefficient test
differ from metafor's. This spec gives `design_matrix` R's coding rule. The models already drop redundant columns as metafor does. It also puts the
reference grid of `emmprep`, which builds its own model matrix through formulaic, on the same builder. Since the coefficient-names spec shipped,
that builder no longer matches the fit for an interaction written out of first-appearance order.

## Motivation

The `TODO.md` item "A two-factor interaction with neither main effect gets other columns than in R", as filed:

<!-- doc-counts: accept-start found on 2026-11-27 - a dated record, not a claim about the present -->
> Found on 2026-11-27 at `e9a5f1d` (macOS arm64, R 4.6.2) while writing `agents/specs/coefficient-names-and-print-tests-follow-r/spec.md`, which
> leaves it out of scope (its review item S6, approved 2026-11-27): on a small data frame with `deltype` (three levels) and `outcome` (two),
> `model.matrix(~ deltype:outcome)` gives seven columns (`(Intercept)`, then `deltypecovert:outcomeAL` to `deltypeovert:outcomePD`, both factors
> coded in full), while `metafopy.formula.design_matrix('~ deltype:outcome')` gives six (`intrcpt`, `outcomePD`, then the four `deltypegeneral` and
> `deltypeovert` products), coding the second factor by contrasts.
<!-- doc-counts: accept-end -->

Measured on 2026-11-27 at `cb24530` (macOS arm64, R 4.6.2, metafor 5.2-1 and emmeans from the audit's pinned library `.audit/Rlib`; scripts in the
session scratchpad, not kept; the figures are the record):

<!-- doc-counts: accept-start measured on 2026-11-27 - a dated record, not a claim about the present -->
- **Fits.** On 12 rows (`deltype` three levels, `outcome` two, `method='FE'`), metafor warns "Redundant predictors dropped from the model." for
  `~ deltype:outcome`, drops `deltypeovert:outcomePD` and reports `intrcpt` 0.491705, `deltypecovert:outcomeAL` -0.104487,
  `deltypegeneral:outcomeAL` -0.253475, `deltypeovert:outcomeAL` -1.322183, `deltypecovert:outcomePD` -0.488539, `deltypegeneral:outcomePD`
  0.591267. metafopy gives no warning and reports `intrcpt` 0.387218, `outcomePD` -0.384052, `deltypegeneral:outcomeAL` -0.148988,
  `deltypeovert:outcomeAL` -1.217696, `deltypegeneral:outcomePD` 1.079807, `deltypeovert:outcomePD` 0.488539. The fitted values agree. The same
  holds for `rma.mv` with `mods = ~ f:g` and for a location-scale fit with `scale = ~ f:g`, where metafor warns "Redundant predictors dropped from
  the scale model." (24 rows, `f` three levels, `g` two).
- **Shapes.** Over 20 formulas on one data frame (factors `f`, `g`, `h`; numerics `x`, `z`), metafopy's names and rank equal R's except
  `~ f:g`, `~ f:g:h`, `~ f:g + g:h` and `~ 0 + f:g + h`. These are exactly the formulas whose R matrix is rank-deficient. A factor with a numeric
  (`~ x:f`, `~ x:z:f`, `~ 0 + x:f + g`), nested terms (`~ f + f:g`, `~ x + x:f`), main effects and `0 +` already match.
- **The repository.** None of the `mods=`, `scale=` and test formulas in `src/`, `docs/` and `README.md` is one of the affected shapes: every
  interaction in them either has the margins R needs or pairs a factor with a numeric.
- **`emmprep` since the coefficient-names spec.** `src/metafopy/methods/emm_grid.py:409` rebuilds the model matrix with `formulaic.model_matrix`
  on the formula as written, without the `r_term_order` that `design_matrix` has applied since `367c13d` ("Task 3: order interaction variables as
  R does"). For `rma_uni(mods='~ g + f:g')` the fit's columns are `gp:fb, gq:fb, ...`, while the grid's are `fb:gp, fc:gp, ...`, so `emmprep` raises
  "Cannot match the model matrix of the fit with that rebuilt from the formula and the data". At `c8fe0fe`, before the change, the same call
  succeeds. In R, `emmprep` on the same fit succeeds and notes the nesting `f %in% g`.
<!-- doc-counts: accept-end -->

## Module location

- `src/metafopy/formula.py`: the coding rule (a new materializer and `r_model_matrix`), used by `design_matrix`
- `src/metafopy/methods/emm_grid.py`: `reference_grid` builds its model matrix through `r_model_matrix`
- `src/tests/r/generate_formula_names.R`, `src/tests/reference/formula_names.json` (regenerated), `src/tests/test_formula_names.py`
- `src/tests/r/generate_formula_rank.R` (new), `src/tests/reference/formula_rank.json` (new, generated), `src/tests/test_formula_rank.py` (new)
- `src/tests/test_formula.py`, `src/tests/test_emmeans_nesting.py`

## API

No public signature changes. Two new module-level names in `metafopy.formula` (not in `__all__`):

```python
class RPandasMaterializer(formulaic.materializers.PandasMaterializer):
    """formulaic's pandas materializer with R's ``model.matrix`` rule for coding the factors of each term.

    In term ``k``, a factor is coded by contrasts when the term without it is empty and the formula has an intercept, or is contained in one of
    the terms before ``k``; otherwise it is coded by indicators for every level. With no intercept, the first factor that is a main effect is
    coded by indicators and the empty term then counts as present. Each formula term becomes one scoped term, so a term is never split to keep
    the matrix full rank: as in R, a matrix can be rank-deficient, and the models drop the redundant columns.

    See ``TermCode`` in https://github.com/wch/r-source/blob/trunk/src/library/stats/src/model.c.
    """


def r_model_matrix(rhs: str, data: pandas.DataFrame, context: dict[str, typing.Any]) -> formulaic.ModelMatrix:
    """The model matrix of the right-hand side ``rhs`` as R's ``model.matrix`` builds it: interaction variables in R's order (``r_term_order``),
    ``^`` inside calls as a power (``r_power``), and factors coded by ``RPandasMaterializer``. Rows with missing values are kept.

    Raises:
        metafopy.errors.InternalError: an interaction's variable could not be placed in the formula text (see ``r_term_order``).
    """
```

`design_matrix(formula, data)` keeps its signature and returns. It calls `r_model_matrix` where it now calls `formulaic.model_matrix`.
`emm_grid.reference_grid` calls `r_model_matrix` on `r_logical_formula(rhs)` in place of its own `formulaic.model_matrix`, and the grid's
`spec.get_model_matrix` keeps the materializer, because formulaic stores it in the model spec.

## Implementation

1. **Scoping.** `RPandasMaterializer` overrides formulaic 1.2.2's `_get_scoped_terms(terms, ensure_full_rank)` (marked with
   `interface_meta.override`, which formulaic's interface check requires). It walks the terms in formulaic's order, which sorts by degree as R's
   `terms` does. For each term it takes the evaluated factors as the base method does, sets aside constant factors (their product is the scoped
   term's `scale`), and gives each categorical factor `ScopedFactor(reduced=...)` by the rule in the docstring. "Contained in" means its variables
   are a subset of an earlier term's. A numeric factor is never reduced. It yields exactly one `ScopedTerm` per term. `ensure_full_rank` is ignored:
   R's rule decides.
2. **No intercept.** Before the walk, the empty term counts as present when the formula has an intercept. Without one, it starts absent, and the
   first categorical factor met in a main-effect term is coded in full, after which it counts as present. That gives R's `~ 0 + f + g`
   (`fa fb fc gq`) and `~ 0 + x + f` (`x fa fb fc`).
3. **One builder.** `r_model_matrix` is the only call of `formulaic.model_matrix` in `metafopy`. It passes `materializer=RPandasMaterializer`, and
   `design_matrix` and `emm_grid.reference_grid` call it, so the fit and its reference grid always share term order and coding.
4. **Redundant columns are already handled.** `rma_uni` (`src/metafopy/models/rma_uni.py:1059`), `rma_mv` (`rma_mv.py:868`), `rma_glmm`
   (`rma_glmm.py:237`) and the location-scale fit (`rma_ls_fit.py:706` and `:733`, the scale model) drop the columns `drop_redundant` finds and warn
   with metafor's messages. `drop_redundant` keeps columns in order and drops each that adds no rank, as R's pivoted QR does, so `~ deltype:outcome`
   drops `deltypeovert:outcomePD` as metafor does. No model changes.
5. **`emmprep` on a rank-deficient fit** keeps raising metafor's "Cannot use function when some redundant predictors were dropped from the model."
   (`emm_grid.py:255`), as R does for `~ f:g` (measured 2026-11-27).

## Tests

- **R names oracle.** `src/tests/r/generate_formula_names.R` gains a factor `arm` (two levels) in its data frame and these formulas:
  `~ deltype:outcome`, `~ outcome:deltype`, `~ deltype:outcome:arm`, `~ deltype:outcome + outcome:arm`, `~ 0 + deltype:outcome + arm`,
  `~ 0 + deltype + outcome`, `~ 0 + x + deltype`, `~ x:deltype`, `~ outcome + deltype:outcome`. `src/tests/test_formula_names.py` gains the same
  column in `DATA`. Regenerate `formula_names.json` with `LC_ALL=C.UTF-8 R_LIBS=.audit/Rlib Rscript` (`audit.config.R_LIB`, which holds jsonlite).
  Every existing entry must come out unchanged: compare the old and new JSON for the old keys before committing.
- **R fits oracle.** `src/tests/r/generate_formula_rank.R` writes `src/tests/reference/formula_rank.json`. It holds the 12-row data set of the
  motivation, written into the script as literal values, not drawn from R's RNG, and for each fit the coefficient names, estimates, standard errors
  and the warning metafor gave. The fits are `rma(yi, vi, mods = ~ deltype:outcome, method = "FE")`,
  `rma.mv(yi, vi, mods = ~ deltype:outcome, random = ~ 1 | id)` and `rma(yi, vi, mods = ~ year, scale = ~ deltype:outcome)`. Pick the `year` values
  and starting data so that the location-scale fit converges in R, and record the R and metafor versions. `src/tests/test_formula_rank.py` fits each
  in metafopy and checks the names, the estimates and standard errors (`rtol=1e-6`), and the warning text with `pytest.warns`.
- **`emmprep` with an out-of-order interaction.** `src/tests/test_emmeans_nesting.py` gains a case `rma_uni(mods='~ grp + lat:grp')` on its nested
  data, checking that `emmprep` succeeds and that `emmeans(grid, 'lat')` gives the same estimates as the same fit written `~ grp + grp:lat`.
- **`RPandasMaterializer`** gets direct tests in `src/tests/test_formula.py`: one scoped term per formula term, constant factors kept as scale,
  `ensure_full_rank=False` giving the same result.

## Output changes

Approving this spec is the confirmation `claude/rules/metafopy.md` asks for before metafopy's output changes. What changes is a formula whose R
matrix is rank-deficient: an interaction of factors written without the terms R needs to code it by contrasts (`~ f:g`, `~ f:g:h`, `~ f:g + g:h`,
`~ 0 + f:g + h`). For such a formula:

- the coefficients, their names and their number follow metafor: one more column before the drop, then the redundant one dropped with metafor's
  warning, in every model that takes `mods=` or `scale=`;
- every per-coefficient statistic (`se`, `zval`, `pval`, CIs) and every `btt=` selection by name follow the new coefficients. The fitted values,
  `QM` for all moderators, `tau2` and the log-likelihood do not change;
- `emmprep` on such a fit raises, as metafor's does, where it now succeeds.

On 2026-11-27 no formula in the repository's tests and docs is affected (see Motivation). `emmprep` on a fit whose interaction is written out of
first-appearance order (`~ g + f:g`) works again, as it did before `367c13d`.

## Error handling

`r_model_matrix` raises `metafopy.errors.InternalError` only through `r_term_order`. Everything else stays as it is: `design_matrix` raises
`metafopy.errors.InputError` for an invalid formula or data, and formulaic's own parse errors reach the caller unchanged. If a later formulaic
renames `_get_scoped_terms`, every test that builds a matrix fails. formulaic is pinned at 1.2.2 in `uv.lock`, so this happens only on a deliberate
upgrade, and the R oracles show whether the override still holds.

## Documentation

Nothing in `docs/differences.md` records this difference, so nothing is removed there. `docs/guide/models.md` gets one sentence where it
describes `mods`: a formula whose matrix is rank-deficient is fitted as in metafor, with the redundant columns dropped and a warning.

## Out of scope

`docs/differences.md`'s `C()` section (operator's decision of 2026-11-27) and every naming rule are unchanged. Contrasts other than treatment
coding (R's `contrasts=` and `options(contrasts=)`) are not added.

## Settings

None.

## External dependencies

- formulaic 1.2.2 (pinned in `uv.lock`): `PandasMaterializer`, `ScopedTerm` and `ScopedFactor` from `formulaic.materializers.types`, and
  `Factor.Kind`. `_get_scoped_terms` is a private method (see Error handling).
- interface-meta 2.0.1 (already installed as formulaic's dependency, in `uv.lock`): `interface_meta.override`. Add it to `pyproject.toml`'s
  dependencies as `interface-meta>=2.0`, because metafopy imports it directly, and run `uv lock` to check the lock is unchanged apart from that
  entry.

## Decisions

- **2026-11-27, made in writing this spec (reversible):** the rule is applied by overriding formulaic's private `_get_scoped_terms` rather than by
  rewriting the formula (`C(x, spans_intercept=False)` gives R's columns for `~ f:g` but not for `~ f:g + g:h`, measured 2026-11-27) or by building
  the matrix outside formulaic, because the override is one method that leaves naming, NA handling and encoding to formulaic, and the R oracles
  catch any upgrade that breaks it.
- **2026-11-27, made in writing this spec (reversible):** `emm_grid` moves to the shared builder in this spec rather than in its own item, because
  its regression has the same cause (a second model-matrix builder) and the same fix.
- **2026-11-27, made in writing this spec (reversible):** no compatibility switch for the old coding, because metafopy is at version 0.0.1 and its
  contract is to give metafor's coefficients.
