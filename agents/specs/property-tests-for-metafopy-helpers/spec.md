# Property tests for metafopy helpers

**Version:** 1.1
**Status:** approved
**Date:** 2026-11-22

Every function under `src/metafopy` whose domain supports a property gets a hypothesis test that calls it directly, in the way the auditme
T2 question counts. The property comes from the audit's own verdict for the function. The numerical kernels come first, because a property
test there can find a real bug that a test against R at a few fixed points misses.

## Motivation

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
The `TODO.md` item "Numerical cores have no property-based tests" was found on 2026-11-21 by auditme review-1
(`agents/audit-reviews/review-1.md`, question T2, at `31d809f`). 144 pure functions with well-defined domains have no hypothesis test, 132 of
them under `src/metafopy`. Decision D5 of that review (answered 2026-11-21) filed the item. The only property tests,
`src/tests/correctness/test_properties.py`, test whole fits. [measurements.md](measurements.md) splits the 132 into three groups, measured
on 2026-11-22:

| Group | Count | What it holds |
|---|---|---|
| A. Numerical kernels | 43 | `core/bvn`, `core/dbinom`, `core/farebrother`, `core/tukey`, `core/nchg`, the distribution functions of `methods/emm_adjust`, the weight functions of `models/sel_spec`, `models/sel_lik` maps, `pkendall`, `rtet`, `i2_from_q` |
| B. Number formatting and layout | 40 | `printing`, `core/labelled`, `core/timing`, `plot/rutils`, `r_as_character`, `format_number`, `fmt_*`, `level_to_alpha` |
| C. Names, formulas and strings | 49 | `formula`, `methods/matreg`, `methods/predict`, `models/call`, `make_unique`, `match_arg`, `pmatch`-style matchers, `abbreviate` |
<!-- doc-counts: accept-end -->

The [direct-tests spec](../direct-tests-for-metafopy-helpers/spec.md) (implemented, merged in `7c8bfeb` on 2026-11-22) made these helpers private.
Its decision D2 left the kernels to this item. Under T2 a private function still needs a property test that calls it directly, by its full module path.

## Module location

- `src/tests/rstrategies.py`: new, shared hypothesis strategies, imported as `import rstrategies` like the other helper modules in `src/tests` (`refutil`, `plotref`)
- `src/tests/test_<module>.py`: the property tests, in the existing file for each module (`test_tukey.py`, `test_nchg.py`, `test_formula.py`,
  ...), or a new file named after the module where none exists
- `agents/audit-reviews/ledger/T2.jsonl`: the re-judged T2 verdicts
- `src/metafopy/**`: no change. A property that fails is filed, not fixed (Error handling)

## API

No package API changes. The shared strategies module exposes plain functions that return hypothesis strategies:

```python
import hypothesis.strategies


def probabilities(*, open_interval: bool = True) -> hypothesis.strategies.SearchStrategy[float]:
    """Floats in (0, 1), or [0, 1] when open_interval is False; never NaN."""


def df_values(*, allow_inf: bool = False) -> hypothesis.strategies.SearchStrategy[float]:
    """Positive finite floats from 0.5 to 1e6, plus inf when allow_inf is True."""


def finite_floats(lo: float = -1e12, hi: float = 1e12) -> hypothesis.strategies.SearchStrategy[float]:
    """Finite floats in [lo, hi], with no NaN or infinity."""


def r_identifiers() -> hypothesis.strategies.SearchStrategy[str]:
    """Syntactically valid R names that are not R reserved words."""
```

Each property test is a plain pytest function in the project's style:

```python
@hypothesis.given(p=rstrategies.probabilities(), k=hypothesis.strategies.integers(2, 20), df=rstrategies.df_values())
def test_qtukey_inverts_ptukey(p: float, k: int, df: float) -> None:
    """ptukey(qtukey(p, k, df), k, df) is within 1e-4 of p (T2 finding for core/tukey.py::qtukey)."""
```

## Error handling

- **A property fails inside the function's domain.** The test has found a defect. The task stops as `blocked`, naming the function and
  hypothesis's shrunk example. The defect goes to `TODO.md` with `/todo-add`, and it is never fixed inside this work: an output change needs
  the user's confirmation (`claude/rules/metafopy.md`).
- **A property fails only outside the domain the R original documents** (for example, a negative degrees of freedom). The strategy is
  narrowed to that domain, and the test's docstring cites the R documentation or source line that sets it.
- **A property the ledger states is false by design** (for example, R's own function breaks it in the same way, checked in `Rscript`). The
  task records the R output in the test file's comment, tests the property R does keep, and notes the change in the ledger `reason` when it
  re-judges.
- **A test runs for more than 10 seconds** at the settings below. It gets the existing `slow` marker (`pyproject.toml`).

## Implementation

1. **Re-judge T2 at HEAD.** Run `phaseb.py plan . --question T2`. Judge each planned candidate with the T2 brief
   (`~/.claude/skills/auditme/phaseb/briefs/T2.md`), then `check` and `merge`. The open `src/metafopy` list from `report` is the work for the
   steps below. A candidate under `src/audit` or `bin` is judged but not tested here: it belongs to the item "Audit-harness and bin helpers
   with branches have no direct test".
2. **Shared strategies.** Write `src/tests/rstrategies.py` as in the API section, with a test of each strategy's bounds.
3. **Group A, the numerical kernels.** For each open candidate, write a test that calls it directly and checks the property its ledger
   `reason` states.
   - **Oracles.** Use the scipy oracle the reason names where it names one: `scipy.stats` distributions, `scipy.special.expit`,
     `math.lgamma`, or `scipy.stats.multivariate_normal` for `bvn`. scipy is already a runtime dependency.
   - **Tolerances.** Compare with scipy at a relative tolerance of 1e-8 (1e-6 for the series in `farebrother` and `tukey`). Check a bound
     or monotonicity with a slack of 4 ulps. Use the tolerance the reason gives where it gives one (1e-4 for `qtukey`).
   - **Relation to the R tests.** The R parity fixtures stay the bit-level check. A property test adds coverage of the domain, not a second
     reference value.
4. **Group B, number formatting and layout.** The same for each open candidate: round trips through `float()`, rounding bounds, idempotence,
   and equal column widths, as each reason states.
5. **Group C, names, formulas and strings.** The same: invariants of R's `pmatch`, `make.unique` and `make.names`, parse-back round trips for
   formulas built from `r_identifiers()`, and the oracles the reasons name. For example, `matreg.match_columns` agrees with
   `predict.match_names`, and `is_subsequence` agrees with a brute-force check.
6. **Close.** Re-judge T2 for the files this work touched (`phaseb.py plan` re-plans them). `phaseb.py report . --question T2` then lists
   no open `src/metafopy` candidate. Then move the `TODO.md` item to `DONE.md`.

Every test uses `@hypothesis.settings(max_examples=100, deadline=None)`. One hundred examples is hypothesis's default; `deadline=None`
because the series kernels vary in time with their inputs. A test names its T2 candidate id in its docstring.

## Settings

None.

## External dependencies

- hypothesis (already a dev dependency, `hypothesis>=6.100`), pytest, scipy (already a runtime dependency): no new dependency.
- The auditme Phase B tooling (`phaseb.py` and the T2 brief) under `~/.claude/skills/auditme`, from the auditor repository. Check it is in
  place with `python3 ~/.claude/skills/auditme/checks/phaseb.py --help`. It is run, not changed.

## Decisions

Answered 2026-11-22 in [the spec review](../../reviews/property-tests-for-metafopy-helpers-spec.md):

- **D1, scope:** all three groups, A then B then C (Implementation steps 3 to 5).
- **D2, the harness findings:** the T2 findings under `src/audit` and `bin` moved to the `TODO.md` item "Audit-harness and bin helpers with
  branches have no direct test" (a dated note on both items records it), so this spec closes its item.
