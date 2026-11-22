# Direct tests for metafopy helpers

**Version:** 1.1
**Status:** implemented
**Date:** 2026-11-21

Every function under `src/metafopy` that branches gets a test that reaches it, in the way the auditme T1 question counts. A public function
needs a test that calls it directly. A private one needs a test of a caller that reaches it. Internal helpers become private, without a leading
underscore, by leaving their module's `__all__`. Then a test of their caller counts, and the direct tests go to the functions users call.

## Motivation

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
The `TODO.md` item "Public metafopy helpers with branches have no direct test" was found on 2026-11-21 by auditme review-1
(`agents/audit-reviews/review-1.md`, question T1, at `31d809f`). 162 functions under `src/metafopy` that branch have no test calling them
directly, under the T1 rulings that the review's section "Thresholds settled during the run" records. Decision D4 of that review (answered
2026-11-21) filed the item. [measurements.md](measurements.md) splits the 162 by how public each one is, measured on 2026-11-21:

| Group | Count | What it holds |
|---|---|---|
| Exported from `metafopy` | 2 functions, 28 methods | `replmiss`, `fmtx`; methods and properties of exported result classes such as `RMAMVResult.random` and `GLMMResult.render` |
| Named only in their own module's `__all__` | 108 functions, 19 methods | helpers that other modules of the package call, such as `core.glm.glm_fit`, `models.mv_inputs.prepare_v` and `printing.fmt_df` |
| Already private | 5 | private, but no test reaches them, such as `models.mv_inputs.block_diagonal` and `models.mv_structs.distance_from_matrix` |
<!-- doc-counts: accept-end -->

The middle group is the bulk. These functions are in their module's `__all__` only by habit: nothing reads a module's `__all__` except
`auditme.core.is_private` (checked 2026-11-21). No module does `import *`, the API reference names each object explicitly, and the code reaches
helpers by their full module path.

## Module location

- `src/metafopy/**/*.py`: `__all__` lists only (no behaviour change)
- `src/tests/`: new tests, in the existing file for each module, or `src/tests/test_<module>.py` where none exists
- `agents/audit-reviews/ledger/T1.jsonl`: the re-judged T1 verdicts

## API

No public API changes. A name leaves a module's `__all__` only if it is not in `metafopy.__all__` or a subpackage's `__all__`. Every new test is
a plain pytest function in the project's style:

```python
def test_rmamv_random_lists_each_component() -> None:
    """RMAMVResult.random names the random terms of a fit with two components."""


def test_block_diagonal_takes_a_list_of_blocks() -> None:
    """rma_mv with V given as a list of block matrices builds the block-diagonal V."""
```

The T1 tooling, from the auditor checker installed at `~/.claude/skills/auditme`:

```text
python3 ~/.claude/skills/auditme/checks/phaseb.py plan . --question T1 --out <dir>   # batches of candidates whose hash or brief changed
python3 ~/.claude/skills/auditme/checks/phaseb.py check <dir>
python3 ~/.claude/skills/auditme/checks/phaseb.py merge . --question T1 <dir>
python3 ~/.claude/skills/auditme/checks/phaseb.py report . --question T1
```

## Error handling

- A new test that shows a defect (wrong output or wrong error) stops the task as `blocked`, naming the function and the input. The defect goes
  to a `TODO.md` item; it is never fixed inside this work, since an output change needs the user's confirmation (`claude/rules/metafopy.md`).
- Removing a name from `__all__` that breaks an import, a test or `mypy` means something reads that list. The name goes back and stays public
  (it then needs a direct test), and the task records why.
- A candidate the T1 brief cannot settle under review-1's rulings stops the task as `blocked`, naming the candidate.

## Implementation

1. **Make the internal helpers private.** For every name in the group "Named only in their own module's `__all__`" of
   [measurements.md](measurements.md), remove the name from its module's `__all__`. Leave the `__all__` empty, or delete it where nothing is
   left. A name in `metafopy.__all__` or a subpackage `__all__` stays. Run `mypy` and the tests of each module touched.
2. **Re-judge T1 at HEAD.** Run `phaseb.py plan` for T1. Every candidate whose code or `private` flag changed since `31d809f` is
   re-planned. Judge each with the T1 brief (`~/.claude/skills/auditme/phaseb/briefs/T1.md`) and review-1's rulings, then `check` and `merge`.
   The open list from `report` is the work for steps 3 and 4.
3. **Direct tests for public functions and methods.** For each open public candidate (the two functions and the exported-class methods),
   write a test that calls it, by name or through a re-export, on a small fit from the bundled datasets. The test asserts a value metafopy already
   produces and R metafor agrees with (a parity fixture value, or a field another test already checks), and exercises at least the branch the
   T1 finding names. A `__post_init__` gets a test that builds the object itself. A `render` or `title_lines` hook gets a test that calls it
   on a fitted result.
4. **Caller tests for private helpers no test reaches.** For each open private candidate, add a test of a caller on an input that takes the
   helper's branch, such as `rma_mv` with `V` as a list of blocks or with `dist` as a matrix.
5. **Close.** Run `phaseb.py report` for T1: no `src/metafopy` candidate is open. Then move the `TODO.md` item to `DONE.md`.

Numerical kernels that are also on the property-test item ("Numerical cores have no property-based tests", which follows this one in
`TODO.md`) become private in step 1 like the rest. Their caller tests count for T1, so this spec writes no property tests; that item does.

## Settings

None.

## External dependencies

- pytest (already a dev dependency); no new dependency.
- The auditme Phase B tooling (`phaseb.py` and the T1 brief) under `~/.claude/skills/auditme`, from the auditor repository. Check it is in
  place with `python3 ~/.claude/skills/auditme/checks/phaseb.py --help`. It is run, not changed.

## Decisions

Answered 2026-11-21 in [the spec review](../../reviews/direct-tests-for-metafopy-helpers-spec.md):

- **D1, internal helpers:** they are made private by leaving their module's `__all__` (Implementation step 1), rather than tested one by one.
- **D2, the numerical kernels:** they stay with the property-test item; this spec writes no hypothesis tests.
