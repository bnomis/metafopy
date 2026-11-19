# Review: the ls profile feasible start spec

**Review page:** https://claude.ai/artifact/LPG3ozit2KpHMqeshtFgzN
**Date:** 2026-11-19

**What is being reviewed and how.** One new spec, `agents/specs/ls-profile-feasible-start/spec.md` (with `measurements.md`), written by `/todo-do`
for the `TODO.md` item "metafopy's profile CI misses the identity-link lower bound of the ls scale intercept". It adds a feasible start to
identity-link profile refits that are left with only the default start, which changes one metafopy `confint` output. Approve the spec (approval is the
confirmation `claude/rules/metafopy.md` asks for before an output changes), or flag it with a comment.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
- Pass: with the start patched in, `pytest src/tests/test_rma_ls.py src/tests/correctness` changes one result only, the strict expected failure
  `ls.confint.scale.ML.identity.bcg.alpha1_lb` passing (1501 passed, 1 skipped, 18 xfailed, against 19 xfailed today).
- Pass: the bound is -0.21432298822, the reference `ls_profile_bounds` -0.214322988803 (6e-10 apart).
<!-- doc-counts: accept-end -->

## The items

### 1. The spec

- **What it changes:** a new `feasible_start` in `src/metafopy/methods/ls_profile.py` (a `scipy.optimize.linprog` start of the largest minimum
  `tau2_i`, as the reference's `ls_profile_identity_starts`), used by `profile_starts` only when every start but the default was dropped as infeasible.
- **Output change:** the ML identity-link `dat.bcg` lower bound of `alpha.1` becomes -0.2143, where it was NaN with "Cannot obtain lower bound". R
  metafor 5.2-1 gives NA there; `docs/differences.md` gains a sentence saying why.
- **Choice made, reversible:** the start is added only as a fallback, not to every refit. Both give the same results; the fallback keeps `confint`
  of the ML fit at 0.49 s (0.39 s today) where adding it everywhere takes 1.14 s, and leaves every refit that has a feasible start unchanged.
- **Out of scope, settled already:** the alpha.1 upper bound stays `>0.2399` (end of the interval searched); the alpha.2 bounds already equal the
  reference, and R's narrower ones are the recorded difference "Profile-likelihood bounds of the scale coefficients".
- **Recommendation:** approve.

## Decisions

None: the one choice the spec makes is reported above.

## Answers

Answered on the review page on 2026-11-19 (16:30 UTC), read back from its `answers` store; no comment:

| Item | Answer |
|---|---|
| 1. The spec | approve |
