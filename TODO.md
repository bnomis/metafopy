# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Location-scale fits can stop at a lower local maximum than the test reference

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
Found on 2026-11-24 while measuring `ls.permutest.location.exact.ML.z.bcglat` on macOS arm64 at `282a6b9`
(`agents/specs/location-scale-fits-stable-across-platforms/measurements.md`). On 8 of the 1210 regular permutations (indices 68, 264, 433, 651,
805, 979, 1110, 1278 in `distinct_label_orders` order) metafopy's `z` differs from the test reference's by more than 1e-4. On permutation 68
metafopy and R both stop at log-likelihood `-12.58248229` (alpha `[-0.778, -0.951]`); the test's reference fit reaches the higher
`-12.57323737` (alpha `[-3.042, 2.526]`). Permutations 264 and 433 likewise.
<!-- doc-counts: accept-end -->

Cause: unconfirmed. The likelihood has at least two local maxima on these permutations; metafopy and R share the lower one, the reference
fit, which starts elsewhere, reaches the higher.

Fix direction: decide which maximum metafopy should report (the higher one is the better fit, but R's is the lower), compare R's behaviour on
the eight permutations, and if metafopy should search more starts, write a spec first. Filed apart from
`agents/specs/location-scale-fits-stable-across-platforms/spec.md`, whose decision D3 checks the exact permutation cases against widened
bounds only.

Note 2026-11-24: permutation 433 alone puts metafopy one over the reference bound of `ls.permutest.location.exact.ML.z.bcglat`, now a
strict-xfail mismatch naming this item; remove its `status = "mismatch"` once this item is fixed.

## Auditor rev 133ca2c reports unaccepted hits under three new rules

<!-- doc-counts: accept-start found on 2026-11-24 - a dated record, not a claim about the present -->
Found on 2026-11-24 when the auditor hooks' `rev` in `.pre-commit-config.yaml` was moved from `3e232368c42ae4147aa8607ac73570adfc3f786f` to
`133ca2c819d32fbc369ff921c1eba491ab36e222` (commit `d35c7a7`): the pre-push hook `auditme-all` refused the push of `main` with 29 unaccepted hits,
all under rules the earlier rev does not have. The bump was reverted so the CI fix could be pushed (operator choice, 2026-11-24).

- `discarded-tuple-return` (a returned tuple is thrown away): `src/metafopy/plot/gosh.py:280`, `:282`, `:295`, `src/metafopy/plot/infl.py:111`,
  `:266`, `src/metafopy/plot/llplot.py:278`, `src/metafopy/plot/profile.py:83`, `src/metafopy/plot/radial.py:194`, `src/metafopy/plot/rma.py:77`,
  `:90`, `src/metafopy/plot/selmodel.py:230` (each discards `metafopy.plot.plain.plain_plot`'s tuple), `src/tests/correctness/test_derived.py:343`,
  `src/tests/test_glmm_extras.py:141`;
- `assert-never-missing` (an `if/elif` chain on a string ends in a fall-through `else`, not `typing.assert_never`): `src/audit/pycall.py:88`,
  `src/metafopy/bias/fsn.py:96`, `src/metafopy/core/padjust.py:69`, `src/metafopy/effects/measures.py:1999`, `src/metafopy/methods/anova.py:195`,
  `src/metafopy/methods/emm_adjust.py:223`, `:308`, `src/metafopy/methods/emm_contrasts.py:221`, `src/metafopy/models/mv_structs.py:538`,
  `src/metafopy/models/tau2.py:353`, `src/metafopy/plot/forest.py:1238`, `src/metafopy/plot/labbe.py:356`, `src/metafopy/plot/rutils.py:339`;
- `pure-name-has-side-effect` (a query-like name with a side effect): `src/audit/steps.py:98`, `src/metafopy/core/rng.py:267`,
  `src/tests/auditsnippets.py:44`.
<!-- doc-counts: accept-end -->

Cause: the new rules are stricter than the code was written to; each hit is either a real cleanup or a site to accept with a reason.

Fix direction: move the rev to `133ca2c` again on a branch, then fix or accept each hit (`# auditme: accept[<rule>] <reason>`); an
`assert-never-missing` fix that changes which input raises changes behaviour, so check each chain's `else` before replacing it. Push only when
`pre-commit run auditme-all --hook-stage manual --all-files` passes.

