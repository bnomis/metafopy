# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `RMAMVResult.R` returns a known correlation matrix as given, not cut and scaled as R stores it

<!-- doc-counts: accept-start found on 2026-11-23 - a dated record, not a claim about the present -->
Found on 2026-11-23 by the direct test of `RMAMVResult.R`
(`src/tests/test_rma_mv.py::test_known_correlation_matrix_is_cut_to_the_levels_with_data`, strict `xfail`): for a fit with
`R={'species': <8 x 8 matrix>}` on data with six species, R metafor 5.2-1 gives `res$R$species` as the 6 x 6 matrix of the levels with
data, sorted (`sp1 sp3 sp4 sp6 sp7 sp8`) and scaled by `Rscale` (`all.equal(res$R$species, Rsp[lev, lev])`); metafopy returns all eight
levels as given.
<!-- doc-counts: accept-end -->

Cause: `rma.mv` replaces each given matrix by `R[[j]][s.levels[[j]], s.levels[[j]]]` and then applies `Rscale`
(`.audit/source/metafor/R/rma.mv.R:1484-1510`) before storing it; `RMAMVResult.R` in `src/metafopy/models/mv_result.py:334` returns
`self.spec['R']` unchanged. Whether the fit itself uses the cut and scaled matrix (so only the stored attribute differs) is not checked.

Fix direction: return the matrix the fit used, cut to the levels with data and scaled as `Rscale` says; this changes an output, so it needs
the operator's confirmation first. Then remove the strict `xfail` marker of that test.
