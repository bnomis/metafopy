# metafopy notes

Some notes.

## Correctness

Correct algorithm implementation is more important that reproducing metafor. That is, if metafor has bugs we don't want to reproduce them.

## Citkowicz and Vevea (2017): revisit if the paper turns up

Added 2026-11-15. Citkowicz M, Vevea JL (2017). A parsimonious weight function
for modeling publication bias. Psychological Methods 22(1):28-41.
doi:10.1037/met0000119. It defines the beta weight function of
`selmodel(type='beta')`.

We could not get the PDF (it needs APA access). The papers folder
(`/Users/simonb/src/metafopy papers`, outside the repository) holds only
Citkowicz's 2012 dissertation and the article's supplementary materials:
`MCitkowicz_suppmat.docx` (the authors' R code for the likelihood) and
`MCitkowicz_suppmat_data.docx` (the `dat.baskerville2012` example data).
The supplement does not print the paper's estimates.

So the publication-bias correctness tests
(`agents/specs/correctness-publication-bias/spec.md`) check the beta case
against the authors' code run in R, not against the paper's printed
values; the paper is marked dropped on 2026-11-15. If we find the paper
later, revisit: add its printed estimates for `dat.baskerville2012` as a
published case (they may not be reproducible, since that likelihood is
flat; see `docs/differences.md`).
