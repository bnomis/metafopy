# Correctness tests: mismatches and sources needed

**Spec:** `agents/specs/correctness-tests/spec.md`
**Status:** items are being collected by the case-writing tasks; Task 13 of `agents/tasks/correctness-tests/tasks-1.md` completes this document in the
review-document layout and publishes its page.

## Mismatches

### rma_uni confint type "ht" with a REML fit centres on a different H

- **Case:** `confint.ht.reml.bcg.none` in `src/tests/correctness/cases/inference.toml` (strict xfail).
- **Source:** Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine 21:1539-1558, equations 24 and 25.
- **Reference value** (derived, `formulas.higgins_thompson_bounds`, on `dat.bcg` with log risk ratios): tau2 interval 0.19972 to 0.47038. The interval is
  `ln H +- z se` with H from Q (`H2 = Q / (k - 1)` = 12.686), as the paper defines it.
- **metafopy:** 0.20274 to 0.47703. Its bounds are centred on the `H2` of the REML fit (12.856), with the standard error computed from Q. With a DL fit
  (where `H2` equals `Q / (k - 1)`) the two agree, case `confint.ht.dl.bcg.none` passes.
- **metafor:** not checked. R `metafor` is not installed in the library this session can reach, and `src/tests/reference/` holds no `ht` fixture.
- **Options:** follow the reference; keep metafor (the REML-centred form is a deliberate generalisation of the paper); investigate (run metafor, then
  decide).

## Sources needed

No `published` case has been written yet. Each seed source of the spec was looked up on 2026-11-11 through the PubMed full-text tool (PubMed Central
only). A value is recorded only when it can be read from the source and the data it was computed from are in a metafopy dataset or printed beside it;
none of the sources below met both. The user is asked to supply the table or example named.

| Source | Wanted | Why no case |
|---|---|---|
| Veroniki et al. (2016), Res Synth Methods 7:55-79, doi 10.1002/jrsm.1164 (PMC4950030) | Table 4: tau2 by estimator for four MRC meta-analyses (Sarcoma, Cervix2, NSCLC1, NSCLC4) | Fetched and Table 4 read, but the trial-level effects and variances are in the MRC review (Bowden et al.), not in the article or any metafopy dataset. Supply those data and the cases can be written. |
| DerSimonian & Laird (1986), Control Clin Trials 7:177-188 | Worked example: tau2 and pooled estimate | Not open access; not in PubMed Central. |
| Hedges (1983), Psychol Bull 93:388-395 | Worked example for HE | Not open access. |
| Hunter & Schmidt (2004), Methods of Meta-Analysis, 2nd ed. | Worked example for HS | Textbook. |
| Sidik & Jonkman (2005), JRSS C 54:367-384 | Worked example for SJ | Not open access. |
| Viechtbauer (2005), JEBS 30:261-293; Morris (1983), JASA 78:47-55 | Worked examples for ML, REML, EB | Not open access. |
| Paule & Mandel (1982), J Res NBS 87:377-385 | Worked example for PM | Open at NIST, but not reachable through the PubMed, Paperclip or alphaXiv tools. |
| DerSimonian & Kacker (2007), Contemp Clin Trials 28:105-114 | Worked example for GENQ | Not open access. |
| Knapp & Hartung (2003), Stat Med 22:2693-2710 | Worked example for the KH test | Not open access. |
| Higgins & Thompson (2002), Stat Med 21:1539-1558 | I2 and H2 of the worked examples | Not open access; not in PubMed Central. |
| Mantel & Haenszel (1959), JNCI 22:719-748; Robins, Breslow & Greenland (1986), Biometrics 42:311-323 | Worked examples for MH estimate and variance | Not open access. |
| Yusuf et al. (1985), Prog Cardiovasc Dis 27:335-371 | Peto summary of `dat.yusuf1985` | Not open access. |
| Hedges (1981), JEBS 6:107-128 | Worked example for SMD | Not open access. |
| Hedges, Gurevitch & Curtis (1999), Ecology 80:1150-1156 | ROM on `dat.curtis1998` | Not open access. |
| Borenstein et al., Introduction to Meta-Analysis; Normand (1999), Stat Med 18:321-359; Raudenbush (2009) | Worked examples on `dat.normand1999`, `dat.raudenbush1985` | Textbooks and paywalled papers. |
