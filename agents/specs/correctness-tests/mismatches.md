# Correctness tests: mismatches and sources needed

**Spec:** `agents/specs/correctness-tests/spec.md`
**Task list:** `agents/tasks/correctness-tests/tasks-1.md` (Task 13)
**Review page:** https://claude.ai/artifact/UTpCxfSZ1588zUnpgL9T1T

## What is reviewed and how

The case-writing tasks compared metafopy with the defining publications first and R metafor second. This document holds every case where the
reference and metafopy disagree (one mismatch), and every published source that could not be read (the table of sources needed). For each mismatch
you choose *follow the reference*, *keep metafor* or *investigate*. For each source needed you supply the table or example, or drop the source. The
list stops until every item has an answer.

## Automatic checks

- Every `mismatch` case has an item: `grep -rn 'status = "mismatch"' src/tests/correctness/cases` finds one case, `confint.ht.reml.bcg.none`, and
  the item below covers it.
- The correctness gate: run `pytest src/tests/correctness -m 'not slow'` (result recorded in the Task 13 commit).

## Mismatches

### rma_uni confint type "ht" with a REML fit centres on a different H

- **Case:** `confint.ht.reml.bcg.none` in `src/tests/correctness/cases/inference.toml` (strict xfail).
- **Source:** Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine 21:1539-1558, equations 24 and 25.
- **Reference value** (derived, `formulas.higgins_thompson_bounds`, on `dat.bcg` with log risk ratios): tau2 interval 0.19972 to 0.47038. The interval is
  `ln H +- z se` with H from Q (`H2 = Q / (k - 1)` = 12.686), as the paper defines it.
- **metafopy:** 0.20274 to 0.47703. Its bounds are centred on the `H2` of the REML fit (12.856), with the standard error computed from Q. With a DL fit
  (where `H2` equals `Q / (k - 1)`) the two agree, case `confint.ht.dl.bcg.none` passes.
- **metafor:** 5.2-1 gives 0.20274 to 0.47703, the same as metafopy (checked 2026-11-12 with the audit's R library). `confint.rma.uni`'s `ht` branch
  computes `se[ln H]` from Q and k as the paper does, then centres on `log(x$H2)`, the REML fit's H2, not `Q / (k - 1)`.
- **The paper:** section 4.1.1 and Appendix method III build the interval "involving only Q and k", with `Q = (k - 1) H^2`, so H is from Q. The
  conversion from the H interval to a tau2 interval (`tau2 = I2 v / (1 - I2)`) is metafor's; the paper gives intervals for H and I2 only, and the
  reference value applies the same conversion to the paper's H.
- **metafor's documentation** (read 2026-11-12, metafor 5.2-1): `?print.rma` (Note) documents `H2 = (tau2 + v) / v` as a deliberate, more general
  definition for the printed H2 and I2, so that they agree with the estimated tau2, and says it equals `Q / (k - 1)` under DL. `?confint.rma` says
  `type="HT"` uses method III of Higgins & Thompson (2002), "see also Borenstein et al., 2009, chapter 16", and says nothing about centring the
  interval on the fit's H2. Borenstein et al. (2009) equations 16.14 to 16.19 centre on `ln(Q / df)` and map to tau2 with `df (L^2 - 1) / C`
  (`df / C` is the paper's `v`), which gives the reference value exactly. So both sources metafor cites centre on Q; the REML centring looks like
  the general H2 carried into `confint` without comment.
- **Options:** follow the reference; keep metafor (the REML-centred form is a deliberate generalisation of the paper); investigate (run metafor, then
  decide).

### Decision: what to do with the `ht` interval under REML

- **What is decided:** which interval `confint(type='ht')` returns for a REML fit.
- **Background:** the case is a strict expected failure. The DL fit agrees with the paper; the REML fit centres on the REML `H2`, a generalisation
  of the paper rather than its definition. Metafor 5.2-1 does the same as metafopy (run on 2026-11-12), and does not document it.
- **What it affects:** the `ht` branch of `src/metafopy/methods/confint.py`, `docs/differences.md`, and the R parity fixtures if metafor has the type.
- **Options:**
  - *Follow the reference*: a `TODO.md` item to centre on `H` from Q; the case passes after the fix.
  - *Keep metafor*: the case is rewritten to the REML-centred convention; its only support is metafor's general H2 for printed statistics.
  - *Investigate*: the case stays a strict expected failure and nothing is filed; metafor has already been run and its code read.
- **Recommendation:** *follow the reference* (revised on 2026-11-12 from *investigate*, once metafor had been run and the paper read). The paper is
  unambiguous and `NOTES.md` puts correct algorithms above reproducing metafor; `docs/differences.md` records the difference from metafor.

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

### Full references

Every DOI below was checked against Crossref on 2026-11-12. Hunter and Schmidt (2004) and Raudenbush (2009) have no DOI and were not checked
against a registry.

- Veroniki AA, Jackson D, Viechtbauer W, Bender R, Bowden J, Knapp G, Kuss O, Higgins JPT, Langan D, Salanti G (2016). Methods to estimate the between-study variance and its uncertainty in meta-analysis. Research Synthesis Methods 7(1):55-79 (PMC4950030). doi:10.1002/jrsm.1164 Trial data: Bowden J, Tierney JF, Copas AJ, Burdett S (2011). Quantifying, displaying and accounting for heterogeneity in the meta-analysis of RCTs using standard and generalised Q statistics. BMC Medical Research Methodology 11:41 (open access; not yet checked for the data). doi:10.1186/1471-2288-11-41
- DerSimonian R, Laird N (1986). Meta-analysis in clinical trials. Controlled Clinical Trials 7(3):177-188. doi:10.1016/0197-2456(86)90046-2
- Hedges LV (1983). A random effects model for effect sizes. Psychological Bulletin 93(2):388-395. doi:10.1037/0033-2909.93.2.388
- Hunter JE, Schmidt FL (2004). Methods of Meta-Analysis: Correcting Error and Bias in Research Findings, 2nd ed. Sage, Thousand Oaks, CA.
- Sidik K, Jonkman JN (2005). Simple heterogeneity variance estimation for meta-analysis. Journal of the Royal Statistical Society, Series C (Applied Statistics) 54(2):367-384. doi:10.1111/j.1467-9876.2005.00489.x
- Viechtbauer W (2005). Bias and efficiency of meta-analytic variance estimators in the random-effects model. Journal of Educational and Behavioral Statistics 30(3):261-293. Morris CN (1983). Parametric empirical Bayes inference: theory and applications. Journal of the American Statistical Association 78(381):47-55. doi:10.3102/10769986030003261 doi:10.1080/01621459.1983.10477920
- Paule RC, Mandel J (1982). Consensus values and weighting factors. Journal of Research of the National Bureau of Standards 87(5):377-385 (free at NIST). doi:10.6028/jres.087.022
- DerSimonian R, Kacker R (2007). Random-effects model for meta-analysis of clinical trials: an update. Contemporary Clinical Trials 28(2):105-114. doi:10.1016/j.cct.2006.04.004
- Knapp G, Hartung J (2003). Improved tests for a random effects meta-regression with a single covariate. Statistics in Medicine 22(17):2693-2710. doi:10.1002/sim.1482
- Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine 21(11):1539-1558. doi:10.1002/sim.1186
- Mantel N, Haenszel W (1959). Statistical aspects of the analysis of data from retrospective studies of disease. Journal of the National Cancer Institute 22(4):719-748. Robins J, Breslow N, Greenland S (1986). Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. Biometrics 42(2):311-323. doi:10.1093/jnci/22.4.719 doi:10.2307/2531052
- Yusuf S, Peto R, Lewis J, Collins R, Sleight P (1985). Beta blockade during and after myocardial infarction: an overview of the randomized trials. Progress in Cardiovascular Diseases 27(5):335-371. doi:10.1016/S0033-0620(85)80003-7
- Hedges LV (1981). Distribution theory for Glass's estimator of effect size and related estimators. Journal of Educational Statistics 6(2):107-128. doi:10.3102/10769986006002107
- Hedges LV, Gurevitch J, Curtis PS (1999). The meta-analysis of response ratios in experimental ecology. Ecology 80(4):1150-1156. doi:10.1890/0012-9658(1999)080[1150:TMAORR]2.0.CO;2
- Borenstein M, Hedges LV, Higgins JPT, Rothstein HR (2009). Introduction to Meta-Analysis. Wiley, Chichester. Normand SLT (1999). Meta-analysis: formulating, evaluating, combining, and reporting. Statistics in Medicine 18(3):321-359. Raudenbush SW (2009). Analyzing effect sizes: random-effects models. In Cooper H, Hedges LV, Valentine JC (eds), The Handbook of Research Synthesis and Meta-Analysis, 2nd ed., pp. 295-315. Russell Sage Foundation, New York. doi:10.1002/9780470743386 doi:10.1002/(SICI)1097-0258(19990215)18:3<321::AID-SIM28>3.0.CO;2-P

On 2026-11-12 the user saved the sources in `/Users/simonb/src/metafopy papers` (outside the repository; the papers are not committed).
Every row has its source there, Raudenbush (2009) as chapter 16 of the whole Handbook (2nd ed., 2009), except Hunter & Schmidt (2004): that
file is Schmidt's chapter "Meta-Analysis" (chapter 22 of a Wiley handbook), not the 2004 book. The Handbook's chapter 17 (Schmidt, Le and Oh) also
covers the Hunter-Schmidt approach.

Each source row is a decision: supply the source, or drop it. Recommendation for every row: drop it unless you already hold the source; the
estimators have metafor parity, derived, property and simulation cases.

## Answers

Answered by the user on the review page on 2026-11-12 and copied here the same day.

- **Decision: the `ht` interval under REML** (`confint.ht.reml.bcg.none`): *follow the reference*. No comment.

Sources needed: every source is *supply*. The files are in `/Users/simonb/src/metafopy papers`, outside the repository; they are not committed.

- Veroniki et al. (2016), with Bowden et al. (2011) for the trial data: *supply*.
- DerSimonian & Laird (1986): *supply*.
- Hedges (1983): *supply*.
- Hunter & Schmidt (2004): the file is Schmidt's chapter "Meta-Analysis" (chapter 22 of a Wiley handbook), not the 2004 book; the Handbook of
  Research Synthesis and Meta-Analysis, chapter 17 (Schmidt, Le and Oh) also covers the approach. Whether either has an HS worked example is checked
  when the case is written: *supply*.
- Sidik & Jonkman (2005): *supply*.
- Viechtbauer (2005) and Morris (1983): *supply*.
- Paule & Mandel (1982): *supply*.
- DerSimonian & Kacker (2007): *supply*.
- Knapp & Hartung (2003): *supply*.
- Higgins & Thompson (2002): *supply*.
- Mantel & Haenszel (1959) and Robins, Breslow & Greenland (1986): *supply*.
- Yusuf et al. (1985): *supply*.
- Hedges (1981): *supply*.
- Hedges, Gurevitch & Curtis (1999): *supply*.
- Borenstein et al. (2009), Normand (1999), and Raudenbush (2009) as chapter 16 of the Handbook of Research Synthesis and Meta-Analysis, 2nd ed.: *supply*.
