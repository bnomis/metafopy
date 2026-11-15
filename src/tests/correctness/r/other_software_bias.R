# Other-software publication-bias cases from the R packages weightr and meta, written to src/tests/correctness/cases/other_bias.toml.
#
# Usage (from the repository root), with weightr and meta installed in a library outside this repository (not the audit's pinned library) and
# their dependencies (ggplot2, scales, metafor, metadat, ...) on R's library path, for example the audit's library through R_LIBS:
#   METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_bias.R
#
# What is independent of metafor: weightr (Vevea-Hedges step-function models) does not depend on metafor, so its cases check selmodel's step
# functions against separate code. meta computes its own Egger, Begg and Peters tests and its own trim-and-fill, but it takes tau2 from
# metafor::rma.uni for every method other than DL, so the random-effects refits (trimfill) use DL, which meta computes itself; the remaining
# dependence on metafor is the pipeline around tau2, not the estimator.

WEIGHTR_VERSION <- "2.0.2"
META_VERSION <- "8.5-0"

lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")
if (!nzchar(lib)) stop("METAFOPY_CORRECTNESS_RLIB is not set: name the R library that holds weightr ", WEIGHTR_VERSION, " and meta ", META_VERSION)
.libPaths(c(lib, .libPaths()))
for (pkg in c("weightr", "meta")) {
  want <- if (pkg == "weightr") WEIGHTR_VERSION else META_VERSION
  if (!requireNamespace(pkg, quietly = TRUE)) stop(pkg, " is not installed in ", lib)
  if (packageVersion(pkg) != package_version(want)) stop(pkg, " ", packageVersion(pkg), " found in ", lib, "; this script is pinned to ", want)
}
suppressPackageStartupMessages({
  library(weightr)
  library(meta)
})

g17 <- function(x) sprintf("%.17g", unname(as.vector(x)))
r_version <- paste(R.version$major, R.version$minor, sep = ".")
today <- format(Sys.Date())
weightr_citation <- sprintf(
  "R package weightr %s (Coburn KM, Vevea JL (2019). weightr: Estimating Weight-Function Models for Publication Bias), run on R %s. Vevea JL, Hedges LV (1995). A general linear model for estimating effect size in the presence of publication bias. Psychometrika 60:419-435.",
  WEIGHTR_VERSION, r_version
)
meta_citation <- sprintf(
  "R package meta %s (Balduzzi S, Ruecker G, Schwarzer G (2019). How to perform a meta-analysis with R: a practical tutorial. Evidence-Based Mental Health 22:153-160), run on R %s.",
  META_VERSION, r_version
)

toml_value <- function(v) {
  if (is.list(v)) {
    if (!is.null(names(v))) return(paste0("{ ", paste(sprintf("%s = %s", names(v), vapply(v, toml_value, "")), collapse = ", "), " }"))
    return(paste0("[", paste(vapply(v, toml_value, ""), collapse = ", "), "]"))
  }
  if (is.character(v)) return(if (length(v) == 1) sprintf("\"%s\"", v) else paste0("[", paste(sprintf("\"%s\"", v), collapse = ", "), "]"))
  if (is.logical(v)) return(tolower(as.character(v)))
  if (length(v) > 1) return(paste0("[", paste(g17(v), collapse = ", "), "]"))
  g17(v)
}

cases <- character(0)
# data is either a bundled dataset name (dataset) or inline columns (inline, a named list of equal-length vectors).
add_case <- function(id, func, args, expect, citation, locator, url, dataset = NULL, inline = NULL, setup = NULL, tolerance = NULL, reason = NULL) {
  lines <- c("[[case]]", sprintf("id = \"%s\"", id), "kind = \"other\"", sprintf("function = \"%s\"", func))
  if (!is.null(dataset)) lines <- c(lines, sprintf("dataset = \"%s\"", dataset))
  if (!is.null(inline)) lines <- c(lines, sprintf("data = %s", toml_value(lapply(inline, as.list))))
  if (!is.null(setup)) lines <- c(lines, sprintf("setup = %s", toml_value(setup)))
  lines <- c(
    lines,
    sprintf("args = %s", toml_value(args)),
    sprintf("expect = { %s }", paste(sprintf("\"%s\" = %s", names(expect), vapply(expect, g17, "")), collapse = ", "))
  )
  if (!is.null(tolerance)) lines <- c(lines, sprintf("tolerance = %s", g17(tolerance)))
  if (!is.null(reason)) lines <- c(lines, "status = \"mismatch\"", sprintf("reason = \"%s\"", reason))
  lines <- c(
    lines, "", "[case.source]",
    sprintf("citation = \"%s\"", citation),
    sprintf("locator = \"%s\"", locator),
    sprintf("read = %s", today),
    sprintf("url = \"%s\"", url),
    ""
  )
  cases <<- c(cases, lines)
}

### weightr: Vevea-Hedges step-function models ----------------------------------------------------------------------------
# weightr uses one-sided p-values for positive effects, as selmodel does for alternative = "greater"; estimation is maximum likelihood.
# adj_est is (tau2, mean, weights of the intervals after the first); metafopy's delta holds the first weight (1) as well.
weightr_url <- "https://cran.r-project.org/package=weightr"
bd <- metadat::dat.bangertdrowns2004
bd <- bd[!is.na(bd$yi) & !is.na(bd$vi), ]
step_case <- function(id, steps, weights = NULL, tolerance = 1e-4) {
  fit <- weightfunct(effect = bd$yi, v = bd$vi, steps = steps, weights = weights)
  est <- as.vector(fit$adj_est)
  se <- as.vector(fit$adj_se)
  if (is.null(weights)) {
    free <- est[-(1:2)]
    expect <- c(
      list(`beta[0]` = est[2], `se[0]` = se[2], tau2 = est[1]),
      setNames(as.list(free), sprintf("delta[%d]", seq_along(free))),
      list(LRT = 2 * (fit$output_unadj$value - fit$output_adj$value))
    )
    add_case(
      id, "selmodel",
      list(type = "stepfun", steps = as.list(steps)), expect, weightr_citation,
      sprintf("weightfunct(steps = c(%s)) on dat.bangertdrowns2004 (k = %d): adj_est (tau2, mean, weights), adj_se, and the likelihood-ratio statistic 2 * (logLik adjusted - logLik unadjusted)", paste(steps, collapse = ", "), nrow(bd)),
      weightr_url, dataset = "dat.bangertdrowns2004", setup = list(rma_uni = list(method = "ML")), tolerance = tolerance
    )
  } else {
    # with a priori weights weightr returns no standard errors and pads adj_est with placeholders after (tau2, mean)
    expect <- list(`beta[0]` = est[2], tau2 = est[1])
    add_case(
      id, "selmodel",
      list(type = "stepfun", steps = as.list(steps), delta = as.list(weights)), expect, weightr_citation,
      sprintf("weightfunct(steps = c(%s), weights = c(%s)) on dat.bangertdrowns2004 (k = %d): adj_est (tau2, mean), with the a priori weights fixed", paste(steps, collapse = ", "), paste(weights, collapse = ", "), nrow(bd)),
      weightr_url, dataset = "dat.bangertdrowns2004", setup = list(rma_uni = list(method = "ML")), tolerance = tolerance
    )
  }
}
step_case("other.bias.weightr.estimated.two_intervals", c(0.025, 1))
# the three-interval likelihood is flat in the weights: the two optimisers stop 2e-4 apart in delta, so the tolerance is 1e-3
step_case("other.bias.weightr.estimated.three_intervals", c(0.025, 0.5, 1), tolerance = 1e-3)
step_case("other.bias.weightr.apriori.three_intervals", c(0.025, 0.5, 1), weights = c(1, 0.7, 0.5))

### meta: Egger, Begg and Peters tests, and trim and fill ----------------------------------------------------------------
meta_url <- "https://cran.r-project.org/package=meta"
hack <- metadat::dat.hackshaw1998
hm <- metagen(TE = hack$yi, seTE = sqrt(hack$vi), common = TRUE, random = FALSE)

eg <- metabias(hm, method.bias = "Egger")
add_case(
  "other.bias.meta.egger.hackshaw1998", "regtest", list(x = "yi", vi = "vi", model = "lm", predictor = "sei"),
  list(zval = eg$statistic, pval = eg$p.value, est = eg$intercept), meta_citation,
  "metabias(metagen(yi, sqrt(vi)), method.bias = 'Egger'): statistic (t on the bias, k - 2 degrees of freedom), p.value, intercept (the limit estimate at zero standard error)",
  meta_url, dataset = "dat.hackshaw1998", tolerance = 1e-8
)

eggerdat <- metadat::dat.egger2001[-16, ]
eggerdat <- metafor::escalc(measure = "OR", ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = eggerdat)
em <- metagen(TE = eggerdat$yi, seTE = sqrt(eggerdat$vi), common = TRUE, random = FALSE)
eg2 <- metabias(em, method.bias = "Egger")
add_case(
  "other.bias.meta.egger.egger2001", "regtest", list(x = "yi", vi = "vi", model = "lm", predictor = "sei"),
  list(zval = eg2$statistic, pval = eg2$p.value, est = eg2$intercept), meta_citation,
  "metabias(metagen(yi, sqrt(vi)), method.bias = 'Egger'): statistic, p.value, intercept; dat.egger2001 without row 16 (ISIS-4, as Egger et al. 1997), log odds ratios from metafor::escalc with its default 0.5 correction",
  meta_url, dataset = "dat.egger2001",
  setup = list(drop = list(15), escalc = list(measure = "OR", ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i")), tolerance = 1e-8
)

bg <- metabias(hm, method.bias = "Begg")
k <- nrow(hack)
add_case(
  "other.bias.meta.begg.hackshaw1998", "ranktest", list(x = "yi", vi = "vi", exact = FALSE),
  list(tau = bg$estimate[["ks"]] / (k * (k - 1) / 2), pval = bg$p.value), meta_citation,
  "metabias(metagen(yi, sqrt(vi)), method.bias = 'Begg'): p.value (normal approximation) and Kendall tau = ks / (k (k - 1) / 2) from estimate['ks']",
  meta_url, dataset = "dat.hackshaw1998", tolerance = 1e-8
)

# Peters' test regresses the log odds ratio on 1 / n with weights from the event counts; meta needs the counts, so the case carries inline data.
pm <- metabin(ai, n1i, ci, n2i, data = metadat::dat.egger2001[-16, ], sm = "OR", method = "Inverse", common = TRUE, random = FALSE)
pe <- metabias(pm, method.bias = "Peters")
pdat <- metadat::dat.egger2001[-16, ]
add_case(
  "other.bias.meta.peters.egger2001", "regtest", list(x = "yi", vi = "vi", model = "lm", predictor = "ninv", ni = "ni"),
  list(zval = pe$statistic, pval = pe$p.value), meta_citation,
  "metabias(metabin(sm = 'OR'), method.bias = 'Peters'): statistic and p.value; dat.egger2001 without row 16 (ISIS-4); meta weights by the event counts, metafopy by 1/vi",
  meta_url,
  inline = list(yi = pm$TE, vi = pm$seTE^2, ni = pdat$n1i + pdat$n2i), tolerance = 1e-8,
  reason = sprintf("meta's Peters test (Peters et al. 2006) is a weighted regression with weights 1 / (1 / (a + c) + 1 / (b + d)) from the event counts and gives t = %s; metafopy's regtest(model = 'lm', predictor = 'ninv') weights by 1 / vi, as metafor does, and gives t = -2.32297, and regtest accepts no other weights", g17(pe$statistic))
)

# Trim and fill: meta's own DerSimonian-Laird tau2, left side determined as metafopy does, the fixed-effect fit not used to place the centre.
for (est in c("L0", "R0")) {
  tf <- trimfill(metagen(TE = hack$yi, seTE = sqrt(hack$vi), method.tau = "DL", common = FALSE, random = TRUE), estimator = est, ma.common = FALSE, common = FALSE, random = TRUE)
  add_case(
    sprintf("other.bias.meta.trimfill.%s.hackshaw1998", tolower(est)), "trimfill", list(estimator = est),
    list(k0 = tf$k0, beta = tf$TE.random), meta_citation,
    sprintf("trimfill(metagen(method.tau = 'DL'), estimator = '%s', ma.common = FALSE): k0 and TE.random of the filled data", est),
    meta_url, dataset = "dat.hackshaw1998", setup = list(rma_uni = list(method = "DL")), tolerance = 1e-8
  )
}

header <- c(
  "# Other-software publication-bias cases from R weightr and R meta, written by src/tests/correctness/r/other_software_bias.R; do not edit by hand.",
  "#",
  "# weightr does not depend on metafor, so its step-function cases check selmodel against separate code. meta's Egger, Begg and Peters tests and its",
  "# trim and fill are its own code; its random-effects refits take tau2 from DerSimonian-Laird (meta's own), so they check the pipeline around tau2,",
  "# not the estimator. meta's Harbord test has no counterpart in metafopy's regtest (its predictors are sei, vi, ni, ninv, sqrtni and sqrtninv), so",
  "# it has no case.",
  ""
)
out <- file.path("src", "tests", "correctness", "cases", "other_bias.toml")
writeLines(c(header, cases), out)
cat("wrote", out, "\n")
