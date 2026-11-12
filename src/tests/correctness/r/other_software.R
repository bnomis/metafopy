# Other-software correctness cases from the R package meta, written to src/tests/correctness/cases/other.toml.
#
# Usage (from the repository root), with meta installed in a library outside this repository (not the audit's pinned library) and its
# dependencies (metafor, metadat, ...) on R's library path, for example the audit's library through R_LIBS:
#   METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software.R
#
# What is independent of metafor: meta computes its own effect sizes (metabin, metacont, metainc) and its own Mantel-Haenszel and Peto
# estimates. It obtains tau2, and so the random-effects estimate, from metafor::rma.uni, so the rma_uni cases check only the pipeline around
# tau2 (weights, Knapp-Hartung intervals), not the estimator.

META_VERSION <- "8.5-0"

lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")
if (!nzchar(lib)) stop("METAFOPY_CORRECTNESS_RLIB is not set: name the R library that holds meta ", META_VERSION)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("meta", quietly = TRUE)) stop("meta is not installed in ", lib)
if (packageVersion("meta") != package_version(META_VERSION)) {
  stop("meta ", packageVersion("meta"), " found in ", lib, "; this script is pinned to ", META_VERSION)
}
suppressPackageStartupMessages(library(meta))

g17 <- function(x) sprintf("%.17g", unname(as.vector(x)))
r_version <- paste(R.version$major, R.version$minor, sep = ".")
today <- format(Sys.Date())
citation <- sprintf(
  "R package meta %s (Balduzzi S, Ruecker G, Schwarzer G (2019). How to perform a meta-analysis with R: a practical tutorial. Evidence-Based Mental Health 22:153-160), run on R %s.",
  META_VERSION, r_version
)

toml_value <- function(v) {
  if (is.list(v)) {
    return(paste0("{ ", paste(sprintf("%s = %s", names(v), vapply(v, toml_value, "")), collapse = ", "), " }"))
  }
  if (is.character(v)) return(sprintf("\"%s\"", v))
  if (is.logical(v)) return(tolower(as.character(v)))
  g17(v)
}

cases <- character(0)
add_case <- function(id, func, args, expect, locator, dataset = NULL, setup = NULL) {
  lines <- c(
    "[[case]]",
    sprintf("id = \"%s\"", id),
    "kind = \"other\"",
    sprintf("function = \"%s\"", func)
  )
  if (!is.null(dataset)) lines <- c(lines, sprintf("dataset = \"%s\"", dataset))
  if (!is.null(setup)) lines <- c(lines, sprintf("setup = %s", toml_value(list(escalc = setup))))
  lines <- c(
    lines,
    sprintf("args = %s", toml_value(args)),
    sprintf("expect = { %s }", paste(sprintf("\"%s\" = %s", names(expect), vapply(expect, g17, "")), collapse = ", ")),
    "",
    "[case.source]",
    sprintf("citation = \"%s\"", citation),
    sprintf("locator = \"%s\"", locator),
    sprintf("read = %s", today),
    "url = \"https://cran.r-project.org/package=meta\"",
    ""
  )
  cases <<- c(cases, lines)
}

bcg <- metadat::dat.bcg
two_by_two <- list(ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg")
expect_vector <- function(prefix, x) setNames(as.list(x), sprintf("%s[%d]", prefix, seq_along(x) - 1L))

### escalc: meta's own effect sizes ------------------------------------------------------------------
for (sm in c("RR", "OR", "RD")) {
  m <- metabin(tpos, tpos + tneg, cpos, cpos + cneg, data = bcg, sm = sm, method = "Inverse", common = TRUE, random = FALSE)
  add_case(
    sprintf("other.escalc.%s.bcg", tolower(sm)), "escalc", c(list(measure = sm), two_by_two),
    c(expect_vector("yi", m$TE), expect_vector("vi", m$seTE^2)),
    sprintf("metabin(sm = '%s', method = 'Inverse'): TE and seTE^2", sm),
    dataset = "dat.bcg"
  )
}

normand <- metadat::dat.normand1999
for (sm in c("MD", "SMD")) {
  m <- metacont(n1i, m1i, sd1i, n2i, m2i, sd2i, data = normand, sm = sm, method.smd = "Hedges", exact.smd = TRUE, common = TRUE, random = FALSE)
  add_case(
    sprintf("other.escalc.%s.normand1999", tolower(sm)), "escalc",
    c(list(measure = sm, m1i = "m1i", sd1i = "sd1i", n1i = "n1i", m2i = "m2i", sd2i = "sd2i", n2i = "n2i"), if (sm == "SMD") list(vtype = "UB")),
    c(expect_vector("yi", m$TE), expect_vector("vi", m$seTE^2)),
    sprintf("metacont(sm = '%s', method.smd = 'Hedges', exact.smd = TRUE): TE and seTE^2%s", sm,
            if (sm == "SMD") "; meta's variance is the unbiased estimator, metafor's vtype = 'UB'" else ""),
    dataset = "dat.normand1999"
  )
}

### rma_mh and rma_peto: meta's own estimators -------------------------------------------------------
for (sm in c("OR", "RR", "RD")) {
  m <- metabin(tpos, tpos + tneg, cpos, cpos + cneg, data = bcg, sm = sm, method = "MH", common = TRUE, random = FALSE)
  add_case(
    sprintf("other.rma_mh.%s.bcg", tolower(sm)), "rma_mh", c(two_by_two, list(measure = sm)),
    if (sm == "RD") list(beta = m$TE.common) else list(beta = m$TE.common, se = m$seTE.common),
    if (sm == "RD") {
      "metabin(sm = 'RD', method = 'MH'): TE.common only; meta's seTE.common is the Greenland and Robins (1985) variance, metafor documents Sato, Greenland and Robins (1989)"
    } else {
      sprintf("metabin(sm = '%s', method = 'MH'): TE.common and seTE.common", sm)
    },
    dataset = "dat.bcg"
  )
}
m <- metabin(tpos, tpos + tneg, cpos, cpos + cneg, data = bcg, sm = "OR", method = "Peto", common = TRUE, random = FALSE)
add_case(
  "other.rma_peto.bcg", "rma_peto", two_by_two, list(beta = m$TE.common, se = m$seTE.common),
  "metabin(sm = 'OR', method = 'Peto'): TE.common and seTE.common", dataset = "dat.bcg"
)

hart <- metadat::dat.hart1999
for (sm in c("IRR", "IRD")) {
  m <- metainc(x1i, t1i, x2i, t2i, data = hart, sm = sm, method = "MH", common = TRUE, random = FALSE)
  add_case(
    sprintf("other.rma_mh.%s.hart1999", tolower(sm)), "rma_mh", list(x1i = "x1i", t1i = "t1i", x2i = "x2i", t2i = "t2i", measure = sm),
    list(beta = m$TE.common, se = m$seTE.common),
    sprintf("metainc(sm = '%s', method = 'MH'): TE.common and seTE.common", sm),
    dataset = "dat.hart1999"
  )
}

### rma_uni: weights, Q and Knapp-Hartung around metafor's tau2 ----------------------------------------
setup_rr <- c(list(measure = "RR"), two_by_two)
rr <- metabin(tpos, tpos + tneg, cpos, cpos + cneg, data = bcg, sm = "RR", method = "Inverse", common = TRUE, random = FALSE)
for (tau in c("DL", "REML")) {
  m <- metagen(rr$TE, rr$seTE, method.tau = tau, common = FALSE, random = TRUE)
  expect <- list(beta = m$TE.random, se = m$seTE.random, tau2 = m$tau2, QE = m$Q)
  if (tau == "DL") expect$I2 <- 100 * m$I2
  add_case(
    sprintf("other.rma_uni.%s.z.bcg", tolower(tau)), "rma_uni", list(yi = "yi", vi = "vi", method = tau, test = "z"), expect,
    sprintf("metagen(method.tau = '%s'): TE.random, seTE.random, tau2, Q%s", tau, if (tau == "DL") ", I2 (as a percentage)" else ""),
    dataset = "dat.bcg", setup = setup_rr
  )
  k <- metagen(rr$TE, rr$seTE, method.tau = tau, method.random.ci = "HK", common = FALSE, random = TRUE)
  add_case(
    sprintf("other.rma_uni.%s.knha.bcg", tolower(tau)), "rma_uni", list(yi = "yi", vi = "vi", method = tau, test = "knha"),
    list(se = k$seTE.random, ci_lb = k$lower.random, ci_ub = k$upper.random),
    sprintf("metagen(method.tau = '%s', method.random.ci = 'HK'): seTE.random, lower.random, upper.random", tau),
    dataset = "dat.bcg", setup = setup_rr
  )
}

header <- c(
  "# Other-software cases from R meta, written by src/tests/correctness/r/other_software.R; do not edit by hand.",
  "#",
  "# meta's effect sizes, Mantel-Haenszel and Peto estimates are its own code. Its tau2 comes from metafor::rma.uni, so the rma_uni cases check",
  "# the weights, Q and Knapp-Hartung intervals around tau2, not the estimator.",
  ""
)
out <- file.path("src", "tests", "correctness", "cases", "other.toml")
writeLines(c(header, cases), out)
cat("wrote", out, "\n")
