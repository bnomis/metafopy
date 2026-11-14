# Other-software correctness cases for rma_mv and robust from the R packages mvmeta, nlme, robumeta and sandwich, written to
# src/tests/correctness/cases/other_mv.toml.
#
# Usage (from the repository root), with the four packages in the audit's pinned library read through R_LIBS and METAFOPY_CORRECTNESS_RLIB naming a
# further library (it may be empty) that is put first on R's library path, as r/other_software.R does:
#   METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_mv.R
#
# None of the four packages depends on metafor (checked below), so the values are independent of it. Nothing is installed.
#
# Conventions pinned here:
#   - robumeta's correlated-effects weights come with a working model whose covariance matrix is diag(1 / weight): the case gives metafopy
#     V = 1 / weight and W = weight, so that the CR2 adjustment of the club_sandwich path sees the same working model as robumeta.
#   - nlme's lme with sigma fixed at 1 is compared by ML. With method = "REML" and sigma fixed, nlme maximises a criterion that is not the restricted
#     likelihood of a model with known sampling variances (observed on 2026-11-14: the restricted log-likelihood of that model, computed with
#     another implementation, is higher at that implementation's own estimates than at nlme's), so the two give different variance components;
#     with method = "ML" they agree.
#   - sandwich::vcovCL has no factor n / (n - p) (metafor's adjust = TRUE); its HC0 with cadjust = TRUE is G / (G - 1) (metafor's "Stata2") and
#     its HC1 with cadjust = TRUE is G / (G - 1) * (N - 1) / (N - p) (metafor's "Stata1"), so those and the unadjusted form are compared.

PACKAGE_VERSIONS <- c(mvmeta = "1.0.3", nlme = "3.1-171", robumeta = "2.1", sandwich = "3.1-3")

lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")
if (!nzchar(lib)) stop("METAFOPY_CORRECTNESS_RLIB is not set: name an R library to put first on the library path")
.libPaths(c(lib, .libPaths()))
for (pkg in names(PACKAGE_VERSIONS)) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop(pkg, " is not installed on the library path")
  if (packageVersion(pkg) != package_version(PACKAGE_VERSIONS[[pkg]])) {
    stop(pkg, " ", as.character(packageVersion(pkg)), " found; this script is pinned to ", PACKAGE_VERSIONS[[pkg]])
  }
}
dependencies <- tools::package_dependencies(names(PACKAGE_VERSIONS), db = installed.packages(), recursive = TRUE)
if (any(vapply(dependencies, function(d) "metafor" %in% d, logical(1)))) stop("one of ", paste(names(PACKAGE_VERSIONS), collapse = ", "), " depends on metafor")
suppressPackageStartupMessages({
  library(mvmeta)
  library(nlme)
  library(robumeta)
  library(sandwich)
})

g17 <- function(x) sprintf("%.17g", unname(as.vector(x)))
r_version <- paste(R.version$major, R.version$minor, sep = ".")
today <- format(Sys.Date())

cases <- character(0)
add_case <- function(id, func, dataset = NULL, setup = NULL, args, expect, then = NULL, data = NULL, pkg, citation, locator, tolerance = NULL) {
  lines <- c("[[case]]", sprintf("id = \"%s\"", id), "kind = \"other\"", sprintf("function = \"%s\"", func))
  if (!is.null(dataset)) lines <- c(lines, sprintf("dataset = \"%s\"", dataset))
  if (!is.null(setup)) lines <- c(lines, sprintf("setup = %s", setup))
  lines <- c(
    lines,
    sprintf("args = %s", args),
    sprintf("expect = { %s }", paste(sprintf("\"%s\" = %s", names(expect), vapply(expect, g17, "")), collapse = ", "))
  )
  if (!is.null(then)) lines <- c(lines, sprintf("then = %s", then))
  if (!is.null(tolerance)) lines <- c(lines, sprintf("tolerance = %s", g17(tolerance)))
  lines <- c(lines, "")
  if (!is.null(data)) {
    lines <- c(lines, "[case.data]", unlist(lapply(names(data), function(n) {
      values <- if (is.character(data[[n]])) sprintf("\"%s\"", data[[n]]) else g17(data[[n]])
      sprintf("%s = [%s]", n, paste(values, collapse = ", "))
    })), "")
  }
  lines <- c(
    lines,
    "[case.source]",
    sprintf("citation = \"%s\"", sprintf("%s %s, run on R %s. %s", pkg, as.character(packageVersion(pkg)), r_version, citation)),
    sprintf("locator = \"%s\"", locator),
    sprintf("read = %s", today),
    sprintf("url = \"https://cran.r-project.org/package=%s\"", pkg),
    ""
  )
  cases <<- c(cases, lines)
}
vector_expect <- function(prefix, x) setNames(as.list(as.vector(x)), sprintf("%s[%d]", prefix, seq_along(x) - 1L))

### mvmeta: unstructured between-trial covariance, REML, on dat.berkey1998 ---------------------------
berkey <- metadat::dat.berkey1998
trials <- unique(berkey$trial)
wide <- do.call(rbind, lapply(trials, function(t) {
  rows <- berkey[berkey$trial == t, ]
  al <- rows[rows$outcome == "AL", ]
  pd <- rows[rows$outcome == "PD", ]
  data.frame(yAL = al$yi, yPD = pd$yi, vAL = al$vi, cov = al$v1i, vPD = pd$vi)
}))
if (!isTRUE(all.equal(wide$cov, do.call(rbind, lapply(trials, function(t) berkey[berkey$trial == t & berkey$outcome == "PD", "v2i"]))[, 1]))) {
  stop("the covariance of the two outcomes differs between the two rows of a trial")
}
mv <- mvmeta(cbind(yAL, yPD), S = as.matrix(wide[, c("vAL", "cov", "vPD")]), data = wide, method = "reml", control = list(reltol = 1e-12, maxit = 1000))
psi <- mv$Psi
add_case(
  "other.mv.un.berkey", "rma_mv", dataset = "dat.berkey1998",
  setup = "{ vblocks = { cluster = \"trial\", columns = [\"v1i\", \"v2i\"] } }",
  args = "{ yi = \"yi\", mods = \"~ 0 + outcome\", random = \"~ outcome | trial\", struct = \"UN\" }",
  expect = c(
    vector_expect("beta", coef(mv)), vector_expect("se", sqrt(diag(vcov(mv)))),
    vector_expect("tau2", diag(psi)), list("rho[0]" = psi[1, 2] / sqrt(psi[1, 1] * psi[2, 2]))
  ),
  pkg = "mvmeta",
  citation = "Gasparrini A, Armstrong B, Kenward MG (2012). Multivariate meta-analysis for non-linear and other multi-parameter associations. Statistics in Medicine 31:3821-3839.",
  locator = "mvmeta(cbind(yAL, yPD), S, method = 'reml'): coef, sqrt(diag(vcov)), Psi (outcomes ordered AL, PD as the outcome factor of the metafopy case)",
  tolerance = 1e-5
)

### nlme: three-level model with known sampling variances --------------------------------------------
kon <- metadat::dat.konstantopoulos2011
kon$district <- factor(kon$district)
kon$school <- factor(kon$school)
fit <- lme(yi ~ year, random = ~ 1 | district / school, weights = varFixed(~ vi), data = kon, method = "ML", control = lmeControl(sigma = 1))
vc <- VarCorr(fit)
sigma2 <- as.numeric(c(vc[2, "Variance"], vc[4, "Variance"]))
add_case(
  "other.mv.threelevel.kon", "rma_mv", dataset = "dat.konstantopoulos2011",
  args = "{ yi = \"yi\", V = \"vi\", random = \"~ 1 | district/school\", mods = \"~ year\", method = \"ML\" }",
  expect = c(vector_expect("beta", fixef(fit)), vector_expect("se", sqrt(diag(vcov(fit)))), vector_expect("sigma2", sigma2)),
  pkg = "nlme",
  citation = "Pinheiro JC, Bates DM (2000). Mixed-effects models in S and S-PLUS. Springer.",
  locator = "lme(yi ~ year, random = ~ 1 | district/school, weights = varFixed(~ vi), control = lmeControl(sigma = 1)), method ML: fixef, vcov, VarCorr (district, then school within district); sigma fixed at 1 makes the residual variance the known vi",
  tolerance = 1e-5
)

### robumeta: correlated-effects RVE with Tipton's small-sample correction -----------------------------
tanner <- metadat::dat.tannersmith2016
robu_fit <- robu(yi ~ propmale, data = tanner, studynum = studyid, var.eff.size = vi, modelweights = "CORR", rho = 0.8, small = TRUE)
if (!identical(robu_fit$data.full$effect.size, tanner$yi)) stop("robumeta reordered the rows")
weights <- robu_fit$data.full$r.weights
table <- robu_fit$reg_table
frame <- list(studyid = tanner$studyid, yi = tanner$yi, vi = tanner$vi, propmale = tanner$propmale, w = weights, vw = 1 / weights)
add_case(
  "other.robust.club.robumeta.tannersmith", "rma_mv",
  args = "{ yi = \"yi\", V = \"vw\", W = \"w\", mods = \"~ propmale\", method = \"FE\" }",
  then = "{ function = \"robust\", args = { cluster = \"studyid\", club_sandwich = true, vcov = \"CR2\" } }",
  data = frame,
  expect = c(
    vector_expect("beta", table$b.r), vector_expect("se", table$SE), vector_expect("dfs", table$dfs),
    vector_expect("pval", table$prob), vector_expect("ci_lb", table$CI.L), vector_expect("ci_ub", table$CI.U)
  ),
  pkg = "robumeta",
  citation = "Fisher Z, Tipton E (2015). robumeta: an R-package for robust variance estimation in meta-analysis. arXiv:1503.02220.",
  locator = "robu(yi ~ propmale, studynum = studyid, var.eff.size = vi, modelweights = 'CORR', rho = 0.8, small = TRUE): reg_table (b.r, SE, dfs, prob, CI.L, CI.U); the weights are data.full$r.weights, the working covariance is diag(1 / weight)",
  tolerance = 1e-6
)

### sandwich: cluster-robust covariance of a weighted lm -------------------------------------------------
lm_fit <- lm(yi ~ propmale, data = tanner, weights = 1 / vi)
variants <- list(
  list(id = "cr0", adjust = "false", type = "HC0", cadjust = FALSE, label = "type = 'HC0', cadjust = FALSE (no factor)"),
  list(id = "stata2", adjust = "\"Stata2\"", type = "HC0", cadjust = TRUE, label = "type = 'HC0', cadjust = TRUE (G / (G - 1))"),
  list(id = "stata1", adjust = "\"Stata1\"", type = "HC1", cadjust = TRUE, label = "type = 'HC1', cadjust = TRUE (G / (G - 1) * (N - 1) / (N - p))")
)
for (v in variants) {
  vc <- vcovCL(lm_fit, cluster = tanner$studyid, type = v$type, cadjust = v$cadjust)
  add_case(
    sprintf("other.robust.sandwich.%s.tannersmith", v$id), "rma_mv", dataset = "dat.tannersmith2016",
    args = "{ yi = \"yi\", V = \"vi\", mods = \"~ propmale\", method = \"FE\" }",
    then = sprintf("{ function = \"robust\", args = { cluster = \"studyid\", adjust = %s } }", v$adjust),
    expect = c(vector_expect("se", sqrt(diag(vc))), vector_expect("vb", as.vector(t(vc)))),
    pkg = "sandwich",
    citation = "Zeileis A, Koell S, Graham N (2020). Various versatile variances: an object-oriented implementation of clustered covariances in R. Journal of Statistical Software 95(1):1-36.",
    locator = sprintf("vcovCL(lm(yi ~ propmale, weights = 1 / vi), cluster = studyid, %s): sqrt(diag) and the matrix by rows", v$label)
  )
}

header <- c(
  "# Other-software cases for rma_mv and robust, written by src/tests/correctness/r/other_software_mv.R; do not edit by hand.",
  "#",
  "# mvmeta, nlme, robumeta and sandwich do not depend on metafor, so these values are independent of it. See the script for the conventions it pins.",
  ""
)
out <- file.path("src", "tests", "correctness", "cases", "other_mv.toml")
writeLines(c(header, cases), out)
cat("wrote", out, "\n")
