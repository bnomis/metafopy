# Generate JSON reference fixtures from R metafor.
#
# Usage (from the repository root):  Rscript src/tests/r/generate_fixtures.R
#
# Writes one src/tests/reference/<case>.json per case in cases.R holding the
# escalc() output and rma.uni() results for every estimator in `methods`,
# with z and Knapp-Hartung tests, with and without moderators.

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})
source("src/tests/r/cases.R")

out_dir <- "src/tests/reference"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

fit_summary <- function(res) {
  out <- list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), vb = unname(res$vb),
    coef_names = rownames(res$beta),
    tau2 = res$tau2, se_tau2 = num(res$se.tau2),
    k = res$k, p = res$p, dfs = num(res$ddf),
    QE = res$QE, QEp = res$QEp, QM = res$QM, QMp = res$QMp,
    I2 = num(res$I2), H2 = num(res$H2), R2 = num(res$R2),
    fit_stats_ml = num(fitstats(res, REML = FALSE)),
    fit_stats_reml = num(fitstats(res, REML = TRUE)),
    weights = num(weights(res)), hat = num(hatvalues(res))
  )
  if (res$p == 1) {
    pr <- predict(res)
    out$predict <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb),
                        ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
  }
  if (!res$method %in% c("EE", "FE", "CE")) {
    ci <- tryCatch(confint(res), error = function(e) NULL)
    if (!is.null(ci)) {
      out$confint_random <- list(estimate = num(ci$random[, "estimate"]),
                                 ci_lb = num(ci$random[, "ci.lb"]),
                                 ci_ub = num(ci$random[, "ci.ub"]),
                                 rows = rownames(ci$random))
    }
  }
  out
}

try_fit <- function(args) {
  # do.call() passes evaluated vectors, sidestepping rma()'s non-standard evaluation
  tryCatch(fit_summary(do.call(rma, args)), error = function(e) list(error = conditionMessage(e)))
}

for (case_name in names(cases)) {
  case <- cases[[case_name]]
  dat <- get(case$data)
  if (!is.null(case$escalc)) {
    args <- c(case$escalc, list(data = dat))
    # escalc() takes column names unquoted; pass the columns themselves
    for (a in setdiff(names(case$escalc), "measure")) args[[a]] <- dat[[case$escalc[[a]]]]
    args$data <- NULL
    es <- do.call(escalc, args)
    dat$yi <- as.vector(es$yi)
    dat$vi <- as.vector(es$vi)
  }

  fits <- list()
  for (m in methods) {
    for (test in c("z", "knha")) {
      if (test == "knha" && m %in% c("EE")) next
      fits[[length(fits) + 1]] <- list(method = m, test = test, mods = NULL,
                                       result = try_fit(list(yi = dat$yi, vi = dat$vi, method = m, test = test)))
      if (!is.null(case$mods)) {
        f <- as.formula(case$mods)
        fits[[length(fits) + 1]] <- list(method = m, test = test, mods = case$mods,
                                         result = try_fit(list(yi = dat$yi, vi = dat$vi, mods = f, data = dat, method = m, test = test)))
      }
    }
  }

  fixture <- list(
    case = case_name,
    dataset = case$data,
    escalc = case$escalc,
    yi = dat$yi, vi = dat$vi,
    fits = fits,
    meta = list(metafor = as.character(packageVersion("metafor")),
                metadat = as.character(packageVersion("metadat")),
                R = R.version.string)
  )
  write_json(fixture, file.path(out_dir, paste0(case_name, ".json")),
             auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
  n_err <- sum(vapply(fits, function(f) !is.null(f$result$error), logical(1)))
  cat(sprintf("%-22s %3d fits (%d errors)\n", case_name, length(fits), n_err))
}
