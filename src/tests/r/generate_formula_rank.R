# Reference fits for formulas whose model matrix is rank-deficient (`~ f:g` with neither main effect): metafor drops the redundant column and
# warns. Records the coefficient names, estimates, standard errors and the warning for rma(), rma.mv() and a location-scale rma(), so that
# metafopy's handling is checked against R rather than against values written by hand.
# Writes src/tests/reference/formula_rank.json.
#
# Usage (from the repository root):
#   LC_ALL=C.UTF-8 R_LIBS=.audit/Rlib Rscript src/tests/r/generate_formula_rank.R

suppressPackageStartupMessages({
  library(jsonlite)
  library(metafor)
})

options(useFancyQuotes = FALSE)

# 12 rows: deltype three levels, outcome two; literal values, not R's RNG
dat <- data.frame(
  yi = c(0.31, -0.12, 0.45, -0.52, 0.18, 0.77, -0.25, 0.09, -0.61, 0.38, 0.52, -0.33),
  vi = c(0.04, 0.06, 0.05, 0.08, 0.03, 0.07, 0.05, 0.04, 0.09, 0.06, 0.03, 0.05),
  year = c(1980, 1985, 1990, 1995, 2000, 2005, 1982, 1988, 1992, 1998, 2002, 2007),
  deltype = rep(c("covert", "general", "overt"), 4),
  outcome = rep(c("AL", "AL", "AL", "PD", "PD", "PD"), 2),
  id = 1:12,
  stringsAsFactors = FALSE
)

# 24 rows for the location-scale fit, which needs more rows to converge
dat_ls <- data.frame(
  yi = c(0.31, -0.12, 0.45, -0.52, 0.18, 0.77, -0.25, 0.09, -0.61, 0.38, 0.52, -0.33,
         0.22, 0.64, -0.41, 0.05, 0.93, -0.18, 0.27, -0.7, 0.11, 0.48, -0.05, 0.36),
  vi = c(0.04, 0.06, 0.05, 0.08, 0.03, 0.07, 0.05, 0.04, 0.09, 0.06, 0.03, 0.05,
         0.07, 0.03, 0.06, 0.05, 0.1, 0.04, 0.08, 0.06, 0.03, 0.07, 0.05, 0.09),
  year = c(1980, 1985, 1990, 1995, 2000, 2005, 1982, 1988, 1992, 1998, 2002, 2007,
           1981, 1986, 1991, 1996, 2001, 2006, 1983, 1989, 1993, 1999, 2003, 2008),
  deltype = rep(c("covert", "general", "overt"), 8),
  outcome = rep(c("AL", "AL", "AL", "PD", "PD", "PD"), 4),
  stringsAsFactors = FALSE
)

# run a fit, keeping the first warning's text
capture <- function(expr) {
  warning_text <- NULL
  fit <- withCallingHandlers(expr, warning = function(w) {
    if (is.null(warning_text)) warning_text <<- conditionMessage(w)
    invokeRestart("muffleWarning")
  })
  list(fit = fit, warning = warning_text)
}

summarise <- function(res, data_name) {
  fit <- res$fit
  list(
    data = data_name,
    names = rownames(fit$beta),
    beta = as.numeric(fit$beta),
    se = as.numeric(fit$se),
    warning = res$warning
  )
}

out <- list()
out$uni <- summarise(capture(rma(yi, vi, mods = ~ deltype:outcome, data = dat, method = "FE")), "data")
out$mv <- summarise(capture(rma.mv(yi, vi, mods = ~ deltype:outcome, random = ~ 1 | id, data = dat)), "data")

ls_res <- capture(rma(yi, vi, mods = ~ year, scale = ~ deltype:outcome, data = dat_ls))
# rma() stops with an error when the optimiser does not converge, so reaching this line means it did
out$ls <- summarise(ls_res, "data_ls")
out$ls$alpha_names <- rownames(ls_res$fit$alpha)
out$ls$alpha <- as.numeric(ls_res$fit$alpha)
out$ls$alpha_se <- as.numeric(ls_res$fit$se.alpha)

out$data <- as.list(dat)
out$data_ls <- as.list(dat_ls)
out$meta <- list(R = R.version.string, metafor = as.character(packageVersion("metafor")))
dir.create("src/tests/reference", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/formula_rank.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/formula_rank.json\n")
