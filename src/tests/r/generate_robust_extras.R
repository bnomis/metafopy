# Reference values for the clubSandwich options of robust() that generate_robust.R does not cover: coef_test = "saddlepoint" (with the
# default and an explicit conf_test) and the Wald tests HTA, HTB, EDF and EDT (besides the others) for rma.uni and rma.mv fits with two or
# three moderators, including btt subsets and other variance estimators (dat.konstantopoulos2011).
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_robust_extras.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(clubSandwich)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}

summ <- function(r) {
  list(
    beta = num(r$beta), se = num(r$se), zval = num(r$zval), pval = num(r$pval), ci_lb = num(r$ci.lb), ci_ub = num(r$ci.ub),
    vb = mat(r$vb), QM = r$QM, QMdf = num(r$QMdf), QMp = r$QMp, ddf = num(r$ddf), n = r$n, tcl = num(r$tcl),
    vbest = r$vbest, robumethod = r$robumethod, test = r$test, coef_test = r$coef_test, conf_test = r$conf_test, wald_test = r$wald_test,
    dfs = num(r$dfs), print = printed(r)
  )
}
fit <- function(expr) {
  warn <- character()
  res <- withCallingHandlers(
    tryCatch(summ(expr), error = function(e) list(error = conditionMessage(e))),
    warning = function(w) {
      warn <<- c(warn, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  res$warnings <- warn
  res
}

out <- list()
kon <- dat.konstantopoulos2011
kon$yr <- kon$year - 1990
kon$yr2 <- kon$yr^2
kon$yr3 <- kon$yr^3

uni0 <- rma(yi, vi, data = kon)
uni1 <- rma(yi, vi, mods = ~yr, data = kon)
uni2 <- rma(yi, vi, mods = ~ yr + yr2, data = kon)
uni3 <- rma(yi, vi, mods = ~ yr + yr2 + yr3, data = kon)
uni3b <- rma(yi, vi, mods = ~ yr + yr2 + yr3, data = kon, btt = 2:3)
uni3c <- rma(yi, vi, mods = ~ yr + yr2 + yr3, data = kon, btt = c(2, 4))
uni4 <- rma(yi, vi, mods = ~ yr + yr2, data = kon, weights = 1 / sqrt(vi))
uni5 <- rma(yi, vi, mods = ~ yr + yr2, data = kon, method = "FE")
mv0 <- rma.mv(yi, vi, random = ~ 1 | district / school, data = kon)
mv1 <- rma.mv(yi, vi, mods = ~yr, random = ~ 1 | district / school, data = kon)
mv2 <- rma.mv(yi, vi, mods = ~ yr + yr2, random = ~ 1 | district / school, data = kon)
mv3 <- rma.mv(yi, vi, mods = ~ yr + yr2 + yr3, random = ~ 1 | district / school, data = kon, btt = 2:3)
mv4 <- rma.mv(yi, vi, mods = ~ yr + yr2, random = ~ 1 | district / school, data = kon, W = 1 / vi)

models <- list(uni_int = uni0, uni_mods = uni1, uni_two = uni2, uni_three = uni3, uni_btt23 = uni3b, uni_btt24 = uni3c, uni_weights = uni4, uni_fe = uni5,
               mv_int = mv0, mv_mods = mv1, mv_two = mv2, mv_btt23 = mv3, mv_weights = mv4)

for (nm in names(models)) {
  m <- models[[nm]]
  out[[paste0(nm, "_saddle")]] <- fit(robust(m, cluster = district, clubSandwich = TRUE, coef_test = "saddlepoint"))
  out[[paste0(nm, "_saddle_conf_z")]] <- fit(robust(m, cluster = district, clubSandwich = TRUE, coef_test = "saddlepoint", conf_test = "naive-t"))
  for (wt in c("chi-sq", "Naive-F", "Naive-Fp", "HTA", "HTB", "HTZ", "EDF", "EDT")) {
    out[[paste0(nm, "_wald_", wt)]] <- fit(robust(m, cluster = district, clubSandwich = TRUE, wald_test = wt))
  }
}
for (type in c("CR0", "CR1", "CR1p", "CR1S", "CR3")) {
  out[[paste0("uni_two_saddle_", type)]] <- fit(robust(uni2, cluster = district, clubSandwich = TRUE, vcov = type, coef_test = "saddlepoint"))
  out[[paste0("mv_two_saddle_", type)]] <- fit(robust(mv2, cluster = district, clubSandwich = TRUE, vcov = type, coef_test = "saddlepoint"))
  for (wt in c("HTA", "HTB", "EDF", "EDT")) {
    out[[paste0("uni_two_", type, "_", wt)]] <- fit(robust(uni2, cluster = district, clubSandwich = TRUE, vcov = type, wald_test = wt))
  }
}
out$uni_bad_conf <- fit(robust(uni1, cluster = district, clubSandwich = TRUE, coef_test = "saddlepoint", conf_test = "saddlepoint"))

out$meta <- list(metafor = as.character(packageVersion("metafor")), clubSandwich = as.character(packageVersion("clubSandwich")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase6", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase6/robust_extras.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
