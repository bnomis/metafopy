# Reference values for robust() on rma.uni and rma.mv fits (dat.konstantopoulos2011, dat.assink2016): the default cluster-robust
# estimator (CR0/CR1/Stata adjustments, residual dfs) and clubSandwich (CR0, CR1, CR1p, CR1S, CR2, CR3 with z, naive-t, naive-tp and
# Satterthwaite tests and the chi-sq, Naive-F, Naive-Fp and HTZ Wald tests), with printed output.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_robust.R

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
    vbest = r$vbest, robumethod = r$robumethod, test = r$test, print = printed(r)
  )
}
fit <- function(expr) tryCatch(summ(expr), error = function(e) list(error = conditionMessage(e)))

out <- list()
kon <- dat.konstantopoulos2011

uni0 <- rma(yi, vi, data = kon)
uni1 <- rma(yi, vi, mods = ~ I(year - 1990), data = kon)
uni2 <- rma(yi, vi, mods = ~ I(year - 1990), data = kon, method = "FE")
uni3 <- rma(yi, vi, mods = ~ I(year - 1990), data = kon, weights = 1 / sqrt(vi))
uni4 <- rma(yi, vi, mods = ~ I(year - 1990), data = kon, subset = district != 11)
out$uni_int <- fit(robust(uni0, cluster = district))
out$uni_mods <- fit(robust(uni1, cluster = district))
out$uni_mods_cr0 <- fit(robust(uni1, cluster = district, adjust = FALSE))
out$uni_mods_stata1 <- fit(robust(uni1, cluster = district, adjust = "Stata1"))
out$uni_mods_stata2 <- fit(robust(uni1, cluster = district, adjust = "Stata2"))
out$uni_fe <- fit(robust(uni2, cluster = district))
out$uni_weights <- fit(robust(uni3, cluster = district))
out$uni_subset <- fit(robust(uni4, cluster = district))
for (type in c("CR0", "CR1", "CR1p", "CR1S", "CR2", "CR3")) {
  out[[paste0("uni_cs_int_", type)]] <- fit(robust(uni0, cluster = district, clubSandwich = TRUE, vcov = type))
  out[[paste0("uni_cs_mods_", type)]] <- fit(robust(uni1, cluster = district, clubSandwich = TRUE, vcov = type))
}
out$uni_cs_fe <- fit(robust(uni2, cluster = district, clubSandwich = TRUE))
out$uni_cs_weights <- fit(robust(uni3, cluster = district, clubSandwich = TRUE))
out$uni_cs_subset <- fit(robust(uni4, cluster = district, clubSandwich = TRUE))
for (ct in c("z", "naive-t", "naive-tp")) {
  out[[paste0("uni_cs_coef_", ct)]] <- fit(robust(uni1, cluster = district, clubSandwich = TRUE, coef_test = ct))
}
for (wt in c("chi-sq", "Naive-F", "Naive-Fp")) {
  out[[paste0("uni_cs_wald_", wt)]] <- fit(robust(uni1, cluster = district, clubSandwich = TRUE, wald_test = wt))
}
out$uni_cs_conf <- fit(robust(uni1, cluster = district, clubSandwich = TRUE, conf_test = "z"))
uni5 <- rma(yi, vi, mods = ~ I(year - 1990) + I((year - 1990)^2), data = kon)
out$uni_cs_btt <- fit(robust(uni5, cluster = district, clubSandwich = TRUE))
out$uni_cs_level90 <- fit(robust(rma(yi, vi, mods = ~ I(year - 1990), data = kon, level = 90), cluster = district, clubSandwich = TRUE))

mv0 <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon)
mv1 <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon)
mv2 <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, W = 1 / vi)
mv3 <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, sparse = TRUE)
mv4 <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, subset = district != 11)
out$mv_int <- fit(robust(mv0, cluster = district))
out$mv_mods <- fit(robust(mv1, cluster = district))
out$mv_mods_cr0 <- fit(robust(mv1, cluster = district, adjust = FALSE))
out$mv_mods_stata1 <- fit(robust(mv1, cluster = district, adjust = "Stata1"))
out$mv_mods_stata2 <- fit(robust(mv1, cluster = district, adjust = "Stata2"))
out$mv_weights <- fit(robust(mv2, cluster = district))
out$mv_sparse <- fit(robust(mv3, cluster = district))
out$mv_subset <- fit(robust(mv4, cluster = district))
out$mv_school <- fit(robust(mv1, cluster = school))
for (type in c("CR0", "CR1", "CR1p", "CR1S", "CR2", "CR3")) {
  out[[paste0("mv_cs_int_", type)]] <- fit(robust(mv0, cluster = district, clubSandwich = TRUE, vcov = type))
  out[[paste0("mv_cs_mods_", type)]] <- fit(robust(mv1, cluster = district, clubSandwich = TRUE, vcov = type))
}
out$mv_cs_weights <- fit(robust(mv2, cluster = district, clubSandwich = TRUE))
out$mv_cs_subset <- fit(robust(mv4, cluster = district, clubSandwich = TRUE))
for (ct in c("z", "naive-t", "naive-tp")) {
  out[[paste0("mv_cs_coef_", ct)]] <- fit(robust(mv1, cluster = district, clubSandwich = TRUE, coef_test = ct))
}
for (wt in c("chi-sq", "Naive-F", "Naive-Fp")) {
  out[[paste0("mv_cs_wald_", wt)]] <- fit(robust(mv1, cluster = district, clubSandwich = TRUE, wald_test = wt))
}

ass <- dat.assink2016
mva <- rma.mv(yi, vi, mods = ~ year, random = list(~ 1 | study, ~ 1 | esid), data = ass)
out$ass_default <- fit(robust(mva, cluster = study))
out$ass_cs <- fit(robust(mva, cluster = study, clubSandwich = TRUE))
out$ass_cs_cr1 <- fit(robust(mva, cluster = study, clubSandwich = TRUE, vcov = "CR1"))

# robust fits feed predict()
rob <- robust(uni1, cluster = district)
pr <- predict(rob, newmods = c(-5, 0, 5))
out$predict_uni <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
rob <- robust(mv1, cluster = district)
pr <- predict(rob, newmods = c(-5, 0, 5))
out$predict_mv <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))

out$meta <- list(metafor = as.character(packageVersion("metafor")), clubSandwich = as.character(packageVersion("clubSandwich")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase4", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase4/robust.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
