# Reference values for rma.mv() on dat.konstantopoulos2011, dat.assink2016, dat.berkey1998 and small synthetic data sets:
# nested/crossed ~ 1 | id terms, moderators, fixed variance components, ML/REML/FE/EE, test="t" with dfs="contain",
# sparse=TRUE, known correlation matrices (R=), the structures CS, HCS, UN, AR, HAR, CAR, ID, DIAG, SPEXP, GEN and GDIAG,
# two ~ inner | outer terms, custom weights, missing values and subsets; with printed output, hatvalues(), weights(),
# residuals() and profile-likelihood confidence intervals for the variance components.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_rma_mv.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}
frame_cols <- function(d) lapply(as.list(as.data.frame(d)), function(col) if (is.numeric(col)) num(col) else as.character(col))
warns <- function(expr) {
  w <- character()
  value <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = w)
}

core <- function(res, w = character(), extra = list()) {
  out <- list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), vb = mat(res$vb),
    sigma2 = num(res$sigma2), tau2 = num(res$tau2), rho = num(res$rho), gamma2 = num(res$gamma2), phi = num(res$phi),
    withS = res$withS, withG = res$withG, withH = res$withH,
    G = mat(res$G), H = mat(res$H),
    QE = res$QE, QEdf = res$QEdf, QEp = res$QEp, QM = res$QM, QMdf = num(res$QMdf), QMp = res$QMp,
    k = res$k, k_f = res$k.f, p = res$p, parms = res$parms, m = res$m,
    fit_ml = num(res$fit.stats$ML), fit_reml = num(res$fit.stats$REML),
    ddf = num(res$ddf), s2w = res$s2w, test = res$test, method = res$method,
    s_nlevels = num(res$s.nlevels), s_names = res$s.names, g_names = res$g.names, h_names = res$h.names,
    g_nlevels = num(res$g.nlevels), g_nlevels_f = num(res$g.nlevels.f), h_nlevels = num(res$h.nlevels), h_nlevels_f = num(res$h.nlevels.f),
    g_levels_k = num(res$g.levels.k), g_levels_comb_k = num(res$g.levels.comb.k), g_levels_f = res$g.levels.f[[1]],
    h_levels_k = num(res$h.levels.k), h_levels_comb_k = num(res$h.levels.comb.k), h_levels_f = res$h.levels.f[[1]],
    sigma2_fix = num(res$vc.fix$sigma2), tau2_fix = num(res$vc.fix$tau2), rho_fix = num(res$vc.fix$rho),
    gamma2_fix = num(res$vc.fix$gamma2), phi_fix = num(res$vc.fix$phi),
    struct = res$struct, Rfix = res$Rfix, allvipos = res$allvipos, int_only = res$int.only, int_incl = res$int.incl,
    coef_names = rownames(res$beta), btt = num(res$btt),
    print = printed(res), print_showfit = printed(res, showfit = TRUE),
    hat = num(hatvalues(res)), weights = num(weights(res)), residuals = num(residuals(res)), fitted = num(fitted(res)),
    M = mat(res$M), warnings = w
  )
  if (res$int.only) out$weights_rowsum <- num(weights(res, type = "rowsum"))
  c(out, extra)
}

fit <- function(expr, extra = function(res) list()) {
  tryCatch(
    {
      r <- warns(expr)
      core(r$value, r$warnings, extra(r$value))
    },
    error = function(e) list(error = conditionMessage(e))
  )
}

out <- list()

### Konstantopoulos (2011): schools within districts --------------------------------------------------------

kon <- dat.konstantopoulos2011
kon_cols <- frame_cols(kon)
out$kon_data <- kon_cols

out$kon_nested <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon))
out$kon_nested_ml <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, method = "ML"))
out$kon_list <- fit(rma.mv(yi, vi, random = list(~ 1 | district, ~ 1 | study), data = kon))
out$kon_mods <- fit(rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon))
out$kon_t <- fit(rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, test = "t"))
out$kon_contain <- fit(rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, test = "t", dfs = "contain"))
out$kon_dfs_num <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, dfs = 20))
out$kon_knha <- fit(rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, test = "knha"))
out$kon_adhoc <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, test = "adhoc"))
out$kon_fix_sigma2 <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, sigma2 = c(0.05, NA)))
out$kon_fix_both <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, sigma2 = c(0.05, 0.02)))
out$kon_fix_all <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, sigma2 = 0.03))
out$kon_sparse <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, sparse = TRUE))
out$kon_fe <- fit(rma.mv(yi, vi, data = kon, method = "FE"))
out$kon_ee_random_ignored <- fit(rma.mv(yi, vi, random = ~ 1 | district, data = kon, method = "EE"))
out$kon_subset <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, subset = district != 11))
out$kon_level90 <- fit(rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, level = 90))
out$kon_btt <- fit(rma.mv(yi, vi, mods = ~ I(year - 1990) + I((year - 1990)^2), random = ~ 1 | district/school, data = kon, btt = 3))
out$kon_formula <- fit(rma.mv(yi ~ I(year - 1990), vi, random = ~ 1 | district/school, data = kon))
out$kon_interaction <- fit(rma.mv(yi, vi, random = ~ 1 | interaction(district, school), data = kon))
w <- kon$vi^-1
out$kon_weights <- fit(rma.mv(yi, vi, W = w, random = ~ 1 | district/school, data = kon))
out$kon_spexp <- fit(rma.mv(yi, vi, random = ~ year | district, struct = "SPEXP", data = kon))

### Assink and Wibbelink (2016): effect sizes within studies ----------------------------------------------

ass <- dat.assink2016
out$ass_data <- frame_cols(ass)
out$ass_cs <- fit(rma.mv(yi, vi, random = list(~ 1 | study, ~ 1 | esid), data = ass))
out$ass_cs_ml <- fit(rma.mv(yi, vi, random = ~ 1 | study/esid, data = ass, method = "ML"))
out$ass_mods <- fit(rma.mv(yi, vi, mods = ~ factor(deltype) + year, random = ~ 1 | study/esid, data = ass))
out$ass_mods_t <- fit(rma.mv(yi, vi, mods = ~ factor(deltype) + year, random = ~ 1 | study/esid, data = ass, test = "t", dfs = "contain"))
V <- vcalc(vi, cluster = study, obs = esid, data = ass, rho = 0.6)
out$ass_V <- mat(V)
out$ass_vcalc <- fit(rma.mv(yi, V, random = ~ 1 | study/esid, data = ass))
out$ass_vcalc_sparse <- fit(rma.mv(yi, V, random = ~ 1 | study/esid, data = ass, sparse = TRUE))

### Berkey et al. (1998): two correlated outcomes -------------------------------------------------------------

ber <- dat.berkey1998
Vb <- bldiag(lapply(split(ber[, c("v1i", "v2i")], ber$trial), as.matrix))
out$ber_data <- frame_cols(ber)
out$ber_V <- mat(Vb)
bf <- function(...) fit(rma.mv(yi, Vb, mods = ~ 0 + outcome, data = ber, ...))
out$ber_un <- bf(random = ~ outcome | trial, struct = "UN")
out$ber_un_ml <- bf(random = ~ outcome | trial, struct = "UN", method = "ML")
out$ber_un_nochol <- bf(random = ~ outcome | trial, struct = "UN", control = list(cholesky = FALSE))
out$ber_un_t <- bf(random = ~ outcome | trial, struct = "UN", test = "t")
out$ber_un_mods <- fit(rma.mv(yi, Vb, mods = ~ 0 + outcome + I(year - 1985):outcome, random = ~ outcome | trial, struct = "UN", data = ber))
out$ber_cs <- bf(random = ~ outcome | trial, struct = "CS")
out$ber_hcs <- bf(random = ~ outcome | trial, struct = "HCS")
out$ber_ar <- bf(random = ~ outcome | trial, struct = "AR")
out$ber_har <- bf(random = ~ outcome | trial, struct = "HAR")
out$ber_diag <- bf(random = ~ outcome | trial, struct = "DIAG")
out$ber_id <- bf(random = ~ outcome | trial, struct = "ID")
out$ber_un_fixrho <- bf(random = ~ outcome | trial, struct = "UN", rho = 0.5)
out$ber_un_fixtau2 <- bf(random = ~ outcome | trial, struct = "UN", tau2 = c(NA, 0.01))
out$ber_hcs_fixrho <- bf(random = ~ outcome | trial, struct = "HCS", rho = 0)
out$ber_cs_sparse <- bf(random = ~ outcome | trial, struct = "CS", sparse = TRUE)
out$ber_un_sparse <- bf(random = ~ outcome | trial, struct = "UN", sparse = TRUE)
out$ber_gen <- fit(rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ 0 + outcome | trial, struct = "GEN", data = ber))
out$ber_gdiag <- fit(rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ 0 + outcome || trial, struct = "GEN", data = ber))
out$ber_car <- fit(rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ as.numeric(outcome == "AL") | trial, struct = "CAR", data = ber))

### synthetic data ------------------------------------------------------------------------------------------------

set.seed(20261013)
ns <- 14
sim <- do.call(rbind, lapply(seq_len(ns), function(s) {
  nout <- sample(2:4, 1)
  data.frame(study = s, group = sample(c("a", "b", "c"), 1), species = paste0("sp", sample(1:8, 1)), outcome = sort(sample(1:4, nout)), stringsAsFactors = FALSE)
}))
sim$id <- seq_len(nrow(sim))
sim$x1 <- round(rnorm(nrow(sim)), 3)
sim$x2 <- round(runif(nrow(sim), 0, 2), 3)
sim$time <- sim$outcome * 1.5 + round(runif(nrow(sim), 0, 0.4), 2)
sim$vi <- round(runif(nrow(sim), 0.02, 0.08), 4)
ustudy <- rnorm(ns, 0, 0.3)
sim$yi <- round(0.2 + 0.3 * sim$x1 + ustudy[sim$study] + rnorm(nrow(sim), 0, 0.15) + rnorm(nrow(sim), 0, sqrt(sim$vi)), 4)
sim$outcome <- factor(sim$outcome, labels = paste0("o", 1:4)[sort(unique(sim$outcome))])
sim$outcome <- as.character(sim$outcome)
out$sim_data <- frame_cols(sim)
Rsp <- cov2cor(crossprod(matrix(rnorm(64), 8)))
dimnames(Rsp) <- list(paste0("sp", 1:8), paste0("sp", 1:8))
out$sim_R <- mat(Rsp)
out$sim_R_names <- rownames(Rsp)
sim_V <- diag(sim$vi)
for (s in unique(sim$study)) {
  i <- which(sim$study == s)
  sim_V[i, i] <- 0.4 * outer(sqrt(sim$vi[i]), sqrt(sim$vi[i]))
  diag(sim_V)[i] <- sim$vi[i]
}
out$sim_V <- mat(sim_V)

out$sim_cs <- fit(rma.mv(yi, sim_V, random = ~ outcome | study, struct = "CS", data = sim))
out$sim_un <- fit(rma.mv(yi, sim_V, random = ~ outcome | study, struct = "UN", data = sim))
out$sim_hcs <- fit(rma.mv(yi, sim_V, random = ~ outcome | study, struct = "HCS", data = sim))
out$sim_car <- fit(rma.mv(yi, sim_V, random = ~ time | study, struct = "CAR", data = sim))
out$sim_spgau <- fit(rma.mv(yi, sim_V, random = ~ time | study, struct = "SPGAU", data = sim))
out$sim_splin <- fit(rma.mv(yi, sim_V, random = ~ time | study, struct = "SPLIN", data = sim))
out$sim_gen <- fit(rma.mv(yi, sim_V, mods = ~ x1, random = ~ x1 + x2 | study, struct = "GEN", data = sim))
out$sim_gdiag <- fit(rma.mv(yi, sim_V, mods = ~ x1, random = ~ x1 + x2 || study, struct = "GEN", data = sim))
out$sim_s_and_g <- fit(rma.mv(yi, sim_V, random = list(~ 1 | id, ~ outcome | study), struct = "CS", data = sim))
out$sim_two_inner_outer <- fit(rma.mv(yi, sim_V, random = list(~ outcome | study, ~ outcome | group), struct = c("CS", "HCS"), data = sim))
out$sim_R_known <- fit(rma.mv(yi, sim_V, random = list(~ 1 | species, ~ 1 | study), R = list(species = Rsp), data = sim))
out$sim_R_cov0 <- fit(rma.mv(yi, sim_V, random = list(~ 1 | species, ~ 1 | study), R = list(species = Rsp), Rscale = "cor0", data = sim))
out$sim_R_none <- fit(rma.mv(yi, sim_V, random = list(~ 1 | species, ~ 1 | study), R = list(species = Rsp), Rscale = "none", data = sim))
out$sim_R_sparse <- fit(rma.mv(yi, sim_V, random = list(~ 1 | species, ~ 1 | study), R = list(species = Rsp), data = sim, sparse = TRUE))
sim_na <- sim
sim_na$yi[3] <- NA
sim_na$x1[10] <- NA
out$sim_na <- fit(rma.mv(yi, sim_V, mods = ~ x1, random = ~ 1 | study, data = sim_na))
out$sim_na_un <- fit(rma.mv(yi, sim_V, random = ~ outcome | study, struct = "UN", data = sim_na))
out$sim_subset <- fit(rma.mv(yi, sim_V, mods = ~ x1 + x2, random = ~ 1 | study, data = sim, subset = x2 > 0.3))

### profile-likelihood confidence intervals for the variance components -------------------------------------------------

ci_list <- function(res, ...) {
  r <- confint(res, ...)
  if (inherits(r, "list.confint.rma")) {
    return(lapply(r, function(x) list(random = mat(x$random), fixed = mat(x$fixed), digits = NULL)))
  }
  list(random = mat(r$random), fixed = mat(r$fixed))
}
cires <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon)
out$ci_kon_sigma2_1 <- ci_list(cires, sigma2 = 1)
out$ci_kon_sigma2_2 <- ci_list(cires, sigma2 = 2)
out$ci_kon_fixed <- ci_list(cires, fixed = TRUE)
out$ci_kon_level90 <- ci_list(cires, sigma2 = 1, level = 90)
berres <- rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ outcome | trial, struct = "UN", data = ber)
out$ci_ber_tau2_1 <- ci_list(berres, tau2 = 1)
out$ci_ber_rho <- ci_list(berres, rho = 1)
cs_res <- rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ outcome | trial, struct = "CS", data = ber)
out$ci_ber_cs_tau2 <- ci_list(cs_res, tau2 = 1)
out$ci_ber_cs_rho <- ci_list(cs_res, rho = 1)
assres <- rma.mv(yi, vi, random = list(~ 1 | study, ~ 1 | esid), data = ass)
out$ci_ass_sigma2_1 <- ci_list(assres, sigma2 = 1)
out$ci_ass_sigma2_2 <- ci_list(assres, sigma2 = 2)

### predict, anova, fitstats ---------------------------------------------------------------------------------------------

pm <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon)
pr <- predict(pm, newmods = c(-5, 0, 5) )
out$predict_kon <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
pr <- predict(pm)
out$predict_kon_fitted <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub))
pt <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, test = "t")
pr <- predict(pt, newmods = c(-5, 0, 5))
out$predict_kon_t <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
m0 <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, method = "ML")
m1 <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon, method = "ML")
a <- anova(m0, m1)
out$anova_lrt <- list(LRT = a$LRT, pval = a$pval, fit_stats_f = num(unlist(a$fit.stats.f)), fit_stats_r = num(unlist(a$fit.stats.r)), parms = c(a$parms.f, a$parms.r), print = printed(a))
a <- anova(pm, btt = 2)
out$anova_wald <- list(QM = a$QM, QMp = a$QMp, print = printed(a))
fs <- fitstats(pm)
out$fitstats <- num(fs)

out$meta <- list(metafor = as.character(packageVersion("metafor")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase4", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase4/rma_mv.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase4/rma_mv.json\n")
