# Reference values for the rma.mv() arguments optbeta (with beta, lambda1 and lambda2), vccon and cvvc on
# dat.konstantopoulos2011, dat.assink2016 and dat.berkey1998: the estimates, their covariance matrix, the number of parameters,
# and the Hessian of the log-likelihood (numDeriv, r = 8) and its inverse for the variance components.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_rma_mv_extras.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x) || (length(x) == 1 && is.na(x))) {
    return(NULL)
  }
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}
warns <- function(expr) {
  w <- character()
  value <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = w)
}

core <- function(res, w = character()) {
  list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), vb = mat(res$vb),
    sigma2 = num(res$sigma2), tau2 = num(res$tau2), rho = num(res$rho), gamma2 = num(res$gamma2), phi = num(res$phi),
    QE = res$QE, QM = res$QM, QMp = res$QMp, k = res$k, p = res$p, parms = res$parms, optbeta = res$optbeta,
    fit_ml = num(res$fit.stats$ML), fit_reml = num(res$fit.stats$REML),
    hessian = mat(res$hessian), vvc = mat(res$vvc), hessian_names = colnames(res$hessian), vvc_names = colnames(res$vvc),
    warnings = w
  )
}

fit <- function(expr) {
  tryCatch(
    {
      r <- warns(expr)
      core(r$value, r$warnings)
    },
    error = function(e) list(error = conditionMessage(e))
  )
}

out <- list()

kon <- dat.konstantopoulos2011
ass <- dat.assink2016
ber <- dat.berkey1998
Vb <- bldiag(lapply(split(ber[, c("v1i", "v2i")], ber$trial), as.matrix))
out$ber_V <- mat(Vb)

nested <- ~ 1 | district / school
kr <- function(...) rma.mv(yi, vi, random = nested, data = kon, ...)
ar <- function(...) rma.mv(yi, vi, random = list(~ 1 | study, ~ 1 | esid), data = ass, ...)
br <- function(...) rma.mv(yi, Vb, mods = ~ 0 + outcome, random = ~ outcome | trial, data = ber, ...)

### vccon -----------------------------------------------------------------------------------------------------------------

out$kon_vccon_sigma2 <- fit(kr(vccon = list(sigma2 = c(1, 1))))
out$kon_vccon_sigma2_ml <- fit(kr(vccon = list(sigma2 = c(1, 1)), method = "ML"))
out$kon_vccon_mods <- fit(kr(mods = ~ I(year - 1990), vccon = list(sigma2 = c(1, 1))))
out$kon_vccon_distinct <- fit(kr(vccon = list(sigma2 = c(1, 2))))
out$kon_vccon_fixed <- fit(kr(sigma2 = c(0.04, 0.04), vccon = list(sigma2 = c(1, 1))))
out$kon_vccon_fixed_mixed <- fit(kr(sigma2 = c(0.04, NA), vccon = list(sigma2 = c(1, 1))))
out$ass_vccon_sigma2 <- fit(ar(vccon = list(sigma2 = c(1, 1))))
out$ass_vccon_sigma2_ml <- fit(ar(vccon = list(sigma2 = c(1, 1)), method = "ML"))
out$ber_un_vccon_tau2 <- fit(br(struct = "UN", vccon = list(tau2 = c(1, 1))))
out$ber_un_vccon_tau2_nochol <- fit(br(struct = "UN", vccon = list(tau2 = c(1, 1)), control = list(cholesky = FALSE)))
out$ber_hcs_vccon_tau2 <- fit(br(struct = "HCS", vccon = list(tau2 = c(1, 1))))
out$ber_un_vccon_sigma2_only <- fit(br(struct = "UN", vccon = list(sigma2 = 1)))

### optbeta, beta, lambda1 and lambda2 -----------------------------------------------------------------------------------------

out$kon_optbeta <- fit(kr(optbeta = TRUE))
out$kon_optbeta_ml <- fit(kr(optbeta = TRUE, method = "ML"))
out$kon_optbeta_mods <- fit(kr(mods = ~ I(year - 1990), optbeta = TRUE))
out$kon_optbeta_mods_t <- fit(kr(mods = ~ I(year - 1990), optbeta = TRUE, test = "t"))
out$kon_optbeta_fixbeta <- fit(kr(mods = ~ I(year - 1990), optbeta = TRUE, beta = c(NA, 0.01)))
out$kon_fixbeta <- fit(kr(mods = ~ I(year - 1990), beta = c(NA, 0.01)))
out$kon_fixbeta_ml <- fit(kr(mods = ~ I(year - 1990), beta = c(0.1, NA), method = "ML"))
out$kon_ridge <- fit(kr(mods = ~ I(year - 1990), lambda2 = 5))
out$kon_ridge_ml <- fit(kr(mods = ~ I(year - 1990), lambda2 = 5, method = "ML"))
out$kon_ridge_big <- fit(kr(mods = ~ I(year - 1990) + I((year - 1990)^2), lambda2 = 50))
out$kon_lasso <- fit(kr(mods = ~ I(year - 1990), lambda1 = 0.05))
out$ass_optbeta <- fit(ar(mods = ~ factor(deltype) + year, optbeta = TRUE))
out$ass_ridge <- fit(ar(mods = ~ factor(deltype) + year, lambda2 = 2))
out$ber_un_optbeta <- fit(br(struct = "UN", optbeta = TRUE))
out$ber_cs_optbeta <- fit(br(struct = "CS", optbeta = TRUE))
out$ber_un_optbeta_ml <- fit(br(struct = "UN", optbeta = TRUE, method = "ML"))
out$kon_fe_optbeta <- fit(rma.mv(yi, vi, data = kon, method = "FE", mods = ~ I(year - 1990), optbeta = TRUE))

### cvvc ----------------------------------------------------------------------------------------------------------------

out$kon_cvvc_true <- fit(kr(cvvc = TRUE))
out$kon_cvvc_varcor <- fit(kr(cvvc = "varcor"))
out$kon_cvvc_transf <- fit(kr(cvvc = "transf"))
out$kon_cvvc_varcov <- fit(kr(cvvc = "varcov"))
out$kon_cvvc_ml <- fit(kr(cvvc = TRUE, method = "ML"))
out$kon_cvvc_mods <- fit(kr(mods = ~ I(year - 1990), cvvc = TRUE))
out$kon_cvvc_fixed <- fit(kr(sigma2 = c(NA, 0.02), cvvc = TRUE))
out$kon_cvvc_fixed_varcov <- fit(kr(sigma2 = c(NA, 0.02), cvvc = "varcov"))
out$kon_cvvc_vccon <- fit(kr(vccon = list(sigma2 = c(1, 2)), cvvc = TRUE))
out$ass_cvvc_true <- fit(ar(cvvc = TRUE))
out$ass_cvvc_transf <- fit(ar(cvvc = "transf"))
out$ass_cvvc_ml <- fit(ar(cvvc = TRUE, method = "ML"))
out$ber_un_cvvc_true <- fit(br(struct = "UN", cvvc = TRUE))
out$ber_un_cvvc_transf <- fit(br(struct = "UN", cvvc = "transf"))
out$ber_un_cvvc_varcov <- fit(br(struct = "UN", cvvc = "varcov"))
out$ber_un_nochol_cvvc_transf <- fit(br(struct = "UN", cvvc = "transf", control = list(cholesky = FALSE)))
out$ber_un_nochol_cvvc_varcor <- fit(br(struct = "UN", cvvc = "varcor", control = list(cholesky = FALSE)))
out$ber_cs_cvvc_true <- fit(br(struct = "CS", cvvc = TRUE))
out$ber_cs_cvvc_varcov <- fit(br(struct = "CS", cvvc = "varcov"))
out$ber_hcs_cvvc_true <- fit(br(struct = "HCS", cvvc = TRUE))
out$ber_hcs_cvvc_transf <- fit(br(struct = "HCS", cvvc = "transf"))
out$ber_un_fixrho_cvvc <- fit(br(struct = "UN", rho = 0.5, cvvc = TRUE))
out$ber_ar_cvvc_true <- fit(br(struct = "AR", cvvc = TRUE))
out$ber_diag_cvvc_true <- fit(br(struct = "DIAG", cvvc = TRUE))

out$meta <- list(metafor = as.character(packageVersion("metafor")), numDeriv = as.character(packageVersion("numDeriv")), R = R.version.string)
dir.create("src/tests/reference/extras", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/extras/rma_mv_extras.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/extras/rma_mv_extras.json\n")
