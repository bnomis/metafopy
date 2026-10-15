# Reference values for the 'link' and 'family' arguments of rma.glmm() (passed through ...) on the bcg, collins1985a, nielweise2007 and hart1999 data:
# the probit, cauchit, cloglog, log, identity and sqrt links for the unconditional models (UM.FS, UM.RS) of odds ratios and incidence rate
# ratios and for the one-group measures, with the fixed-effects (glm) and mixed-effects (lme4::glmer) fits, predict() and the printed output.
# lme4 often stops early with non-canonical links; its warnings are stored under "warnings" and errors under "error".
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_glmm_links.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

core <- function(expr) {
  warns <- character()
  res <- tryCatch(withCallingHandlers(expr, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
                  error = function(e) conditionMessage(e))
  if (is.character(res)) return(list(error = res, warnings = I(warns)))
  list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval), ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub),
    vb = num(res$vb), coef_names = rownames(res$beta),
    tau2 = res$tau2, sigma2 = res$sigma2, I2 = res$I2, H2 = res$H2, vt = res$vt,
    QE_Wld = res$QE.Wld, QEp_Wld = res$QEp.Wld, QE_LRT = res$QE.LRT, QEp_LRT = res$QEp.LRT, QE_df = res$QE.df,
    QM = res$QM, QMp = res$QMp, QMdf = num(res$QMdf),
    k = res$k, k_f = res$k.f, k_yi = res$k.yi, k_eff = res$k.eff, p = res$p, p_eff = res$p.eff, parms = res$parms,
    int_only = res$int.only, method = res$method, model = res$model,
    fit_ml = num(res$fit.stats$ML),
    warnings = I(warns),
    print = printed(res), print_showfit = printed(res, showfit = TRUE)
  )
}

pred <- function(res, ...) {
  out <- predict(res, ...)
  keep <- c("pred", "se", "ci.lb", "ci.ub", "pi.lb", "pi.ub")
  r <- lapply(keep, function(n) if (is.null(out[[n]])) NULL else num(out[[n]]))
  names(r) <- gsub("\\.", "_", keep)
  r$print <- printed(out)
  r
}

bcg <- dat.bcg
col <- dat.collins1985a
nw <- dat.nielweise2007
hart <- dat.hart1999
b2 <- function(measure = "OR", ...) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = measure, ...)
c2 <- function(...) rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", ...)
n2 <- function(...) rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "OR", ...)
h2 <- function(...) rma.glmm(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = hart, measure = "IRR", ...)
p1 <- function(measure = "PLO", ...) rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = measure, ...)
r1 <- function(...) rma.glmm(xi = tpos, ti = tpos + tneg, data = bcg, measure = "IRLN", ...)

out <- list()

### odds ratios with the unconditional models
# the cauchit link has heavy tails: lme4 fits with a random-effect variance of about 200 (and stops with a convergence warning), so only
# the fixed-effects models are used for it
for (lk in c("probit", "cloglog", "cauchit")) {
  if (lk != "cauchit") {
    out[[paste0("or_fs_", lk)]] <- core(b2(model = "UM.FS", link = lk))
    out[[paste0("or_fs_", lk, "_mod")]] <- core(b2(model = "UM.FS", link = lk, mods = ~ ablat))
  }
  out[[paste0("or_fs_", lk, "_fe")]] <- core(b2(model = "UM.FS", link = lk, method = "FE"))
  out[[paste0("or_fs_", lk, "_mod_fe")]] <- core(b2(model = "UM.FS", link = lk, mods = ~ ablat + year, method = "FE"))
  if (lk != "cauchit") {
    out[[paste0("or_rs_", lk, "_fe")]] <- core(b2(model = "UM.RS", link = lk, method = "FE"))
  }
}
out$or_rs_probit <- core(b2(model = "UM.RS", link = "probit"))
out$or_rs_cloglog <- core(b2(model = "UM.RS", link = "cloglog"))
out$or_fs_probit_t <- core(b2(model = "UM.FS", link = "probit", test = "t", level = 90))
out$or_fs_probit_nw <- core(n2(model = "UM.FS", link = "probit"))
out$or_fs_probit_nw_fe <- core(n2(model = "UM.FS", link = "probit", method = "FE"))
out$or_fs_cloglog_col <- core(c2(model = "UM.FS", link = "cloglog"))
out$or_fs_cloglog_col_fe <- core(c2(model = "UM.FS", link = "cloglog", method = "FE"))
out$or_fs_logit <- core(b2(model = "UM.FS", link = "logit"))
out$or_fs_sqrt_fe <- core(b2(model = "UM.FS", link = "sqrt", method = "FE"))

### family objects
out$or_fam_probit <- core(b2(model = "UM.FS", family = binomial(link = "probit")))
out$or_fam_binomial <- core(b2(model = "UM.FS", family = binomial))
out$or_fam_cloglog_fe <- core(b2(model = "UM.FS", family = binomial(link = "cloglog"), method = "FE"))
out$rr_fam_binomial <- core(b2("RR", model = "UM.FS", family = binomial))
out$rr_link_probit <- core(b2("RR", model = "UM.FS", link = "probit"))
out$rd_link_probit_fe <- core(b2("RD", model = "UM.FS", link = "probit", method = "FE"))
out$plo_fam_cloglog <- core(p1("PLO", family = binomial(link = "cloglog")))

### conditional models and the argument are not affected
out$al_probit <- core(b2(model = "CM.AL", link = "probit"))
out$el_probit_fe <- core(b2(model = "CM.EL", link = "probit", method = "FE"))
out$al_family <- core(b2(model = "CM.AL", family = binomial(link = "probit")))

### incidence rate ratios
for (lk in c("identity", "sqrt")) {
  out[[paste0("irr_fs_", lk)]] <- core(h2(model = "UM.FS", link = lk))
  out[[paste0("irr_fs_", lk, "_fe")]] <- core(h2(model = "UM.FS", link = lk, method = "FE"))
}
out$irr_rs_sqrt_fe <- core(h2(model = "UM.RS", link = "sqrt", method = "FE"))
out$irr_fam_poisson_sqrt <- core(h2(model = "UM.FS", family = poisson(link = "sqrt"), method = "FE"))
out$irr_fam_poisson <- core(h2(model = "UM.FS", family = poisson))

### one-group measures
for (lk in c("probit", "cloglog", "cauchit")) {
  if (lk != "cauchit") {
    out[[paste0("plo_", lk)]] <- core(p1("PLO", link = lk))
    out[[paste0("plo_", lk, "_mod")]] <- core(p1("PLO", link = lk, mods = ~ ablat))
  }
  out[[paste0("plo_", lk, "_fe")]] <- core(p1("PLO", link = lk, method = "FE"))
}
out$plo_probit_t <- core(p1("PLO", link = "probit", test = "t", level = 90))
out$plo_family_probit <- core(p1("PLO", family = binomial(link = "probit")))
out$pr_probit <- core(p1("PR", link = "probit"))
out$pr_logit <- core(p1("PR", link = "logit"))
out$pr_logit_fe <- core(p1("PR", link = "logit", method = "FE"))
out$pln_cloglog <- core(p1("PLN", link = "cloglog"))
out$pln_logit <- core(p1("PLN", link = "logit"))
# the identity and square root links are not identifiable for the random-effects model of the rates (lme4 diverges), so only FE
out$irln_identity_fe <- core(r1(link = "identity", method = "FE"))
out$irln_sqrt_fe <- core(r1(link = "sqrt", method = "FE"))
out$irln_family_sqrt_fe <- core(r1(family = poisson(link = "sqrt"), method = "FE"))

### predict
pr <- list()
m1 <- b2(model = "UM.FS", link = "probit", mods = ~ ablat)
pr$or_probit_mod <- pred(m1, newmods = c(10, 30, 50))
pr$or_probit_mod_exp <- pred(m1, newmods = c(10, 30, 50), transf = exp)
pr$or_probit <- pred(b2(model = "UM.FS", link = "probit"))
pr$or_cloglog_fe <- pred(b2(model = "UM.FS", link = "cloglog", method = "FE"), transf = exp)
pr$plo_probit <- pred(p1("PLO", link = "probit"), transf = pnorm)
pr$plo_cloglog_mod <- pred(p1("PLO", link = "cloglog", mods = ~ ablat), newmods = c(10, 30, 50))
pr$irln_sqrt_fe <- pred(r1(link = "sqrt", method = "FE"))
pr$irr_identity_fe <- pred(h2(model = "UM.FS", link = "identity", method = "FE"), transf = exp)
out$predict <- pr

### link functions of make.link() / the families, on a grid
eta <- c(-40, -9, -3, -1.5, -0.3, 0, 0.2, 1, 2.5, 8.5, 25, 40)
mu <- c(1e-12, 1e-5, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-5)
lf <- list()
for (lk in c("logit", "probit", "cauchit", "cloglog", "log", "identity", "sqrt")) {
  fam <- binomial(link = lk)
  lf[[lk]] <- list(eta = eta, linkinv = num(fam$linkinv(eta)), mu_eta = num(fam$mu.eta(eta)), mu = mu, linkfun = num(suppressWarnings(fam$linkfun(mu))),
                   valideta = fam$valideta(eta), valideta_pos = fam$valideta(abs(eta) + 1))
}
out$link_functions <- lf

### errors
err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(
  bad_link = err(b2(model = "UM.FS", link = "bogus")),
  bad_link_fe = err(b2(model = "UM.FS", link = "bogus", method = "FE"))
)

out$meta <- list(metafor = as.character(packageVersion("metafor")), lme4 = as.character(packageVersion("lme4")), R = R.version.string)
dir.create("src/tests/reference/phase9", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase9/glmm_links.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase9/glmm_links.json\n")
