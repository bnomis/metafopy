# Reference values for the experimental rma.glmm() measures (RR and RD for 2x2 tables, PR and PLN for proportions) and for
# predict() of rma.glmm objects, on the bcg, collins1985a and nielweise2007 data (the data of generate_glmm.R).
# rma.glmm() fits these with glm() and lme4::glmer() using log and identity links, so they are as unstable as lme4 makes them:
# several fits end with warnings (stored under "warnings") or errors (stored under "error").
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_glmm_extras.R

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
    tau2 = res$tau2, se_tau2 = res$se.tau2, sigma2 = res$sigma2, rho = res$rho, ci_lb_tau2 = res$ci.lb.tau2, ci_ub_tau2 = res$ci.ub.tau2,
    I2 = res$I2, H2 = res$H2, vt = res$vt,
    QE_Wld = res$QE.Wld, QEp_Wld = res$QEp.Wld, QE_LRT = res$QE.LRT, QEp_LRT = res$QEp.LRT, QE_df = res$QE.df,
    QM = res$QM, QMp = res$QMp, QMdf = num(res$QMdf),
    k = res$k, k_f = res$k.f, k_yi = res$k.yi, k_eff = res$k.eff, k_all = res$k.all, p = res$p, p_eff = res$p.eff, parms = res$parms,
    int_only = res$int.only, method = res$method, model = res$model,
    fit_ml = num(res$fit.stats$ML), yi = num(res$yi), vi = num(res$vi), X = num(res$X), slab = res$slab,
    warnings = I(warns),
    print = printed(res), print_showfit = printed(res, showfit = TRUE)
  )
}

# predict() output: the list components without the attributes (pi.lb/pi.ub are absent for fixed-effects models)
pred <- function(res, ...) {
  out <- predict(res, ...)
  keep <- c("pred", "se", "ci.lb", "ci.ub", "pi.lb", "pi.ub")
  r <- lapply(keep, function(n) if (is.null(out[[n]])) NULL else num(out[[n]]))
  names(r) <- gsub("\\.", "_", keep)
  r$slab <- as.character(out$slab)
  r$ddf <- if (is.null(out$ddf)) NULL else num(out$ddf)
  r$print <- printed(out)
  r
}

out <- list()
bcg <- dat.bcg
col <- dat.collins1985a
nw <- dat.nielweise2007
bna <- bcg
bna$ablat[3] <- NA
b2 <- function(measure, ...) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = measure, ...)
c2 <- function(measure, ...) rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = measure, ...)
n2 <- function(measure, ...) rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = measure, ...)
p1 <- function(measure, ...) rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = measure, ...)

### risk ratios --------------------------------------------------------------------------------------------

out$rr_fs_ml <- core(b2("RR", model = "UM.FS"))
out$rr_fs_fe <- core(b2("RR", model = "UM.FS", method = "FE"))
out$rr_fs_mod <- core(b2("RR", model = "UM.FS", mods = ~ ablat))
out$rr_fs_mod_fe <- core(b2("RR", model = "UM.FS", mods = ~ ablat + year, method = "FE"))
out$rr_fs_t <- core(b2("RR", model = "UM.FS", test = "t", level = 90))
out$rr_fs_mod_na <- core(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna, measure = "RR", model = "UM.FS", mods = ~ ablat))
out$rr_rs_ml <- core(b2("RR", model = "UM.RS"))
out$rr_rs_fe <- core(b2("RR", model = "UM.RS", method = "FE"))
out$rr_collins <- core(c2("RR", model = "UM.FS"))
out$rr_collins_fe <- core(c2("RR", model = "UM.FS", method = "FE"))
out$rr_nw <- core(n2("RR", model = "UM.FS"))
out$rr_nw_fe <- core(n2("RR", model = "UM.FS", method = "FE"))

### risk differences ---------------------------------------------------------------------------------------

out$rd_fs_ml <- core(b2("RD", model = "UM.FS"))
out$rd_fs_fe <- core(b2("RD", model = "UM.FS", method = "FE"))
out$rd_fs_mod <- core(b2("RD", model = "UM.FS", mods = ~ ablat))
out$rd_fs_mod_fe <- core(b2("RD", model = "UM.FS", mods = ~ ablat, method = "FE"))
out$rd_rs_ml <- core(b2("RD", model = "UM.RS"))
out$rd_rs_fe <- core(b2("RD", model = "UM.RS", method = "FE"))
out$rd_collins <- core(c2("RD", model = "UM.FS"))
out$rd_collins_fe <- core(c2("RD", model = "UM.FS", method = "FE"))
out$rd_nw <- core(n2("RD", model = "UM.FS"))
out$rd_nw_fe <- core(n2("RD", model = "UM.FS", method = "FE"))

### proportions and log proportions -------------------------------------------------------------------------

out$pr_ml <- core(p1("PR"))
out$pr_fe <- core(p1("PR", method = "FE"))
out$pr_mod <- core(p1("PR", mods = ~ ablat))
out$pr_mod_fe <- core(p1("PR", mods = ~ ablat, method = "FE"))
out$pln_ml <- core(p1("PLN"))
out$pln_fe <- core(p1("PLN", method = "FE"))
out$pln_mod <- core(p1("PLN", mods = ~ ablat))
out$pln_t <- core(p1("PLN", test = "t", level = 90))
out$pln_model_arg <- core(p1("PLN", model = "CM.EL"))

### arguments and errors -------------------------------------------------------------------------------

err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(
  rr_cm_el = err(b2("RR", model = "CM.EL")),
  rr_cm_al = err(b2("RR", model = "CM.AL")),
  rd_cm_el = err(b2("RD", model = "CM.EL")),
  rd_cm_al = err(b2("RD", model = "CM.AL"))
)

### predict() ----------------------------------------------------------------------------------------------------

pr <- list()
m_or <- b2("OR", model = "UM.FS")
m_or_mod <- b2("OR", model = "UM.FS", mods = ~ ablat)
m_or_mod2 <- b2("OR", model = "UM.FS", mods = ~ ablat + year)
m_or_fe <- b2("OR", model = "UM.FS", method = "FE")
m_or_t <- b2("OR", model = "UM.FS", test = "t", mods = ~ ablat)
m_al <- b2("OR", model = "CM.AL")
m_el <- b2("OR", model = "CM.EL")
m_rs <- b2("OR", model = "UM.RS")
m_irr <- rma.glmm(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = dat.hart1999, measure = "IRR", model = "CM.EL")
m_plo <- rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = "PLO")
m_plo_mod <- rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = "PLO", mods = ~ ablat)
m_irln <- rma.glmm(xi = tpos, ti = tpos + tneg, data = bcg, measure = "IRLN")
m_na <- rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna, measure = "OR", model = "UM.FS", mods = ~ ablat)
m_rr <- b2("RR", model = "UM.FS", mods = ~ ablat)
m_pln <- p1("PLN")

pr$or <- pred(m_or)
pr$or_transf_exp <- pred(m_or, transf = exp)
pr$or_transf_ilogit <- pred(m_or, transf = transf.ilogit)
pr$or_transf_ilogit_int <- pred(m_or, transf = transf.ilogit.int)
pr$or_transf_exp_mode <- pred(m_or, transf = transf.exp.mode)
pr$or_targs <- pred(m_or, transf = function(x, targs) x * targs$m, targs = list(m = 3))
pr$or_level <- pred(m_or, level = 90)
pr$or_fe <- pred(m_or_fe)
pr$or_fe_exp <- pred(m_or_fe, transf = exp)
pr$or_mod <- pred(m_or_mod)
pr$or_mod_newmods <- pred(m_or_mod, newmods = c(10, 30, 50))
pr$or_mod_newmods_exp <- pred(m_or_mod, newmods = c(10, 30, 50), transf = exp)
pr$or_mod_newmods_level <- pred(m_or_mod, newmods = c(10, 30), level = 80)
pr$or_mod_nointercept <- pred(m_or_mod, newmods = c(10, 30), intercept = FALSE)
pr$or_mod2_newmods <- pred(m_or_mod2, newmods = rbind(c(10, 1950), c(30, 1980)))
pr$or_mod2_row <- pred(m_or_mod2, newmods = c(10, 1950))
pr$or_mod2_p <- pred(m_or_mod2, newmods = rbind(c(1, 10, 1950), c(1, 30, 1980)))
pr$or_t <- pred(m_or_t)
pr$or_t_newmods <- pred(m_or_t, newmods = c(10, 30, 50), level = 99)
pr$al <- pred(m_al)
pr$el <- pred(m_el)
pr$rs <- pred(m_rs)
pr$irr <- pred(m_irr, transf = exp)
pr$plo <- pred(m_plo, transf = transf.ilogit)
pr$plo_mod <- pred(m_plo_mod, newmods = c(10, 30, 50), transf = transf.ilogit)
pr$irln <- pred(m_irln, transf = exp)
pr$na_mod <- pred(m_na)
pr$rr_mod <- pred(m_rr, newmods = c(10, 30), transf = exp)
pr$pln <- pred(m_pln, transf = exp)
out$predict <- pr

out$meta <- list(metafor = as.character(packageVersion("metafor")), lme4 = as.character(packageVersion("lme4")), R = R.version.string)
dir.create("src/tests/reference/phase6", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase6/glmm_extras.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase6/glmm_extras.json\n")
