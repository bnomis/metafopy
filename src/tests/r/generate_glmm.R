# Reference values for rma.glmm() (UM.FS, UM.RS, CM.AL, CM.EL for odds ratios and incidence rate ratios, PLO and IRLN)
# on the bcg, nielweise2007, collins1985a and hart1999 data. rma.glmm() fits its mixed models with lme4::glmer(),
# whose optimiser stops at a loose tolerance, so the estimates are only reproducible to about 1e-4.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_glmm.R

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
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(withCallingHandlers(expr, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
                  error = function(e) conditionMessage(e))
  if (is.character(res)) return(list(error = res, warnings = warns))
  elapsed <- proc.time()[["elapsed"]] - t0
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
    warnings = warns, elapsed = elapsed,
    print = printed(res), print_showfit = printed(res, showfit = TRUE)
  )
}

out <- list()
bcg <- dat.bcg
col <- dat.collins1985a
nw <- dat.nielweise2007
bcgargs <- function(...) rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = "OR", ...)

### unconditional models --------------------------------------------------------------------------------

out$fs_ml <- core(bcgargs(model = "UM.FS"))
out$fs_fe <- core(bcgargs(model = "UM.FS", method = "FE"))
out$fs_ml_mod <- core(bcgargs(model = "UM.FS", mods = ~ ablat))
out$fs_ml_mod_fe <- core(bcgargs(model = "UM.FS", mods = ~ ablat + year, method = "FE"))
out$fs_ml_coding1 <- core(bcgargs(model = "UM.FS", coding = 1))
out$fs_ml_t <- core(bcgargs(model = "UM.FS", test = "t", level = 90))
out$fs_collins <- core(rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", model = "UM.FS"))
out$fs_nw <- core(rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "OR", model = "UM.FS"))
out$fs_subset <- core(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = "OR", model = "UM.FS", subset = alloc == "random"))
bna <- bcg
bna$ablat[3] <- NA
out$fs_ml_mod_na <- core(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna, measure = "OR", model = "UM.FS", mods = ~ ablat))
out$rs_ml <- core(bcgargs(model = "UM.RS"))
out$rs_ml_cor <- core(bcgargs(model = "UM.RS", cor = TRUE))
out$rs_fe <- core(bcgargs(model = "UM.RS", method = "FE"))
out$rs_ml_mod <- core(bcgargs(model = "UM.RS", mods = ~ ablat))
out$rs_collins <- core(rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", model = "UM.RS"))

### conditional models ----------------------------------------------------------------------------------

out$al_ml <- core(bcgargs(model = "CM.AL"))
out$al_fe <- core(bcgargs(model = "CM.AL", method = "FE"))
out$al_ml_mod <- core(bcgargs(model = "CM.AL", mods = ~ ablat))
out$al_collins <- core(rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", model = "CM.AL"))
out$al_nw <- core(rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "OR", model = "CM.AL"))
out$el_ml <- core(bcgargs(model = "CM.EL"))
out$el_fe <- core(bcgargs(model = "CM.EL", method = "FE"))
out$el_fe_mod <- core(bcgargs(model = "CM.EL", mods = ~ ablat, method = "FE"))
out$el_collins <- core(rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, measure = "OR", model = "CM.EL"))
out$el_nw <- core(rma.glmm(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "OR", model = "CM.EL"))

### incidence rate ratios -----------------------------------------------------------------------------

hart <- dat.hart1999
irr <- function(...) rma.glmm(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = hart, measure = "IRR", ...)
out$irr_fs <- core(irr(model = "UM.FS"))
out$irr_fs_fe <- core(irr(model = "UM.FS", method = "FE"))
out$irr_rs <- core(irr(model = "UM.RS"))
out$irr_el <- core(irr(model = "CM.EL"))
out$irr_el_mod <- core(irr(model = "CM.EL", mods = ~ year))
out$irr_al <- core(irr(model = "CM.AL"))

### one-group measures -------------------------------------------------------------------------------

out$plo <- core(rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = "PLO"))
out$plo_fe <- core(rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = "PLO", method = "FE"))
out$plo_mod <- core(rma.glmm(xi = tpos, mi = tneg, data = bcg, measure = "PLO", mods = ~ ablat))
out$irln <- core(rma.glmm(xi = tpos, ti = tpos + tneg, data = bcg, measure = "IRLN"))
out$irln_mod <- core(rma.glmm(xi = tpos, ti = tpos + tneg, data = bcg, measure = "IRLN", mods = ~ ablat))

### arguments and errors -------------------------------------------------------------------------------

err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(
  no_measure = err(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg)),
  bad_model = err(bcgargs(model = "XX")),
  bad_method = err(bcgargs(method = "REML")),
  bad_coding = err(bcgargs(coding = 2)),
  bad_measure = err(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = "SMD")),
  too_many = err(rma.glmm(ai = tpos[1:2], bi = tneg[1:2], ci = cpos[1:2], di = cneg[1:2], data = bcg, measure = "OR", mods = ~ ablat + year))
)

out$meta <- list(metafor = as.character(packageVersion("metafor")), lme4 = as.character(packageVersion("lme4")), R = R.version.string)
dir.create("src/tests/reference/phase3", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase3/glmm.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase3/glmm.json\n")
