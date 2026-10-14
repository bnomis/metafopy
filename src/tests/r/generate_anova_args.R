# Reference values for p.adjust() and for anova(..., adjust=, refit=) of rma.uni and rma.mv fits.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_anova_args.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) unname(as.vector(x))
p1 <- c(0.01, 0.04, 0.03, 0.005, 0.2, 0.04, 0.8, 0.0001)
p2 <- c(0.02, NA, 0.5, 0.001, 0.02)
p3 <- c(0.03, 0.04)
out <- list(padjust = list())
for (m in p.adjust.methods) {
  out$padjust[[m]] <- list(p1 = num(p.adjust(p1, m)), p2 = num(p.adjust(p2, m)), p3 = num(p.adjust(p3, m)), n = num(p.adjust(p3, m, n = 5)))
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ablat + year + alloc, data = dat)
res0 <- rma(yi, vi, data = dat)
L <- rbind(c(0, 1, 0, 0, 0), c(0, 0, 1, 0, 0), c(0, 0, 0, 1, -1))
out$uni_list_holm <- sapply(anova(res, btt = list(2, 3, 4:5), adjust = "holm"), function(a) a$QMp)
out$uni_list_bonf <- sapply(anova(res, btt = list(2, 3, 4:5), adjust = TRUE), function(a) a$QMp)
out$uni_names_BH <- sapply(anova(res, btt = c("ablat", "year"), adjust = "BH"), function(a) a$QMp)
out$uni_single <- anova(res, btt = 2:3, adjust = "holm")$QMp
out$uni_rhs <- anova(res, btt = 2, rhs = -0.02)$QM
out$uni_X_hommel <- num(anova(res, X = L, adjust = "hommel")$pval)
out$uni_X_BY <- num(anova(res, X = L, adjust = "BY")$pval)
lr <- anova(res, res0, refit = TRUE)
out$uni_refit <- list(LRT = lr$LRT, pval = lr$pval, fit_f = num(lr$fit.stats.f), fit_r = num(lr$fit.stats.r), tau2_f = lr$tau2.f, tau2_r = lr$tau2.r, R2 = lr$R2, parms_f = lr$parms.f, parms_r = lr$parms.r)

d2 <- dat.konstantopoulos2011
mv <- rma.mv(yi, vi, mods = ~I(year - 1990), random = ~1 | district/school, data = d2)
mv0 <- rma.mv(yi, vi, random = ~1 | district/school, data = d2)
lr2 <- anova(mv, mv0, refit = TRUE)
out$mv_refit <- list(LRT = lr2$LRT, pval = lr2$pval, fit_f = num(lr2$fit.stats.f), fit_r = num(lr2$fit.stats.r))
out$mv_X_holm <- num(anova(mv, X = rbind(c(1, 0), c(1, 10), c(0, 1)), adjust = "holm")$pval)
out$mv_list <- sapply(anova(mv, btt = list(1, 2), adjust = "hochberg"), function(a) a$QMp)
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase7/anova_args.json")
