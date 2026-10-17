# Reference values for the arguments added after audit 8: vif(joinb=), time=, and the rest of the argument gaps.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit8_args.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) unname(as.vector(x))
out <- list()

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ablat + year + alloc, data = dat)

# vif(joinb=): the columns with the same label are permuted together
v <- vif(res, sim = 20, seed = 7, joinb = c(1, 2, 3, 4, 4))
out$vif_joinb <- list(vif = num(sapply(v$vif, function(x) x$vif)), sim = unname(v$sim), prop = num(v$prop))

# anova(): L, df and test for location-scale models, fixed
ls1 <- rma(yi, vi, mods = ~ablat + year, scale = ~ablat, data = dat, method = "ML")
ls0 <- rma(yi, vi, mods = ~ablat, scale = ~1, data = dat, method = "ML")
a <- anova(ls1, ls0, df = 2)
out$ls_lrt_df <- list(parms_f = a$parms.f, parms_r = a$parms.r, LRT = a$LRT, pval = a$pval)
lsr1 <- rma(yi, vi, mods = ~ablat + year, scale = ~ablat, data = dat)
lsr0 <- rma(yi, vi, mods = ~ablat, scale = ~ablat, data = dat)
a <- anova(lsr1, lsr0, test = "Wald")
out$ls_wald <- list(QM = a$QM, QMp = a$QMp, btt = num(a$btt))
a <- anova(lsr1, L = rbind(c(0, 1, 0), c(0, 0, 1)))
out$ls_L <- list(QM = a$QM, pval = num(a$pval), zval = num(a$zval))
a <- anova(lsr1, btt = "year", fixed = TRUE)
out$ls_fixed <- list(QM = a$QM, QMp = a$QMp)
a <- anova(ls1, ls0, df = 3)
out$ls_lrt_df3 <- list(parms_f = a$parms.f, pval = a$pval)
hack <- dat.hackshaw1998
base <- rma(yi, vi, data = hack, method = "ML")
se1 <- selmodel(base, type = "stepfun", steps = c(0.025, 0.1, 0.5))
se0 <- selmodel(base, type = "stepfun", steps = c(0.025, 0.5))
a <- anova(se1, se0, df = 2)
out$sel_lrt_df <- list(parms_f = a$parms.f, parms_r = a$parms.r, LRT = a$LRT, pval = a$pval)

# confint(extint=TRUE): the search interval is widened where the bound lies outside it
ml <- rma(yi, vi, data = dat, method = "ML")
ci0 <- confint(ml, type = "PL", xlim = c(0, 0.5))$random
ci1 <- confint(ml, type = "PL", xlim = c(0, 0.5), extint = TRUE)$random
out$confint_extint_uni <- list(without = num(ci0["tau^2", ]), with = num(ci1["tau^2", ]))
mvf <- rma.mv(yi, vi, random = ~1 | trial, data = dat, method = "ML")
m0 <- confint(mvf, sigma2 = 1, xlim = c(0, 0.5))$random
m1 <- confint(mvf, sigma2 = 1, xlim = c(0, 0.5), extint = TRUE)$random
out$confint_extint_mv <- list(without = num(m0), with = num(m1))
lsf <- rma(yi, vi, scale = ~ablat, data = dat, method = "ML")
print(coef(lsf, type = "alpha"))
l0 <- confint(lsf, alpha = 2, xlim = c(-0.1, 0.1))$random
l1 <- confint(lsf, alpha = 2, xlim = c(-0.1, 0.1), extint = TRUE)$random
out$confint_extint_ls <- list(without = num(l0), with = num(l1))

# rma.uni(): the arguments read through '...'
k1 <- rma(yi, vi, data = dat, knha = TRUE)
out$knha <- list(test = k1$test, ddf = k1$ddf, pval = num(k1$pval), ci = c(k1$ci.lb, k1$ci.ub))
k0 <- rma(yi, vi, data = dat, test = "knha", knha = FALSE)
out$knha_false <- list(test = k0$test)
d5 <- rma(yi, vi, mods = ~ablat, data = dat, test = "t", dfs = 5)
out$dfs <- list(ddf = d5$ddf, pval = num(d5$pval), QMp = d5$QMp, ci_lb = num(d5$ci.lb))
d5l <- rma(yi, vi, mods = ~ablat, scale = ~ablat, data = dat, test = "t", dfs = 5)
out$dfs_ls <- list(ddf = d5l$ddf, ddf_alpha = d5l$ddf.alpha, pval = num(d5l$pval))
i2 <- rma(yi, vi, data = dat, i2def = "2")
out$i2def <- list(I2 = i2$I2, H2 = i2$H2)
i2m <- rma(yi, vi, mods = ~ablat, data = dat, i2def = "2")
out$i2def_mods <- list(I2 = i2m$I2, H2 = i2m$H2)
out$r2def <- list()
for (d in c("1", "1v", "2", "2v", "3", "3v", "4", "4w", "5", "6", "7", "8")) {
  out$r2def[[d]] <- list(reml = rma(yi, vi, mods = ~ablat, data = dat, r2def = d)$R2, ml = rma(yi, vi, mods = ~ablat, data = dat, method = "ML", r2def = d)$R2)
}
out$skipr2 <- is.null(rma(yi, vi, mods = ~ablat, data = dat, skipr2 = TRUE)$R2)
ab <- rma(yi, vi, mods = ~factor(alloc) + I(ablat^2), data = dat, abbrev = TRUE)
out$abbrev <- rownames(ab$beta)
ab2 <- rma(yi, vi, mods = ~poly(ablat, degree = 2, raw = TRUE), scale = ~factor(alloc), data = dat, abbrev = TRUE)
out$abbrev_ls <- list(beta = rownames(ab2$beta), alpha = rownames(ab2$alpha))

# rcalc(): rowid, vnames and noid for a list of matrices
R1 <- matrix(c(1, 0.4, 0.3, 0.4, 1, 0.2, 0.3, 0.2, 1), 3, 3, dimnames = list(c("a", "b", "c"), c("a", "b", "c")))
R2 <- matrix(c(1, 0.5, 0.5, 1), 2, 2, dimnames = list(c("a", "b"), c("a", "b")))
rl <- list(s1 = R1, s2 = R2)
rc0 <- rcalc(rl, ni = c(40, 60))
out$rcalc_base <- list(dat = rc0$dat, V = unname(as.matrix(rc0$V)))
ids <- c("s2.a.s2.b", "s1.b.s1.c", "s1.a.s1.b", "s1.a.s1.c")
rc1 <- rcalc(rl, ni = c(40, 60), rowid = ids, vnames = c("ri", "x", "y", "study"), noid = TRUE)
out$rcalc_args <- list(dat = rc1$dat, V = unname(as.matrix(rc1$V)), names = names(rc1$dat))
# pairmat(fixed=TRUE)
pm <- pairmat(res, btt = "alloc", fixed = TRUE)
out$pairmat_fixed <- list(rows = rownames(pm), values = unname(as.matrix(pm)))

write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase7/audit8_args.json")
