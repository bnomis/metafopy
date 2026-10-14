# Reference values for se(), nobs(), df.residual(), formula(), model.matrix(), update() and the fitted values and
# residuals of rma.glmm fits, for rma.uni, rma.mv, rma.mh, rma.peto, rma.glmm, location-scale and selmodel fits.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_accessors.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) unname(as.vector(x))
fstr <- function(f) if (is.null(f)) NULL else paste(deparse(f), collapse = "")
info <- function(o) {
  s <- se(o)
  list(nobs = nobs(o), nobs_all = as.list(nobs(o, all = TRUE)), df_residual = df.residual(o), formula = fstr(formula(o)),
       se = if (is.list(s)) lapply(s, num) else num(s),
       model_matrix = if (is.list(model.matrix(o))) lapply(model.matrix(o), unname) else unname(model.matrix(o)))
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$yi[3] <- NA
res <- suppressWarnings(rma(yi, vi, mods = ~ablat, data = dat))
resy <- suppressWarnings(rma(yi ~ ablat, vi, data = dat))
sel <- suppressWarnings(selmodel(rma(yi, vi, data = dat), type = "beta"))
gl <- rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, mods = ~ablat)
mh <- rma.mh(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, subset = 1:5)
pt <- rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
ls <- suppressWarnings(rma(yi, vi, mods = ~ablat, scale = ~year, data = dat))
mv <- rma.mv(yi, vi, random = ~1 | district/school, data = dat.konstantopoulos2011)
sub <- rma(yi, vi, data = dat, subset = 1:10)

out <- list()
out$uni <- info(res)
out$uni_yi <- c(info(resy), list(formula_yi = fstr(formula(resy, type = "yi"))))
out$uni_subset <- info(sub)
out$sel <- info(sel)
out$glmm <- c(info(gl), list(fitted = num(fitted(gl)), residuals = num(residuals(gl))))
out$mh <- info(mh)
out$peto <- info(pt)
out$ls <- c(info(ls), list(formula_scale = fstr(formula(ls, type = "scale"))))
out$mv <- info(mv)

out$update_formula <- num(coef(suppressWarnings(update(res, ~ . + year))))
out$update_drop <- num(coef(suppressWarnings(update(update(res, ~ . + year), ~ . - ablat))))
out$update_ml <- c(num(coef(suppressWarnings(update(res, method = "ML")))), suppressWarnings(update(res, method = "ML"))$tau2)
out$update_yi <- num(coef(suppressWarnings(update(resy, . ~ . + year))))
out$update_mv <- c(num(coef(update(mv, mods = ~I(year - 1990)))), update(mv, mods = ~I(year - 1990))$sigma2)
out$update_mh <- num(coef(update(mh, measure = "RR")))
out$update_peto <- num(coef(update(pt, subset = 1:6)))
out$update_glmm <- num(coef(update(gl, ~ . + year)))
out$update_ls <- num(coef(suppressWarnings(update(ls, scale = ~ablat)))$alpha)
out$update_sel <- c(num(coef(suppressWarnings(update(sel, type = "halfnorm")))$beta), num(coef(suppressWarnings(update(sel, type = "halfnorm")))$delta))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), "src/tests/reference/phase7/accessors.json")
