# Reference values for the gaps closed after the fourth audit in the area "vcov": vcov(type = "obs"/"marginal"/"fitted"/"resid") of every fit class
# under each na.action, the default vcov(type = "fixed") of location-scale and selection models, the data summaries (M, b, k.eff, p.eff, ni, I2, H2, QE,
# QEp of selection models), coef(summary(res)), and cbind()/rbind() of escalc data frames.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit4_vcov.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  x <- as.matrix(x)
  list(dim = dim(x), rows = rownames(x), values = matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x)))
}
quiet <- function(expr) suppressWarnings(suppressMessages(expr))

ref <- list()
dat0 <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat <- dat0
dat$ablat[5] <- NA
dat$yi[3] <- NA

### the fits ----------------------------------------------------------------------------------------------------------------------------------
fits <- list(
  uni = quiet(rma(yi, vi, data = dat0)),
  uni_mods = quiet(rma(yi, vi, mods = ~ablat, data = dat)),
  uni_w = quiet(rma(yi, vi, mods = ~ablat, weights = 1 / sqrt(vi), data = dat0)),
  mv = quiet(rma.mv(yi, vi, random = ~ 1 | trial, data = dat0)),
  mv_mods = quiet(rma.mv(yi, vi, mods = ~ablat, random = ~ 1 | trial, data = dat)),
  mh = quiet(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)),
  peto = quiet(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)),
  ls = quiet(rma(yi, vi, scale = ~ablat, data = dat0)),
  ls_na = quiet(rma(yi, vi, mods = ~ablat, scale = ~ablat, data = dat)),
  sel = quiet(selmodel(rma(yi, vi, data = dat0), type = "halfnorm")),
  sel_mods = quiet(selmodel(rma(yi, vi, mods = ~ablat, data = dat), type = "halfnorm")),
  rob = quiet(robust(rma(yi, vi, data = dat0), cluster = trial)),
  rob_mods = quiet(robust(rma(yi, vi, mods = ~ablat, data = dat), cluster = trial)),
  rob_cs = quiet(robust(rma(yi, vi, mods = ~ablat, data = dat0), cluster = alloc, clubSandwich = TRUE)),
  rob_mv = quiet(robust(rma.mv(yi, vi, random = ~ 1 | trial, data = dat0), cluster = trial)),
  tf = quiet(trimfill(rma(yi, vi, data = dat0)))
)

### vcov() by type and na.action ------------------------------------------------------------------------------------------------------------
vc <- list()
for (n in names(fits)) {
  vc[[n]] <- list()
  for (ty in c("fixed", "beta", "alpha", "delta", "obs", "marginal", "fitted", "resid")) {
    vc[[n]][[ty]] <- list()
    for (na in c("na.omit", "na.exclude", "na.pass", "na.fail")) {
      options(na.action = na)
      r <- tryCatch(quiet(vcov(fits[[n]], type = ty)), error = function(e) conditionMessage(e))
      key <- sub("^na\\.", "", na)
      vc[[n]][[ty]][[key]] <- if (is.character(r)) list(error = r) else if (is.list(r)) lapply(r, mat) else mat(r)
    }
  }
}
options(na.action = "na.omit")
ref$vcov <- vc

### data summaries --------------------------------------------------------------------------------------------------------------------------
fields <- function(x, nms) {
  res <- lapply(nms, function(nm) {
    v <- x[[nm]]
    if (is.null(v)) NULL else if (is.matrix(v) || nm == "M") mat(v) else num(v)
  })
  names(res) <- nms
  res
}
ref$fields <- list(
  uni = fields(fits$uni, c("b", "k.eff", "p.eff", "k.all", "ni", "M")),
  uni_mods = fields(fits$uni_mods, c("b", "k.eff", "p.eff", "k.all", "ni", "M")),
  mv = fields(fits$mv, c("b", "k.eff", "p.eff", "M")),
  mv_mods = fields(fits$mv_mods, c("b", "k.eff", "p.eff", "M")),
  mh = fields(fits$mh, c("b", "k.eff", "p.eff", "ni")),
  peto = fields(fits$peto, c("b", "k.eff", "p.eff", "ni")),
  ls = fields(fits$ls, c("b", "k.eff", "p.eff", "M")),
  sel = fields(fits$sel, c("b", "k.eff", "p.eff", "ni", "M", "I2", "H2", "vt", "QE", "QEp")),
  rob = fields(fits$rob, c("b", "M")),
  tf = fields(fits$tf, c("b", "k.eff", "M"))
)
dat_mh <- dat.bcg
dat_mh$tpos[2] <- NA
mh_na <- quiet(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat_mh))
peto_na <- quiet(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat_mh))
ref$fields$mh_na <- fields(mh_na, c("k.eff", "p.eff", "ni"))
ref$fields$peto_na <- fields(peto_na, c("k.eff", "p.eff", "ni"))
ref$vcov_na_tables <- list(
  mh_fitted = lapply(c("na.omit", "na.exclude"), function(na) {
    options(na.action = na)
    mat(vcov(mh_na, type = "fitted"))
  })
)
options(na.action = "na.omit")

### coef(summary()) -------------------------------------------------------------------------------------------------------------------------
tab <- function(x) {
  x <- as.data.frame(x)
  list(names = names(x), rows = rownames(x), values = lapply(x, num))
}
tabs <- function(x) if (is.data.frame(x)) tab(x) else lapply(x, tab)
cs <- list(
  uni = quiet(rma(yi, vi, data = dat0)),
  uni_mods = quiet(rma(yi, vi, mods = ~ablat + alloc, data = dat)),
  uni_knha = quiet(rma(yi, vi, mods = ~ablat, data = dat0, test = "knha")),
  uni_t = quiet(rma(yi, vi, data = dat0, test = "t")),
  mv = fits$mv,
  mv_t = quiet(rma.mv(yi, vi, mods = ~ablat, random = ~ 1 | trial, data = dat, test = "t")),
  mh = fits$mh,
  peto = fits$peto,
  ls = fits$ls,
  ls_knha = quiet(rma(yi, vi, scale = ~ablat, data = dat0, test = "knha")),
  sel = fits$sel,
  sel_mods = fits$sel_mods,
  rob = fits$rob,
  rob_mods = fits$rob_mods,
  rob_cs = fits$rob_cs,
  tf = fits$tf
)
ref$coef_summary <- lapply(cs, function(x) tabs(coef(summary(x))))
ref$coef_summary_type <- list(
  ls_beta = tab(coef(summary(cs$ls), type = "beta")), ls_alpha = tab(coef(summary(cs$ls), type = "alpha")),
  sel_beta = tab(coef(summary(cs$sel), type = "beta")), sel_delta = tab(coef(summary(cs$sel), type = "delta")),
  uni_alpha = tab(coef(summary(cs$uni), type = "alpha"))
)

### cbind() and rbind() of escalc data frames ------------------------------------------------------------------------------------------------
a <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:5, ], slab = paste0("s", 1:5), digits = 3)
b <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[6:9, ], slab = paste0("s", 6:9), digits = 5)
b_dup <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[6:9, ], slab = paste0("s", 1:4), digits = 5)
nia <- escalc(measure = "SMD", m1i = m1i, m2i = m2i, sd1i = sd1i, sd2i = sd2i, n1i = n1i, n2i = n2i,
              data = data.frame(m1i = c(10, 12), m2i = c(9, 10), sd1i = c(2, 3), sd2i = c(2, 3), n1i = c(20, 30), n2i = c(20, 30)))
frame <- function(x) {
  at <- attributes(x)
  nm <- c("yi.names", "vi.names", "sei.names", "zi.names", "pval.names", "ci.lb.names", "ci.ub.names")
  yn <- at$yi.names[1]
  list(
    names = names(x), rows = rownames(x), nrow = nrow(x),
    numeric = lapply(x[sapply(x, is.numeric)], num),
    attrs = lapply(at[intersect(nm, names(at))], as.character),
    digits = if (is.null(at$digits)) NULL else num(at$digits)[1],
    measure = attr(x[[yn]], "measure"), slab = as.character(attr(x[[yn]], "slab")), ni = num(attr(x[[yn]], "ni"))
  )
}
ref$bind <- list(
  rbind_ab = frame(rbind(a, b)),
  rbind_dup = frame(rbind(a, b_dup)),
  rbind_aa = frame(rbind(a, a)),
  cbind_summary = frame(cbind(a, summary(a)[, c("sei", "zi")])),
  cbind_extra = frame(cbind(a, z = 1:5)),
  cbind_aa = frame(cbind(a, a)),
  cbind_names = frame(cbind(escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:5, ], var.names = c("y1", "v1"))[, c("y1", "v1")],
                            escalc(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:5, ], var.names = c("y2", "v2"), digits = 2)[, c("y2", "v2")])),
  rbind_ni = frame(rbind(nia, nia))
)

write_json(ref, "src/tests/reference/phase12/audit4_vcov.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
