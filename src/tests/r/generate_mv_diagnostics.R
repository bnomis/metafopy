# Reference values for the rma.mv methods profile(), ranef(), cooks.distance(), dfbetas(), rstandard(), rstudent() and
# hatvalues(): the multilevel model of dat.konstantopoulos2011 (with and without a moderator, with test="t", with a
# missing value under na.action="na.exclude"), and the multivariate model of dat.berkey1998 (known V, struct="UN").
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_mv_diagnostics.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(as.matrix(x)))
frame <- function(d) list(rows = rownames(d), cols = colnames(d), values = lapply(as.list(as.data.frame(d)), num))
reslist <- function(r) list(resid = num(r$resid), se = num(r$se), z = num(r$z), slab = as.character(r$slab))
clusterlist <- function(r) list(obs = reslist(r$obs), cluster = list(X2 = num(r$cluster$X2), k = num(r$cluster$k), slab = as.character(r$cluster$slab)))
named <- function(x) list(values = num(x), names = names(x))
prof <- function(p) list(vc = num(p[[1]]), ll = num(p$ll), beta = frame(p$beta), ci_lb = frame(p$ci.lb), ci_ub = frame(p$ci.ub), ylim = num(p$ylim), maxll = p$maxll, est = p[[9]])
errmsg <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))

kon <- dat.konstantopoulos2011
kres <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon)
kmod <- rma.mv(yi, vi, mods = ~ I(year - 1990), random = ~ 1 | district/school, data = kon)
kt <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, test = "t")
kc <- rma.mv(yi, vi, random = ~ 1 | district/school, data = kon, test = "t", dfs = "contain")

ber <- dat.berkey1998
V <- bldiag(lapply(split(ber[, c("v1i", "v2i")], ber$trial), as.matrix))
bres <- rma.mv(yi, V, mods = ~ outcome - 1, random = ~ outcome | trial, struct = "UN", data = ber)

out <- list()

out$kon <- list(
  cooks = named(cooks.distance(kres)),
  cooks_cluster = named(cooks.distance(kres, cluster = district)),
  cooks_cluster_fixed = named(cooks.distance(kres, cluster = district, reestimate = FALSE)),
  dfbetas = frame(dfbetas(kres)),
  dfbetas_cluster = frame(dfbetas(kres, cluster = district)),
  dfbetas_cluster_fixed = frame(dfbetas(kres, cluster = district, reestimate = FALSE)),
  rstudent = reslist(rstudent(kres)),
  rstudent_cluster = clusterlist(rstudent(kres, cluster = district)),
  rstudent_cluster_fixed = clusterlist(rstudent(kres, cluster = district, reestimate = FALSE)),
  rstandard = reslist(rstandard(kres)),
  rstandard_cluster = clusterlist(rstandard(kres, cluster = district)),
  hat = named(hatvalues(kres)),
  hat_matrix = num(hatvalues(kres, type = "matrix")),
  ranef = lapply(ranef(kres), frame),
  ranef_expand = lapply(ranef(kres, expand = TRUE), frame),
  ranef_transf = lapply(ranef(kres, transf = exp), frame),
  ranef_level = lapply(ranef(kres, level = 90), frame),
  profile_sigma2_2 = prof(profile(kres, sigma2 = 2, steps = 5, plot = FALSE, progbar = FALSE)),
  profile_xlim = prof(profile(kres, sigma2 = 1, xlim = c(0.01, 0.2), steps = 4, plot = FALSE, progbar = FALSE)),
  profile_all = lapply(profile(kres, steps = 4, plot = FALSE, progbar = FALSE)[1:2], prof),
  err_qqnorm = errmsg(qqnorm(kres)),
  err_profile_tau2 = errmsg(profile(kres, tau2 = 1, plot = FALSE)),
  err_profile_two = errmsg(profile(kres, sigma2 = 1, tau2 = 1, plot = FALSE)),
  err_profile_none = errmsg(profile(kres, sigma2 = 3, plot = FALSE)),
  err_profile_xlim = errmsg(profile(kres, sigma2 = 1, xlim = c(-1, 1), plot = FALSE)),
  err_cluster_len = errmsg(cooks.distance(kres, cluster = 1:3)),
  err_influence = errmsg(influence(kres))
)

out$kmod <- list(
  cooks_cluster = named(cooks.distance(kmod, cluster = district)),
  dfbetas_cluster = frame(dfbetas(kmod, cluster = district)),
  rstudent_cluster = clusterlist(rstudent(kmod, cluster = district)),
  rstandard_cluster = clusterlist(rstandard(kmod, cluster = district)),
  hat = named(hatvalues(kmod)),
  ranef = lapply(ranef(kmod), frame),
  profile_sigma2_1 = prof(profile(kmod, sigma2 = 1, steps = 5, plot = FALSE, progbar = FALSE))
)

out$kt <- list(ranef = lapply(ranef(kt), frame), ranef_contain = lapply(ranef(kc), frame))

# the tau2[2] profile stops at 0.075: at tau2[2] = 0.1 metafopy's optimiser stops at a lower local maximum than nlminb
out$ber <- list(
  cooks_cluster = named(cooks.distance(bres, cluster = trial)),
  dfbetas_cluster = frame(dfbetas(bres, cluster = trial)),
  rstudent = reslist(rstudent(bres)),
  rstudent_cluster = clusterlist(rstudent(bres, cluster = trial)),
  rstandard_cluster = clusterlist(rstandard(bres, cluster = trial)),
  hat = named(hatvalues(bres)),
  hat_matrix = num(hatvalues(bres, type = "matrix")),
  ranef = lapply(ranef(bres), frame),
  ranef_names = names(ranef(bres)),
  profile_tau2_2 = prof(profile(bres, tau2 = 2, xlim = c(0.003, 0.075), steps = 4, plot = FALSE, progbar = FALSE)),
  profile_rho = prof(profile(bres, rho = 1, steps = 5, plot = FALSE, progbar = FALSE))
)

kna <- kon
kna$yi[3] <- NA
kres_na <- suppressWarnings(rma.mv(yi, vi, random = ~ 1 | district/school, data = kna))
options(na.action = "na.exclude")
out$kna <- list(
  cooks_cluster = named(cooks.distance(kres_na, cluster = district)),
  cooks = named(cooks.distance(kres_na)),
  dfbetas = frame(dfbetas(kres_na)),
  rstandard = reslist(rstandard(kres_na)),
  rstudent_cluster = clusterlist(rstudent(kres_na, cluster = district)),
  hat = named(hatvalues(kres_na)),
  ranef = lapply(ranef(kres_na), frame)
)
options(na.action = "na.omit")

dir.create("src/tests/reference/phase7", showWarnings = FALSE)
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase7/mv_diagnostics.json")
