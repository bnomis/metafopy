# Reference values for vif(..., sim = n, seed = s) on location-scale models (class "rma.ls"): the simulated (G)VIFs under independence of
# the moderators (and of the scale variables), their proportions ("prop") and printed output, on dat.bangertdrowns2004. The model
# matrices are given as matrices so that metafor permutes their columns, with set.seed(seed + l) for iteration l, which metafopy reproduces
# with its port of R's Mersenne-Twister and sample().
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_ls_vif_sim.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

bd <- dat.bangertdrowns2004
bd$ni100 <- bd$ni / 100
bd <- bd[complete.cases(bd[, c("yi", "vi", "ni100", "meta", "imag")]), ]

Xm <- cbind(ni100 = bd$ni100, meta = bd$meta)
Zm <- cbind(intrcpt = 1, ni100 = bd$ni100, imag = bd$imag)

part <- function(v) {
  list(
    vifs = num(v$vifs), names = names(v$vifs),
    sim = if (is.null(v$sim)) NULL else lapply(seq_len(nrow(v$sim)), function(i) num(v$sim[i, ])),
    prop = num(v$prop), sim_type = if (is.null(v$sim)) NULL else attr(v$sim, "type")
  )
}
both <- function(v) c(list(beta = part(v$beta), alpha = part(v$alpha)), list(print = printed(v)))

out <- list()

res <- rma(bd$yi, bd$vi, mods = Xm, scale = Zm)
out$both <- both(vif(res, sim = 12, seed = 42, progbar = FALSE))
out$both_btt <- both(vif(res, btt = 2, att = list(2, 2:3), sim = 8, seed = 7, progbar = FALSE))
out$both_reest <- both(vif(res, reestimate = TRUE, sim = 6, seed = 3, progbar = FALSE))

res_ml <- rma(bd$yi, bd$vi, mods = Xm, scale = Zm, method = "ML", link = "identity")
out$ml_identity <- both(vif(res_ml, sim = 8, seed = 11, progbar = FALSE))

# three location predictors
X3 <- cbind(ni100 = bd$ni100, meta = bd$meta, imag = bd$imag)
res3 <- rma(bd$yi, bd$vi, mods = X3, scale = Zm)
out$three <- both(vif(res3, sim = 10, seed = 21, progbar = FALSE))

# scale predictors only
res_s <- rma(bd$yi, bd$vi, scale = Zm)
v <- vif(res_s, sim = 8, seed = 5, progbar = FALSE)
out$scale_only <- c(part(v), list(print = printed(v)))
# location predictors only
res_l <- rma(bd$yi, bd$vi, mods = Xm, scale = ~ 1)
v <- vif(res_l, sim = 8, seed = 5, progbar = FALSE)
out$loc_only <- c(part(v), list(print = printed(v)))

dir.create("src/tests/reference/phase6", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase6/ls_vif_sim.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
