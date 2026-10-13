# Reference values for rma.mh() and rma.peto() (fits, printed output, weights(), rstandard(),
# rstudent(), leave1out(), cumul(), confint(), fitstats()) on the bcg, nielweise2007, collins1985a,
# hart1999 and yusuf1985 data, with variations of add/to/drop00, correct, level, subset and NAs.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_mh_peto.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
resid_list <- function(x) list(resid = num(x$resid), se = num(x$se), z = num(x$z), slab = x$slab)
tab_list <- function(x) {
  out <- lapply(unclass(x)[c("estimate", "se", "zval", "pval", "ci.lb", "ci.ub", "Q", "Qp", "I2", "H2", "k")[c("estimate", "se", "zval", "pval", "ci.lb", "ci.ub", "Q", "Qp", "I2", "H2", "k") %in% names(x)]], num)
  out$slab <- x$slab
  out$print <- printed(x)
  out
}

core <- function(res) {
  ordv <<- if (res$k.all > 2) c(2, 1, rev(3:res$k.all)) else seq_len(res$k.all) # a fixed non-trivial order for cumul()
  out <- list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), QE = res$QE, QEp = res$QEp, I2 = res$I2, H2 = res$H2,
    k = res$k, k_f = res$k.f, k_yi = res$k.yi, k_pos = res$k.pos, k_all = res$k.all,
    yi = num(res$yi), vi = num(res$vi), yi_f = num(res$yi.f), vi_f = num(res$vi.f),
    fit_ml = num(res$fit.stats$ML), fit_reml = num(res$fit.stats$REML),
    weights = num(weights(res)), weights_matrix_diag = tryCatch(num(diag(weights(res, type = "matrix"))), error = function(e) NULL),
    slab = res$slab, ids = num(res$ids),
    not_na = num(res$not.na), not_na_yivi = num(res$not.na.yivi),
    outdat = lapply(res$outdat, num), outdat_f = lapply(res$outdat.f, num),
    print = printed(res), print_showfit = printed(res, showfit = TRUE),
    confint = num(unlist(confint(res)$fixed)), confint_exp = num(unlist(confint(res, transf = TRUE)$fixed)),
    rstandard = resid_list(rstandard(res)), rstudent = resid_list(rstudent(res)),
    leave1out = if (res$k > 1) tab_list(leave1out(res)), leave1out_exp = if (res$k > 1) tab_list(leave1out(res, transf = TRUE)),
    cumul = tab_list(cumul(res)), cumul_ord = tab_list(cumul(res, order = ordv)), cumul_ord_dec = tab_list(cumul(res, order = ordv, decreasing = TRUE))
  )
  for (n in c("CO", "COp", "MH", "MHp", "BD", "BDp", "TA", "TAp"))
    if (!is.null(res[[n]])) out[[tolower(n)]] <- res[[n]]
  out
}

out <- list()
bcg <- dat.bcg
bcg$label <- paste(bcg$author, bcg$year)
nw <- dat.nielweise2007
col <- dat.collins1985a

### Mantel-Haenszel ----------------------------------------------------------------------------

out$mh_bcg_or <- core(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label))
out$mh_bcg_or_nocorrect <- core(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, correct = FALSE, level = 90))
out$mh_bcg_rr <- core(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = "RR"))
out$mh_bcg_rd <- core(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, measure = "RD"))
out$mh_bcg_subset <- core(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, subset = alloc == "random"))
out$mh_nw_or <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw))
out$mh_nw_or_addto <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, add = c(1/2, 1/2), to = c("only0", "all")))
out$mh_nw_or_if0all <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, add = c(1/2, 1/4), to = c("if0all", "only0")))
out$mh_nw_or_nodrop <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, drop00 = FALSE))
out$mh_nw_or_drop2 <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, drop00 = c(TRUE, TRUE)))
out$mh_nw_rr <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "RR"))
out$mh_nw_rd <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, measure = "RD"))
out$mh_col_or <- core(rma.mh(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, slab = paste(trial, year)))
out$mh_irr <- core(rma.mh(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = dat.hart1999, measure = "IRR"))
out$mh_ird <- core(rma.mh(x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = dat.hart1999, measure = "IRD"))
bna <- bcg
bna$tpos[3] <- NA
out$mh_bcg_na <- core(suppressWarnings(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna)))
out$mh_yusuf <- core(rma.mh(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = dat.yusuf1985, subset = table == 6))
x <- cbind(ai = c(5, 3, 0, 8), bi = c(95, 47, 20, 92), ci = c(8, 6, 0, 3), di = c(92, 44, 20, 97))
out$mh_vectors <- core(rma.mh(ai = x[, 1], bi = x[, 2], ci = x[, 3], di = x[, 4]))
out$mh_k1 <- core(rma.mh(ai = 5, bi = 95, ci = 8, di = 92))

### Peto -------------------------------------------------------------------------------------

out$peto_bcg <- core(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, slab = label))
out$peto_bcg_level <- core(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, level = 90, subset = alloc != "random"))
out$peto_nw <- core(rma.peto(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw))
out$peto_nw_addto <- core(rma.peto(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, add = c(1/2, 1/2), to = c("only0", "all")))
out$peto_nw_nodrop <- core(rma.peto(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, drop00 = FALSE))
out$peto_nw_drop2 <- core(rma.peto(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = nw, drop00 = c(TRUE, TRUE)))
out$peto_col <- core(rma.peto(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = col, slab = paste(trial, year)))
out$peto_yusuf <- core(rma.peto(ai = ai, n1i = n1i, ci = ci, n2i = n2i, data = dat.yusuf1985, subset = table == 6))
out$peto_bcg_na <- core(suppressWarnings(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bna)))

### argument errors / edge cases ------------------------------------------------------------

err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
out$errors <- list(
  mh_measure = err(rma.mh(ai = 1, bi = 2, ci = 3, di = 4, measure = "SMD")),
  peto_no_events = err(rma.peto(ai = c(0, 0), bi = c(5, 6), ci = c(0, 0), di = c(5, 6)))
)

out$meta <- list(metafor = as.character(packageVersion("metafor")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase3", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase3/mh_peto.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase3/mh_peto.json\n")
