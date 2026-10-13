# Reference values for the multilevel phylogenetic meta-analysis of dat.moura2021 (the example of ?dat.moura2021) with rma.mv():
# random effects for study, effect size, species and the phylogeny (known correlation matrix via R=). The package 'ape' is not needed:
# the phylogenetic covariance matrix is computed from the edge table of the tree with the tree's own branch lengths.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_moura.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

dat <- escalc(measure = "ZCOR", ri = ri, ni = ni, data = dat.moura2021$dat)
tree <- dat.moura2021$tree
ntip <- length(tree$tip.label)
parent <- tree$edge[, 1]
child <- tree$edge[, 2]
len <- tree$edge.length
paths <- matrix(0, ntip, length(child))
for (t in seq_len(ntip)) {
  node <- t
  while (node %in% child) {
    e <- which(child == node)
    paths[t, e] <- 1
    node <- parent[e]
  }
}
C <- (paths * rep(len, each = ntip)) %*% t(paths)
dimnames(C) <- list(tree$tip.label, tree$tip.label)
A <- cov2cor(C)
dat$species.id.phy <- dat$species.id

res <- rma.mv(yi, vi,
  random = list(~ 1 | study.id, ~ 1 | effect.size.id, ~ 1 | species.id, ~ 1 | species.id.phy),
  R = list(species.id.phy = A), data = dat, sparse = TRUE)

out <- list(
  sigma2 = num(res$sigma2), beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
  ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), QE = res$QE, QEp = res$QEp, k = res$k,
  loglik = num(logLik(res)), fit_reml = num(res$fit.stats$REML), s_nlevels = num(res$s.nlevels), s_names = res$s.names,
  print = printed(res),
  vcv_check = list(dim = dim(C), diag_head = num(diag(C)[1:5]), cor_corner = num(A[1:3, 1:3]), tips = tree$tip.label[1:3])
)
out$meta <- list(metafor = as.character(packageVersion("metafor")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase4", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase4/moura2021.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
