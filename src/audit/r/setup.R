# Installs the R side of the audit into a private library and writes the installed versions.
#
#   Rscript src/audit/r/setup.R <library-dir> <versions-json>
#
# metafor and metadat are pinned; the other packages are installed at the newest version the running R accepts and their versions are then recorded,
# so a later run on a different library is reported by `audit check-r` instead of silently comparing against another oracle.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]
out <- args[2]
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4L)

pinned <- c(metafor = "5.2-1", metadat = "1.6-0", gsl = "2.1-8")
loose <- c("jsonlite", "remotes", "mvtnorm", "CompQuadForm", "numDeriv", "Matrix", "clubSandwich", "lme4", "GLMMadaptive", "glmmTMB",
           "emmeans", "survival", "Epi", "estmeansd", "metaBLUE", "BiasedUrn", "nloptr", "minqa", "dfoptim", "ucminf", "optimParallel",
           "Rsolnp", "alabama", "BB", "pracma", "subplex", "lbfgs", "lbfgsb3c", "metafor", "testthat", "xtable", "survival", "sandwich")

have <- rownames(installed.packages(lib.loc = lib))
install.packages(setdiff(c("jsonlite", "remotes"), have))
for (p in names(pinned)) {
  if (!(p %in% have) || packageVersion(p, lib.loc = lib) != package_version(pinned[[p]])) {
    remotes::install_version(p, version = pinned[[p]], upgrade = "never", dependencies = TRUE)
  }
}
rest <- setdiff(loose, c(names(pinned), have))
for (p in rest) {
  try(install.packages(p, dependencies = TRUE), silent = TRUE)
}
inst <- installed.packages(lib.loc = lib)
vers <- as.list(setNames(inst[, "Version"], inst[, "Package"]))
vers$R <- paste(R.version$major, R.version$minor, sep = ".")
writeLines(jsonlite::toJSON(vers[order(names(vers))], auto_unbox = TRUE, pretty = TRUE), out)
missing <- setdiff(c(names(pinned), loose), names(vers))
if (length(missing)) cat("NOT INSTALLED:", missing, "\n")
