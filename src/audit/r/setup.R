# Installs the R side of the audit into a private library and writes the installed versions.
#
#   Rscript src/audit/r/setup.R <library-dir> <versions-json>
#
# Every package is installed at the version pinned in src/audit/data/r-versions.json: metafor, metadat and gsl first, then each other package whose
# installed version differs, in the pin file's order. R itself is not installed: `audit check-r` refuses a library that differs from the pin, and the
# pin is changed by copying a run's r-versions.json over it after reviewing what the new versions change.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]
out <- args[2]
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4L)

have <- rownames(installed.packages(lib.loc = .libPaths()))
install.packages(setdiff(c("jsonlite", "remotes"), have))
pin <- jsonlite::fromJSON(file.path(here, "..", "data", "r-versions.json"), simplifyVector = FALSE)
pin$R <- NULL
first <- c("metafor", "metadat", "gsl")
# jsonlite names a package found in two library paths (a private and a system one) "name.1" for the second copy; that copy is not a package
shadow <- grepl("\\.[0-9]+$", names(pin)) & sub("\\.[0-9]+$", "", names(pin)) %in% names(pin)
pin <- pin[!shadow]
wanted <- c(intersect(first, names(pin)), setdiff(names(pin), first))
installed_version <- function(p) {
  inst <- installed.packages(lib.loc = .libPaths())
  if (p %in% rownames(inst)) inst[p, "Version"] else NA_character_
}
for (p in wanted) {
  version <- pin[[p]]
  if (!identical(installed_version(p), version)) {
    try(remotes::install_version(p, version = version, upgrade = "never", dependencies = FALSE), silent = TRUE)
  }
}
inst <- installed.packages(lib.loc = .libPaths())
vers <- as.list(setNames(inst[, "Version"], inst[, "Package"]))
vers <- vers[!(names(vers) %in% rownames(installed.packages(priority = "base")))]
vers$R <- paste(R.version$major, R.version$minor, sep = ".")
writeLines(jsonlite::toJSON(vers[order(names(vers))], auto_unbox = TRUE, pretty = TRUE), out)
failed <- Filter(function(p) !identical(installed_version(p), pin[[p]]), wanted)
for (p in failed) cat("NOT INSTALLED:", p, pin[[p]], "\n")
