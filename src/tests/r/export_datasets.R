# Export the metadat datasets used by the fixtures to CSV for metafopy.datasets.
#
# Usage (from the repository root):  Rscript src/tests/r/export_datasets.R

suppressPackageStartupMessages(library(metadat))
source("src/tests/r/cases.R")

out_dir <- "src/metafopy/datasets/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# datasets used only by the Phase 3 (binary and count data) fixture scripts
extra <- c("dat.collins1985a", "dat.yusuf1985")

for (name in sort(unique(c(vapply(cases, `[[`, "", "data"), extra)))) {
  dat <- get(name)
  write.csv(dat, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE, na = "")
  cat(sprintf("%-28s %4d rows\n", name, nrow(dat)))
}
