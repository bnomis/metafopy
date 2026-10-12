# Export the metadat datasets used by the fixtures to CSV for metafopy.datasets.
#
# Usage (from the repository root):  Rscript tests/r/export_datasets.R

suppressPackageStartupMessages(library(metadat))
source("tests/r/cases.R")

out_dir <- "src/metafopy/datasets/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (name in sort(unique(vapply(cases, `[[`, "", "data")))) {
  dat <- get(name)
  write.csv(dat, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE, na = "")
  cat(sprintf("%-28s %4d rows\n", name, nrow(dat)))
}
