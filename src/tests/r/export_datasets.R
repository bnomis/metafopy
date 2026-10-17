# Export the metadat datasets used by the fixtures to CSV for metafopy.datasets.
#
# Usage (from the repository root):  Rscript src/tests/r/export_datasets.R

suppressPackageStartupMessages(library(metadat))
source("src/tests/r/cases.R")

out_dir <- "src/metafopy/datasets/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# every data frame of metadat (lists such as dat.moura2021, dat.lim2014, dat.maire2019 and dat.metap need their own export)
metadat_names <- data(package = "metadat")$results[, "Item"]
frames <- Filter(function(name) is.data.frame(get(name)), metadat_names)

for (name in sort(unique(c(vapply(cases, `[[`, "", "data"), frames)))) {
  dat <- get(name)
  write.csv(dat, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE, na = "")
  cat(sprintf("%-28s %4d rows\n", name, nrow(dat)))
}

# character columns that read as numbers (dat.yusuf1985 `id` is "5.10"): pandas must keep them as text
for (name in frames) {
  chr <- names(Filter(function(col) is.character(col) || is.factor(col), get(name)))
  numeric_like <- Filter(function(col) all(grepl("^[-+]?[0-9.]+([eE][-+]?[0-9]+)?$", as.character(get(name)[[col]])[!is.na(get(name)[[col]])])) && any(!is.na(get(name)[[col]])), chr)
  if (length(numeric_like)) cat(sprintf("TEXT %s: %s\n", name, paste(numeric_like, collapse = ", ")))
}

# character columns with strings that pandas reads as missing ("", "NA", "None", ...): list them in MISSING_STRINGS of src/metafopy/datasets/base.py
# (write.csv(na = "") makes R's NA an empty field, so the column must be listed there to keep the strings apart from R's NA)
pandas_na <- c("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null")
for (name in frames) {
  for (col in names(get(name))) {
    x <- as.character(get(name)[[col]])
    if (any(!is.na(x) & x %in% pandas_na)) cat(sprintf("MISSING_STRINGS %s: %s\n", name, col))
  }
}

# dat.moura2021 is a list with the data frame `dat` and the phylogenetic tree `tree` (class "phylo"); the data go to
# dat.moura2021.csv and the tree to dat.moura2021.tree.csv, one row per edge with the label of its child node
moura <- dat.moura2021
write.csv(moura$dat, file.path(out_dir, "dat.moura2021.csv"), row.names = FALSE, na = "")
tree <- moura$tree
labels <- c(tree$tip.label, tree$node.label)
edges <- data.frame(parent = tree$edge[, 1], child = tree$edge[, 2], length = tree$edge.length, child_label = labels[tree$edge[, 2]])
write.csv(edges, file.path(out_dir, "dat.moura2021.tree.csv"), row.names = FALSE, na = "")
cat(sprintf("%-28s %4d rows\n%-28s %4d edges\n", "dat.moura2021", nrow(moura$dat), "dat.moura2021.tree", nrow(edges)))
