# Export the metadat datasets used by the fixtures to CSV for metafopy.datasets.
#
# Usage (from the repository root):  Rscript src/tests/r/export_datasets.R

suppressPackageStartupMessages(library(metadat))
source("src/tests/r/cases.R")

out_dir <- "src/metafopy/datasets/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# datasets used only by the Phase 3 (binary and count data) and Phase 4 (vcalc, rcalc) fixture scripts, and by the plot gallery
# in the user manual (dat.damico2009, dat.pignon2000), and the data-preparation helpers (dat.hasselblad1998, dat.ishak2007, dat.senn2013)
extra <- c("dat.berkey1998", "dat.collins1985a", "dat.craft2003", "dat.damico2009", "dat.hasselblad1998", "dat.ishak2007", "dat.knapp2017", "dat.pignon2000", "dat.senn2013", "dat.yusuf1985")

for (name in sort(unique(c(vapply(cases, `[[`, "", "data"), extra)))) {
  dat <- get(name)
  write.csv(dat, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE, na = "")
  cat(sprintf("%-28s %4d rows\n", name, nrow(dat)))
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
