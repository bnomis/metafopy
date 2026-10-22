"""Phylogenetic trees bundled with the datasets (e.g. the tree of ``dat.moura2021``)."""

import dataclasses

import numpy
import numpy.typing
import pandas

import metafopy.datasets.base
import metafopy.errors

__all__ = ['PhyloTree', 'load_tree']

FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class PhyloTree:
    """A rooted tree stored as an edge table, like the `phylo` class of the R package ape.

    Tips are the nodes that are never a parent; the tips are taken in the order of their node ids
    (as in ape, and as in the bundled tree, the tips have the ids 1 to the number of tips).

    Attributes:
        edges: data frame with one row per edge and the columns `parent` and `child` (node ids), `length` (edge length) and
            `child_label` (label of the child node).
    """

    edges: pandas.DataFrame

    @property
    def tip_labels(self) -> tuple[str, ...]:
        """Labels of the tips, in the order of their node ids."""
        tips = self.edges[~self.edges['child'].isin(self.edges['parent'])].sort_values('child')
        return tuple(str(s) for s in tips['child_label'])

    def vcv(self, corr: bool = False) -> pandas.DataFrame:
        """Variance-covariance matrix of a Brownian-motion trait along the tree (`ape::vcv.phylo`).

        Entry `(i, j)` is the summed edge length shared by the root-to-tip paths of tips `i` and `j`.

        Args:
            corr: whether to return the correlation matrix instead (default False).

        Returns:
            Square data frame with the tip labels as index and columns.

        Raises:
            metafopy.errors.InputError: the edge table does not describe a tree (a node has two parents, or a cycle can be reached from a tip).
        """
        if self.edges['child'].duplicated().any():
            raise metafopy.errors.InputError('the edge table does not describe a tree: a node has more than one parent')
        parent_of = dict(zip(self.edges['child'], self.edges['parent'], strict=True))
        length_of = dict(zip(self.edges['child'], self.edges['length'], strict=True))
        tips = sorted(set(self.edges['child']) - set(self.edges['parent']))
        index = {child: i for i, child in enumerate(self.edges['child'])}
        paths = numpy.zeros((len(tips), len(index)))
        for t, tip in enumerate(tips):
            node = tip
            seen = 0
            while node in parent_of:
                seen += 1
                if seen > len(index):
                    raise metafopy.errors.InputError('the edge table contains a cycle and does not describe a tree')
                paths[t, index[node]] = 1.0
                node = parent_of[node]
        lengths = numpy.array([length_of[c] for c in self.edges['child']], dtype=numpy.float64)
        cov = (paths * lengths) @ paths.T
        if corr:
            sd = numpy.sqrt(numpy.diag(cov))
            cov = cov / numpy.outer(sd, sd)
        labels = self.tip_labels
        return pandas.DataFrame(cov, index=list(labels), columns=list(labels))


def load_tree(name: str) -> PhyloTree:
    """Load a bundled phylogenetic tree by dataset name, for example `load_tree('moura2021')`.

    Args:
        name: name of the dataset the tree belongs to, with or without the `dat.` prefix.

    Returns:
        The tree as a [PhyloTree][metafopy.datasets.PhyloTree].

    Raises:
        metafopy.errors.NotFoundError: there is no such tree.
    """
    base = name if name.startswith('dat.') else f'dat.{name}'
    frame = metafopy.datasets.base.read(base.removesuffix('.tree') + '.tree').copy()
    if not {'parent', 'child', 'length', 'child_label'} <= set(frame.columns):
        raise metafopy.errors.NotFoundError(f'{name!r} is not a tree dataset')
    return PhyloTree(edges=frame)
