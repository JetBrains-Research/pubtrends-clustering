import logging

from pysrc.papers.analysis.graph import build_papers_graph

logger = logging.getLogger(__name__)


def get_direct_references_subgraph(analyzer, pmid):
    """
    Extract subgraph of the papers graph containing only direct references
    of the paper with given `pmid`.
    """
    logger.info('Analyzing papers graph')
    analyzer.papers_graph = build_papers_graph(
        analyzer.df, analyzer.cit_df, analyzer.cocit_grouped_df, analyzer.bibliographic_coupling_df,
    )

    references = list(analyzer.cit_df[analyzer.cit_df['id_out'] == pmid]['id_in'])
    references.append(pmid)
    return analyzer.papers_graph.subgraph(references)


def align_clustering_for_sklearn(partition, ground_truth):
    # Get clustering subset only with IDs present in ground truth dict
    actual_clustering = {k: v for k, v in partition.items() if k in ground_truth}

    # Align clustering
    labels_true = []
    labels_pred = []

    for pmid in actual_clustering:
        labels_true.append(ground_truth[pmid])
        labels_pred.append(actual_clustering[pmid])

    return labels_true, labels_pred
