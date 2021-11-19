import logging
from collections import Counter
from functools import lru_cache
from itertools import chain

import community
import numpy as np
from celery import Celery
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from pysrc.fasttext.fasttext import PRETRAINED_MODEL_CACHE
from pysrc.papers.analysis.graph import to_weighted_graph
from pysrc.papers.analysis.node2vec import node2vec
from pysrc.papers.analysis.text import build_stemmed_corpus, vectorize_corpus, tokens_embeddings, texts_embeddings
from pysrc.papers.analysis.topics import cluster_and_sort, compute_topics_similarity_matrix
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.utils import SEED
from utils.analysis import get_direct_references_subgraph, align_clustering_for_sklearn
from utils.io import load_analyzer, load_clustering, get_review_pmids
from utils.preprocessing import preprocess_clustering, get_clustering_level

celery_app = Celery('grid_search', backend='redis://localhost:6379', broker='redis://localhost:6379')

# Without extension
OUTPUT_NAME = 'grid_search_2021_11_17'

# Save all generated partitions for further investigation
# (might consume a LOT of space)
SAVE_PARTITION = True

similarity_weights = [1, 3, 10]
topics_sizes = [(20, 10), (10, 50), (50, 10)]

louvain_resolution = [0.8]
embeddings_factor = [1, 3, 10]
pca_components = [15]
min_df = [0.001]
max_df = [0.8]
max_features = [10000]

louvain_params = dict(
    method=['louvain'],
    similarity_bibliographic_coupling=similarity_weights.copy(), similarity_cocitation=similarity_weights.copy(),
    similarity_citation=similarity_weights.copy(), louvain_resolution=louvain_resolution.copy(),
    topics_sizes=topics_sizes.copy()
)

lda_params = dict(
    method=['lda'],
    min_df=min_df.copy(), max_df=max_df.copy(), max_features=max_features.copy(),
    topics_sizes=topics_sizes.copy()
)

embeddings_params = dict(
    method=['embeddings'],
    similarity_bibliographic_coupling=similarity_weights.copy(), similarity_cocitation=similarity_weights.copy(),
    similarity_citation=similarity_weights.copy(),
    min_df=min_df.copy(), max_df=max_df.copy(), max_features=max_features.copy(),
    embeddings_factor_graph=embeddings_factor.copy(), embeddings_factor_text=embeddings_factor.copy(),
    pca_components=pca_components.copy(),
    topics_sizes=topics_sizes.copy()
)

dbscan_params = dict(
    method=['dbscan'],
    similarity_bibliographic_coupling=similarity_weights.copy(), similarity_cocitation=similarity_weights.copy(),
    similarity_citation=similarity_weights.copy(),
    min_df=min_df.copy(), max_df=max_df.copy(), max_features=max_features.copy(),
    graph_embeddings_factor=embeddings_factor.copy(), text_embeddings_factor=embeddings_factor.copy(),
    topics_sizes=topics_sizes.copy()
)

param_grid = [
    embeddings_params,
    louvain_params,
    lda_params
]


def similarity_param(i):
    if i == 0:
        return 'similarity_bibliographic_coupling'
    elif i == 1:
        return 'similarity_cocitation'
    elif i == 2:
        return 'similarity_citation'
    else:
        raise Exception(f'Illegal param {i}')


# Add zero's combinations for similarity features
for i in range(3):
    pi = similarity_param(i)
    param_grid.extend([
        dict(embeddings_params, **{pi: [0]}),
        dict(louvain_params, **{pi: [0]})
    ])
    for j in range(i + 1, 3):
        pj = similarity_param(j)
        param_grid.extend([
            dict(embeddings_params, **{pi: [0], pj: [0]}),
            dict(louvain_params, **{pi: [0], pj: [0]})
        ])

# Add zero's combinations for embeddings features
for i in range(2):
    pi = 'embeddings_factor_graph' if i == 0 else 'embeddings_factor_text'
    param_grid.extend([
        dict(embeddings_params, **{pi: [0]}),
        dict(dbscan_params, **{pi: [0]}),
    ])

print('Parameters grid size', len(ParameterGrid(param_grid)))

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@lru_cache(maxsize=1000)
def get_similarity_func(similarity_bibliographic_coupling,
                        similarity_cocitation,
                        similarity_citation):
    def inner(d):
        # Ignore single cocitation because of review paper
        return similarity_bibliographic_coupling * np.log1p(d.get('bibcoupling', 0)) + \
               similarity_cocitation * (np.log1p(max(0, d.get('cocitation', 0) - 1))) + \
               similarity_citation * d.get('citation', 0)

    return inner


# LDA
def cluster_lda(X, topics_max_numbers, topic_min_size):
    logging.debug('Looking for an appropriate number of clusters,'
                  f'min_cluster_size={topic_min_size}, max_clusters={topics_max_numbers}')
    r = min(int(X.shape[0] / topic_min_size), topics_max_numbers) + 1
    l = 1

    prev_min_size = None
    while l < r - 1:
        n_clusters = int((l + r) / 2)
        lda = LatentDirichletAllocation(n_components=n_clusters, random_state=SEED).fit(X)
        clusters = lda.transform(X).argmax(axis=1)
        clusters_counter = Counter(clusters)
        min_size = clusters_counter.most_common()[-1][1]
        logger.debug(f'l={l}, r={r}, n_clusters={n_clusters}, min_cluster_size={topic_min_size}, '
                     f'prev_min_size={prev_min_size}, min_size={min_size}')
        if min_size < topic_min_size:
            if prev_min_size is not None and min_size <= prev_min_size:
                break
            r = n_clusters + 1
        else:
            l = n_clusters
        prev_min_size = min_size

    logger.debug(f'Number of clusters = {n_clusters}')
    logger.debug(f'Min cluster size = {prev_min_size}')
    logger.debug('Reorder clusters by size descending')
    reorder_map = {c: i for i, (c, _) in enumerate(clusters_counter.most_common())}
    return [reorder_map[c] for c in clusters]


@lru_cache(maxsize=1000)
def preprocess_lda(analyzer, subgraph, max_df, min_df, max_features):
    # Use only papers present in the subgraph
    subgraph_df = analyzer.df[analyzer.df.id.isin(subgraph.nodes)]
    node_ids = list(subgraph_df.id.values)

    # Build and vectorize corpus for LDA
    papers_sentences_corpus = build_stemmed_corpus(subgraph_df)
    logger.debug(f'Vectorize corpus')
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        preprocessor=lambda t: t,
        tokenizer=lambda t: t
    )
    counts = vectorizer.fit_transform([list(chain(*sentences)) for sentences in papers_sentences_corpus])
    logger.debug(f'Vectorized corpus size {counts.shape}')

    return counts, node_ids


def topic_analysis_lda(analyzer, subgraph, **settings):
    X, node_ids = preprocess_lda(analyzer, subgraph,
                                 max_df=settings['max_df'],
                                 min_df=settings['min_df'],
                                 max_features=settings['max_features'])
    topics_max_number, topic_min_size = settings['topics_sizes']
    clusters = cluster_lda(X, topics_max_numbers=topics_max_number, topic_min_size=topic_min_size)
    return dict(zip(node_ids, clusters))


# Louvain
def topic_analysis_louvain(similarity_graph, **settings):
    """
    Performs clustering of similarity topics with different cluster resolution
    :param similarity_graph: Similarity graph
    :param settings: contains all tunable parameters
    :return: merged_partition
    """
    logging.debug('Compute aggregated similarity')
    similarity_func = get_similarity_func(settings['similarity_bibliographic_coupling'],
                                          settings['similarity_cocitation'],
                                          settings['similarity_citation'])
    for _, _, d in similarity_graph.edges(data=True):
        d['similarity'] = similarity_func(d)

    logging.debug('Graph clustering via Louvain community algorithm')
    partition_louvain = community.best_partition(
        similarity_graph, weight='similarity', random_state=SEED, resolution=settings['louvain_resolution']
    )
    logging.debug(f'Best partition {len(set(partition_louvain.values()))} components')
    components = set(partition_louvain.values())
    comp_sizes = {c: sum([partition_louvain[node] == c for node in partition_louvain.keys()]) for c in components}
    logging.debug(f'Components: {comp_sizes}')

    if len(similarity_graph.edges) > 0:
        modularity = community.modularity(partition_louvain, similarity_graph)
        logging.debug(f'Graph modularity (possible range is [-1, 1]): {modularity :.3f}')

    logging.debug('Merge small components')
    similarity_matrix = compute_similarity_graph_matrix(similarity_graph, similarity_func, partition_louvain)
    topics_max_number, topic_min_size = settings['topics_sizes']
    return merge_components(partition_louvain, similarity_matrix,
                            topics_max_number=topics_max_number, topic_min_size=topic_min_size)


def compute_similarity_graph_matrix(similarity_graph, similarity_func, partition):
    logging.debug('Computing mean similarity of all edges between topics')
    n_comps = len(set(partition.values()))
    edges = len(similarity_graph.edges)
    sources = [None] * edges
    targets = [None] * edges
    similarities = [0.0] * edges
    i = 0
    for u, v, data in similarity_graph.edges(data=True):
        sources[i] = u
        targets[i] = v
        similarities[i] = similarity_func(data)
        i += 1
    df = pd.DataFrame(partition.items(), columns=['id', 'comp'])
    similarity_df = pd.DataFrame(data={'source': sources, 'target': targets, 'similarity': similarities})
    similarity_topics_df = similarity_df.merge(df, how='left', left_on='source', right_on='id') \
        .merge(df, how='left', left_on='target', right_on='id')
    logging.debug('Calculate mean similarity between for topics')
    mean_similarity_topics_df = \
        similarity_topics_df.groupby(['comp_x', 'comp_y'])['similarity'].mean().reset_index()
    similarity_matrix = np.zeros(shape=(n_comps, n_comps))
    for index, row in mean_similarity_topics_df.iterrows():
        cx, cy = int(row['comp_x']), int(row['comp_y'])
        similarity_matrix[cx, cy] = similarity_matrix[cy, cx] = row['similarity']
    return similarity_matrix


def merge_components(partition, similarity_matrix, topics_max_number, topic_min_size):
    """
    Merge small topics to required number of topics and minimal size, reorder topics by size
    """
    logging.debug(f'Merging: max {topics_max_number} components with min size {topic_min_size}')
    comp_sizes = {c: sum([partition[paper] == c for paper in partition.keys()])
                  for c in (set(partition.values()))}
    logging.debug(f'{len(comp_sizes)} comps, comp_sizes: {comp_sizes}')

    merge_index = 1
    while len(comp_sizes) > 1 and \
            (len(comp_sizes) > topics_max_number or min(comp_sizes.values()) < topic_min_size):
        logging.debug(f'{merge_index}. Pick minimal and merge it with the closest by similarity')
        merge_index += 1
        min_comp = min(comp_sizes.keys(), key=lambda c: comp_sizes[c])
        comp_to_merge = max([c for c in partition.values() if c != min_comp],
                            key=lambda c: similarity_matrix[min_comp, c])
        logging.debug(f'Merging with most similar comp {comp_to_merge}')
        comp_update = min(min_comp, comp_to_merge)
        comp_sizes[comp_update] = comp_sizes[min_comp] + comp_sizes[comp_to_merge]
        if min_comp != comp_update:
            del comp_sizes[min_comp]
        else:
            del comp_sizes[comp_to_merge]
        logging.debug(f'Merged comps: {len(comp_sizes)}, updated comp_sizes: {comp_sizes}')
        for (paper, c) in list(partition.items()):
            if c == min_comp or c == comp_to_merge:
                partition[paper] = comp_update

        logging.debug('Update similarities')
        for i in range(len(similarity_matrix)):
            similarity_matrix[i, comp_update] = \
                (similarity_matrix[i, min_comp] + similarity_matrix[i, comp_to_merge]) / 2
            similarity_matrix[comp_update, i] = \
                (similarity_matrix[min_comp, i] + similarity_matrix[comp_to_merge, i]) / 2

    logging.debug('Sorting comps by size descending')
    sorted_components = {
        c: i for i, c in enumerate(sorted(set(comp_sizes), key=lambda c: comp_sizes[c], reverse=True))
    }
    logging.debug(f'Comps reordering by size: {sorted_components}')
    merged_partition = {paper: sorted_components[c] for paper, c in partition.items()}
    components_counter = Counter(merged_partition.values())
    for k, v in components_counter.most_common():
        logging.debug(f'Component {k}: {v} ({int(100 * v / len(merged_partition))}%)')
    return merged_partition


@lru_cache(maxsize=1000)
def preprocess_embeddings(
        subgraph,
        subgraph_df_ids,
        subgraph_df_titles,
        subgraph_df_abstracts,
        subgraph_df_meshs,
        subgraph_df_keywords,
        min_df, max_df, max_features,
        similarity_bibliographic_coupling,
        similarity_cocitation,
        similarity_citation,
        embeddings_factor_graph,
        embeddings_factor_text
):
    subgraph_df = pd.DataFrame(dict(id=subgraph_df_ids, title=subgraph_df_titles, abstract=subgraph_df_abstracts,
                                    mesh=subgraph_df_meshs, keywords=subgraph_df_keywords))
    corpus, corpus_tokens, corpus_counts = vectorize_corpus(
        subgraph_df,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    logger.debug('Analyzing tokens embeddings')
    model = PRETRAINED_MODEL_CACHE.download_and_load_model
    corpus_tokens_embedding = np.array([
        model.get_vector(t) if model.has_index_for(t)
        else np.zeros(model.vector_size)  # Support out-of-dictionary missing embeddings
        for t in corpus_tokens
    ])
    logger.debug('Analyzing texts embeddings')
    texts_embds = texts_embeddings(
        corpus_counts, corpus_tokens_embedding
    )
    logger.debug('Analyzing papers graph embeddings')
    similarity_func = get_similarity_func(similarity_bibliographic_coupling,
                                          similarity_cocitation,
                                          similarity_citation)
    weighted_similarity_graph = to_weighted_graph(subgraph, weight_func=similarity_func)
    logger.debug('Prepare sparse graph for visualization')

    sparse_papers_graph = PapersAnalyzer.prepare_sparse_papers_graph(subgraph, weighted_similarity_graph)
    graph_embds = node2vec(subgraph_df['id'], sparse_papers_graph)
    logger.debug('Computing aggregated graph and text embeddings for papers')
    papers_embeddings = np.concatenate(
        (graph_embds * embeddings_factor_graph,
         texts_embds * embeddings_factor_text), axis=1)

    return papers_embeddings


def topic_analysis_embeddings(analyzer, subgraph, **settings):
    """
    Rerun topic analysis based on embeddings
    """
    subgraph_df = analyzer.df[analyzer.df.id.isin(subgraph.nodes)]
    embeddings = preprocess_embeddings(
        subgraph,
        # Dataframe cannot be hashed, so use tuples of columns
        tuple(list(subgraph_df['id'])),
        tuple(list(subgraph_df['title'])),
        tuple(list(subgraph_df['abstract'])),
        tuple(list(subgraph_df['mesh'])),
        tuple(list(subgraph_df['keywords'])),
        settings['min_df'], settings['max_df'], settings['max_features'],
        settings['similarity_bibliographic_coupling'],
        settings['similarity_cocitation'],
        settings['similarity_citation'],
        settings['embeddings_factor_graph'], settings['embeddings_factor_text']
    )
    logger.debug('Computing PCA projection')
    pca = PCA(n_components=min(len(embeddings), settings['pca_components']))
    t = StandardScaler().fit_transform(embeddings)
    pca_coords = pca.fit_transform(t)
    logger.debug(f'Explained variation {int(np.sum(pca.explained_variance_ratio_) * 100)}%')
    topics_max_number, topic_min_size = settings['topics_sizes']
    clusters, _ = cluster_and_sort(
        pca_coords, topics_max_number, topic_min_size
    )
    node_ids = list(subgraph_df.id.values)
    return dict(zip(node_ids, clusters))


def topic_analysis_dbscan(analyzer, subgraph, **settings):
    """
    Rerun topic analysis based on embeddings
    """
    subgraph_df = analyzer.df[analyzer.df.id.isin(subgraph.nodes)]
    embeddings = preprocess_embeddings(
        subgraph,
        # Dataframe cannot be hashed, so use tuples of columns
        tuple(list(subgraph_df['id'])),
        tuple(list(subgraph_df['title'])),
        tuple(list(subgraph_df['abstract'])),
        tuple(list(subgraph_df['mesh'])),
        tuple(list(subgraph_df['keywords'])),
        settings['min_df'], settings['max_df'], settings['max_features'],
        settings['similarity_bibliographic_coupling'],
        settings['similarity_cocitation'],
        settings['similarity_citation'],
        settings['embeddings_factor_graph'], settings['embeddings_factor_text']
    )
    topics_max_number, topic_min_size = settings['topics_sizes']
    db = DBSCAN(min_samples=topic_min_size).fit(embeddings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_ + 1  # Zero is noise cluster
    similarity_matrix = compute_topics_similarity_matrix(embeddings, labels)
    return merge_components(dict(zip(subgraph_df['id'], labels)), similarity_matrix,
                            topics_max_number=topics_max_number, topic_min_size=topic_min_size)


# Topic analysis
def topic_analysis(analyzer, subgraph, method, **method_params):
    """
    Returns partition - dictionary {pmid (str): cluster (int)}
    """
    if method == 'embeddings':
        return topic_analysis_embeddings(analyzer, subgraph, **method_params)
    if method == 'dbscan':
        return topic_analysis_dbscan(analyzer, subgraph, **method_params)
    elif method == 'louvain':
        return topic_analysis_louvain(subgraph, **method_params)
    elif method == 'lda':
        return topic_analysis_lda(analyzer, subgraph, **method_params)
    else:
        raise ValueError(f'Unknown clustering method {method}')


def run_grid_search(analyzer, subgraph, ground_truth, metrics, param_grid, save_partition=False):
    # Accumulate grid results for all hierarchy levels
    grid_results = []
    partitions = []

    parameter_grid = ParameterGrid(param_grid)
    grid_size = len(parameter_grid)
    for i, param_values in enumerate(parameter_grid):
        partition = topic_analysis(analyzer, subgraph, **param_values)
        if save_partition:
            param_partition = param_values.copy()
            param_partition['partition'] = partition
            partitions.append(param_partition)

        # Iterate over hierarchy levels to avoid re-calculating same clustering for different ground truth
        for level, ground_truth_partition in ground_truth.items():
            result = param_values.copy()
            result['level'] = level
            labels_true, labels_pred = align_clustering_for_sklearn(partition, ground_truth_partition)
            result['n_clusters'] = len(set(labels_pred))

            # Evaluate different metrics
            for metric in metrics:
                result[metric.__name__] = metric(labels_true, labels_pred)

            grid_results.append(result)

        if (i + 1) % 10 == 0:
            print(f' {i + 1} / {grid_size}\n')
    print('\n')

    return grid_results, partitions


@celery_app.task(name='run_single_parameter')
def run_single_parameter(pmid):
    clustering = load_clustering(pmid)
    analyzer = load_analyzer(pmid)

    # Pre-calculate all hierarchy levels before grid search to avoid re-calculation of clusterings
    ground_truth = {}
    for level in range(1, get_clustering_level(clustering)):
        ground_truth[level] = preprocess_clustering(clustering, level,
                                                    include_box_sections=False,
                                                    uniqueness_method='unique_only')
    subgraph = get_direct_references_subgraph(analyzer, pmid)
    return run_grid_search(
        analyzer, subgraph, ground_truth, metrics, param_grid, save_partition=SAVE_PARTITION
    )


def reg_v_score(labels_true, labels_pred, reg=0.01):
    """ Regularized v score """
    v_score = v_measure_score(labels_true, labels_pred)
    n_clusters = len(set(labels_pred))
    return v_score - reg * n_clusters


metrics = [adjusted_mutual_info_score, reg_v_score]

import json
import logging
import time

import pandas as pd
from celery.result import AsyncResult

if __name__ == '__main__':

    # Code to start the worker
    def run_worker():
        # Set the worker up to run in-place instead of using a pool
        celery_app.conf.CELERYD_CONCURRENCY = 32
        celery_app.conf.CELERYD_POOL = 'prefork'
        celery_app.worker_main(
            argv=['worker', '--loglevel=info', '--concurrency=32', '--without-gossip']
        )


    # Create a thread and run the workers in it
    import threading

    t = threading.Thread(target=run_worker)
    t.setDaemon(True)
    t.start()

    review_pmids = get_review_pmids()
    n_reviews = len(review_pmids)

    logger.info('Submitting all review pmid tasks')
    tasks = {}
    for i, pmid in enumerate(review_pmids):
        logger.info(f'({i + 1} / {n_reviews}) {pmid} - starting grid search')
        tasks[pmid] = run_single_parameter.delay(pmid).id

    logger.info('Waiting for tasks to finish')
    results_df = pd.DataFrame()
    partitions_overall = []

    i = 1
    while len(tasks):
        tasks_alive = {}
        for pmid, task in tasks.items():
            job = AsyncResult(task, app=celery_app)
            if job.state == 'PENDING':
                tasks_alive[pmid] = task
            elif job.state == 'FAILURE':
                print('Error', pmid, task)
            elif job.state == 'SUCCESS':
                grid_results, partitions = job.result
                grid_results_df = pd.DataFrame(grid_results)
                grid_results_df['pmid'] = pmid
                results_df = results_df.append(grid_results_df, ignore_index=True)
                partitions_overall.append({
                    'pmid': pmid,
                    'partitions': partitions
                })
                logger.info(f'Done {pmid}')

        logger.info(f'Done {len(partitions_overall)} / {n_reviews}')
        tasks = tasks_alive
        time.sleep(60)
    logger.info('All tasks are finished')

    results_df.fillna(0, inplace=True)
    results_df.to_csv(f'{OUTPUT_NAME}.csv', index=False)

    with open(f'{OUTPUT_NAME}.json', 'w') as f:
        json.dump(partitions_overall, f)
