import gzip
import json
import os

from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
CLUSTERING_FOLDER = os.path.normpath(os.path.join(FILE_DIR, '../../clustering'))
PUBTRENDS_EXPORT_FOLDER = os.path.normpath(os.path.join(FILE_DIR, '../../pubtrends-export'))
PUBTRENDS_CONFIG = PubtrendsConfig(test=False)


def reload_exported_analyzer(path_to_archive, source='Pubmed'):
    """
    Load analysis data from json.gz archive generated by PubTrends.
    Restores missing derivable fields.
    """
    with gzip.open(path_to_archive, 'rt', encoding='UTF-8') as zipfile:
        data = json.load(zipfile)

    loader, url_prefix = Loaders.get_loader_and_url_prefix(source, PUBTRENDS_CONFIG)
    analyzer = PapersAnalyzer(loader, PUBTRENDS_CONFIG)
    analyzer.init(data)

    analyzer.ids = set(analyzer.df['id'])
    analyzer.n_papers = len(analyzer.ids)
    analyzer.pub_types = list(set(analyzer.df['type']))
    analyzer.query = 'restored from PubTrends export'

    analyzer.components = set(analyzer.df['comp'].unique())
    if -1 in analyzer.components:
        analyzer.components.remove(-1)

    return analyzer


def get_review_pmids(folder=CLUSTERING_FOLDER):
    return sorted([f.split('.')[0] for f in os.listdir(folder)])


def load_analyzer(pmid, folder=PUBTRENDS_EXPORT_FOLDER):
    analyzer_file = os.path.join(folder, f'{pmid}.json.gz')
    return reload_exported_analyzer(analyzer_file)


def load_clustering(pmid, folder=CLUSTERING_FOLDER):
    clustering_file = os.path.join(folder, f'{pmid}.json')
    with open(clustering_file, 'r') as f:
        clustering = json.load(f)
    return clustering
