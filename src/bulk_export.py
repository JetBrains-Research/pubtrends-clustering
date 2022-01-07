import gzip
import json
import logging
import os

from pysrc.papers.analysis.expand import expand_ids
from pysrc.papers.analyzer import PapersAnalyzer
from pysrc.papers.config import PubtrendsConfig
from pysrc.papers.db.loaders import Loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
TARGET_FOLDER = os.path.normpath(os.path.join(FILE_DIR, '../pubtrends-export/'))

TARGET_PMIDS = [26667849, 26678314, 26688349, 26688350, 26580716, 26580717, 26656254, 26675821, 27834397, 27834398,
                27890914, 27916977, 27677859, 27677860, 27904142, 28003656, 29147025, 29170536, 28853444, 28920587,
                28792006, 28852220, 29213134, 29321682, 30578414, 30842595, 30644449, 30679807, 30108335, 30390028,
                30459365, 30467385, 31686003, 31806885, 31836872, 32005979, 31937935, 32020081, 32042144, 32699292]

SOURCE = 'Pubmed'
LIMIT = 1000


def export_analysis(pmid):
    logging.info(f'Started analysis for PMID {pmid}')

    ids = [pmid]
    query = f'Paper ID: {pmid}'

    # extracted from 'analyze_id_list' Celery task
    config = PubtrendsConfig(test=False)
    loader = Loaders.get_loader(SOURCE, config)
    analyzer = PapersAnalyzer(loader, config)
    try:
        ids = expand_ids(loader=loader, ids=ids, single_paper=True, limit=LIMIT, max_expand=PapersAnalyzer.EXPAND_LIMIT,
                         citations_q_low=PapersAnalyzer.EXPAND_CITATIONS_Q_LOW,
                         citations_q_high=PapersAnalyzer.EXPAND_CITATIONS_Q_HIGH,
                         citations_sigma=PapersAnalyzer.EXPAND_CITATIONS_SIGMA,
                         similarity_threshold=PapersAnalyzer.EXPAND_SIMILARITY_THRESHOLD,
                         single_paper_impact=PapersAnalyzer.SINGLE_PAPER_IMPACT)

        analyzer.analyze_papers(ids, query, task=None)
    finally:
        loader.close_connection()

    dump = analyzer.dump()

    # export as JSON
    path = os.path.join(TARGET_FOLDER, f'{pmid}.json.gz')
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dump).encode('utf-8'))

    logging.info(f'Finished analysis for PMID {pmid}\n')


def main():
    for pmid in TARGET_PMIDS:
        export_analysis(pmid)


if __name__ == "__main__":
    main()
