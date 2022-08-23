# PubTrends clustering analysis based on Nature Reviews papers

This is the source code for the [Open Access Paper](https://doi.org/10.1145/3459930.3469501). \
Shpynov, O. and Nikolai, K., 2021, August. PubTrends: a scientific literature explorer. In Proceedings of the 12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (pp. 1-1).


## Results

We analyzed 40 papers of 5 different fields, for each paper we prepared 3 clusters levels according to the review paper structure. Average number of clusters for each level is shown on the ![figure 1](results/grid_search_2021_11_02_ground_truth_clusters.png?raw=true "Figure1. Nature Review clusters")

The following clustering algorithms were evaluated:
* LDA (Latent Dirichlet Allocation)
* Louvain communities detection algorithm, followed by merging tiny clusters
* Hierarchical clustering of word2vec based embeddings for citation graph and texts
* DBScan of embeddings, followed by merging tiny clusters

We use AMI (Adjusted Mutual Information) as a quality metrics for comparison with ground truth clusters.
Average number of clusters and AMI for best params are shown on the figures:

![figure 2](results/grid_search_2021_11_02_best_clusters_number.png?raw=true "Figure2. Average number of clusters")
![figure 3](results/grid_search_2021_11_02_best_adjusted_mutual_information.png?raw=true "Figure3. Average AMI")

Best parameters citations graph parameters for Louvain, Hierarchical and DBScan methods are shown on the figure 4:
![figure4](results/grid_search_2021_11_02_params.png?raw=true "Figure4. Best citations graph params")

See [Jupyter notebook](src/grid_search_optimization.ipynb) and [results](results) folder for details.

## How To reproduce 

1. The source code depends on the main [pubtrends](https://github.com/JetBrains-Research/pubtrends) repo. \
   Please clone this repository under pubtrends root first.
2. Use `src/collect_papers.py` to create database of papers. 
3. See ground truth clustering collection section.
4. Launch `src/bulk_export.py` to analyze selected papers.
5. Launch `src/grid_search.py` to launch grid search analysis.
6. Launch `src/grid_search_optimization.ipynb` for results visualization.

## Ground Truth Clustering Collection

**Current status: 40/40 papers processed, 99.9% references mapped (other seem to be erroneous)**

### Validation

1. Grouped references file should be valid JSONs.
2. Grouped references files should not contain nulls, reference IDs should be unique.
3. Fix inconsistent use of tabs and spaces.
4. References should be validated.
5. Paragraph structure (hierarchy) should be validated somehow (maybe via Nature's website?).

### Description

This folder contains:
* `clustering/` - the final result of preprocessing for each paper
* `grouped_refs_validated/` - hand-curated mapping of references by section of the paper
* `refs_validated/` - files with references that were originally extracted by Grobid and later partially fixed to get better mapping with Pubmed titles
* `refs_selected/` - files with references that are have got a special mention by review authors (potential key papers)
 
Important things:
 * clustering contains PMIDs of references grouped by paper sections in the exact order, which means:
   * PMIDs within a cluster may repeat
   * one PMID may occur in several clusters
 * there are special kinds of sections, which can be treated in a different way, see titles of these sections below:
   * INTRODUCTION
   * CONCLUSION
   * PERSPECTIVES
   * Box N | Title goes here
 * sections without references are deleted from the markup, same for figure captions
 * several papers might contain clustering-related info in tables, not processed currently:
   * 29147025 - tables 1 and 2 with references
   * 27677859 - table may contain a good clustering (p. 46)
   * 27677860 - structured conclusion
   * 29213134 - interesting review about attention
   * 26667849 - timeline of key findings about DNA with references
   * 27890914, 28920587, 30467385, 30679807, 30842595, 31806885, 32005979, 32020081 - another structured table
   * 31937935 - systematic!
 * not all references are currently mapped due to Grobid parsing errors, hopefully, will be fixed in the near future, details below:

	```
	26580716: [100%] 152 / 152 references mapped
	26580717: [100%] 91 / 91 references mapped
	26656254: [100%] 160 / 160 references mapped
	26667849: [100%] 99 / 99 references mapped
	26675821: [100%] 123 / 123 references mapped
	26678314: [100%] 198 / 198 references mapped
	26688349: [100%] 106 / 106 references mapped
	26688350: [100%] 105 / 105 references mapped
	27677859: [100%] 111 / 111 references mapped
	27677860: [100%] 178 / 178 references mapped
	27834397: [100%] 200 / 200 references mapped
	27834398: [100%] 240 / 240 references mapped
	27890914: [100%] 254 / 254 references mapped
	27904142: 100 / 101 references mapped
	--> Not a reference: regional heterogeneity in the response of astrocytes following traumatic brain injury in the adult rat
	27916977: [100%] 106 / 106 references mapped
	28003656: [100%] 196 / 196 references mapped
	28792006: [100%] 126 / 126 references mapped
	28852220: [100%] 137 / 137 references mapped
	28853444: [100%] 89 / 89 references mapped
	28920587: [100%] 188 / 188 references mapped
	29147025: [100%] 172 / 172 references mapped
	29170536: [100%] 161 / 161 references mapped
	29213134: [100%] 151 / 151 references mapped
	29321682: [100%] 185 / 185 references mapped
	30108335: 276 / 277 references mapped
	--> Not a reference: hepatitis c virus replication-specific inhibition of microrna activity with self-cleavable allosteric ribozyme
	30390028: [100%] 162 / 162 references mapped
	30459365: [100%] 333 / 333 references mapped
	30467385: [100%] 177 / 177 references mapped
	30578414: [100%] 127 / 127 references mapped
	30644449: 163 / 164 references mapped
	--> Wrong reference: pillars article: approaching the asymptote? evolution and revolution in immunology. cold spring harb symp quant biol. 1989. 54: 1-13
	30679807: [100%] 157 / 157 references mapped
	30842595: [100%] 209 / 209 references mapped
	31686003: [100%] 195 / 195 references mapped
	31806885: [100%] 203 / 203 references mapped
	31836872: [100%] 79 / 79 references mapped
	31937935: [100%] 279 / 279 references mapped
	32005979: [100%] 139 / 139 references mapped
	32020081: [100%] 178 / 178 references mapped
	32042144: 202 / 204 references mapped
	--> Not a reference: interregional synaptic maps among engram cells underlie memory formation
	--> Not a reference: memory-associated dynamic regulation of the "stable" core of the chromatin particle
	32699292: [100%] 153 / 153 references mapped
	```
  
  * other thoughts:
    * 26667849 - timeline of key findings about DNA with references, articles like these might serve as a ground truth for topic evolution analysis
    * 31937935 - very systematic!
