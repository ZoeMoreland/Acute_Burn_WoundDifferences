# Computational Analysis of Transcriptomic Incision and Burn Wounds Project

This repository contains Python scripts for preprocessing, filtering, clustering, comparing, and biologically interpreting transcriptomic wound healing datasets. The workflow compares incision and burn healing using fold-change filtering, correlation clustering, dynamic time warping, and gene ontology enrichment.

## Repository Structure
The Python scripts are stored in a folder called `Code`.

```text
Code/
├── cleaning.py
├── fc.py
├── correlation_cluster.py
├── corr_clust_plot.py
├── DTW.py
└── gene_ontology_analysis.py
```

The GitHub repository also includes four data files:
1. Two clustered gene expression files that have been averaged by day.
2. Two Gene Ontology enrichment data tables generated from the clusters in `gene_ontology_analysis.py`

## Project Overview
The goal of this code is to analyze wound-healing gene expression trajectories across time. The scripts clean raw incision and burn datasets, selects highly variable genes, cluster genes with similar temporal expression patterns, compare clusters between datasts, and assign biological meaning to clusters using Gene Ontology Biological Process enrichment.

The main workflow is:
1. Clean and preprocess raw datasets.
2. Select the top 10% most variable genes by fold change.
3. Cluster genes using correlation-based clustering.
4. Normalize and average replicate expression values by day.
5. Compare clusters across datasets using gene overlap and trajectory plots.
6. se Dynamic Time Warping to compare cluster expression trajectories.
7. Perform Gene Ontology enrichment and wound-healing phase scoring.


## File Descriptions

### `Code/cleaning.py`
Cleans and preprocesses the incision and burn datasets.
This script:
1. Loads the raw datasets.
2. Standardizes gene names.
3. Renames burn sample columns.
4. Removes genes with inconsistent replicates.
5. Collapses repeated gene entries based on correlation.
6. Removes mostly-zero genes.
7. Detects and removes sample-level outliers.
8. Saves the cleaned datasets.

### `Code/fc.py`
Selects the top 10% most variable genes by fold change.
This script:
1. Loads the cleaned acute and burn datasets.
2. Computes fold change for each gene.
3. Keeps the top 10% of genes with the highest fold change.
4. Saves the filtered datasets.

### `Code/corrlection_cluster.py`
Clusters incision and burn datasets using correlation-based clustering.
This script:
1. Loads fold-change filtered acute and burn datasets.
2. Removes Day 0 columns before clustering.
3. Clusters acute and burn genes by correlation.
4. Keeps clusters with at least 10 genes.
5. Saves cluster assignment files.
6. Prints shared-gene overlap between acute and burn datasets.

### `Code/corr_clust_plot.py`
Processes clustered gene expression datasets and creates cleaned cluster-expression tables, overlap summaries, and comparison plots.
This script:
1. Loads cluster assignment files for the incision and burn datasets.
2. Loads the corresponding gene expression datasets.
3. Prints basic dataset summaries, including:
- number of genes in each expression dataset
- number of clusters in each dataset
- number of genes assigned to each cluster
- number of genes shared between acute/burn datasets
4. Normalizes the expression data:
- burn RNA-seq data is converted from counts to CPM, log2(CPM + 1), then normalized relative to the average Day 0 expression
- incision Affymetrix data is already log2/RMA-normalized, so it is only normalized relative to the average Day 0 expression
5. Averages replicate samples within each day so each dataset has one column per timepoint.
6. Joins cluster labels with the day-averaged expression data to create clustered expression tables.
7. Saves the cleaned clustered day-averaged datasets.
8. Creates and saves an incision-vs-burn cluster overlap table showing how many genes are shared between each pair of clusters.
9. Identifies shared genes between selected incision and burn cluster pairs.
10. Generates stacked trajectory plots comparing selected incision and burn clusters, with each cluster shown in its own panel and its mean trajectory highlighted.

### `Code/DTW.py`
Performs Dynamic Time Warping cluster trajectory analysis.
This script:
1. Loads clustered, day-averaged expression files.
2. Computes mean cluster trajectories.
3. Computes DTW distances between two datasets.
4. Saves a ranked cluster match table.
5. Plots a DTW heatmap.
6. Optionally plots DTW alignments for selected cluster pairs.

### `Code/gene_ontology_analysis.py`
Runs the Gene Ontology phase scoring and trajectory workflow.
This script:
1. Loads clustered gene expression tables for burn and incision datasets.
2. Creates cluster average expression tables by averaging genes within each cluster.
3. Runs GO Biological Process enrichment for each cluster using Enrichr through GSEApy.
4. Saves GO enrichment barplots for each dataset and cluster.
5. Uses a custom keyword dictionary to assign enriched GO terms to wound-healing phase categories.
6. Scores each cluster by wound-healing phase using both matched GO term counts and weighted scores based on -log10(adjusted p-value).
7. Saves cluster-level GO phase scores in both long and wide table formats.
8. Combines cluster average expression trajectories with GO phase scores to create phase-weighted expression trajectories over time.
9. Saves and plots GO phase-weighted trajectories for each dataset.
10. Creates a long-format acute vs burn table for downstream phase-level boxplots.


## Included Data Files
This repository includes four data files used by the later analysis scripts.

The two clustered day-averaged gene expression files are:

Data/cluster/burn_clustered_dayAvg.csv
Data/cluster/acute_clustered_dayAvg.csv

These files contain gene expression values averaged by timepoint, with cluster labels included for each gene.

The other two data files are Gene Ontology enrichment result tables generated by `Code/gene_ontology_analysis.py`. These tables contain the GO Biological Process enrichment results for the wound-healing clusters and are used for downstream GO phase scoring and biological interpretation.
