"""
Cluster acute and burn wound-healing datasets using correlation-based clustering.

This script:
1. Loads fold-change filtered acute and burn datasets
2. Removes Day 0 columns before clustering
3. Clusters acute and burn genes by correlation
4. Keeps clusters with at least 10 genes
5. Saves cluster assignment files
6. Prints shared-gene overlap for:
   - acute vs burn
"""

import numpy as np
import pandas as pd


# -----------------------------
# File paths
# -----------------------------
ACUTE_PATH = "Data/PreProcessing_Data/fc_acute.csv"
BURN_PATH = "Data/PreProcessing_Data/fc_burn.csv"

ACUTE_CLUSTER_OUTPUT = "Data/cluster/acute_corr_cluster.csv"
ACUTE_CLUSTER_EXPR_OUTPUT = "Data/cluster/acute_corr_cluster_with_expression.csv"
ACUTE_REP_AVG_OUTPUT = "Data/cluster/acute_cluster_replicateAvg.csv"
ACUTE_CLUSTER_REP_AVG_OUTPUT = "Data/cluster/acute_clusterAvg_replicateAvg.csv"


BURN_CLUSTER_OUTPUT = "Data/cluster/burn_corr_cluster.csv"
BURN_CLUSTER_EXPR_OUTPUT = "Data/cluster/burn_corr_cluster_with_expression.csv"
BURN_REP_AVG_OUTPUT = "Data/cluster/burn_cluster_replicateAvg.csv"
BURN_CLUSTER_REP_AVG_OUTPUT = "Data/cluster/burn_clusterAvg_replicateAvg.csv"



# -----------------------------
# Settings
# -----------------------------
ACUTE_SEED_THRESHOLD = 0.75

BURN_SEED_THRESHOLD = 0.70

MEAN_CORR_THRESHOLD = 0.50
MIN_CLUSTER_SIZE = 10
METHOD = "pearson"
FDR_THRESHOLD = 0.05


# -----------------------------
# Data loading
# -----------------------------
def load_datasets():
    """Load acute and burn datasets."""

    acute_df = pd.read_csv(ACUTE_PATH, sep="\t", index_col=0)
    burn_df = pd.read_csv(BURN_PATH, sep="\t", index_col=0)

    return acute_df, burn_df



# -----------------------------
# Correlation clustering
# -----------------------------
def correlation_clustering(
    df,
    seed_threshold,
    mean_corr_threshold=MEAN_CORR_THRESHOLD,
    method=METHOD,
    min_cluster_size=2,
    verbose=True
):
    """
    Cluster genes based on:
    1. Correlation to a seed gene
    2. Mean pairwise correlation within the proposed cluster
    """

    remaining = df.copy()
    clusters = []
    cluster_stats = []

    while len(remaining) > 0:
        seed_gene = remaining.index[0]
        seed_profile = remaining.loc[seed_gene]

        if method == "pearson":
            seed_corr = remaining.apply(
                lambda row: row.corr(seed_profile),
                axis=1
            )
        elif method == "spearman":
            seed_corr = remaining.apply(
                lambda row: row.corr(seed_profile, method="spearman"),
                axis=1
            )
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        proposed_genes = seed_corr[seed_corr >= seed_threshold].index.tolist()

        if len(proposed_genes) < min_cluster_size:
            clusters.append(proposed_genes)

            cluster_stats.append({
                "cluster_id": len(clusters),
                "seed_gene": seed_gene,
                "initial_size": len(proposed_genes),
                "final_size": len(proposed_genes),
                "mean_pairwise_corr": np.nan
            })

            remaining = remaining.drop(proposed_genes)

            if verbose:
                print(
                    f"Cluster {len(clusters)}: {len(proposed_genes)} gene(s) "
                    f"(seed = {seed_gene}, no mean-corr filtering applied)"
                )

            continue

        cluster_df = remaining.loc[proposed_genes]
        corr_matrix = cluster_df.T.corr(method=method)
        corr_matrix = corr_matrix.mask(
            np.eye(corr_matrix.shape[0], dtype=bool)
        )

        mean_corr_per_gene = corr_matrix.mean(axis=1)

        filtered_genes = mean_corr_per_gene[
            mean_corr_per_gene >= mean_corr_threshold
        ].index.tolist()

        if len(filtered_genes) == 0:
            filtered_genes = [seed_gene]

        final_mean_pairwise = np.nan

        if len(filtered_genes) >= 2:
            final_cluster_df = remaining.loc[filtered_genes]
            final_corr_matrix = final_cluster_df.T.corr(method=method)

            final_corr_matrix = final_corr_matrix.mask(
                np.eye(final_corr_matrix.shape[0], dtype=bool)
            )

            final_mean_pairwise = final_corr_matrix.stack().mean()

        clusters.append(filtered_genes)

        cluster_stats.append({
            "cluster_id": len(clusters),
            "seed_gene": seed_gene,
            "initial_size": len(proposed_genes),
            "final_size": len(filtered_genes),
            "mean_pairwise_corr": final_mean_pairwise
        })

        remaining = remaining.drop(filtered_genes)

        if verbose:
            print(
                f"Cluster {len(clusters)}: "
                f"{len(filtered_genes)} genes "
                f"(seed = {seed_gene}, initial = {len(proposed_genes)})"
            )

    return clusters, pd.DataFrame(cluster_stats)


# -----------------------------
# Cluster utilities
# -----------------------------
def remove_day0_columns(df):
    """Remove Day 0 columns before clustering."""

    return df[
        [col for col in df.columns if not str(col).startswith("0")]
    ].copy()


def filter_clusters_by_size(clusters, min_size=MIN_CLUSTER_SIZE):
    """Keep only clusters with at least min_size genes."""

    return [cluster for cluster in clusters if len(cluster) >= min_size]


def assignments_df(clusters):
    """Create a dataframe mapping each gene to its cluster label."""

    records = []

    for cluster_id, cluster in enumerate(clusters, start=1):
        for gene in cluster:
            records.append((gene, cluster_id))

    df = pd.DataFrame(records, columns=["Gene", "Cluster"])
    df = df.set_index("Gene")
    df = df.sort_values("Cluster")

    return df


def print_cluster_sizes(clusters, dataset_name):
    """Print the number of genes in each cluster."""

    print(f"\n{dataset_name} clusters:")

    for cluster_id, cluster in enumerate(clusters, start=1):
        print(f"Cluster {cluster_id}: {len(cluster)} genes")


def print_shared_gene_summary(reference_clusters, burn_clusters, reference_name):
    """Print shared-gene overlap between a reference dataset and burn."""

    reference_genes = set(
        gene for cluster in reference_clusters for gene in cluster
    )

    burn_genes = set(
        gene for cluster in burn_clusters for gene in cluster
    )

    shared_genes = reference_genes.intersection(burn_genes)
    reference_only = reference_genes - burn_genes
    burn_only = burn_genes - reference_genes

    print(f"\n{reference_name} vs Burn shared-gene summary")
    print(f"Total {reference_name} genes: {len(reference_genes)}")
    print(f"Total burn genes: {len(burn_genes)}")
    print(f"Shared genes: {len(shared_genes)}")
    print(f"{reference_name}-only genes: {len(reference_only)}")
    print(f"Burn-only genes: {len(burn_only)}")

    print(f"\nShared genes per cluster pair: {reference_name} vs Burn")

    for i, reference_cluster in enumerate(reference_clusters, start=1):
        reference_set = set(reference_cluster)

        for j, burn_cluster in enumerate(burn_clusters, start=1):
            burn_set = set(burn_cluster)
            overlap = reference_set.intersection(burn_set)

            if len(overlap) > 0:
                print(
                    f"{reference_name} Cluster {i} & Burn Cluster {j}: "
                    f"{len(overlap)} shared genes"
                )


def cluster_dataset(df, dataset_name, seed_threshold):
    """Remove Day 0 columns, cluster genes, and filter small clusters."""

    print(f"\nClustering {dataset_name}")

    cluster_df = remove_day0_columns(df)

    clusters, cluster_stats = correlation_clustering(
        cluster_df,
        seed_threshold=seed_threshold,
        mean_corr_threshold=MEAN_CORR_THRESHOLD,
        method=METHOD
    )

    clusters_filtered = filter_clusters_by_size(
        clusters,
        min_size=MIN_CLUSTER_SIZE
    )

    print(f"Total clusters {dataset_name}: {len(clusters)}")
    print(
        f"Clusters with >= {MIN_CLUSTER_SIZE} genes - "
        f"{dataset_name}: {len(clusters_filtered)}"
    )

    print_cluster_sizes(clusters_filtered, dataset_name)

    return clusters_filtered, cluster_stats



def clustered_expression_df(original_df, clusters):
    """
    Create a dataframe that includes:
    - Gene
    - Cluster
    - all original expression data for that gene
    """

    assign_df = assignments_df(clusters)

    clustered_df = original_df.loc[assign_df.index].copy()
    clustered_df.insert(0, "Cluster", assign_df["Cluster"])

    clustered_df = clustered_df.sort_values("Cluster")

    return clustered_df

# average replicates
def average_replicates(df):
    """Average replicate columns for each gene."""

    averaged_df = df.copy()
    averaged_df = averaged_df.apply(pd.to_numeric, errors="coerce")

    timepoint_groups = {}

    for col in df.columns:
        if "_" in str(col):
            timepoint = str(col).split("_")[0]

            if timepoint not in timepoint_groups:
                timepoint_groups[timepoint] = []

            timepoint_groups[timepoint].append(col)

    for timepoint, cols in timepoint_groups.items():
        averaged_df[timepoint] = averaged_df[cols].mean(axis=1)

    return averaged_df[[col for col in averaged_df.columns if "_" not in str(col)]]


# average cluster expression
def average_cluster_expression(clustered_df):
    """Average expression of all genes in each cluster at each timepoint."""

    cluster_expr = clustered_df.copy()
    cluster_expr = cluster_expr.apply(pd.to_numeric, errors="coerce")

    cluster_groups = cluster_expr.groupby("Cluster")

    averaged_cluster_expr = cluster_groups.mean()

    return averaged_cluster_expr

# -----------------------------
# Main workflow
# -----------------------------
def main():
    """Run significance testing and correlation clustering."""

    acute_df, burn_df = load_datasets()

    print("Original acute shape:", acute_df.shape)
    print("Original burn shape:", burn_df.shape)

    acute_clusters, acute_cluster_stats = cluster_dataset(
        acute_df,
        dataset_name="Acute",
        seed_threshold=ACUTE_SEED_THRESHOLD
    )


    burn_clusters, burn_cluster_stats = cluster_dataset(
        burn_df,
        dataset_name="Burn",
        seed_threshold=BURN_SEED_THRESHOLD
    )

    acute_assign = assignments_df(acute_clusters)
    burn_assign = assignments_df(burn_clusters)

    acute_assign.to_csv(ACUTE_CLUSTER_OUTPUT)
    burn_assign.to_csv(BURN_CLUSTER_OUTPUT)
    
    
    
    acute_cluster_expr = clustered_expression_df(acute_df, acute_clusters)
    burn_cluster_expr = clustered_expression_df(burn_df, burn_clusters)

    acute_cluster_expr.to_csv(ACUTE_CLUSTER_EXPR_OUTPUT)
    burn_cluster_expr.to_csv(BURN_CLUSTER_EXPR_OUTPUT)


    acute_rep_avg = average_replicates(acute_cluster_expr)
    burn_rep_avg = average_replicates(burn_cluster_expr)

    acute_rep_avg.to_csv(ACUTE_REP_AVG_OUTPUT)
    burn_rep_avg.to_csv(BURN_REP_AVG_OUTPUT)
    
    acute_cluster_rep_avg = average_cluster_expression(acute_rep_avg)
    burn_cluster_rep_avg = average_cluster_expression(burn_rep_avg)

    acute_cluster_rep_avg.to_csv(ACUTE_CLUSTER_REP_AVG_OUTPUT)
    burn_cluster_rep_avg.to_csv(BURN_CLUSTER_REP_AVG_OUTPUT)

    print("\nCluster assignment files saved.")

    print_shared_gene_summary(
        reference_clusters=acute_clusters,
        burn_clusters=burn_clusters,
        reference_name="Acute"
    )


if __name__ == "__main__":
    main()