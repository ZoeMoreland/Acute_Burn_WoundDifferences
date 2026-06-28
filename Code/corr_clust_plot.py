import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""
This script processes clustered wound-healing gene expression datasets and
creates cleaned cluster-expression tables, overlap summaries, and comparison
plots.

Steps:
1. Load cluster assignment files for the incision/acute, burn, and acute
   NanoString datasets.

2. Load the corresponding gene expression datasets.

3. Print basic dataset summaries, including:
   - number of genes in each expression dataset
   - number of clusters in each dataset
   - number of genes assigned to each cluster
   - number of genes shared between acute/burn and acute2/burn datasets

4. Normalize the expression data:
   - burn RNA-seq data is converted from counts to CPM, log2(CPM + 1),
     then normalized relative to the average Day 0 expression
   - acute Affymetrix data is already log2/RMA-normalized, so it is only
     normalized relative to the average Day 0 expression
   - acute NanoString data is currently kept as-is unless additional
     normalization is needed

5. Average replicate samples within each day so each dataset has one column
   per timepoint.

6. Join cluster labels with the day-averaged expression data to create
   clustered expression tables.

7. Save the cleaned clustered day-averaged datasets.

8. Create and save an acute-vs-burn cluster overlap table showing how many
   genes are shared between each pair of clusters.

9. Identify shared genes between selected acute and burn cluster pairs.

10. Generate stacked trajectory plots comparing selected acute and burn
    clusters, with each cluster shown in its own panel and its mean
    trajectory highlighted.
"""

# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "figure.dpi": 100,
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})


# ============================================================
# File paths
# ============================================================

ACUTE_CLUSTER_PATH = "Data/cluster/acute_clustered_reduced_again.csv"
BURN_CLUSTER_PATH = "Data/cluster/burn_corr_cluster.csv"

ACUTE_EXPR_PATH = "Data/PreProcessing_Data/fc_acute.csv"
BURN_EXPR_PATH = "Data/PreProcessing_Data/fc_burn.csv"

OUTPUT_DIR = "Data/cluster"
PLOT_DIR = "Plots/Correlation/Selected Acute vs Burn Clusters"


# ============================================================
# Helper functions
# ============================================================

def load_cluster_file(path, sep=","):
    """
    Load a cluster file and keep only the Cluster column.
    """
    df = pd.read_csv(path, sep=sep, index_col=0)

    if "Cluster" not in df.columns:
        raise ValueError(f"'Cluster' column not found in {path}")

    return df[["Cluster"]]


def load_expression_file(path, sep="\t"):
    """
    Load expression data with genes as the index.
    """
    return pd.read_csv(path, sep=sep, index_col=0)


def cpm_log2(count_df):
    """
    Convert raw counts to log2(CPM + 1).
    Use this for RNA-seq count data, such as burn.
    """
    library_sizes = count_df.sum(axis=0)
    cpm = count_df.divide(library_sizes, axis=1) * 1_000_000
    return np.log2(cpm + 1)


def normalize_to_day0(log_df):
    """
    Subtract each gene's average Day 0 expression from all timepoints.
    Returns log2 fold-change relative to Day 0.
    """
    day0_cols = [c for c in log_df.columns if str(c).startswith("0_")]

    if len(day0_cols) == 0:
        raise ValueError("No Day 0 columns found. Expected columns like '0_1', '0_2'.")

    day0_mean = log_df[day0_cols].mean(axis=1)
    return log_df.subtract(day0_mean, axis=0)


def average_by_day(df):
    """
    Average replicate columns by day.

    Example:
        0_1, 0_2 -> 0
        1_1, 1_2 -> 1
        3_1, 3_2 -> 3
    """
    day_cols = {}

    for col in df.columns:
        day = float(str(col).split("_")[0])
        day_cols.setdefault(day, []).append(col)

    averaged_df = pd.DataFrame({
        day: df[cols].mean(axis=1)
        for day, cols in day_cols.items()
    })

    return averaged_df.reindex(sorted(averaged_df.columns), axis=1)


def join_clusters(cluster_df, expr_df):
    """
    Join Cluster column to expression data.
    """
    joined = cluster_df.join(expr_df, how="inner")
    return joined.sort_values("Cluster")


def print_dataset_summary(name, cluster_df, expr_df):
    """
    Print basic dataset information.
    """
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Number of genes in expression data: {expr_df.shape[0]}")
    print(f"Number of clusters: {cluster_df['Cluster'].nunique()}")

    print("\nGenes per cluster:")
    print(cluster_df["Cluster"].value_counts().sort_index())


def get_gene_overlap(df1, df2):
    """
    Return overlapping gene names between two dataframes.
    """
    return set(df1.index).intersection(set(df2.index))


def make_cluster_overlap_table(df1, df2, df1_name="Acute", df2_name="Burn"):
    """
    Create a table showing how many genes overlap between clusters.

    Rows = df2 clusters
    Columns = df1 clusters
    """
    df1_clusters = sorted(df1["Cluster"].unique())
    df2_clusters = sorted(df2["Cluster"].unique())

    overlap_table = pd.DataFrame(
        0,
        index=[f"{df2_name} Cluster {c}" for c in df2_clusters],
        columns=[f"{df1_name} Cluster {c}" for c in df1_clusters]
    )

    for c1 in df1_clusters:
        df1_genes = set(df1[df1["Cluster"] == c1].index)

        for c2 in df2_clusters:
            df2_genes = set(df2[df2["Cluster"] == c2].index)
            overlap_table.loc[
                f"{df2_name} Cluster {c2}",
                f"{df1_name} Cluster {c1}"
            ] = len(df1_genes.intersection(df2_genes))

    return overlap_table


def get_shared_genes(df1, df2, cluster1, cluster2):
    """
    Return genes shared between one cluster from df1 and one cluster from df2.
    """
    genes1 = set(df1[df1["Cluster"] == cluster1].index)
    genes2 = set(df2[df2["Cluster"] == cluster2].index)

    shared_genes = sorted(genes1.intersection(genes2))

    print(f"\nShared genes between Cluster {cluster1} and Cluster {cluster2}:")
    print(f"{len(shared_genes)} genes")
    print(shared_genes)

    return shared_genes


def extract_day(col):
    """
    Convert timepoint column names into numeric day values.

    Works for:
        0, 0.25, 0.5, 1, 3
        "0", "0.25", "0.5", "1", "3"
        "0_1", "0.25_2", "3_1"
        "Day 0", "Day 0.25"
    """
    col = str(col)
    col = col.replace("Day", "").strip()

    if "_" in col:
        col = col.split("_")[0]

    return float(col)


def plot_cluster_with_mean(
    df,
    cluster_number,
    dataset_name="Dataset",
    save_path=None
):
    """
    Plot every gene in a cluster with the cluster mean highlighted.
    """
    cluster_df = df[df["Cluster"] == cluster_number].copy()
    time_cols = [col for col in cluster_df.columns if col != "Cluster"]

    x = [extract_day(col) for col in time_cols]
    values = cluster_df[time_cols].astype(float)

    plt.figure(figsize=(10, 6))

    for _, row in values.iterrows():
        plt.plot(x, row.values, alpha=0.2)

    mean_curve = values.mean(axis=0)
    plt.plot(x, mean_curve.values, linewidth=3, label="Cluster mean")

    plt.xticks(x, time_cols, rotation=45)
    plt.title(f"{dataset_name} Cluster {cluster_number} (n={len(cluster_df)})")
    plt.xlabel("Day")
    plt.ylabel("log2 fold-change relative to Day 0")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cluster_pair(
    acute_df,
    acute_cluster_number,
    burn_df,
    burn_cluster_number,
    acute_name="Incision",
    burn_name="Burn",
    save_path=None
):
    """
    Plot one acute/incision cluster and one burn cluster as stacked plots.
    Each plot has its own y-axis.
    """
    acute_cluster_df = acute_df[acute_df["Cluster"] == acute_cluster_number].copy()
    burn_cluster_df = burn_df[burn_df["Cluster"] == burn_cluster_number].copy()

    acute_time_cols = [col for col in acute_cluster_df.columns if col != "Cluster"]
    burn_time_cols = [col for col in burn_cluster_df.columns if col != "Cluster"]

    x_acute = [extract_day(col) for col in acute_time_cols]
    x_burn = [extract_day(col) for col in burn_time_cols]

    acute_values = acute_cluster_df[acute_time_cols].astype(float)
    burn_values = burn_cluster_df[burn_time_cols].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    fig.suptitle(
        f"{acute_name} Cluster {acute_cluster_number} vs "
        f"{burn_name} Cluster {burn_cluster_number}",
        fontsize=16,
        y=1.02
    )

    # ---------------- Acute cluster ----------------
    ax = axes[0]

    for _, row in acute_values.iterrows():
        ax.plot(x_acute, row.values, color="gray", alpha=0.25, linewidth=1)

    acute_mean = acute_values.mean(axis=0)
    ax.plot(x_acute, acute_mean.values, color="black", linewidth=3, label="Mean trajectory")

    ax.set_title(f"{acute_name} Cluster {acute_cluster_number} (n={acute_values.shape[0]} genes)")
    ax.set_xticks(x_acute)
    ax.set_xticklabels(acute_time_cols, rotation=45)
    ax.set_ylabel("Gene Expression")
    ax.legend(fontsize=9, loc="best")

    # ---------------- Burn cluster ----------------
    ax = axes[1]

    for _, row in burn_values.iterrows():
        ax.plot(x_burn, row.values, color="gray", alpha=0.25, linewidth=1)

    burn_mean = burn_values.mean(axis=0)
    ax.plot(x_burn, burn_mean.values, color="black", linewidth=3, label="Mean trajectory")

    ax.set_title(f"{burn_name} Cluster {burn_cluster_number} (n={burn_values.shape[0]} genes)")
    ax.set_xticks(x_burn)
    ax.set_xticklabels(burn_time_cols, rotation=45)
    ax.set_xlabel("Time in Days")
    ax.set_ylabel("Gene Expression")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cluster_pairs(acute_clustered, burn_clustered, cluster_pairs):
    """
    Plot multiple acute/burn cluster pairs.
    """
    for acute_cluster_num, burn_cluster_num in cluster_pairs:
        save_path = (
            f"{PLOT_DIR}/"
            f"acute{acute_cluster_num}_burn{burn_cluster_num}_comparison.png"
        )

        plot_cluster_pair(
            acute_df=acute_clustered,
            acute_cluster_number=acute_cluster_num,
            burn_df=burn_clustered,
            burn_cluster_number=burn_cluster_num,
            save_path=save_path
        )


# ============================================================
# Main workflow
# ============================================================

# ---------------- Load cluster files ----------------

acute_cluster = load_cluster_file(ACUTE_CLUSTER_PATH, sep="\t")
burn_cluster = load_cluster_file(BURN_CLUSTER_PATH, sep=",")

# ---------------- Load expression files ----------------

acute_raw = load_expression_file(ACUTE_EXPR_PATH, sep="\t")
burn_raw = load_expression_file(BURN_EXPR_PATH, sep="\t")

# ---------------- Print summaries before processing ----------------

print_dataset_summary("Acute GSE23006", acute_cluster, acute_raw)
print_dataset_summary("Burn RNA-seq", burn_cluster, burn_raw)

# ---------------- Gene overlaps before processing ----------------

acute_burn_overlap = get_gene_overlap(acute_raw, burn_raw)

print(f"\nGenes shared between acute and burn: {len(acute_burn_overlap)}")

# ---------------- Normalize expression ----------------

# Burn RNA-seq: counts -> CPM -> log2(CPM + 1) -> Day 0 centered
burn_log = cpm_log2(burn_raw)
burn_log2fc = normalize_to_day0(burn_log)

# Acute GSE32006: already RMA-normalized and log2 -> Day 0 centered
acute_log2fc = normalize_to_day0(acute_raw)



# ---------------- Average replicates by day ----------------

acute_day_avg = average_by_day(acute_log2fc)
burn_day_avg = average_by_day(burn_log2fc)

# ---------------- Join clusters to day-averaged expression ----------------

acute_clustered = join_clusters(acute_cluster, acute_day_avg)
burn_clustered = join_clusters(burn_cluster, burn_day_avg)

# ---------------- Save cleaned clustered datasets ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

acute_clustered.to_csv(f"{OUTPUT_DIR}/acute_clustered_dayAvg.csv", sep="\t")
burn_clustered.to_csv(f"{OUTPUT_DIR}/burn_clustered_dayAvg.csv", sep="\t")

print("\nSaved clustered day-averaged datasets.")

# ---------------- Make overlap table ----------------

overlap_table = make_cluster_overlap_table(
    acute_clustered,
    burn_clustered,
    df1_name="Acute",
    df2_name="Burn"
)

overlap_table.to_csv(f"{OUTPUT_DIR}/acute_burn_cluster_overlap_table.csv")
print("\nSaved acute/burn cluster overlap table.")
print(overlap_table)

# ---------------- Shared genes example ----------------

shared_genes = get_shared_genes(
    acute_clustered,
    burn_clustered,
    cluster1=3,
    cluster2=1
)

# ---------------- Plot selected cluster pairs ----------------

cluster_pairs = [
    (2, 8),
    (3, 2),
    (3, 5),
    (4, 2),
    (4, 5),
    (5, 4),
    (6, 4),
    (6, 6),
    (8, 1),
    (8, 8),
    (9, 2),
    (9, 5),
    (9, 6),
    (10, 8),
    (11, 8),
    (12, 4),
    (12, 6),
    (13, 2),
    (13, 5),
]

plot_cluster_pairs(acute_clustered, burn_clustered, cluster_pairs)