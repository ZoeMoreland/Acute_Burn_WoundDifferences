"""
Select the top 10% most variable genes by fold change.

This script:
1. Loads the cleaned acute and burn datasets
2. Computes fold change for each gene
3. Keeps the top 10% of genes with the highest fold change
4. Saves the filtered datasets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests



# -----------------------------
# File paths
# -----------------------------
ACUTE_PATH = "Data/PreProcessing_Data/GSE23006_Cleaned_data.csv"
BURN_PATH = "Data/PreProcessing_Data/burn_fem_cleaned_data.csv"

ACUTE_OUTPUT_PATH = "Data/PreProcessing_Data/fc_acute.csv"
BURN_OUTPUT_PATH = "Data/PreProcessing_Data/fc_burn.csv"


# -----------------------------
# Settings
# -----------------------------
TOP_FRAC = 0.10
EPSILON = 1e-10


# -----------------------------
# Functions
# -----------------------------
def load_datasets():
    """Load cleaned acute and burn datasets."""

    acute = pd.read_csv(ACUTE_PATH, sep="\t", index_col=0)
    burn = pd.read_csv(BURN_PATH, sep=",", index_col=0)

    return acute, burn






def get_day_from_column(col):
    """
    Extracts day from column names like:
    '0_1', '0.25_2', '1_3', '28_25'
    """
    return str(col).split("_")[0]



def counts_to_log2_cpm(count_df):
    """
    Convert raw RNA-seq counts to log2(CPM + 1).
    Rows = genes, columns = samples.
    """

    library_sizes = count_df.sum(axis=0)

    cpm = count_df.divide(library_sizes, axis=1) * 1_000_000

    log2_cpm = np.log2(cpm + 1)

    return log2_cpm


def subtract_day0_mean_per_gene(df, day0_label="0"):
    """
    Subtracts the Day 0 mean expression from every sample for each gene.

    Rows = genes
    Columns = samples
    Values = log2 expression values
    """

    df = df.copy()

    day_labels = pd.Series(df.columns, index=df.columns).apply(get_day_from_column)

    day0_cols = day_labels[day_labels == day0_label].index

    if len(day0_cols) == 0:
        raise ValueError(f"No Day {day0_label} columns found. Check your column names.")

    day0_mean = df[day0_cols].mean(axis=1)

    df_day0_normalized = df.subtract(day0_mean, axis=0)

    return df_day0_normalized

def compute_volcano_table(
    df,
    day0_label="0",
    top_frac=0.10
):
    """
    Creates a table for volcano-style plotting.

    Parameters
    ----------
    df : pandas DataFrame
        Rows = genes, columns = samples.
        Values should already be log2 expression or log2 normalized expression.

    day0_label : str
        Label for baseline day, usually "0".

    top_frac : float
        Fraction of genes to highlight, usually 0.10 for top 10%.

    Returns
    -------
    volcano_df : pandas DataFrame
        Contains max log2 fold change, ANOVA p-value, FDR, and top 10% label.
    """

    df = df.copy()

    # Get day labels from columns
    day_labels = pd.Series(df.columns, index=df.columns).apply(get_day_from_column)

    # Make sure Day 0 exists
    day0_cols = day_labels[day_labels == day0_label].index

    if len(day0_cols) == 0:
        raise ValueError(f"No Day {day0_label} columns found.")

    # Day 0 mean expression per gene
    day0_mean = df[day0_cols].mean(axis=1)

    # Calculate mean expression per day
    unique_days = sorted(day_labels.unique(), key=lambda x: float(x))

    day_means = pd.DataFrame(index=df.index)

    for day in unique_days:
        cols = day_labels[day_labels == day].index
        day_means[day] = df[cols].mean(axis=1)

    # Since data are log2, subtracting Day 0 gives log2 fold change
    log2fc_by_day = day_means.subtract(day0_mean, axis=0)

    # Maximum absolute log2 fold change across time
    max_abs_log2fc = log2fc_by_day.abs().max(axis=1)

    # Signed max log2 fold change:
    # keeps direction from the timepoint with the largest absolute change
    signed_max_log2fc = log2fc_by_day.apply(
        lambda row: row.loc[row.abs().idxmax()],
        axis=1
    )

    # ANOVA p-value across timepoints
    pvals = []

    for gene in df.index:
        groups = []
        for day in unique_days:
            cols = day_labels[day_labels == day].index
            values = df.loc[gene, cols].dropna().values

            if len(values) > 0:
                groups.append(values)

        # Need at least two groups with data
        if len(groups) >= 2:
            try:
                pval = f_oneway(*groups).pvalue
            except Exception:
                pval = np.nan
        else:
            pval = np.nan

        pvals.append(pval)

    pvals = pd.Series(pvals, index=df.index)

    # FDR adjustment
    valid = pvals.notna()

    fdr = pd.Series(np.nan, index=df.index)

    if valid.sum() > 0:
        fdr.loc[valid] = multipletests(
            pvals.loc[valid],
            method="fdr_bh"
        )[1]

    # Identify top 10% by max absolute log2 fold change
    cutoff = max_abs_log2fc.quantile(1 - top_frac)
    selected_top_genes = max_abs_log2fc >= cutoff

    volcano_df = pd.DataFrame({
        "signed_max_log2FC": signed_max_log2fc,
        "max_abs_log2FC": max_abs_log2fc,
        "p_value": pvals,
        "FDR": fdr,
        "selected_top_10_percent": selected_top_genes
    })

    # Avoid infinite values on plot
    volcano_df["minus_log10_FDR"] = -np.log10(volcano_df["FDR"].replace(0, np.nan))

    return volcano_df


def plot_volcano(
    volcano_df,
    title="Volcano plot",
    fc_cutoff=None,
    fdr_cutoff=0.05,
    save_path=None
):
    """
    Plots volcano-style figure with selected top 10% genes highlighted.
    """

    plot_df = volcano_df.dropna(subset=["signed_max_log2FC", "minus_log10_FDR"]).copy()

    selected = plot_df["selected_top_10_percent"]

    plt.figure(figsize=(7, 6))

    # Background genes
    plt.scatter(
        plot_df.loc[~selected, "signed_max_log2FC"],
        plot_df.loc[~selected, "minus_log10_FDR"],
        alpha=0.35,
        s=12,
        label="Other genes"
    )

    # Selected top 10% genes
    plt.scatter(
        plot_df.loc[selected, "signed_max_log2FC"],
        plot_df.loc[selected, "minus_log10_FDR"],
        alpha=0.75,
        s=18,
        label="Top 10% selected genes"
    )

    # FDR threshold line
    if fdr_cutoff is not None:
        plt.axhline(
            -np.log10(fdr_cutoff),
            linestyle="--",
            linewidth=1,
            label=f"FDR = {fdr_cutoff}"
        )

    # Optional fold-change cutoff lines
    if fc_cutoff is not None:
        plt.axvline(fc_cutoff, linestyle="--", linewidth=1)
        plt.axvline(-fc_cutoff, linestyle="--", linewidth=1)

    plt.xlabel("Signed maximum log2 fold change from Day 0")
    plt.ylabel("-log10(FDR)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    
def plot_volcano_with_fc_threshold(
    volcano_df,
    title="Volcano-style selection plot",
    fdr_cutoff=0.05,
    show_fdr_line=True,
    save_path=None
):
    """
    Volcano-style plot highlighting genes selected by top 10% absolute fold-change.
    Adds vertical threshold lines showing the fold-change cutoff used for selection.
    """

    plot_df = volcano_df.dropna(
        subset=["signed_max_log2FC", "minus_log10_FDR"]
    ).copy()

    selected = plot_df["selected_top_10_percent"]

    # Find the actual fold-change cutoff from selected genes
    fc_cutoff = plot_df.loc[selected, "max_abs_log2FC"].min()

    plt.figure(figsize=(7, 6))

    # Other genes
    plt.scatter(
        plot_df.loc[~selected, "signed_max_log2FC"],
        plot_df.loc[~selected, "minus_log10_FDR"],
        alpha=0.30,
        s=10,
        label="Other eligible genes"
    )

    # Top 10% genes
    plt.scatter(
        plot_df.loc[selected, "signed_max_log2FC"],
        plot_df.loc[selected, "minus_log10_FDR"],
        alpha=0.75,
        s=16,
        label="Top 10% by absolute fold-change"
    )

    # Fold-change cutoff lines
    plt.axvline(
        fc_cutoff,
        linestyle="--",
        linewidth=1.2,
        label=f"Top 10% fold-change cutoff = ±{fc_cutoff:.2f}"
    )

    plt.axvline(
        -fc_cutoff,
        linestyle="--",
        linewidth=1.2
    )

    # Optional FDR line
    if show_fdr_line and fdr_cutoff is not None:
        plt.axhline(
            -np.log10(fdr_cutoff),
            linestyle=":",
            linewidth=1,
            label=f"FDR = {fdr_cutoff}"
        )

    plt.xlabel("maximum absolute log2 expression change from Day 0")
    plt.ylabel("-log10(FDR)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    
    
def compute_fold_change(data):
    """
    Compute fold change for each gene.

    Fold change is calculated as:

        delta_G = (max expression - min expression) / min expression

    A small epsilon is added to avoid division by zero.
    """

    i_max = data.max(axis=1)
    i_min = data.min(axis=1)

    delta_g = (i_max - i_min) / (i_min + EPSILON)

    return delta_g


def select_top_fraction_by_fold_change(data, frac=TOP_FRAC):
    """
    Select the top fraction of genes with the highest fold change.
    """

    fold_change = compute_fold_change(data)

    top_n = int(len(fold_change) * frac)

    top_genes = fold_change.sort_values(ascending=False).head(top_n).index

    filtered_data = data.loc[top_genes]

    return filtered_data


def save_datasets(acute_fc, burn_fc):
    """Save fold-change filtered datasets."""

    acute_fc.to_csv(ACUTE_OUTPUT_PATH, sep="\t")
    burn_fc.to_csv(BURN_OUTPUT_PATH, sep="\t")








# -----------------------------
# Main workflow
# -----------------------------
def main():
    """Run fold-change filtering workflow."""

    acute, burn = load_datasets()

    print("Original acute shape:", acute.shape)
    print("Original burn shape:", burn.shape)


    acute_day0_normalized = subtract_day0_mean_per_gene(
    acute,
    day0_label="0"
    )

    acute_volcano = compute_volcano_table(
    acute_day0_normalized,
    day0_label="0",
    top_frac=0.10
    )

    # plot_volcano(
    #     acute_volcano,
    #     title="Incision dataset: selected top 10% genes",
    #     save_path="acute_volcano_top10.png"
    # )



    burn_log2_cpm = counts_to_log2_cpm(burn)


    burn_day0_normalized = subtract_day0_mean_per_gene(
        burn_log2_cpm,
        day0_label="0"
    )
    
    burn_volcano = compute_volcano_table(
        burn_day0_normalized,
        day0_label="0",
        top_frac=0.10
    )

    # plot_volcano(
    #     burn_volcano,
    #     title="Burn dataset: selected top 10% genes",
    #     save_path="burn_volcano_top10.png"
    # )


    plot_volcano_with_fc_threshold(
    acute_volcano,
    title="Incision dataset: top 10% fold-change genes selected for clustering",
    show_fdr_line=False,
    save_path="acute_volcano_top10_no_fdr_line.png"
    )
    
    
    plot_volcano_with_fc_threshold(
    burn_volcano,
    title="Burn dataset: top 10% fold-change genes selected for clustering",
    show_fdr_line=False,
    save_path="burn_volcano_top10_no_fdr_line.png"
    )
    
    acute_fc = select_top_fraction_by_fold_change(acute, frac=TOP_FRAC)
    burn_fc = select_top_fraction_by_fold_change(burn, frac=TOP_FRAC)

    print("Top 10% acute shape:", acute_fc.shape)
    print("Top 10% burn shape:", burn_fc.shape)

    save_datasets(acute_fc, burn_fc)

    print("Fold-change filtered datasets saved.")





if __name__ == "__main__":
    main()
    
    
    