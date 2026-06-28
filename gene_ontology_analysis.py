"""
GO phase scoring and trajectory workflow
========================================

This script does the following:

1. Loads clustered gene expression tables for burn and acute datasets.
2. Creates cluster average expression tables by averaging genes within each cluster.
3. Runs GO Biological Process enrichment for each cluster using Enrichr through GSEApy.
4. Saves GO enrichment barplots for each dataset and cluster.
5. Uses a custom keyword dictionary to assign enriched GO terms to wound healing phase categories.
6. Scores each cluster by wound healing phase using both matched GO term counts and
   weighted scores based on -log10(adjusted p-value).
7. Saves cluster-level GO phase scores in both long and wide table formats.
8. Combines cluster average expression trajectories with GO phase scores to create
   phase-weighted expression trajectories over time.
9. Saves and plots GO phase-weighted trajectories for each dataset.
10. Creates a long-format acute vs burn table for downstream phase-level boxplots.

Inputs expected:
    Data/cluster/burn_clustered_dayAvg.csv
    Data/cluster/acute_clustered_dayAvg.csv

Main outputs:
    enrichr_results/
    Plots/GO/
    GO_phase_scores/
    GO_phase_trajectories/
    Plots/GO_phase_trajectories/
    GO_phase_boxplot_tables/acute_burn_phase_cluster_long_table.csv
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gseapy import barplot as gp_barplot


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 100,
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})


# -----------------------------------------------------------------------------
# File paths and settings
# -----------------------------------------------------------------------------

DATA_PATHS = {
    "burn": "Data/cluster/burn_clustered_dayAvg.csv",
    "acute": "Data/cluster/acute_clustered_dayAvg.csv",
}

CLUSTER_COL = "Cluster"
ORGANISM = "mouse"
GENE_SETS = "GO_Biological_Process_2021"
SIG_CUTOFF = 0.05
TOP_TERM = 10
ENRICHR_SLEEP_SECONDS = 3

MAIN_PHASES = [
    "Inflammation",
    "Proliferation / tissue formation",
    "ECM / remodeling",
    "Stress / damage response",
    "Cellular regulation / turnover",
    "Ambiguous / specialized signaling",
]


# -----------------------------------------------------------------------------
# Wound healing phase keyword dictionary
# -----------------------------------------------------------------------------

PHASE_KEYWORD_DICTIONARY = {
    "Inflammation": {
        "Inflammation": [
            "immune", "inflammatory", "inflammation", "cytokine", "chemokine",
            "interleukin", "tumor necrosis factor", "tnf", "neutrophil",
            "macrophage", "monocyte", "leukocyte", "lymphocyte", "t cell",
            "b cell", "mast cell", "myeloid", "phagocyt", "degranulation",
            "complement", "defense response", "response to bacterium",
            "response to virus", "pathogen", "microglial", "microglia",
            "microglial cell activation", "lipopolysaccharide",
            "molecule of bacterial origin", "bacterial origin", "interferon-gamma",
            "interferon gamma", "t-helper 17", "T-helper", "th17",
            "t helper 17", "biotic stimulus", "response to biotic stimulus",
        ],
    },
    "Proliferation / tissue formation": {
        "Proliferation / tissue formation": [
            "cell proliferation", "epithelial", "epithelium", "keratinocyte",
            "skin", "morphogenesis", "development", "differentiation",
            "tissue development", "wound healing", "regeneration", "axon guidance",
            "axonogenesis", "cell migration", "fibroblast",
        ],
        "Cell cycle / proliferation": [
            "cell cycle", "mitotic", "mitosis", "spindle",
            "chromosome segregation", "cytokinesis", "dna replication",
            "metaphase", "anaphase", "checkpoint",
            "microtubule cytoskeleton organization involved in mitosis",
            "sister chromatid segregation", "chromatid segregation",
            "microtubule polymerization", "microtubule depolymerization",
            "nuclear membrane reassembly", "nuclear envelope reassembly",
            "negative regulation of cell growth", "negative regulation of growth",
            "regulation of cell growth", "cell growth",
        ],
        "Angiogenesis / vascular repair": [
            "angiogenesis", "vasculature", "vascular", "blood vessel",
            "endothelial", "vegf", "vascular endothelial growth factor",
        ],
        "Development / differentiation ambiguous": [
            "differentiation", "development", "morphogenesis", "organ development",
            "trophoblast", "hepatocyte", "meiotic",
        ],
        "ECM deposition / proliferation": [
            "extracellular matrix", "extracellular matrix organization",
            "extracellular structure organization",
            "external encapsulating structure organization", "collagen fibril organization",
            "supramolecular fiber organization", "hyaluronan", "hyaluronic acid",
            "fibronectin", "granulation", "myofibroblast",
        ],
    },
    "ECM / remodeling": {
        "ECM remodeling / maturation": [
            "matrix remodeling", "matrix metalloproteinase", "collagen crosslinking",
            "collagen fibril crosslinking", "basement membrane", "scar", "elastin",
            "proteoglycan", "glycosaminoglycan", "keratan sulfate", "integrin",
            "mechanical stimulus",
        ],
        "Muscle / contractile remodeling": [
            "muscle", "contraction", "contractile", "actomyosin", "actin-myosin",
            "myofibril", "sarcomere", "striated muscle", "smooth muscle",
            "muscle filament",
        ],
    },
    "Stress / damage response": {
        "Stress / damage response": [
            "stress", "response to heat", "heat shock", "response to uv", "uv-b",
            "uv protection", "oxidative stress", "hypoxia", "dna damage", "cell death",
            "apoptotic", "necrotic", "reactive oxygen species", "fatty acid",
            "fatty acid beta-oxidation", "fatty acid oxidation", "light stimulus",
            "unfolded protein response", "endoplasmic reticulum unfolded protein response",
            "er unfolded protein response", "er stress", "external stimulus",
            "response to external stimulus",
        ],
    },
    "Cellular regulation / turnover": {
        "Protein turnover / cellular regulation": [
            "proteasomal", "proteasome", "ubiquitin", "ubiquitination",
            "protein catabolic", "protein modification", "small protein conjugation",
            "nucleic acid-templated transcription", "transcription, dna-templated",
            "regulation of gene expression", "gene expression",
            "oligosaccharide biosynthetic", "carbohydrate biosynthetic",
            "cellular component organization", "negative regulation of cellular component organization",
            "cellular organization", "protein polymerization",
            "negative regulation of protein polymerization",
        ],
        "Cellular processing / endocytosis": [
            "endocytosis", "receptor-mediated endocytosis", "vacuolar", "lysosome",
            "lysosomal", "acidification", "autophagy", "phagosome", "cellular process",
        ],
    },
    "Ambiguous / specialized signaling": {
        "Neural repair / ambiguous": [
            "axon", "neuron", "neuronal", "synaptic", "dendritic", "nerve",
            "neurogenesis", "synaptic plasticity",
        ],
        "Ambiguous signaling": [
            "signaling", "signal transduction", "receptor signaling", "cation channel",
            "ion channel", "calcium", "kinase", "phosphorylation",
            "transforming growth factor beta", "tgf", "smad", "growth hormone stimulus",
            "growth hormone", "adenylate cyclase", "camp signaling", "cyclic amp",
        ],
    },
}


# -----------------------------------------------------------------------------
# Data loading and cleaning
# -----------------------------------------------------------------------------

def load_clustered_dataset(path: str, sep: str = "\t", cluster_col: str = CLUSTER_COL) -> pd.DataFrame:
    """Load a clustered expression table and check that the cluster column exists."""
    df = pd.read_csv(path, sep=sep, index_col=0)

    if cluster_col not in df.columns:
        raise ValueError(f"Expected column '{cluster_col}' in {path}, but it was not found.")

    return df


def keep_only_cluster_column(df: pd.DataFrame, cluster_col: str = CLUSTER_COL) -> pd.DataFrame:
    """Keep only gene IDs as the index and the cluster assignment column."""
    return df[[cluster_col]].copy()


def get_genes_in_cluster(df: pd.DataFrame, cluster_num, cluster_col: str = CLUSTER_COL) -> list[str]:
    """Return the gene names/index values assigned to one cluster."""
    return df.loc[df[cluster_col] == cluster_num].index.tolist()


def clean_cluster_avg_table(cluster_avg_df: pd.DataFrame, cluster_col: str = CLUSTER_COL) -> pd.DataFrame:
    """
    Clean a cluster-average expression table.

    Expected final format:
        rows = clusters
        columns = averaged timepoints
        values = average expression
    """
    df = cluster_avg_df.copy()

    if cluster_col in df.columns:
        df = df.set_index(cluster_col)

    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    sorted_cols = sorted(df.columns, key=extract_timepoint)
    return df[sorted_cols]


def extract_timepoint(value):
    """Extract a numeric timepoint from labels like 0, 0.25, Day 3, or 3_2."""
    value_str = str(value).replace("Day", "").strip()

    if "_" in value_str:
        value_str = value_str.split("_")[0]

    try:
        return float(value_str)
    except ValueError:
        nums = re.findall(r"\d+\.?\d*", value_str)
        if nums:
            return float(nums[0])
        return value_str


# -----------------------------------------------------------------------------
# GO enrichment and plotting
# -----------------------------------------------------------------------------

def run_go_enrichment_for_cluster(
    df: pd.DataFrame,
    cluster_num,
    title: str,
    cluster_col: str = CLUSTER_COL,
    column: str = "Adjusted P-value",
    organism: str = ORGANISM,
    gene_sets: str = GENE_SETS,
    sig_cutoff: float = SIG_CUTOFF,
    cutoff: int = 1,
    top_term: int = TOP_TERM,
    figsize: tuple[int, int] = (8, 6),
    outdir: str = "enrichr_results",
    ofname: Optional[str] = None,
):
    """Run Enrichr for one cluster, save a barplot, and return the top significant GO terms."""
    gene_list = get_genes_in_cluster(df, cluster_num, cluster_col=cluster_col)

    if not gene_list:
        print(f"No genes found for cluster {cluster_num}. Skipping.")
        return None, None, None

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=f"{outdir}/cluster_{cluster_num}",
        )
    except ValueError as err:
        print(f"No enrichment results for cluster {cluster_num}. Skipping.")
        print(f"GSEApy message: {err}")
        return None, None, None
    except Exception as err:
        print(f"Error running enrichment for cluster {cluster_num}. Skipping.")
        print(f"Error message: {err}")
        return None, None, None

    if enr.results is None or enr.results.empty:
        print(f"No enrichment results found for cluster {cluster_num}. Skipping.")
        return enr, None, None

    results = enr.results.copy()
    numeric_cols = ["Adjusted P-value", "P-value", "Combined Score"]

    for col in numeric_cols:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    results = results.dropna(subset=numeric_cols)

    if results.empty:
        print(f"No usable enrichment results found for cluster {cluster_num}. Skipping.")
        return enr, None, None

    significant_results = results.loc[results[column] < sig_cutoff].copy()

    if significant_results.empty:
        print(
            f"No significant GO terms found for cluster {cluster_num} "
            f"using {column} < {sig_cutoff}. Skipping."
        )
        return enr, None, None

    top_results = significant_results.sort_values(column).head(top_term).copy()

    if ofname is not None:
        os.makedirs(os.path.dirname(ofname), exist_ok=True)

    try:
        ax = gp_barplot(
            df=top_results,
            column=column,
            title=title,
            cutoff=cutoff,
            top_term=top_term,
            figsize=figsize,
            ofname=ofname,
        )
    except ValueError as err:
        print(f"Could not make plot for cluster {cluster_num}. Skipping plot.")
        print(f"GSEApy plotting message: {err}")
        return enr, top_results, None
    except Exception as err:
        print(f"Unexpected plotting error for cluster {cluster_num}. Skipping plot.")
        print(f"Error message: {err}")
        return enr, top_results, None

    return enr, top_results, ax


# -----------------------------------------------------------------------------
# GO phase scoring
# -----------------------------------------------------------------------------

def score_go_terms_by_phase_weighted(
    top_results: pd.DataFrame,
    phase_keyword_dictionary: dict,
    term_col: str = "Term",
    pval_col: str = "Adjusted P-value",
    genes_col: str = "Genes",
):
    """
    Score enriched GO terms using the wound healing phase keyword dictionary.

    Each matched GO term contributes:
        count += 1
        weighted_score += -log10(adjusted p-value)

    The function also stores the matched GO term, matched keyword, subcategory,
    adjusted p-value, weight, and genes associated with the enriched term.
    """
    broad_phase_counts = {phase: 0 for phase in phase_keyword_dictionary}
    broad_phase_weighted_scores = {phase: 0.0 for phase in phase_keyword_dictionary}
    matched_terms = defaultdict(list)
    unmatched_terms = []

    results = top_results.copy()
    results[pval_col] = pd.to_numeric(results[pval_col], errors="coerce")
    results = results.dropna(subset=[term_col, pval_col])

    tiny = 1e-300

    for _, row in results.iterrows():
        term = str(row[term_col])
        term_lower = term.lower()
        adj_p = max(float(row[pval_col]), tiny)
        weight = -np.log10(adj_p)

        genes_for_term = row[genes_col] if genes_col in results.columns else ""
        genes_for_term = "" if pd.isna(genes_for_term) else str(genes_for_term)

        matched_any_phase = False

        for broad_phase, subcategory_dict in phase_keyword_dictionary.items():
            matched_this_phase = False

            for subcategory, keywords in subcategory_dict.items():
                for keyword in keywords:
                    if keyword.lower() in term_lower:
                        broad_phase_counts[broad_phase] += 1
                        broad_phase_weighted_scores[broad_phase] += weight

                        matched_terms[broad_phase].append({
                            "term": term,
                            "subcategory": subcategory,
                            "matched_keyword": keyword,
                            "adjusted_p_value": adj_p,
                            "weight": weight,
                            "genes": genes_for_term,
                        })

                        matched_any_phase = True
                        matched_this_phase = True
                        break

                if matched_this_phase:
                    break

        if not matched_any_phase:
            unmatched_terms.append(term)

    score_df = pd.DataFrame({
        "Matched GO Term Count": broad_phase_counts,
        "Weighted Score (-log10 adj p)": broad_phase_weighted_scores,
    }).sort_values("Weighted Score (-log10 adj p)", ascending=False)

    return score_df, dict(matched_terms), unmatched_terms


def run_go_phase_scoring_for_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    phase_keyword_dictionary: dict,
    cluster_col: str = CLUSTER_COL,
    organism: str = ORGANISM,
    gene_sets: str = GENE_SETS,
    sig_cutoff: float = SIG_CUTOFF,
    top_term: int = TOP_TERM,
    go_plot_dir: str = "Plots/GO",
    enrichr_outdir: str = "enrichr_results",
    score_outdir: str = "GO_phase_scores",
):
    """Run GO enrichment and GO phase scoring for every cluster in one dataset."""
    os.makedirs(go_plot_dir, exist_ok=True)
    os.makedirs(enrichr_outdir, exist_ok=True)
    os.makedirs(score_outdir, exist_ok=True)

    all_go_results = {}

    for cluster_num in sorted(df[cluster_col].dropna().unique()):
        print(f"\nRunning GO enrichment for {dataset_name} Cluster {cluster_num}")

        _, top_results, _ = run_go_enrichment_for_cluster(
            df=df,
            cluster_num=cluster_num,
            title=f"{dataset_name} Cluster {cluster_num} GO Biological Process",
            cluster_col=cluster_col,
            organism=organism,
            gene_sets=gene_sets,
            sig_cutoff=sig_cutoff,
            top_term=top_term,
            outdir=f"{enrichr_outdir}/{dataset_name}",
            ofname=f"{go_plot_dir}/{dataset_name}_cluster{cluster_num}_go_barplot.png",
        )

        time.sleep(ENRICHR_SLEEP_SECONDS)

        if top_results is not None:
            all_go_results[cluster_num] = top_results

    cluster_phase_scores = {}
    cluster_matched_terms = {}
    cluster_unmatched_terms = {}

    for cluster_num, top_results in all_go_results.items():
        score_df, matched_terms, unmatched_terms = score_go_terms_by_phase_weighted(
            top_results=top_results,
            phase_keyword_dictionary=phase_keyword_dictionary,
        )

        cluster_phase_scores[cluster_num] = score_df
        cluster_matched_terms[cluster_num] = matched_terms
        cluster_unmatched_terms[cluster_num] = unmatched_terms

    long_table = make_go_phase_long_table(
        dataset_name=dataset_name,
        cluster_phase_scores=cluster_phase_scores,
        cluster_matched_terms=cluster_matched_terms,
        cluster_unmatched_terms=cluster_unmatched_terms,
    )

    wide_table = make_go_phase_wide_table(cluster_phase_scores)

    save_go_phase_tables(
        dataset_name=dataset_name,
        long_table=long_table,
        wide_table=wide_table,
        score_outdir=score_outdir,
    )

    return {
        "go_results": all_go_results,
        "phase_scores": cluster_phase_scores,
        "matched_terms": cluster_matched_terms,
        "unmatched_terms": cluster_unmatched_terms,
        "long_table": long_table,
        "wide_table": wide_table,
    }


def make_go_phase_long_table(
    dataset_name: str,
    cluster_phase_scores: dict,
    cluster_matched_terms: dict,
    cluster_unmatched_terms: dict,
) -> pd.DataFrame:
    """Create a long-format GO phase score table for one dataset."""
    rows = []

    for cluster_num, score_df in cluster_phase_scores.items():
        unmatched_terms = cluster_unmatched_terms.get(cluster_num, [])

        for phase, row in score_df.iterrows():
            matched_items = cluster_matched_terms.get(cluster_num, {}).get(phase, [])

            rows.append({
                "Dataset": dataset_name,
                "Cluster": cluster_num,
                "Phase": phase,
                "Matched GO Term Count": row["Matched GO Term Count"],
                "Weighted Score (-log10 adj p)": row["Weighted Score (-log10 adj p)"],
                "Matched GO Terms": "; ".join(item["term"] for item in matched_items),
                "Matched Subcategories": "; ".join(item["subcategory"] for item in matched_items),
                "Matched Keywords": "; ".join(item["matched_keyword"] for item in matched_items),
                "Matched Genes": "; ".join(item["genes"] for item in matched_items),
                "Matched GO Terms with Genes": " | ".join(
                    f'{item["term"]}: {item["genes"]}' for item in matched_items
                ),
                "Matched Terms with Weights": "; ".join(
                    f'{item["term"]} ({item["subcategory"]}: {item["matched_keyword"]}, '
                    f'weight={item["weight"]:.3f})'
                    for item in matched_items
                ),
                "Unmatched GO Term Count": len(unmatched_terms),
                "Unmatched GO Terms": "; ".join(unmatched_terms),
            })

    long_table = pd.DataFrame(rows)

    if not long_table.empty:
        long_table = long_table.sort_values(
            by=["Cluster", "Weighted Score (-log10 adj p)"],
            ascending=[True, False],
        )

    return long_table


def make_go_phase_wide_table(cluster_phase_scores: dict) -> pd.DataFrame:
    """Create a wide-format table with clusters as rows and phases as columns."""
    if not cluster_phase_scores:
        return pd.DataFrame()

    wide_table = pd.DataFrame({
        cluster_num: score_df["Weighted Score (-log10 adj p)"]
        for cluster_num, score_df in cluster_phase_scores.items()
    }).T

    wide_table.index.name = "Cluster"
    return wide_table


def save_go_phase_tables(
    dataset_name: str,
    long_table: pd.DataFrame,
    wide_table: pd.DataFrame,
    score_outdir: str = "GO_phase_scores",
) -> None:
    """Save long and wide GO phase score tables."""
    os.makedirs(score_outdir, exist_ok=True)

    if not long_table.empty:
        long_file = f"{score_outdir}/{dataset_name}_all_cluster_phase_keyword_weighted_scores.csv"
        long_table.to_csv(long_file, index=False)
        print(f"\nSaved long weighted phase scores to: {long_file}")
    else:
        print(f"\nNo GO phase scores found for {dataset_name}. Long file not saved.")

    if not wide_table.empty:
        wide_file = f"{score_outdir}/{dataset_name}_cluster_phase_weighted_score_table.csv"
        wide_table.to_csv(wide_file)
        print(f"Saved wide weighted score table to: {wide_file}")
    else:
        print(f"No wide score table saved for {dataset_name}.")


# -----------------------------------------------------------------------------
# GO phase trajectories
# -----------------------------------------------------------------------------

def compute_go_phase_trajectories_from_cluster_avg(
    cluster_avg_df: pd.DataFrame,
    phase_weight_table: pd.DataFrame,
    normalize_phase_weights: bool = True,
) -> pd.DataFrame:
    """
    Combine cluster-average expression trajectories with GO phase scores.

    Returns a table where:
        rows = GO phases
        columns = timepoints
        values = GO phase-weighted expression over time
    """
    cluster_avg = clean_cluster_avg_table(cluster_avg_df)
    phase_weights = phase_weight_table.copy()

    phase_weights.index = phase_weights.index.astype(str)
    cluster_avg.index = cluster_avg.index.astype(str)

    shared_clusters = cluster_avg.index.intersection(phase_weights.index)

    if len(shared_clusters) == 0:
        raise ValueError(
            "No matching clusters found between cluster_avg_df and phase_weight_table. "
            "Check whether cluster labels are formatted the same way."
        )

    cluster_avg = cluster_avg.loc[shared_clusters]
    phase_weights = phase_weights.loc[shared_clusters].fillna(0)

    if normalize_phase_weights:
        phase_weights = phase_weights.div(
            phase_weights.sum(axis=0).replace(0, np.nan),
            axis=1,
        ).fillna(0)

    return phase_weights.T @ cluster_avg


def plot_go_phase_trajectories(
    phase_trajectories: pd.DataFrame,
    dataset_name: str,
    phases_to_plot: Optional[list[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12.5, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot GO phase-weighted expression trajectories with real day spacing on the x-axis."""
    plot_df = phase_trajectories.copy()

    if phases_to_plot is not None:
        phases_to_plot = [phase for phase in phases_to_plot if phase in plot_df.index]
        plot_df = plot_df.loc[phases_to_plot]

    time_cols = plot_df.columns
    x_values = [extract_timepoint(col) for col in time_cols]

    plt.figure(figsize=figsize)

    for phase in plot_df.index:
        plt.plot(x_values, plot_df.loc[phase].values, marker="o", linewidth=2, label=phase)

    if title is None:
        title = f"{dataset_name}: GO Phase-Weighted Expression Trajectories"

    plt.title(title, pad=18)
    plt.xlabel("Time in Days", labelpad=10)
    plt.ylabel("GO phase-weighted average expression", labelpad=14)
    plt.xticks(ticks=x_values, labels=[f"{day:g}" for day in x_values], rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, frameon=True, borderaxespad=0.5)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def save_and_plot_phase_trajectories(
    cluster_avg_datasets: dict[str, pd.DataFrame],
    all_dataset_results: dict,
    phases_to_plot: list[str] = MAIN_PHASES,
) -> dict[str, pd.DataFrame]:
    """Create, save, and plot phase-weighted trajectories for all datasets."""
    all_phase_trajectories = {}
    os.makedirs("GO_phase_trajectories", exist_ok=True)

    for dataset_name, cluster_avg_df in cluster_avg_datasets.items():
        print("\n############################################################")
        print(f"Computing GO phase trajectories for: {dataset_name}")
        print("############################################################")

        phase_weight_table = all_dataset_results[dataset_name]["wide_table"]

        phase_trajectories = compute_go_phase_trajectories_from_cluster_avg(
            cluster_avg_df=cluster_avg_df,
            phase_weight_table=phase_weight_table,
            normalize_phase_weights=True,
        )

        all_phase_trajectories[dataset_name] = phase_trajectories
        phase_trajectories.to_csv(f"GO_phase_trajectories/{dataset_name}_go_phase_weighted_trajectories.csv")

        plot_go_phase_trajectories(
            phase_trajectories=phase_trajectories,
            dataset_name=dataset_name,
            phases_to_plot=phases_to_plot,
            save_path=f"Plots/GO_phase_trajectories/{dataset_name}_go_phase_trajectories.png",
        )

    return all_phase_trajectories


# -----------------------------------------------------------------------------
# Boxplot table creation
# -----------------------------------------------------------------------------

def make_phase_cluster_long_table(
    cluster_avg_df: pd.DataFrame,
    phase_weight_table: pd.DataFrame,
    dataset_name: str,
    phases_to_keep: Optional[list[str]] = None,
    normalize_phase_weights: bool = True,
) -> pd.DataFrame:
    """
    Create a long table for downstream boxplots.

    Output columns:
        Dataset | Cluster | Phase | Timepoint | Expression | Phase Weight | Weighted Expression
    """
    cluster_avg = clean_cluster_avg_table(cluster_avg_df)
    phase_weights = phase_weight_table.copy()

    phase_weights.index = phase_weights.index.astype(str)
    cluster_avg.index = cluster_avg.index.astype(str)

    shared_clusters = cluster_avg.index.intersection(phase_weights.index)

    if len(shared_clusters) == 0:
        raise ValueError(
            f"No matching clusters found for {dataset_name}. "
            "Check cluster labels in cluster_avg_df and phase_weight_table."
        )

    cluster_avg = cluster_avg.loc[shared_clusters]
    phase_weights = phase_weights.loc[shared_clusters]

    if phases_to_keep is not None:
        phases_to_keep = [phase for phase in phases_to_keep if phase in phase_weights.columns]
        phase_weights = phase_weights[phases_to_keep]

    phase_weights = phase_weights.fillna(0)

    if normalize_phase_weights:
        phase_weights = phase_weights.div(
            phase_weights.sum(axis=0).replace(0, np.nan),
            axis=1,
        ).fillna(0)

    rows = []

    for cluster in shared_clusters:
        for phase in phase_weights.columns:
            phase_weight = phase_weights.loc[cluster, phase]

            if phase_weight == 0:
                continue

            for timepoint in cluster_avg.columns:
                expression = cluster_avg.loc[cluster, timepoint]

                rows.append({
                    "Dataset": dataset_name,
                    "Cluster": cluster,
                    "Phase": phase,
                    "Timepoint": timepoint,
                    "Expression": expression,
                    "Phase Weight": phase_weight,
                    "Weighted Expression": expression * phase_weight,
                })

    return pd.DataFrame(rows)


def save_acute_burn_boxplot_table(
    acute_cluster_avg: pd.DataFrame,
    burn_cluster_avg: pd.DataFrame,
    all_dataset_results: dict,
    phases_to_keep: list[str] = MAIN_PHASES,
    out_file: str = "GO_phase_boxplot_tables/acute_burn_phase_cluster_long_table.csv",
) -> pd.DataFrame:
    """Create and save the acute vs burn cluster-level phase table for boxplots."""
    dataset_info = {
        "acute": {
            "cluster_avg": acute_cluster_avg,
            "phase_weights": all_dataset_results["acute"]["wide_table"],
        },
        "burn": {
            "cluster_avg": burn_cluster_avg,
            "phase_weights": all_dataset_results["burn"]["wide_table"],
        },
    }

    all_tables = []

    for dataset_name, info in dataset_info.items():
        print(f"Making cluster-level phase boxplot table for {dataset_name}")

        long_df = make_phase_cluster_long_table(
            cluster_avg_df=info["cluster_avg"],
            phase_weight_table=info["phase_weights"],
            dataset_name=dataset_name,
            phases_to_keep=phases_to_keep,
            normalize_phase_weights=True,
        )

        all_tables.append(long_df)

    phase_cluster_long_df = pd.concat(all_tables, ignore_index=True)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    phase_cluster_long_df.to_csv(out_file, index=False)

    print(f"Saved cluster-level phase boxplot table to: {out_file}")
    return phase_cluster_long_df


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the full GO enrichment, scoring, trajectory, and boxplot-table workflow."""
    clustered_datasets = {
        name: load_clustered_dataset(path)
        for name, path in DATA_PATHS.items()
    }

    cluster_avg_datasets = {
        name: df.groupby(CLUSTER_COL).mean(numeric_only=True)
        for name, df in clustered_datasets.items()
    }

    go_input_datasets = {
        name: keep_only_cluster_column(df)
        for name, df in clustered_datasets.items()
    }

    all_dataset_results = {}

    for dataset_name, df in go_input_datasets.items():
        print("\n############################################################")
        print(f"Running full GO phase scoring workflow for: {dataset_name}")
        print("############################################################")

        all_dataset_results[dataset_name] = run_go_phase_scoring_for_dataset(
            df=df,
            dataset_name=dataset_name,
            phase_keyword_dictionary=PHASE_KEYWORD_DICTIONARY,
            cluster_col=CLUSTER_COL,
            organism=ORGANISM,
            gene_sets=GENE_SETS,
            sig_cutoff=SIG_CUTOFF,
            top_term=TOP_TERM,
        )

    save_and_plot_phase_trajectories(
        cluster_avg_datasets=cluster_avg_datasets,
        all_dataset_results=all_dataset_results,
        phases_to_plot=MAIN_PHASES,
    )

    save_acute_burn_boxplot_table(
        acute_cluster_avg=cluster_avg_datasets["acute"],
        burn_cluster_avg=cluster_avg_datasets["burn"],
        all_dataset_results=all_dataset_results,
        phases_to_keep=MAIN_PHASES,
    )


if __name__ == "__main__":
    main()
