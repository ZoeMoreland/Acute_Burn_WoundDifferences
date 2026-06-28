"""
DTW cluster trajectory analysis

This script:
1. Loads clustered, day-averaged expression files.
2. Computes mean cluster trajectories.
3. Computes DTW distances between two datasets.
4. Saves a ranked cluster match table.
5. Plots a DTW heatmap.
6. Optionally plots DTW alignments for selected cluster pairs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tslearn.metrics import dtw, dtw_path


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------
DATA_DIR = Path("Data/cluster")
PLOT_DIR = Path("Plots/DTW")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "acute": DATA_DIR / "acute_clustered_dayAvg.csv",
    "burn": DATA_DIR / "burn_clustered_dayAvg.csv",
}

CLUSTER_COL = "Cluster"
SEP = "\t"


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


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def safe_name(name: str) -> str:
    """Make dataset names safer for filenames."""
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )


def load_clustered_data(files: dict[str, Path], sep: str = "\t") -> dict[str, pd.DataFrame]:
    """Load all clustered expression datasets."""
    return {
        name: pd.read_csv(path, index_col=0, sep=sep)
        for name, path in files.items()
    }


def get_time_columns(df: pd.DataFrame, cluster_col: str = CLUSTER_COL) -> list[str]:
    """
    Return expression columns that can be interpreted as numeric timepoints.
    This avoids relying on df.columns[1:], which can break if column order changes.
    """
    time_cols = []

    for col in df.columns:
        if col == cluster_col:
            continue
        try:
            float(col)
            time_cols.append(col)
        except ValueError:
            pass

    return sorted(time_cols, key=float)


def zscore_series(values: np.ndarray) -> np.ndarray:
    """Z-score one trajectory. If std is 0, center only."""
    values = np.asarray(values, dtype=float)
    std = np.std(values)

    if std == 0:
        return values - np.mean(values)

    return (values - np.mean(values)) / std


def get_cluster_trajectories(
    df: pd.DataFrame,
    cluster_col: str = CLUSTER_COL,
) -> pd.DataFrame:
    """Average expression across genes within each cluster."""
    time_cols = get_time_columns(df, cluster_col=cluster_col)

    if cluster_col not in df.columns:
        raise ValueError(f"Missing required cluster column: {cluster_col}")
    if not time_cols:
        raise ValueError("No numeric time columns were found.")

    return df.groupby(cluster_col)[time_cols].mean()


def compute_dtw_distance_matrix(
    traj1: pd.DataFrame,
    traj2: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute pairwise DTW distances between clusters from two trajectory tables."""
    distances = pd.DataFrame(index=traj1.index, columns=traj2.index, dtype=float)

    for cluster1 in traj1.index:
        series1 = traj1.loc[cluster1].to_numpy(dtype=float)
        if normalize:
            series1 = zscore_series(series1)

        for cluster2 in traj2.index:
            series2 = traj2.loc[cluster2].to_numpy(dtype=float)
            if normalize:
                series2 = zscore_series(series2)

            distances.loc[cluster1, cluster2] = dtw(series1, series2)

    return distances


def distance_matrix_to_matches(
    dist_matrix: pd.DataFrame,
    dataset1_label: str,
    dataset2_label: str,
    remove_self_matches: bool = False,
) -> pd.DataFrame:
    """Convert a DTW distance matrix into a ranked long-format table."""
    matches = (
        dist_matrix
        .rename_axis(f"{dataset1_label}_Cluster")
        .reset_index()
        .melt(
            id_vars=f"{dataset1_label}_Cluster",
            var_name=f"Matched_{dataset2_label}_Cluster",
            value_name="DTW_Distance",
        )
    )

    if remove_self_matches:
        matches = matches[
            matches[f"{dataset1_label}_Cluster"] != matches[f"Matched_{dataset2_label}_Cluster"]
        ]

    return matches.sort_values("DTW_Distance").reset_index(drop=True)


def plot_dtw_heatmap(
    dist_matrix: pd.DataFrame,
    dataset1_label: str,
    dataset2_label: str,
    output_dir: Path = PLOT_DIR,
    annot: bool = True,
) -> None:
    """Plot and save a DTW distance heatmap."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        dist_matrix,
        cmap="viridis",
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "DTW Distance"},
    )

    plt.xlabel(f"{dataset2_label} Clusters")
    plt.ylabel(f"{dataset1_label} Clusters")
    plt.title("DTW Distance Between Cluster Trajectories")
    plt.tight_layout()

    filename = f"dtw_heatmap_{safe_name(dataset1_label)}_vs_{safe_name(dataset2_label)}.png"
    plt.savefig(output_dir / filename, dpi=300)
    plt.show()


def plot_cluster(
    df: pd.DataFrame,
    cluster_number: int,
    dataset_label: str = "Dataset",
    cluster_col: str = CLUSTER_COL,
) -> None:
    """Plot all gene trajectories and the mean trajectory for one cluster."""
    time_cols = get_time_columns(df, cluster_col=cluster_col)
    cluster_df = df[df[cluster_col] == cluster_number]

    if cluster_df.empty:
        raise ValueError(f"No rows found for cluster {cluster_number} in {dataset_label}.")

    days = np.array(time_cols, dtype=float)

    plt.figure(figsize=(10, 6))

    for _, row in cluster_df.iterrows():
        plt.plot(days, row[time_cols].to_numpy(dtype=float), alpha=0.2)

    mean_curve = cluster_df[time_cols].mean().to_numpy(dtype=float)
    plt.plot(days, mean_curve, linewidth=3)

    plt.xticks(days, [f"{day:g}" for day in days], rotation=45)
    plt.title(f"{dataset_label} Cluster {cluster_number} (n={len(cluster_df)})")
    plt.xlabel("Timepoint in Days")
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.show()


def print_dtw_day_mapping(
    path: list[tuple[int, int]],
    days1: np.ndarray,
    days2: np.ndarray,
    name1: str,
    name2: str,
    cluster1: int,
    cluster2: int,
) -> None:
    """Print day-to-day DTW alignment mapping."""
    print(f"\nDTW Day Mapping: {name1} cluster {cluster1} → {name2} cluster {cluster2}")
    print("-" * 60)

    for i, j in path:
        print(f"{name1} Day {days1[i]:g}  →  {name2} Day {days2[j]:g}")


def plot_dtw_alignment_two_panels(
    traj1: pd.DataFrame,
    traj2: pd.DataFrame,
    cluster1: int,
    cluster2: int,
    name1: str = "Dataset1",
    name2: str = "Dataset2",
    normalize: bool = True,
    output_dir: Path = PLOT_DIR,
) -> None:
    """Plot two cluster trajectories and draw DTW alignment connections."""
    series1 = traj1.loc[cluster1].to_numpy(dtype=float)
    series2 = traj2.loc[cluster2].to_numpy(dtype=float)

    if normalize:
        series1 = zscore_series(series1)
        series2 = zscore_series(series2)

    days1 = np.array(traj1.columns, dtype=float)
    days2 = np.array(traj2.columns, dtype=float)

    path, distance = dtw_path(series1, series2)
    print_dtw_day_mapping(path, days1, days2, name1, name2, cluster1, cluster2)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
    )

    ax1.plot(days1, series1, marker="o", label=f"{name1} cluster {cluster1}")
    ax1.set_title(f"{name1} Cluster {cluster1}")
    ax1.set_ylabel("Average Expression")
    ax1.set_xticks(days1)
    ax1.set_xticklabels([f"{day:g}" for day in days1], rotation=45)

    ax2.plot(days2, series2, marker="o", label=f"{name2} cluster {cluster2}")
    ax2.set_title(f"{name2} Cluster {cluster2}")
    ax2.set_ylabel("Average Expression")
    ax2.set_xlabel("Timepoint in Days")
    ax2.set_xticks(days2)
    ax2.set_xticklabels([f"{day:g}" for day in days2], rotation=45)

    for i, j in path:
        connection = ConnectionPatch(
            xyA=(days1[i], series1[i]),
            coordsA=ax1.transData,
            xyB=(days2[j], series2[j]),
            coordsB=ax2.transData,
            linestyle="--",
            linewidth=1,
            alpha=0.4,
        )
        fig.add_artist(connection)

    fig.suptitle(
        f"DTW Alignment\n"
        f"{name1} Cluster {cluster1} vs {name2} Cluster {cluster2}\n"
        f"DTW distance = {distance:.3f} ({'z-scored' if normalize else 'raw'})",
        y=0.98,
    )

    plt.tight_layout()

    filename = (
        f"dtw_alignment_{safe_name(name1)}_C{cluster1}"
        f"_vs_{safe_name(name2)}_C{cluster2}.png"
    )
    plt.savefig(output_dir / filename, dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
def main() -> None:
    data = load_clustered_data(FILES, sep=SEP)

    acute_traj = get_cluster_trajectories(data["acute"])
    burn_traj = get_cluster_trajectories(data["burn"])

    # Save averaged cluster trajectories if useful later.
    acute_traj.to_csv(DATA_DIR / "acute_clustered_averaged.csv", sep=SEP)
    burn_traj.to_csv(DATA_DIR / "burn_cluster_averaged.csv", sep=SEP)

    # Compare Incision/GSE23006 vs Burn.
    acute_burn_dtw = compute_dtw_distance_matrix(acute_traj, burn_traj, normalize=True)

    acute_burn_matches = distance_matrix_to_matches(
        acute_burn_dtw,
        dataset1_label="Acute",
        dataset2_label="Burn",
        remove_self_matches=False,
    )

    acute_burn_matches.to_csv(
        DATA_DIR / "all_acute_burn_cluster_matches.csv",
        index=False,
        sep=SEP,
    )

    plot_dtw_heatmap(
        acute_burn_dtw,
        dataset1_label="Incision_GSE23006",
        dataset2_label="Burn",
        annot=True,
    )

    # Optional example alignment plot. Change these clusters as needed.
    plot_dtw_alignment_two_panels(
        acute_traj,
        burn_traj,
        cluster1=5,
        cluster2=3,
        name1="Acute",
        name2="Burn",
        normalize=True,
    )

if __name__ == "__main__":
    main()
