"""
Clean and preprocess incision and burn wound-healing datasets.

Steps:
1. Load raw datasets
2. Standardize gene names
3. Rename burn sample columns
4. Remove genes with inconsistent replicates
5. Collapse repeated gene entries based on correlation
6. Remove mostly-zero genes
7. Detect and remove sample-level outliers
8. Save cleaned datasets
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations
from typing import Any, Dict, Tuple




# -----------------------------
# File paths
# -----------------------------
ACUTE_RAW_PATH = "Data/Raw Data/GSE23006_Raw_data_Gene_Mapped.csv"
BURN_RAW_PATH = "Data/Raw Data/raw_burn_fem_data.csv"

ACUTE_OUTPUT_PATH = "Data/PreProcessing_Data/GSE23006_Cleaned_data.csv"
BURN_OUTPUT_PATH = "Data/PreProcessing_Data/burn_fem_cleaned_data.csv"


# -----------------------------
# Cleaning settings
# -----------------------------
REPLICATE_THRESHOLD_K = 4.0
DUPLICATE_CORRELATION_THRESHOLD = 0.90
MIN_DETECTION_FRAC = 0.30
OUTLIER_Z_THRESHOLD = 3.0


def load_datasets():
    """Load raw acute and burn datasets."""

    acute_df = pd.read_csv(
        ACUTE_RAW_PATH,
        sep="\t",
        index_col=0
    )

    burn_df = pd.read_csv(
        BURN_RAW_PATH,
        sep=",",
        header=1,
        index_col=0
    )

    return acute_df, burn_df


def standardize_gene_names(*dfs):
    """Convert gene names in dataframe indices to uppercase."""

    standardized = []

    for df in dfs:
        df = df.copy()
        df.index = df.index.str.upper()
        standardized.append(df)

    return standardized



def rename_burn_columns(burn_df):
    """
    Rename burn columns from formats like '0.1', '0.2'
    to '0_1', '0_2', etc.
    """

    burn_df = burn_df.copy()

    burn_cols = pd.Series(burn_df.columns, dtype="string")
    burn_days = burn_cols.str.split(".", n=1, expand=True)[0].astype(int)

    day_counts = {}
    new_cols = []

    for day in burn_days:
        day_counts[day] = day_counts.get(day, 0) + 1
        new_cols.append(f"{day}_{day_counts[day]}")

    burn_df.columns = new_cols

    return burn_df


def filter_genes_by_replicate_consistency(
    expr: pd.DataFrame,
    threshold_k: float = 4.0,
    min_reps_per_day: int = 2,
    denom_epsilon: float = 1e-8,
    drop_days_with_few_reps: bool = True,
    return_diagnostics: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Replicate-consistency gene filter modeled after the paper in your screenshot.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene x Sample matrix. Rows = genes, columns like "0_1", "0_2", "14_7", etc.
        Values should be non-negative expression (raw counts, normalized counts, etc.).
    threshold_k : float
        Number of standard deviations above the mean for the day-specific threshold (paper uses 4).
    min_reps_per_day : int
        Minimum number of replicates required to compute replicate consistency for a day.
    denom_epsilon : float
        Small value added to denominators to avoid division-by-zero (only used when day mean is 0).
    drop_days_with_few_reps : bool
        If True, days with < min_reps_per_day are ignored (not used for filtering).
        If False, raises ValueError when encountering such a day.
    return_diagnostics : bool
        If True, returns a diagnostics dictionary with thresholds, per-gene max errors, and flags.

    Returns
    -------
    filtered_expr : pd.DataFrame
        Subset of expr with inconsistent genes removed.
    diagnostics : dict
        Contains:
          - day_to_cols: mapping of day -> list of sample columns
          - day_threshold: pd.Series of r_Thres per day
          - day_ravg, day_rstd: pd.Series mean/std of relative error per day
          - gene_day_maxerr: pd.DataFrame (genes x days) max % error per gene per day
          - gene_flag: pd.Series boolean (True = dropped)
          - kept_genes, dropped_genes: lists
    """

    if not isinstance(expr, pd.DataFrame):
        raise TypeError("expr must be a pandas DataFrame")

    # --- Parse day from column names like "0_1" ---
    cols = expr.columns.astype(str)
    parts = pd.Series(cols, index=cols).str.split("_", n=1, expand=True)

    if parts.shape[1] < 2:
        raise ValueError("Column names must look like 'day_sample' (e.g., '0_1').")

    day = pd.to_numeric(parts[0], errors="coerce")
    if day.isna().any():
        bad = list(cols[day.isna()])
        raise ValueError(f"Could not parse day from columns: {bad[:10]}{'...' if len(bad)>10 else ''}")

    day = day.astype(int)

    # Group columns by day
    day_to_cols: Dict[int, list] = {}
    for c, d in zip(cols, day.values):
        day_to_cols.setdefault(int(d), []).append(c)

    # Sort days for consistent outputs
    days_sorted = sorted(day_to_cols.keys())

    # Validate replicate counts
    usable_days = []
    for d in days_sorted:
        if len(day_to_cols[d]) < min_reps_per_day:
            if drop_days_with_few_reps:
                continue
            else:
                raise ValueError(f"Day {d} has only {len(day_to_cols[d])} replicates (< {min_reps_per_day}).")
        usable_days.append(d)

    if len(usable_days) == 0:
        raise ValueError("No days have enough replicates to run this filter.")

    # We'll store: per-day thresholds and per-gene/day max errors
    day_ravg = {}
    day_rstd = {}
    day_threshold = {}
    gene_day_maxerr = pd.DataFrame(index=expr.index, columns=usable_days, dtype=float)

    # --- Main loop over days ---
    for d in usable_days:
        cols_d = day_to_cols[d]
        Xd = expr[cols_d]  # genes x reps

        # Gene-wise mean across replicates for this day (G in the paper, for this timepoint)
        Gd = Xd.mean(axis=1)  # genes

        # Relative percent error per gene per replicate (S^k in the paper)
        # Use epsilon only where mean is 0 to avoid division-by-zero.
        denom = Gd.copy()
        denom = denom.where(denom != 0, other=denom_epsilon)

        # Broadcast denom across columns, compute percent error
        rel_err = (Xd.sub(Gd, axis=0).abs().div(denom, axis=0)) * 100.0  # genes x reps

        # Compute r_AVG and r_STD for this day across all genes and replicates
        # Flatten all values for this day
        vals = rel_err.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            # If everything is NaN, skip day (should be rare)
            continue

        r_avg = float(np.mean(vals))
        r_std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        r_thres = r_avg + threshold_k * r_std

        day_ravg[d] = r_avg
        day_rstd[d] = r_std
        day_threshold[d] = r_thres

        # For each gene, max replicate error at this day (S_max in the paper generalized)
        gene_day_maxerr[d] = rel_err.max(axis=1)

    # Some days might have been skipped if rel_err had no finite values
    gene_day_maxerr = gene_day_maxerr.dropna(axis=1, how="all")
    final_days = list(gene_day_maxerr.columns)

    if len(final_days) == 0:
        raise ValueError("All usable days were skipped due to missing/invalid values.")

    # Align threshold series to final days
    day_ravg_s = pd.Series({d: day_ravg[d] for d in final_days}, name="r_avg")
    day_rstd_s = pd.Series({d: day_rstd[d] for d in final_days}, name="r_std")
    day_thr_s  = pd.Series({d: day_threshold[d] for d in final_days}, name="r_thres")

    # --- Flag genes: drop if ANY day exceeds that day’s threshold ---
    # Compare each gene's max error at day d to threshold(d)
    exceeds = gene_day_maxerr.gt(day_thr_s, axis=1)  # genes x days boolean
    gene_flag = exceeds.any(axis=1)  # True = drop gene

    filtered_expr = expr.loc[~gene_flag].copy()

    diagnostics = {}
    if return_diagnostics:
        diagnostics = {
            "day_to_cols": {d: day_to_cols[d] for d in days_sorted},
            "usable_days": usable_days,
            "final_days": final_days,
            "day_ravg": day_ravg_s,
            "day_rstd": day_rstd_s,
            "day_threshold": day_thr_s,
            "gene_day_maxerr": gene_day_maxerr,
            "gene_flag": gene_flag,
            "kept_genes": filtered_expr.index.tolist(),
            "dropped_genes": expr.index[gene_flag].tolist(),
            "n_genes_in": expr.shape[0],
            "n_genes_out": filtered_expr.shape[0],
        }

    return filtered_expr, diagnostics



def collapse_repeated_genes(expr, threshold=0.9, min_overlap=2, return_report=False, verbose=False):
    collapsed_rows = []
    report = {"kept": [], "dropped": []}

    for gene, df_g in expr.groupby(level=0):
        n = len(df_g)

        if n == 1:
            s = df_g.iloc[0].copy()
            s.name = str(gene)
            collapsed_rows.append(s)
            report["kept"].append((gene, 1, str(gene)))
            continue

        edges = [set() for _ in range(n)]

        for i, j in combinations(range(n), 2):
            a = df_g.iloc[i].to_numpy(dtype=float, copy=False)
            b = df_g.iloc[j].to_numpy(dtype=float, copy=False)

            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < min_overlap:
                continue

            aa, bb = a[mask], b[mask]
            a_std, b_std = aa.std(), bb.std()

            if a_std < 1e-12 or b_std < 1e-12:
                r = 1.0 if (a_std < 1e-12 and b_std < 1e-12 and np.allclose(aa, bb, atol=1e-8)) else np.nan
            else:
                r = pearsonr(aa, bb)[0]

            if np.isfinite(r) and r >= threshold:
                edges[i].add(j)
                edges[j].add(i)

        seen = set()
        components = []
        for i in range(n):
            if i in seen or not edges[i]:
                continue
            stack, comp = [i], set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                seen.add(u)
                stack.extend(edges[u] - comp)
            components.append(sorted(comp))

        if not components:
            report["dropped"].append(gene)
            continue

        for k, comp in enumerate(components, start=1):
            mean_row = df_g.iloc[comp].mean(axis=0, skipna=True)
            out_name = str(gene) if len(components) == 1 else f"{gene}__part{k}"
            mean_row.name = out_name
            collapsed_rows.append(mean_row)
            report["kept"].append((gene, len(comp), out_name))

    collapsed_df = pd.DataFrame(collapsed_rows)
    if len(collapsed_df):
        collapsed_df.index = collapsed_df.index.astype(str)

    if verbose:
        print(f"collapse_repeated_genes(threshold={threshold})")
        print(f"  kept clusters: {len(report['kept'])}")
        print(f"  dropped genes (no correlated pairs): {len(report['dropped'])}")

    if return_report:
        return collapsed_df, report
    return collapsed_df



def filter_by_detection(expr, min_frac=0.3, min_value=0):
    """
    expr: genes × samples DataFrame
    min_frac: fraction of samples a gene must be detected in
    min_value: detection threshold (0 for counts, >0 for logs)
    """
    n_samples = expr.shape[1]
    detected = (expr > min_value).sum(axis=1)
    keep = detected >= int(np.ceil(min_frac * n_samples))
    return expr.loc[keep]






def detect_sample_outliers_by_day(
    expr: pd.DataFrame,
    z_thresh: float = 3.0,
    log_transform: bool = True,
    return_report: bool = True,
):
    """
    Detect within-day sample outliers using z-scored distance-to-day-mean.

    Parameters
    ----------
    expr : pd.DataFrame
        Gene x Sample matrix. Columns like '1_1', '1_2', ...
    z_thresh : float
        Z-score threshold for declaring a sample an outlier (default = 3).
    log_transform : bool
        Whether to log2(x + 1) before computing distances.
    return_report : bool
        If True, return diagnostics table.

    Returns
    -------
    expr_filtered : pd.DataFrame
        Expression matrix with outlier samples removed.
    report : pd.DataFrame (optional)
        Per-sample diagnostics (day, distance, z-score, outlier flag).
    """

    X = expr.copy()

    if log_transform:
        X = np.log2(X + 1)

    # Parse day from column names
    day_map = {}
    for c in X.columns:
        s = str(c).strip()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)_(\d+)$", s)
        if not m:
            raise ValueError(f"Column name not understood: {c}")

        day_map[c] = float(m.group(1))

    report_rows = []
    outlier_cols = []

    for day in sorted(set(day_map.values())):
        day_cols = [c for c, d in day_map.items() if d == day]

        if len(day_cols) < 2:
            # Too few samples to define outliers reliably
            continue

        X_day = X[day_cols]

        # Day-specific mean profile
        mean_profile = X_day.mean(axis=1)

        # Distance of each sample to the day mean
        distances = {
            c: np.linalg.norm(X_day[c] - mean_profile)
            for c in day_cols
        }

        dist_vals = np.array(list(distances.values()))
        mu = dist_vals.mean()
        sigma = dist_vals.std(ddof=1)

        for c, dist in distances.items():
            z = (dist - mu) / sigma if sigma > 0 else 0.0
            is_outlier = abs(z) > z_thresh

            report_rows.append({
                "sample": c,
                "day": day,
                "distance": dist,
                "z_score": z,
                "outlier": is_outlier
            })

            if is_outlier:
                outlier_cols.append(c)

    expr_filtered = expr.drop(columns=outlier_cols, errors="ignore")

    report = pd.DataFrame(report_rows)

    if return_report:
        return expr_filtered, report.sort_values(["day", "z_score"])
    else:
        return expr_filtered




def clean_dataset(
    df,
    dataset_name,
    apply_log_transform_for_outlier_detection=True
):
    """
    Apply the shared cleaning workflow to one dataset.
    """

    print(f"\nCleaning {dataset_name}")
    print("Original shape:", df.shape)

    df, replicate_diag = filter_genes_by_replicate_consistency(
        df,
        threshold_k=REPLICATE_THRESHOLD_K
    )
    print("After replicate-consistency filtering:", df.shape)

    df, duplicate_report = collapse_repeated_genes(
        df,
        threshold=DUPLICATE_CORRELATION_THRESHOLD,
        return_report=True,
        verbose=False
    )
    print("After collapsing repeated genes:", df.shape)

    df = filter_by_detection(
        df,
        min_frac=MIN_DETECTION_FRAC
    )
    print("After mostly-zero filtering:", df.shape)

    df, sample_qc = detect_sample_outliers_by_day(
        df,
        z_thresh=OUTLIER_Z_THRESHOLD,
        log_transform=apply_log_transform_for_outlier_detection
    )

    print("Outlier samples:")
    print(sample_qc[sample_qc["outlier"]])

    return df, replicate_diag, duplicate_report, sample_qc






def main():
    """Run the full cleaning pipeline."""

    acute_df, burn_df = load_datasets()

    burn_df = rename_burn_columns(burn_df)

    acute_df, burn_df = standardize_gene_names(
        acute_df,
        burn_df
    )

    acute_clean, acute_diag, acute_report, sample_qc_acute = clean_dataset(
        acute_df,
        dataset_name="Acute incision",
        apply_log_transform_for_outlier_detection=False
    )


    burn_clean, burn_diag, burn_report, sample_qc_burn = clean_dataset(
        burn_df,
        dataset_name="Burn",
        apply_log_transform_for_outlier_detection=True
    )

    acute_clean.to_csv(ACUTE_OUTPUT_PATH, sep="\t")
    burn_clean.to_csv(BURN_OUTPUT_PATH, sep=",")

    print("\nCleaned datasets saved.")
    
    
    
if __name__ == "__main__":
    main()