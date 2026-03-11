"""
Per-cluster MOAKS scoring for the DINOv3 track.

For each cluster:
  - Joins cluster assignments to the MOAKS CSV (MOAK_L.csv or MOAK_R.csv)
    on patient ID.
  - Computes the mean score for every MOAKS variable.
  - Reports the count of patients in each cluster that have MOAKS data.

Outputs:
  {csv_dir}/moaks_cluster_stats_{side}.csv
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd


def run_moaks_scoring(
    df_clusters: pd.DataFrame,
    moaks_csv: str,
    side: str,
    csv_dir: str,
) -> pd.DataFrame:
    """
    Compute per-cluster MOAKS statistics and save results.

    Args:
        df_clusters: DataFrame with columns ["ID", "cluster"]
        moaks_csv:   Path to MOAK_L.csv or MOAK_R.csv
        side:        "left" or "right" (used only for output filenames)
        csv_dir:     Directory to save the output CSV

    Returns:
        DataFrame with one row per cluster containing:
          - cluster:       cluster index
          - n_total:       total patients in cluster
          - n_with_moaks:  patients whose ID appears in the MOAKS CSV
          - <var>_mean:    mean score per MOAKS variable (NaN if no data)
    """
    df_moaks = pd.read_csv(moaks_csv)
    # Strip whitespace from column names and all string values
    df_moaks.columns = df_moaks.columns.str.strip()
    df_moaks = df_moaks.apply(
        lambda col: col.str.strip() if col.dtype == object else col
    )
    # Cast every column (except ID) to numeric — the CSV stores values as strings
    moaks_vars = [c for c in df_moaks.columns if c != "ID"]
    df_moaks[moaks_vars] = df_moaks[moaks_vars].apply(
        pd.to_numeric, errors="coerce"
    )

    # Normalise ID types to int for reliable joining
    df_clusters = df_clusters.copy()
    df_clusters["ID"] = pd.to_numeric(df_clusters["ID"], errors="coerce")
    df_moaks["ID"] = pd.to_numeric(df_moaks["ID"], errors="coerce")

    # Join: keep all cluster rows (left join) so n_total is always correct
    merged = df_clusters.merge(df_moaks, on="ID", how="left")

    results = []
    for cluster_id, group in merged.groupby("cluster"):
        row = {"cluster": cluster_id, "n_total": len(group)}

        # Count patients with MOAKS data (at least one non-NaN MOAKS column)
        has_moaks = group[moaks_vars].notna().any(axis=1)
        row["n_with_moaks"] = int(has_moaks.sum())

        # Mean per variable (uses only non-NaN entries automatically)
        for var in moaks_vars:
            row[f"{var}_mean"] = group[var].mean()

        results.append(row)

    df_stats = pd.DataFrame(results)

    os.makedirs(csv_dir, exist_ok=True)
    out_path = os.path.join(csv_dir, f"moaks_cluster_stats_{side}.csv")
    df_stats.to_csv(out_path, index=False)
    print(f"[{side}] Saved MOAKS cluster stats to {out_path}")

    # Print a summary to stdout
    print(f"\n[{side}] MOAKS cluster summary:")
    summary_cols = ["cluster", "n_total", "n_with_moaks"]
    print(df_stats[summary_cols].to_string(index=False))

    return df_stats
