"""
Cluster quality evaluation for the DINOv3 (and base AE) tracks.

Metrics computed
----------------
* NMI  — Normalised Mutual Information between cluster labels and KL grades.
* ARI  — Adjusted Rand Index between cluster labels and KL grades.
* Spearman ρ — ordinal correlation between cluster label and WOMAC total score.
* Kendall τ  — ordinal correlation between cluster label and WOMAC total score.
* Kruskal-Wallis H — non-parametric one-way ANOVA of WOMAC across clusters.
* Dunn post-hoc    — pairwise Dunn test (Bonferroni correction) for all
                     cluster pairs that show a significant KW result.

Outputs
-------
A plain-text report saved to ``output_path``.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def _womac_col(side: str) -> str:
    """Return the WOMAC total-score column name for *side*."""
    return f"V00WOMTS{side.upper()[0]}"  # V00WOMTSL or V00WOMTSR


def _kl_col(side: str) -> str:
    """Return the KL-grade column name for *side*."""
    return f"V00XRKL_{side.upper()[0]}"  # V00XRKL_L or V00XRKL_R


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(
    clusters_csv: str,
    clinical_csv: str,
    side: str,
    output_path: str,
) -> str:
    """
    Evaluate cluster quality against clinical labels and save a text report.

    Parameters
    ----------
    clusters_csv : str
        Path to the cluster-assignment CSV produced by K-means, with columns
        ``["ID", "cluster"]``.
    clinical_csv : str
        Path to the cleaned clinical CSV (e.g. ``csv/clinical00_cleaned.csv``).
        Must contain columns for KL grade and WOMAC total score for *side*.
    side : str
        ``"left"`` or ``"right"``.
    output_path : str
        Destination for the plain-text evaluation report.

    Returns
    -------
    str
        The full text of the evaluation report (also written to *output_path*).
    """
    df_clusters = pd.read_csv(clusters_csv)
    df_clusters.columns = df_clusters.columns.str.strip()

    df_clinical = pd.read_csv(clinical_csv)
    df_clinical.columns = df_clinical.columns.str.strip()

    # Normalise ID to int for reliable joining
    df_clusters["ID"] = pd.to_numeric(df_clusters["ID"], errors="coerce")
    df_clinical["ID"] = pd.to_numeric(df_clinical["ID"], errors="coerce")

    merged = df_clusters.merge(df_clinical, on="ID", how="inner")

    kl_col = _kl_col(side)
    womac_col = _womac_col(side)

    missing = [c for c in [kl_col, womac_col] if c not in merged.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} not found in clinical CSV. "
            f"Available: {list(merged.columns)}"
        )

    report_lines = [
        "=" * 70,
        f"Cluster Evaluation Report — side={side}",
        "=" * 70,
        f"Patients in clusters CSV : {len(df_clusters)}",
        f"Patients after join      : {len(merged)}",
        f"KL-grade column          : {kl_col}",
        f"WOMAC column             : {womac_col}",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. NMI / ARI vs KL grades
    # ------------------------------------------------------------------
    valid_kl = merged[[kl_col, "cluster"]].dropna()
    if len(valid_kl) < 2:
        report_lines += [
            "NMI / ARI",
            "---------",
            "  Skipped — fewer than 2 patients with valid KL grades.",
            "",
        ]
    else:
        kl_labels = valid_kl[kl_col].astype(int).values
        cluster_labels = valid_kl["cluster"].astype(int).values

        nmi = normalized_mutual_info_score(kl_labels, cluster_labels)
        ari = adjusted_rand_score(kl_labels, cluster_labels)

        report_lines += [
            "NMI / ARI vs KL grades",
            "----------------------",
            f"  NMI  = {nmi:.4f}",
            f"  ARI  = {ari:.4f}",
            f"  N    = {len(valid_kl)}",
            "",
        ]

    # ------------------------------------------------------------------
    # 2. Spearman / Kendall correlation: cluster label vs WOMAC
    # ------------------------------------------------------------------
    valid_womac = merged[[womac_col, "cluster"]].dropna()
    if len(valid_womac) < 3:
        report_lines += [
            "Spearman / Kendall (cluster vs WOMAC)",
            "--------------------------------------",
            "  Skipped — fewer than 3 patients with valid WOMAC scores.",
            "",
        ]
    else:
        x = valid_womac["cluster"].astype(float).values
        y = valid_womac[womac_col].astype(float).values

        sp_rho, sp_p = stats.spearmanr(x, y)
        kt_tau, kt_p = stats.kendalltau(x, y)

        report_lines += [
            "Spearman / Kendall ordinal correlation (cluster vs WOMAC)",
            "----------------------------------------------------------",
            f"  Spearman ρ = {sp_rho:.4f}  (p = {sp_p:.4e})",
            f"  Kendall  τ = {kt_tau:.4f}  (p = {kt_p:.4e})",
            f"  N          = {len(valid_womac)}",
            "",
        ]

    # ------------------------------------------------------------------
    # 3. Kruskal-Wallis + Dunn post-hoc
    # ------------------------------------------------------------------
    valid_kw = merged[[womac_col, "cluster"]].dropna()
    groups = [
        grp[womac_col].values
        for _, grp in valid_kw.groupby("cluster")
        if len(grp) >= 2
    ]

    if len(groups) < 2:
        report_lines += [
            "Kruskal-Wallis + Dunn post-hoc",
            "--------------------------------",
            "  Skipped — fewer than 2 clusters with ≥2 patients.",
            "",
        ]
    else:
        kw_stat, kw_p = stats.kruskal(*groups)

        report_lines += [
            "Kruskal-Wallis test (WOMAC across clusters)",
            "--------------------------------------------",
            f"  H = {kw_stat:.4f}  (p = {kw_p:.4e})",
            f"  N = {len(valid_kw)}",
            "",
        ]

        if kw_p < 0.05:
            try:
                import scikit_posthocs as sp

                dunn = sp.posthoc_dunn(
                    valid_kw,
                    val_col=womac_col,
                    group_col="cluster",
                    p_adjust="bonferroni",
                )
                report_lines += [
                    "Dunn post-hoc (Bonferroni-corrected p-values)",
                    "----------------------------------------------",
                    dunn.to_string(),
                    "",
                ]
            except ImportError:
                report_lines += [
                    "Dunn post-hoc",
                    "-------------",
                    "  scikit-posthocs not installed; skipping Dunn test.",
                    "",
                ]
        else:
            report_lines += [
                "Dunn post-hoc",
                "-------------",
                "  Skipped — KW p-value ≥ 0.05 (no significant overall difference).",
                "",
            ]

    # ------------------------------------------------------------------
    # 4. Per-cluster WOMAC summary
    # ------------------------------------------------------------------
    report_lines += [
        "Per-cluster WOMAC summary",
        "-------------------------",
    ]
    for cluster_id, grp in valid_kw.groupby("cluster"):
        w = grp[womac_col]
        report_lines.append(
            f"  Cluster {cluster_id:>2d} — N={len(w):>4d}  "
            f"mean={w.mean():.2f}  median={w.median():.2f}  "
            f"std={w.std():.2f}"
        )
    report_lines += ["", "=" * 70]

    report = "\n".join(report_lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(report)

    print(f"[{side}] Evaluation report saved to {output_path}")
    return report
