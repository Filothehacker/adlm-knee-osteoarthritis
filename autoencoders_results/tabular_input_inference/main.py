import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

REPO_ROOT = os.path.dirname(PROJECT_ROOT)
AUTOENCODERS_DIR = os.path.join(REPO_ROOT, "autoencoders")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if AUTOENCODERS_DIR not in sys.path:
    sys.path.append(AUTOENCODERS_DIR)

from tabular_input_inference.infer import run_inference
from inference.cluster import run_kmeans, clusters_stats
from inference.tsne_visualization import tsne_plot_with_clusters


def load_features_npz(features_dir: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    feat_path = os.path.join(features_dir, f"features_{side}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features file not found for side={side}: {feat_path}")

    data = np.load(feat_path, allow_pickle=True)
    return data["ids"], data["features"]


def save_features_npz(features_dir: str, side: str, ids: np.ndarray, features: np.ndarray) -> str:
    os.makedirs(features_dir, exist_ok=True)
    feat_path = os.path.join(features_dir, f"features_{side}.npz")
    np.savez(feat_path, ids=ids, features=features)
    return feat_path


def read_test_ids(side: str) -> set[str]:
    path = os.path.join(AUTOENCODERS_DIR, "test_ids", f"tab_input_ae_{side}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test-IDs file not found for side={side}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def split_features_by_test_ids(
    ids: np.ndarray,
    features: np.ndarray,
    test_ids: set[str],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    ids_str = np.array([str(x) for x in ids])

    is_test = np.array([pid in test_ids for pid in ids_str], dtype=bool)
    is_non_test = ~is_test

    test_ids_arr = ids_str[is_test]
    test_feat_arr = features[is_test]

    non_test_ids_arr = ids_str[is_non_test]
    non_test_feat_arr = features[is_non_test]

    return (test_ids_arr, test_feat_arr), (non_test_ids_arr, non_test_feat_arr)


def run_clustering(features_dir: str, csv_dir: str, side: str, k: int = 5):
    ids, features = load_features_npz(features_dir, side)
    patients_features = {str(pid): features[i] for i, pid in enumerate(ids)}

    df_clusters = run_kmeans(patients_features, k=k)

    os.makedirs(csv_dir, exist_ok=True)
    out_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    df_clusters.to_csv(out_path, index=False)
    print(f"[{side}] Saved cluster assignments to {out_path}")
    return df_clusters


def run_stats(csv_dir: str, clinical_csv: str, side: str) -> None:
    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster CSV not found for side={side}: {clusters_path}")

    df_clinical = pd.read_csv(clinical_csv)
    df_clinical.columns = df_clinical.columns.str.strip()

    df_clusters = pd.read_csv(clusters_path)

    df_stats = clusters_stats(df_clusters, df_clinical, side=side)
    print(f"\n[{side}] Cluster stats:\n")
    print(df_stats.to_string(index=False))

    stats_path = os.path.join(csv_dir, f"cluster_stats_{side}.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"[{side}] Saved cluster stats to {stats_path}")


def run_tsne(features_dir: str, csv_dir: str, plots_dir: str, side: str, n_components: int = 3) -> None:
    ids, features = load_features_npz(features_dir, side)

    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster CSV not found for side={side}: {clusters_path}")

    df_clusters = pd.read_csv(clusters_path)

    patients_features = {
        str(pid): torch.from_numpy(features[i]).unsqueeze(0)
        for i, pid in enumerate(ids)
    }

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"tsne_{side}_{n_components}d.png")

    print(f"[{side}] Running t-SNE with existing clusters...")
    tsne_plot_with_clusters(
        patients_features=patients_features,
        df_clusters=df_clusters,
        n_components=n_components,
        output_path=out_path,
    )
    print(f"[{side}] Saved t-SNE plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full MRI + tabular input pipeline: inference, clustering, stats, t-SNE (non-masked model)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to model weights.",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="left",
    )
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="csv/clinical00_cleaned.csv",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tsne_components", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()

    weights_path = args.weights_path or "./weights_tabular_input_ae/best_tabular_input_ae_training_phase.pth"
    print(f"Weights: {weights_path}")

    results_dir = os.path.join("results", "tabular_input_ae")
    os.makedirs(results_dir, exist_ok=True)

    sides = ["left", "right"] if args.side == "both" else [args.side]

    for side in sides:
        print(f"\n=== SIDE: {side} ===")

        features_dir_all = os.path.join(results_dir, "features")
        run_inference(
            data_root=args.data_root,
            clinical_csv_path=args.clinical_csv,
            side=side,
            weights_path=weights_path,
            max_patients=args.max_patients,
            features_dir=features_dir_all,
        )

        #  Split into TEST vs NON-TEST using saved holdout IDs
        test_ids = read_test_ids(side=side)
        ids_all, feats_all = load_features_npz(features_dir_all, side)

        (ids_test, feats_test), (ids_non_test, feats_non_test) = split_features_by_test_ids(
            ids=ids_all,
            features=feats_all,
            test_ids=test_ids,
        )

        subsets = [
            ("test", ids_test, feats_test),
            ("non_test", ids_non_test, feats_non_test),
        ]

        for subset_name, subset_ids, subset_feats in subsets:
            subset_root = os.path.join(results_dir, subset_name)
            features_dir = os.path.join(subset_root, "features")
            csv_dir = os.path.join(subset_root, "csv")
            plots_dir = os.path.join(subset_root, "plots")
            os.makedirs(subset_root, exist_ok=True)

            save_features_npz(features_dir, side, subset_ids, subset_feats)

            print(f"\n--- SUBSET: {subset_name} ({len(subset_ids)} patients) ---")

            # Clustering
            run_clustering(features_dir=features_dir, csv_dir=csv_dir, side=side, k=args.k)

            # Stats
            run_stats(csv_dir=csv_dir, clinical_csv=args.clinical_csv, side=side)

            # t-SNE
            run_tsne(
                features_dir=features_dir,
                csv_dir=csv_dir,
                plots_dir=plots_dir,
                side=side,
                n_components=args.tsne_components,
            )


if __name__ == "__main__":
    main()
