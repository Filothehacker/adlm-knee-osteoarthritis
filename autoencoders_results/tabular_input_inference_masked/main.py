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

from tabular_input_inference_masked.infer import run_inference
from inference.cluster import run_kmeans, clusters_stats
from inference.tsne_visualization import tsne_plot_with_clusters


def load_features_npz(features_dir: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    feat_path = os.path.join(features_dir, f"features_{side}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features file not found for side={side}: {feat_path}")

    data = np.load(feat_path, allow_pickle=True)
    return data["ids"], data["features"]


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
        description="Full MRI + tabular input pipeline: inference, clustering, stats, t-SNE (masked AE)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
    )
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--side", type=str, choices=["left", "right", "both"], default="left")
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--clinical_csv", type=str, default="csv/clinical00_cleaned.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tsne_components", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()

    weights_path = args.weights_path or "./weights_ae_masked/best_ae_masked_training_phase.pth"
    print(f"Weights: {weights_path}")

    results_dir = os.path.join("results", "tabular_input_ae_masked")
    features_dir = os.path.join(results_dir, "features")
    csv_dir = os.path.join(results_dir, "csv")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)

    sides = ["left", "right"] if args.side == "both" else [args.side]

    for side in sides:
        print(f"\n=== SIDE: {side} ===")

        run_inference(
            data_root=args.data_root,
            clinical_csv_path=args.clinical_csv,
            side=side,
            weights_path=weights_path,
            max_patients=args.max_patients,
            features_dir=features_dir,
        )

        run_clustering(features_dir=features_dir, csv_dir=csv_dir, side=side, k=args.k)
        run_stats(csv_dir=csv_dir, clinical_csv=args.clinical_csv, side=side)
        run_tsne(
            features_dir=features_dir,
            csv_dir=csv_dir,
            plots_dir=plots_dir,
            side=side,
            n_components=args.tsne_components,
        )


if __name__ == "__main__":
    main()
