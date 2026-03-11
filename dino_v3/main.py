"""
DINOv3 MRI clustering pipeline.

Steps:
  1. infer.py    — extract per-patient feature vectors from 3D MRI via DINOv3
  2. cluster.py  — K-means (k=5) on the feature vectors
  3. score_moaks.py — per-cluster MOAKS mean scores + patient counts
  4. t-SNE visualization colored by cluster

Usage:
  uv run dino_v3/main.py --data_root <path> --weights_path <path> --side <left|right> [--k 5]
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from dino_v3.infer import run_inference
from dino_v3.cluster import load_features, run_clustering
from dino_v3.score_moaks import run_moaks_scoring
from inference.tsne_visualization import tsne_plot_with_clusters


def run_tsne(features_dir: str, csv_dir: str, plots_dir: str, side: str,
             n_components: int = 2) -> None:
    import pandas as pd

    ids, features = load_features(features_dir, side)

    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster CSV not found: {clusters_path}")
    df_clusters = pd.read_csv(clusters_path)

    patients_features = {
        str(pid): torch.from_numpy(features[i]).unsqueeze(0)
        for i, pid in enumerate(ids)
    }

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"tsne_{side}_{n_components}d.png")

    print(f"[{side}] Running t-SNE (n_components={n_components}) ...")
    tsne_plot_with_clusters(
        patients_features=patients_features,
        df_clusters=df_clusters,
        n_components=n_components,
        output_path=out_path,
    )
    print(f"[{side}] Saved t-SNE plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINOv3 MRI clustering pipeline"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
        help="Root folder containing patient MRI data",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights_dinov3/dinov3_vitb16_pretrain_lvd1689m.pth",
        help="Path to DINOv3 pretrained weights (.pth)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Which knee side to process",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of K-means clusters (default 5)",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit on number of patients (for debugging)",
    )
    parser.add_argument(
        "--moaks_csv_dir",
        type=str,
        default="csv",
        help="Directory containing MOAK_L.csv and MOAK_R.csv",
    )
    parser.add_argument(
        "--tsne_components",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Number of t-SNE dimensions for visualization",
    )
    args = parser.parse_args()

    results_dir = os.path.join("dino_v3_results", args.side)
    features_dir = os.path.join(results_dir, "features")
    csv_dir = os.path.join(results_dir, "csv")
    plots_dir = os.path.join(results_dir, "plots")

    # MOAKS CSV: left -> MOAK_L.csv, right -> MOAK_R.csv
    moaks_suffix = "L" if args.side == "left" else "R"
    moaks_csv = os.path.join(args.moaks_csv_dir, f"MOAK_{moaks_suffix}.csv")

    print(f"\n=== DINOv3 pipeline | side={args.side} | k={args.k} ===\n")

    # 1) Feature extraction
    run_inference(
        data_root=args.data_root,
        side=args.side,
        weights_path=args.weights_path,
        features_dir=features_dir,
        max_patients=args.max_patients,
    )

    # 2) Clustering
    df_clusters = run_clustering(
        features_dir=features_dir,
        csv_dir=csv_dir,
        side=args.side,
        k=args.k,
    )

    # 3) MOAKS scoring
    run_moaks_scoring(
        df_clusters=df_clusters,
        moaks_csv=moaks_csv,
        side=args.side,
        csv_dir=csv_dir,
    )

    # 4) t-SNE visualization
    run_tsne(
        features_dir=features_dir,
        csv_dir=csv_dir,
        plots_dir=plots_dir,
        side=args.side,
        n_components=args.tsne_components,
    )

    print(f"\n=== Done. Results in {results_dir}/ ===")


if __name__ == "__main__":
    main()
'''
  1. Input: [1, 1, 160, 224, 224] — one patient's full 3D volume                                                    
  2. Unbatch slices: reshape to [160, 3, 224, 224] — 160 independent 2D images
     (The 3 is the RGB channels — DINOv3 was pretrained on natural images so it expects 3-channel input. Since MRI      
      slices are grayscale, we just repeat the single channel 3 times)                                    
  3. 2D ViT forward: backbone processes all 160 slices → [160, 768] CLS tokens                                      
  4. Max-pool: .max(dim=1).values → [768] — for each of the 768 feature dimensions, keep the highest activation     
  across all slices 
  the CLS token is a single [768]-dim vector that summarizes the whole slice — that's the whole point of   
  the CLS token in ViT: it aggregates global information from all 196 patch tokens via self-attention. So per slice 
  you get one [768] vector, and after max-pooling across 160 slices you get one [768] vector per patient.                                                                                                
  5. Output: one [768]-dim vector per patient representing "the most activated response" seen anywhere in the volume
  
  '''