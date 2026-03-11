"""
K-means clustering on DINOv3 features.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(ids: np.ndarray, features: np.ndarray, k: int = 5) -> pd.DataFrame:
    """
    Fit K-means on feature matrix and return a DataFrame with columns [ID, cluster].

    Args:
        ids:      np.ndarray of patient ID strings, shape [N]
        features: np.ndarray of shape [N, F]
        k:        number of clusters (default 5)

    Returns:
        DataFrame with columns ["ID", "cluster"]
    """
    n_samples = features.shape[0]
    if n_samples < k:
        print(f"Warning: n_samples={n_samples} < k={k}, clamping k to {n_samples}.")
        k = n_samples

    print(f"Running K-means with k={k} on {n_samples} patients "
          f"(feature dim {features.shape[1]}) ...")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    df = pd.DataFrame({"ID": ids, "cluster": labels})
    print(f"K-means complete. Cluster sizes:\n{df['cluster'].value_counts().sort_index()}")
    return df


def load_features(features_dir: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Load features_{side}.npz and return (ids, features)."""
    path = os.path.join(features_dir, f"features_{side}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["ids"], data["features"]


def run_clustering(features_dir: str, csv_dir: str, side: str, k: int = 5) -> pd.DataFrame:
    """
    Load features, run K-means, save mri_clusters_{side}.csv, return DataFrame.
    """
    ids, features = load_features(features_dir, side)
    df_clusters = run_kmeans(ids, features, k=k)

    os.makedirs(csv_dir, exist_ok=True)
    out_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    df_clusters.to_csv(out_path, index=False)
    print(f"[{side}] Saved cluster assignments to {out_path}")
    return df_clusters
