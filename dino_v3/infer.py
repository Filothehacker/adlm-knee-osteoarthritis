"""
Feature extraction for the DINOv3 track.

For each patient in the dataset, passes the 3D MRI volume through the
DINOv3 backbone (slice-by-slice with mean-pooling) and saves the resulting
feature vectors to a .npz file.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from dino_v3.data import iter_mri_dataset
from dino_v3.model import build_volumetric_model


def run_inference(
    data_root: str,
    side: str,
    weights_path: str,
    features_dir: str,
    max_patients: int | None = None,
) -> str:
    """
    Run DINOv3 feature extraction on all patients for the given side.

    Saves:
        {features_dir}/features_{side}.npz
            - ids:      np.ndarray of patient IDs (str)
            - features: np.ndarray of shape [N, embed_dim]

    Returns:
        Path to the saved .npz file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_volumetric_model(weights_path=weights_path, device=device)

    print(f"[{side}] Streaming MRIs from {data_root} ...")

    patient_ids = []
    feature_list = []
    processed = 0

    with torch.no_grad():
        for p_id, vol in iter_mri_dataset(data_root, side=side, max_patients=max_patients):
            # vol: [1, D, 224, 224] -> [1, 1, D, 224, 224]
            vol = vol.unsqueeze(0).to(device)

            # [1, embed_dim]
            feats = model(vol)

            patient_ids.append(p_id)
            feature_list.append(feats.squeeze(0).cpu().numpy())
            processed += 1
            print(f"[{side}] Processed {processed} patients (last ID: {p_id})")

    if not feature_list:
        print(f"[{side}] No patients processed, nothing to save.")
        os.makedirs(features_dir, exist_ok=True)
        return os.path.join(features_dir, f"features_{side}.npz")

    os.makedirs(features_dir, exist_ok=True)
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    np.savez(
        feat_out,
        ids=np.array(patient_ids),
        features=np.stack(feature_list),  # [N, embed_dim]
    )
    print(f"[{side}] Saved features ({np.stack(feature_list).shape}) to {feat_out}")
    return feat_out
