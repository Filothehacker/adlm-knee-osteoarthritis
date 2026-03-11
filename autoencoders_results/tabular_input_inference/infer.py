import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

REPO_ROOT = os.path.dirname(PROJECT_ROOT)
AUTOENCODERS_DIR = os.path.join(REPO_ROOT, "autoencoders")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if AUTOENCODERS_DIR not in sys.path:
    sys.path.append(AUTOENCODERS_DIR)

from tabular_input_inference.data import iter_mri_dataset_with_tabular_inputs, TAB_INPUTS


def build_feature_extractor(weights_path: str, device: torch.device, side: str) -> nn.Module:
    """
    Loads the AE that takes (MRI, tab_x[T]) and returns z features.
    """
    from ae_tabular_input.model import build_tabular_input_ae

    num_tabular_inputs = len(TAB_INPUTS)

    model = build_tabular_input_ae(
        pretrained_ae_path=None,
        device=device,
        latent_channels=64,
        num_tabular_inputs=num_tabular_inputs,
        tabular_latent_dim=4,
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def extract_features(model: nn.Module, vol: torch.Tensor, tab_x: torch.Tensor) -> torch.Tensor:
    """
    vol:   [B, 1, D, H, W]
    tab_x: [B, T]
    Returns:
        feats: [B, 64]
    """
    encoded_mri = model.encoder(vol)            # [B, C, D', H', W']
    encoded_tab = model.tabular_encoder(tab_x)  # [B, tab_lat]

    B, C, D, H, W = encoded_mri.shape
    encoded_tab = encoded_tab.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)

    z = torch.cat([encoded_mri, encoded_tab], dim=1)
    z = model.channel_projection(z)             # [B, 64, D', H', W']
    feats = z.mean(dim=(2, 3, 4))               # [B, 64]
    return feats


def run_inference(
    data_root: str,
    clinical_csv_path: str,
    side: str,
    weights_path: str,
    max_patients: int | None = None,
    features_dir: str = "features",
) -> str:
    """
    Streams patients and saves:
        features_{side}.npz with arrays: ids, features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor(weights_path=weights_path, device=device, side=side)

    print(f"[{side}] Streaming MRIs + tabular inputs from {data_root} ...")

    patients_features: Dict[str, torch.Tensor] = {}
    processed = 0

    with torch.no_grad():
        for pid, vol, tab_x in iter_mri_dataset_with_tabular_inputs(
            dataset_root=data_root,
            clinical_csv_path=clinical_csv_path,
            side=side,
            max_patients=max_patients,
        ):
            vol = vol.unsqueeze(0).to(device)     # [1, 1, D, 224, 224]
            tab_x = tab_x.unsqueeze(0).to(device) # [1, T]

            feats = extract_features(model=model, vol=vol, tab_x=tab_x)  # [1, 64]
            patients_features[pid] = feats.cpu()

            processed += 1
            if processed % 10 == 0:
                print(f"[{side}] Processed {processed} patients (last ID: {pid})")

    os.makedirs(features_dir, exist_ok=True)
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    if not patients_features:
        np.savez(feat_out, ids=np.array([]), features=np.array([]))
        print(f"[{side}] No patients processed, saved empty {feat_out}")
        return feat_out

    patient_ids = np.array(list(patients_features.keys()))
    features = np.stack([t.numpy().squeeze() for t in patients_features.values()])  # [N, 64]
    np.savez(feat_out, ids=patient_ids, features=features)

    print(f"[{side}] Saved features to {feat_out}")
    return feat_out
