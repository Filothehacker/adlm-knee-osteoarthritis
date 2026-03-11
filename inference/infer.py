import os
import sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

AUTOENCODERS_DIR = os.path.join(PROJECT_ROOT, "autoencoders")
if AUTOENCODERS_DIR not in sys.path:
    sys.path.append(AUTOENCODERS_DIR)

from inference.data import iter_mri_dataset
from typing import Literal

ModelName = Literal["resnet50", "autoencoder"]

sys.path.append(os.path.abspath("MedicalNet"))
#from models.resnet import resnet50


def build_feature_extractor(
    model_name: ModelName,
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Return a model that, given vol [B, 1, D, H, W], outputs an embedding [B, F].
    All model-specific stuff stays in here.
    """
    if model_name == "resnet50":
        base_model = resnet50(
            sample_input_D=64,
            sample_input_H=224,
            sample_input_W=224,
            num_seg_classes=1,
        )

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        base_model.load_state_dict(state_dict, strict=False)

        # remove classification head -> keep feature extractor
        feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
        return feature_extractor

    elif model_name == "autoencoder":
        
        from ae_filippo.model import Autoencoder3D

        ae = Autoencoder3D()
        state_dict = torch.load(weights_path, map_location=device)
        ae.load_state_dict(state_dict)
        ae = ae.to(device)
        ae.eval()
        return ae
    
    elif model_name == "autoencoder_pain":
        from ae_pain.model import build_pain_ae

        # You can adjust latent_channels / num_pain_outputs as needed
        model = build_pain_ae(
            pretrained_ae_path=None,
            device=device,
            latent_channels=64,
            num_pain_outputs=1,
        )

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def extract_features(
    model: nn.Module,
    vol: torch.Tensor,
    model_name: ModelName,
) -> torch.Tensor:
    """
    Forward one volume through the model and return a [1, F] embedding tensor.
    """
    if model_name == "resnet50":
        # model outputs [B, 2048, D', H', W']
        feats = model(vol)
        feats = feats.mean(dim=(2, 3, 4))  # -> [B, 2048]
        return feats

    elif model_name == "autoencoder":
        # Encode volume -> [B, C, D, H, W]
        z = model.encoder(vol)

        # Global average pooling over spatial dims -> [B, C]
        z = z.mean(dim=(2, 3, 4))
        return z
    
    elif model_name == "autoencoder_pain":
        # Pain AE also has .encoder; we ignore the pain head, just use latent
        z = model.encoder(vol)  # [B, C, D, H, W]
        z = z.mean(dim=(2, 3, 4))
        return z


    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_inference(
    data_root: str,
    side: str,
    weights_path: str = "MedicalNet/pretrain/resnet_50.pth",
    max_patients: int | None = None,
    features_dir: str = "features",
    model_name: ModelName = "resnet50",
) -> str:
    """
    Run inference on MRIs for a given side with a chosen feature extractor and save latent features.

    Saves:
        {features_dir}/features_{side}.npz
            - ids: np.ndarray of patient IDs (str)
            - features: np.ndarray of shape [N, F] where F depends on the model.

    Returns:
        Path to the saved .npz file.
    """
    # Build feature extractor for the chosen model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor(
        model_name=model_name,
        weights_path=weights_path,
        device=device,
    )

    print(f"[{side}] Streaming MRIs from {data_root} ...")

    patients_features: dict[str, torch.Tensor] = {}
    processed = 0

    with torch.no_grad():
        for p_id, vol in iter_mri_dataset(
            data_root,
            side=side,
            max_patients=max_patients,
        ):
            # vol: [1, D, 224, 224] -> [1, 1, D, 224, 224]
            vol = vol.unsqueeze(0).to(device)

            # Generic feature extraction for the chosen model -> [1, F]
            feats = extract_features(
                model=model,
                vol=vol,
                model_name=model_name,
            )

            patients_features[p_id] = feats.cpu()
            processed += 1
            print(f"[{side}] Processed {processed} patients (last ID: {p_id})")

    if not patients_features:
        print(f"[{side}] No patients processed, nothing to save.")
        os.makedirs(features_dir, exist_ok=True)
        return os.path.join(features_dir, f"features_{side}.npz")

    # Save latent features
    os.makedirs(features_dir, exist_ok=True)
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    patient_ids = np.array(list(patients_features.keys()))
    features = np.stack(
        [t.detach().numpy().squeeze() for t in patients_features.values()]
    )  # [N, F]

    np.savez(feat_out, ids=patient_ids, features=features)
    print(f"[{side}] Saved features to {feat_out}")

    return feat_out
