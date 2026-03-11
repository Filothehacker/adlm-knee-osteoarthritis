import os
import tarfile
import tempfile
import shutil
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms


EXCLUDE_L = {"9070923", "9388265", "9594253", "9860568"}
EXCLUDE_R = {"9004315", "9537947", "9637394", "9462278", "9522128"}

TAB_INPUTS_L = [
    "V00WOMTSL", "V00KOOSKPL", "V00KOOSYML", "V00KOOSQOL",
    "V00XRJSL_L", "V00XRJSM_L",
    "V00XRSCFM_L", "V00XRSCFL_L",
    "V00XRSCTM_L", "V00XRSCTL_L",
    "V00XROSFM_L", "V00XROSFL_L",
    "V00XROSTM_L", "V00XROSTL_L",
    "V00XRKL_L",
    "V99ELKVSAF",
]

TAB_INPUTS_R = [
    "V00WOMTSR", "V00KOOSKPR", "V00KOOSYMR", "V00KOOSQOL",
    "V00XRJSL_R", "V00XRJSM_R",
    "V00XRSCFM_R", "V00XRSCFL_R",
    "V00XRSCTM_R", "V00XRSCTL_R",
    "V00XROSFM_R", "V00XROSFL_R",
    "V00XROSTM_R", "V00XROSTL_R",
    "V00XRKL_R",
    "V99ERKVSAF",
]

_transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def normalize_tab(col: str, raw: float) -> float:
    if col.startswith(("V00WOMTS", "V00KOOS")):
        return raw / 100.0
    if col.startswith("V00XRKL"):
        return raw / 4.0
    if col.startswith("V00XR"):
        return raw / 3.0
    if col.startswith("V99"):
        return raw
    raise ValueError(f"Unknown column for normalization: {col}")


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return bool(pd.isna(v))


def list_patient_ids_fast(dataset_root: str, side: str = "left", max_patients: Optional[int] = None) -> List[str]:
    subset_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    ids: List[str] = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for pid in os.listdir(subset_path):
            if os.path.isdir(os.path.join(subset_path, pid, "mri", side)):
                ids.append(str(pid))
    if max_patients:
        ids = ids[:max_patients]
    return ids


def reconstruct_mri_from_tar(tar_path: str) -> Optional[torch.Tensor]:
    temp_dir = tempfile.mkdtemp()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        dicom_files: List[str] = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.startswith("._") or f.startswith("."):
                    continue
                dicom_files.append(os.path.join(root, f))

        if not dicom_files:
            return None

        dicoms = []
        for f in dicom_files:
            try:
                d = pydicom.dcmread(f, force=True)
                if hasattr(d, "pixel_array"):
                    dicoms.append(d)
            except Exception:
                continue

        if not dicoms:
            return None

        dicoms.sort(key=lambda d: float(getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))))

        try:
            volume = np.stack([d.pixel_array for d in dicoms], axis=0).astype(np.float32)
        except ValueError:
            return None

        v_min = volume.min()
        v_max = volume.max()
        if v_max - v_min > 0:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = np.zeros_like(volume)

        resized_slices = []
        for i in range(volume.shape[0]):
            img = Image.fromarray((volume[i] * 255).astype(np.uint8))
            img = _transform_2d.transforms[0](img)  # resize only
            resized_slices.append(np.array(img))

        volume = np.stack(resized_slices, axis=0).astype(np.float32) / 255.0
        vol = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # [1, D, 224, 224]
        vol = (vol - 0.5) / 0.5
        return vol
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_single_patient_mri(dataset_root: str, patient_id: str, side: str = "left") -> Optional[torch.Tensor]:
    subset_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        patient_path = os.path.join(subset_path, patient_id)
        mri_side_dir = os.path.join(patient_path, "mri", side)

        if not os.path.isdir(mri_side_dir):
            continue

        tar_files = [f for f in os.listdir(mri_side_dir) if f.endswith(".tar.gz")]
        if not tar_files:
            continue

        tar_path = os.path.join(mri_side_dir, tar_files[0])
        tensor = reconstruct_mri_from_tar(tar_path)
        if tensor is not None:
            return tensor
    return None


def _build_tab_vector_and_mask(row: Dict[str, Any], cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    vals: List[float] = []
    mask: List[float] = []
    for c in cols:
        v = row.get(c, None)
        if is_missing(v):
            vals.append(0.0)
            mask.append(0.0)
        else:
            vals.append(normalize_tab(c, float(v)))
            mask.append(1.0)
    return torch.tensor(vals, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def iter_mri_dataset_with_tabular_inputs(
    dataset_root: str,
    clinical_csv_path: str,
    side: str = "left",
    max_patients: Optional[int] = None,
) -> Generator[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Yields (patient_id, vol[1,D,224,224], tab_x[T], tab_mask[T]) streamed.
    """
    all_ids = list_patient_ids_fast(dataset_root, side=side, max_patients=max_patients)

    if side == "left":
        all_ids = [pid for pid in all_ids if pid not in EXCLUDE_L]
        cols = TAB_INPUTS_L
    elif side == "right":
        all_ids = [pid for pid in all_ids if pid not in EXCLUDE_R]
        cols = TAB_INPUTS_R
    else:
        raise ValueError("side must be 'left' or 'right'")

    df = pd.read_csv(clinical_csv_path)
    df.columns = df.columns.str.strip()
    df["ID"] = df["ID"].astype(str)

    needed_cols = {"ID"} | set(cols)
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    valid_ids = set(df["ID"].tolist())
    all_ids = [pid for pid in all_ids if pid in valid_ids]
    clinical = df.set_index("ID").to_dict("index")

    processed = 0
    for pid in all_ids:
        if max_patients is not None and processed >= max_patients:
            return

        vol = load_single_patient_mri(dataset_root, pid, side=side)
        if vol is None:
            continue

        tab_x, tab_mask = _build_tab_vector_and_mask(clinical[pid], cols)

        processed += 1
        if processed == 1:
            print(
                f"[{side}] Example MRI tensor shape: {tuple(vol.shape)}; "
                f"tab_x shape: {tuple(tab_x.shape)}; tab_mask shape: {tuple(tab_mask.shape)}"
            )

        yield pid, vol, tab_x, tab_mask
