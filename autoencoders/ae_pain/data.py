import os
import tarfile
import tempfile
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
from PIL import Image
from torchvision import transforms

transform_resize = transforms.Resize((224, 224))

def reconstruct_mri_from_tar(tar_path):
    """
    Robust MRI loader.
    1. Extracts TAR.
    2. Tries to read EVERY file as a DICOM (ignoring extensions).
    3. Normalizes to [-1, 1].
    """
    temp_dir = tempfile.mkdtemp()
    try:
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
        except Exception as e:
            print(f"[Error] Corrupt tar file: {tar_path} -> {e}")
            return None

        candidate_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.startswith("._") or f.startswith("."): 
                    continue
                candidate_files.append(os.path.join(root, f))

        if not candidate_files:
            print(f"[Error] Empty archive: {tar_path}")
            return None

        dicoms = []
        for f in candidate_files:
            try:
                d = pydicom.dcmread(f, force=True)
                
                if hasattr(d, "pixel_array"):
                    dicoms.append(d)
            except:
                continue

        if not dicoms:
            print(f"[Error] No valid DICOMs found inside {tar_path}")
            return None

        dicoms.sort(key=lambda d: float(getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))))

        try:
            volume = np.stack([d.pixel_array for d in dicoms], axis=0).astype(np.float32)
        except ValueError as e:
            print(f"[Error] DICOM shape mismatch in {tar_path}: {e}")
            return None

        v_min = volume.min()
        v_max = volume.max()
        if v_max - v_min > 0:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = np.zeros_like(volume)

        resized_slices = []
        for i in range(volume.shape[0]):
            img = Image.fromarray(volume[i], mode="F") 
            img = transform_resize(img)
            resized_slices.append(np.array(img))

        volume = np.stack(resized_slices, axis=0)
        volume = torch.from_numpy(volume).unsqueeze(0) # (1, D, H, W)

        # Normalize to [-1, 1]
        volume = (volume - 0.5) / 0.5
        
        return volume.float()

    except Exception as e:
        print(f"[Error] Unexpected error loading {tar_path}: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def load_single_patient_mri(dataset_root, patient_id, side="left"):
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

def list_patient_ids_fast(dataset_root, side="left", max_patients=None):
    subset_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    ids = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for pid in os.listdir(subset_path):
            if os.path.isdir(os.path.join(subset_path, pid, "mri", side)):
                ids.append(pid)
    if max_patients: ids = ids[:max_patients]
    return ids

def iter_mri_dataset(dataset_root, side="left", max_patients=None):
    pass