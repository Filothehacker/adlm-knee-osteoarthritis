import torch
from torch.utils.data import Dataset
import random
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.data import iter_mri_dataset, load_single_patient_mri

def list_patient_ids_fast(dataset_root, side="left", max_patients=None):
    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    ids = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for pid in os.listdir(subset_path):
            mri_side_dir = os.path.join(subset_path, pid, "mri", side)
            if os.path.isdir(mri_side_dir):
                ids.append(pid)

    if max_patients is not None:
        ids = ids[:max_patients]

    return ids

class KneeMRIDataset(Dataset):
    def __init__(self, root, side='left',split='train', max_patients=None):
        self.root = root
        self.side = side
        self.max_patients = max_patients

        all_ids = []
        EXCLUDE_L = {
        "9070923",
        "9388265",
        "9594253",
        "9860568",}
        EXCLUDE_R = {
        "9004315",
        "9537947",
        "9637394",
        "9462278",
        "9522128",}
        
        
        all_ids = list_patient_ids_fast(root, side, max_patients)

        if side == "left":
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_L]
        else:
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_R]


        random.shuffle(all_ids)
        n_total = len(all_ids)
        n_train = int(0.7 * n_total)
        n_val   = int(0.15 * n_total)

        if split == "train":
            self.patient_ids = all_ids[:n_train]
        elif split == "val":
            self.patient_ids = all_ids[n_train:n_train+n_val]
        elif split == "test":
            self.patient_ids = all_ids[n_train+n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        target_pid = self.patient_ids[idx]

        vol = load_single_patient_mri(self.root, target_pid, self.side)
        if not torch.is_tensor(vol):
            vol = torch.tensor(vol, dtype=torch.float32)

        return vol  # (1, D, 224, 224)
