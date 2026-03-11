import os
import sys
import random
from typing import Optional, List

import torch
from torch.utils.data import Dataset
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_pain.data import list_patient_ids_fast, load_single_patient_mri


EXCLUDE_L = {
    "9070923",
    "9388265",
    "9594253",
    "9860568",
}
EXCLUDE_R = {
    "9004315",
    "9537947",
    "9637394",
    "9462278",
    "9522128",
}


class KneeMRIPainDataset(Dataset):
    """
    Dataset for multi-task AE with KOOS/WOMAC/etc. supervision.

    Returns:
        vol: FloatTensor, shape (1, D, 224, 224), normalized to [-1, 1]
        y:   FloatTensor, shape (1,), e.g. KOOS Pain score in [0, 100]
    """

    def __init__(
        self,
        root: str,
        clinical_csv_path: str,
        tabular_variable: str,
        side: str = "left",
        split: str = "train",
        max_patients: Optional[int] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()

        self.root = root
        self.side = side
        self.tabular_variable = tabular_variable

        # --------- collect MRI patient IDs ----------
        all_ids: List[str] = list_patient_ids_fast(root, side, max_patients)
        all_ids = [str(pid) for pid in all_ids]

        if side == "left":
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_L]
        elif side == "right":
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_R]
        else:
            raise ValueError("side must be 'left' or 'right'")

        # --------- load clinical CSV ----------
        df = pd.read_csv(clinical_csv_path)
        df.columns = df.columns.str.strip()

        if "ID" not in df.columns:
            raise ValueError(
                f"CSV has no 'ID' column after stripping. Columns are: {list(df.columns)}"
            )

        if self.tabular_variable not in df.columns:
            raise ValueError(
                f"tabular_variable '{self.tabular_variable}' not found in CSV columns. "
                f"Available columns: {list(df.columns)}"
            )

        df["ID"] = df["ID"].astype(str)

        # keep only patients that exist in both MRI and clinical CSV
        valid_ids = set(df["ID"].tolist())
        all_ids = [pid for pid in all_ids if pid in valid_ids]

        if len(all_ids) == 0:
            raise RuntimeError("No overlapping patients between MRI and clinical CSV.")

        self.clinical_dict = df.set_index("ID").to_dict("index")

        # --------- split into train / val / test ----------
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        n_total = len(all_ids)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        if split == "train":
            self.patient_ids = all_ids[:n_train]
        elif split == "val":
            self.patient_ids = all_ids[n_train : n_train + n_val]
        elif split == "test":
            self.patient_ids = all_ids[n_train + n_val :]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        max_retries = 10
        retries = 0
        
        while retries < max_retries:
            pid = self.patient_ids[idx]

            vol = load_single_patient_mri(self.root, pid, self.side)

            if vol is None:
                print(f"[Warning] Loading failed for patient {pid}. Skipping to a new random patient...")
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

            if isinstance(vol, tuple):
                loaded_pid, vol_tensor = vol
                vol = vol_tensor

            if not torch.is_tensor(vol):
                print(f"[Warning] Invalid data type for {pid}. Skipping...")
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

            vol = vol.float()

            v_min = vol.min()
            v_max = vol.max()
            
            if v_max > 5.0: 
                vol = vol - v_min
                if vol.max() > 0:
                    vol = vol / vol.max()
                vol = (vol * 2.0) - 1.0
            elif v_min >= 0.0 and v_max <= 1.0:
                 vol = (vol * 2.0) - 1.0

            # Load Pain Label
            try:
                raw_value = float(self.clinical_dict[pid][self.tabular_variable])
                value = raw_value / 100.0 
                y = torch.tensor([value], dtype=torch.float32)
                
                return vol, y 
                
            except KeyError:
                print(f"[Warning] Clinical data missing for {pid}. Skipping...")
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

        raise RuntimeError(f"Failed to load any valid data after {max_retries} retries. Check dataset path.")
