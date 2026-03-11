import os
import sys
import random
from typing import Optional, List, Tuple, Any

import torch
from torch.utils.data import Dataset
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_pain.data import list_patient_ids_fast, load_single_patient_mri


EXCLUDE_L = {"9070923", "9388265", "9594253", "9860568"}
EXCLUDE_R = {"9004315", "9537947", "9637394", "9462278", "9522128"}

TAB_INPUTS = ["V00AGE", "V00ABCIRC", "P02SEX"]

TARGETS_L = ["V00WOMTSL", "V00XRJSM_L", "V99ELKVSAF"]
TARGETS_R = ["V00WOMTSR", "V00XRJSM_R", "V99ERKVSAF"]


def write_test_ids(test_ids: List[str], side: str, out_dir: str = "test_ids") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"masked_ae_{side}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for pid in test_ids:
            f.write(f"{pid}\n")
    return path


def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------
# Normalization
# -------------------------

def normalize_tab(col: str, raw: float) -> float:
    if col == "V00AGE":
        return raw / 100.0
    if col == "V00ABCIRC":
        return raw / 200.0
    if col == "P02SEX":
        # coded as 1/2 → 0/1
        if raw == 1.0:
            return 0.0
        if raw == 2.0:
            return 1.0
        return 0.0 if raw <= 1.0 else 1.0

    if col.startswith("V00WOMTS"):
        return raw / 100.0
    if col.startswith("V00XRJSM"):
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


class KneeMRITabularDataset(Dataset):
    """
    Returns:
        vol      : (1, D, H, W)
        tab_x    : (T,)
        tab_mask : (T,)
        y        : (3,)
        y_mask   : (3,)
    """

    def __init__(
        self,
        root: str,
        clinical_csv_path: str,
        patient_ids: Optional[List[str]] = None,
        side: str = "left",
        max_patients: Optional[int] = None,
        require_all_targets: bool = False,
        require_all_inputs: bool = False,
    ):
        super().__init__()

        self.root = root
        self.side = side
        self.tab_inputs = TAB_INPUTS

        if side == "left":
            self.targets = TARGETS_L
            exclude = EXCLUDE_L
        elif side == "right":
            self.targets = TARGETS_R
            exclude = EXCLUDE_R
        else:
            raise ValueError("side must be 'left' or 'right'")

        df = pd.read_csv(clinical_csv_path)
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(str)

        needed_cols = {"ID"} | set(self.tab_inputs) | set(self.targets)
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        self.clinical = df.set_index("ID").to_dict("index")
        valid_ids = set(df["ID"].tolist())

        if patient_ids is None:
            all_ids = list_patient_ids_fast(root, side, max_patients)
            patient_ids = [str(pid) for pid in all_ids]
        else:
            patient_ids = [str(pid) for pid in patient_ids]
            if max_patients is not None:
                patient_ids = patient_ids[:max_patients]

        patient_ids = [pid for pid in patient_ids if pid not in exclude]
        patient_ids = [pid for pid in patient_ids if pid in valid_ids]

        sub = df.set_index("ID")

        if require_all_targets:
            patient_ids = [
                pid for pid in patient_ids
                if all(not is_missing(sub.loc[pid][t]) for t in self.targets)
            ]

        if require_all_inputs:
            patient_ids = [
                pid for pid in patient_ids
                if all(not is_missing(sub.loc[pid][c]) for c in self.tab_inputs)
            ]

        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_mri(self, pid: str) -> Optional[torch.Tensor]:
        vol = load_single_patient_mri(self.root, pid, self.side)
        if vol is None:
            return None
        if isinstance(vol, tuple):
            vol = vol[1]
        if not torch.is_tensor(vol):
            return None

        vol = vol.float()
        vmin, vmax = vol.min(), vol.max()

        if vmax > 5.0:
            vol = vol - vmin
            if vol.max() > 0:
                vol = vol / vol.max()
            vol = vol * 2.0 - 1.0
        elif 0.0 <= vmin and vmax <= 1.0:
            vol = vol * 2.0 - 1.0

        return vol

    def _build_tab(self, pid: str, cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.clinical[pid]
        vals, mask = [], []

        for c in cols:
            v = row[c]
            if is_missing(v):
                vals.append(0.0)
                mask.append(0.0)
            else:
                vals.append(normalize_tab(c, float(v)))
                mask.append(1.0)

        return (
            torch.tensor(vals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def __getitem__(self, idx: int):
        retries = 0
        while retries < 10:
            pid = self.patient_ids[idx]
            vol = self._load_mri(pid)
            if vol is None:
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

            tab_x, tab_mask = self._build_tab(pid, self.tab_inputs)
            y, y_mask = self._build_tab(pid, self.targets)

            return vol, tab_x, tab_mask, y, y_mask

        raise RuntimeError("Failed to load valid sample after retries")
