import os
import sys
import random
from typing import Optional, List, Any

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


def write_test_ids(
    test_ids: List[str],
    side: str,
    out_dir: str = "test_ids",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tab_input_ae_{side}.txt")

    with open(path, "w", encoding="utf-8") as f:
        for pid in test_ids:
            f.write(f"{pid}\n")

    return path


def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_tab(col: str, raw: float) -> float:
    # -----------------
    # Inputs
    # -----------------
    if col == "V00AGE":
        return raw / 100.0

    if col == "V00ABCIRC":
        return raw / 200.0

    if col == "P02SEX":
        # coded as 1/2 -> map to 0/1
        if raw == 1.0:
            return 0.0
        if raw == 2.0:
            return 1.0
        return 0.0 if raw <= 1.0 else 1.0

    # -----------------
    # Targets
    # -----------------
    if col in ("V00WOMTSL", "V00WOMTSR"):
        return raw / 100.0

    if col in ("V00XRJSM_L", "V00XRJSM_R"):
        return raw / 3.0

    if col in ("V99ELKVSAF", "V99ERKVSAF"):
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
        vol   : FloatTensor, shape (1, D, H, W), normalized to [-1, 1]
        tab_x : FloatTensor, shape (T,)
        y     : FloatTensor, shape (3,)
    """

    def __init__(
        self,
        root: str,
        clinical_csv_path: str,
        patient_ids: Optional[List[str]] = None,
        side: str = "left",
        max_patients: Optional[int] = None,
        require_all_targets: bool = True,
        require_all_inputs: bool = True,
    ):
        super().__init__()

        self.root = root
        self.side = side

        if side == "left":
            self.tab_vars = TAB_INPUTS
            self.targets = TARGETS_L
            exclude = EXCLUDE_L
        elif side == "right":
            self.tab_vars = TAB_INPUTS
            self.targets = TARGETS_R
            exclude = EXCLUDE_R
        else:
            raise ValueError("side must be 'left' or 'right'")

        if patient_ids is None:
            all_ids = list_patient_ids_fast(root, side, max_patients)
            patient_ids = [str(pid) for pid in all_ids]
        else:
            patient_ids = [str(pid) for pid in patient_ids]
            if max_patients is not None:
                patient_ids = patient_ids[:max_patients]

        patient_ids = [pid for pid in patient_ids if pid not in exclude]

        df = pd.read_csv(clinical_csv_path)
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(str)

        needed_cols = {"ID"} | set(self.tab_vars) | set(self.targets)
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        self.clinical = df.set_index("ID").to_dict("index")

        valid_ids = set(df["ID"].tolist())
        patient_ids = [pid for pid in patient_ids if pid in valid_ids]

        if require_all_targets or require_all_inputs:
            sub = df.set_index("ID")
            keep: List[str] = []
            for pid in patient_ids:
                row = sub.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                ok_targets = True
                ok_inputs = True

                if require_all_targets:
                    ok_targets = all(not is_missing(row[t]) for t in self.targets)

                if require_all_inputs:
                    ok_inputs = all(not is_missing(row[c]) for c in self.tab_vars)

                if ok_targets and ok_inputs:
                    keep.append(pid)

            patient_ids = keep

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
        elif vmin >= 0.0 and vmax <= 1.0:
            vol = vol * 2.0 - 1.0
        return vol

    def _build_tab(self, pid: str, cols: List[str]) -> torch.Tensor:
        row = self.clinical[pid]
        vals: List[float] = []

        for c in cols:
            v = row[c]
            if is_missing(v):
                vals.append(0.0)
            else:
                vals.append(normalize_tab(c, float(v)))

        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int):
        retries = 0
        while retries < 10:
            pid = self.patient_ids[idx]
            vol = self._load_mri(pid)
            if vol is None:
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

            tab_x = self._build_tab(pid, self.tab_vars)
            y = self._build_tab(pid, self.targets)

            return vol, tab_x, y

        raise RuntimeError("Failed to load valid sample after retries")
