"""
Feature extraction for the DINOv3 track.

For each patient, the 3-D MRI volume (decoded from a per-patient ``.tar.gz`` of
DICOM slices) is passed through :class:`~dino_v3.model.VolumetricDINOv3` and
max-pooled to a single 768-d vector.

Production behaviour on the VM
------------------------------
* **Resume-safe.** Each patient vector is cached atomically at
  ``{features_dir}/per_patient/{patient_id}.npy``.  Re-running skips patients
  whose vector already exists (unless ``overwrite=True``).
* **Crash-safe per patient.** A failure on one patient is logged to
  ``{run_output_dir}/failed_files.csv`` and the run continues.
* **Parallel decode.** A ``DataLoader`` with ``num_workers`` decodes tar.gz /
  DICOM on CPU workers while the GPU runs the backbone.
* **GPU + mixed precision.** Runs under ``torch.inference_mode()`` and, on
  CUDA, ``torch.autocast(float16)`` (disable with ``use_amp=False``).

After all patients are processed the per-patient cache is assembled into the
``{features_dir}/features_{side}.npz`` file that the clustering stage expects
(keys ``ids`` and ``features``), so the downstream pipeline is unchanged.
"""

import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from dino_v3.data import (
    KneeMRITarDataset,
    collate_keep_list,
    list_image_files,
    list_patient_archives,
)
from dino_v3.model import build_volumetric_model


def _atomic_save_npy(array: np.ndarray, final_path: str) -> None:
    """Write *array* to ``final_path`` atomically (tmp file + ``os.replace``)."""
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    np.save(tmp_path, array)
    # np.save appends .npy if missing; normalise the tmp name it actually wrote.
    written = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npy"
    os.replace(written, final_path)


def _atomic_savez(final_path: str, **arrays) -> None:
    """Write a ``.npz`` to *final_path* atomically."""
    tmp_path = f"{final_path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, final_path)


def _record_failure(failed_csv: str, patient_id: str, tar_path: str, error: str) -> None:
    """Append one row to ``failed_files.csv`` (creating it with a header)."""
    new_file = not os.path.exists(failed_csv)
    os.makedirs(os.path.dirname(failed_csv) or ".", exist_ok=True)
    with open(failed_csv, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["patient_id", "tar_path", "error"])
        writer.writerow([patient_id, tar_path, error])


def _append_manifest(manifest_csv: str, patient_id: str, npy_path: str) -> None:
    """Append one processed-patient row to the side manifest."""
    new_file = not os.path.exists(manifest_csv)
    with open(manifest_csv, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["patient_id", "timestamp", "npy_path"])
        writer.writerow([patient_id, time.strftime("%Y-%m-%dT%H:%M:%S"), npy_path])


def _write_progress(progress_json: str, payload: dict) -> None:
    """Atomically write a machine-readable progress snapshot."""
    os.makedirs(os.path.dirname(progress_json) or ".", exist_ok=True)
    tmp_path = f"{progress_json}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, progress_json)


def run_inference(
    data_root: str,
    side: str,
    features_dir: str,
    run_output_dir: str,
    weights_path: str | None = None,
    model=None,
    device: torch.device | None = None,
    max_patients: int | None = None,
    batch_size: int = 1,
    num_workers: int = 4,
    overwrite: bool = False,
    use_amp: bool = True,
) -> str:
    """
    Run DINOv3 feature extraction for one knee *side*, resume-safe.

    Either a prebuilt *model* (preferred, so weights load once for both sides)
    or a *weights_path* must be supplied.

    Saves per-patient vectors under ``{features_dir}/per_patient/`` and the
    aggregated ``{features_dir}/features_{side}.npz``.  Returns the npz path.
    """
    if model is None:
        if weights_path is None:
            raise ValueError("run_inference requires either model or weights_path")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_volumetric_model(weights_path=weights_path, device=device)
    if device is None:
        device = next(model.parameters()).device

    amp_enabled = use_amp and device.type == "cuda"

    per_patient_dir = os.path.join(features_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)
    os.makedirs(run_output_dir, exist_ok=True)

    failed_csv = os.path.join(run_output_dir, "failed_files.csv")
    manifest_csv = os.path.join(run_output_dir, f"manifest_{side}.csv")
    progress_json = os.path.join(run_output_dir, f"progress_{side}.json")

    # --- enumerate patients, apply resume filter before building the loader ---
    archives = list_patient_archives(data_root, side)
    if not archives:
        image_files = list_image_files(data_root)
        if image_files:
            print(
                f"[{side}] Found {len(image_files)} supported image files under "
                f"{data_root}, but this DINOv3 pipeline expects the MRI archive "
                "layout <subset>/<patient>/mri/<side>/*.tar.gz."
            )
    if max_patients is not None:
        archives = archives[:max_patients]
    total_found = len(archives)

    if overwrite:
        todo = archives
    else:
        todo = [
            (pid, tp)
            for (pid, tp) in archives
            if not os.path.exists(os.path.join(per_patient_dir, f"{pid}.npy"))
        ]
    already = total_found - len(todo)

    print(
        f"[{side}] {total_found} patient archives found under {data_root} | "
        f"{already} already cached | {len(todo)} to process | "
        f"amp={amp_enabled} device={device}"
    )
    _write_progress(
        progress_json,
        {
            "side": side,
            "input_dir": data_root,
            "total_found": total_found,
            "already_cached": already,
            "to_process": len(todo),
            "processed_new": 0,
            "failed": 0,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "images_per_second": 0.0,
            "device": str(device),
            "amp": amp_enabled,
        },
    )

    if todo:
        dataset = KneeMRITarDataset(todo)
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_keep_list,
            "pin_memory": (device.type == "cuda"),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(dataset, **loader_kwargs)

        processed = 0
        failed = 0
        start = time.time()
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                for patient_id, volume, tar_path in batch:
                    npy_path = os.path.join(per_patient_dir, f"{patient_id}.npy")
                    if not overwrite and os.path.exists(npy_path):
                        print(f"[{side}] SKIP {patient_id}: cached during run")
                        continue
                    if volume is None:
                        _record_failure(
                            failed_csv, patient_id, tar_path, "decode_failed_or_none"
                        )
                        failed += 1
                        print(f"[{side}] FAILED {patient_id}: decode returned None")
                        _write_progress(
                            progress_json,
                            {
                                "side": side,
                                "input_dir": data_root,
                                "total_found": total_found,
                                "already_cached": already,
                                "to_process": len(todo),
                                "processed_new": processed,
                                "failed": failed,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "images_per_second": processed
                                / max(time.time() - start, 1e-6),
                                "last_failure": patient_id,
                                "device": str(device),
                                "amp": amp_enabled,
                            },
                        )
                        continue
                    try:
                        vol = volume.unsqueeze(0).to(device, non_blocking=True)
                        if amp_enabled:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                feats = model(vol)
                        else:
                            feats = model(vol)
                        vec = feats.squeeze(0).float().cpu().numpy()

                        _atomic_save_npy(vec, npy_path)
                        _append_manifest(manifest_csv, patient_id, npy_path)

                        processed += 1
                        rate = processed / max(time.time() - start, 1e-6)
                        print(
                            f"[{side}] {processed}/{len(todo)} | {patient_id} | "
                            f"{rate:.2f} patients/s"
                        )
                        _write_progress(
                            progress_json,
                            {
                                "side": side,
                                "input_dir": data_root,
                                "total_found": total_found,
                                "already_cached": already,
                                "to_process": len(todo),
                                "processed_new": processed,
                                "failed": failed,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "images_per_second": rate,
                                "last_success": patient_id,
                                "device": str(device),
                                "amp": amp_enabled,
                            },
                        )
                    except Exception as exc:
                        _record_failure(failed_csv, patient_id, tar_path, repr(exc))
                        failed += 1
                        print(f"[{side}] FAILED {patient_id}: {exc!r}")
                        _write_progress(
                            progress_json,
                            {
                                "side": side,
                                "input_dir": data_root,
                                "total_found": total_found,
                                "already_cached": already,
                                "to_process": len(todo),
                                "processed_new": processed,
                                "failed": failed,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "images_per_second": processed
                                / max(time.time() - start, 1e-6),
                                "last_failure": patient_id,
                                "device": str(device),
                                "amp": amp_enabled,
                            },
                        )
                        continue

        elapsed = time.time() - start
        print(
            f"[{side}] Feature extraction done: {processed} new in {elapsed:.1f}s "
            f"({processed / max(elapsed, 1e-6):.2f} patients/s)"
        )
        _write_progress(
            progress_json,
            {
                "side": side,
                "input_dir": data_root,
                "total_found": total_found,
                "already_cached": already,
                "to_process": len(todo),
                "processed_new": processed,
                "failed": failed,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": elapsed,
                "images_per_second": processed / max(elapsed, 1e-6),
                "device": str(device),
                "amp": amp_enabled,
            },
        )

    # --- assemble aggregated features_{side}.npz from the per-patient cache ---
    return _assemble_npz(per_patient_dir, features_dir, side)


def _assemble_npz(per_patient_dir: str, features_dir: str, side: str) -> str:
    """Stack all cached per-patient vectors into ``features_{side}.npz``."""
    # Exclude any half-written temp files left by a crash mid-_atomic_save_npy
    # (np.save always appends .npy, so a stray tmp also ends in .npy).
    npy_files = sorted(
        f
        for f in os.listdir(per_patient_dir)
        if f.endswith(".npy") and ".tmp." not in f
    )
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    if not npy_files:
        print(f"[{side}] No cached vectors found, nothing to assemble.")
        return feat_out

    ids = [os.path.splitext(f)[0] for f in npy_files]
    features = np.stack(
        [np.load(os.path.join(per_patient_dir, f)) for f in npy_files]
    )
    _atomic_savez(feat_out, ids=np.array(ids), features=features)
    print(f"[{side}] Assembled features {features.shape} ({len(ids)} patients) -> {feat_out}")
    return feat_out
