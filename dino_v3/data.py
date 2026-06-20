"""
Data loading for the DINOv3 track.

Re-exports the canonical AE data helpers (``iter_mri_dataset``,
``reconstruct_mri_from_tar``) and adds a PyTorch ``Dataset`` over the local
``<subset>/<patient>/mri/<side>/*.tar.gz`` tree so feature extraction can use
a ``DataLoader`` with multiple workers (parallel tar.gz read + DICOM decode)
while the GPU runs the backbone.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AUTOENCODERS_DIR = os.path.join(PROJECT_ROOT, "autoencoders")
if AUTOENCODERS_DIR not in sys.path:
    sys.path.insert(0, AUTOENCODERS_DIR)

from torch.utils.data import Dataset

from ae_filippo.data import iter_mri_dataset, reconstruct_mri_from_tar  # noqa: F401

SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
)

__all__ = [
    "iter_mri_dataset",
    "reconstruct_mri_from_tar",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "list_image_files",
    "list_patient_archives",
    "KneeMRITarDataset",
    "collate_keep_list",
]


def list_image_files(input_dir: str) -> list[str]:
    """
    Recursively enumerate supported local image files under *input_dir*.

    This is intentionally a streaming-friendly path discovery helper: it only
    returns file paths and never opens image payloads, so a large dataset is
    not loaded into RAM during discovery.
    """
    if input_dir.startswith("gs://"):
        raise ValueError(
            "list_image_files expects a local path. Copy the dataset to "
            "local disk (e.g. /mnt/data/dataset/...) before running."
        )
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files: list[str] = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_files.append(os.path.join(dirpath, filename))
    return sorted(image_files)


def list_patient_archives(dataset_root: str, side: str) -> list[tuple[str, str]]:
    """
    Enumerate ``(patient_id, tar_path)`` pairs for one knee *side* under a
    **local** dataset root laid out as
    ``<root>/<subset>/<patient_id>/mri/<side>/*.tar.gz``.

    Patients without a ``.tar.gz`` for the requested side are skipped.  The
    result is sorted by ``patient_id`` so runs are deterministic and resume
    behaviour is stable.

    Parameters
    ----------
    dataset_root : str
        Local directory path (``gs://`` is not supported here — download /
        copy the data locally first, e.g. to ``/mnt/data/dataset/...``).
    side : str
        ``"left"`` or ``"right"``.

    Returns
    -------
    list[tuple[str, str]]
        Sorted list of ``(patient_id, absolute_tar_path)``.
    """
    if dataset_root.startswith("gs://"):
        raise ValueError(
            "list_patient_archives expects a local path. Copy the dataset to "
            "local disk (e.g. /mnt/data/dataset/...) before running."
        )
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Input directory not found: {dataset_root}")

    archives: list[tuple[str, str]] = []
    for subset in sorted(os.listdir(dataset_root)):
        subset_path = os.path.join(dataset_root, subset)
        if not os.path.isdir(subset_path):
            continue
        for patient_id in sorted(os.listdir(subset_path)):
            mri_side_dir = os.path.join(subset_path, patient_id, "mri", side)
            if not os.path.isdir(mri_side_dir):
                continue
            tar_files = sorted(
                f for f in os.listdir(mri_side_dir) if f.endswith(".tar.gz")
            )
            if not tar_files:
                continue
            archives.append(
                (patient_id, os.path.abspath(os.path.join(mri_side_dir, tar_files[0])))
            )
    return sorted(archives, key=lambda item: (item[0], item[1]))


class KneeMRITarDataset(Dataset):
    """
    Map-style dataset yielding ``(patient_id, volume_or_None, tar_path)`` for a
    fixed list of patient archives.

    ``__getitem__`` decodes one ``.tar.gz`` into a ``[1, D, 224, 224]`` tensor
    via :func:`reconstruct_mri_from_tar`.  Decode failures return ``None`` as
    the volume (instead of raising) so a ``DataLoader`` worker never kills the
    whole job — the caller records the failure and moves on.

    Volumes have a variable depth ``D`` across patients, so batching uses
    :func:`collate_keep_list` (a list of items, no stacking).
    """

    def __init__(self, archives: list[tuple[str, str]]):
        self.archives = archives

    def __len__(self) -> int:
        return len(self.archives)

    def __getitem__(self, idx: int):
        patient_id, tar_path = self.archives[idx]
        try:
            volume = reconstruct_mri_from_tar(tar_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[load error] {patient_id} ({tar_path}): {exc}")
            volume = None
        return patient_id, volume, tar_path


def collate_keep_list(batch):
    """Collate that returns the batch list unchanged (volumes vary in depth)."""
    return list(batch)
