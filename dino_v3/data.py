"""
Data loading for the DINOv3 track.

Thin wrapper around the existing AE data pipeline so that dino_v3 scripts
can import from one canonical place inside this package.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AUTOENCODERS_DIR = os.path.join(PROJECT_ROOT, "autoencoders")
if AUTOENCODERS_DIR not in sys.path:
    sys.path.insert(0, AUTOENCODERS_DIR)

from ae_filippo.data import iter_mri_dataset, reconstruct_mri_from_tar  # noqa: F401

__all__ = ["iter_mri_dataset", "reconstruct_mri_from_tar"]
