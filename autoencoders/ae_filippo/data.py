import io
import os
import tarfile
import tempfile
import shutil
import numpy as np
import torch
import pydicom
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _volume_from_dicoms(dicoms):
    """
    Convert a sorted list of pydicom datasets into a normalised torch tensor.

    Parameters
    ----------
    dicoms : list of pydicom.Dataset
        DICOM slices sorted by InstanceNumber / SliceLocation.

    Returns
    -------
    torch.Tensor
        Shape ``[1, D, 224, 224]``, values in ``[-1, 1]``.
    """
    # Stack into 3D volume (D, H, W)
    volume = np.stack([d.pixel_array for d in dicoms], axis=0).astype(np.float32)

    # Min-max normalize -> [0,1]
    volume -= volume.min()
    if volume.max() > 0:
        volume /= volume.max()

    # Resize each slice to 224×224
    resized_slices = []
    for i in range(volume.shape[0]):
        img = Image.fromarray(volume[i].astype(np.float32), mode="F")
        img = transform_2d.transforms[0](img)  # transforms.Resize((224, 224))

        # img is still float; values already in [0,1], no /255.0 here
        arr = np.array(img).astype(np.float32)  # stays in [0,1]
        resized_slices.append(arr)

    volume = np.stack(resized_slices, axis=0)  # [D, 224, 224]

    # To torch: [D, 224, 224] -> [1, D, 224, 224]
    volume = torch.from_numpy(volume).unsqueeze(0)

    # Normalize to [-1, 1]
    volume = (volume - 0.5) / 0.5
    return volume


def reconstruct_mri_from_tar(tar_path, _gcs_client=None):
    """
    Open a ``.tar.gz`` MRI archive (local path or ``gs://`` URI), extract all
    DICOM slices, reconstruct the 3-D volume, normalise intensities, resize
    slices to 224×224, and return a torch tensor of shape
    ``[1, D, 224, 224]`` in ``[-1, 1]``.

    Parameters
    ----------
    tar_path : str
        Local file path **or** a ``gs://bucket/blob`` URI.  When a GCS URI is
        supplied the archive is streamed into an in-memory ``io.BytesIO``
        buffer and DICOM members are read directly from the archive without
        writing any temporary files to disk.
    _gcs_client : google.cloud.storage.Client, optional
        An existing GCS client to reuse.  When ``None`` (default) a new client
        is created.  Callers that process many patients should pass a shared
        client to avoid the per-call overhead of client initialisation.

    Returns
    -------
    torch.Tensor or None
        Shape ``[1, D, 224, 224]`` in ``[-1, 1]``, or ``None`` on error.
    """
    if isinstance(tar_path, str) and tar_path.startswith("gs://"):
        return _reconstruct_mri_from_gcs(tar_path, client=_gcs_client)

    # --- Local path: original behaviour (extract to temp dir) ---
    temp_dir = tempfile.mkdtemp()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        dicom_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                dicom_files.append(os.path.join(root, f))

        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(
            key=lambda d: getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))
        )

        return _volume_from_dicoms(dicoms)

    except Exception as e:
        print(f"Error reconstructing {tar_path}: {e}")
        return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _reconstruct_mri_from_gcs(gcs_uri, client=None):
    """
    Stream a ``.tar.gz`` MRI archive from GCS into memory and reconstruct the
    3-D volume without writing any temporary files to disk.

    Parameters
    ----------
    gcs_uri : str
        A ``gs://bucket/blob`` URI pointing to a ``.tar.gz`` DICOM archive.
    client : google.cloud.storage.Client, optional
        An existing GCS client to reuse.  A new client is created when
        ``None`` (default).

    Returns
    -------
    torch.Tensor or None
        Shape ``[1, D, 224, 224]`` in ``[-1, 1]``, or ``None`` on error.
    """
    from google.cloud import storage as gcs

    try:
        without_scheme = gcs_uri[len("gs://"):]
        bucket_name, blob_name = without_scheme.split("/", 1)
        if client is None:
            client = gcs.Client()
        blob = client.bucket(bucket_name).blob(blob_name)

        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)

        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            dicoms = []
            for m in members:
                fobj = tar.extractfile(m)
                if fobj is not None:
                    try:
                        dicoms.append(pydicom.dcmread(io.BytesIO(fobj.read())))
                    except Exception:
                        pass

        if not dicoms:
            print(f"No DICOM members found in {gcs_uri}")
            return None

        dicoms.sort(
            key=lambda d: getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0))
        )
        return _volume_from_dicoms(dicoms)

    except Exception as e:
        print(f"Error reconstructing {gcs_uri}: {e}")
        return None


def _iter_mri_dataset_gcs(bucket_name, blob_prefix, side, max_patients):
    """
    Generator that yields ``(patient_id, tensor[1, D, 224, 224])`` for all
    patients found under a GCS prefix, streaming each archive from GCS.

    Parameters
    ----------
    bucket_name : str
        GCS bucket name (without the ``gs://`` scheme).
    blob_prefix : str
        Object prefix inside the bucket, e.g.
        ``"cleaned_images_baseline"``.
    side : str
        ``"left"`` or ``"right"``.
    max_patients : int or None
        Cap on the number of patients to process (``None`` means unlimited).

    Yields
    ------
    tuple[str, torch.Tensor]
        ``(patient_id, volume_tensor)`` where the tensor has shape
        ``[1, D, 224, 224]`` and values in ``[-1, 1]``.
    """
    from google.cloud import storage as gcs

    client = gcs.Client()
    prefix = blob_prefix.rstrip("/") + "/"

    # List subset-level "directories" (one delimiter level)
    subset_prefixes = []
    blobs_iter = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    for page in blobs_iter.pages:
        subset_prefixes.extend(page.prefixes)

    # List patient-level "directories" under each subset
    all_patients = []
    for subset_prefix in subset_prefixes:
        patient_iter = client.list_blobs(bucket_name, prefix=subset_prefix, delimiter="/")
        for page in patient_iter.pages:
            for patient_prefix in page.prefixes:
                patient_id = patient_prefix.rstrip("/").rsplit("/", 1)[-1]
                all_patients.append((patient_prefix, patient_id))

    if max_patients is not None:
        all_patients = all_patients[:max_patients]

    pbar = tqdm(all_patients, desc=f"Loading {side} MRI volumes", unit="patient")
    count = 0

    for patient_prefix, patient_id in pbar:
        mri_side_prefix = patient_prefix + f"mri/{side}/"
        tar_blobs = [
            b
            for b in client.list_blobs(bucket_name, prefix=mri_side_prefix)
            if b.name.endswith(".tar.gz")
        ]
        if not tar_blobs:
            continue

        tar_uri = f"gs://{bucket_name}/{tar_blobs[0].name}"
        tensor = reconstruct_mri_from_tar(tar_uri, _gcs_client=client)

        if tensor is not None:
            count += 1
            if count == 1:
                print(
                    f"This is the shape of the tensor we extract from the MRI image: {tensor.shape}"
                )
            yield patient_id, tensor


def iter_mri_dataset(dataset_root, side="left", max_patients=None):
    """
    Generator that yields ``(patient_id, tensor[1, D, 224, 224])`` for every
    patient in ``dataset_root``, with a progress bar.

    Supports both local directory paths and ``gs://`` URIs.  When a GCS URI
    is supplied each ``.tar.gz`` archive is streamed directly from GCS into
    memory without writing to disk.

    Parameters
    ----------
    dataset_root : str
        Local directory path **or** a ``gs://bucket/prefix`` URI pointing to
        the root of the dataset (the directory that contains the subset
        sub-directories such as ``0.C.2/`` and ``0.E.1/``).
    side : str
        ``"left"`` or ``"right"``.
    max_patients : int or None
        Optional cap on the number of patients to process.

    Yields
    ------
    tuple[str, torch.Tensor]
        ``(patient_id, volume_tensor)`` where the tensor has shape
        ``[1, D, 224, 224]`` and values in ``[-1, 1]``.
    """
    if dataset_root.startswith("gs://"):
        without_scheme = dataset_root[len("gs://"):]
        bucket_name, blob_prefix = without_scheme.split("/", 1)
        yield from _iter_mri_dataset_gcs(bucket_name, blob_prefix, side, max_patients)
        return

    # --- Local path: original behaviour ---
    subset_dirs = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    all_patients = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for patient_id in os.listdir(subset_path):
            all_patients.append((subset, patient_id))

    if max_patients is not None:
        all_patients = all_patients[:max_patients]

    pbar = tqdm(all_patients, desc=f"Loading {side} MRI volumes", unit="patient")
    count = 0

    for subset, patient_id in pbar:
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
            count += 1

            if count == 1:
                print(
                    f"This is the shape of the tensor we extract from the MRI image: {tensor.shape}"
                )

            yield patient_id, tensor


def load_single_patient_mri(dataset_root, patient_id, side="left"):
    """Returns the MRI tensor for a given patient."""
    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

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
