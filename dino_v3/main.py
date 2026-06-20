"""
DINOv3 MRI clustering pipeline — GCS-native edition.

Steps (per side)
----------------
1. infer.py        — extract per-patient feature vectors from 3-D MRI via
                     DINOv3 ViT-B/16 backbone (slices 60-99, hook-based patch
                     token extraction, max-pooling to 768-d vector).
2. cluster.py      — K-means (k=5 by default).
3. score_moaks.py  — per-cluster MOAKS mean scores + patient counts.
4. evaluate_clusters.py — NMI/ARI, Spearman/Kendall, Kruskal-Wallis + Dunn.
5. tsne_visualization.py — t-SNE plots.

When ``--side both`` is supplied the pipeline runs sequentially for left
then right, mirroring the pattern in ``inference/main.py``.

All output paths (``--output_dir``) support both local directories and
``gs://`` URIs.  For GCS, artifacts are written to a local staging directory
first and then uploaded to GCS after each side completes.

Usage
-----
::

    uv run dino_v3/main.py \\
        --data_root gs://koa-thesis-oai-data/cleaned_images_baseline \\
        --weights_path gs://koa-thesis-oai-data/weights/dinov3_vitb16_pretrain_lvd1689m.pth \\
        --output_dir gs://koa-thesis-oai-data/results \\
        --side both \\
        --k 5
"""

import argparse
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch

from dino_v3.infer import run_inference
from dino_v3.cluster import load_features, run_clustering
from dino_v3.model import build_volumetric_model
from dino_v3.score_moaks import run_moaks_scoring
from inference.evaluate_clusters import run_evaluation


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _gcs_upload_dir(local_dir: str, gcs_prefix: str) -> None:
    """
    Recursively upload every file under *local_dir* to *gcs_prefix*.

    Parameters
    ----------
    local_dir : str
        Root of the local staging directory.
    gcs_prefix : str
        Destination ``gs://bucket/prefix`` (trailing slash optional).
    """
    from google.cloud import storage as gcs

    without_scheme = gcs_prefix[len("gs://"):]
    bucket_name, blob_prefix = without_scheme.split("/", 1)
    blob_prefix = blob_prefix.rstrip("/")

    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    for dirpath, _, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_name = f"{blob_prefix}/{rel_path}" if blob_prefix else rel_path
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"  Uploaded {local_path} → gs://{bucket_name}/{blob_name}")


def _resolve_gcs_file(uri: str, label: str = "file") -> str:
    """
    Return a local path to a GCS file, downloading it if necessary.

    Parameters
    ----------
    uri : str
        Local path or ``gs://bucket/blob`` URI.
    label : str
        Human-readable name used in log messages.

    Returns
    -------
    str
        Local file path.
    """
    if not uri.startswith("gs://"):
        return uri

    from google.cloud import storage as gcs
    from google.cloud.exceptions import NotFound

    without_scheme = uri[len("gs://"):]
    bucket_name, blob_name = without_scheme.split("/", 1)
    client = gcs.Client()
    blob = client.bucket(bucket_name).blob(blob_name)

    try:
        blob.reload()
    except NotFound:
        raise FileNotFoundError(f"GCS {label} not found: {uri}")

    suffix = os.path.splitext(blob_name)[-1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    blob.download_to_filename(tmp.name)
    print(f"Downloaded {label} from {uri} to {tmp.name}")
    return tmp.name


def _resolve_moaks_csv_dir(moaks_csv_dir: str) -> str:
    """
    Return a local directory containing ``MOAK_L.csv`` and ``MOAK_R.csv``.

    If *moaks_csv_dir* starts with ``gs://`` both files are downloaded to a
    temporary local directory and that directory path is returned.

    Parameters
    ----------
    moaks_csv_dir : str
        Local directory path or ``gs://bucket/prefix`` URI.

    Returns
    -------
    str
        Local directory path.
    """
    if not moaks_csv_dir.startswith("gs://"):
        return moaks_csv_dir

    local_dir = tempfile.mkdtemp(prefix="moaks_csv_")
    for fname in ("MOAK_L.csv", "MOAK_R.csv"):
        uri = moaks_csv_dir.rstrip("/") + "/" + fname
        dest = os.path.join(local_dir, fname)
        tmp_path = _resolve_gcs_file(uri, label=fname)
        os.replace(tmp_path, dest)
    return local_dir


def _resolve_clinical_csv(clinical_csv: str) -> str:
    """
    Return a local path to the clinical CSV.

    If *clinical_csv* starts with ``gs://`` the file is downloaded to a
    temporary location and that path is returned.  The caller is responsible
    for cleanup (the temp file persists for the lifetime of the process).

    Parameters
    ----------
    clinical_csv : str
        Local path or ``gs://bucket/blob`` URI.

    Returns
    -------
    str
        Local file path.
    """
    return _resolve_gcs_file(clinical_csv, label="clinical CSV")


def configure_runtime_dirs(
    cache_dir: str,
    tmp_dir: str,
    pip_cache_dir: str | None = None,
) -> None:
    """
    Force runtime caches and scratch files into caller-controlled directories.

    The GCP VM root disk is intentionally not used for model caches, hub
    downloads, matplotlib caches, or temporary files.  Environment variables
    inherited from a login shell are overridden because they commonly point at
    ``$HOME`` or ``/tmp``.
    """
    torch_cache = os.path.join(cache_dir, "torch")
    hf_cache = os.path.join(cache_dir, "huggingface")
    matplotlib_cache = os.path.join(cache_dir, "matplotlib")
    if pip_cache_dir is None:
        cache_parent = os.path.dirname(os.path.abspath(cache_dir.rstrip(os.sep)))
        pip_cache_dir = os.path.join(cache_parent, "pip-cache")

    for path in (
        cache_dir,
        torch_cache,
        hf_cache,
        matplotlib_cache,
        tmp_dir,
        pip_cache_dir,
    ):
        os.makedirs(path, exist_ok=True)

    os.environ["PIP_CACHE_DIR"] = pip_cache_dir
    os.environ["TORCH_HOME"] = torch_cache
    os.environ["HF_HOME"] = hf_cache
    os.environ["XDG_CACHE_HOME"] = cache_dir
    os.environ["MPLCONFIGDIR"] = matplotlib_cache
    os.environ["TMPDIR"] = tmp_dir
    tempfile.tempdir = tmp_dir


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def run_tsne(
    features_dir: str,
    csv_dir: str,
    plots_dir: str,
    side: str,
    n_components: int = 2,
) -> None:
    """
    Load features and cluster assignments for *side* and save a t-SNE plot.

    Parameters
    ----------
    features_dir : str
        Directory containing ``features_{side}.npz``.
    csv_dir : str
        Directory containing ``mri_clusters_{side}.csv``.
    plots_dir : str
        Destination directory for the output PNG.
    side : str
        ``"left"`` or ``"right"``.
    n_components : int
        Number of t-SNE dimensions (1, 2, or 3).
    """
    from inference.tsne_visualization import tsne_plot_with_clusters

    ids, features = load_features(features_dir, side)

    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster CSV not found: {clusters_path}")
    df_clusters = pd.read_csv(clusters_path)

    patients_features = {
        str(pid): torch.from_numpy(features[i]).unsqueeze(0)
        for i, pid in enumerate(ids)
    }

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"tsne_{side}_{n_components}d.png")

    print(f"[{side}] Running t-SNE (n_components={n_components}) ...")
    tsne_plot_with_clusters(
        patients_features=patients_features,
        df_clusters=df_clusters,
        n_components=n_components,
        output_path=out_path,
    )
    print(f"[{side}] Saved t-SNE plot to {out_path}")


def run_side(
    side: str,
    data_root: str,
    weights_path: str,
    local_output_root: str,
    moaks_csv_dir: str,
    clinical_csv_local: str,
    k: int,
    max_patients,
    tsne_components: int,
    model=None,
    device=None,
    batch_size: int = 1,
    num_workers: int = 4,
    overwrite: bool = False,
    use_amp: bool = True,
) -> None:
    """
    Execute the full DINOv3 pipeline for a single knee side.

    Parameters
    ----------
    side : str
        ``"left"`` or ``"right"``.
    data_root : str
        Local or ``gs://`` path to the MRI dataset root.
    weights_path : str
        Local or ``gs://`` path to pretrained weights.
    local_output_root : str
        Local directory where all output files are written (may be a temp
        staging dir when the final destination is GCS).
    moaks_csv_dir : str
        Local directory containing ``MOAK_L.csv`` / ``MOAK_R.csv``.
    clinical_csv_local : str
        Local path to the cleaned clinical CSV.
    k : int
        Number of K-means clusters.
    max_patients : int or None
        Patient cap (``None`` for unlimited).
    tsne_components : int
        Number of t-SNE dimensions.
    """
    results_dir = os.path.join(local_output_root, side)
    features_dir = os.path.join(results_dir, "features")
    csv_dir = os.path.join(results_dir, "csv")
    plots_dir = os.path.join(results_dir, "plots")

    moaks_suffix = "L" if side == "left" else "R"
    moaks_csv = os.path.join(moaks_csv_dir, f"MOAK_{moaks_suffix}.csv")

    print(f"\n=== DINOv3 pipeline | side={side} | k={k} ===\n")

    # 1) Feature extraction (resume-safe; per-patient cache + manifest)
    run_inference(
        data_root=data_root,
        side=side,
        features_dir=features_dir,
        run_output_dir=local_output_root,
        weights_path=weights_path,
        model=model,
        device=device,
        max_patients=max_patients,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
        use_amp=use_amp,
    )

    # 2) Clustering
    df_clusters = run_clustering(
        features_dir=features_dir,
        csv_dir=csv_dir,
        side=side,
        k=k,
    )

    # 3) MOAKS scoring
    run_moaks_scoring(
        df_clusters=df_clusters,
        moaks_csv=moaks_csv,
        side=side,
        csv_dir=csv_dir,
    )

    # 4) Cluster evaluation (NMI/ARI, Spearman/Kendall, KW + Dunn)
    clusters_csv_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    eval_report_path = os.path.join(results_dir, "evaluation_report.txt")
    run_evaluation(
        clusters_csv=clusters_csv_path,
        clinical_csv=clinical_csv_local,
        side=side,
        output_path=eval_report_path,
    )

    # 5) t-SNE visualisation
    run_tsne(
        features_dir=features_dir,
        csv_dir=csv_dir,
        plots_dir=plots_dir,
        side=side,
        n_components=tsne_components,
    )

    print(f"\n=== Done [{side}]. Results in {results_dir}/ ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the DINOv3 pipeline for one or both sides."""
    parser = argparse.ArgumentParser(
        description="DINOv3 MRI clustering pipeline (GCS-native)"
    )
    parser.add_argument(
        "--input_dir",
        "--data_root",
        dest="data_root",
        type=str,
        default=os.environ.get(
            "KOA_INPUT_DIR", "/mnt/data/dataset/cleaned_images_baseline"
        ),
        help=(
            "Local root directory containing the patient MRI tar.gz tree "
            "(<subset>/<patient>/mri/<side>/*.tar.gz). Must be a local path — "
            "copy the data to disk first (resume-safe extraction reads it many "
            "times). Env: KOA_INPUT_DIR. "
            "Default: /mnt/data/dataset/cleaned_images_baseline."
        ),
    )
    parser.add_argument(
        "--weights_path",
        "--checkpoint_path",
        dest="weights_path",
        type=str,
        default=os.environ.get(
            "KOA_WEIGHTS_PATH",
            "/mnt/data/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth",
        ),
        help=(
            "Path (local or gs://) to DINOv3 pretrained weights (.pth). "
            "Env: KOA_WEIGHTS_PATH."
        ),
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help=(
            'Which knee side(s) to process.  "both" runs left then right '
            "sequentially (default: both)."
        ),
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="csv/clinical00_cleaned.csv",
        help=(
            "Path (local or gs://) to the cleaned clinical CSV "
            "(default: csv/clinical00_cleaned.csv)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("KOA_OUTPUT_DIR", "/mnt/data/outputs"),
        help=(
            "Root output directory (local or gs://) for all result files. "
            "Env: KOA_OUTPUT_DIR. Default: /mnt/data/outputs."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device. 'cuda' falls back to CPU with a warning if "
        "no GPU is available (default: cuda).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Patients pulled per DataLoader step (default: 1). Volumes have "
        "variable depth, so each is run through the backbone individually; "
        "raise this only to deepen worker prefetch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for parallel tar.gz/DICOM decode (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite per-patient feature vectors even if "
        "they are already cached (default: skip cached).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA float16 autocast (faster, less memory). OFF by "
        "default: extraction runs in fp32 to match the prior baseline. "
        "Max-pooling over fp16 features can shift K-means cluster boundaries, "
        "so validate fp16-vs-fp32 before trusting AMP results.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("KOA_CACHE_DIR", "/mnt/data/cache"),
        help="Cache root for model/hub downloads (TORCH_HOME/HF_HOME). "
        "Env: KOA_CACHE_DIR. Default: /mnt/data/cache.",
    )
    parser.add_argument(
        "--pip_cache_dir",
        type=str,
        default=os.environ.get("KOA_PIP_CACHE_DIR", "/mnt/data/pip-cache"),
        help="pip cache directory. Env: KOA_PIP_CACHE_DIR. "
        "Default: /mnt/data/pip-cache.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=os.environ.get("KOA_TMP_DIR", "/mnt/data/tmp"),
        help="Scratch directory for temp files (TMPDIR). "
        "Env: KOA_TMP_DIR. Default: /mnt/data/tmp.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of K-means clusters (default: 5).",
    )
    parser.add_argument(
        "--tsne_components",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Number of t-SNE dimensions for visualisation (default: 2).",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional patient cap for debugging (default: None = unlimited).",
    )
    parser.add_argument(
        "--moaks_csv_dir",
        type=str,
        default="csv",
        help="Directory containing MOAK_L.csv and MOAK_R.csv (default: csv).",
    )
    args = parser.parse_args()

    if not args.data_root:
        raise ValueError("--input_dir / --data_root must not be empty.")

    # --- Keep all caches/temp off the tiny root disk ---------------------
    configure_runtime_dirs(args.cache_dir, args.tmp_dir, args.pip_cache_dir)

    # --- Device resolution with explicit CPU fallback --------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: --device cuda requested but torch.cuda.is_available() is "
            "False. Falling back to CPU (this will be very slow)."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        print(
            f"Using GPU: {torch.cuda.get_device_name(0)} "
            f"(CUDA {torch.version.cuda}, torch {torch.__version__})"
        )
    else:
        print(f"Using CPU (torch {torch.__version__})")

    use_amp = (device.type == "cuda") and args.amp
    if use_amp:
        print("AMP enabled (float16 autocast). Validate clustering vs fp32.")

    sides = ["left", "right"] if args.side == "both" else [args.side]
    use_gcs_output = args.output_dir.startswith("gs://")

    # Build the (frozen) model once and reuse it for both sides.
    model = build_volumetric_model(
        weights_path=args.weights_path,
        device=device,
        tmp_dir=args.tmp_dir,
    )

    # Resolve clinical CSV and MOAKS CSV dir to local paths (download from GCS if needed)
    clinical_csv_local = _resolve_clinical_csv(args.clinical_csv)
    moaks_csv_dir_local = _resolve_moaks_csv_dir(args.moaks_csv_dir)

    if use_gcs_output:
        staging_dir = tempfile.mkdtemp(prefix="dino_v3_staging_")
        print(f"GCS output mode — staging locally at {staging_dir}")
    else:
        staging_dir = args.output_dir
        os.makedirs(staging_dir, exist_ok=True)

    try:
        for side in sides:
            run_side(
                side=side,
                data_root=args.data_root,
                weights_path=args.weights_path,
                local_output_root=staging_dir,
                moaks_csv_dir=moaks_csv_dir_local,
                clinical_csv_local=clinical_csv_local,
                k=args.k,
                max_patients=args.max_patients,
                tsne_components=args.tsne_components,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
                use_amp=use_amp,
            )

        if use_gcs_output:
            print(f"\nUploading results to {args.output_dir} ...")
            _gcs_upload_dir(staging_dir, args.output_dir)
            print("Upload complete.")

    finally:
        if use_gcs_output:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\n=== All done. Results in {args.output_dir}/ ===")


if __name__ == "__main__":
    main()
