# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

```bash
uv sync                  # Install all dependencies
uv run <script.py>       # Run a script using the project venv
uv add <package>         # Add a dependency
uv remove <package>      # Remove a dependency
```

**Formatting:** `black` is the configured formatter.

**Cluster training:** Each training directory under `autoencoders/` contains a `.slurm` file. All SLURM scripts must be submitted from the **project root** (the `python` commands reference paths like `autoencoders/ae_filippo/train.py`).

## Project Overview

Knee osteoarthritis analysis from TUM ADLM practical (WS 2025-2026). The project trains 3D autoencoders on knee MRI volumes and uses the learned latent representations to cluster patients and correlate clusters with clinical outcomes (WOMAC, JSN, KL grades, surgery). A second track using DINOv3 is also being developed (`dinov3/`).

## Repository Structure

```
autoencoders/          # Training code for all AE variants
  ae_filippo/          # Base 3D autoencoder
  ae_pain/             # + pain prediction head
  ae_tabular_input/    # + tabular fusion + WOMAC/JSN/surgery heads
  ae_tabular_input_masked/  # same, with missing-data masking
autoencoders_results/  # Inference and results for AE models
  tabular_input_inference/
  tabular_input_inference_masked/
  results/             # Model checkpoints, cluster CSVs, plots
  recon_samples_20/    # PNG reconstructions at 20 epochs
  recon_samples_50/    # PNG reconstructions at 50 epochs
inference/             # General inference pipeline (base AE / pain AE)
dinov3/                # DINOv3-based track (in development)
weights_dinov3/        # DINOv3 pretrained weights
csv/                   # Clinical data (WOMAC, KOOS, JSN, KL grades, surgery)
```

## sys.path Convention

All scripts use a `PROJECT_ROOT` pattern so they can be run from the repo root without installing packages. The key rule:

- Files inside `autoencoders/ae_*/` → `PROJECT_ROOT` = `autoencoders/` (makes `ae_filippo`, `ae_pain`, etc. importable)
- Files inside `inference/` → `PROJECT_ROOT` = repo root; additionally appends `autoencoders/` for AE model imports
- Files inside `autoencoders_results/*/` → `PROJECT_ROOT` = `autoencoders_results/`; additionally appends repo root (for `inference.*`) and `autoencoders/` (for `ae_*.*`)

When adding a new script in one of these directories, follow the same pattern that already exists in the sibling files.

## Architecture

### Data Pipeline
- MRI data stored as DICOM files in `tar.gz` archives
- `autoencoders/ae_filippo/data.py`: Extracts DICOM archives, reconstructs 3D volumes (160×224×224), normalizes to [-1, 1]
- `autoencoders/ae_filippo/data_t.py`: `KneeMRIDataset` — handles train/val/test splits
- Clinical data: `csv/clinical00_cleaned.csv`
- Variable mappings per knee side: `variables.json` (full), `variables_t.json` (truncated)

### Model Variants

All models share the same base encoder/decoder from `autoencoders/ae_filippo/model.py`:
- **Encoder3D:** 4× Conv3d blocks with stride-2 downsampling (1→32→64→128→64 channels) → 64-dim latent vector
- **Decoder3D:** 4× ConvTranspose3d blocks with Tanh output

| Directory | Model | Extra heads |
|-----------|-------|-------------|
| `autoencoders/ae_filippo/` | Base 3D autoencoder | None |
| `autoencoders/ae_pain/` | + `PainHead` (global avg pool → linear) | Pain score (regression) |
| `autoencoders/ae_tabular_input/` | + `TabularEncoder` (2-layer MLP) + fusion projection | WOMAC (regression), JSN (4-class), Surgery (binary) |
| `autoencoders/ae_tabular_input_masked/` | Same as above | Same, with missing-data masking |

**Tabular fusion:** MRI latent + encoded tabular features are concatenated and projected back to 64-dim before decoding.

**Test IDs:** The tabular AE train scripts write held-out test patient IDs to `autoencoders/test_ids/`. The inference scripts in `autoencoders_results/` read from that same location.

### Inference Pipeline (`inference/`)

`inference/main.py` orchestrates the full pipeline for the base and pain AE models:
1. **Feature extraction** (`infer.py`): Run encoder on dataset → latent vectors
2. **Clustering** (`cluster.py`): K-means on latent features (configurable `--k`, default 5)
3. **Statistics** (`score.py`): Correlate cluster assignments with WOMAC, JSN, KL grades, surgery %
4. **Visualization** (`tsne_visualization.py`): t-SNE plots (1D/2D/3D) colored by cluster

```bash
uv run inference/main.py --data_root <path> --weights_path <path> --model_name <autoencoder|autoencoder_pain|resnet50> --side <left|right|both> --k <int>
```

For tabular input models, use the corresponding `autoencoders_results/tabular_input_inference/main.py` or `autoencoders_results/tabular_input_inference_masked/main.py`.
