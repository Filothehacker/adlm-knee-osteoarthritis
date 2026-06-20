#!/usr/bin/env bash
#
# run_vm_inference.sh — run the DINOv3 inference pipeline on the GCP VM
# (koa-machine-1, NVIDIA Tesla T4).
#
# Everything is kept on the large /mnt/data volume; nothing of size is written
# to /, /tmp, or $HOME. Safe to re-run after an SSH drop or reboot — feature
# extraction resumes from the per-patient cache under the output dir.
#
# Override any path/knob via the env vars below, e.g.:
#   NUM_WORKERS=8 BATCH_SIZE=1 ./run_vm_inference.sh --side left --k 5
# Extra args after the script name are forwarded to dino_v3/main.py.
#
# SURVIVING SSH DISCONNECTS: this runs python in the foreground, which SIGHUP
# kills on disconnect. Launch it under tmux (recommended) or nohup so it keeps
# running, e.g.:
#   tmux new -s koa './run_vm_inference.sh --side both --k 5'
#   nohup ./run_vm_inference.sh --side both --k 5 &
# Either way the run is resume-safe: re-running skips already-cached patients,
# so at most the in-flight patient is lost.

set -euo pipefail

# --- Paths (all on the large disk) -----------------------------------------
export DATA_ROOT="${DATA_ROOT:-/mnt/data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/pip-cache}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/cache/torch}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATA_ROOT}/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${DATA_ROOT}/cache/matplotlib}"

INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/dataset/cleaned_images_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATA_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth}"
CODE_DIR="${CODE_DIR:-${DATA_ROOT}/code/adlm-knee-osteoarthritis}"
VENV_DIR="${VENV_DIR:-${DATA_ROOT}/venvs/inference}"

# T4-safe runtime defaults (16 GB GPU, n1-standard-8 = 8 vCPUs).
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Optional: GCS location to fetch the weights from if not already on disk.
WEIGHTS_GCS="${WEIGHTS_GCS:-gs://koa-inference-data-filippo-20260620/knee-osteoarthritis/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth}"

# --- Create the on-disk layout ---------------------------------------------
mkdir -p \
  "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" \
  "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${OUTPUT_DIR}" \
  "$(dirname "${CHECKPOINT_PATH}")"

export KOA_INPUT_DIR="${INPUT_DIR}"
export KOA_OUTPUT_DIR="${OUTPUT_DIR}"
export KOA_WEIGHTS_PATH="${CHECKPOINT_PATH}"
export KOA_CACHE_DIR="${XDG_CACHE_HOME}"
export KOA_PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export KOA_TMP_DIR="${TMPDIR}"

# --- Activate the venv ------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${CODE_DIR}"

# --- Fetch weights if missing ----------------------------------------------
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Weights not found at ${CHECKPOINT_PATH}; downloading from ${WEIGHTS_GCS} ..."
  gcloud storage cp "${WEIGHTS_GCS}" "${CHECKPOINT_PATH}"
fi

# --- GPU sanity check (non-fatal) ------------------------------------------
nvidia-smi || echo "WARNING: nvidia-smi failed; check GPU/driver."

echo "=== Launching DINOv3 inference ==="
echo "  input_dir   = ${INPUT_DIR}"
echo "  output_dir  = ${OUTPUT_DIR}"
echo "  checkpoint  = ${CHECKPOINT_PATH}"
echo "  device      = ${DEVICE} | batch_size=${BATCH_SIZE} | num_workers=${NUM_WORKERS}"

LOG_FILE="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "  log         = ${LOG_FILE}"

# pipefail (set above) preserves python's exit code through the tee.
python dino_v3/main.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --cache_dir "${XDG_CACHE_HOME}" \
  --pip_cache_dir "${PIP_CACHE_DIR}" \
  --tmp_dir "${TMPDIR}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "=== Done. Results under ${OUTPUT_DIR} (log: ${LOG_FILE}) ==="
