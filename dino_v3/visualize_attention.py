"""
Attention map visualization for DINOv3 on 3D MRI slices.

For each selected slice of a patient's MRI volume, registers a forward hook
on the last transformer block's SelfAttention module to extract the raw Q and K
projections and manually compute softmax attention weights (bypassing
scaled_dot_product_attention which discards them).

The [CLS] token attention over patch tokens is reshaped to a 14×14 spatial
grid and overlaid as a heatmap on the original MRI slice.

Usage:
  uv run dino_v3/visualize_attention.py \
      --data_root <path> \
      --weights_path weights_dinov3/dinov3_vitb16_pretrain_lvd1689m.pth \
      --patient_id <id> \
      --side left \
      [--n_slices 16]          # how many top max-pool winning slices to show
      [--output_dir dino_v3_results/attention_maps]
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DINOV3_PKG = os.path.join(PROJECT_ROOT, "dinov3")
if DINOV3_PKG not in sys.path:
    sys.path.insert(0, DINOV3_PKG)

AUTOENCODERS_DIR = os.path.join(PROJECT_ROOT, "autoencoders")
if AUTOENCODERS_DIR not in sys.path:
    sys.path.insert(0, AUTOENCODERS_DIR)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ae_filippo.data import load_single_patient_mri
from dino_v3.model import build_dino_model

# ViT-B/16 on 224×224: 14×14 = 196 patch tokens
PATCH_GRID = 14
N_STORAGE = 4   # n_storage_tokens for ViT-B/16
# Token layout: [CLS(1), storage(4), patches(196)]
PATCH_START = 1 + N_STORAGE  # = 5


def _make_attn_hook(store: dict):
    """
    Forward hook for SelfAttention.forward.
    Manually recomputes softmax(QK^T / scale) to recover attention weights
    that scaled_dot_product_attention discards.
    """
    def hook(module, inputs, output):
        x = inputs[0]           # [B, N, C]  (after LayerNorm)
        B, N, C = x.shape
        head_dim = C // module.num_heads

        with torch.no_grad():
            qkv = module.qkv(x.float())           # [B, N, 3C]
            qkv = qkv.reshape(B, N, 3, module.num_heads, head_dim)
            q, k, _ = torch.unbind(qkv, dim=2)    # each [B, N, heads, head_dim]
            q = q.transpose(1, 2)                  # [B, heads, N, head_dim]
            k = k.transpose(1, 2)

            scale = head_dim ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            # attn: [B, heads, N, N]
            store["attn"] = attn.detach().cpu()
    return hook


def extract_cls_attn_map(attn: torch.Tensor) -> np.ndarray:
    """
    From attn [B, heads, N, N], extract CLS→patch attention,
    average over heads, and reshape to [PATCH_GRID, PATCH_GRID].
    """
    # CLS row (token 0), patch columns [PATCH_START:]
    cls_attn = attn[0, :, 0, PATCH_START:]     # [heads, 196]
    cls_attn = cls_attn.mean(dim=0)             # [196]
    cls_attn = cls_attn.reshape(PATCH_GRID, PATCH_GRID).numpy()
    # Min-max normalise to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


def visualize_patient(
    data_root: str,
    patient_id: str,
    side: str,
    weights_path: str,
    n_slices: int,
    output_dir: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dino_model(weights_path=weights_path, device=device)
    backbone = model.backbone

    # Register hook on last block's attention module
    last_block = backbone.blocks[-1]
    attn_store = {}
    hook_handle = last_block.attn.register_forward_hook(_make_attn_hook(attn_store))

    # Load volume: [1, D, 224, 224]
    print(f"Loading MRI for patient {patient_id} ({side}) ...")
    vol = load_single_patient_mri(data_root, patient_id, side=side)
    if vol is None:
        print(f"Could not load MRI for patient {patient_id}.")
        return

    D = vol.shape[1]

    # Find the slices that contribute most to the max-pooled feature vector.
    # Run all D slices through the backbone to get CLS tokens: [D, embed_dim]
    print("Finding max-pool winning slices ...")
    all_slices = vol[0].unsqueeze(1).expand(-1, 3, -1, -1).to(device)  # [D, 3, 224, 224]
    with torch.no_grad():
        all_cls = backbone(all_slices)  # [D, embed_dim]

    # For each feature dim, which slice had the max? Count wins per slice.
    winning_indices = all_cls.argmax(dim=0).cpu().numpy()  # [embed_dim]
    counts = np.bincount(winning_indices, minlength=D)     # [D]

    # Top n_slices by win count, sorted back to anatomical order
    slice_indices = np.sort(np.argsort(counts)[::-1][:n_slices])

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2, n_slices,
        figsize=(n_slices * 2.5, 6),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    print(f"Extracting attention maps for {n_slices} slices ...")
    for col, s_idx in enumerate(slice_indices):
        # Single slice: [1, 224, 224] -> [1, 3, 224, 224]
        raw_slice = vol[0, s_idx]               # [224, 224]  in [-1, 1]
        img_np = ((raw_slice.numpy() + 1) / 2)  # [0, 1]

        slice_tensor = raw_slice.unsqueeze(0).unsqueeze(0)   # [1, 1, 224, 224]
        slice_3ch = slice_tensor.expand(-1, 3, -1, -1).to(device)  # [1, 3, 224, 224]

        with torch.no_grad():
            backbone(slice_3ch)  # triggers the hook

        attn_map = extract_cls_attn_map(attn_store["attn"])  # [14, 14]

        # Upsample attention map to 224×224 for overlay
        attn_up = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
        attn_up = torch.nn.functional.interpolate(
            attn_up, size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Row 0: raw MRI slice
        axes[0, col].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"slice {s_idx}", fontsize=7)
        axes[0, col].axis("off")

        # Row 1: attention heatmap overlaid on slice
        axes[1, col].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(attn_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[1, col].axis("off")

    # Add row labels
    axes[0, 0].set_ylabel("MRI", fontsize=8)
    axes[1, 0].set_ylabel("Attention", fontsize=8)
    for ax in axes[:, 0]:
        ax.yaxis.set_visible(True)
        ax.set_yticks([])

    fig.suptitle(
        f"DINOv3 [CLS] attention — patient {patient_id} ({side} knee)",
        fontsize=10,
    )

    out_path = os.path.join(output_dir, f"attention_{patient_id}_{side}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention map to {out_path}")

    hook_handle.remove()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 attention map visualizer")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--weights_path",
        type=str,
        default="weights_dinov3/dinov3_vitb16_pretrain_lvd1689m.pth",
    )
    parser.add_argument("--patient_id", type=str, required=True,
                        help="Patient ID to visualize (e.g. 9002316)")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--n_slices", type=int, default=16,
                        help="Number of evenly-spaced slices to visualise")
    parser.add_argument("--output_dir", type=str,
                        default="dino_v3_results/attention_maps")
    args = parser.parse_args()

    visualize_patient(
        data_root=args.data_root,
        patient_id=args.patient_id,
        side=args.side,
        weights_path=args.weights_path,
        n_slices=args.n_slices,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
