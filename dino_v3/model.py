import copy
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make the dinov3 package importable (it lives at <repo>/dinov3/)
DINOV3_PKG = os.path.join(PROJECT_ROOT, "dinov3")
if DINOV3_PKG not in sys.path:
    sys.path.insert(0, DINOV3_PKG)

import torch
import torch.nn as nn


def _sinusoidal_1d(positions: torch.Tensor, d: int, base: int = 10000) -> torch.Tensor:
    """
    Build sinusoidal 1-D positional embeddings using the RoPE frequency formula.

    Parameters
    ----------
    positions : torch.Tensor
        1-D integer tensor of positions, shape [N].
    d : int
        Embedding dimensionality (must be even).
    base : int
        Frequency base for the RoPE formula (default: 10 000).

    Returns
    -------
    torch.Tensor
        Shape [N, d].  Even indices hold cosine values, odd indices sine values.
    """
    assert d % 2 == 0, "d must be even"
    theta = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    pos = positions.float().unsqueeze(1)  # [N, 1]
    freqs = pos * theta.unsqueeze(0)  # [N, d//2]
    emb = torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [N, d//2, 2]
    return emb.reshape(len(positions), d)  # [N, d]


def _build_depth_encoding(
    depth_indices: torch.Tensor,
    n_patches_per_slice: int,
    d_model: int = 768,
    base: int = 10000,
) -> torch.Tensor:
    """
    Pre-compute an additive depth positional buffer.

    Each of the 40 slices gets a unique sinusoidal embedding of dimension
    ``d_model``, which is then broadcast across all ``n_patches_per_slice``
    patch tokens of that slice.  Row and column axes are intentionally omitted
    because the backbone's 2-D positional encoding has already mixed spatial
    (row, col) information into the patch token values through 11 transformer
    blocks.  Depth is the only axis with no pre-existing positional signal.

    Parameters
    ----------
    depth_indices : torch.Tensor
        Anatomical slice indices, e.g. ``torch.arange(60, 100)``.  Using the
        actual anatomical indices (rather than re-indexed 0–39) preserves
        anatomical meaning across patients.
    n_patches_per_slice : int
        Number of patch tokens per slice (196 for 224×224 with patch_size=16).
    d_model : int
        Total feature dimension (768).
    base : int
        Frequency base for the sinusoidal formula.

    Returns
    -------
    torch.Tensor
        Shape ``[1, D*n_patches_per_slice, d_model]`` — ready to be registered
        as a buffer and broadcast-added to a ``[B, D*P, d_model]`` sequence.
    """
    D = len(depth_indices)
    depth_emb = _sinusoidal_1d(depth_indices, d_model, base)  # [D, 768]
    # Broadcast the same depth embedding across all patches of that slice
    depth_emb = depth_emb.unsqueeze(1).expand(D, n_patches_per_slice, d_model)
    return depth_emb.reshape(1, D * n_patches_per_slice, d_model)  # [1, 7840, 768]


class VolumetricDINOv3(nn.Module):
    """
    Volumetric DINOv3 inference model for 3-D knee MRI feature extraction.

    Pipeline
    --------
    1. Center-crop the input volume to 40 anatomically relevant slices (60–99).
    2. Extract dense patch tokens from the frozen DINOv3 ViT-B/16 backbone via
       a forward hook on ``backbone.blocks[-1].attn``.
    3. Apply additive depth positional embeddings (depth axis only).
    4. Pass the sequence through 4 frozen deep-copied DINOv3 transformer blocks
       followed by the copied LayerNorm.
    5. Max-pool across all 7840 tokens to a single ``[B, 768]`` patient vector.

    Parameters
    ----------
    backbone : nn.Module
        Frozen DINOv3 ViT-B/16 backbone.

    Notes
    -----
    * All components are frozen (``requires_grad=False``) — this is a
      purely inference-time feature extractor with no trainable parameters.
    * Max-pooling captures the most extreme activation per feature dimension,
      which aligns with how OA severity is clinically assessed: by the
      worst-affected region, not the average.
    """

    SLICE_START: int = 60
    SLICE_END: int = 100  # exclusive → 40 slices
    N_SLICES: int = 40
    PATCH_SIZE: int = 16
    IMG_SIZE: int = 224
    N_PATCHES_PER_DIM: int = 14  # 224 // 16
    N_PATCHES_PER_SLICE: int = 196  # 14 * 14
    N_TOKENS: int = 7840  # 40 * 196
    D_MODEL: int = 768

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Deep-copy last 4 transformer blocks and freeze them
        self.vol_blocks = nn.ModuleList(
            [copy.deepcopy(backbone.blocks[i]) for i in range(-4, 0)]
        )
        for block in self.vol_blocks:
            block.requires_grad_(False)

        # Deep-copy and freeze the final LayerNorm
        self.vol_norm = copy.deepcopy(backbone.norm)
        self.vol_norm.requires_grad_(False)


        # Depth positional encoding buffer — pre-computed, not a trainable parameter.
        # Row/col information is already embedded in the patch token values by the
        # backbone's 2-D positional encoding; only depth is genuinely missing.
        depth_indices = torch.arange(self.SLICE_START, self.SLICE_END)
        depth_enc = _build_depth_encoding(
            depth_indices,
            n_patches_per_slice=self.N_PATCHES_PER_SLICE,
            d_model=self.D_MODEL,
        )
        self.register_buffer("depth_enc", depth_enc)  # [1, 7840, 768]

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vol : torch.Tensor
            MRI volume of shape ``[B, 1, D, 224, 224]``.  ``B=1`` during
            inference.

        Returns
        -------
        torch.Tensor
            Patient feature vector of shape ``[B, 768]``.
        """
        B, C, D, H, W = vol.shape
        assert C == 1, "Expected single-channel MRI volume"
        assert vol.shape[2] >= 100, (
            f"Volume has fewer than 100 slices, cannot center-crop 60-99"
        )

        # --- Step 1: center-crop slices 60–99 and expand channel ---
        # [B, 1, D, H, W] -> [B, 1, 40, H, W]
        vol_crop = vol[:, :, self.SLICE_START : self.SLICE_END, :, :]
        # [B, 1, 40, H, W] -> [B*40, 1, H, W] -> [B*40, 3, H, W]
        slices = vol_crop.permute(0, 2, 1, 3, 4).reshape(B * self.N_SLICES, 1, H, W)
        slices = slices.expand(-1, 3, -1, -1)

        # --- Step 2: extract patch tokens via forward hook ---
        patch_tokens_store: list[torch.Tensor] = []

        def _hook(module, inputs, output):
            # inputs[0]: [B*40, n_all_tokens, d_model]
            # Skip CLS (1) + register tokens (4) = first 5 tokens
            patch_tokens_store.append(inputs[0][:, 5:, :])

        handle = self.backbone.blocks[-1].attn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.backbone(slices)
        finally:
            handle.remove()

        patch_tokens = patch_tokens_store[0]  # [B*40, 196, 768]
        assert patch_tokens.shape == (
            B * self.N_SLICES,
            self.N_PATCHES_PER_SLICE,
            self.D_MODEL,
        ), f"Unexpected patch token shape: {patch_tokens.shape}"

        # --- Step 3: reshape and inject depth encoding ---
        x = patch_tokens.reshape(B, self.N_TOKENS, self.D_MODEL)  # [B, 7840, 768]
        x = x + self.depth_enc  # depth-only additive positional signal

        # --- Step 4: 4 frozen transformer blocks + LayerNorm ---
        for block in self.vol_blocks:
            x = block(x)
        x = self.vol_norm(x)

        # --- Step 5: max-pool across the token sequence ---
        out = x.max(dim=1).values  # [B, 768]

        assert out.shape == (B, self.D_MODEL), f"Unexpected output shape: {out.shape}"
        return out


def build_volumetric_model(
    weights_path: str, device: torch.device, tmp_dir: str | None = None
) -> VolumetricDINOv3:
    """
    Build and return a :class:`VolumetricDINOv3` model loaded from local or
    GCS weights.

    Parameters
    ----------
    weights_path : str
        Local file path **or** a ``gs://bucket/blob`` URI pointing to the
        pretrained ViT-B/16 ``.pth`` weights file, e.g.
        ``gs://koa-thesis-oai-data/weights/dinov3_vitb16_pretrain_lvd1689m.pth``.
    device : torch.device
        Target device.
    tmp_dir : str, optional
        Scratch directory for temporary GCS checkpoint downloads.  Defaults to
        ``TMPDIR`` when set.

    Returns
    -------
    VolumetricDINOv3
        Model in eval mode on the requested device.

    Raises
    ------
    FileNotFoundError
        If ``weights_path`` is a local path that does not exist, or if the
        GCS blob does not exist in the specified bucket.
    """
    from dinov3.hub.backbones import dinov3_vitb16

    print(f"Loading DINOv3 ViT-B/16 weights from {weights_path} ...")

    if isinstance(weights_path, str) and weights_path.startswith("gs://"):
        import tempfile
        from google.cloud import storage as gcs
        from google.cloud.exceptions import NotFound

        without_scheme = weights_path[len("gs://"):]
        bucket_name, blob_name = without_scheme.split("/", 1)
        client = gcs.Client()
        blob = client.bucket(bucket_name).blob(blob_name)

        try:
            blob.reload()  # raises NotFound if blob does not exist
        except NotFound:
            raise FileNotFoundError(
                f"GCS weights blob not found: {weights_path}"
            )

        suffix = os.path.splitext(blob_name)[-1] or ".pth"
        if tmp_dir is not None:
            os.makedirs(tmp_dir, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
            dir=tmp_dir or os.environ.get("TMPDIR"),
        )
        tmp.close()  # close our handle so download_to_filename can write cleanly
        try:
            blob.download_to_filename(tmp.name)
            resolved_path = tmp.name
        except Exception:
            os.unlink(tmp.name)
            raise
    else:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Local weights file not found: {weights_path}"
            )
        resolved_path = weights_path
        tmp = None

    try:
        backbone = dinov3_vitb16(pretrained=True, weights=resolved_path)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)

    backbone = backbone.to(device)
    backbone.eval()

    model = VolumetricDINOv3(backbone).to(device)
    model.eval()
    return model
