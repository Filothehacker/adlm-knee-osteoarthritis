import os
import torch
import matplotlib.pyplot as plt

def save_reconstruction(model, dataset, device, outdir="recon_samples", num_samples=5):
    os.makedirs(outdir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            vol = dataset[i]            # (1, D, 224, 224)
            vol = vol.unsqueeze(0)      # (B=1, C=1, D, H, W)
            vol = vol.to(device)

            recon, _ = model(vol)

            # Move to CPU
            vol = vol.cpu()[0]      # (1, D, H, W)
            recon = recon.cpu()[0]  # (1, D, H, W)

            original = vol[0]       # (D, 224, 224)
            decoded = recon[0]      # (D, 224, 224)

            # pick 5 representative slices evenly spaced in depth
            D = original.shape[0]
            slice_ids = [D//6, D//3, D//2, 2*D//3, 5*D//6]

            fig, axes = plt.subplots(5, 2, figsize=(6, 12))
            fig.suptitle(f"Patient {i} Reconstruction", fontsize=16)

            for idx, s in enumerate(slice_ids):
                axes[idx, 0].imshow(original[s].numpy(), cmap="gray")
                axes[idx, 0].set_title(f"Original slice {s}")
                axes[idx, 0].axis("off")

                axes[idx, 1].imshow(decoded[s].numpy(), cmap="gray")
                axes[idx, 1].set_title(f"Reconstructed slice {s}")
                axes[idx, 1].axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"patient_{i}_recon.png"))
            plt.close(fig)

    print(f"Saved reconstructions to {outdir}/")

