import argparse
import os
import sys
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

REPO_ROOT = os.path.dirname(PROJECT_ROOT)
AUTOENCODERS_DIR = os.path.join(REPO_ROOT, "autoencoders")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if AUTOENCODERS_DIR not in sys.path:
    sys.path.append(AUTOENCODERS_DIR)

from ae_tabular_input.model import AutoencoderWithTabularInput, Autoencoder3D
from ae_tabular_input.dataset import KneeMRITabularDataset

def build_tabular_input_ae_from_weights(weights_path: str, device: torch.device, in_channels: int, latent_channels: int, num_tabular_inputs: int, num_tabular_outputs: int, tabular_latent_dim: int) -> AutoencoderWithTabularInput:

    base_ae = Autoencoder3D(in_channels=in_channels, latent_channels=latent_channels)
    base_ae.to(device)

    model = AutoencoderWithTabularInput(
        base_ae=base_ae,
        latent_channels=latent_channels,
        num_tabular_inputs=num_tabular_inputs,
        num_tabular_outputs=num_tabular_outputs,
        tabular_latent_dim=tabular_latent_dim,
    ).to(device)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"[build_tabular_input_ae_from_weights] Loaded pretrained tabular input AE from {weights_path}")

    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_reconstruction(model, dataset, device, outdir="recon_samples", num_samples=5):
    os.makedirs(outdir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            vol, tabular_vars = dataset[i]

            vol = vol.unsqueeze(0)
            vol = vol.to(device)

            tabular_vars = tabular_vars.unsqueeze(0)
            tabular_vars = tabular_vars.to(device)

            recon, _, _ = model(vol, tabular_vars)

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruction of MRI images using the tabular input autoencoder."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
        help="Root folder containing 0.C.2 and 0.E.1 directories",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./weights_tabular_input_ae/best_tabular_input_ae_training_phase.pth",
        help="Path to model weights. If omitted, a model-specific default is used.",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="left",
        help="Which side(s) to process",
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="csv/clinical00_cleaned.csv",
        help="Path to cleaned clinical CSV.",
    )
    parser.add_argument(
        "--recons_dir",
        type=str,
        default="reconstruction/recon_samples",
        help="Directory to save reconstructions to.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of reconstructions to generate.",
    )

    # NOTE: We can set this in a comma delimited way as follows: "tab_var_1,tab_var_2,tab_var_3"
    parser.add_argument(
        "--tabular_variables",
        type=str,
        default="V00KOOSKPL",
        help="Tabular inputs to consider together with the MRI image.",
    )

    args = parser.parse_args()

    tabular_variables = [item for item in args.tabular_variables.split(',')]
    num_tabular_inputs = len(tabular_variables)
    clinical_csv = args.clinical_csv
    data_root = args.data_root
    num_samples = args.num_samples

    # Choose default weights if none provided.
    if args.weights_path is None:
        weights_path = "./weights_tabular_input_ae/best_tabular_input_ae_training_phase.pth"
    else:
        weights_path = args.weights_path

    print(f"Weights: {weights_path}")


    # Reconstruction directory.
    results_dir = os.path.join("results", "tabular_input_ae")
    recon_dir = os.path.join(results_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    if args.side == "both":
        sides = ["left", "right"]
    else:
        sides = [args.side]

    device = get_device()
    print(f"Using device: {device}")

    # ----------------- BUILD MODEL -----------------
    print("Building model...")
    model = build_tabular_input_ae_from_weights(weights_path=weights_path, device=device, in_channels=1, latent_channels=64, num_tabular_inputs=num_tabular_inputs, num_tabular_outputs=1, tabular_latent_dim=4)

    for side in sides:

        # ----------------- LOAD DATA -----------------
        print(f"Loading dataset...")
        
        recon_set = KneeMRITabularDataset(
            root=data_root,
            clinical_csv_path=clinical_csv, 
            tabular_variables=tabular_variables, 
            side=side, 
            split="val", 
        )

        save_reconstruction(model=model, dataset=recon_set, device=device, outdir=recon_dir, num_samples=num_samples)
        
    return

if __name__ == "__main__":
    main()