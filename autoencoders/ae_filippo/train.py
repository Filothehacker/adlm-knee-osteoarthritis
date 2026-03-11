import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D
from ae_filippo.train_f import train_autoencoder
from ae_filippo.data_t import KneeMRIDataset
from ae_filippo.save import save_reconstruction
import wandb


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    device = get_device()
    print(f"Using device: {device}")

    # Load train + validation sets
    train_set = KneeMRIDataset(data_root, side="left", split="train")
    val_set   = KneeMRIDataset(data_root, side="left", split="val")

    if len(train_set) == 0:
        print("ERROR: No training volumes found! Check dataset path and splits.")
        return


    # Dataloaders

    is_cuda = device == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda and len(val_set) > 0,
    )

    # Model
    model = Autoencoder3D(in_channels=1, latent_channels=64).to(device)

    # Train with validation support
    num_epochs=50
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=1e-4,
        device=device,
        use_amp=is_cuda,
        use_wandb=False,  # change to True if using wandb
    )

    # Save final model
    
    torch.save(trained_model.state_dict(), "weights_ae/"+str(num_epochs)+"_ae.pth")
    print("Model saved to trained_knee_3d_autoencoder.pth")

    # Save example reconstruction
    if len(val_set) > 0:
        save_reconstruction(trained_model, val_set, device)
    else:
        save_reconstruction(trained_model, train_set, device)


if __name__ == "__main__":
    main()
