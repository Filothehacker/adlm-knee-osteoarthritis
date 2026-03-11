import os
import sys
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_tabular_input.dataset import KneeMRITabularDataset, write_test_ids
from ae_tabular_input.model import build_tabular_input_ae


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_ids_disjoint(
    all_ids: List[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:

    ids = list(all_ids)  
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    return train_ids, val_ids, test_ids


def train_tabular_input_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 50,
    lr: float = 1e-4,
    lambda_tabular: float = 0.5,
    device: str = "cuda",
    use_amp: bool = True,
    phase_name: str = "train",
    weights_dir: str = "weights_tabular_input_ae",
) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    os.makedirs(weights_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ----------------- TRAIN -----------------
        model.train()
        recon_sum = womac_sum = jsn_sum = surg_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs}")
        for x, tab_x, y in pbar:
            x = x.to(device, non_blocking=True)
            tab_x = tab_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                x_hat, womac_pred, jsn_pred, surg_pred, _ = model(x, tab_x)

                target_womac = y[:, 0:1]
                target_jsn = (y[:, 1] * 3).round().clamp(0, 3).long()
                target_surg = torch.clamp(y[:, 2:3], 0.0, 1.0)

                loss_recon = nn.functional.mse_loss(x_hat, x)
                loss_womac = nn.functional.mse_loss(womac_pred, target_womac)
                loss_jsn = nn.functional.cross_entropy(jsn_pred, target_jsn)
                loss_surg = nn.functional.binary_cross_entropy_with_logits(surg_pred, target_surg)

                loss = loss_recon + lambda_tabular * (loss_womac + loss_jsn + loss_surg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            recon_sum += loss_recon.item()
            womac_sum += loss_womac.item()
            jsn_sum += loss_jsn.item()
            surg_sum += loss_surg.item()

        n = max(len(train_loader), 1)
        print(
            f"[{phase_name} Epoch {epoch}] "
            f"Train: recon={recon_sum/n:.6f}, "
            f"womac={womac_sum/n:.6f}, "
            f"jsn={jsn_sum/n:.6f}, "
            f"surgery={surg_sum/n:.6f}"
        )

        # ----------------- VALIDATION -----------------
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            recon_sum = womac_sum = jsn_sum = surg_sum = 0.0

            with torch.no_grad():
                for x, tab_x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    tab_x = tab_x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with autocast(enabled=use_amp):
                        x_hat, womac_pred, jsn_pred, surg_pred, _ = model(x, tab_x)

                        target_womac = y[:, 0:1]
                        target_jsn = (y[:, 1] * 3).round().clamp(0, 3).long()
                        target_surg = torch.clamp(y[:, 2:3], 0.0, 1.0)

                        loss_recon = nn.functional.mse_loss(x_hat, x)
                        loss_womac = nn.functional.mse_loss(womac_pred, target_womac)
                        loss_jsn = nn.functional.cross_entropy(jsn_pred, target_jsn)
                        loss_surg = nn.functional.binary_cross_entropy_with_logits(surg_pred, target_surg)

                    recon_sum += loss_recon.item()
                    womac_sum += loss_womac.item()
                    jsn_sum += loss_jsn.item()
                    surg_sum += loss_surg.item()

            n = max(len(val_loader), 1)
            val_recon = recon_sum / n
            val_womac = womac_sum / n
            val_jsn = jsn_sum / n
            val_surg = surg_sum / n
            val_loss = val_recon + lambda_tabular * (val_womac + val_jsn + val_surg)

            print(
                f"[{phase_name} Epoch {epoch}] "
                f"Val:   recon={val_recon:.6f}, "
                f"womac={val_womac:.6f}, "
                f"jsn={val_jsn:.6f}, "
                f"surgery={val_surg:.6f}"
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(weights_dir, f"best_tabular_input_ae_{phase_name}.pth")
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved to {best_path}")
        else:
            scheduler.step(recon_sum / max(len(train_loader), 1))

    return model


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    clinical_csv_path = "./csv/clinical00_cleaned.csv"
    side = "left"

    batch_size = 8
    num_workers = 8
    num_epochs = 50
    lr = 1e-4
    lambda_tabular = 0.5
    weights_dir = "weights_tabular_input_ae"

    train_frac = 0.70
    val_frac = 0.15
    seed = 42  

    device = get_device()
    is_cuda = device == "cuda"


    master = KneeMRITabularDataset(
        root=data_root,
        clinical_csv_path=clinical_csv_path,
        patient_ids=None,  
        side=side,
        require_all_targets=True,
        require_all_inputs=True,
    )
    all_ids = sorted(master.patient_ids)

    # Split ids disjointly 
    train_ids, val_ids, test_ids = split_ids_disjoint(
        all_ids=all_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    # Persist test ids for inference later 
    out_dir = os.path.join(PROJECT_ROOT, "test_ids")
    write_test_ids(test_ids=test_ids, side=side, out_dir=out_dir)

    # Create datasets from the fixed ids 
    train_set = KneeMRITabularDataset(
        root=data_root,
        clinical_csv_path=clinical_csv_path,
        patient_ids=train_ids,
        side=side,
        require_all_targets=True,
        require_all_inputs=True,
    )
    val_set = KneeMRITabularDataset(
        root=data_root,
        clinical_csv_path=clinical_csv_path,
        patient_ids=val_ids,
        side=side,
        require_all_targets=True,
        require_all_inputs=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if is_cuda else 0,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
    )

    model = build_tabular_input_ae(
        pretrained_ae_path=None,
        device=torch.device(device),
        in_channels=1,
        latent_channels=64,
        num_tabular_inputs=len(train_set.tab_vars),
    )

    train_tabular_input_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        lambda_tabular=lambda_tabular,
        device=device,
        use_amp=is_cuda,
        phase_name="training_phase",
        weights_dir=weights_dir,
    )

    os.makedirs(weights_dir, exist_ok=True)
    final_path = os.path.join(weights_dir, f"final_tabular_input_ae_{num_epochs}_epochs.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
