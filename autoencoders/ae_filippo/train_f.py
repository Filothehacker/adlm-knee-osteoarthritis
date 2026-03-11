import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def train_autoencoder(
    model,
    train_loader,
    val_loader=None,
    num_epochs=20,
    lr=1e-4,
    device="cuda",
    use_amp=True,
    use_wandb=False,
):
    """
    Trains the autoencoder with optional validation.
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):

        # TRAIN
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} (train)")

        for x in pbar:                     
            x = x.to(device)
            y = x                           # Autoencoder target = input

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                x_hat, _ = model(x)         # model returns (reconstruction, latent)
                loss = criterion(x_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}")

        if use_wandb:
            import wandb
            wandb.log({"train_loss": train_loss})

        # VALIDATION
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for x in val_loader:        
                    x = x.to(device)
                    y = x

                    with autocast(enabled=use_amp):
                        x_hat, _ = model(x)
                        loss = criterion(x_hat, y)

                    val_loss_sum += loss.item()

            val_loss = val_loss_sum / len(val_loader)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.6f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_autoencoder_val.pth")
                print("✨ New best validation model saved.")

        else:
            # No validation set → fall back to training loss
            scheduler.step(train_loss)

    return model
