"""
04_autoencoder.py
Trains a deep autoencoder on the gene expression training set.
The encoder compresses 2000 genes → 64-dim latent space.
Saves encoder weights and latent embeddings for downstream classification.
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "models")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

# ── Hyperparameters ───────────────────────────────────────────────────────────
INPUT_DIM   = 2000
LATENT_DIM  = 64
HIDDEN_DIMS = [1024, 256]   # encoder layers (decoder mirrors these)
EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
DROPOUT     = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 2000 → 1024 → 256 → 64
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIMS[0]),
            nn.BatchNorm1d(HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.BatchNorm1d(HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIMS[1], LATENT_DIM),
            nn.ReLU(),
        )

        # Decoder: 64 → 256 → 1024 → 2000
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIMS[1]),
            nn.BatchNorm1d(HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[0]),
            nn.BatchNorm1d(HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIMS[0], INPUT_DIM),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_tensors():
    X_train = torch.tensor(np.load(os.path.join(PROC_DIR, "X_train.npy")), dtype=torch.float32)
    X_val   = torch.tensor(np.load(os.path.join(PROC_DIR, "X_val.npy")),   dtype=torch.float32)
    X_test  = torch.tensor(np.load(os.path.join(PROC_DIR, "X_test.npy")),  dtype=torch.float32)
    return X_train, X_val, X_test


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_tensor, criterion, optimizer, scheduler):
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss  = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(DEVICE))
            val_loss  = criterion(val_recon, val_tensor.to(DEVICE)).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} — train loss: {train_loss:.5f}  val loss: {val_loss:.5f}")

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train", color="#0072B2")
    ax.plot(val_losses,   label="Val",   color="#E69F00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "autoencoder_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def save_latent_embeddings(model, X_train, X_val, X_test):
    model.eval()
    with torch.no_grad():
        for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
            z = model.encode(X.to(DEVICE)).cpu().numpy()
            np.save(os.path.join(PROC_DIR, f"Z_{split}.npy"), z)
            print(f"  ✓ Latent {split}: {z.shape}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    X_train, X_val, X_test = load_tensors()
    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

    model     = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    print(f"\nModel architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    print("Training autoencoder...")
    train_losses, val_losses = train(model, train_loader, X_val, criterion, optimizer, scheduler)

    # Save model
    model_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  ✓ Model saved: {model_path}")

    plot_losses(train_losses, val_losses, FIG_DIR)

    print("\nGenerating latent embeddings...")
    save_latent_embeddings(model, X_train, X_val, X_test)

    print("\nDone.")


if __name__ == "__main__":
    main()