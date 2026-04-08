"""
05_classifier.py
Trains two MLP classifiers:
  A) Raw features  : 2000-dim gene expression → tumor type
  B) Latent features: 64-dim autoencoder embedding → tumor type
Saves both models and a comparison of validation metrics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES = 5
EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 1e-3
DROPOUT     = 0.3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(prefix: str, use_latent: bool):
    """Load X (raw or latent) and y for one split."""
    x_key = f"Z_{prefix}" if use_latent else f"X_{prefix}"
    X = torch.tensor(np.load(os.path.join(PROC_DIR, f"{x_key}.npy")), dtype=torch.float32)
    y = torch.tensor(np.load(os.path.join(PROC_DIR, f"y_{prefix}.npy")), dtype=torch.long)
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_X, val_y, label: str):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        scheduler.step()

        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            preds = model(val_X.to(DEVICE)).argmax(dim=1).cpu().numpy()
        val_accs.append(accuracy_score(val_y.numpy(), preds))

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:3d}/{EPOCHS} — "
                  f"loss: {train_losses[-1]:.4f}  val acc: {val_accs[-1]:.4f}")

    return train_losses, val_accs


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_comparison(raw_accs, latent_accs, save_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(raw_accs,    label="MLP — raw (2000-dim)",    color="#0072B2", linewidth=1.8)
    ax.plot(latent_accs, label="MLP — latent (64-dim AE)", color="#E69F00", linewidth=1.8,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("MLP Classifier: Raw vs Autoencoder Latent Features",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "classifier_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    results = {}

    for use_latent, label, hidden in [
        (False, "raw",    [512, 128]),
        (True,  "latent", [128,  64]),
    ]:
        print(f"── Training MLP on {'latent' if use_latent else 'raw'} features ──")
        X_train, y_train = load_split("train", use_latent)
        X_val,   y_val   = load_split("val",   use_latent)

        input_dim    = X_train.shape[1]
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=BATCH_SIZE, shuffle=True)

        model = MLP(input_dim, hidden, NUM_CLASSES, DROPOUT).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Input dim: {input_dim}  |  Params: {total:,}")

        train_losses, val_accs = train_model(model, train_loader, X_val, y_val, label)

        # Save model
        path = os.path.join(MODEL_DIR, f"mlp_{label}.pt")
        torch.save(model.state_dict(), path)
        print(f"  ✓ Model saved: {path}")

        results[label] = {"model": model, "val_accs": val_accs,
                          "best_val_acc": max(val_accs)}
        print()

    # Summary
    print("── Results Summary ──")
    for label, r in results.items():
        print(f"  {label:8s} — best val acc: {r['best_val_acc']:.4f}")

    plot_comparison(results["raw"]["val_accs"], results["latent"]["val_accs"], FIG_DIR)

    # Save best val accs for evaluate script
    np.save(os.path.join(MODEL_DIR, "val_accs_raw.npy"),    results["raw"]["val_accs"])
    np.save(os.path.join(MODEL_DIR, "val_accs_latent.npy"), results["latent"]["val_accs"])

    print("\nDone.")


if __name__ == "__main__":
    main()