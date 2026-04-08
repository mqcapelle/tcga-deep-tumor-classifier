"""
06_evaluate.py
Final evaluation of both MLP classifiers on the held-out test set.
Produces:
  - Confusion matrices (raw + latent)
  - Per-class precision / recall / F1 table
  - UMAP of autoencoder latent space, coloured by true vs predicted label
  - Summary figure suitable for a portfolio README
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")
FIG_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PALETTE   = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


# ── Re-declare model (must match 05_classifier.py exactly) ───────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_test(use_latent: bool):
    x_key = "Z_test" if use_latent else "X_test"
    X = torch.tensor(np.load(os.path.join(PROC_DIR, f"{x_key}.npy")), dtype=torch.float32)
    y = torch.tensor(np.load(os.path.join(PROC_DIR, "y_test.npy")),   dtype=torch.long)
    return X, y


def load_model(label: str, input_dim: int, hidden: list[int]) -> MLP:
    model = MLP(input_dim, hidden, 5).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"mlp_{label}.pt"),
                                     map_location=DEVICE))
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(X.to(DEVICE)).argmax(dim=1).cpu().numpy()


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_latent_umap(Z_all, y_all, y_pred, classes, save_dir):
    print("  Running UMAP on latent space (~10s)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=42, verbose=False)
    Z_2d = reducer.fit_transform(Z_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("UMAP of Autoencoder Latent Space (64-dim → 2-dim)",
                 fontsize=13, fontweight="bold")

    for ax, labels, title in [
        (axes[0], y_all,   "True Labels"),
        (axes[1], y_pred,  "Predicted Labels"),
    ]:
        for i, (cls, color) in enumerate(zip(classes, PALETTE)):
            mask = labels == i
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                       c=color, label=cls, alpha=0.8, s=30, edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(title="Tumor type", fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "latent_umap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_summary_card(results: dict, classes, save_dir):
    """Single-figure portfolio summary card."""
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("TCGA Pan-Cancer Deep Learning Classifier — Test Set Results",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── Accuracy bar ──────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    labels = list(results.keys())
    accs   = [results[k]["acc"] for k in labels]
    bars   = ax0.bar(["Raw\n(2000-dim)", "Latent\n(64-dim AE)"],
                     accs, color=["#0072B2", "#E69F00"], width=0.5)
    ax0.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax0.set_ylim(0, 1.08)
    ax0.set_ylabel("Test Accuracy")
    ax0.set_title("Accuracy Comparison", fontweight="bold")
    ax0.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax0)

    # ── Per-class F1 — raw ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    f1_raw = results["raw"]["f1"]
    ax1.barh(classes, f1_raw, color="#0072B2", alpha=0.85)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("F1 Score")
    ax1.set_title("Per-class F1 — Raw MLP", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax1)

    # ── Per-class F1 — latent ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    f1_latent = results["latent"]["f1"]
    ax2.barh(classes, f1_latent, color="#E69F00", alpha=0.85)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("F1 Score")
    ax2.set_title("Per-class F1 — Latent MLP", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    sns.despine(ax=ax2)

    plt.tight_layout()
    path = os.path.join(save_dir, "summary_card.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    classes = list(le.classes_)

    results = {}

    for use_latent, label, hidden in [
        (False, "raw",    [512, 128]),
        (True,  "latent", [128,  64]),
    ]:
        print(f"\n── Evaluating MLP ({label}) on test set ──")
        X_test, y_test = load_test(use_latent)
        model  = load_model(label, X_test.shape[1], hidden)
        y_pred = predict(model, X_test)
        y_true = y_test.numpy()

        acc = accuracy_score(y_true, y_pred)
        print(f"  Test accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=classes))

        # Per-class F1 from report
        report = classification_report(y_true, y_pred, target_names=classes,
                                       output_dict=True)
        f1_per_class = [report[c]["f1-score"] for c in classes]

        results[label] = {"acc": acc, "f1": f1_per_class,
                          "y_true": y_true, "y_pred": y_pred}

        plot_confusion(
            y_true, y_pred, classes,
            title=f"Confusion Matrix — MLP {label.capitalize()} Features (Test Set)",
            path=os.path.join(FIG_DIR, f"confusion_{label}.png")
        )

    # UMAP of latent space with predicted labels
    print("\n── Latent space UMAP ──")
    Z_all = np.vstack([
        np.load(os.path.join(PROC_DIR, "Z_train.npy")),
        np.load(os.path.join(PROC_DIR, "Z_val.npy")),
        np.load(os.path.join(PROC_DIR, "Z_test.npy")),
    ])
    y_all = np.concatenate([
        np.load(os.path.join(PROC_DIR, "y_train.npy")),
        np.load(os.path.join(PROC_DIR, "y_val.npy")),
        np.load(os.path.join(PROC_DIR, "y_test.npy")),
    ])
    # Use test predictions mapped back into full array positions
    y_pred_all = y_all.copy()
    test_start = len(np.load(os.path.join(PROC_DIR, "y_train.npy"))) + \
                 len(np.load(os.path.join(PROC_DIR, "y_val.npy")))
    y_pred_all[test_start:] = results["latent"]["y_pred"]
    plot_latent_umap(Z_all, y_all, y_pred_all, classes, FIG_DIR)

    print("\n── Summary Card ──")
    plot_summary_card(results, classes, FIG_DIR)

    print("\n✅ Evaluation complete. All figures saved to results/figures/")


if __name__ == "__main__":
    main()