"""
03_explore.py
PCA and UMAP visualizations of the preprocessed gene expression data.
Saves figures to results/figures/.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.decomposition import PCA
import umap

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

# ── Palette (colorblind-friendly) ─────────────────────────────────────────────
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def load_data():
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(PROC_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"))

    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    # Combine all splits for visualization
    X = np.vstack([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])
    print(f"Loaded — X: {X.shape}, y: {y.shape}")
    return X, y, le


def plot_pca(X: np.ndarray, y: np.ndarray, le, save_dir: str) -> None:
    print("Running PCA...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    # ── Scree plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA of TCGA Pan-Cancer Gene Expression", fontsize=14, fontweight="bold")

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].plot(range(1, 51), cumvar, marker="o", markersize=3,
                 color="#0072B2", linewidth=1.5)
    axes[0].axhline(90, color="grey", linestyle="--", linewidth=0.8, label="90% threshold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Cumulative Explained Variance (%)")
    axes[0].set_title("Scree Plot")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── PC1 vs PC2 scatter ────────────────────────────────────────────────────
    for i, (cls, color) in enumerate(zip(le.classes_, PALETTE)):
        mask = y == i
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, label=cls, alpha=0.75, s=25, edgecolors="none")

    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title("PC1 vs PC2 by Tumor Type")
    axes[1].legend(title="Tumor type", framealpha=0.9)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "pca.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")

    # Return PCA-reduced data for UMAP input (faster than raw 2000-dim)
    return X_pca


def plot_umap(X_pca: np.ndarray, y: np.ndarray, le, save_dir: str) -> None:
    print("Running UMAP (this takes ~30s)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=42, verbose=False)
    X_umap = reducer.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (cls, color) in enumerate(zip(le.classes_, PALETTE)):
        mask = y == i
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=color, label=cls, alpha=0.80, s=30, edgecolors="none")

    ax.set_title("UMAP of TCGA Pan-Cancer Gene Expression\n(top 2000 variable genes, log1p-normalized)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Tumor type", framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "umap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_class_distribution(y: np.ndarray, le, save_dir: str) -> None:
    counts = {cls: (y == i).sum() for i, cls in enumerate(le.classes_)}

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.keys(), counts.values(), color=PALETTE, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Sample Count per Tumor Type", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tumor Type")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    X, y, le = load_data()
    plot_class_distribution(y, le, FIG_DIR)
    X_pca = plot_pca(X, y, le, FIG_DIR)
    plot_umap(X_pca, y, le, FIG_DIR)
    print("\nAll figures saved.")


if __name__ == "__main__":
    main()