"""
02_preprocess.py
Loads raw TCGA RNA-seq data, applies log-normalization, filters low-variance
genes, and saves train/val/test splits as numpy arrays ready for PyTorch.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "TCGA-PANCAN-HiSeq-801x20531")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DATA_CSV   = os.path.join(RAW_DIR, "data.csv")
LABELS_CSV = os.path.join(RAW_DIR, "labels.csv")

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOP_GENES       = 2000   # keep top N genes by variance after log-transform
VAL_SIZE        = 0.10
TEST_SIZE       = 0.10
RANDOM_STATE    = 42


def load_raw(data_path: str, labels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0).squeeze()
    print(f"Raw X shape : {X.shape}")
    print(f"Raw y shape : {y.shape}")
    print(f"Class counts:\n{y.value_counts()}")
    return X, y


def preprocess(X: pd.DataFrame, y: pd.Series):
    # 1. Log1p normalization (standard for RNA-seq count data)
    X_log = np.log1p(X.values.astype(np.float32))
    print(f"\nAfter log1p  — min: {X_log.min():.3f}, max: {X_log.max():.3f}")

    # 2. Filter to top N most variable genes
    gene_vars = X_log.var(axis=0)
    top_idx   = np.argsort(gene_vars)[-TOP_GENES:]
    X_filtered = X_log[:, top_idx]
    print(f"After gene filter — shape: {X_filtered.shape}")

    # 3. Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)
    print(f"Classes: {list(le.classes_)}")

    return X_filtered, y_enc, le


def split_and_scale(X: np.ndarray, y: np.ndarray):
    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Val / test split
    rel_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Scale using train statistics only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\nSplit sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save(X_train, X_val, X_test, y_train, y_val, y_test, scaler, le) -> None:
    os.makedirs(PROC_DIR, exist_ok=True)
    np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROC_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)
    with open(os.path.join(PROC_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"\nAll arrays + artifacts saved to {PROC_DIR}")


def main() -> None:
    X, y          = load_raw(DATA_CSV, LABELS_CSV)
    X_proc, y_enc, le = preprocess(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X_proc, y_enc)
    save(X_train, X_val, X_test, y_train, y_val, y_test, scaler, le)


if __name__ == "__main__":
    main()