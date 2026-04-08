"""
01_download.py
Downloads the TCGA pan-cancer gene expression dataset from UCI ML Repository.
Dataset: 801 samples x 20,531 genes, 5 tumor types (BRCA, KIRC, COAD, LUAD, PRAD)
Source: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq
"""

import os
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
ZIP_PATH = os.path.join(DATA_DIR, "tcga_raw.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "raw")


def download_file(url: str, dest: str) -> None:
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        desc="Downloading", total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract_all(zip_path: str, extract_dir: str) -> None:
    """Unzip, then untar the inner tar.gz."""
    import tarfile

    os.makedirs(extract_dir, exist_ok=True)

    # Step 1: unzip outer zip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
        print(f"Zip contents: {z.namelist()}")

    # Step 2: find and extract the tar.gz
    for fname in os.listdir(extract_dir):
        if fname.endswith(".tar.gz"):
            tar_path = os.path.join(extract_dir, fname)
            print(f"Extracting tar.gz: {fname}")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            print("tar.gz extracted.")
            break


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print("Zip already downloaded, skipping.")
    else:
        print(f"Downloading dataset to {ZIP_PATH} ...")
        download_file(ZIP_URL, ZIP_PATH)

    print(f"Extracting to {EXTRACT_DIR} ...")
    extract_all(ZIP_PATH, EXTRACT_DIR)

    # Walk all extracted files to find csvs
    print("\nAll extracted files:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    main()
