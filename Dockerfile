# Dockerfile
# Reproducible environment for the TCGA pan-cancer deep learning classifier.
# Build:  docker build -t tcga-classifier .
# Run:    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results tcga-classifier

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/       ./src/
COPY Snakefile  .

# ── Create expected directories ───────────────────────────────────────────────
RUN mkdir -p data results/figures results/models

# ── Default command: run full pipeline via Snakemake ─────────────────────────
CMD ["snakemake", "--cores", "1"]