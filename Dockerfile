# ── Stage 1 : builder (installe les dépendances) ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# ── Stage 2 : image finale ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copier les packages installés depuis le builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copier le code source
COPY src/         ./src/
COPY config.yaml  .

# Dossiers attendus par le code
RUN mkdir -p models artifacts data/raw data/processed

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

EXPOSE 8000

# Healthcheck Docker natif
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Lancer avec Uvicorn (workers selon CPU)
CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
