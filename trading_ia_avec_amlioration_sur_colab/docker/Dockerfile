# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Trading IA Multi-Modèles — Dockerfile                                   ║
# ║  Build : docker build -t trading-ia .                                    ║
# ║  Run   : docker compose up  (voir docker-compose.yml)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Base image ────────────────────────────────────────────────────────────────
# CPU-only : python:3.11-slim
# GPU      : nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04  (décommentez)
FROM python:3.11-slim
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Trading IA <vous@exemple.com>"
LABEL description="Trading IA Multi-Modèles — PatchTST · LSTM-Attention · CNN-Transformer · TFT-Lite"

# ── Système ───────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Répertoire de travail ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances Python ────────────────────────────────────────────────────────
# Copier requirements en premier pour profiter du cache Docker layer
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY src/          ./src/
COPY web/          ./web/
COPY config.yaml   .

# ── Dossiers de données / modèles (volumes montés en prod) ────────────────────
RUN mkdir -p data/raw data/processed data/sentiment_raw \
             models artifacts logs

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8000 : API FastAPI + Dashboard
# 5000 : MLFlow UI
EXPOSE 8000 5000

# ── Variables d'environnement par défaut ──────────────────────────────────────
ENV PORT=8000 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    PYTHONPATH=/app

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# ── Commande de démarrage ─────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
