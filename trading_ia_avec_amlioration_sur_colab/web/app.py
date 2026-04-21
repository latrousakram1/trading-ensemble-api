"""
web/app.py — API FastAPI pour Trading IA Multi-Modèles
Endpoints : /api/dashboard  /api/predict/{asset}  /api/metrics  /healthz
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advanced_models import MODEL_REGISTRY, build_model
from src.data import download_binance_ohlcv, load_market_data
from src.features import FEATURE_COLS, add_features
from src.targets import build_target
from src.train_utils import apply_scaler, fit_scaler
from src.utils import load_config, setup_logging

# ─────────────────────────────────────────────────────────────────────────────
logger = setup_logging("api")
cfg    = load_config(ROOT / "config.yaml")
app    = FastAPI(title="Trading IA API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Static dashboard ──────────────────────────────────────────────────────────
WEB_DIR = Path(__file__).parent
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# In-memory model cache
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Dict[str, torch.nn.Module] = {}
_scaler_cache: Dict[str, dict] = {}
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

def _load_models() -> None:
    models_dir = ROOT / "models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in MODEL_REGISTRY:
        for suffix in ("_tuned.pt", "_final.pt"):
            ckpt = models_dir / f"{name}{suffix}"
            if ckpt.exists():
                try:
                    model = build_model(name, cfg)
                    state = torch.load(ckpt, map_location=device)
                    model.load_state_dict(state)
                    model.eval().to(device)
                    _model_cache[name] = model
                    logger.info(f"Loaded {name} from {ckpt.name}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load {name}: {e}")

def _get_live_sequence(asset: str) -> Optional[np.ndarray]:
    """Download recent bars and build feature matrix for inference."""
    try:
        seq_len = cfg["model"]["seq_len"]
        bars_needed = seq_len + 60  # buffer
        df = download_binance_ohlcv(
            symbols=[asset],
            interval=cfg["market"]["interval"],
            lookback_days=max(7, bars_needed // 24 + 2),
        )
        if df is None or df.empty:
            return None
        df = add_features(df)
        df = df.dropna(subset=FEATURE_COLS).tail(seq_len)
        if len(df) < seq_len:
            return None
        return df[FEATURE_COLS].values.astype(np.float32)
    except Exception as e:
        logger.error(f"Live data error for {asset}: {e}")
        return None

def _ensemble_predict(seq: np.ndarray) -> dict:
    """Run all loaded models and return ensemble probabilities."""
    if not _model_cache:
        return {"error": "no models loaded"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Z-score normalise
    mu  = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    seq_norm = (seq - mu) / std

    x = torch.tensor(seq_norm[np.newaxis], dtype=torch.float32).to(device)
    all_probs = []
    for name, model in _model_cache.items():
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx  = int(np.argmax(avg_probs))
    return {
        "signal":     LABEL_MAP[pred_idx],
        "confidence": float(avg_probs[pred_idx]),
        "probabilities": {
            "SELL": float(avg_probs[0]),
            "HOLD": float(avg_probs[1]),
            "BUY":  float(avg_probs[2]),
        },
        "models_used": list(_model_cache.keys()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Loading models…")
    _load_models()
    logger.info(f"API ready — {len(_model_cache)} model(s) loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def health():
    return {"status": "ok", "models_loaded": len(_model_cache),
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def serve_dashboard():
    idx = WEB_DIR / "dashboard.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"msg": "Dashboard not found. Serve web/dashboard.html separately."})


@app.get("/api/predict/{asset}")
def predict(asset: str):
    """Real-time prediction for a single asset."""
    asset = asset.upper()
    allowed = cfg["market"]["symbols"]
    if asset not in allowed:
        raise HTTPException(400, f"Asset {asset} not in {allowed}")
    seq = _get_live_sequence(asset)
    if seq is None:
        raise HTTPException(503, "Could not fetch live data")
    result = _ensemble_predict(seq)
    result["asset"]     = asset
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/api/predict")
def predict_all():
    """Real-time predictions for all configured assets."""
    symbols = cfg["market"]["symbols"]
    results = {}
    for asset in symbols:
        seq = _get_live_sequence(asset)
        if seq is None:
            results[asset] = {"error": "data unavailable"}
            continue
        res = _ensemble_predict(seq)
        res["timestamp"] = datetime.now(timezone.utc).isoformat()
        results[asset] = res
    return results


@app.get("/api/metrics")
def get_metrics():
    """Return latest saved metrics from artifacts/."""
    artifacts = ROOT / "artifacts"
    metrics: dict = {}
    for jf in sorted(artifacts.glob("*_metrics*.json")):
        try:
            with open(jf) as f:
                metrics[jf.stem] = json.load(f)
        except Exception:
            pass
    return metrics or {"info": "No artifact metrics found. Run NB02 first."}


class SignalEntry(BaseModel):
    asset: str
    signal: str
    confidence: float
    timestamp: str


@app.get("/api/dashboard")
def dashboard_data():
    """Aggregate payload consumed by dashboard.html."""
    symbols = cfg["market"]["symbols"]
    signals = {}
    for asset in symbols:
        seq = _get_live_sequence(asset)
        if seq is not None:
            res = _ensemble_predict(seq)
            signals[asset] = res
        else:
            signals[asset] = {
                "signal": "HOLD",
                "confidence": 0.0,
                "probabilities": {"SELL": 0.33, "HOLD": 0.34, "BUY": 0.33},
                "models_used": [],
            }

    metrics_data  = get_metrics()
    models_status = {
        name: ("loaded" if name in _model_cache else "not_loaded")
        for name in MODEL_REGISTRY
    }

    return {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "signals":       signals,
        "metrics":       metrics_data,
        "models_status": models_status,
        "config": {
            "symbols":   symbols,
            "interval":  cfg["market"]["interval"],
            "horizon":   cfg["target"]["horizon"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
