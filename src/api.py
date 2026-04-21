"""
api.py  —  Trading Ensemble API  v3.0.0
========================================
Endpoints :
  GET  /                         Statut global
  GET  /health                   Santé détaillée
  GET  /models                   Modèles disponibles
  GET  /validation               Résultats walk-forward
  GET  /dashboard                Dashboard HTML Chart.js
  GET  /dashboard/data           Données JSON du dashboard
  POST /predict/{model_name}     Prédiction individuelle
  POST /predict/ensemble         Prédiction ensemble pondéré
  GET  /predict/ensemble/scan    Scan multi-actifs

Corrections :
  - Import try/except sur realtime (compatible Colab + module)
  - sentiment_score ajouté au frame temps réel
  - endpoint /validation exposant les résultats walk-forward
  - _load_backtest_series depuis artifact_dir (configurable)
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from src.advanced_models import build_model
    from src.data import load_market_data
    from src.features import add_features
    from src.metrics import compute_reliable_metrics
    from src.realtime import BinanceRealtimeBuffer
    from src.targets import build_target
    from src.utils import load_config, load_json
    from src.train_utils import get_feature_cols
except ImportError:
    from advanced_models import build_model
    from data import load_market_data
    from features import add_features
    from metrics import compute_reliable_metrics
    from realtime import BinanceRealtimeBuffer
    from targets import build_target
    from utils import load_config, load_json
    from train_utils import get_feature_cols

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("trading-api")

ENSEMBLE_MEMBERS = ["patchtst", "lstm_attention", "cnn_transformer"]

_state: dict[str, Any] = {
    "cfg": None, "df": None, "loaded": {}, "weights": {},
    "start_time": None, "request_count": 0,
    "market_source": "static", "realtime": None,
}


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _load_checkpoint(model_name: str, cfg: dict) -> dict:
    if model_name in _state["loaded"]:
        return _state["loaded"][model_name]

    model_dir = Path(cfg["paths"]["model_dir"])
    ckpt_path = None
    for suffix in ("_tuned.pt", "_final.pt"):
        candidate = model_dir / f"{model_name}{suffix}"
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Aucun checkpoint pour '{model_name}' dans {model_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_cols = ckpt["feature_cols"]
    scaler = {
        "mean": np.array(ckpt["scaler_mean"], dtype="float32"),
        "std":  np.array(ckpt["scaler_std"],  dtype="float32"),
    }
    scaler["std"][scaler["std"] == 0] = 1.0

    saved_name = ckpt.get("model_name", model_name).replace("_tuned", "").replace("_final", "")
    model = build_model(saved_name, len(feature_cols), cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    entry = {"model": model, "feature_cols": feature_cols, "scaler": scaler,
             "n_params": ckpt.get("n_params", 0), "ckpt_path": str(ckpt_path)}
    _state["loaded"][model_name] = entry
    log.info("Modele charge: %s [%s]", model_name, ckpt_path.name)
    return entry


def _get_market_frame(asset: str | None = None) -> pd.DataFrame:
    realtime = _state.get("realtime")
    if realtime is not None and realtime.is_ready():
        frame = realtime.snapshot(asset)
        if not frame.empty:
            # ── Ajout sentiment_score manquant pour le buffer temps réel ──
            if "sentiment_score" not in frame.columns:
                frame = frame.copy()
                frame["sentiment_score"] = 0.0
            _state["market_source"] = "binance_websocket"
            return frame

    df = _state.get("df")
    if df is None:
        return pd.DataFrame()
    _state["market_source"] = "static"
    if asset is not None:
        return df[df["asset"] == asset.upper()].copy()
    return df.copy()


def _build_sequence(asset: str, feature_cols: list[str], scaler: dict,
                     cfg: dict, sentiment_score: float = 0.0):
    df = _get_market_frame(asset)
    if df.empty:
        raise ValueError(f"Actif '{asset}' introuvable.")

    if "sentiment_score" not in df.columns:
        df = df.copy()
        df["sentiment_score"] = float(sentiment_score)
    else:
        df = df.copy()
        df["sentiment_score"] = df["sentiment_score"].fillna(float(sentiment_score))

    df = add_features(df)
    df = build_target(df, horizon=cfg["target"]["horizon"],
                      buy_threshold=cfg["target"]["buy_threshold"],
                      sell_threshold=cfg["target"]["sell_threshold"])

    seq_len = cfg["model"]["seq_len"]
    g = df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(g) < seq_len:
        raise ValueError(f"Données insuffisantes pour '{asset}': {len(g)} < {seq_len}")

    window = g[feature_cols].astype("float32").values[-seq_len:]
    std = scaler["std"].copy(); std[std == 0] = 1.0
    if len(scaler["mean"]) == len(feature_cols):
        window = (window - scaler["mean"]) / std

    return np.expand_dims(window, 0).astype("float32"), g.iloc[-1]


def _infer(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(
            model(torch.tensor(X, dtype=torch.float32)), dim=1
        ).cpu().numpy()[0]


def _signal(probs: np.ndarray, threshold: float, allow_short: bool,
             hold_threshold: float = 0.0) -> tuple[str, float]:
    p_sell, p_hold, p_buy = probs
    if p_hold >= hold_threshold > 0:
        return "HOLD", 0.0
    if p_buy >= threshold and p_buy > p_sell:
        return "BUY", 1.0
    if allow_short and p_sell >= threshold and p_sell > p_buy:
        return "SELL", -1.0
    return "HOLD", 0.0


def _load_backtest_series() -> tuple[list, list, dict]:
    """Charge equity curve et signaux depuis les artifacts."""
    cfg = _state["cfg"]
    artifact_dir = Path(cfg["paths"]["artifact_dir"])

    equity_curve = []
    backtest_prices = []
    reliable = {}

    equity_path = artifact_dir / "portfolio_equity.csv"
    if equity_path.exists():
        eq = pd.read_csv(equity_path, parse_dates=["timestamp"])
        eq["equity"] = (1 + eq["strategy_ret"].fillna(0)).cumprod()
        equity_curve = [
            {"timestamp": str(row.timestamp), "equity": round(float(row.equity), 6)}
            for row in eq.tail(500).itertuples()
        ]

    predictions_path = artifact_dir / "backtest_predictions.csv"
    if predictions_path.exists():
        preds = pd.read_csv(predictions_path, parse_dates=["timestamp"])
        bt_asset = preds[preds["asset"] == cfg.get("api", {}).get("dashboard_default_asset", "BTCUSDT")]
        backtest_prices = [
            {"timestamp": str(row.timestamp), "close": float(row.close),
             "signal": "BUY" if row.signal > 0 else ("SELL" if row.signal < 0 else "HOLD")}
            for row in bt_asset.tail(300).itertuples()
        ]

    reliable_path = artifact_dir / "backtest_reliable_metrics.json"
    if reliable_path.exists():
        reliable = load_json(reliable_path)

    return backtest_prices, equity_curve, reliable


# ─── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["start_time"] = time.time()
    cfg = load_config()
    _state["cfg"] = cfg

    # Données statiques
    preferred   = Path(cfg["sentiment"]["aligned_output_csv"])
    market_path = preferred if preferred.exists() else Path(cfg["market"]["output_csv"])
    if market_path.exists():
        raw = load_market_data(market_path)
        if "sentiment_score" not in raw.columns:
            raw["sentiment_score"] = 0.0
        _state["df"] = raw
        log.info("Données statiques chargées : %d lignes", len(raw))

    # Buffer temps réel (si configuré)
    rt_cfg = cfg.get("realtime", {})
    if rt_cfg.get("enabled", False):
        buf = BinanceRealtimeBuffer(
            symbols=cfg["market"]["symbols"],
            interval=cfg["market"]["interval"],
            buffer_hours=cfg["model"]["seq_len"] + 10,
            snapshot_csv=rt_cfg.get("snapshot_csv"),
        )
        await buf.initialize()
        buf.start()
        _state["realtime"] = buf
        log.info("Buffer temps réel Binance démarré")

    # Pré-chargement des modèles + calcul des poids
    f1_map = {}
    for name in ENSEMBLE_MEMBERS:
        try:
            _load_checkpoint(name, cfg)
            metrics_path = Path(cfg["paths"]["artifact_dir"]) / f"{name}_metrics.json"
            if metrics_path.exists():
                m = load_json(metrics_path)
                f1_map[name] = m.get("test_f1_macro", 1.0)
            else:
                f1_map[name] = 1.0
        except FileNotFoundError as e:
            log.warning("%s", e)

    if f1_map:
        total = sum(f1_map.values())
        _state["weights"] = {n: round(f / total, 4) for n, f in f1_map.items()}
        log.info("Poids ensemble : %s", _state["weights"])

    log.info("API prête — %d modèle(s) chargé(s)", len(_state["loaded"]))
    yield

    if _state.get("realtime"):
        await _state["realtime"].stop()


# ─── Application ───────────────────────────────────────────────────────────────
cfg_init = load_config()
app = FastAPI(
    title="Trading Ensemble API",
    description="Ensemble PatchTST · LSTM Attention · CNN Transformer",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg_init.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    _state["request_count"] += 1
    return await call_next(request)


# ─── Schémas Pydantic ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    asset:           str   = Field(...,  description="Ex: BTCUSDT")
    threshold:       float = Field(0.45, ge=0.0, le=1.0)
    allow_short:     bool  = Field(True)
    sentiment_score: float = Field(0.0)


class ModelPrediction(BaseModel):
    model: str; signal: str; signal_value: float; confidence: float
    prob_sell: float; prob_hold: float; prob_buy: float


class PredictResponse(BaseModel):
    asset: str; timestamp: str; close: float; model: str; signal: str
    signal_value: float; confidence: float; prob_sell: float; prob_hold: float
    prob_buy: float; latency_ms: float; data_source: str


class EnsembleResponse(BaseModel):
    asset: str; timestamp: str; close: float; signal: str; signal_value: float
    confidence: float; prob_sell: float; prob_hold: float; prob_buy: float
    weights: dict[str, float]; model_contributions: dict[str, ModelPrediction]
    latency_ms: float; data_source: str


class AssetScan(BaseModel):
    asset: str; signal: str; confidence: float
    prob_sell: float; prob_hold: float; prob_buy: float


class ScanResponse(BaseModel):
    scanned_at: str; threshold: float; results: list[AssetScan]
    latency_ms: float; data_source: str


# ─── Endpoints info ────────────────────────────────────────────────────────────
@app.get("/", tags=["info"])
def root():
    market_df = _get_market_frame()
    assets = sorted(market_df["asset"].unique().tolist()) if not market_df.empty else []
    return {
        "name": "Trading Ensemble API", "version": "3.0.0",
        "uptime_s": round(time.time() - (_state["start_time"] or time.time()), 1),
        "models_loaded": list(_state["loaded"].keys()),
        "ensemble_weights": _state["weights"],
        "assets": assets,
        "requests_served": _state["request_count"],
        "market_source": _state["market_source"],
    }


@app.get("/health", tags=["info"])
def health():
    cfg = _state["cfg"]
    available = []
    for name in ENSEMBLE_MEMBERS:
        found = None
        for suffix in ("_tuned.pt", "_final.pt"):
            c = Path(cfg["paths"]["model_dir"]) / f"{name}{suffix}"
            if c.exists(): found = c.name; break
        available.append({"name": name, "file": found, "loaded": name in _state["loaded"]})

    market_df = _get_market_frame()
    realtime  = _state.get("realtime")
    rt_status = realtime.status() if realtime else {"enabled": False, "ready": False}
    return {
        "status": "ok" if not market_df.empty and _state["loaded"] else "degraded",
        "data_loaded": not market_df.empty, "data_rows": len(market_df),
        "models": available, "ensemble_weights": _state["weights"],
        "uptime_s": round(time.time() - (_state["start_time"] or time.time()), 1),
        "requests_served": _state["request_count"],
        "market_source": _state["market_source"], "realtime": rt_status,
    }


@app.get("/models", tags=["info"])
def list_models():
    cfg = _state["cfg"]
    result = []
    for name in ENSEMBLE_MEMBERS:
        file_name = None
        for suffix in ("_tuned.pt", "_final.pt"):
            c = Path(cfg["paths"]["model_dir"]) / f"{name}{suffix}"
            if c.exists(): file_name = c.name; break
        if file_name is None: continue
        entry = _state["loaded"].get(name, {})
        result.append({"name": name, "file": file_name,
                        "n_params": entry.get("n_params", "?"),
                        "loaded": name in _state["loaded"],
                        "weight_in_ensemble": _state["weights"].get(name)})
    return {"models": result, "ensemble_members": list(_state["loaded"].keys())}


@app.get("/validation", tags=["info"])
def get_validation():
    """Résultats du walk-forward mensuel (robustesse du modèle)."""
    cfg = _state["cfg"]
    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    results = {}
    for name in ENSEMBLE_MEMBERS:
        p = artifact_dir / f"walk_forward_{name}.json"
        if p.exists():
            results[name] = load_json(p)
    if not results:
        raise HTTPException(
            404,
            "Walk-forward non encore lancé. "
            "Exécutez : python src/walk_forward.py --model lstm_attention"
        )
    return results


# ─── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/test", tags=["test"])
def test_endpoint():
    return {"message": "Test endpoint works!", "timestamp": "2026-04-18"}


@app.get("/dashboard", tags=["dashboard"])
def dashboard():
    """Sert le dashboard HTML avec Chart.js"""
    try:
        # Chemin vers le fichier dashboard.html
        dashboard_path = Path(__file__).resolve().parents[1] / "web" / "dashboard.html"
        
        if not dashboard_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard introuvable")
        
        # Lire le contenu du fichier HTML
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur dashboard: {str(e)}")


@app.get("/dashboard/data", tags=["dashboard"])
def dashboard_data(
    asset: str | None = None,
    threshold: float = 0.45,
    allow_short: bool = True,
    sentiment_score: float = 0.0,
):
    cfg = _state["cfg"]
    asset = (asset or cfg.get("api", {}).get("dashboard_default_asset", "BTCUSDT")).upper()
    market_df = _get_market_frame(asset).sort_values("timestamp").tail(
        cfg["model"]["seq_len"] + 48
    )
    if market_df.empty:
        raise HTTPException(404, f"Aucune donnée disponible pour {asset}.")

    candles = [
        {"timestamp": row.timestamp.isoformat(),
         "open": float(row.open), "high": float(row.high),
         "low": float(row.low),  "close": float(row.close),
         "volume": float(row.volume)}
        for row in market_df.itertuples()
    ]
    ensemble = predict_ensemble(PredictRequest(
        asset=asset, threshold=threshold,
        allow_short=allow_short, sentiment_score=sentiment_score,
    ))
    scan = scan_all_assets(threshold=threshold, allow_short=allow_short,
                           sentiment_score=sentiment_score)
    backtest_prices, equity_curve, reliable = _load_backtest_series()
    return {
        "asset": asset, "market_source": _state["market_source"],
        "candles": candles, "latest_signal": ensemble.model_dump(),
        "scan": scan.model_dump(), "backtest_price_signals": backtest_prices,
        "equity_curve": equity_curve, "reliable_metrics": reliable,
        "weights": _state["weights"],
    }


# ─── Prédictions ───────────────────────────────────────────────────────────────
@app.post("/predict/{model_name}", response_model=PredictResponse, tags=["predict"])
def predict_model(model_name: str, req: PredictRequest):
    t0 = time.time()
    cfg = _state["cfg"]
    hold_threshold = float(cfg["backtest"].get("hold_probability_threshold", 0.0))
    try:
        art = _load_checkpoint(model_name, cfg)
        X, row = _build_sequence(req.asset, art["feature_cols"], art["scaler"],
                                  cfg, req.sentiment_score)
        probs = _infer(art["model"], X)
        signal, signal_value = _signal(probs, req.threshold, req.allow_short, hold_threshold)
        return PredictResponse(
            asset=req.asset, timestamp=str(row["timestamp"]), close=float(row["close"]),
            model=model_name, signal=signal, signal_value=signal_value,
            confidence=float(max(probs)), prob_sell=float(probs[0]),
            prob_hold=float(probs[1]), prob_buy=float(probs[2]),
            latency_ms=round((time.time() - t0) * 1000, 1),
            data_source=_state["market_source"],
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.post("/predict/ensemble", response_model=EnsembleResponse, tags=["predict"])
def predict_ensemble(req: PredictRequest):
    t0 = time.time()
    cfg = _state["cfg"]
    hold_threshold = float(cfg["backtest"].get("hold_probability_threshold", 0.0))

    contributions: dict[str, ModelPrediction] = {}
    weighted_probs: list[np.ndarray] = []
    row = None

    for name in ENSEMBLE_MEMBERS:
        try:
            art = _load_checkpoint(name, cfg)
            X, cur_row = _build_sequence(req.asset, art["feature_cols"], art["scaler"],
                                          cfg, req.sentiment_score)
            probs = _infer(art["model"], X)
            weight = _state["weights"].get(name, 1.0 / len(ENSEMBLE_MEMBERS))
            weighted_probs.append(probs * weight)
            signal, signal_value = _signal(probs, req.threshold, req.allow_short, hold_threshold)
            contributions[name] = ModelPrediction(
                model=name, signal=signal, signal_value=signal_value,
                confidence=float(max(probs)), prob_sell=float(probs[0]),
                prob_hold=float(probs[1]), prob_buy=float(probs[2]),
            )
            if row is None: row = cur_row
        except Exception as e:
            log.warning("Modele %s ignoré: %s", name, e)

    if not weighted_probs:
        raise HTTPException(503, "Aucun modèle disponible pour l'ensemble.")

    avg     = np.sum(weighted_probs, axis=0)
    total_w = sum(_state["weights"].get(n, 1.0 / len(ENSEMBLE_MEMBERS)) for n in contributions)
    avg     = avg / (total_w + 1e-12)
    signal, signal_value = _signal(avg, req.threshold, req.allow_short, hold_threshold)

    return EnsembleResponse(
        asset=req.asset, timestamp=str(row["timestamp"]) if row is not None else "",
        close=float(row["close"]) if row is not None else 0.0,
        signal=signal, signal_value=signal_value,
        confidence=float(max(avg)), prob_sell=float(avg[0]),
        prob_hold=float(avg[1]), prob_buy=float(avg[2]),
        weights=_state["weights"], model_contributions=contributions,
        latency_ms=round((time.time() - t0) * 1000, 1),
        data_source=_state["market_source"],
    )


@app.get("/predict/ensemble/scan", response_model=ScanResponse, tags=["predict"])
def scan_all_assets(threshold: float = 0.45, allow_short: bool = True,
                     sentiment_score: float = 0.0):
    t0  = time.time()
    df  = _get_market_frame()
    if df.empty:
        raise HTTPException(503, "Données non disponibles.")

    results = []
    for asset in sorted(df["asset"].unique().tolist()):
        try:
            r = predict_ensemble(PredictRequest(
                asset=asset, threshold=threshold,
                allow_short=allow_short, sentiment_score=sentiment_score,
            ))
            results.append(AssetScan(
                asset=asset, signal=r.signal, confidence=r.confidence,
                prob_sell=r.prob_sell, prob_hold=r.prob_hold, prob_buy=r.prob_buy,
            ))
        except Exception as e:
            log.warning("Scan %s: %s", asset, e)

    return ScanResponse(
        scanned_at=pd.Timestamp.utcnow().isoformat(), threshold=threshold,
        results=results, latency_ms=round((time.time() - t0) * 1000, 1),
        data_source=_state["market_source"],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Erreur non gérée sur %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
