"""
sentiment.py  v2.0  Score QDM temporel
========================================
Score de sentiment qui varie dans le temps grace a :
  1. Fear & Greed Index (alternative.me) -- quotidien
  2. FinBERT (ProsusAI/finbert) -- NLP sur headlines
  3. Price momentum -- signal technique normalise

Le score resultant (QDM) est dans [-1, +1] et change heure par heure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import requests

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# -- Fear & Greed Index --------------------------------------------------------
def download_fear_greed(days: int = 200) -> pd.DataFrame:
    """Telecharge le Fear & Greed Index depuis alternative.me."""
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["timestamp"] = df["timestamp"].dt.normalize()
        df["fg_value"]  = pd.to_numeric(df["value"], errors="coerce")
        df["fg_norm"]   = (df["fg_value"] - 50) / 50.0
        return (df[["timestamp", "fg_value", "fg_norm", "value_classification"]]
                .sort_values("timestamp").reset_index(drop=True))
    except Exception:
        dates = pd.date_range(
            end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC"
        )
        return pd.DataFrame({
            "timestamp": dates,
            "fg_value": 50,
            "fg_norm": 0.0,
            "value_classification": "Neutral",
        })


# -- FinBERT -------------------------------------------------------------------
def finbert_asset_scores(headlines_by_asset: dict[str, list[str]]) -> dict[str, float]:
    """
    Calcule le score FinBERT pour chaque actif.
    Retourne {asset: score} avec score dans [-1, +1].
    """
    try:
        from transformers import pipeline
        import torch
        device_id = 0 if torch.cuda.is_available() else -1
        nlp = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=device_id,
            top_k=None,
        )
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        results = {}
        for asset, texts in headlines_by_asset.items():
            if not texts:
                results[asset] = 0.0
                continue
            scores = []
            for text in texts[:20]:
                try:
                    out = nlp(text[:512], truncation=True)[0]
                    s = sum(label_map.get(r["label"].lower(), 0.0) * r["score"] for r in out)
                    scores.append(s)
                except Exception:
                    scores.append(0.0)
            results[asset] = float(np.mean(scores)) if scores else 0.0
        return results
    except Exception:
        return {a: 0.0 for a in headlines_by_asset}


# -- Score QDM temporel --------------------------------------------------------
def build_qdm_sentiment_feature(
    market_df: pd.DataFrame,
    fg_df: pd.DataFrame | None = None,
    finbert_scores: dict[str, float] | None = None,
    alpha: float = 0.50,
    beta: float = 0.30,
    gamma: float = 0.20,
) -> pd.DataFrame:
    """
    Construit le score QDM temporel.

    QDM(t) = alpha*FG_norm(t) + beta*FinBERT(asset) + gamma*PriceMomentum(t)

    Parametres
    ----------
    market_df      : DataFrame OHLCV avec timestamp, asset, close
    fg_df          : Fear & Greed (download_fear_greed)
    finbert_scores : dict {asset: score}
    alpha          : poids Fear & Greed     (defaut 0.50)
    beta           : poids FinBERT          (defaut 0.30)
    gamma          : poids Price Momentum   (defaut 0.20)
    """
    out = market_df.copy().sort_values(["asset", "timestamp"])

    # 1. Fear & Greed par date
    if fg_df is None:
        fg_df = download_fear_greed(days=200)
    fg_lookup = (fg_df.copy()
                 .assign(date=lambda d: d["timestamp"].dt.date.astype(str))
                 .groupby("date")["fg_norm"].mean()
                 .to_dict())
    out["date"]    = out["timestamp"].dt.date.astype(str)
    out["fg_norm"] = out["date"].map(fg_lookup).fillna(0.0)

    # 2. FinBERT par actif
    if finbert_scores is None:
        finbert_scores = {}
    out["finbert_norm"] = out["asset"].map(finbert_scores).fillna(0.0)

    # 3. Price momentum normalise
    out["ret_24h"]   = out.groupby("asset")["close"].pct_change(24)
    out["price_mom"] = out.groupby("asset")["ret_24h"].transform(
        lambda s: (s - s.rolling(168, min_periods=24).mean())
                  / (s.rolling(168, min_periods=24).std() + 1e-8)
    )
    out["price_mom"] = out["price_mom"].clip(-3, 3) / 3.0

    # 4. Score QDM combine
    out["sentiment_score"] = (
        alpha * out["fg_norm"]
        + beta  * out["finbert_norm"]
        + gamma * out["price_mom"].fillna(0.0)
    ).clip(-1.0, 1.0)

    out = out.drop(
        columns=["date", "fg_norm", "finbert_norm", "price_mom", "ret_24h"],
        errors="ignore",
    )
    return out


# -- Proxy simple (fallback sans API) ------------------------------------------
def build_proxy_sentiment_feature(
    market_df: pd.DataFrame,
    sentiment_labeled_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fallback sans API externe. Utilise uniquement le price momentum.
    Compatible avec prepare_datasets.py.
    """
    if sentiment_labeled_df is not None and len(sentiment_labeled_df) > 0:
        fb_scores = {}
        if "sentiment_score" in sentiment_labeled_df.columns:
            fb_scores = (sentiment_labeled_df
                         .groupby("asset")["sentiment_score"]
                         .mean().to_dict())
        return build_qdm_sentiment_feature(
            market_df, fg_df=None, finbert_scores=fb_scores,
            alpha=0.0, beta=1.0, gamma=0.0,
        )
    return build_qdm_sentiment_feature(
        market_df, fg_df=None, finbert_scores=None,
        alpha=0.0, beta=0.0, gamma=1.0,
    )


# -- HuggingFace datasets (compatibilite ancienne version) ---------------------
HF_SOURCES = [
    {"target_name": "zeroshot/twitter-financial-news-sentiment",
     "candidates": [("zeroshot/twitter-financial-news-sentiment", None)]},
    {"target_name": "takala/financial_phrasebank",
     "candidates": [("takala/financial_phrasebank", "sentences_allagree")]},
]


def _canonical_label(x) -> str:
    s = str(x).strip().lower()
    mapping = {
        "0": "negative", "1": "neutral", "2": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive",
        "bearish": "negative", "bullish": "positive",
    }
    return mapping.get(s, "neutral")


def _label_to_score(x) -> float:
    return {"negative": -1.0, "neutral": 0.0, "positive": 1.0}.get(_canonical_label(x), 0.0)


def _extract_asset(text: str) -> str:
    text = str(text).upper()
    if any(t in text for t in ["BTC", "BITCOIN"]):  return "BTCUSDT"
    if any(t in text for t in ["ETH", "ETHEREUM"]): return "ETHUSDT"
    if "BNB" in text:                                return "BNBUSDT"
    if any(t in text for t in ["SOL", "SOLANA"]):   return "SOLUSDT"
    return "ALL"


def download_sentiment_datasets(output_dir) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    if not HF_AVAILABLE:
        return {"error": "datasets library not available"}
    for source in HF_SOURCES:
        tname = source["target_name"]
        try:
            for dname, subset in source["candidates"]:
                try:
                    ds = (load_dataset(dname, subset) if subset
                          else load_dataset(dname))
                    split = "train" if "train" in ds else list(ds.keys())[0]
                    df = ds[split].to_pandas()
                    out = output_dir / (tname.replace("/", "__") + ".csv")
                    df.to_csv(out, index=False)
                    saved[tname] = str(out)
                    break
                except Exception:
                    continue
        except Exception as e:
            saved[tname] = f"ERROR: {e}"
    return saved


def merge_downloaded_sentiment_data(output_dir, output_csv) -> pd.DataFrame:
    output_dir = Path(output_dir)
    frames = []
    for csv_path in output_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            continue
        df["text"] = df["text"].astype(str)
        df["label_text"] = df.get("label", pd.Series("neutral", index=df.index)).map(
            _canonical_label
        )
        df["sentiment_score"] = df["label_text"].map(_label_to_score)
        df["asset"] = df["text"].map(_extract_asset)
        frames.append(df[["text", "sentiment_score", "asset"]])
    if not frames:
        return pd.DataFrame(columns=["text", "sentiment_score", "asset"])
    merged = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["text"])
              .reset_index(drop=True))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged
