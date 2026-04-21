"""
sentiment.py v4.0 — Score QDM Temporel
QDM(t) = 0.5 * FearGreed_norm(t)  +  0.3 * FinBERT_score(asset)  +  0.2 * PriceMomentum(t)

Fix v4.0 :
  - _to_day_utc_naive : tz_localize(None) → tz_convert(None)  [lève exception pandas 2.x sur series tz-aware]
  - Mapping FG : merge_asof remplacé par string "%Y-%m-%d" + dict.map() + ffill/bfill explicite
  - Diagnostic verbeux : prints directs pour debug Colab (pas seulement logger)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ALPHA = 0.50   # Fear & Greed
BETA  = 0.30   # FinBERT
GAMMA = 0.20   # Price momentum


# ─────────────────────────────────────────────────────────────────────────────
# Helper : normalisation des dates
# ─────────────────────────────────────────────────────────────────────────────

def _to_date_str(col: pd.Series) -> pd.Series:
    """
    Convertit n'importe quelle série temporelle en chaîne "YYYY-MM-DD" UTC.
    Gère : tz-aware, tz-naive, string ISO, int unix.

    Utilise tz_convert(None) (pas tz_localize) pour pandas >= 2.0.
    """
    s = pd.to_datetime(col, utc=True, errors="coerce")   # → toujours UTC tz-aware
    s = s.dt.tz_convert(None)                            # → tz-naïf (UTC)  ← FIX
    return s.dt.normalize().dt.strftime("%Y-%m-%d")      # → "2024-03-15"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fear & Greed
# ─────────────────────────────────────────────────────────────────────────────

def download_fear_greed(days: int = 220) -> pd.DataFrame:
    """
    Télécharge l'index Fear & Greed (alternative.me).
    Retourne un DataFrame avec : date_str (YYYY-MM-DD), fg_norm (float [-1,+1]).
    """
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()["data"]
        df = pd.DataFrame(raw)

        df["date_str"] = _to_date_str(
            pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        )
        df["fg_value"] = pd.to_numeric(df["value"], errors="coerce")
        df["fg_norm"]  = (df["fg_value"] - 50) / 50.0

        df = (
            df[["date_str", "fg_value", "fg_norm", "value_classification"]]
            .dropna(subset=["fg_norm"])
            .sort_values("date_str")
            .reset_index(drop=True)
        )
        print(f"[QDM] Fear & Greed : {len(df)} jours "
              f"({df['date_str'].iloc[0]} → {df['date_str'].iloc[-1]})")
        return df

    except Exception as e:
        logger.warning(f"API Fear & Greed indisponible ({e}) — fallback synthétique")
        end = pd.Timestamp.utcnow().normalize()
        dates = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
        date_strs = dates.strftime("%Y-%m-%d").tolist()
        fg_vals = np.clip(
            50 + 20 * np.sin(np.linspace(0, 4 * np.pi, days))
            + np.random.normal(0, 8, days), 0, 100
        ).astype(int)
        print(f"[QDM] Fear & Greed synthétique : {len(date_strs)} jours "
              f"({date_strs[0]} → {date_strs[-1]})")
        return pd.DataFrame({
            "date_str"             : date_strs,
            "fg_value"             : fg_vals,
            "fg_norm"              : (fg_vals - 50) / 50.0,
            "value_classification" : "Neutral (synthetic)",
        })


# ─────────────────────────────────────────────────────────────────────────────
# 2. FinBERT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HEADLINES: Dict[str, list] = {
    "BTCUSDT": [
        "Bitcoin surges past resistance as institutional buying accelerates",
        "BTC reaches new high amid growing ETF demand",
        "Bitcoin network hashrate drops sharply raising concerns",
        "Regulatory crackdown fears crypto market participants",
        "Bitcoin holds steady despite market uncertainty",
        "Major bank announces Bitcoin custody for institutions",
    ],
    "ETHUSDT": [
        "Ethereum upgrade deployed improving transaction throughput",
        "ETH staking yields decline as validators join network",
        "Ethereum DeFi TVL reaches record on renewed activity",
        "Smart contract vulnerability found in major protocol",
    ],
    "BNBUSDT": [
        "Binance expands despite regulatory headwinds",
        "BNB Chain launches developer incentives for ecosystem",
        "Binance faces probe increasing BNB token uncertainty",
    ],
    "SOLUSDT": [
        "Solana network experiences brief outage disrupting transactions",
        "Solana NFT marketplace volume surges to record levels",
        "SOL validator count increases improving decentralization",
    ],
}


def _score_texts(texts: list, pipe) -> float:
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    for text in texts:
        try:
            result = pipe(text[:512], truncation=True)[0]
            s = sum(label_map.get(r["label"].lower(), 0.0) * r["score"] for r in result)
            scores.append(s)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_finbert_scores(
    headlines: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
    if headlines is None:
        headlines = DEFAULT_HEADLINES
    try:
        import torch
        from transformers import pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        finbert = pipeline("text-classification", model="ProsusAI/finbert",
                           device=device_id, top_k=None)
        print(f"[QDM] FinBERT chargé sur {'GPU' if device_id == 0 else 'CPU'}")
        scores = {}
        for asset, texts in headlines.items():
            scores[asset] = _score_texts(texts, finbert)
            print(f"  FinBERT {asset}: {scores[asset]:+.4f}")
        return scores
    except Exception as e:
        logger.warning(f"FinBERT non disponible ({e}) — scores neutres 0.0")
        return {asset: 0.0 for asset in headlines}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Score QDM
# ─────────────────────────────────────────────────────────────────────────────

def build_qdm_sentiment_feature(
    df: pd.DataFrame,
    fg_df: Optional[pd.DataFrame] = None,
    finbert_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construit QDM(t) pour chaque ligne du DataFrame.
    QDM(t) = 0.5*FG(t) + 0.3*FinBERT(asset) + 0.2*Momentum(t)
    """
    df = df.copy()
    asset_col = "symbol" if "symbol" in df.columns else "asset"
    df = df.sort_values([asset_col, "timestamp"]).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # A. Fear & Greed — mapping via chaînes "YYYY-MM-DD" UTC
    #    Évite tout problème de dtype datetime / timezone
    # ══════════════════════════════════════════════════════════════════════════
    if fg_df is None:
        fg_df = download_fear_greed(days=220)

    # Construire le lookup { "YYYY-MM-DD" → fg_norm }
    # Si fg_df a "date_str" (v4), l'utiliser directement
    # Si fg_df a "day" ou "timestamp" (anciennes versions), convertir
    if "date_str" in fg_df.columns:
        fg_series = fg_df.set_index("date_str")["fg_norm"]
    elif "day" in fg_df.columns:
        fg_series = fg_df.copy()
        fg_series["date_str"] = _to_date_str(fg_series["day"])
        fg_series = fg_series.set_index("date_str")["fg_norm"]
    else:
        fg_series = fg_df.copy()
        fg_series["date_str"] = _to_date_str(fg_series["timestamp"])
        fg_series = fg_series.set_index("date_str")["fg_norm"]

    fg_series = fg_series.groupby(level=0).mean()   # déduplique les dates
    fg_lookup = fg_series.to_dict()

    # Convertir les timestamps du df en "YYYY-MM-DD" UTC
    df["_date_str"] = _to_date_str(df["timestamp"])

    # Diagnostic croisement
    df_dates  = set(df["_date_str"].unique())
    fg_dates  = set(fg_lookup.keys())
    overlap   = df_dates & fg_dates
    print(f"[QDM] Dates df     : {min(df_dates)} → {max(df_dates)}  ({len(df_dates)} jours)")
    print(f"[QDM] Dates FG     : {min(fg_dates)} → {max(fg_dates)}  ({len(fg_dates)} jours)")
    print(f"[QDM] Chevauchement: {len(overlap)} jours sur {len(df_dates)}")

    if len(overlap) == 0:
        print("[QDM] ⚠ AUCUN chevauchement — vérifiez la période téléchargée !")

    # Mapping direct
    df["fg_norm"] = df["_date_str"].map(fg_lookup)

    n_nan = df["fg_norm"].isna().sum()
    print(f"[QDM] NaN après mapping : {n_nan} / {len(df)}  "
          f"({100*n_nan/len(df):.1f}%)")

    if n_nan > 0:
        # Étendre le lookup ±3 jours pour les jours sans données FG
        extended = {}
        for d in sorted(fg_dates):
            day = pd.Timestamp(d)
            for offset in range(1, 4):
                for delta in [pd.Timedelta(days=offset), pd.Timedelta(days=-offset)]:
                    key = (day + delta).strftime("%Y-%m-%d")
                    if key not in fg_lookup and key not in extended:
                        extended[key] = fg_lookup[d]
        fg_lookup_ext = {**fg_lookup, **extended}
        df["fg_norm"] = df["_date_str"].map(fg_lookup_ext)
        n_nan2 = df["fg_norm"].isna().sum()
        print(f"[QDM] NaN après extension ±3j : {n_nan2} / {len(df)}")

        if n_nan2 > 0:
            # Dernier recours : ffill / bfill par ordre temporel
            df = df.sort_values("_date_str")
            df["fg_norm"] = df["fg_norm"].ffill().bfill().fillna(0.0)
            df = df.sort_values([asset_col, "timestamp"]).reset_index(drop=True)

    std_fg = df["fg_norm"].std()
    print(f"[QDM] fg_norm : std={std_fg:.4f}  "
          f"min={df['fg_norm'].min():.3f}  max={df['fg_norm'].max():.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # B. FinBERT (statique par actif)
    # ══════════════════════════════════════════════════════════════════════════
    if finbert_scores is None:
        finbert_scores = {a: 0.0 for a in df[asset_col].unique()}

    df["finbert_norm"] = df[asset_col].map(finbert_scores).fillna(0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # C. Price Momentum (z-score rolling)
    # ══════════════════════════════════════════════════════════════════════════
    def _momentum(g: pd.DataFrame) -> pd.Series:
        ret   = g["close"].pct_change(24)
        mu    = ret.rolling(168, min_periods=24).mean()
        sigma = ret.rolling(168, min_periods=24).std() + 1e-8
        return ((ret - mu) / sigma).clip(-3, 3) / 3.0

    df["price_mom"] = (
        df.groupby(asset_col, group_keys=False)
        .apply(_momentum)
        .fillna(0.0)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # D. Score QDM final
    # ══════════════════════════════════════════════════════════════════════════
    df["sentiment_score"] = (
        ALPHA * df["fg_norm"]
        + BETA  * df["finbert_norm"]
        + GAMMA * df["price_mom"]
    ).clip(-1.0, 1.0)

    df = df.drop(columns=["_date_str", "fg_norm", "finbert_norm", "price_mom"],
                 errors="ignore")

    # Validation finale
    std_q  = df["sentiment_score"].std()
    uniq_q = df["sentiment_score"].nunique()

    if std_q <= 0.01 or uniq_q <= 10:
        print(f"[QDM] 🚨 SCORE STATIQUE : std={std_q:.6f}, n_unique={uniq_q}")
        print(f"      période données : {df['timestamp'].min()} → {df['timestamp'].max()}")
    else:
        print(f"[QDM] ✅ Score temporel OK : std={std_q:.4f} | "
              f"n_unique={uniq_q:,} | "
              f"min={df['sentiment_score'].min():.3f} | "
              f"max={df['sentiment_score'].max():.3f}")

    return df.reset_index(drop=True)
