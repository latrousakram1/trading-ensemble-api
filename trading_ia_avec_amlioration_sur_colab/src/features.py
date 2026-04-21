"""features.py — Feature engineering : 20 indicateurs techniques + sentiment QDM."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Colonnes de features ──────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # Rendements
    "ret_1h", "ret_4h", "ret_24h", "ret_168h",
    # Moyennes mobiles
    "sma_ratio_24", "sma_ratio_168", "ema_ratio_24",
    # RSI
    "rsi_14", "rsi_28",
    # Volatilité
    "vol_24h", "vol_168h",
    # Z-score
    "zscore_24h", "zscore_168h",
    # Structure de bougie
    "candle_body", "candle_upper_wick", "candle_lower_wick",
    # Momentum
    "momentum_12", "momentum_24",
    # Calendrier
    "hour_sin", "hour_cos",
    # Sentiment QDM
    "sentiment_score",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI classique Wilder."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _compute_qdm_auto(df: pd.DataFrame, asset_col: str) -> pd.Series:
    """
    Calcule le score QDM sans dépendance externe :
      - Fear & Greed : téléchargement API (fallback synthétique si indispo)
      - FinBERT      : 0.0 (non disponible sans NB00)
      - Price Mom    : z-score rolling 24h/168h sur les prix (100% local)

    Appelé automatiquement si sentiment_score est constant (= 0.0 partout).
    """
    try:
        from src.sentiment import build_qdm_sentiment_feature
        print("[features] sentiment_score constant détecté → calcul QDM automatique")
        df_qdm = build_qdm_sentiment_feature(df.copy())
        return df_qdm["sentiment_score"]
    except Exception as e:
        # Fallback minimal : price momentum seul (100% local, std > 0)
        logger.warning(f"[features] build_qdm_sentiment_feature échoué ({e}) "
                       f"→ fallback price momentum seul")

        def _mom(g: pd.DataFrame) -> pd.Series:
            ret   = g["close"].pct_change(24)
            mu    = ret.rolling(168, min_periods=24).mean()
            sigma = ret.rolling(168, min_periods=24).std() + 1e-8
            return ((ret - mu) / sigma).clip(-3, 3) / 3.0

        mom = (
            df.groupby(asset_col, group_keys=False)
            .apply(_mom)
            .fillna(0.0)
        )
        std = mom.std()
        print(f"[features] Fallback price momentum : std={std:.4f}")
        return mom * 0.20   # poids gamma seulement


def add_features(df: pd.DataFrame, auto_qdm: bool = True) -> pd.DataFrame:
    """
    Calcule les 20 features techniques + sentiment sur un DataFrame OHLCV.

    Parameters
    ----------
    df        : DataFrame avec symbol, timestamp, open, high, low, close, volume
    auto_qdm  : Si True (défaut), calcule automatiquement le score QDM quand
                sentiment_score est constant ou absent (évite QDM=0 permanent).

    Returns
    -------
    pd.DataFrame avec les colonnes de FEATURE_COLS ajoutées.
    """
    df = df.copy()

    # ── Normaliser le nom de colonne actif ────────────────────────────────────
    asset_col = "symbol" if "symbol" in df.columns else "asset"
    df = df.sort_values([asset_col, "timestamp"])

    # ── Sentiment : initialiser à 0 si absent ────────────────────────────────
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    def _per_asset(g: pd.DataFrame) -> pd.DataFrame:
        c = g["close"]

        # ── Rendements ──────────────────────────────────────────────────────
        g["ret_1h"]   = c.pct_change(1)
        g["ret_4h"]   = c.pct_change(4)
        g["ret_24h"]  = c.pct_change(24)
        g["ret_168h"] = c.pct_change(168)

        # ── Moyennes mobiles ────────────────────────────────────────────────
        sma24  = c.rolling(24,  min_periods=12).mean()
        sma168 = c.rolling(168, min_periods=48).mean()
        ema24  = c.ewm(span=24, min_periods=12).mean()

        g["sma_ratio_24"]  = (c / (sma24  + 1e-12)) - 1
        g["sma_ratio_168"] = (c / (sma168 + 1e-12)) - 1
        g["ema_ratio_24"]  = (c / (ema24  + 1e-12)) - 1

        # ── RSI ─────────────────────────────────────────────────────────────
        g["rsi_14"] = _rsi(c, 14) / 100.0
        g["rsi_28"] = _rsi(c, 28) / 100.0

        # ── Volatilité ───────────────────────────────────────────────────────
        g["vol_24h"]  = g["ret_1h"].rolling(24,  min_periods=8).std()
        g["vol_168h"] = g["ret_1h"].rolling(168, min_periods=24).std()

        # ── Z-score ─────────────────────────────────────────────────────────
        m24  = c.rolling(24,  min_periods=12).mean()
        s24  = c.rolling(24,  min_periods=12).std()
        m168 = c.rolling(168, min_periods=48).mean()
        s168 = c.rolling(168, min_periods=48).std()
        g["zscore_24h"]  = (c - m24)  / (s24  + 1e-12)
        g["zscore_168h"] = (c - m168) / (s168 + 1e-12)

        # ── Structure de bougie ─────────────────────────────────────────────
        body   = (g["close"] - g["open"]).abs() / (c + 1e-12)
        u_wick = (g["high"] - g[["close","open"]].max(axis=1)) / (c + 1e-12)
        l_wick = (g[["close","open"]].min(axis=1) - g["low"]) / (c + 1e-12)
        g["candle_body"]       = body.clip(-0.5, 0.5)
        g["candle_upper_wick"] = u_wick.clip(0, 0.5)
        g["candle_lower_wick"] = l_wick.clip(0, 0.5)

        # ── Momentum ────────────────────────────────────────────────────────
        g["momentum_12"] = c.pct_change(12).clip(-0.5, 0.5)
        g["momentum_24"] = c.pct_change(24).clip(-0.5, 0.5)

        return g

    df = df.groupby(asset_col, group_keys=False).apply(_per_asset)

    # ── Calendrier (cyclique) ────────────────────────────────────────────────
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── Sentiment QDM ────────────────────────────────────────────────────────
    # Détecter si le score est constant (cas typique : CSV chargé sans NB00)
    # → déclencher le calcul QDM automatiquement
    score_std    = df["sentiment_score"].std()
    score_unique = df["sentiment_score"].nunique()
    is_constant  = (score_std < 1e-6) or (score_unique <= 1)

    if is_constant and auto_qdm:
        df["sentiment_score"] = _compute_qdm_auto(df, asset_col).values
        # Vérification post-calcul
        new_std = df["sentiment_score"].std()
        print(f"[features] QDM calculé automatiquement : std={new_std:.4f}")
    else:
        df["sentiment_score"] = df["sentiment_score"].fillna(0.0).clip(-1.0, 1.0)
        if not is_constant:
            logger.info(f"[features] sentiment_score déjà varié : std={score_std:.4f}")

    # ── Clips finaux ─────────────────────────────────────────────────────────
    for col in ["ret_1h", "ret_4h", "ret_24h", "ret_168h"]:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)
    for col in ["vol_24h", "vol_168h"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 0.2)
    for col in ["zscore_24h", "zscore_168h"]:
        if col in df.columns:
            df[col] = df[col].clip(-5, 5)
    for col in ["sma_ratio_24", "sma_ratio_168", "ema_ratio_24"]:
        if col in df.columns:
            df[col] = df[col].clip(-0.5, 0.5)

    df = df.reset_index(drop=True)
    n_features = sum(1 for c in df.columns if c in FEATURE_COLS)
    logger.info(f"Features calculées : {n_features}/{len(FEATURE_COLS)}")
    return df
