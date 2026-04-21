"""
data.py — Téléchargement OHLCV multi-sources + chargement CSV

Sources par ordre de priorité :
  1. Binance.com  (API publique, bloquée sur Colab/US → HTTP 451)
  2. yfinance     (Yahoo Finance — fallback automatique si 451)

La détection est automatique : aucune config à changer.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# Endpoints miroirs Binance — essayés dans l'ordre
# data-api.binance.vision = endpoint données publiques, passe les blocages Colab/US
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
INTERVAL_MS  = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
# Mapping Binance symbol → Yahoo Finance ticker
YAHOO_TICKER = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "BNBUSDT": "BNB-USD",
    "SOLUSDT": "SOL-USD",
    "ADAUSDT": "ADA-USD",
    "XRPUSDT": "XRP-USD",
    "DOGEUSDT":"DOGE-USD",
}


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 1 : Binance
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_klines_binance(
    symbol: str, interval: str,
    start_ms: int, end_ms: int, limit: int = 1000,
) -> pd.DataFrame:
    """
    Un batch de klines Binance.
    Essaie tous les endpoints BINANCE_ENDPOINTS dans l'ordre.
    data-api.binance.vision passe les blocages Colab/US (HTTP 451).
    """
    params = dict(symbol=symbol, interval=interval,
                  startTime=start_ms, endTime=end_ms, limit=limit)
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","n_trades",
            "taker_buy_base","taker_buy_quote","ignore"]

    last_error = None
    for url in BINANCE_ENDPOINTS:
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code == 451:
                    logger.debug(f"451 sur {url} — endpoint suivant")
                    break   # sortir de la boucle attempt, essayer endpoint suivant
                r.raise_for_status()
                data = r.json()
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data, columns=cols)
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                df["symbol"] = symbol
                return df[["timestamp","symbol","open","high","low","close","volume"]]
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                last_error = e
                logger.warning(f"Binance {symbol} [{url}] attempt {attempt+1}/3 : {e} — attente {wait}s")
                time.sleep(wait)

    raise RuntimeError(f"Tous les endpoints Binance ont échoué pour {symbol} : {last_error}")


def _download_binance(
    symbols: List[str], interval: str,
    lookback_days: int, limit_per_call: int,
) -> pd.DataFrame:
    interval_ms = INTERVAL_MS.get(interval, 3_600_000)
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - lookback_days * 24 * 3600 * 1000
    all_dfs  = []

    for symbol in symbols:
        symbol_dfs = []
        cursor = start_ms
        while cursor < now_ms:
            end_chunk = min(cursor + limit_per_call * interval_ms, now_ms)
            chunk = _fetch_klines_binance(symbol, interval, cursor, end_chunk, limit_per_call)
            if chunk.empty:
                break
            symbol_dfs.append(chunk)
            cursor = int(chunk["timestamp"].iloc[-1].timestamp() * 1000) + interval_ms
            time.sleep(0.12)
        if symbol_dfs:
            df_sym = pd.concat(symbol_dfs, ignore_index=True)
            df_sym = df_sym.drop_duplicates("timestamp").sort_values("timestamp")
            all_dfs.append(df_sym)
            logger.info(f"  Binance {symbol} : {len(df_sym):,} lignes")

    if not all_dfs:
        raise RuntimeError("Binance : aucune donnée")
    return pd.concat(all_dfs, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 2 : yfinance (fallback Colab / IP bloqué)
# ═════════════════════════════════════════════════════════════════════════════

def _download_yfinance(
    symbols: List[str], interval: str, lookback_days: int,
) -> pd.DataFrame:
    """
    Télécharge via yfinance (Yahoo Finance).
    Mapping automatique : BTCUSDT → BTC-USD, etc.
    Installe yfinance si absent.
    """
    try:
        import yfinance as yf
    except ImportError:
        import subprocess, sys
        logger.info("Installation de yfinance…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
        import yfinance as yf

    # Mapping intervalle Binance → Yahoo
    yf_interval_map = {
        "1m":"1m", "5m":"5m", "15m":"15m",
        "1h":"1h", "4h":"1h",   # Yahoo n'a pas 4h → on prend 1h
        "1d":"1d",
    }
    yf_interval = yf_interval_map.get(interval, "1h")

    # Yahoo limite l'historique selon l'intervalle
    # 1h → max 730 jours ; 1d → illimité
    if yf_interval in ("1m","5m","15m") and lookback_days > 59:
        logger.warning(f"yfinance : intervalle {yf_interval} limité à 60 jours — ajustement")
        lookback_days = min(lookback_days, 59)

    period = f"{lookback_days}d"
    all_dfs = []

    for symbol in symbols:
        ticker = YAHOO_TICKER.get(symbol, symbol.replace("USDT", "-USD"))
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=yf_interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning(f"yfinance : aucune donnée pour {ticker}")
                continue

            # Aplatir MultiIndex si présent (yfinance >= 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.reset_index().rename(columns={"index":"timestamp","Datetime":"timestamp","Date":"timestamp"})

            # Normaliser le nom de la colonne timestamp
            ts_col = [c for c in df.columns if c.lower() in ("datetime","date","timestamp")]
            if ts_col:
                df = df.rename(columns={ts_col[0]: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = symbol

            df = df[["timestamp","symbol","open","high","low","close","volume"]].dropna()
            df = df.sort_values("timestamp").reset_index(drop=True)
            all_dfs.append(df)
            logger.info(f"  yfinance {symbol} ({ticker}) : {len(df):,} lignes")

        except Exception as e:
            logger.error(f"yfinance erreur {symbol} : {e}")

    if not all_dfs:
        raise RuntimeError("yfinance : aucune donnée récupérée")
    return pd.concat(all_dfs, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE PUBLIQUE
# ═════════════════════════════════════════════════════════════════════════════

def download_binance_ohlcv(
    symbols: List[str],
    interval: str = "1h",
    lookback_days: int = 180,
    limit_per_call: int = 1000,
    force_source: Optional[str] = None,   # "binance" | "yfinance" | None (auto)
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV avec fallback automatique.

    Ordre :
      1. Binance.com  (rapide, précis, bloqué sur Colab US → HTTP 451)
      2. yfinance     (fallback automatique si 451 ou erreur réseau)

    Parameters
    ----------
    symbols        : ['BTCUSDT', 'ETHUSDT', ...]
    interval       : '1h', '4h', '1d', ...
    lookback_days  : nombre de jours d'historique
    limit_per_call : taille des batches Binance
    force_source   : forcer une source ('binance' ou 'yfinance')

    Returns
    -------
    DataFrame : timestamp, symbol, open, high, low, close, volume
    """
    logger.info(f"Téléchargement : {symbols}")

    if force_source == "yfinance":
        logger.info("Source forcée : yfinance")
        df = _download_yfinance(symbols, interval, lookback_days)
    else:
        try:
            df = _download_binance(symbols, interval, lookback_days, limit_per_call)
            logger.info("✓ Source : Binance")
        except RuntimeError as e:
            if "451" in str(e) or "impossible" in str(e).lower():
                logger.warning(
                    f"Binance inaccessible ({e})\n"
                    f"→ Fallback automatique sur yfinance (Yahoo Finance)"
                )
                df = _download_yfinance(symbols, interval, lookback_days)
                logger.info("✓ Source : yfinance (fallback)")
            else:
                raise

    df = (
        df.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    logger.info(
        f"✓ Total : {len(df):,} lignes | "
        f"{df['symbol'].nunique()} actifs | "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Chargement CSV
# ═════════════════════════════════════════════════════════════════════════════

def load_market_data(path_or_cfg) -> pd.DataFrame:
    """
    Charge un CSV de données de marché.
    Accepte un chemin (str / Path) OU le dict cfg complet.
    """
    # Résolution du chemin
    if isinstance(path_or_cfg, dict):
        path = path_or_cfg.get("market", path_or_cfg).get("output_csv", "data/raw/market_data.csv")
    else:
        path = str(path_or_cfg)

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    else:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

    # Normaliser le nom de colonne actif
    if "asset" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"asset": "symbol"})

    asset_col = "symbol" if "symbol" in df.columns else df.columns[1]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(f"Données chargées : {path} → {len(df):,} lignes")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Split temporel
# ═════════════════════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict sans data leakage."""
    timestamps = df["timestamp"].sort_values().unique()
    n  = len(timestamps)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    train_ts = set(timestamps[:t1])
    val_ts   = set(timestamps[t1:t2])
    test_ts  = set(timestamps[t2:])

    train_df = df[df["timestamp"].isin(train_ts)].copy()
    val_df   = df[df["timestamp"].isin(val_ts)].copy()
    test_df  = df[df["timestamp"].isin(test_ts)].copy()

    logger.info(
        f"Split temporel : "
        f"Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
