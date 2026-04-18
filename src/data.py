from __future__ import annotations
from pathlib import Path
from typing import Iterable
import time
import requests
import pandas as pd

REQUIRED_COLS = {"timestamp", "asset", "open", "high", "low", "close", "volume"}
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = _read_table(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return (
        df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
          .sort_values(["asset", "timestamp"])
          .reset_index(drop=True)
    )


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return _read_table(path)
    csv_fallback = path.with_suffix('.csv')
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback, parse_dates=["timestamp"])
    parquet_fallback = path.with_suffix('.parquet')
    if parquet_fallback.exists():
        return pd.read_parquet(parquet_fallback)
    raise FileNotFoundError(f"Dataset transformé introuvable: {path}, {csv_fallback} ou {parquet_fallback}")


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def _download_symbol_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    session = requests.Session()
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": min(limit, 1000),
        }
        response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.15)
    return rows


def download_binance_ohlcv(symbols: Iterable[str], interval: str = '1h', lookback_days: int = 180, limit: int = 1000) -> pd.DataFrame:
    end_ts = pd.Timestamp.utcnow()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    frames: list[pd.DataFrame] = []
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'n_trades', 'taker_buy_base_volume',
        'taker_buy_quote_volume', 'ignore'
    ]

    for symbol in symbols:
        rows = _download_symbol_klines(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms, limit=limit)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=columns)
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.rename(columns={'open_time': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['asset'] = symbol
        frames.append(df[['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume']])

    if not frames:
        raise RuntimeError('Aucune donnée Binance téléchargée. Vérifie la connexion internet ou les symboles demandés.')

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna().sort_values(['asset', 'timestamp']).reset_index(drop=True)
    return out
