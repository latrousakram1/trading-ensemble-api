from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from pathlib import Path

import pandas as pd
import websockets

try:
    from src.data import download_binance_ohlcv
except ImportError:
    from data import download_binance_ohlcv

log = logging.getLogger("trading-realtime")


class BinanceRealtimeBuffer:
    def __init__(
        self,
        symbols: list[str],
        interval: str = "1h",
        buffer_hours: int = 96,
        bootstrap_hours: int = 120,
        websocket_url: str = "wss://stream.binance.com:9443/stream",
        reconnect_delay_s: int = 5,
        snapshot_csv: str | None = None,
    ) -> None:
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval = interval
        self.buffer_hours = max(int(buffer_hours), 96)
        self.bootstrap_hours = max(int(bootstrap_hours), self.buffer_hours)
        self.websocket_url = websocket_url.rstrip("/")
        self.reconnect_delay_s = max(int(reconnect_delay_s), 1)
        self.snapshot_csv = Path(snapshot_csv) if snapshot_csv else None

        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()
        self._ready = False
        self._running = False
        self._task: asyncio.Task | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._bootstrap)
        self._ready = True

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._run_loop(), name="binance-realtime-buffer")
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and any(not frame.empty for frame in self._frames.values())

    def status(self) -> dict:
        with self._lock:
            latest = {}
            for symbol, frame in self._frames.items():
                if frame.empty:
                    latest[symbol] = None
                else:
                    latest[symbol] = str(frame["timestamp"].iloc[-1])
            return {
                "enabled": True,
                "ready": self._ready,
                "symbols": self.symbols,
                "buffer_hours": self.buffer_hours,
                "latest_timestamps": latest,
            }

    def snapshot(self, asset: str | None = None) -> pd.DataFrame:
        with self._lock:
            if asset is not None:
                frame = self._frames.get(asset.upper(), pd.DataFrame()).copy()
                return frame.reset_index(drop=True)
            frames = [frame.copy() for frame in self._frames.values() if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=["timestamp", "asset", "open", "high", "low", "close", "volume"])
        return pd.concat(frames, ignore_index=True).sort_values(["asset", "timestamp"]).reset_index(drop=True)

    def update_from_closed_candle(self, symbol: str, candle: dict) -> None:
        symbol = symbol.upper()
        row = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(int(candle["t"]), unit="ms", utc=True),
                    "asset": symbol,
                    "open": float(candle["o"]),
                    "high": float(candle["h"]),
                    "low": float(candle["l"]),
                    "close": float(candle["c"]),
                    "volume": float(candle["v"]),
                }
            ]
        )
        with self._lock:
            current = self._frames.get(symbol, pd.DataFrame())
            frame = pd.concat([current, row], ignore_index=True)
            frame = (
                frame.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .tail(self.buffer_hours)
                .reset_index(drop=True)
            )
            self._frames[symbol] = frame
        self._persist_snapshot()

    def _bootstrap(self) -> None:
        lookback_days = max(2, math.ceil(self.bootstrap_hours / 24))
        df = download_binance_ohlcv(
            symbols=self.symbols,
            interval=self.interval,
            lookback_days=lookback_days,
        )
        with self._lock:
            self._frames = {}
            for symbol in self.symbols:
                frame = (
                    df[df["asset"] == symbol]
                    .sort_values("timestamp")
                    .tail(self.buffer_hours)
                    .reset_index(drop=True)
                )
                self._frames[symbol] = frame
        self._persist_snapshot()
        log.info("Bootstrap realtime charge: %s", {k: len(v) for k, v in self._frames.items()})

    async def _run_loop(self) -> None:
        if not self.symbols:
            return

        stream_names = "/".join(f"{symbol.lower()}@kline_{self.interval}" for symbol in self.symbols)
        ws_url = f"{self.websocket_url}?streams={stream_names}"

        while self._running:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as websocket:
                    log.info("Websocket Binance connecte: %s", ws_url)
                    async for raw_message in websocket:
                        payload = json.loads(raw_message)
                        data = payload.get("data", {})
                        candle = data.get("k", {})
                        if candle.get("i") != self.interval or not candle.get("x"):
                            continue
                        symbol = data.get("s")
                        if symbol:
                            self.update_from_closed_candle(symbol, candle)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Websocket Binance indisponible: %s", exc)
                await asyncio.sleep(self.reconnect_delay_s)

    def _persist_snapshot(self) -> None:
        if self.snapshot_csv is None:
            return
        snapshot = self.snapshot()
        self.snapshot_csv.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(self.snapshot_csv, index=False)
