from __future__ import annotations
from pathlib import Path

try:
    from src.data import download_binance_ohlcv
    from src.utils import load_config
except ImportError:
    from data import download_binance_ohlcv
    from utils import load_config


def main():
    cfg = load_config()
    out_path = Path(cfg['market']['output_csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = download_binance_ohlcv(
        symbols=cfg['market']['symbols'],
        interval=cfg['market']['interval'],
        lookback_days=cfg['market']['lookback_days'],
        limit=cfg['market']['limit_per_call'],
    )
    df.to_csv(out_path, index=False)
    print(f'Données marché sauvegardées dans: {out_path}')
    print(df.groupby('asset').size())
    print(df.head())


if __name__ == '__main__':
    main()
