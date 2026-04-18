from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.data import load_market_data
    from src.sentiment import build_proxy_sentiment_feature
    from src.utils import load_config
except ImportError:
    from data import load_market_data
    from sentiment import build_proxy_sentiment_feature
    from utils import load_config


def main():
    cfg = load_config()
    market_path = Path(cfg["market"]["output_csv"])
    sentiment_path = Path(cfg["sentiment"]["merged_output_csv"])
    out_path = Path(cfg["sentiment"]["aligned_output_csv"])

    if not market_path.exists():
        raise FileNotFoundError(
            f"Le fichier marche n'existe pas encore: {market_path}. "
            "Lance d'abord python -m src.download_market_data"
        )
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"Le fichier sentiment n'existe pas encore: {sentiment_path}. "
            "Lance d'abord python -m src.download_sentiment_data"
        )

    market_df = load_market_data(market_path)
    sent_df = pd.read_csv(sentiment_path)
    aligned_df = build_proxy_sentiment_feature(market_df, sent_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(out_path, index=False)
    print(f"Dataset prepare: {out_path}")
    print(aligned_df.head())


if __name__ == "__main__":
    main()
