from __future__ import annotations
from pathlib import Path
from sentiment import download_sentiment_datasets, merge_downloaded_sentiment_data
from utils import load_config

def main():
    cfg = load_config()
    out_dir = Path(cfg['sentiment']['output_dir'])
    merged_path = Path(cfg['sentiment']['merged_output_csv'])
    status = download_sentiment_datasets(out_dir)
    print('Téléchargement des bases de sentiment:')
    for k, v in status.items():
        print('-', k, '->', v)
    merged = merge_downloaded_sentiment_data(out_dir, merged_path)
    print(f'Base de sentiment fusionnée: {merged_path}')
    print(merged.head())
    print('Distribution des labels:')
    print(merged['label_text'].value_counts(dropna=False))

if __name__ == '__main__':
    main()
