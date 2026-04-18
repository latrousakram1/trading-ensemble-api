from __future__ import annotations
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.model import PatchTSTLite
    from src.train_utils import get_feature_cols, fit_scaler, SeqDataset
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from model import PatchTSTLite
    from train_utils import get_feature_cols, fit_scaler, SeqDataset

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, seq_len=96):
        self.samples = []
        for _, g in df.groupby('asset'):
            g = g.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
            X = g[feature_cols].astype('float32').values
            y = g['target'].astype(int).values
            for i in range(seq_len, len(g)):
                self.samples.append((X[i-seq_len:i], y[i]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

def objective(trial):
    cfg = load_config()
    set_seed(cfg['project']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    market_path = Path(cfg['sentiment']['aligned_output_csv']) if Path(cfg['sentiment']['aligned_output_csv']).exists() else Path(cfg['market']['output_csv'])
    df = load_market_data(market_path)
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    df = add_features(df)
    df = build_target(df, horizon=cfg['target']['horizon'], buy_threshold=cfg['target']['buy_threshold'], sell_threshold=cfg['target']['sell_threshold'])
    train_df, val_df, _ = temporal_split(df, 0.70, 0.15)
    feature_cols = [c for c in df.columns if c.startswith(('ret_', 'sma_', 'ema_', 'rsi_', 'vol_', 'zscore_'))] + ['range_pct', 'body_pct', 'volume_zscore_20', 'hour', 'dayofweek', 'month', 'sentiment_score']
    seq_len = cfg['model']['seq_len']
    train_ds = SeqDataset(train_df, feature_cols, seq_len)
    val_ds = SeqDataset(val_df, feature_cols, seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)

    model = PatchTSTLite(
        n_features=len(feature_cols), seq_len=seq_len,
        patch_len=cfg['model']['patch_len'], stride=cfg['model']['stride'],
        d_model=d_model, n_heads=cfg['model']['n_heads'], n_layers=n_layers,
        ff_dim=cfg['model']['ff_dim'], dropout=dropout, n_classes=cfg['model']['n_classes']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            ys.extend(yb.numpy().tolist())
    return f1_score(ys, preds, average='macro')

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=load_config()['optuna']['n_trials'])
    result = {'best_value': study.best_value, 'best_params': study.best_params}
    save_json(result, Path(load_config()['paths']['artifact_dir']) / 'optuna_best.json')
    print(result)

if __name__ == '__main__':
    main()
