from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.model import PatchTSTLite
    from src.train_utils import SeqDataset, fit_scaler, get_feature_cols, evaluate, make_loaders
    from src.metrics import compute_reliable_metrics
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from model import PatchTSTLite
    from train_utils import SeqDataset, fit_scaler, get_feature_cols, evaluate, make_loaders
    from metrics import compute_reliable_metrics

FEATURE_SUFFIXES = ('ret_', 'sma_', 'ema_', 'rsi_', 'vol_', 'zscore_')
EXTRA_FEATURES = ['range_pct', 'body_pct', 'volume_zscore_20', 'hour', 'dayofweek', 'month', 'sentiment_score']

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, seq_len=96, scaler=None):
        self.samples = []
        self.labels = []
        for _, g in df.groupby('asset'):
            g = g.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
            if len(g) <= seq_len:
                continue
            X = g[feature_cols].astype('float32').values
            if scaler is not None:
                X = (X - scaler['mean']) / scaler['std']
            y = g['target'].astype(int).values
            for i in range(seq_len, len(g)):
                self.samples.append(X[i-seq_len:i])
                self.labels.append(int(y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def get_feature_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + EXTRA_FEATURES
    return [c for c in cols if c in df.columns]

def fit_scaler(df: pd.DataFrame, feature_cols):
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
    mean = x.mean(axis=0).values.astype('float32')
    std = x.std(axis=0).replace(0, 1.0).values.astype('float32')
    return {'mean': mean, 'std': std}

def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(dim=1)
            ys.extend(yb.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(ys, probs, multi_class='ovr')
    except Exception:
        auc = float('nan')
    report = classification_report(ys, preds, output_dict=True, zero_division=0)
    return {'acc': acc, 'f1_macro': f1, 'auc_ovr': auc, 'report': report}

def main():
    cfg = load_config()
    set_seed(cfg['project']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preferred = Path(cfg['sentiment']['aligned_output_csv'])
    market_path = preferred if preferred.exists() else Path(cfg['market']['output_csv'])
    df = load_market_data(market_path)
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    df = add_features(df)
    df = build_target(df,
                      horizon=cfg['target']['horizon'],
                      buy_threshold=cfg['target']['buy_threshold'],
                      sell_threshold=cfg['target']['sell_threshold'])
    train_df, val_df, test_df = temporal_split(df, 0.70, 0.15)
    feature_cols = get_feature_cols(df)
    scaler = fit_scaler(train_df, feature_cols)

    seq_len = cfg['model']['seq_len']
    train_ds = SeqDataset(train_df, feature_cols, seq_len, scaler=scaler)
    val_ds = SeqDataset(val_df, feature_cols, seq_len, scaler=scaler)
    test_ds = SeqDataset(test_df, feature_cols, seq_len, scaler=scaler)

    y_train = np.array(train_ds.labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    sample_weights = np.array([class_weights[int(y)].item() for y in train_ds.labels], dtype='float64')
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = PatchTSTLite(n_features=len(feature_cols), **cfg['model']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg['training']['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    mlflow.set_experiment(cfg['project']['mlflow_experiment'])
    best_f1, patience_count, best_state = -1.0, 0, None

    with mlflow.start_run(run_name='patchtst_improved'):
        mlflow.log_params({
            'seq_len': seq_len,
            'd_model': cfg['model']['d_model'],
            'n_layers': cfg['model']['n_layers'],
            'dropout': cfg['model']['dropout'],
            'lr': cfg['training']['lr'],
            'n_features': len(feature_cols),
            'buy_threshold': cfg['target']['buy_threshold'],
            'sell_threshold': cfg['target']['sell_threshold'],
            'probability_threshold': cfg['backtest']['probability_threshold'],
        })

        for epoch in range(1, cfg['training']['epochs'] + 1):
            model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                optimizer.step()
                losses.append(loss.item())

            train_loss = float(np.mean(losses)) if losses else float('nan')
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics['f1_macro'])
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_acc': val_metrics['acc'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_auc_ovr': val_metrics['auc_ovr'],
            }, step=epoch)
            print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_acc={val_metrics['acc']:.4f} | val_f1={val_metrics['f1_macro']:.4f} | val_auc={val_metrics['auc_ovr']:.4f}")

            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= cfg['training']['patience']:
                print('Early stopping activé.')
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate(model, test_loader, device)
        mlflow.log_metrics({
            'test_acc': test_metrics['acc'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_auc_ovr': test_metrics['auc_ovr'],
        })
        out_model = Path(cfg['paths']['model_dir']) / 'patchtst_final.pt'
        out_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_cols': feature_cols,
            'scaler_mean': scaler['mean'].tolist(),
            'scaler_std': scaler['std'].tolist(),
            'config': cfg,
        }, out_model)
        summary = {
            'test_acc': test_metrics['acc'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_auc_ovr': test_metrics['auc_ovr'],
            'classification_report': test_metrics['report'],
            'feature_cols': feature_cols,
            'model_path': str(out_model),
        }
        save_json(summary, Path(cfg['paths']['artifact_dir']) / 'patchtst_metrics.json')
        print('TEST:', summary)

if __name__ == '__main__':
    main()
