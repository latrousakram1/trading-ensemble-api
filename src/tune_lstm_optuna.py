"""
tune_lstm_optuna.py
====================
Optimisation des hyperparamètres du modèle LSTMAttention avec Optuna.

Objectif  : maximiser le F1-macro sur le jeu de validation
Modèle    : LSTMAttention (LSTM bidirectionnel + self-attention)
Trials    : 20 (configurable via --n-trials)

Espace de recherche
-------------------
  hidden_dim   : [64, 128, 256]
  n_layers     : 1 → 4
  n_heads      : [2, 4, 8]
  dropout      : 0.1 → 0.5
  lr           : 1e-4 → 5e-3 (log)
  weight_decay : 1e-5 → 1e-2 (log)
  batch_size   : [32, 64, 128]
  label_smooth : 0.0 → 0.1
  seq_len      : [48, 96]        ← fenêtre temporelle
  buy_thresh   : 0.004 → 0.015  ← seuil de labellisation
  sell_thresh  : 0.004 → 0.015  (symétrique négatif)

Usage
-----
  python src/tune_lstm_optuna.py
  python src/tune_lstm_optuna.py --n-trials 50
  python src/tune_lstm_optuna.py --n-trials 20 --n-startup 5
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import mlflow
import mlflow.pytorch
try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.advanced_models import LSTMAttention
    from src.train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate, make_loaders
    from src.metrics import compute_reliable_metrics
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from advanced_models import LSTMAttention
    from train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate, make_loaders
    from metrics import compute_reliable_metrics

FEATURE_SUFFIXES = ('ret_', 'sma_', 'ema_', 'rsi_', 'vol_', 'zscore_')
EXTRA_FEATURES   = ['range_pct', 'body_pct', 'volume_zscore_20',
                    'hour', 'dayofweek', 'month', 'sentiment_score']


# ─── Dataset (avec scaler intégré) ────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len, scaler=None):
        self.samples, self.labels = [], []
        for _, g in df.groupby('asset'):
            g = g.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
            if len(g) <= seq_len:
                continue
            X = g[feature_cols].astype('float32').values
            if scaler is not None:
                X = (X - scaler['mean']) / scaler['std']
            y = g['target'].astype(int).values
            for i in range(seq_len, len(g)):
                self.samples.append(X[i - seq_len:i])
                self.labels.append(int(y[i]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx],  dtype=torch.long))


def get_feature_cols(df):
    cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + EXTRA_FEATURES
    return [c for c in cols if c in df.columns]


def fit_scaler(df, feature_cols):
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
    return {'mean': x.mean(axis=0).values.astype('float32'),
            'std':  x.std(axis=0).replace(0, 1.0).values.astype('float32')}


# ─── Fonction objectif Optuna ──────────────────────────────────────────────────
def make_objective(df_full, device, n_epochs_per_trial=8):
    """
    Retourne la fonction objectif pour Optuna.

    On pré-charge les données une seule fois (df_full) et on
    reconstruit les datasets à chaque trial avec les hyperparamètres
    proposés (seq_len, buy/sell threshold peuvent varier).
    """
    cfg = load_config()

    def objective(trial: optuna.Trial) -> float:
        # ── 1. Hyperparamètres à optimiser ──────────────────────────
        hidden_dim   = trial.suggest_categorical('hidden_dim',   [128, 256])
        n_layers     = trial.suggest_int(        'n_layers',     1, 4)
        n_heads_opts = [h for h in [2, 4, 8] if hidden_dim * 2 % h == 0]
        n_heads      = trial.suggest_categorical('n_heads',      n_heads_opts)
        dropout      = trial.suggest_float(      'dropout',      0.1, 0.5)
        lr           = trial.suggest_float(      'lr',           1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float(      'weight_decay', 1e-5, 1e-2, log=True)
        batch_size   = trial.suggest_categorical('batch_size',   [32, 64, 128])
        label_smooth = trial.suggest_float(      'label_smooth', 0.0, 0.1)
        seq_len      = trial.suggest_categorical('seq_len',      [48, 96])
        buy_thr      = trial.suggest_float(      'buy_threshold',  0.004, 0.015)
        sell_thr     = trial.suggest_float(      'sell_threshold', 0.004, 0.015)

        # ── 2. Rebuild target avec les seuils du trial ──────────────
        df = build_target(df_full.copy(),
                          horizon=cfg['target']['horizon'],
                          buy_threshold=buy_thr,
                          sell_threshold=-sell_thr)

        train_df, val_df, _ = temporal_split(df, 0.70, 0.15)
        feature_cols = get_feature_cols(df)
        scaler = fit_scaler(train_df, feature_cols)

        train_ds = SeqDataset(train_df, feature_cols, seq_len, scaler)
        val_ds   = SeqDataset(val_df,   feature_cols, seq_len, scaler)

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise optuna.TrialPruned()

        # WeightedSampler pour équilibrer les classes
        y_train = np.array(train_ds.labels)
        cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
        multipliers = np.array(
            cfg['training'].get('class_weight_multipliers', [1.0, 1.0, 1.0]),
            dtype='float32',
        )
        if multipliers.shape[0] == cw.shape[0]:
            cw = cw * multipliers
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
        sw = np.array([cw[y] for y in train_ds.labels], dtype='float64')
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # ── 3. Modèle ───────────────────────────────────────────────
        n_features = len(feature_cols)
        model = LSTMAttention(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            n_classes=cfg['model']['n_classes'],
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smooth)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs_per_trial, eta_min=lr * 0.1
        )

        # ── 4. Entraînement rapide + pruning intermédiaire ──────────
        best_f1 = 0.0
        for epoch in range(1, n_epochs_per_trial + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            # Évaluation intermédiaire + pruning
            model.eval()
            ys, preds = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                    ys.extend(yb.numpy())
            f1 = float(f1_score(ys, preds, average='macro', zero_division=0))

            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            best_f1 = max(best_f1, f1)

        return best_f1

    return objective


# ─── Entraînement final avec les meilleurs paramètres ─────────────────────────
def train_best_model(best_params: dict, df_full, device, cfg, n_epochs: int = 20):
    """
    Ré-entraîne le modèle LSTMAttention avec les meilleurs hyperparamètres
    sur train+val, puis évalue sur le test set.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    df = build_target(df_full.copy(),
                      horizon=cfg['target']['horizon'],
                      buy_threshold=best_params['buy_threshold'],
                      sell_threshold=-best_params['sell_threshold'])

    train_df, val_df, test_df = temporal_split(df, 0.70, 0.15)
    feature_cols = get_feature_cols(df)
    # Scaler sur train uniquement
    scaler = fit_scaler(train_df, feature_cols)

    seq_len = best_params['seq_len']
    bs      = best_params['batch_size']

    train_ds = SeqDataset(train_df, feature_cols, seq_len, scaler)
    val_ds   = SeqDataset(val_df,   feature_cols, seq_len, scaler)
    test_ds  = SeqDataset(test_df,  feature_cols, seq_len, scaler)

    y_train = np.array(train_ds.labels)
    cw      = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
    multipliers = np.array(
        cfg['training'].get('class_weight_multipliers', [1.0, 1.0, 1.0]),
        dtype='float32',
    )
    if multipliers.shape[0] == cw.shape[0]:
        cw = cw * multipliers
    cw_t    = torch.tensor(cw, dtype=torch.float32, device=device)
    sw      = np.array([cw[y] for y in train_ds.labels], dtype='float64')
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    n_features = len(feature_cols)
    model = LSTMAttention(
        n_features=n_features,
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        n_heads=best_params['n_heads'],
        dropout=best_params['dropout'],
        n_classes=cfg['model']['n_classes'],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=cw_t, label_smoothing=best_params['label_smooth']
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_f1, best_state, patience_count = -1.0, None, 0
    patience = cfg['training']['patience']

    print('\n─── Entraînement final avec les meilleurs hyperparamètres ───')
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                ys.extend(yb.numpy())
        f1 = float(f1_score(ys, preds, average='macro', zero_division=0))
        scheduler.step(f1)
        print(f'  Epoch {epoch:02d} | loss={np.mean(losses):.4f} | val_f1={f1:.4f}')

        if f1 > best_f1:
            best_f1, patience_count = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= patience:
            print('  Early stopping.')
            break

    model.load_state_dict(best_state)

    # Évaluation test set
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = torch.softmax(model(xb.to(device)), dim=1)
            preds.extend(p.argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            ys.extend(yb.numpy())

    acc = accuracy_score(ys, preds)
    f1m = f1_score(ys, preds, average='macro', zero_division=0)
    try:    auc = float(roc_auc_score(ys, probs, multi_class='ovr'))
    except: auc = float('nan')

    print(f'\n  TEST → acc={acc:.4f} | f1={f1m:.4f} | auc={auc:.4f}')
    print(classification_report(ys, preds, target_names=['Sell','Hold','Buy'], zero_division=0))

    return model, {
        'test_acc': round(acc, 4), 'test_f1_macro': round(f1m, 4),
        'test_auc_ovr': round(auc, 4),
        'feature_cols': feature_cols, 'scaler': scaler,
        'best_params': best_params,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Optuna — LSTM Attention F1-macro')
    parser.add_argument('--n-trials',   type=int, default=50, help='Nombre de trials Optuna')
    parser.add_argument('--n-startup',  type=int, default=5,  help='Trials aléatoires avant TPE')
    parser.add_argument('--n-epochs',   type=int, default=12,  help='Epochs par trial (tuning rapide)')
    parser.add_argument('--final-epochs', type=int, default=20, help='Epochs entraînement final')
    parser.add_argument('--study-name', type=str, default='lstm_attention_f1',
                        help='Nom du study Optuna (pour reprise)')
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg['project']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    print(f'Trials : {args.n_trials}  |  Epochs/trial : {args.n_epochs}')

    # ── Charger les données une seule fois ──────────────────────────
    preferred = Path(cfg['sentiment']['aligned_output_csv'])
    market_path = preferred if preferred.exists() else Path(cfg['market']['output_csv'])
    df_raw = load_market_data(market_path)
    if 'sentiment_score' not in df_raw.columns:
        df_raw['sentiment_score'] = 0.0
    df_raw = add_features(df_raw)
    print(f'Données : {len(df_raw):,} lignes, {df_raw["asset"].nunique()} actifs')

    # ── Study Optuna avec pruning MedianPruner ──────────────────────
    # MedianPruner abandonne les trials dont la métrique intermédiaire
    # est inférieure à la médiane des trials précédents → 30-50% de gains de temps
    sampler = TPESampler(n_startup_trials=args.n_startup, seed=cfg['project']['seed'])
    pruner  = MedianPruner(n_startup_trials=args.n_startup, n_warmup_steps=3)

    mlflow.set_experiment(cfg['project']['mlflow_experiment'] + '-Optuna-LSTM')

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    print('\n── Démarrage de l\'optimisation ──')

    def optuna_mlflow_callback(study, trial):
        """Log chaque trial complété dans MLflow."""
        if trial.value is not None:
            with mlflow.start_run(run_name=f'trial_{trial.number:03d}', nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric('val_f1_macro', trial.value)
                mlflow.log_metric('trial_number', trial.number)
            print(f'  Trial {trial.number:02d} → F1={trial.value:.4f} | params={trial.params}')
        else:
            print(f'  Trial {trial.number:02d} → pruned')

    objective = make_objective(df_raw, device, n_epochs_per_trial=args.n_epochs)

    with mlflow.start_run(run_name=f'optuna_study_{args.study_name}') as parent_run:
        mlflow.log_params({
            'n_trials':         args.n_trials,
            'n_startup_trials': args.n_startup,
            'n_epochs_trial':   args.n_epochs,
            'final_epochs':     args.final_epochs,
            'objective':        'val_f1_macro',
            'model':            'lstm_attention',
        })

        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            callbacks=[optuna_mlflow_callback],
        )

        # Log résultats globaux dans le run parent
        mlflow.log_metric('best_val_f1', study.best_value)
        mlflow.log_params({f'best_{k}': v for k, v in study.best_params.items()})
        n_pruned = len([t for t in study.trials if t.value is None])
        mlflow.log_metric('n_pruned_trials', n_pruned)
        print(f'  MLflow parent run_id : {parent_run.info.run_id}')

    # ── Résultats ────────────────────────────────────────────────────
    print('\n═══════════════════════════════════════')
    print(f'  Meilleur F1-macro (val) : {study.best_value:.4f}')
    print(f'  Meilleurs paramètres :')
    for k, v in study.best_params.items():
        print(f'    {k:<20} = {v}')

    # Top 5 trials
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:5]
    print('\n  Top 5 trials :')
    print(f"  {'#':>3}  {'F1-macro':>9}  paramètres clés")
    for t in top5:
        p = t.params
        print(f"  {t.number:>3}  {t.value:>9.4f}  "
              f"hidden={p.get('hidden_dim')}  layers={p.get('n_layers')}  "
              f"lr={p.get('lr', 0):.2e}  drop={p.get('dropout', 0):.2f}")

    # Importance des hyperparamètres
    try:
        importance = optuna.importance.get_param_importances(study)
        print('\n  Importance des hyperparamètres :')
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(imp * 30)
            print(f'    {param:<20} {bar} {imp:.3f}')
    except Exception:
        pass

    # ── Sauvegarder les résultats ────────────────────────────────────
    artifact_dir = Path(cfg['paths']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    all_trials = [
        {'number': t.number, 'value': t.value, 'params': t.params,
         'state': str(t.state)}
        for t in study.trials
    ]
    save_json({
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'top5': [{'number': t.number, 'value': t.value, 'params': t.params} for t in top5],
        'all_trials': all_trials,
    }, artifact_dir / 'optuna_lstm_best.json')

    # ── Entraînement final avec les meilleurs paramètres ────────────
    print('\n── Entraînement final ──')
    best_model, final_metrics = train_best_model(
        study.best_params, df_raw, device, cfg, n_epochs=args.final_epochs
    )

    # Sauvegarder le modèle optimisé
    model_dir = Path(cfg['paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_name': 'lstm_attention_tuned',
        'model_state_dict': best_model.state_dict(),
        'feature_cols': final_metrics['feature_cols'],
        'scaler_mean': final_metrics['scaler']['mean'].tolist(),
        'scaler_std':  final_metrics['scaler']['std'].tolist(),
        'best_params': study.best_params,
        'optuna_best_f1': study.best_value,
        'test_metrics': {k: v for k, v in final_metrics.items()
                         if k not in ('feature_cols', 'scaler', 'best_params')},
        'config': cfg,
    }, model_dir / 'lstm_attention_tuned.pt')

    save_json({
        'test_acc':       final_metrics['test_acc'],
        'test_f1_macro':  final_metrics['test_f1_macro'],
        'test_auc_ovr':   final_metrics['test_auc_ovr'],
        'optuna_best_f1': round(study.best_value, 4),
        'best_params':    study.best_params,
    }, artifact_dir / 'lstm_attention_tuned_metrics.json')

    # Log métriques finales dans MLflow (run parent toujours actif)
    try:
        with mlflow.start_run(run_name='final_training_best_params', nested=True):
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({
                'final_test_acc':      final_metrics['test_acc'],
                'final_test_f1_macro': final_metrics['test_f1_macro'],
                'final_test_auc_ovr':  final_metrics['test_auc_ovr'],
                'optuna_best_val_f1':  study.best_value,
            })
            mlflow.pytorch.log_model(best_model, 'lstm_attention_tuned')
    except Exception as e:
        print(f'  MLflow final log warning : {e}')

    print(f'\n✓ Modèle optimisé sauvegardé → {model_dir / "lstm_attention_tuned.pt"}')
    print(f'✓ Métriques sauvegardées     → {artifact_dir / "lstm_attention_tuned_metrics.json"}')
    print(f'✓ Tous les trials            → {artifact_dir / "optuna_lstm_best.json"}')


if __name__ == '__main__':
    main()
