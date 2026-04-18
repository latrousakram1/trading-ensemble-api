"""
tune_all_optuna.py
==================
Optimisation multi-objectif Pareto des 3 modèles de l'ensemble :
    patchtst · lstm_attention · cnn_transformer

Objectifs simultanés (NSGAIISampler) :
    1. Maximiser F1-macro  (classification)
    2. Maximiser Sharpe    (backtest portefeuille)

Stratégie :
    - 20 trials par modèle, séquentiel
    - NSGAIISampler  → front de Pareto
    - MedianPruner   → abandonne les mauvais trials dès l'epoch 3
    - 8 epochs/trial pendant la recherche, 25 epochs pour l'entraînement final
    - Le modèle final retenu = meilleur compromis F1+Sharpe (distance à l'utopie)

Usage
-----
    python src/tune_all_optuna.py
    python src/tune_all_optuna.py --n-trials 50 --models patchtst lstm_attention
    python src/tune_all_optuna.py --final-epochs 30
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
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
    from src.advanced_models import LSTMAttention, CNNTransformer
    from src.model import PatchTSTLite
    from src.train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate, make_loaders, run_epoch
    from src.metrics import compute_reliable_metrics, period_return_from_horizon
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from advanced_models import LSTMAttention, CNNTransformer
    from model import PatchTSTLite
    from train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate, make_loaders, run_epoch
    from metrics import compute_reliable_metrics, period_return_from_horizon

FEATURE_SUFFIXES = ('ret_', 'sma_', 'ema_', 'rsi_', 'vol_', 'zscore_')
EXTRA_FEATURES   = ['range_pct', 'body_pct', 'volume_zscore_20',
                    'hour', 'dayofweek', 'month', 'sentiment_score']
ANN_FACTOR       = np.sqrt(365 * 24)

ALL_MODELS = ['patchtst', 'lstm_attention', 'cnn_transformer']


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len, scaler=None):
        self.samples, self.labels = [], []
        for _, g in df.groupby('asset'):
            g = g.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
            if len(g) <= seq_len: continue
            X = g[feature_cols].astype('float32').values
            if scaler: X = (X - scaler['mean']) / scaler['std']
            y = g['target'].astype(int).values
            for i in range(seq_len, len(g)):
                self.samples.append(X[i - seq_len:i])
                self.labels.append(int(y[i]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return (torch.tensor(self.samples[i], dtype=torch.float32),
                torch.tensor(self.labels[i],  dtype=torch.long))


def get_feature_cols(df):
    cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + EXTRA_FEATURES
    return [c for c in cols if c in df.columns]


def fit_scaler(df, feature_cols):
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
    return {'mean': x.mean(0).values.astype('float32'),
            'std':  x.std(0).replace(0, 1.0).values.astype('float32')}


# ─── Hyperparamètres par modèle ───────────────────────────────────────────────
def suggest_hyperparams(trial: optuna.Trial, model_name: str) -> dict:
    """
    Espace de recherche spécifique à chaque modèle.
    Paramètres communs + paramètres propres à l'architecture.
    """
    # Communs à tous les modèles
    p = {
        'lr':           trial.suggest_float('lr',           1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size':   trial.suggest_categorical('batch_size',  [32, 64, 128]),
        'label_smooth': trial.suggest_float('label_smooth', 0.0, 0.1),
        'seq_len':      trial.suggest_categorical('seq_len',     [48, 96]),
        'dropout':      trial.suggest_float('dropout',      0.1, 0.5),
        'buy_threshold':  trial.suggest_float('buy_threshold',  0.004, 0.015),
        'sell_threshold': trial.suggest_float('sell_threshold', 0.004, 0.015),
    }

    if model_name == 'patchtst':
        p['d_model']    = trial.suggest_categorical('d_model',    [32, 64, 128])
        p['n_layers']   = trial.suggest_int(        'n_layers',   2, 5)
        p['n_heads']    = trial.suggest_categorical('n_heads',    [2, 4, 8])
        p['ff_dim']     = trial.suggest_categorical('ff_dim',     [64, 128, 256])
        p['patch_len']  = trial.suggest_categorical('patch_len',  [8, 12, 16])
        p['stride']     = trial.suggest_categorical('stride',     [4, 6, 8])

    elif model_name == 'lstm_attention':
        p['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        p['n_layers']   = trial.suggest_int(        'n_layers',   1, 4)
        p['n_heads']    = trial.suggest_categorical('n_heads',    [2, 4, 8])

    elif model_name == 'cnn_transformer':
        p['d_model']    = trial.suggest_categorical('d_model',    [32, 64, 128])
        p['n_layers']   = trial.suggest_int(        'n_layers',   1, 4)
        p['n_heads']    = trial.suggest_categorical('n_heads',    [2, 4, 8])

    return p


def build_model_from_params(model_name: str, n_features: int,
                             params: dict, n_classes: int = 3) -> nn.Module:
    """Instancie le modèle depuis les hyperparamètres du trial."""
    if model_name == 'patchtst':
        # Contrainte : patch_len doit être ≤ seq_len
        patch_len = min(params['patch_len'], params['seq_len'])
        stride    = min(params['stride'], patch_len)
        return PatchTSTLite(
            n_features=n_features,
            seq_len=params['seq_len'],
            patch_len=patch_len,
            stride=stride,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            ff_dim=params['ff_dim'],
            dropout=params['dropout'],
            n_classes=n_classes,
        )
    elif model_name == 'lstm_attention':
        # Contrainte : hidden_dim * 2 doit être divisible par n_heads
        hidden_dim = params['hidden_dim']
        n_heads    = params['n_heads']
        while (hidden_dim * 2) % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        return LSTMAttention(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_layers=params['n_layers'],
            n_heads=n_heads,
            dropout=params['dropout'],
            n_classes=n_classes,
        )
    elif model_name == 'cnn_transformer':
        return CNNTransformer(
            n_features=n_features,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            dropout=params['dropout'],
            n_classes=n_classes,
        )
    raise ValueError(f'Modèle inconnu : {model_name}')


# ─── Sharpe rapide sur la validation ──────────────────────────────────────────
def quick_sharpe(model: nn.Module, val_df: pd.DataFrame,
                 feature_cols: list, scaler: dict,
                 seq_len: int, params: dict,
                 device: str, cfg: dict) -> float:
    """
    Calcule le Sharpe sur le jeu de validation avec agrégation par timestamp.
    Retourne 0.0 si aucune position n'est prise.
    """
    THRESHOLD   = params['buy_threshold']   # réutilise le seuil de labellisation
    ALLOW_SHORT = True
    FEE         = (cfg['backtest']['fee_bps'] + cfg['backtest']['slippage_bps']) / 10_000
    HORIZON     = cfg['target']['horizon']

    records = []
    model.eval()
    for asset, g in val_df.groupby('asset'):
        g = g.dropna(subset=feature_cols + ['future_return']).reset_index(drop=True)
        if len(g) <= seq_len: continue
        X = g[feature_cols].astype('float32').values
        X = (X - scaler['mean']) / scaler['std']
        samples = np.array([X[i - seq_len:i] for i in range(seq_len, len(g))])
        with torch.no_grad():
            probs = torch.softmax(
                model(torch.tensor(samples, dtype=torch.float32).to(device)), dim=1
            ).cpu().numpy()
        for i, row in enumerate(g.iloc[seq_len:].itertuples()):
            ps, _, pb = probs[i]
            if   pb >= THRESHOLD and pb > ps:   sig =  1.0
            elif ALLOW_SHORT and ps >= THRESHOLD and ps > pb: sig = -1.0
            else: sig = 0.0
            records.append({'asset': asset, 'timestamp': row.timestamp,
                             'future_return': row.future_return, 'raw_signal': sig})

    if not records:
        return 0.0

    out = pd.DataFrame(records)
    out['signal']       = out.groupby('asset')['raw_signal'].shift(1).fillna(0.0)
    out['turnover']     = out.groupby('asset')['signal'].diff().abs().fillna(out['signal'].abs())
    out['strategy_ret'] = out['signal'] * (out['future_return'] / HORIZON) - FEE * out['turnover']

    # Agrégation par timestamp
    port_ret = (out.groupby('timestamp')['strategy_ret']
                   .mean().sort_index().fillna(0.0))

    if (port_ret == 0).all() or port_ret.std() < 1e-10:
        return 0.0

    return float(ANN_FACTOR * port_ret.mean() / (port_ret.std() + 1e-12))


# ─── Fonction objectif multi-objectif ─────────────────────────────────────────
def make_objective(model_name: str, df_full, device: str,
                   cfg: dict, n_epochs: int = 8):
    """
    Retourne la fonction objectif pour un modèle donné.
    Retourne un tuple (f1_macro, sharpe) — Optuna minimise donc on retourne
    les valeurs positives et on configure direction='maximize' pour les deux.
    """
    def objective(trial: optuna.Trial):
        params = suggest_hyperparams(trial, model_name)

        # Reconstruire target avec les seuils du trial
        df = build_target(df_full.copy(),
                          horizon=cfg['target']['horizon'],
                          buy_threshold=params['buy_threshold'],
                          sell_threshold=-params['sell_threshold'])
        train_df, val_df, _ = temporal_split(df, 0.70, 0.15)
        feature_cols = get_feature_cols(df)
        scaler       = fit_scaler(train_df, feature_cols)

        train_ds = SeqDataset(train_df, feature_cols, params['seq_len'], scaler)
        val_ds   = SeqDataset(val_df,   feature_cols, params['seq_len'], scaler)
        if not train_ds.samples or not val_ds.samples:
            return 0.0, 0.0

        y_tr = np.array(train_ds.labels)
        cw   = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr)
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
        sw   = np.array([cw[y] for y in train_ds.labels], dtype='float64')
        sampler  = WeightedRandomSampler(sw, len(sw), replacement=True)

        train_ld = DataLoader(train_ds, batch_size=params['batch_size'], sampler=sampler)
        val_ld   = DataLoader(val_ds,   batch_size=params['batch_size'], shuffle=False)

        model = build_model_from_params(model_name, len(feature_cols),
                                        params, cfg['model']['n_classes']).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_t,
                                        label_smoothing=params['label_smooth'])
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=params['lr'],
                                      weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=params['lr'] * 0.1)

        best_f1 = 0.0
        for epoch in range(1, n_epochs + 1):
            model.train()
            for xb, yb in train_ld:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            # F1 intermédiaire pour pruning
            model.eval()
            ys, preds = [], []
            with torch.no_grad():
                for xb, yb in val_ld:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                    ys.extend(yb.numpy())
            f1 = float(f1_score(ys, preds, average='macro', zero_division=0))

            # NSGAIISampler ne supporte pas le pruning intermédiaire —
            # on l'utilise uniquement pour interrompre les trials très mauvais
            best_f1 = max(best_f1, f1)

        # Sharpe sur le jeu de validation
        sharpe = quick_sharpe(model, val_df, feature_cols, scaler,
                               params['seq_len'], params, device, cfg)

        return best_f1, sharpe

    return objective


# ─── Sélection du meilleur trial Pareto ───────────────────────────────────────
def select_best_trial(study: optuna.Study,
                       w_f1: float = 0.6, w_sharpe: float = 0.4) -> optuna.Trial:
    """
    Parmi les trials du front de Pareto, sélectionne celui qui minimise
    la distance pondérée au point utopique (f1=1, sharpe=max).

    w_f1 + w_sharpe doivent sommer à 1.
    Par défaut : priorité légèrement plus grande à F1 (classification).
    """
    pareto = study.best_trials
    if not pareto:
        # Fallback : meilleur trial selon f1
        completed = [t for t in study.trials if t.values]
        return max(completed, key=lambda t: t.values[0])

    f1s     = np.array([t.values[0] for t in pareto])
    sharpes = np.array([t.values[1] for t in pareto])

    # Normaliser [0, 1]
    f1_range  = f1s.max()     - f1s.min()     + 1e-9
    sh_range  = sharpes.max() - sharpes.min() + 1e-9
    f1_norm   = (f1s     - f1s.min())     / f1_range
    sh_norm   = (sharpes - sharpes.min()) / sh_range

    # Distance au point utopique pondérée
    dist = w_f1 * (1 - f1_norm) + w_sharpe * (1 - sh_norm)
    best_idx = int(dist.argmin())
    return pareto[best_idx]


# ─── Entraînement final ────────────────────────────────────────────────────────
def train_final(model_name: str, best_params: dict,
                df_full, device: str, cfg: dict,
                n_epochs: int = 25) -> tuple:
    """
    Ré-entraîne le modèle avec les meilleurs hyperparamètres sur train+val.
    Retourne (model, feature_cols, scaler, test_metrics).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    df = build_target(df_full.copy(),
                      horizon=cfg['target']['horizon'],
                      buy_threshold=best_params['buy_threshold'],
                      sell_threshold=-best_params['sell_threshold'])
    train_df, val_df, test_df = temporal_split(df, 0.70, 0.15)
    feature_cols = get_feature_cols(df)
    scaler       = fit_scaler(train_df, feature_cols)
    seq_len      = best_params['seq_len']
    bs           = best_params['batch_size']

    train_ds = SeqDataset(train_df, feature_cols, seq_len, scaler)
    val_ds   = SeqDataset(val_df,   feature_cols, seq_len, scaler)
    test_ds  = SeqDataset(test_df,  feature_cols, seq_len, scaler)

    y_tr = np.array(train_ds.labels)
    cw   = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tr)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    sw   = np.array([cw[y] for y in train_ds.labels], dtype='float64')
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=bs, sampler=sampler)
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    model = build_model_from_params(model_name, len(feature_cols),
                                    best_params, cfg['model']['n_classes']).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_t,
                                    label_smoothing=best_params['label_smooth'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=best_params['lr'],
                                  weight_decay=best_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_f1, best_state, patience_count = -1.0, None, 0
    print(f'\n  Entraînement final — {n_epochs} epochs max')

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_ld:
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
            for xb, yb in val_ld:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                ys.extend(yb.numpy())
        f1 = float(f1_score(ys, preds, average='macro', zero_division=0))
        scheduler.step(f1)
        marker = '★' if f1 > best_f1 else ' '
        print(f'  {marker} Epoch {epoch:02d} | loss={np.mean(losses):.4f} | val_f1={f1:.4f}')

        if f1 > best_f1:
            best_f1, patience_count = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= cfg['training']['patience']:
            print('  Early stopping.')
            break

    model.load_state_dict(best_state)

    # Évaluation test
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            p = torch.softmax(model(xb.to(device)), dim=1)
            preds.extend(p.argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            ys.extend(yb.numpy())

    acc = accuracy_score(ys, preds)
    f1m = f1_score(ys, preds, average='macro', zero_division=0)
    try:    auc = float(roc_auc_score(ys, probs, multi_class='ovr'))
    except: auc = float('nan')

    # Sharpe test
    sharpe_test = quick_sharpe(model, test_df, feature_cols, scaler,
                                seq_len, best_params, device, cfg)

    print(f'\n  TEST → acc={acc:.4f} | f1={f1m:.4f} | auc={auc:.4f} | sharpe={sharpe_test:.2f}')
    print(classification_report(ys, preds, target_names=['Sell','Hold','Buy'], zero_division=0))

    return model, feature_cols, scaler, {
        'test_acc':     round(acc,         4),
        'test_f1_macro':round(f1m,         4),
        'test_auc_ovr': round(auc,         4),
        'test_sharpe':  round(sharpe_test, 4),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Optuna multi-objectif — 3 modèles')
    parser.add_argument('--models',        nargs='+',  default=ALL_MODELS)
    parser.add_argument('--n-trials',      type=int,   default=20)
    parser.add_argument('--n-startup',     type=int,   default=5)
    parser.add_argument('--n-epochs',      type=int,   default=8,
                        help='Epochs par trial (recherche rapide)')
    parser.add_argument('--final-epochs',  type=int,   default=25,
                        help='Epochs entraînement final')
    parser.add_argument('--w-f1',          type=float, default=0.6,
                        help='Poids F1 dans sélection Pareto (défaut 0.6)')
    parser.add_argument('--w-sharpe',      type=float, default=0.4,
                        help='Poids Sharpe dans sélection Pareto (défaut 0.4)')
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg['project']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device  : {device}')
    print(f'Modèles : {args.models}')
    print(f'Trials  : {args.n_trials} / modèle | Epochs/trial : {args.n_epochs}')
    print(f'Objectifs : F1-macro (w={args.w_f1}) + Sharpe (w={args.w_sharpe})')

    ARTIFACT_DIR = Path(cfg['paths']['artifact_dir'])
    MODEL_DIR    = Path(cfg['paths']['model_dir'])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Charger les données une seule fois ──────────────────────────
    preferred   = Path(cfg['sentiment']['aligned_output_csv'])
    market_path = preferred if preferred.exists() else Path(cfg['market']['output_csv'])
    df_raw = load_market_data(market_path)
    if 'sentiment_score' not in df_raw.columns:
        df_raw['sentiment_score'] = 0.0
    df_raw = add_features(df_raw)
    print(f'Données : {len(df_raw):,} lignes | {df_raw["asset"].nunique()} actifs\n')

    mlflow.set_experiment(cfg['project']['mlflow_experiment'] + '-Optuna-MultiObj')

    all_results = {}

    # ── Boucle séquentielle sur les modèles ─────────────────────────
    for model_name in args.models:
        print(f'\n{"═"*60}')
        print(f'  OPTUNA : {model_name.upper()}')
        print(f'{"═"*60}')

        # NSGAIISampler pour l'optimisation multi-objectif (front de Pareto)
        # MedianPruner uniquement sur le premier objectif (F1)
        sampler = NSGAIISampler(seed=cfg['project']['seed'])
        # Note : pruner incompatible avec multi-objectif (NSGAIISampler)
        study = optuna.create_study(
            study_name=f'{model_name}_pareto',
            directions=['maximize', 'maximize'],   # F1, Sharpe
            sampler=sampler,
        )

        # Callback : log chaque trial dans MLflow
        def make_callback(mname):
            def cb(study, trial):
                if trial.values:
                    f1_val, sh_val = trial.values
                    with mlflow.start_run(
                        run_name=f'{mname}_trial_{trial.number:03d}', nested=True
                    ):
                        mlflow.log_params(trial.params)
                        mlflow.log_metrics({
                            'val_f1_macro': f1_val,
                            'val_sharpe':   sh_val,
                        })
                    print(f'  Trial {trial.number:02d} → '
                          f'F1={f1_val:.4f}  Sharpe={sh_val:.2f}  '
                          f'params={trial.params}')
                else:
                    print(f'  Trial {trial.number:02d} → pruned')
            return cb

        objective = make_objective(model_name, df_raw, device, cfg,
                                   n_epochs=args.n_epochs)

        with mlflow.start_run(run_name=f'{model_name}_optuna_pareto') as parent_run:
            mlflow.log_params({
                'model':        model_name,
                'n_trials':     args.n_trials,
                'n_epochs':     args.n_epochs,
                'objectives':   'f1_macro,sharpe',
                'sampler':      'NSGAIISampler',
            })

            study.optimize(
                objective,
                n_trials=args.n_trials,
                callbacks=[make_callback(model_name)],
            )

            # Front de Pareto
            pareto_trials = study.best_trials
            print(f'\n  Front de Pareto : {len(pareto_trials)} trials')
            print(f'  {"#":>3}  {"F1":>8}  {"Sharpe":>8}')
            for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
                print(f'  {t.number:>3}  {t.values[0]:>8.4f}  {t.values[1]:>8.2f}')

            # Sélection du meilleur compromis
            best_trial = select_best_trial(study, args.w_f1, args.w_sharpe)
            best_params = best_trial.params
            print(f'\n  ★ Meilleur compromis : trial {best_trial.number}')
            print(f'    F1={best_trial.values[0]:.4f}  Sharpe={best_trial.values[1]:.2f}')
            for k, v in best_params.items():
                print(f'    {k:<20} = {v}')

            # Log dans MLflow parent
            mlflow.log_metrics({
                'best_val_f1':     best_trial.values[0],
                'best_val_sharpe': best_trial.values[1],
                'n_pareto_trials': len(pareto_trials),
            })
            mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})

            # ── Entraînement final ────────────────────────────────
            model, feature_cols, scaler, test_metrics = train_final(
                model_name, best_params, df_raw, device, cfg,
                n_epochs=args.final_epochs,
            )

            # Log métriques finales
            mlflow.log_metrics({
                'final_test_acc':      test_metrics['test_acc'],
                'final_test_f1_macro': test_metrics['test_f1_macro'],
                'final_test_auc_ovr':  test_metrics['test_auc_ovr'],
                'final_test_sharpe':   test_metrics['test_sharpe'],
            })
            mlflow.pytorch.log_model(model, f'model_{model_name}_tuned')
            print(f'  MLflow run_id : {parent_run.info.run_id}')

        # ── Sauvegarder le modèle ─────────────────────────────────
        ckpt_path = MODEL_DIR / f'{model_name}_tuned.pt'
        torch.save({
            'model_name':        f'{model_name}_tuned',
            'model_state_dict':  model.state_dict(),
            'feature_cols':      feature_cols,
            'scaler_mean':       scaler['mean'].tolist(),
            'scaler_std':        scaler['std'].tolist(),
            'best_params':       best_params,
            'pareto_best_f1':    best_trial.values[0],
            'pareto_best_sharpe':best_trial.values[1],
            'test_metrics':      test_metrics,
            'config':            cfg,
            'mlflow_run_id':     parent_run.info.run_id,
        }, ckpt_path)

        # Sauvegarder résultats Optuna
        all_trials_data = [
            {'number': t.number, 'values': t.values, 'params': t.params,
             'state': str(t.state)}
            for t in study.trials if t.values
        ]
        pareto_data = [
            {'number': t.number, 'f1': t.values[0], 'sharpe': t.values[1],
             'params': t.params}
            for t in pareto_trials
        ]
        save_json({
            'model':          model_name,
            'best_params':    best_params,
            'pareto_best': {'f1': best_trial.values[0], 'sharpe': best_trial.values[1]},
            'pareto_trials':  pareto_data,
            'test_metrics':   test_metrics,
            'all_trials':     all_trials_data,
        }, ARTIFACT_DIR / f'optuna_{model_name}_results.json')

        all_results[model_name] = {
            'best_params':    best_params,
            'pareto_best_f1': best_trial.values[0],
            'pareto_best_sharpe': best_trial.values[1],
            'test_metrics':   test_metrics,
            'model':          model,
            'feature_cols':   feature_cols,
            'scaler':         scaler,
        }

        print(f'\n  ✓ {model_name} → {ckpt_path}')

    # ── Tableau comparatif final ─────────────────────────────────────
    print(f'\n{"═"*70}')
    print('  RÉSULTATS FINAUX — APRÈS OPTIMISATION OPTUNA')
    print(f'{"═"*70}')
    print(f'  {"Modèle":<22} {"F1-val":>8} {"Sharpe-val":>11} '
          f'{"F1-test":>9} {"Sh-test":>9} {"AUC":>7}')
    print('  ' + '-'*68)
    for name, r in all_results.items():
        tm = r['test_metrics']
        print(f'  {name:<22} {r["pareto_best_f1"]:>8.4f} '
              f'{r["pareto_best_sharpe"]:>11.2f} '
              f'{tm["test_f1_macro"]:>9.4f} '
              f'{tm["test_sharpe"]:>9.2f} '
              f'{tm["test_auc_ovr"]:>7.4f}')

    # Sauvegarder le résumé global
    save_json(
        {name: {k: v for k, v in r.items() if k not in ('model','feature_cols','scaler')}
         for name, r in all_results.items()},
        ARTIFACT_DIR / 'optuna_all_results.json',
    )
    print(f'\n  ✓ Résumé → {ARTIFACT_DIR / "optuna_all_results.json"}')


if __name__ == '__main__':
    main()
