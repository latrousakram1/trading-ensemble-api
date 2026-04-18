"""
ensemble_final.py
==================
Construit, évalue et sauvegarde l'ensemble pondéré final :
    lstm_attention + cnn_transformer + patchtst
    (tft_lite exclu — F1=0.40, sous-convergence)

Poids : proportionnels au F1-macro de chaque modèle sur le test set.

Usage
-----
    python src/ensemble_final.py
    python src/ensemble_final.py --weights 0.45 0.35 0.20
    python src/ensemble_final.py --threshold 0.50
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # pas d'affichage GUI en mode script

import mlflow
import mlflow.pytorch
try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.advanced_models import build_model
    from src.metrics import (compute_portfolio_metrics, compute_asset_metrics,
                             compute_reliable_metrics, period_return_from_horizon)
    from src.train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from advanced_models import build_model
    from metrics import (compute_portfolio_metrics, compute_asset_metrics,
                         compute_reliable_metrics, period_return_from_horizon)
    from train_utils import get_feature_cols, fit_scaler, SeqDataset, evaluate

FEATURE_SUFFIXES = ('ret_', 'sma_', 'ema_', 'rsi_', 'vol_', 'zscore_')
EXTRA_FEATURES   = ['range_pct', 'body_pct', 'volume_zscore_20',
                    'hour', 'dayofweek', 'month', 'sentiment_score']

# Modèles retenus (TFT exclu)
ENSEMBLE_MEMBERS = ['lstm_attention', 'cnn_transformer', 'patchtst']


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_feature_cols(df):
    cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + EXTRA_FEATURES
    return [c for c in cols if c in df.columns]


def load_member(model_name: str, cfg: dict, device: str):
    """Charge un checkpoint et reconstruit le modèle."""
    ckpt_path = None
    for suffix in ('_tuned.pt', '_final.pt'):
        candidate = Path(cfg['paths']['model_dir']) / f'{model_name}{suffix}'
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint introuvable pour '{model_name}'. "
            "Lancez d'abord : python src/train_advanced.py --model " + model_name
        )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    feature_cols = ckpt['feature_cols']
    scaler = {
        'mean': np.array(ckpt['scaler_mean'], dtype='float32'),
        'std':  np.array(ckpt['scaler_std'],  dtype='float32'),
    }
    scaler['std'][scaler['std'] == 0] = 1.0
    saved_name = ckpt.get('model_name', model_name)
    model = build_model(saved_name, len(feature_cols), cfg['model'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    return model, feature_cols, scaler


def build_sequences(df, feature_cols, seq_len, scaler):
    """Retourne (X, meta) pour tous les actifs."""
    samples, meta = [], []
    for asset, g in df.groupby('asset'):
        g = g.dropna(subset=feature_cols + ['future_return']).reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        X = g[feature_cols].astype('float32').values
        X = (X - scaler['mean']) / scaler['std']
        for i in range(seq_len, len(g)):
            samples.append(X[i - seq_len:i])
            meta.append({
                'asset':         asset,
                'timestamp':     g.loc[i, 'timestamp'],
                'close':         float(g.loc[i, 'close']),
                'future_return': float(g.loc[i, 'future_return']),
            })
    if not samples:
        return np.empty((0, seq_len, len(feature_cols)), dtype='float32'), []
    return np.array(samples, dtype='float32'), meta


def compute_portfolio_metrics(port_ret: pd.Series,
                               exposure: pd.Series | None = None) -> dict:
    """Métriques portefeuille sur rendements agrégés par timestamp."""
    port_ret = port_ret.fillna(0.0)
    ANN = np.sqrt(365 * 24)
    equity  = (1.0 + port_ret).cumprod()
    n       = max(len(port_ret), 1)
    ppy     = ANN ** 2
    total   = float(equity.iloc[-1] - 1.0)
    annual  = float((1 + total) ** (ppy / n) - 1)
    sharpe  = float(ANN * port_ret.mean() / (port_ret.std() + 1e-12))
    max_dd  = float((equity / equity.cummax() - 1.0).min())
    calmar  = annual / (abs(max_dd) + 1e-12)
    vol     = float(port_ret.std() * ANN)
    active  = port_ret[port_ret != 0]
    wr      = float((active > 0).mean()) if len(active) else 0.0
    exp     = float(exposure.mean()) if exposure is not None \
              else float((port_ret != 0).mean())
    return {
        'total_return':  round(total,  4),
        'annual_return': round(annual, 4),
        'sharpe':        round(sharpe, 4),
        'calmar':        round(calmar, 4),
        'max_drawdown':  round(max_dd, 4),
        'annual_vol':    round(vol,    4),
        'win_rate':      round(wr,     4),
        'exposure':      round(exp,    4),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Ensemble final pondéré')
    parser.add_argument('--members',   nargs='+', default=ENSEMBLE_MEMBERS,
                        help='Noms des modèles membres')
    parser.add_argument('--weights',   nargs='+', type=float, default=None,
                        help='Poids manuels (doit correspondre à --members)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Seuil de décision (défaut : config.yaml)')
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg['project']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # ── MLflow experiment ──────────────────────────────────────────
    mlflow.set_experiment(cfg['project']['mlflow_experiment'] + '-Ensemble')
    THRESHOLD   = args.threshold or float(cfg['backtest']['probability_threshold'])
    HOLD_THRESHOLD = float(cfg['backtest'].get('hold_probability_threshold', 0.0))
    ALLOW_SHORT = bool(cfg['backtest'].get('allow_short', True))
    FEE         = (cfg['backtest']['fee_bps'] + cfg['backtest']['slippage_bps']) / 10_000.0
    HORIZON     = cfg['target']['horizon']
    SEQ_LEN     = cfg['model']['seq_len']
    ARTIFACT_DIR = Path(cfg['paths']['artifact_dir'])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Charger les données ─────────────────────────────────────────
    preferred   = Path(cfg['sentiment']['aligned_output_csv'])
    market_path = preferred if preferred.exists() else Path(cfg['market']['output_csv'])
    df = load_market_data(market_path)
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
    df = add_features(df)
    df = build_target(df, horizon=HORIZON,
                      buy_threshold=cfg['target']['buy_threshold'],
                      sell_threshold=cfg['target']['sell_threshold'])
    _, _, test_df = temporal_split(df, 0.70, 0.15)
    print(f'Test set : {len(test_df):,} lignes')

    # ── 2. Charger les modèles membres ────────────────────────────────
    members = args.members
    models, feature_cols_per_model, scalers = {}, {}, {}
    for name in members:
        try:
            m, fc, sc = load_member(name, cfg, device)
            models[name]            = m
            feature_cols_per_model[name] = fc
            scalers[name]           = sc
            print(f'  ✓ {name} chargé')
        except FileNotFoundError as e:
            print(f'  ✗ {name} : {e}')

    if not models:
        raise RuntimeError('Aucun modèle chargé. Lancez train_advanced.py --model all')

    available = list(models.keys())

    # ── 3. Inférence par modèle ────────────────────────────────────────
    all_probs    = {}   # name → (meta, probs)
    common_meta  = None

    for name in available:
        fc = feature_cols_per_model[name]
        sc = scalers[name]
        X, meta = build_sequences(test_df, fc, SEQ_LEN, sc)
        if len(X) == 0:
            print(f'  ✗ {name} : aucune séquence')
            continue
        with torch.no_grad():
            probs = torch.softmax(
                models[name](torch.tensor(X, dtype=torch.float32).to(device)), dim=1
            ).cpu().numpy()
        all_probs[name] = probs
        common_meta = meta
        print(f'  {name} : {probs.shape}')

    if not all_probs:
        raise RuntimeError('Aucune prédiction générée.')

    # ── 4. Poids ──────────────────────────────────────────────────────
    if args.weights:
        assert len(args.weights) == len(available), "Longueur --weights != nb modèles"
        raw_w = {n: w for n, w in zip(available, args.weights)}
    else:
        # Charger les F1-macro depuis les fichiers de métriques
        f1_map = {}
        for name in available:
            mpath = ARTIFACT_DIR / f'{name}_metrics.json'
            if mpath.exists():
                import json
                d = json.loads(mpath.read_text())
                f1_map[name] = d.get('test_f1_macro', 1.0)
            else:
                f1_map[name] = 1.0   # poids égal si métriques absentes
        total_f1 = sum(f1_map.values())
        raw_w = {n: f / total_f1 for n, f in f1_map.items()}

    # Normaliser
    s = sum(raw_w.values())
    WEIGHTS = {n: round(w / s, 4) for n, w in raw_w.items()}
    print('\nPoids de l\'ensemble :')
    for n, w in WEIGHTS.items():
        print(f'  {n:<22} = {w:.4f}')

    # ── 5. Combinaison pondérée ────────────────────────────────────────
    weighted_probs = sum(WEIGHTS[n] * all_probs[n] for n in available)

    # ── 6. Métriques de classification ────────────────────────────────
    # Reconstruire les vraies étiquettes dans le même ordre que meta
    feature_cols = feature_cols_per_model[available[0]]
    scaler_ref   = scalers[available[0]]
    true_labels  = []
    fc_ref       = feature_cols
    for asset, g in test_df.groupby('asset'):
        g = g.dropna(subset=fc_ref + ['target', 'future_return']).reset_index(drop=True)
        if len(g) <= SEQ_LEN:
            continue
        true_labels.extend(g['target'].astype(int).values[SEQ_LEN:])
    true_labels = np.array(true_labels)

    ens_preds = weighted_probs.argmax(axis=1)
    acc = accuracy_score(true_labels, ens_preds)
    f1m = f1_score(true_labels, ens_preds, average='macro', zero_division=0)
    try:    auc = float(roc_auc_score(true_labels, weighted_probs, multi_class='ovr'))
    except: auc = float('nan')

    print(f'\n=== CLASSIFICATION (test set) ===')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  F1-macro : {f1m:.4f}')
    print(f'  AUC-OVR  : {auc:.4f}')
    print()
    print(classification_report(true_labels, ens_preds,
                                  target_names=['Sell', 'Hold', 'Buy'], zero_division=0))

    # ── 7. Backtest portefeuille ───────────────────────────────────────
    meta_df = pd.DataFrame(common_meta)
    meta_df['prob_sell'] = weighted_probs[:, 0]
    meta_df['prob_hold'] = weighted_probs[:, 1]
    meta_df['prob_buy']  = weighted_probs[:, 2]

    def decide(row):
        if row['prob_hold'] >= HOLD_THRESHOLD:
            return 0.0
        if row['prob_buy'] >= THRESHOLD and row['prob_buy'] > row['prob_sell']:
            return 1.0
        if ALLOW_SHORT and row['prob_sell'] >= THRESHOLD and row['prob_sell'] > row['prob_buy']:
            return -1.0
        return 0.0

    meta_df['raw_signal']  = meta_df.apply(decide, axis=1)
    meta_df['signal']      = meta_df.groupby('asset')['raw_signal'].shift(1).fillna(0.0)
    meta_df['turnover']    = meta_df.groupby('asset')['signal'].diff().abs() \
                              .fillna(meta_df['signal'].abs())
    meta_df['period_ret']  = meta_df['future_return'] / HORIZON
    meta_df['strategy_ret']= meta_df['signal'] * meta_df['period_ret'] \
                              - FEE * meta_df['turnover']

    # Agrégation par timestamp
    meta_df = meta_df.sort_values(['timestamp', 'asset'])
    portfolio = (
        meta_df.groupby('timestamp', as_index=True)
               .agg(strategy_ret=('strategy_ret', 'mean'),
                    exposure=('signal', lambda x: float((x != 0).mean())))
               .sort_index()
    )
    port_ret    = portfolio['strategy_ret'].fillna(0.0)
    bt_metrics  = compute_portfolio_metrics(port_ret, portfolio['exposure'])
    equity_ens  = (1.0 + port_ret).cumprod()

    print(f'=== BACKTEST (portefeuille égal-pondéré) ===')
    for k, v in bt_metrics.items():
        print(f'  {k:<20} : {v:.2%}' if abs(v) < 10 else f'  {k:<20} : {v:.2f}')

    # Métriques par actif
    print('\n  Par actif :')
    for asset, grp in meta_df.groupby('asset'):
        r  = grp['strategy_ret'].fillna(0.0)
        eq = (1.0 + r).cumprod()
        tot= float(eq.iloc[-1] - 1.0)
        sh = float(np.sqrt(365*24) * r.mean() / (r.std() + 1e-12))
        wr = float((r[r!=0] > 0).mean()) if (r!=0).any() else 0.0
        print(f'    {asset}: ret={tot:.2%}  sharpe={sh:.2f}  wr={wr:.2%}')

    # ── 8. Matrice de confusion ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm   = confusion_matrix(true_labels, ens_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Sell', 'Hold', 'Buy'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'Ensemble pondéré\nF1={f1m:.4f}  AUC={auc:.4f}', fontweight='bold')

    dd = equity_ens / equity_ens.cummax() - 1.0
    axes[1].fill_between(range(len(dd)), dd.values, 0, color='#e74c3c', alpha=0.4)
    axes[1].plot(dd.values, color='#e74c3c', linewidth=1)
    axes[1].axhline(bt_metrics['max_drawdown'], color='darkred', linestyle='--',
                    label=f"Max DD={bt_metrics['max_drawdown']:.2%}")
    axes[1].set_title('Drawdown — Ensemble pondéré', fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(ARTIFACT_DIR / 'ensemble_cm_drawdown.png', bbox_inches='tight')
    print(f'\n  Figure → {ARTIFACT_DIR / "ensemble_cm_drawdown.png"}')

    # Equity curve
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(equity_ens.values, color='#2ecc71', linewidth=2,
             label=f"Ensemble ({bt_metrics['total_return']:.1%})")
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Equity curve — Ensemble pondéré final', fontweight='bold')
    ax2.set_xlabel('Pas de temps (heures agrégées)')
    ax2.set_ylabel('Equity (base 1.0)')
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(ARTIFACT_DIR / 'ensemble_equity.png', bbox_inches='tight')
    print(f'  Figure → {ARTIFACT_DIR / "ensemble_equity.png"}')

    # ── 9. Logger dans MLflow et sauvegarder le résumé final ──────────
    with mlflow.start_run(run_name='ensemble_weighted_final') as run:
        mlflow.log_params({
            'members':      str(available),
            'weights':      str(WEIGHTS),
            'threshold':    THRESHOLD,
            'fee_bps':      cfg['backtest']['fee_bps'],
            'slippage_bps': cfg['backtest']['slippage_bps'],
            'horizon':      HORIZON,
        })
        mlflow.log_metrics({
            'ens_accuracy': round(acc, 4),
            'ens_f1_macro': round(f1m, 4),
            'ens_auc_ovr':  round(auc, 4),
        })
        for k, v in bt_metrics.items():
            mlflow.log_metric(f'bt_{k}', v)
        print(f'  MLflow run_id : {run.info.run_id}')

    summary = {
        'best_final_model': {
            'name':         'ensemble_weighted',
            'models_used':  available,
            'weights':      WEIGHTS,
            'threshold':    THRESHOLD,
            'acc':          round(acc, 4),
            'f1_macro':     round(f1m, 4),
            'auc_ovr':      round(auc, 4),
            'backtest':     bt_metrics,
        },
        'excluded_models': [
            {'name': 'tft_lite', 'reason': 'F1-macro=0.40 — sous-convergence'}
        ],
    }
    save_json(summary, ARTIFACT_DIR / 'final_summary.json')
    print(f'\n✓ Résumé final → {ARTIFACT_DIR / "final_summary.json"}')

    # Sauvegarder les prédictions de l'ensemble
    out_path = ARTIFACT_DIR / 'ensemble_predictions.csv'
    meta_df.to_csv(out_path, index=False)
    print(f'✓ Prédictions   → {out_path}')


if __name__ == '__main__':
    main()
