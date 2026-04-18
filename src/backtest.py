"""
backtest.py
===========
Backtest multi-actifs avec portefeuille égal-pondéré.

Corrections appliquées :
  - Agrégation par timestamp avant l'equity curve (bug compounding ×4 résolu)
  - Retour horaire via formule géométrique : (1+r)^(1/h)-1  (non linéaire)
  - Métriques depuis metrics.py (source unique de vérité)
  - Import try/except pour compatibilité Colab et module

Usage
-----
    python src/backtest.py                        # PatchTST par défaut
    python src/backtest.py --model lstm_attention
    python src/backtest.py --model ensemble       # ensemble pondéré
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from src.utils import load_config, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.model import PatchTSTLite
    from src.advanced_models import build_model
    from src.metrics import (
        compute_portfolio_metrics,
        compute_asset_metrics,
        compute_reliable_metrics,
        period_return_from_horizon,
    )
    from src.train_utils import get_feature_cols
except ImportError:
    from utils import load_config, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from model import PatchTSTLite
    from advanced_models import build_model
    from metrics import (
        compute_portfolio_metrics,
        compute_asset_metrics,
        compute_reliable_metrics,
        period_return_from_horizon,
    )
    from train_utils import get_feature_cols


def _load_checkpoint(model_name: str, cfg: dict) -> tuple:
    """Charge un checkpoint. Priorité _tuned.pt > _final.pt."""
    model_dir = Path(cfg["paths"]["model_dir"])
    ckpt_path = None
    for suffix in ("_tuned.pt", "_final.pt"):
        candidate = model_dir / f"{model_name}{suffix}"
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint introuvable pour '{model_name}' dans {model_dir}. "
            "Lancez d'abord : python src/train_advanced.py --model " + model_name
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feature_cols = ckpt["feature_cols"]
    scaler = {
        "mean": np.array(ckpt["scaler_mean"], dtype="float32"),
        "std":  np.array(ckpt["scaler_std"],  dtype="float32"),
    }
    scaler["std"][scaler["std"] == 0] = 1.0

    import inspect
    saved_name = ckpt.get("model_name", model_name).replace("_tuned", "").replace("_final", "")
    if saved_name == "patchtst":
        valid_keys = set(inspect.signature(PatchTSTLite.__init__).parameters) - {"self"}
        model_cfg  = {k: v for k, v in cfg["model"].items() if k in valid_keys}
        model = PatchTSTLite(n_features=len(feature_cols), **model_cfg)
    else:
        model = build_model(saved_name, len(feature_cols), cfg["model"])

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, feature_cols, scaler


def run_backtest(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: dict,
    cfg: dict,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """
    Backtest multi-actifs avec portefeuille égal-pondéré.

    Retourne
    --------
    out          : DataFrame par actif × timestamp avec signaux et rendements
    portfolio    : DataFrame agrégé par timestamp (strategy_ret, exposure)
    portfolio_ret: Series des rendements horaires du portefeuille
    metrics      : dict des métriques (portefeuille + par actif)
    """
    THRESHOLD        = float(cfg["backtest"]["probability_threshold"])
    HOLD_THRESHOLD   = float(cfg["backtest"].get("hold_probability_threshold", 0.0))
    ALLOW_SHORT      = bool(cfg["backtest"].get("allow_short", True))
    FEE              = (cfg["backtest"]["fee_bps"] + cfg["backtest"]["slippage_bps"]) / 10_000.0
    SEQ_LEN          = cfg["model"]["seq_len"]
    HORIZON          = cfg["target"]["horizon"]

    records = []
    model.eval()

    for asset, g in test_df.groupby("asset"):
        g = g.dropna(subset=feature_cols + ["future_return"]).reset_index(drop=True)
        if len(g) <= SEQ_LEN:
            continue
        X = g[feature_cols].astype("float32").values
        std = scaler["std"].copy(); std[std == 0] = 1.0
        X = (X - scaler["mean"]) / std

        samples = np.array([X[i - SEQ_LEN : i] for i in range(SEQ_LEN, len(g))])
        with torch.no_grad():
            probs = torch.softmax(
                model(torch.tensor(samples, dtype=torch.float32).to(device)), dim=1
            ).cpu().numpy()

        for i, row in enumerate(g.iloc[SEQ_LEN:].itertuples()):
            p_sell, p_hold, p_buy = probs[i]
            # Filtre HOLD explicite si configuré
            if p_hold >= HOLD_THRESHOLD > 0:
                sig = 0.0
            elif p_buy >= THRESHOLD and p_buy > p_sell:
                sig = 1.0
            elif ALLOW_SHORT and p_sell >= THRESHOLD and p_sell > p_buy:
                sig = -1.0
            else:
                sig = 0.0

            records.append({
                "asset":         asset,
                "timestamp":     row.timestamp,
                "close":         float(row.close),
                "future_return": float(row.future_return),
                "prob_sell":     float(p_sell),
                "prob_hold":     float(p_hold),
                "prob_buy":      float(p_buy),
                "raw_signal":    sig,
            })

    if not records:
        raise ValueError("Aucune séquence construite — vérifiez les données et features.")

    out = pd.DataFrame(records)
    out = out.sort_values(["timestamp", "asset"]).copy()

    # Décalage no-look-ahead par actif
    out["signal"]   = out.groupby("asset")["raw_signal"].shift(1).fillna(0.0)
    out["turnover"] = out.groupby("asset")["signal"].diff().abs().fillna(out["signal"].abs())

    # ── Retour horaire : formule géométrique ──
    out["period_ret"]   = period_return_from_horizon(out["future_return"], HORIZON)
    out["strategy_ret"] = out["signal"] * out["period_ret"] - FEE * out["turnover"]

    # ── Agrégation portefeuille ────────────────────────────────────
    portfolio = (
        out.groupby("timestamp", as_index=True)
           .agg(
               strategy_ret=("strategy_ret", "mean"),
               exposure=("signal", lambda x: float((x != 0).mean())),
           )
           .sort_index()
    )

    portfolio_ret      = portfolio["strategy_ret"].fillna(0.0)
    portfolio_exposure = portfolio["exposure"]

    # Métriques globales
    global_metrics = compute_portfolio_metrics(portfolio_ret, portfolio_exposure)

    # Métriques par actif
    per_asset = {}
    for asset, grp in out.groupby("asset"):
        per_asset[asset] = compute_asset_metrics(grp["strategy_ret"].fillna(0.0))
    global_metrics["per_asset"] = per_asset

    return out, portfolio, portfolio_ret, global_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest multi-actifs — portefeuille égal-pondéré")
    parser.add_argument("--model", default="patchtst",
                        help="Nom du modèle (patchtst, lstm_attention, cnn_transformer)")
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preferred   = Path(cfg["sentiment"]["aligned_output_csv"])
    market_path = preferred if preferred.exists() else Path(cfg["market"]["output_csv"])
    df = load_market_data(market_path)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df = add_features(df)
    df = build_target(
        df,
        horizon=cfg["target"]["horizon"],
        buy_threshold=cfg["target"]["buy_threshold"],
        sell_threshold=cfg["target"]["sell_threshold"],
    )
    _, _, test_df = temporal_split(df, 0.70, 0.15)

    model, feature_cols, scaler = _load_checkpoint(args.model, cfg)
    model = model.to(device)

    out, portfolio, portfolio_ret, metrics = run_backtest(
        model, test_df, feature_cols, scaler, cfg, device
    )
    reliable = compute_reliable_metrics(portfolio_ret, portfolio["exposure"])

    # Sauvegardes
    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(artifact_dir / "backtest_predictions.csv", index=False)
    portfolio.to_csv(artifact_dir / "portfolio_equity.csv")
    save_json(metrics,   artifact_dir / "backtest_metrics.json")
    save_json(reliable,  artifact_dir / "backtest_reliable_metrics.json")

    print("\n=== BACKTEST — Portefeuille égal-pondéré ===")
    print(f"  Sharpe      : {metrics['sharpe']:.2f}")
    print(f"  Max DD      : {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate    : {metrics['win_rate']:.2%}")
    print(f"  Exposition  : {metrics['exposure']:.2%}")
    print(f"  Retour tot. : {metrics['total_return']:.2%}")
    print(f"\n  [Note] CAGR={metrics['annual_return']:.0%} — non interprétable sur {len(portfolio)} heures.")
    print(f"\n  Par actif :")
    for asset, m in metrics["per_asset"].items():
        print(f"    {asset}: sharpe={m['sharpe']:.2f}  maxdd={m['max_drawdown']:.2%}  wr={m['win_rate']:.2%}")
    print(f"\n  Prédictions → {artifact_dir/'backtest_predictions.csv'}")
    print(f"  Equity      → {artifact_dir/'portfolio_equity.csv'}")


if __name__ == "__main__":
    main()
