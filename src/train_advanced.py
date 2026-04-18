"""
train_advanced.py
=================
Entraînement des 4 modèles + ensemble avec backtest intégré.

Utilise train_utils.py pour éviter la duplication de code.
Corrections :
  - class_weight_multipliers depuis config (boost HOLD)
  - period_ret géométrique dans run_backtest_portfolio
  - metrics depuis metrics.py (source unique)

Usage
-----
    python src/train_advanced.py --model lstm_attention
    python src/train_advanced.py --model all
    python src/train_advanced.py --model patchtst
"""
from __future__ import annotations
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from src.utils import load_config, set_seed, save_json
    from src.data import load_market_data, temporal_split
    from src.features import add_features
    from src.targets import build_target
    from src.advanced_models import build_model, MODEL_REGISTRY
    from src.metrics import (
        compute_portfolio_metrics, compute_asset_metrics,
        compute_reliable_metrics, period_return_from_horizon,
    )
    from src.train_utils import (
        SeqDataset, fit_scaler, get_feature_cols,
        make_loaders, evaluate, build_scheduler, run_epoch,
    )
except ImportError:
    from utils import load_config, set_seed, save_json
    from data import load_market_data, temporal_split
    from features import add_features
    from targets import build_target
    from advanced_models import build_model, MODEL_REGISTRY
    from metrics import (
        compute_portfolio_metrics, compute_asset_metrics,
        compute_reliable_metrics, period_return_from_horizon,
    )
    from train_utils import (
        SeqDataset, fit_scaler, get_feature_cols,
        make_loaders, evaluate, build_scheduler, run_epoch,
    )

ALL_MODELS              = [k for k in MODEL_REGISTRY if k != "ensemble"]
ALL_MODELS_WITH_ENSEMBLE = list(MODEL_REGISTRY.keys())


# ─── Backtest portefeuille ─────────────────────────────────────────────────────
def run_backtest_portfolio(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: dict,
    cfg: dict,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """
    Backtest multi-actifs — portefeuille égal-pondéré.
    Retour géométrique horaire (non linéaire).
    """
    THRESHOLD      = float(cfg["backtest"]["probability_threshold"])
    HOLD_THRESHOLD = float(cfg["backtest"].get("hold_probability_threshold", 0.0))
    ALLOW_SHORT    = bool(cfg["backtest"].get("allow_short", True))
    FEE            = (cfg["backtest"]["fee_bps"] + cfg["backtest"]["slippage_bps"]) / 10_000.0
    SEQ_LEN        = cfg["model"]["seq_len"]
    HORIZON        = cfg["target"]["horizon"]

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
            if p_hold >= HOLD_THRESHOLD > 0:
                sig = 0.0
            elif p_buy >= THRESHOLD and p_buy > p_sell:
                sig = 1.0
            elif ALLOW_SHORT and p_sell >= THRESHOLD and p_sell > p_buy:
                sig = -1.0
            else:
                sig = 0.0
            records.append({
                "asset": asset, "timestamp": row.timestamp,
                "close": float(row.close), "future_return": float(row.future_return),
                "prob_sell": float(p_sell), "prob_hold": float(p_hold),
                "prob_buy": float(p_buy), "raw_signal": sig,
            })

    out = pd.DataFrame(records).sort_values(["timestamp", "asset"]).copy()
    out["signal"]       = out.groupby("asset")["raw_signal"].shift(1).fillna(0.0)
    out["turnover"]     = out.groupby("asset")["signal"].diff().abs().fillna(out["signal"].abs())
    out["period_ret"]   = period_return_from_horizon(out["future_return"], HORIZON)
    out["strategy_ret"] = out["signal"] * out["period_ret"] - FEE * out["turnover"]

    portfolio = (
        out.groupby("timestamp", as_index=True)
           .agg(strategy_ret=("strategy_ret", "mean"),
                exposure=("signal", lambda x: float((x != 0).mean())))
           .sort_index()
    )
    portfolio_ret = portfolio["strategy_ret"].fillna(0.0)
    global_metrics = compute_portfolio_metrics(portfolio_ret, portfolio["exposure"])
    per_asset = {}
    for asset, grp in out.groupby("asset"):
        per_asset[asset] = compute_asset_metrics(grp["strategy_ret"].fillna(0.0))
    global_metrics["per_asset"] = per_asset
    return out, portfolio, portfolio_ret, global_metrics


# ─── Entraînement d'un modèle ──────────────────────────────────────────────────
def train_one_model(
    model_name: str,
    cfg: dict,
    train_loader, val_loader, test_loader,
    feature_cols: list[str],
    scaler: dict,
    device: str,
    class_weights: torch.Tensor,
) -> tuple[dict, torch.nn.Module]:
    print(f"\n{'='*60}\n  Entraînement : {model_name.upper()}\n{'='*60}")

    model = build_model(model_name, len(feature_cols), cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres : {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg["training"]["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler, scheduler_type = build_scheduler(optimizer, cfg, len(train_loader))

    mlflow.set_experiment(cfg["project"]["mlflow_experiment"])
    best_f1, patience_count, best_state = -1.0, 0, None

    with mlflow.start_run(run_name=f"{model_name}_advanced"):
        mlflow.log_params({"model": model_name, "n_params": n_params,
                           "lr": cfg["training"]["lr"], "n_features": len(feature_cols)})

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss = run_epoch(
                model, train_loader, criterion, optimizer, device,
                cfg["training"]["grad_clip"], scheduler, scheduler_type,
            )
            vm = evaluate(model, val_loader, device)
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(vm["f1_macro"])
            elif scheduler_type == "cosine":
                scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "val_acc": vm["acc"],
                "val_f1_macro": vm["f1_macro"], "val_auc_ovr": vm["auc_ovr"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            print(f"  Epoch {epoch:02d} | loss={train_loss:.4f} | "
                  f"acc={vm['acc']:.4f} | f1={vm['f1_macro']:.4f} | "
                  f"auc={vm['auc_ovr']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

            if vm["f1_macro"] > best_f1:
                best_f1 = vm["f1_macro"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= cfg["training"]["patience"]:
                print("  Early stopping.")
                break

        if best_state:
            model.load_state_dict(best_state)

        test_m = evaluate(model, test_loader, device)
        mlflow.log_metrics({"test_acc": test_m["acc"], "test_f1_macro": test_m["f1_macro"],
                             "test_auc_ovr": test_m["auc_ovr"]})
        print(f"  TEST → acc={test_m['acc']:.4f} | f1={test_m['f1_macro']:.4f} | "
              f"auc={test_m['auc_ovr']:.4f}")

        out_model = Path(cfg["paths"]["model_dir"]) / f"{model_name}_final.pt"
        out_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler_mean": scaler["mean"].tolist(),
            "scaler_std":  scaler["std"].tolist(),
            "config": cfg,
            "n_params": n_params,
        }, out_model)

        summary = {
            "model": model_name, "n_params": n_params,
            "test_acc": test_m["acc"], "test_f1_macro": test_m["f1_macro"],
            "test_auc_ovr": test_m["auc_ovr"],
        }
        save_json(summary, Path(cfg["paths"]["artifact_dir"]) / f"{model_name}_metrics.json")
        return summary, model


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm_attention",
                        choices=ALL_MODELS_WITH_ENSEMBLE + ["all"])
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    preferred   = Path(cfg["sentiment"]["aligned_output_csv"])
    market_path = preferred if preferred.exists() else Path(cfg["market"]["output_csv"])
    df = load_market_data(market_path)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df = add_features(df)
    df = build_target(df, horizon=cfg["target"]["horizon"],
                      buy_threshold=cfg["target"]["buy_threshold"],
                      sell_threshold=cfg["target"]["sell_threshold"])

    train_df, val_df, test_df = temporal_split(df, 0.70, 0.15)
    feature_cols = get_feature_cols(df)
    scaler       = fit_scaler(train_df, feature_cols)
    seq_len      = cfg["model"]["seq_len"]

    train_ds = SeqDataset(train_df, feature_cols, seq_len, scaler)
    val_ds   = SeqDataset(val_df,   feature_cols, seq_len, scaler)
    test_ds  = SeqDataset(test_df,  feature_cols, seq_len, scaler)

    multipliers = cfg["training"].get("class_weight_multipliers", [1.0, 1.8, 1.0])
    train_loader, val_loader, test_loader, class_weights = make_loaders(
        train_ds, val_ds, test_ds, cfg["training"]["batch_size"], multipliers
    )

    models_to_train = ALL_MODELS_WITH_ENSEMBLE if args.model == "all" else [args.model]
    all_results = []

    for name in models_to_train:
        summary, trained_model = train_one_model(
            name, cfg, train_loader, val_loader, test_loader,
            feature_cols, scaler, device, class_weights,
        )
        _, _, portfolio_ret, bt = run_backtest_portfolio(
            trained_model, test_df, feature_cols, scaler, cfg, device
        )
        reliable = compute_reliable_metrics(portfolio_ret)
        summary["backtest"]          = {k: v for k, v in bt.items() if k != "per_asset"}
        summary["reliable_backtest"] = reliable
        all_results.append(summary)

        print(f"\n  BACKTEST :")
        print(f"    Sharpe  : {bt['sharpe']:.2f}")
        print(f"    Max DD  : {bt['max_drawdown']:.2%}")
        print(f"    Win Rate: {bt['win_rate']:.2%}")

    if len(all_results) > 1:
        save_json({"results": all_results},
                  Path(cfg["paths"]["artifact_dir"]) / "models_comparison.json")


if __name__ == "__main__":
    main()
