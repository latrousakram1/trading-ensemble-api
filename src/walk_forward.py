from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from src.advanced_models import MODEL_REGISTRY, build_model
    from src.data import load_market_data
    from src.features import add_features
    from src.metrics import compute_portfolio_metrics, compute_reliable_metrics
    from src.targets import build_target
    from src.train_advanced import run_backtest_portfolio
    from src.train_utils import SeqDataset, fit_scaler, get_feature_cols, evaluate, make_loaders, run_epoch
    from src.utils import load_config, save_json, set_seed
except ImportError:
    from advanced_models import MODEL_REGISTRY, build_model
    from data import load_market_data
    from features import add_features
    from metrics import compute_portfolio_metrics, compute_reliable_metrics
    from targets import build_target
    from train_advanced import run_backtest_portfolio
    from train_utils import SeqDataset, fit_scaler, get_feature_cols, evaluate, make_loaders, run_epoch
    from utils import load_config, save_json, set_seed


def monthly_walk_forward_splits(
    df: pd.DataFrame,
    train_min_months: int = 3,
    test_months: int = 1,
    max_windows: int = 6,
) -> list[dict]:
    months = sorted(df["timestamp"].dt.to_period("M").astype(str).unique().tolist())
    windows: list[dict] = []

    for test_start_idx in range(train_min_months, len(months), test_months):
        test_month_slice = months[test_start_idx:test_start_idx + test_months]
        if len(test_month_slice) < test_months:
            break

        train_months = months[:test_start_idx]
        val_month = train_months[-1]
        fit_months = train_months[:-1]
        if not fit_months:
            continue

        windows.append(
            {
                "fit_months": fit_months,
                "val_month": val_month,
                "test_months": test_month_slice,
            }
        )
        if len(windows) >= max_windows:
            break

    return windows


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.extend(prob.tolist())
            preds.extend(prob.argmax(axis=1).tolist())
            ys.extend(yb.numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average="macro", zero_division=0)
    try:
        auc = float(roc_auc_score(ys, probs, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1_macro": f1, "auc_ovr": auc}


def train_window_model(
    model_name: str,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_cols: list[str],
    device: str,
    class_weights: torch.Tensor,
) -> torch.nn.Module:
    model = build_model(model_name, len(feature_cols), cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"]["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_f1 = -1.0
    best_state = None
    patience_count = 0

    for _epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

        metrics = evaluate_model(model, val_loader, device)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= cfg["training"]["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward mensuel pour les modeles de trading")
    parser.add_argument(
        "--model",
        default="lstm_attention",
        choices=[name for name in MODEL_REGISTRY if name != "ensemble"],
    )
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preferred = Path(cfg["sentiment"]["aligned_output_csv"])
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

    wf_cfg = cfg.get("evaluation", {}).get("walk_forward", {})
    windows = monthly_walk_forward_splits(
        df=df,
        train_min_months=wf_cfg.get("train_min_months", 3),
        test_months=wf_cfg.get("test_months", 1),
        max_windows=args.max_windows or wf_cfg.get("max_windows", 6),
    )

    results = []
    for idx, window in enumerate(windows, start=1):
        fit_df = df[df["timestamp"].dt.to_period("M").astype(str).isin(window["fit_months"])].copy()
        val_df = df[df["timestamp"].dt.to_period("M").astype(str) == window["val_month"]].copy()
        test_df = df[df["timestamp"].dt.to_period("M").astype(str).isin(window["test_months"])].copy()

        feature_cols = get_feature_cols(df)
        scaler = fit_scaler(fit_df, feature_cols)
        seq_len = cfg["model"]["seq_len"]

        train_ds = SeqDataset(fit_df, feature_cols, seq_len, scaler)
        val_ds = SeqDataset(val_df, feature_cols, seq_len, scaler)
        test_ds = SeqDataset(test_df, feature_cols, seq_len, scaler)
        if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
            continue

        y_train = np.array(train_ds.labels)
        cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
        multipliers = np.array(
            cfg["training"].get("class_weight_multipliers", [1.0, 1.0, 1.0]),
            dtype="float32",
        )
        if multipliers.shape[0] == cw.shape[0]:
            cw = cw * multipliers
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
        sample_weights = np.array([cw[int(y)] for y in train_ds.labels], dtype="float64")

        batch_size = cfg["training"]["batch_size"]
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = train_window_model(
            model_name=args.model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            feature_cols=feature_cols,
            device=device,
            class_weights=class_weights,
        )
        clf_metrics = evaluate_model(model, test_loader, device)
        _, portfolio, portfolio_ret, bt_metrics = run_backtest_portfolio(
            model=model,
            test_df=test_df,
            feature_cols=feature_cols,
            scaler=scaler,
            cfg=cfg,
            device=device,
        )
        reliable = compute_reliable_metrics(portfolio_ret, portfolio["exposure"])

        result = {
            "window": idx,
            "fit_months": window["fit_months"],
            "val_month": window["val_month"],
            "test_months": window["test_months"],
            "classification": {k: round(v, 4) for k, v in clf_metrics.items()},
            "backtest": {k: v for k, v in bt_metrics.items() if k != "per_asset"},
            "reliable_backtest": reliable,
        }
        results.append(result)
        print(
            f"Window {idx}: test={window['test_months']} "
            f"F1={clf_metrics['f1_macro']:.4f} Sharpe={reliable['sharpe']:.2f}"
        )

    if not results:
        raise RuntimeError("Aucune fenetre walk-forward valide n'a pu etre evaluee.")

    summary = {
        "model": args.model,
        "n_windows": len(results),
        "windows": results,
        "average_classification": {
            "acc": round(float(np.mean([r["classification"]["acc"] for r in results])), 4),
            "f1_macro": round(float(np.mean([r["classification"]["f1_macro"] for r in results])), 4),
            "auc_ovr": round(float(np.nanmean([r["classification"]["auc_ovr"] for r in results])), 4),
        },
        "average_reliable_backtest": {
            "sharpe": round(float(np.mean([r["reliable_backtest"]["sharpe"] for r in results])), 4),
            "max_drawdown": round(float(np.mean([r["reliable_backtest"]["max_drawdown"] for r in results])), 4),
            "win_rate": round(float(np.mean([r["reliable_backtest"]["win_rate"] for r in results])), 4),
            "exposure": round(float(np.mean([r["reliable_backtest"]["exposure"] for r in results])), 4),
            "note": results[0]["reliable_backtest"]["note"],
        },
    }
    out_path = Path(cfg["paths"]["artifact_dir"]) / f"walk_forward_{args.model}.json"
    save_json(summary, out_path)
    print(f"Resume walk-forward sauvegarde -> {out_path}")


if __name__ == "__main__":
    main()
