"""
train_utils.py
==============
Utilitaires d'entraînement partagés entre tous les scripts.
Élimine la duplication de code entre :
  train_advanced.py, tune_all_optuna.py, tune_lstm_optuna.py, walk_forward.py

Contient :
  - SeqDataset       : dataset séquentiel avec normalisation intégrée
  - fit_scaler       : calcule mean/std sur les données d'entraînement
  - get_feature_cols : liste les colonnes de features
  - evaluate         : évalue un modèle sur un DataLoader
  - build_scheduler  : construit le scheduler (onecycle / cosine / plateau)
  - run_epoch        : une epoch d'entraînement
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


# ─── Constantes features ───────────────────────────────────────────────────────
FEATURE_SUFFIXES = ("ret_", "sma_", "ema_", "rsi_", "vol_", "zscore_")
EXTRA_FEATURES = [
    "range_pct", "body_pct", "volume_zscore_20",
    "hour", "dayofweek", "month", "sentiment_score",
]


# ─── Features ──────────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retourne la liste ordonnée des colonnes de features présentes dans df."""
    cols = [c for c in df.columns if c.startswith(FEATURE_SUFFIXES)] + EXTRA_FEATURES
    return [c for c in cols if c in df.columns]


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Calcule mean/std sur df (doit être le jeu d'entraînement uniquement)."""
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().astype("float32")
    return {
        "mean": x.mean(axis=0).values.astype("float32"),
        "std":  x.std(axis=0).replace(0, 1.0).values.astype("float32"),
    }


# ─── Dataset séquentiel ────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    """
    Dataset de séquences temporelles avec normalisation intégrée.

    Construit des fenêtres glissantes de longueur seq_len pour chaque actif.
    Si scaler est fourni, les features sont normalisées (z-score).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = 96,
        scaler: dict | None = None,
    ) -> None:
        self.samples: list[np.ndarray] = []
        self.labels:  list[int]        = []

        for _, g in df.groupby("asset"):
            g = g.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
            if len(g) <= seq_len:
                continue
            X = g[feature_cols].astype("float32").values
            if scaler is not None:
                std = scaler["std"].copy()
                std[std == 0] = 1.0
                X = (X - scaler["mean"]) / std
            y = g["target"].astype(int).values
            for i in range(seq_len, len(g)):
                self.samples.append(X[i - seq_len : i])
                self.labels.append(int(y[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],  dtype=torch.long),
        )


# ─── DataLoaders ───────────────────────────────────────────────────────────────
def make_loaders(
    train_ds: SeqDataset,
    val_ds:   SeqDataset,
    test_ds:  SeqDataset,
    batch_size: int,
    class_weight_multipliers: list[float] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construit les DataLoaders avec WeightedRandomSampler sur le train.
    Retourne (train_loader, val_loader, test_loader, class_weights_tensor).
    """
    y_train = np.array(train_ds.labels)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)

    if class_weight_multipliers is not None:
        mult = np.array(class_weight_multipliers, dtype="float32")
        if mult.shape[0] == cw.shape[0]:
            cw = cw * mult

    class_weights = torch.tensor(cw, dtype=torch.float32)
    sample_weights = np.array([cw[int(y)] for y in train_ds.labels], dtype="float64")
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights


# ─── Évaluation ────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """Évalue le modèle sur loader. Retourne acc, f1_macro, auc_ovr."""
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = torch.softmax(model(xb), dim=1)
            ys.extend(yb.cpu().numpy().tolist())
            preds.extend(p.argmax(1).cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1  = f1_score(ys, preds, average="macro", zero_division=0)
    try:
        auc = float(roc_auc_score(ys, probs, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1_macro": f1, "auc_ovr": auc}


# ─── Scheduler ─────────────────────────────────────────────────────────────────
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> tuple:
    """
    Construit le scheduler selon cfg['training']['scheduler']['type'].
    Supporte : onecycle, cosine, reduce_on_plateau (défaut).
    Retourne (scheduler, type_str).
    """
    sched_cfg  = cfg["training"].get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "reduce_on_plateau")).lower()

    if sched_type == "onecycle":
        total = max(steps_per_epoch * cfg["training"]["epochs"], 1)
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sched_cfg.get("max_lr", cfg["training"]["lr"] * 3),
                total_steps=total,
                pct_start=sched_cfg.get("pct_start", 0.2),
                div_factor=sched_cfg.get("div_factor", 10.0),
                final_div_factor=sched_cfg.get("final_div_factor", 100.0),
            ),
            "onecycle",
        )

    if sched_type == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(cfg["training"]["epochs"], 1),
                eta_min=sched_cfg.get("min_lr", cfg["training"]["lr"] * 0.1),
            ),
            "cosine",
        )

    return (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        ),
        "reduce_on_plateau",
    )


# ─── Epoch d'entraînement ──────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    scheduler=None,
    scheduler_type: str = "reduce_on_plateau",
) -> float:
    """
    Effectue une epoch d'entraînement et retourne la loss moyenne.
    Gère le step du scheduler OneCycleLR (par batch) si nécessaire.
    """
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None and scheduler_type == "onecycle":
            scheduler.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")
