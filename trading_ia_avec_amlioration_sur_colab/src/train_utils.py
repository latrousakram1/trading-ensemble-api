"""train_utils.py — Utilitaires d'entraînement : Dataset, scaler, DataLoader, evaluate."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


# ── Feature columns ───────────────────────────────────────────────────────────

def get_feature_cols(df) -> List[str]:
    """Retourne les colonnes de features présentes dans le DataFrame."""
    from src.features import FEATURE_COLS
    return [c for c in FEATURE_COLS if c in df.columns]


# ── Scaler (normalisation Z-score) ───────────────────────────────────────────

def fit_scaler(train_df, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    """Calcule mean/std sur le train set uniquement."""
    X = train_df[feature_cols].astype("float32").values
    mean = np.nanmean(X, axis=0).astype("float32")
    std  = np.nanstd(X,  axis=0).astype("float32")
    std[std == 0] = 1.0  # éviter division par zéro
    return {"mean": mean, "std": std}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    """Applique la normalisation Z-score."""
    std = scaler["std"].copy()
    std[std == 0] = 1.0
    return (X - scaler["mean"]) / std


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """Dataset de séquences temporelles avec normalisation intégrée."""

    def __init__(
        self,
        df,
        feature_cols: List[str],
        seq_len: int,
        scaler: Dict[str, np.ndarray],
    ):
        self.seq_len = seq_len
        self.samples: List[np.ndarray] = []
        self.labels:  List[int]        = []

        # Construire par actif pour éviter les chevauchements inter-actifs
        for asset, g in df.groupby("symbol"):
            g = g.dropna(subset=feature_cols + ["target_label"]).reset_index(drop=True)
            if len(g) <= seq_len:
                continue
            X = g[feature_cols].astype("float32").values
            X = apply_scaler(X, scaler)
            y = g["target_label"].astype(int).values

            for i in range(seq_len, len(g)):
                self.samples.append(X[i - seq_len : i])
                self.labels.append(int(y[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_loaders(
    train_ds: SeqDataset,
    val_ds: SeqDataset,
    test_ds: SeqDataset,
    batch_size: int,
    class_weight_multipliers: List[float],
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Crée les DataLoaders avec WeightedRandomSampler pour équilibrer les classes.

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights (Tensor)
    """
    y = np.array(train_ds.labels)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    cw = cw * np.array(class_weight_multipliers, dtype="float32")
    class_weights = torch.tensor(cw, dtype=torch.float32)

    # Poids par échantillon pour le sampler
    sample_weights = np.array([cw[int(l)] for l in train_ds.labels], dtype="float64")
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """Évalue le modèle : accuracy, F1-macro, AUC-OVR."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.numpy().tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = probs_arr.argmax(axis=1)

    acc = float(accuracy_score(labels_arr, preds_arr))
    f1  = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(labels_arr, probs_arr, multi_class="ovr"))
    except Exception:
        auc = 0.5

    return {"acc": acc, "f1_macro": f1, "auc_ovr": auc}


# ── Entraînement (1 epoch) ────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
    scheduler=None,
    sched_type: str = "none",
) -> float:
    """Effectue une epoch d'entraînement. Retourne la loss moyenne."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if scheduler is not None and sched_type == "onecycle":
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> Tuple[object, str]:
    """Construit le scheduler selon la configuration."""
    sched_cfg  = cfg["training"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "reduce_on_plateau")
    n_epochs   = cfg["training"].get("epochs", 20)

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
    elif sched_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["training"]["lr"] * 5,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
        )
    else:  # reduce_on_plateau (défaut)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )
        sched_type = "reduce_on_plateau"

    return scheduler, sched_type
