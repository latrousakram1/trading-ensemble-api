"""advanced_models.py — Registre des modèles : PatchTST · LSTM-Attention · CNN-Transformer · TFT-Lite."""

import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── 1. PatchTST ───────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST adapté à la classification.
    Découpe la série temporelle en patches → Transformer → classification.
    """

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        self.seq_len   = cfg.get("seq_len", 96)
        self.patch_len = cfg.get("patch_len", 12)
        self.stride    = cfg.get("stride", 6)
        self.d_model   = cfg.get("d_model", 64)
        n_heads        = cfg.get("n_heads", 4)
        n_layers       = cfg.get("n_layers", 3)
        ff_dim         = cfg.get("ff_dim", 128)
        dropout        = cfg.get("dropout", 0.2)
        n_classes      = cfg.get("n_classes", 3)

        # Nombre de patches
        n_patches = (self.seq_len - self.patch_len) // self.stride + 1

        # Projection des patches
        self.patch_proj = nn.Linear(self.patch_len * n_features, self.d_model)

        # Encodage positionnel
        self.pos_enc = nn.Parameter(torch.zeros(1, n_patches, self.d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN pour stabilité
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(self.d_model)

        # Tête de classification
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, n_features)
        B = x.size(0)

        # Découpe en patches
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]          # (B, patch_len, F)
            patches.append(patch.reshape(B, -1))              # (B, patch_len*F)
        patches = torch.stack(patches, dim=1)                 # (B, n_patches, patch_len*F)

        z = self.patch_proj(patches) + self.pos_enc           # (B, n_patches, d_model)
        z = self.transformer(z)
        z = self.norm(z)
        z = z.mean(dim=1)                                     # global avg pooling
        return self.head(z)


# ── 2. LSTM + Self-Attention ──────────────────────────────────────────────────

class LSTMAttention(nn.Module):
    """LSTM bidirectionnel suivi d'une couche d'attention temporelle."""

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        hidden_dim = cfg.get("hidden_dim", 128)
        n_layers   = cfg.get("n_layers", 2)
        dropout    = cfg.get("dropout", 0.2)
        n_classes  = cfg.get("n_classes", 3)

        self.lstm = nn.LSTM(
            n_features, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        d_out = hidden_dim * 2  # bidirectionnel

        # Attention
        self.attn_w = nn.Linear(d_out, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_out // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                            # (B, T, 2H)
        scores = self.attn_w(out).squeeze(-1)            # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)             # (B, 2H)
        return self.head(context)


# ── 3. CNN + Transformer ──────────────────────────────────────────────────────

class CNNTransformer(nn.Module):
    """1D-CNN pour extraction locale + Transformer pour dépendances longues."""

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        d_model   = cfg.get("d_model", 64)
        n_heads   = cfg.get("n_heads", 4)
        n_layers  = cfg.get("n_layers", 2)
        dropout   = cfg.get("dropout", 0.2)
        n_classes = cfg.get("n_classes", 3)

        # CNN multi-échelle
        self.conv1 = nn.Conv1d(n_features, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_features, d_model, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(n_features, d_model, kernel_size=15, padding=7)
        self.proj  = nn.Linear(d_model * 3, d_model)
        self.bn    = nn.BatchNorm1d(d_model)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        xt = x.permute(0, 2, 1)  # (B, F, T)
        c1 = F.gelu(self.conv1(xt))
        c2 = F.gelu(self.conv2(xt))
        c3 = F.gelu(self.conv3(xt))
        cat = torch.cat([c1, c2, c3], dim=1).permute(0, 2, 1)  # (B, T, 3D)
        z   = self.proj(cat)                                     # (B, T, D)
        z   = self.bn(z.permute(0, 2, 1)).permute(0, 2, 1)
        z   = self.transformer(z)
        return self.head(z.mean(dim=1))


# ── 4. TFT-Lite ───────────────────────────────────────────────────────────────

class TFTLite(nn.Module):
    """
    Version allégée du Temporal Fusion Transformer (Lim et al., 2021).
    Variable Selection → GRN → Attention → Classification.
    """

    class GatedResidualNetwork(nn.Module):
        def __init__(self, d: int, dropout: float):
            super().__init__()
            self.fc1  = nn.Linear(d, d * 2)
            self.fc2  = nn.Linear(d * 2, d)
            self.gate = nn.Linear(d, d)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = F.elu(self.fc1(x))
            h = self.drop(self.fc2(h))
            g = torch.sigmoid(self.gate(x))
            return self.norm(x + g * h)

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        d_model   = cfg.get("d_model", 64)
        n_heads   = cfg.get("n_heads", 4)
        n_layers  = cfg.get("n_layers", 2)
        dropout   = cfg.get("dropout", 0.2)
        n_classes = cfg.get("n_classes", 3)

        self.input_proj = nn.Linear(n_features, d_model)

        # Variable selection simplifiée
        self.var_sel = self.GatedResidualNetwork(d_model, dropout)

        # Temporal processing (LSTM)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)

        # Interpretable multi-head attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.grn  = self.GatedResidualNetwork(d_model, dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.var_sel(z)
        z, _ = self.lstm(z)
        z = self.attn(z)
        z = self.grn(z)
        return self.head(z.mean(dim=1))


# ── Registre ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "patchtst":        PatchTST,
    "lstm_attention":  LSTMAttention,
    "cnn_transformer": CNNTransformer,
    "tft_lite":        TFTLite,
}


def build_model(name: str, n_features: int, cfg: dict) -> nn.Module:
    """Instancie un modèle depuis le registre."""
    name_clean = name.replace("_tuned", "").replace("_final", "")
    cls = MODEL_REGISTRY.get(name_clean)
    if cls is None:
        raise ValueError(
            f"Modèle '{name}' inconnu. Disponibles : {list(MODEL_REGISTRY.keys())}"
        )
    model = cls(n_features, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modèle {name_clean} : {n_params:,} paramètres")
    return model


def infer_cfg_from_state_dict(name: str, state_dict: dict, base_cfg: dict) -> dict:
    """
    Infère les hyperparamètres d'un modèle à partir des shapes de son state_dict.
    Évite le RuntimeError 'size mismatch' quand config.yaml diffère du checkpoint.
    """
    cfg = dict(base_cfg)  # copie pour ne pas modifier l'original
    name_clean = name.replace("_tuned", "").replace("_final", "")

    try:
        if name_clean == "lstm_attention":
            # weight_ih_l0 : shape [4*hidden_dim, n_features]
            # bidirectionnel → 2 * hidden_dim = d_out
            # n_layers : infer depuis les clés (weight_ih_l0, l1, l2 ...)
            ih = state_dict["lstm.weight_ih_l0"]          # [4H, F]
            hidden_dim = ih.shape[0] // 4
            cfg["hidden_dim"] = hidden_dim
            n_layers = sum(
                1 for k in state_dict
                if k.startswith("lstm.weight_ih_l") and "reverse" not in k
            )
            cfg["n_layers"] = n_layers
            logger.info(f"[infer] lstm_attention → hidden_dim={hidden_dim}, n_layers={n_layers}")

        elif name_clean == "patchtst":
            # patch_proj.weight : [d_model, patch_len * n_features]
            pw = state_dict["patch_proj.weight"]
            cfg["d_model"] = pw.shape[0]
            # pos_enc : [1, n_patches, d_model]
            pe = state_dict["pos_enc"]
            # n_patches = (seq_len - patch_len) // stride + 1  — difficile à inverser
            # On garde seq_len/patch_len/stride depuis config
            logger.info(f"[infer] patchtst → d_model={cfg['d_model']}")

        elif name_clean == "cnn_transformer":
            # conv1.weight : [d_model, n_features, 3]
            cfg["d_model"] = state_dict["conv1.weight"].shape[0]
            logger.info(f"[infer] cnn_transformer → d_model={cfg['d_model']}")

        elif name_clean == "tft_lite":
            # input_proj.weight : [d_model, n_features]
            cfg["d_model"] = state_dict["input_proj.weight"].shape[0]
            logger.info(f"[infer] tft_lite → d_model={cfg['d_model']}")

    except KeyError as e:
        logger.warning(f"[infer] Clé manquante pour {name_clean} : {e} — cfg inchangé")

    return cfg


def load_model_from_checkpoint(
    path: str,
    n_features: int,
    base_cfg: dict,
    device: str = "cpu",
) -> "nn.Module":
    """
    Charge un modèle depuis un fichier .pt en inférant automatiquement
    les hyperparamètres depuis le state_dict.

    Gère deux formats de checkpoint :
      - dict avec 'model_state_dict' et 'model_name'  (sauvegarde NB02)
      - state_dict direct (torch.save(model.state_dict(), path))

    Parameters
    ----------
    path       : chemin vers le .pt
    n_features : nombre de features (len(FEATURE_COLS))
    base_cfg   : cfg['model'] depuis config.yaml
    device     : 'cpu' ou 'cuda'

    Returns
    -------
    model en mode eval(), sur device
    """
    import torch

    ckpt = torch.load(path, map_location=device)

    # Détecter le format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        name       = ckpt.get("model_name", "unknown")
    else:
        # state_dict direct — deviner le nom depuis le path
        state_dict = ckpt
        import os
        stem = os.path.basename(path).replace(".pt", "")
        name = stem.replace("_tuned", "").replace("_final", "")

    name_clean = name.replace("_tuned", "").replace("_final", "")

    # Inférer les hyperparamètres depuis les shapes du checkpoint
    inferred_cfg = infer_cfg_from_state_dict(name_clean, state_dict, base_cfg)

    # Construire le modèle avec les bons hyperparamètres
    model = build_model(name_clean, n_features, inferred_cfg)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modèle chargé : {name_clean} ({n_params:,} params) depuis {path}")
    return model
