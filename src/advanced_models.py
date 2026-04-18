"""
advanced_models.py
==================
Modèles avancés pour le trading :

  1. LSTMAttention   – LSTM bidirectionnel + self-attention multi-têtes
  2. TFTLite         – Temporal Fusion Transformer (version légère)
  3. CNNTransformer  – Convolutions temporelles + Transformer
  4. EnsembleModel   – Combinaison pondérée de plusieurs modèles

Chaque classe respecte la même interface que PatchTSTLite :
    forward(x: Tensor[B, T, F]) -> Tensor[B, n_classes]
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. LSTM Bidirectionnel + Self-Attention
# ─────────────────────────────────────────────────────────────
class LSTMAttention(nn.Module):
    """LSTM bidirectionnel suivi d'une couche de self-attention multi-têtes.

    Args:
        n_features: Nombre de features en entrée.
        hidden_dim: Taille cachée du LSTM (par direction).
        n_layers:   Nombre de couches LSTM.
        n_heads:    Nombre de têtes d'attention.
        dropout:    Taux de dropout.
        n_classes:  Nombre de classes de sortie.
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        attn_dim = hidden_dim * 2  # bidirectionnel → ×2
        self.norm1 = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(attn_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)                      # [B, T, H]
        x, _ = self.lstm(x)                         # [B, T, 2H]
        # Self-attention avec résidu
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        # Pooling global par moyenne
        x = self.norm2(x)
        x = x.mean(dim=1)                           # [B, 2H]
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# 2. Temporal Fusion Transformer Lite
# ─────────────────────────────────────────────────────────────
class _GatingLayer(nn.Module):
    """GLU (Gated Linear Unit) utilisé dans TFT."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        h = self.drop(self.fc(x))
        h1, h2 = h.chunk(2, dim=-1)
        out = h1 * torch.sigmoid(h2)
        if residual is not None:
            out = out + residual
        return self.norm(out)


class TFTLite(nn.Module):
    """Temporal Fusion Transformer simplifié.

    Implémente les blocs essentiels du TFT :
    - Variable Selection Network (VSN)
    - LSTM encodeur local
    - Attention Interprétable Multi-têtes
    - Gated Residual Networks (GRN)

    Args:
        n_features: Nombre de features.
        hidden_dim: Dimension cachée principale.
        n_heads:    Têtes d'attention.
        n_lstm_layers: Couches LSTM.
        dropout:    Dropout.
        n_classes:  Classes de sortie.
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Variable Selection Network
        self.vsn_weights = nn.Linear(n_features, n_features)
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # LSTM local
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.lstm_gate = _GatingLayer(hidden_dim, dropout)

        # Attention multi-têtes interprétable
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = _GatingLayer(hidden_dim, dropout)

        # GRN final
        self.grn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variable selection
        vsn = torch.softmax(self.vsn_weights(x), dim=-1)
        x = x * vsn                                # [B, T, F]
        x = self.input_proj(x)                     # [B, T, H]

        # LSTM
        lstm_out, _ = self.lstm(x)                 # [B, T, H]
        lstm_out = self.lstm_gate(lstm_out, x)      # gated residual

        # Self-attention
        attn_in = self.norm_attn(lstm_out)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        attn_out = self.attn_gate(attn_out, lstm_out)

        # GRN + pooling
        out = self.grn(attn_out)
        out = out.mean(dim=1)                      # [B, H]
        return self.head(out)


# ─────────────────────────────────────────────────────────────
# 3. CNN + Transformer
# ─────────────────────────────────────────────────────────────
class CNNTransformer(nn.Module):
    """Convolutions temporelles multi-échelles suivies d'un Transformer.

    Les convolutions extraient des patterns locaux à différentes
    granularités (kernels 3, 7, 15). Le Transformer capture ensuite
    les dépendances globales.

    Args:
        n_features: Nombre de features.
        d_model:    Dimension du modèle Transformer.
        n_heads:    Têtes d'attention.
        n_layers:   Couches Transformer.
        dropout:    Dropout.
        n_classes:  Classes de sortie.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        # Multi-scale CNN
        kernels = [3, 7, 15]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_features, d_model, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])
        # Fusion des branches CNN
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(kernels), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Tête
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x_t = x.permute(0, 2, 1)                   # [B, F, T] pour Conv1d
        branch_outs = []
        for conv in self.convs:
            out = conv(x_t)                         # [B, D, T]
            branch_outs.append(out)
        # Concaténation sur le canal puis fusion
        fused = torch.cat(branch_outs, dim=1)       # [B, D*3, T]
        fused = fused.permute(0, 2, 1)              # [B, T, D*3]
        fused = self.fusion(fused)                  # [B, T, D]
        # Transformer + pooling
        out = self.transformer(fused)               # [B, T, D]
        out = out.mean(dim=1)                       # [B, D]
        return self.head(out)


# ─────────────────────────────────────────────────────────────
# 4. Ensemble Model
# ─────────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    """Ensemble de plusieurs modèles avec pondération apprise.

    Les poids de combinaison sont appris via une couche softmax.
    En mode `trainable=False`, la moyenne simple est utilisée.

    Args:
        models:     Liste de modèles torch.nn.Module.
        n_classes:  Nombre de classes.
        trainable:  Si True, les poids d'ensemble sont apprenables.
        freeze_submodels: Si True, les sous-modèles sont gelés
                          (seuls les poids d'ensemble s'entraînent).
    """
    def __init__(
        self,
        models: list[nn.Module],
        n_classes: int = 3,
        trainable: bool = True,
        freeze_submodels: bool = False,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.trainable = trainable
        n = len(models)
        if trainable:
            self.log_weights = nn.Parameter(torch.zeros(n))
        if freeze_submodels:
            for m in self.models:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack(
            [m(x) for m in self.models], dim=1
        )  # [B, n_models, n_classes]

        if self.trainable:
            weights = torch.softmax(self.log_weights, dim=0)  # [n_models]
            weights = weights.view(1, -1, 1)                  # [1, n_models, 1]
            out = (logits * weights).sum(dim=1)               # [B, n_classes]
        else:
            out = logits.mean(dim=1)                          # [B, n_classes]
        return out

    @torch.no_grad()
    def get_weights(self) -> dict[str, float]:
        """Retourne les poids d'ensemble normalisés."""
        if not self.trainable:
            n = len(self.models)
            return {f"model_{i}": 1.0 / n for i in range(n)}
        w = torch.softmax(self.log_weights, dim=0).cpu().numpy()
        return {f"model_{i}": float(w[i]) for i in range(len(w))}


# ─────────────────────────────────────────────────────────────
# Registre des modèles disponibles
# ─────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, type] = {
    "patchtst":        None,          # importé depuis model.py
    "lstm_attention":  LSTMAttention,
    "tft_lite":        TFTLite,
    "cnn_transformer": CNNTransformer,
    "ensemble":        EnsembleModel,
}


def build_model(name: str, n_features: int, cfg: dict, n_classes: int = 3) -> nn.Module:
    """Construit un modèle à partir de son nom et de la config YAML.

    Args:
        name:       Clé dans MODEL_REGISTRY.
        n_features: Nombre de features d'entrée.
        cfg:        Dict de configuration (section ``model``).
        n_classes:  Nombre de classes.

    Returns:
        Instance nn.Module initialisée.
    """
    try:
        from src.model import PatchTSTLite  # appelé depuis la racine du projet
    except ModuleNotFoundError:
        from model import PatchTSTLite      # appelé depuis src/ directement
    MODEL_REGISTRY["patchtst"] = PatchTSTLite

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modèle inconnu : '{name}'. Disponibles : {list(MODEL_REGISTRY)}"
        )

    cls = MODEL_REGISTRY[name]

    if name == "patchtst":
        # Ne passer que les paramètres acceptés par PatchTSTLite
        import inspect
        valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self"}
        patchtst_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(n_features=n_features, **patchtst_cfg)

    if name == "lstm_attention":
        return cls(
            n_features=n_features,
            hidden_dim=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 4),
            dropout=cfg.get("dropout", 0.2),
            n_classes=n_classes,
        )

    if name == "tft_lite":
        return cls(
            n_features=n_features,
            hidden_dim=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_lstm_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            n_classes=n_classes,
        )

    if name == "cnn_transformer":
        return cls(
            n_features=n_features,
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            n_classes=n_classes,
        )

    if name == "ensemble":
        sub_names = cfg.get("ensemble_members", ["patchtst", "lstm_attention", "cnn_transformer"])
        sub_models = [build_model(n, n_features, cfg, n_classes) for n in sub_names]
        return EnsembleModel(
            models=sub_models,
            n_classes=n_classes,
            trainable=cfg.get("ensemble_trainable", True),
            freeze_submodels=cfg.get("ensemble_freeze_submodels", False),
        )

    raise ValueError(f"build_model: cas non couvert pour '{name}'")
