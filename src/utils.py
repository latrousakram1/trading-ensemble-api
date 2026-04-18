"""
utils.py
========
Fonctions utilitaires partagées par tout le projet.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ─── Configuration ─────────────────────────────────────────────────────────────
_CFG_CACHE: dict | None = None


def load_config(path: str | Path = "config.yaml") -> dict:
    """Charge config.yaml et met en cache le résultat."""
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    for candidate in (Path(path), Path("config.yaml"), Path("../config.yaml")):
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                _CFG_CACHE = yaml.safe_load(f)
            return _CFG_CACHE
    raise FileNotFoundError("config.yaml introuvable.")


# ─── Reproductibilité ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Sauvegarde JSON ───────────────────────────────────────────────────────────
def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
