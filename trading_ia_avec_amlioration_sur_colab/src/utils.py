"""utils.py — Utilitaires partagés pour le projet Trading IA."""

import os
import json
import random
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Charge config.yaml depuis le répertoire courant ou le chemin donné."""
    config_path = Path(path)
    if not config_path.exists():
        # Chercher dans le répertoire parent
        config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml introuvable (cherché dans {path} et {config_path})")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Reproductibilité ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fixe les graines pour la reproductibilité totale."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Seed fixée à {seed}")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data: dict, path: str):
    """Sauvegarde un dictionnaire en JSON (crée les dossiers si nécessaire)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type non sérialisable : {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    logger.info(f"JSON sauvegardé → {path}")


def load_json(path: str) -> dict:
    """Charge un fichier JSON."""
    with open(path) as f:
        return json.load(f)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Configure le logging vers fichier + console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "trading_ia.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("trading_ia")


# ── MLRun integration ─────────────────────────────────────────────────────────

def get_mlrun_project(cfg: dict):
    """
    Initialise ou charge un projet MLRun.
    Retourne None si MLRun n'est pas installé.
    """
    try:
        import mlrun
        project_name = cfg.get("mlrun", {}).get("project_name", "trading-ia")
        artifact_path = cfg.get("mlrun", {}).get("artifact_path", "artifacts/mlrun")
        project = mlrun.get_or_create_project(
            name=project_name,
            context="./",
            parameters={"config": cfg},
        )
        logger.info(f"Projet MLRun : {project_name}")
        return project
    except ImportError:
        logger.warning("MLRun non installé — suivi désactivé")
        return None
    except Exception as e:
        logger.warning(f"MLRun indisponible : {e}")
        return None
