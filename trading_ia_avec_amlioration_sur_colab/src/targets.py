"""targets.py — Construction des labels de classification : 0=Sell, 1=Hold, 2=Buy."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping label → index
LABEL_MAP   = {"Sell": 0, "Hold": 1, "Buy": 2}
LABEL_NAMES = ["Sell", "Hold", "Buy"]


def build_target(
    df: pd.DataFrame,
    cfg_or_horizon=None,          # accepte cfg dict OU int
    buy_threshold: float = 0.008,
    sell_threshold: float = -0.008,
    horizon: int = 12,
) -> pd.DataFrame:
    """
    Ajoute les colonnes future_return et target_label au DataFrame.

    Peut être appelé de deux façons :
        build_target(df, cfg)          ← cfg est le dict de configuration complet
        build_target(df, horizon=12)   ← passage direct des paramètres

    Parameters
    ----------
    df             : DataFrame avec colonnes symbol, timestamp, close
    cfg_or_horizon : dict de config (cfg) OU entier (horizon)
    buy_threshold  : Rendement min pour signal BUY  (ex: +0.8%)
    sell_threshold : Rendement max pour signal SELL (ex: -0.8%)
    horizon        : Nombre de pas temporels (heures) dans le futur (si pas de cfg)

    Returns
    -------
    DataFrame avec colonnes supplémentaires :
        future_return  : float   — rendement sur `horizon` heures
        target_label   : int     — 0=Sell, 1=Hold, 2=Buy
    """
    # ── Résolution des paramètres ─────────────────────────────────────────────
    if isinstance(cfg_or_horizon, dict):
        t = cfg_or_horizon.get("target", cfg_or_horizon)
        horizon        = int(t.get("horizon",        horizon))
        buy_threshold  = float(t.get("buy_threshold",  buy_threshold))
        sell_threshold = float(t.get("sell_threshold", -abs(buy_threshold)))
    elif cfg_or_horizon is not None:
        horizon = int(cfg_or_horizon)

    # ── Nom de colonne actif (robustesse symbol / asset) ─────────────────────
    group_col = "symbol" if "symbol" in df.columns else "asset"  # noqa

    df = df.copy().sort_values([group_col, "timestamp"])

    def _per_asset(g: pd.DataFrame) -> pd.DataFrame:
        # Rendement futur sur `horizon` bougies
        g = g.copy()
        g["future_return"] = g["close"].pct_change(horizon).shift(-horizon)

        # Labels
        ret = g["future_return"]
        g["target_label"] = np.where(
            ret >= buy_threshold,  LABEL_MAP["Buy"],
            np.where(
                ret <= sell_threshold, LABEL_MAP["Sell"],
                LABEL_MAP["Hold"]
            )
        ).astype(int)
        return g

    df = df.groupby(group_col, group_keys=False).apply(_per_asset)
    df = df.reset_index(drop=True)

    # Stats
    dist = df["target_label"].value_counts().sort_index()
    total = len(df.dropna(subset=["target_label"]))
    logger.info(
        f"Labels (horizon={horizon}h) : "
        + " | ".join(
            f"{LABEL_NAMES[i]}={dist.get(i, 0):,} ({dist.get(i, 0)/total:.1%})"
            for i in range(3)
        )
    )
    return df
