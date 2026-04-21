"""metrics.py — Métriques de backtest fiables (Sharpe, MaxDD, Win Rate)."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annualisation pour données horaires
ANN_FACTOR = float(np.sqrt(365 * 24))  # ≈ 93.5


def period_return_from_horizon(future_return: float, horizon: int) -> float:
    """
    Convertit le rendement sur `horizon` heures en rendement horaire.
    Formule géométrique : (1 + r)^(1/h) - 1
    """
    return (1 + future_return) ** (1 / horizon) - 1


def compute_reliable_metrics(
    port_ret: pd.Series,
    exposure: pd.Series,
) -> Dict[str, float]:
    """
    Calcule les métriques fiables du backtest.

    Notes
    -----
    - CAGR volontairement exclu (non interprétable sur < 2 ans)
    - Sharpe annualisé sur données horaires

    Parameters
    ----------
    port_ret : pd.Series — rendements horaires du portefeuille
    exposure : pd.Series — proportion du capital exposé à chaque timestamp

    Returns
    -------
    dict avec sharpe, max_drawdown, win_rate, avg_exposure, n_trades
    """
    r = port_ret.fillna(0.0)

    # Sharpe annualisé
    mu  = float(r.mean())
    std = float(r.std()) + 1e-12
    sharpe = float(ANN_FACTOR * mu / std)

    # Max Drawdown
    equity  = (1 + r).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / (rolling_max + 1e-12)
    max_dd   = float(drawdown.min())

    # Win Rate (sur les périodes avec position)
    active = r[exposure.fillna(0) != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    # Exposition moyenne
    avg_exp = float(exposure.fillna(0).abs().mean())

    # Nombre de trades (changements de signal)
    n_trades = int((exposure.diff().fillna(exposure) != 0).sum())

    return {
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "win_rate":     round(win_rate, 4),
        "avg_exposure": round(avg_exp, 4),
        "n_trades":     n_trades,
        "n_hours":      len(r),
    }


def compute_portfolio_metrics(
    port_ret: pd.Series,
    exposure: pd.Series,
) -> Dict[str, float]:
    """Alias de compute_reliable_metrics pour compatibilité."""
    return compute_reliable_metrics(port_ret, exposure)


def compute_asset_metrics(
    asset_ret: pd.Series,
    signal: pd.Series,
) -> Dict[str, float]:
    """Métriques par actif individuel."""
    return compute_reliable_metrics(asset_ret, signal)
