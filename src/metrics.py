"""
metrics.py
==========
Source unique de vérité pour toutes les métriques financières.
Importé par backtest.py, train_advanced.py, walk_forward.py, ensemble_final.py.

Corrections appliquées :
  - compute_portfolio_metrics : equity curve sur rendements agrégés par timestamp
  - compute_reliable_metrics  : masque CAGR/annual_return (non fiable sur < 2 ans)
  - period_return_from_horizon: conversion géométrique correcte (non linéaire)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Constante d'annualisation pour crypto 1h ─────────────────────────────────
ANN_FACTOR_1H = float(np.sqrt(365 * 24))   # ≈ 92.97


# ─── Conversion retour cumulé → retour horaire ────────────────────────────────
def period_return_from_horizon(cumulative_return: pd.Series | np.ndarray,
                                horizon: int) -> pd.Series | np.ndarray:
    """
    Convertit un retour cumulé sur `horizon` périodes en retour par période
    en utilisant la formule géométrique (correcte) plutôt que la division linéaire.

    cumulative_return : (close[t+h] / close[t]) - 1
    horizon           : nombre de périodes (ex: 12 pour 12h)

    Retourne : (1 + cumulative_return)^(1/horizon) - 1
    """
    return (1.0 + cumulative_return) ** (1.0 / max(horizon, 1)) - 1.0


# ─── Métriques portefeuille (agrégé par timestamp) ────────────────────────────
def compute_portfolio_metrics(
    portfolio_ret: pd.Series,
    portfolio_exposure: pd.Series | None = None,
    ann_factor: float | None = None,
) -> dict:
    """
    Métriques sur le rendement portefeuille déjà agrégé par timestamp.

    portfolio_ret      : rendement moyen à chaque heure (1 valeur par timestamp)
    portfolio_exposure : fraction d'actifs actifs à chaque heure
    ann_factor         : √(périodes/an). Défaut = √(365×24) pour crypto 1h.

    Note : annual_return / CAGR est inclus ici mais doit être lu avec précaution
    sur des backtests < 2 ans. Utiliser compute_reliable_metrics() pour la
    présentation et le rapport officiel.
    """
    portfolio_ret = portfolio_ret.fillna(0.0)
    if ann_factor is None:
        ann_factor = ANN_FACTOR_1H

    equity = (1.0 + portfolio_ret).cumprod()
    n = max(len(portfolio_ret), 1)
    periods_per_year = ann_factor ** 2          # (365 × 24) pour 1h

    total_ret  = float(equity.iloc[-1] - 1.0)
    annual_ret = float((1.0 + total_ret) ** (periods_per_year / n) - 1.0)
    vol        = float(portfolio_ret.std() * ann_factor)
    sharpe     = float(ann_factor * portfolio_ret.mean() / (portfolio_ret.std() + 1e-12))
    max_dd     = float((equity / equity.cummax() - 1.0).min())
    calmar     = annual_ret / (abs(max_dd) + 1e-12)

    active   = portfolio_ret[portfolio_ret != 0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    exposure = (
        float(portfolio_exposure.mean())
        if portfolio_exposure is not None
        else float((portfolio_ret != 0).mean())
    )

    return {
        "total_return":     round(total_ret,  4),
        "annual_return":    round(annual_ret, 4),
        "sharpe":           round(sharpe,     4),
        "calmar":           round(calmar,     4),
        "max_drawdown":     round(max_dd,     4),
        "annual_vol":       round(vol,        4),
        "win_rate":         round(win_rate,   4),
        "exposure":         round(exposure,   4),
        "n_active_periods": int((portfolio_ret != 0).sum()),
    }


# ─── Métriques par actif (equity mono-actif) ──────────────────────────────────
def compute_asset_metrics(ret: pd.Series, ann_factor: float | None = None) -> dict:
    """Métriques individuelles par actif (equity construite sur ses seuls rendements)."""
    ret = ret.fillna(0.0)
    if ann_factor is None:
        ann_factor = ANN_FACTOR_1H

    equity   = (1.0 + ret).cumprod()
    total    = float(equity.iloc[-1] - 1.0)
    n        = max(len(ret), 1)
    annual   = float((1.0 + total) ** (ann_factor ** 2 / n) - 1.0)
    sharpe   = float(ann_factor * ret.mean() / (ret.std() + 1e-12))
    max_dd   = float((equity / equity.cummax() - 1.0).min())
    win_rate = float((ret[ret != 0] > 0).mean()) if (ret != 0).any() else 0.0

    return {
        "total_return":  round(total,    4),
        "annual_return": round(annual,   4),
        "sharpe":        round(sharpe,   4),
        "max_drawdown":  round(max_dd,   4),
        "win_rate":      round(win_rate, 4),
        "n_trades":      int((ret.diff().abs().fillna(ret.abs()) > 0).sum()),
    }


# ─── Métriques fiables (pour la présentation et le rapport) ───────────────────
def compute_reliable_metrics(
    portfolio_ret: pd.Series,
    portfolio_exposure: pd.Series | None = None,
    ann_factor: float | None = None,
) -> dict:
    """
    Sous-ensemble des métriques financièrement fiables sur un backtest court (< 2 ans).

    Exclut volontairement annual_return et CAGR : sur 6 mois de données crypto
    horaires, l'extrapolation annuelle produit des chiffres non interprétables
    (ex : 818 000%). Ces métriques restent correctes mathématiquement mais
    induisent en erreur lors de la présentation.

    Retourne : Sharpe, Max DD, Win Rate, Exposition, n_active_periods.
    """
    full = compute_portfolio_metrics(
        portfolio_ret=portfolio_ret,
        portfolio_exposure=portfolio_exposure,
        ann_factor=ann_factor,
    )
    return {
        "sharpe":           full["sharpe"],
        "max_drawdown":     full["max_drawdown"],
        "win_rate":         full["win_rate"],
        "exposure":         full["exposure"],
        "n_active_periods": full["n_active_periods"],
        "note": (
            "CAGR/annual_return volontairement exclus : l'extrapolation annuelle "
            "sur < 2 ans de données crypto horaires produit des chiffres non "
            "interprétables économiquement. Sharpe, Max DD et Win Rate sont fiables."
        ),
    }
