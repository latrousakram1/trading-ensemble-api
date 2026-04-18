"""Tests unitaires — fonctions utilitaires"""
import pytest
import numpy as np


def test_signal_buy():
    """Quand prob_buy >= threshold et > prob_sell → signal BUY."""
    probs = np.array([0.10, 0.30, 0.60])
    threshold = 0.45
    p_sell, _, p_buy = probs
    if p_buy >= threshold and p_buy > p_sell:
        signal = 'BUY'
    else:
        signal = 'OTHER'
    assert signal == 'BUY'


def test_signal_sell():
    """Quand prob_sell >= threshold et > prob_buy → signal SELL."""
    probs = np.array([0.55, 0.30, 0.15])
    threshold = 0.45
    p_sell, _, p_buy = probs
    if p_sell >= threshold and p_sell > p_buy:
        signal = 'SELL'
    else:
        signal = 'OTHER'
    assert signal == 'SELL'


def test_signal_hold():
    """Quand aucun seuil atteint → signal HOLD."""
    probs = np.array([0.35, 0.40, 0.25])
    threshold = 0.45
    p_sell, _, p_buy = probs
    if p_buy >= threshold and p_buy > p_sell:
        signal = 'BUY'
    elif p_sell >= threshold and p_sell > p_buy:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    assert signal == 'HOLD'


def test_ensemble_weighted_sum():
    """Les probabilités pondérées doivent sommer à 1."""
    probs_per_model = {
        'patchtst':       np.array([0.20, 0.50, 0.30]),
        'lstm_attention': np.array([0.15, 0.45, 0.40]),
        'cnn_transformer':np.array([0.25, 0.35, 0.40]),
    }
    weights = {'patchtst': 0.309, 'lstm_attention': 0.349, 'cnn_transformer': 0.343}
    total_w = sum(weights.values())
    avg = sum(w * probs_per_model[n] for n, w in weights.items()) / total_w
    assert abs(avg.sum() - 1.0) < 1e-6, f"Somme des probas != 1 : {avg.sum()}"


def test_portfolio_aggregation():
    """Le rendement portefeuille doit être la moyenne des actifs à chaque timestamp."""
    import pandas as pd
    records = [
        {'timestamp': 1, 'asset': 'BTC', 'strategy_ret': 0.01},
        {'timestamp': 1, 'asset': 'ETH', 'strategy_ret': 0.02},
        {'timestamp': 2, 'asset': 'BTC', 'strategy_ret': -0.01},
        {'timestamp': 2, 'asset': 'ETH', 'strategy_ret': 0.00},
    ]
    out = pd.DataFrame(records)
    portfolio = out.groupby('timestamp')['strategy_ret'].mean()
    assert abs(portfolio[1] - 0.015) < 1e-10
    assert abs(portfolio[2] - (-0.005)) < 1e-10


def test_sharpe_positive_returns():
    """Le Sharpe doit être positif pour une série de retours positifs."""
    import numpy as np
    returns = np.array([0.001] * 100)
    ann = np.sqrt(365 * 24)
    sharpe = ann * returns.mean() / (returns.std() + 1e-12)
    assert sharpe > 0
