"""Tests unitaires — config.yaml"""
import yaml
import pytest
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def test_config_exists():
    assert CONFIG_PATH.exists(), "config.yaml introuvable"

def test_config_required_keys():
    cfg = load_cfg()
    required = ['model','training','backtest','paths','target','project']
    for key in required:
        assert key in cfg, f"Clé manquante : {key}"

def test_model_config():
    cfg = load_cfg()
    m = cfg['model']
    assert m['seq_len'] > 0
    assert m['n_classes'] == 3
    assert m['dropout'] >= 0 and m['dropout'] <= 1

def test_backtest_config():
    cfg = load_cfg()
    b = cfg['backtest']
    assert 0 < b['probability_threshold'] < 1
    assert b['fee_bps'] >= 0
    assert b['slippage_bps'] >= 0

def test_target_config():
    cfg = load_cfg()
    t = cfg['target']
    assert t['horizon'] > 0
    assert t['buy_threshold'] > 0
