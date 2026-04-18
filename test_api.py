"""
test_api.py
===========
Script de test complet de l'API de trading.
Lance après avoir démarré le serveur.

Usage :
    python test_api.py
    python test_api.py --url http://localhost:8000
    python test_api.py --asset ETHUSDT --threshold 0.50
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import requests

GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):  print(f'  {GREEN}✓{RESET} {msg}')
def err(msg): print(f'  {RED}✗{RESET} {msg}')
def warn(msg):print(f'  {YELLOW}!{RESET} {msg}')
def sec(msg): print(f'\n{BOLD}{msg}{RESET}')


def test_root(base: str) -> bool:
    sec('GET /')
    try:
        r = requests.get(f'{base}/', timeout=10)
        assert r.status_code == 200, f'HTTP {r.status_code}'
        d = r.json()
        ok(f"version       : {d.get('version')}")
        ok(f"uptime        : {d.get('uptime_s')}s")
        ok(f"models_loaded : {d.get('models_loaded')}")
        ok(f"assets        : {d.get('assets')}")
        ok(f"poids ensemble: {d.get('ensemble_weights')}")
        return True
    except Exception as e:
        err(str(e))
        return False


def test_health(base: str) -> bool:
    sec('GET /health')
    try:
        r = requests.get(f'{base}/health', timeout=10)
        assert r.status_code == 200, f'HTTP {r.status_code}'
        d = r.json()
        status = d.get('status')
        col = GREEN if status == 'ok' else YELLOW
        print(f'  {col}● status       : {status}{RESET}')
        ok(f"data_loaded   : {d.get('data_loaded')} ({d.get('data_rows'):,} lignes)")
        for m in d.get('models', []):
            s = '✓' if m['loaded'] else '✗'
            print(f'  {GREEN if m["loaded"] else RED}{s}{RESET} '
                  f"{m['name']:<22} {m.get('file','—')}")
        return status in ('ok', 'degraded')
    except Exception as e:
        err(str(e))
        return False


def test_models(base: str) -> bool:
    sec('GET /models')
    try:
        r = requests.get(f'{base}/models', timeout=10)
        assert r.status_code == 200, f'HTTP {r.status_code}'
        d = r.json()
        for m in d.get('models', []):
            ok(f"{m['name']:<22} params={m.get('n_params','?'):>10}  "
               f"weight={m.get('weight_in_ensemble')}")
        return True
    except Exception as e:
        err(str(e))
        return False


def test_predict_model(base: str, model: str, asset: str, threshold: float) -> bool:
    sec(f'POST /predict/{model}')
    payload = {'asset': asset, 'threshold': threshold, 'allow_short': True}
    try:
        r = requests.post(f'{base}/predict/{model}', json=payload, timeout=30)
        if r.status_code == 404:
            warn(f'Modèle {model} non chargé — skipped')
            return True
        assert r.status_code == 200, f'HTTP {r.status_code} — {r.text[:200]}'
        d = r.json()
        sig_col = GREEN if d['signal'] == 'BUY' else (RED if d['signal'] == 'SELL' else YELLOW)
        ok(f"asset      : {d['asset']}")
        ok(f"timestamp  : {d['timestamp']}")
        ok(f"close      : {d['close']:.2f}")
        print(f'  signal     : {sig_col}{d["signal"]}{RESET}  '
              f'(conf={d["confidence"]:.1%}  '
              f'sell={d["prob_sell"]:.1%}  '
              f'hold={d["prob_hold"]:.1%}  '
              f'buy={d["prob_buy"]:.1%})')
        ok(f"latency    : {d['latency_ms']} ms")
        return True
    except Exception as e:
        err(str(e))
        return False


def test_ensemble(base: str, asset: str, threshold: float) -> bool:
    sec('POST /predict/ensemble')
    payload = {'asset': asset, 'threshold': threshold, 'allow_short': True}
    try:
        r = requests.post(f'{base}/predict/ensemble', json=payload, timeout=30)
        assert r.status_code == 200, f'HTTP {r.status_code} — {r.text[:200]}'
        d = r.json()
        sig_col = GREEN if d['signal'] == 'BUY' else (RED if d['signal'] == 'SELL' else YELLOW)
        ok(f"asset      : {d['asset']}")
        ok(f"timestamp  : {d['timestamp']}")
        ok(f"close      : {d['close']:.2f}")
        print(f'\n  {BOLD}SIGNAL ENSEMBLE : {sig_col}{d["signal"]}{RESET}  '
              f'(confiance={d["confidence"]:.1%})')
        print(f'  Sell={d["prob_sell"]:.1%}  Hold={d["prob_hold"]:.1%}  Buy={d["prob_buy"]:.1%}')
        print(f'\n  {BOLD}Contributions par modèle :{RESET}')
        print(f'  {"Modèle":<22} {"Poids":>6}  {"Sell":>7} {"Hold":>7} {"Buy":>7}  Signal')
        print('  ' + '─' * 65)
        for name, contrib in d.get('model_contributions', {}).items():
            w = d['weights'].get(name, '?')
            sig2 = contrib['signal']
            c2 = GREEN if sig2 == 'BUY' else (RED if sig2 == 'SELL' else YELLOW)
            print(f'  {name:<22} {w:>6.3f}  '
                  f'{contrib["prob_sell"]:>6.1%} {contrib["prob_hold"]:>6.1%} '
                  f'{contrib["prob_buy"]:>6.1%}  {c2}{sig2}{RESET}')
        ok(f"latency    : {d['latency_ms']} ms")
        return True
    except Exception as e:
        err(str(e))
        return False


def test_scan(base: str, threshold: float) -> bool:
    sec('GET /predict/ensemble/scan')
    try:
        r = requests.get(
            f'{base}/predict/ensemble/scan',
            params={'threshold': threshold, 'allow_short': True},
            timeout=60,
        )
        assert r.status_code == 200, f'HTTP {r.status_code} — {r.text[:200]}'
        d = r.json()
        ok(f"scanned_at : {d['scanned_at']}")
        ok(f"threshold  : {d['threshold']}")
        print(f'\n  {"Actif":<12} {"Sell":>8} {"Hold":>8} {"Buy":>8}  {"Signal":<10} Conf')
        print('  ' + '─' * 58)
        for res in d.get('results', []):
            sig_col = (GREEN if res['signal'] == 'BUY'
                       else RED if res['signal'] == 'SELL' else YELLOW)
            print(f'  {res["asset"]:<12} '
                  f'{res["prob_sell"]:>7.1%} {res["prob_hold"]:>7.1%} '
                  f'{res["prob_buy"]:>7.1%}  '
                  f'{sig_col}{res["signal"]:<10}{RESET} '
                  f'{res["confidence"]:.1%}')
        ok(f"latency    : {d['latency_ms']} ms")
        return True
    except Exception as e:
        err(str(e))
        return False


def test_errors(base: str) -> bool:
    sec('Tests erreurs (404 / 400)')
    passed = True

    # Modèle inexistant
    r = requests.post(f'{base}/predict/modele_inexistant',
                      json={'asset': 'BTCUSDT'}, timeout=10)
    if r.status_code == 404:
        ok('Modèle inexistant → 404')
    else:
        err(f'Attendu 404, reçu {r.status_code}')
        passed = False

    # Actif inexistant
    r = requests.post(f'{base}/predict/ensemble',
                      json={'asset': 'FAKEUSDT'}, timeout=10)
    if r.status_code in (400, 422, 503):
        ok(f'Actif inexistant → {r.status_code}')
    else:
        err(f'Attendu 4xx, reçu {r.status_code}')
        passed = False

    return passed


def main():
    parser = argparse.ArgumentParser(description='Test API Trading')
    parser.add_argument('--url',       default='http://localhost:8000')
    parser.add_argument('--asset',     default='BTCUSDT')
    parser.add_argument('--threshold', type=float, default=0.45)
    args = parser.parse_args()

    base = args.url.rstrip('/')
    print(f'\n{BOLD}=== TEST API TRADING ==={RESET}')
    print(f'URL       : {base}')
    print(f'Asset     : {args.asset}')
    print(f'Threshold : {args.threshold}')

    # Vérifier que l'API est disponible
    print(f'\nConnexion à {base}...')
    for attempt in range(5):
        try:
            requests.get(f'{base}/health', timeout=5)
            break
        except Exception:
            if attempt == 4:
                err('API non disponible après 5 tentatives.')
                sys.exit(1)
            warn(f'Tentative {attempt+1}/5 — retry dans 3s')
            time.sleep(3)

    results = {
        'root':    test_root(base),
        'health':  test_health(base),
        'models':  test_models(base),
    }
    for model in ['patchtst', 'lstm_attention', 'cnn_transformer']:
        results[f'predict_{model}'] = test_predict_model(base, model, args.asset, args.threshold)
    results['ensemble'] = test_ensemble(base, args.asset, args.threshold)
    results['scan']     = test_scan(base, args.threshold)
    results['errors']   = test_errors(base)

    # Résumé
    total  = len(results)
    passed = sum(results.values())
    failed = total - passed
    col    = GREEN if failed == 0 else RED
    print(f'\n{BOLD}{"="*40}{RESET}')
    print(f'{BOLD}RÉSUMÉ : {col}{passed}/{total} tests passés{RESET}')
    if failed:
        print(f'{RED}Échecs :{RESET}')
        for name, ok_flag in results.items():
            if not ok_flag:
                print(f'  {RED}✗ {name}{RESET}')
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
