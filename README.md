# Trading AI System - Multi-Model Ensemble

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](./Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

Un système de prédiction de signaux de trading basé sur un **ensemble pondéré de modèles deep learning** : PatchTST, LSTM Attention et CNN Transformer. Le système intègre des données de marché et d'analyse de sentiment pour générer des signaux de trading optimisés.

##  Résultats Clés

| Modèle | F1-Macro | AUC-OVR | Sharpe Ratio |
|--------|----------|---------|--------------|
| PatchTST | 0.535 | 0.746 | ~32 |
| LSTM Attention | 0.604 | 0.780 | ~43 |
| CNN Transformer | 0.593 | 0.775 | ~42 |
| **Ensemble Pondéré** | **0.630** | **0.819** | **49.3** |

##  Architecture

```
trading-ensemble-api/
├── src/
│   ├── api.py                 # API FastAPI (7 endpoints)
│   ├── advanced_models.py     # LSTM, CNN Transformer, Ensemble
│   ├── model.py               # Implémentation PatchTST
│   ├── data.py                # Chargement et prétraitement des données
│   ├── features.py            # Ingénierie des caractéristiques
│   ├── targets.py             # Génération des labels de trading
│   ├── sentiment.py           # Analyse de sentiment
│   ├── backtest.py            # Moteur de backtesting
│   └── utils.py               # Utilitaires
├── models/
│   ├── patchtst_final.pt
│   ├── lstm_attention_final.pt
│   ├── cnn_transformer_final.pt
│   └── ensemble_final.pt
├── data/
│   ├── raw/
│   │   ├── market_data.csv
│   │   └── sentiment_labeled.csv
│   └── processed/
│       └── sentiment_market_aligned.csv
├── artifacts/
│   ├── backtest_metrics.json
│   ├── model_metrics.json
│   └── models_comparison.json
├── tests/
│   ├── test_api_structure.py
│   ├── test_config.py
│   └── test_utils.py
├── notebooks/
│   ├── 01_Setup_And_Download.ipynb
│   ├── 02_Sentiment_Datasets.ipynb
│   └── ... (autres notebooks d'analyse)
├── config.yaml                # Configuration centrale
├── Dockerfile                 # Image Docker multi-étapes
├── docker-compose.yml         # Orchestration des services
├── requirements.txt           # Dépendances complètes
└── requirements-deploy.txt    # Dépendances de production
```

##  Fonctionnalités

- **Modèles Multiples** : Ensemble de 3 architectures deep learning spécialisées
- **Analyse de Sentiment** : Intégration de données textuelles financières
- **API REST** : Interface FastAPI avec documentation interactive
- **Backtesting** : Évaluation des performances sur données historiques
- **Optimisation** : Tuning automatique avec Optuna
- **Containerisation** : Déploiement Docker prêt à la production
- **MLflow Tracking** : Suivi des expériences et métriques

##  Démarrage Rapide

### Prérequis

- Docker & Docker Compose
- 4GB RAM minimum
- Python 3.11+ (pour développement local)

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/latrousakram1/trading-ensemble-api.git
cd trading-ensemble-api

# Placer les modèles entraînés dans models/
# (patchtst_final.pt, lstm_attention_final.pt, cnn_transformer_final.pt)

# Construire et lancer avec Docker
docker compose up -d --build

# Vérifier le déploiement
curl http://localhost:8000/health
```

### Test de l'API

```bash
# Exécuter les tests automatisés
python test_api.py

# Ou tester manuellement
curl -X POST "http://localhost:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "features": [...]}'
```

##  API Documentation

L'API expose 7 endpoints principaux :

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Statut général et poids de l'ensemble |
| `GET` | `/health` | Santé détaillée des modèles |
| `GET` | `/models` | Liste des modèles disponibles |
| `POST` | `/predict/{model}` | Prédiction individuelle (patchtst, lstm, cnn) |
| `POST` | `/predict/ensemble` | Signal de trading ensemble pondéré |
| `GET` | `/predict/ensemble/scan` | Scan de tous les actifs configurés |
| `POST` | `/backtest` | Exécution de backtesting |

### Documentation Interactive

Accédez à **http://localhost:8000/docs** pour explorer l'API avec Swagger UI.

### Exemple d'utilisation

```python
import requests

# Prédiction ensemble
response = requests.post(
    "http://localhost:8000/predict/ensemble",
    json={
        "symbol": "BTCUSDT",
        "market_data": [...],  # Données OHLCV
        "sentiment_data": [...]  # Scores de sentiment
    }
)

signal = response.json()["signal"]  # "BUY", "SELL", ou "HOLD"
confidence = response.json()["confidence"]
```

## 🔧 Développement Local

### Installation des dépendances

```bash
# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installation
pip install -r requirements.txt
```

### Configuration

Le fichier `config.yaml` centralise tous les paramètres :

```yaml
project:
  name: patchtst-trading-multi-model
  seed: 42

market:
  symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  interval: 1h
  lookback_days: 180

model:
  seq_len: 96
  patch_len: 12
  d_model: 64
  n_heads: 4
```

### Lancement du serveur

```bash
# Mode développement avec rechargement automatique
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

##  Tests

```bash
# Tests unitaires
pytest tests/ -v

# Tests d'intégration API
python test_api.py

# Backtesting
python run_backtest.bat
```

##  Modèles et Métriques

### Architectures

- **PatchTST** : Transformer spécialisé séries temporelles avec patching
- **LSTM Attention** : Réseau récurrent avec mécanisme d'attention
- **CNN Transformer** : Convolution 1D + self-attention
- **Ensemble** : Moyenne pondérée optimisée par validation croisée

### Métriques de Performance

- **F1-Macro** : Équilibre précision/rappel sur classes déséquilibrées
- **AUC-OVR** : Capacité discriminative multi-classes
- **Sharpe Ratio** : Ratio rendement/risque annualisé

##  Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Guidelines

- Tests pour toute nouvelle fonctionnalité
- Documentation des endpoints API
- Respect du style de code (black, isort)
- Validation des performances sur données de test

##  Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

##  Disclaimer

Ce système est fourni à des fins éducatives et de recherche. Les signaux de trading générés ne constituent pas des conseils financiers. L'investissement en crypto-monnaies comporte des risques importants. Utilisez à vos propres risques.

##  Support

- Issues GitHub pour les bugs
- Discussions pour les questions générales
- Documentation dans `/docs`

---

*Construit avec ❤️ pour la communauté quant trading*
  - Test de démarrage du conteneur

---

## Tests locaux

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

---

## Technologies

- **Deep Learning** : PyTorch 2.x — PatchTST, LSTM Attention, CNN Transformer
- **API** : FastAPI + Uvicorn
- **Suivi expériences** : MLflow
- **Optimisation** : Optuna (NSGAIISampler multi-objectif)
- **Containerisation** : Docker + Docker Compose
- **CI/CD** : GitHub Actions
- **Données** : Binance crypto (BTC, ETH, BNB, SOL — Oct 2025 → Avr 2026)

---

.
