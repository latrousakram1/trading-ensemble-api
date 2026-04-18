# 📖 Manuel d'Utilisation - Système de Trading IA

## Vue d'ensemble

Ce manuel décrit l'utilisation complète du système de trading IA multi-modèles basé sur un ensemble pondéré de réseaux de neurones deep learning (PatchTST, LSTM Attention, CNN Transformer).

---

## 🎯 Table des Matières

1. [Installation et Configuration](#installation-et-configuration)
2. [Architecture du Système](#architecture-du-système)
3. [Utilisation de l'API](#utilisation-de-lapi)
4. [Prédictions de Trading](#prédictions-de-trading)
5. [Backtesting](#backtesting)
6. [Dashboard et Visualisation](#dashboard-et-visualisation)
7. [Configuration Avancée](#configuration-avancée)
8. [Dépannage](#dépannage)

---

## 🚀 Installation et Configuration

### Prérequis Système

- **Python** : 3.11 ou supérieur
- **Mémoire RAM** : Minimum 4GB, recommandé 8GB+
- **Espace disque** : 2GB pour le code, + modèles (~500MB)
- **Système d'exploitation** : Windows/Linux/MacOS

### Installation Rapide

```bash
# Cloner le dépôt
git clone https://github.com/latrousakram1/trading-ensemble-api.git
cd trading-ensemble-api

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration Initiale

1. **Télécharger les modèles entraînés** dans le dossier `models/` :
   - `patchtst_final.pt`
   - `lstm_attention_final.pt`
   - `cnn_transformer_final.pt`
   - `ensemble_final.pt` (optionnel)

2. **Vérifier la configuration** dans `config.yaml` :
   ```yaml
   market:
     symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]  # Actifs à trader
     interval: 1h                                    # Intervalle temporel
   ```

---

## 🏗️ Architecture du Système

### Composants Principaux

```
trading-ensemble-api/
├── src/
│   ├── api.py              # API FastAPI (interface utilisateur)
│   ├── advanced_models.py  # Modèles LSTM, CNN, Ensemble
│   ├── model.py            # Implémentation PatchTST
│   ├── data.py             # Chargement et prétraitement des données
│   ├── features.py         # Ingénierie des caractéristiques
│   ├── backtest.py         # Moteur de backtesting
│   └── realtime.py         # Buffer de données temps réel
├── models/                 # Modèles entraînés (.pt)
├── data/                   # Données marché et sentiment
├── artifacts/              # Métriques et résultats
└── config.yaml            # Configuration centrale
```

### Modèles Disponibles

| Modèle | Description | Points Forts |
|--------|-------------|--------------|
| **PatchTST** | Transformer spécialisé séries temporelles | Précision, stabilité |
| **LSTM Attention** | Réseau récurrent avec attention | Patterns longs termes |
| **CNN Transformer** | Convolution + self-attention | Patterns locaux |
| **Ensemble** | Moyenne pondérée optimisée | Meilleure performance globale |

---

## 🌐 Utilisation de l'API

### Démarrage du Serveur

```bash
# Mode développement (avec rechargement automatique)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Mode production
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Accès à l'API

- **URL locale** : http://localhost:8000
- **Documentation interactive** : http://localhost:8000/docs
- **Interface alternative** : http://localhost:8000/redoc

### Endpoints Disponibles

#### Informations Générales

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Statut général du système |
| `/health` | GET | Santé détaillée des composants |
| `/models` | GET | Liste des modèles disponibles |
| `/validation` | GET | Résultats de validation walk-forward |

#### Prédictions

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/predict/{model}` | POST | Prédiction d'un modèle spécifique |
| `/predict/ensemble` | POST | Prédiction ensemble pondéré |
| `/predict/ensemble/scan` | GET | Scan de tous les actifs |

#### Visualisation

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/dashboard` | GET | Dashboard HTML interactif |
| `/dashboard/data` | GET | Données JSON du dashboard |

---

## 📊 Prédictions de Trading

### Format des Données d'Entrée

Toutes les prédictions nécessitent des données au format JSON :

```json
{
  "symbol": "BTCUSDT",
  "market_data": [
    [1640995200, 46216.93, 46391.49, 46208.37, 46306.45, 40.676],
    [1640998800, 46306.44, 46589.26, 46253.58, 46556.15, 38.432]
  ],
  "sentiment_data": [0.2, 0.1, -0.1, 0.3, -0.2]
}
```

**Structure des données marché (OHLCV)** :
- `timestamp` : Timestamp Unix
- `open` : Prix d'ouverture
- `high` : Prix le plus haut
- `low` : Prix le plus bas
- `close` : Prix de clôture
- `volume` : Volume échangé

### Exemples d'Utilisation

#### Prédiction Individuelle

```python
import requests

# Prédiction avec PatchTST
response = requests.post(
    "http://localhost:8000/predict/patchtst",
    json={
        "symbol": "BTCUSDT",
        "market_data": [...],  # Vos données OHLCV
        "sentiment_data": [...]  # Scores de sentiment
    }
)

result = response.json()
print(f"Signal: {result['signal']}")  # BUY, SELL, HOLD
print(f"Confiance: {result['confidence']}")
```

#### Prédiction Ensemble

```python
# Prédiction avec l'ensemble (recommandé)
response = requests.post(
    "http://localhost:8000/predict/ensemble",
    json={
        "symbol": "BTCUSDT",
        "market_data": [...],
        "sentiment_data": [...]
    }
)

result = response.json()
print(f"Signal final: {result['signal']}")
print(f"Confiance: {result['confidence']}")
print(f"Poids des modèles: {result['weights']}")
```

#### Scan Multi-Actifs

```python
# Scanner tous les actifs configurés
response = requests.get("http://localhost:8000/predict/ensemble/scan")

signals = response.json()
for asset in signals['assets']:
    print(f"{asset['symbol']}: {asset['signal']} ({asset['confidence']:.2f})")
```

### Interprétation des Résultats

- **BUY** : Signal d'achat (prix attendu en hausse)
- **SELL** : Signal de vente (prix attendu en baisse)
- **HOLD** : Pas de signal clair (conserver position)

**Confiance** : Probabilité du signal (0.0 à 1.0)
- > 0.7 : Signal très fiable
- 0.5-0.7 : Signal modéré
- < 0.5 : Signal faible

---

## 📈 Backtesting

### Exécution d'un Backtest

```bash
# Via script batch (Windows)
run_backtest.bat

# Via Python directement
python -m src.backtest
```

### Résultats du Backtest

Les résultats sont sauvegardés dans `artifacts/backtest_metrics.json` :

```json
{
  "total_return": 0.493,
  "sharpe_ratio": 49.3,
  "max_drawdown": 0.12,
  "win_rate": 0.63,
  "total_trades": 1247,
  "avg_trade_duration": "4.2h"
}
```

### Métriques Clés

- **Total Return** : Rendement total annualisé
- **Sharpe Ratio** : Ratio rendement/risque (>2 considéré bon)
- **Max Drawdown** : Perte maximale (<20% acceptable)
- **Win Rate** : Taux de trades gagnants (>55% bon)
- **Total Trades** : Nombre de transactions exécutées

---

## 📊 Dashboard et Visualisation

### Accès au Dashboard

1. **Démarrer l'API** : `uvicorn src.api:app --host 0.0.0.0 --port 8000`
2. **Ouvrir dans le navigateur** : http://localhost:8000/dashboard

### Fonctionnalités du Dashboard

- **Graphiques de performance** : Évolution du portefeuille
- **Métriques en temps réel** : Sharpe, drawdown, win rate
- **Signaux actifs** : Positions ouvertes et pending
- **Historique des trades** : Liste détaillée des transactions
- **Analyse par actif** : Performance par crypto-monnaie

### API des Données

```python
# Récupérer les données du dashboard
response = requests.get("http://localhost:8000/dashboard/data")
data = response.json()

print("Performance globale:", data['performance'])
print("Trades récents:", data['recent_trades'])
```

---

## ⚙️ Configuration Avancée

### Fichier de Configuration

Le fichier `config.yaml` contrôle tous les paramètres du système :

```yaml
project:
  name: patchtst-trading-multi-model
  seed: 42

market:
  symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]  # Actifs à trader
  interval: 1h                                    # Intervalle: 1m, 5m, 15m, 1h, 4h, 1d
  limit_per_call: 1000                           # Limite d'historique par appel API
  lookback_days: 180                             # Période d'historique

features:
  returns: [1, 3, 6, 12]                         # Périodes de returns
  sma: [5, 10, 20]                              # Moyennes mobiles simples
  ema: [5, 10, 20]                              # Moyennes mobiles exponentielles
  rsi_period: 14                                 # Période RSI

model:
  seq_len: 96                                    # Longueur des séquences
  patch_len: 12                                  # Taille des patches (PatchTST)
  d_model: 64                                    # Dimension des embeddings
  n_heads: 4                                     # Nombre de têtes d'attention
  n_layers: 3                                    # Nombre de couches
```

### Modification des Actifs

Pour ajouter/modifier les actifs trackés :

1. **Éditer `config.yaml`** :
   ```yaml
   market:
     symbols: [BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT]
   ```

2. **Redémarrer l'API** pour prendre en compte les changements

### Ajustement des Paramètres de Trading

```yaml
target:
  horizon: 12        # Horizon de prédiction (en périodes)
  buy_threshold: 0.008   # Seuil d'achat (+0.8%)
  sell_threshold: -0.008 # Seuil de vente (-0.8%)
```

---

## 🔧 Dépannage

### Problèmes Courants

#### "Model file not found"
```
Erreur: torch.load() missing file: models/patchtst_final.pt
```
**Solution** : Télécharger et placer les fichiers `.pt` dans le dossier `models/`

#### "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Solution** : Le système utilise automatiquement CPU si CUDA n'est pas disponible

#### "Port already in use"
```
ERROR: Port 8000 already in use
```
**Solution** :
```bash
# Utiliser un port différent
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

#### "ImportError: No module named 'src'"
```
ImportError: No module named 'src'
```
**Solution** : Installer en mode développement :
```bash
pip install -e .
```

### Logs et Debug

#### Activer les logs détaillés

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Vérifier la santé du système

```bash
curl http://localhost:8000/health
```

### Performance

#### Optimisations recommandées

1. **Utiliser CUDA** si disponible (GPU NVIDIA)
2. **Augmenter la RAM** pour de gros volumes de données
3. **Utiliser SSD** pour un accès rapide aux modèles
4. **Configurer un proxy** pour les appels API fréquents

---

## 📚 Ressources Supplémentaires

### Documentation Technique

- [README.md](README.md) - Vue d'ensemble du projet
- [README_DEPLOY.md](README_DEPLOY.md) - Guide de déploiement
- [config.yaml](config.yaml) - Configuration détaillée

### Scripts Utiles

- `run_api.bat` - Démarrage rapide de l'API
- `run_backtest.bat` - Exécution du backtesting
- `test_api.py` - Tests de validation

### Liens Externes

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)

---

## ⚠️ Avertissements Importants

### Risques du Trading

- **Ce système est éducatif** : Les signaux ne constituent pas des conseils financiers
- **Trading de crypto-monnaies** : Risque de perte totale du capital
- **Backtesting ≠ Performance réelle** : Les conditions passées ne prédisent pas l'avenir
- **Diversification** : Ne pas investir plus que ce que vous pouvez perdre

### Limitations du Système

- **Données historiques** : Performance basée sur des données passées
- **Volatilité** : Les crypto-monnaies sont extrêmement volatiles
- **Liquidité** : Certains actifs peuvent avoir une faible liquidité
- **Frais de trading** : Non pris en compte dans les calculs

### Recommandations

1. **Commencer petit** : Tester avec de petits montants
2. **Surveiller régulièrement** : Vérifier les performances
3. **Diversifier** : Ne pas concentrer sur un seul actif
4. **Stop-loss** : Toujours définir des limites de perte
5. **Éducation** : Continuer à apprendre sur le trading

---

## 📞 Support

### Pour obtenir de l'aide :

1. **Vérifier les logs** : Examiner les messages d'erreur
2. **Consulter la documentation** : README et guides
3. **Tester les endpoints** : Utiliser `/health` et `/docs`
4. **Issues GitHub** : Signaler les bugs sur le dépôt

### Contact

- **Dépôt GitHub** : https://github.com/latrousakram1/trading-ensemble-api
- **Issues** : Pour signaler des bugs
- **Discussions** : Pour les questions générales

---

*Dernière mise à jour : Avril 2026*</content>
<parameter name="filePath">c:\Users\Utilisateur\Desktop\ua4\notbook\final_patchtst_trading_deployable_final\MANUAL.md