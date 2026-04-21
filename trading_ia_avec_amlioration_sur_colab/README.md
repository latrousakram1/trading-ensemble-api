# 🤖 Trading IA Multi-Modèles — Crypto

> Deep learning pour le trading crypto : PatchTST · LSTM-Attention · CNN-Transformer · TFT-Lite  
> Pipeline complet : données → features → entraînement → backtest → optimisation → live

---

## 📋 Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Installation rapide](#3-installation-rapide)
4. [Exécution sur Google Colab](#4-exécution-sur-google-colab)
5. [Exécution locale](#5-exécution-locale)
6. [Docker](#6-docker)
7. [Dashboard & API](#7-dashboard--api)
8. [MLRun / MLFlow](#8-mlrun--mlflow)
9. [Configuration](#9-configuration)
10. [Description des modules src/](#10-description-des-modules-src)
11. [Modèles de deep learning](#11-modèles-de-deep-learning)
12. [Score QDM Sentiment](#12-score-qdm-sentiment)
13. [Métriques et résultats](#13-métriques-et-résultats)
14. [FAQ](#14-faq)

---

## 1. Vue d'ensemble

Ce projet implémente un système de trading algorithmique basé sur 4 modèles de deep learning entraînés
sur des données OHLCV horaires de 4 actifs crypto (BTC, ETH, BNB, SOL).

```
Données Binance (180j × 1h)
      ↓
Features (21 indicateurs techniques + sentiment QDM)
      ↓
Labels  (SELL=0 / HOLD=1 / BUY=2,  horizon=12h)
      ↓
4 modèles deep learning (PatchTST / LSTM-Attention / CNN-Transformer / TFT-Lite)
      ↓
Optimisation Optuna (NSGAIISampler, 2 objectifs : F1 + Sharpe)
      ↓
Ensemble pondéré → Backtest walk-forward → API live → Dashboard
```

### Actifs et configuration par défaut

| Paramètre       | Valeur            |
|-----------------|-------------------|
| Actifs          | BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT |
| Intervalle      | 1h                |
| Lookback        | 180 jours         |
| Fenêtre entrée  | 96 bougies (seq_len) |
| Horizon cible   | 12h               |
| Seuil BUY/SELL  | ±0.8%             |
| Frais backtest  | 5 bps + 2 bps slippage |

---

## 2. Structure du projet

```
trading_ia/
├── config.yaml                    ← Configuration centrale
├── requirements.txt               ← Dépendances Python
├── README.md                      ← Ce fichier
│
├── src/                           ← Modules Python réutilisables
│   ├── utils.py                   ← load_config, set_seed, logging
│   ├── data.py                    ← Téléchargement Binance, chargement CSV
│   ├── features.py                ← 21 indicateurs techniques
│   ├── targets.py                 ← Construction des labels
│   ├── sentiment.py               ← Score QDM (Fear&Greed + FinBERT + Momentum)
│   ├── advanced_models.py         ← PatchTST, LSTM-Attention, CNN-Transformer, TFT-Lite
│   ├── train_utils.py             ← Dataset, scaler, loaders, entraînement
│   └── metrics.py                 ← Sharpe, MaxDD, WinRate
│
├── notebooks/                     ← Notebooks Colab (ordre d'exécution)
│   ├── NB00_Sentiment_QDM.ipynb   ← [OPTIONNEL] Score sentiment
│   ├── NB01_Setup_Donnees_Features.ipynb  ← [REQUIS 1] Données + features
│   ├── NB02_Entrainement_Backtest.ipynb   ← [REQUIS 2] Entraînement
│   ├── NB03_Optuna_Pareto.ipynb   ← [OPTIONNEL] Optimisation HPO
│   ├── NB04_Ensemble_WalkForward.ipynb    ← [REQUIS 3] Ensemble + live
│   └── NB05_EDA.ipynb             ← [OPTIONNEL] Analyse exploratoire
│
├── web/
│   ├── dashboard.html             ← Dashboard interactif (aucune dépendance)
│   └── app.py                     ← API FastAPI (prédictions live)
│
├── docker/
│   ├── Dockerfile                 ← Image Docker
│   └── docker-compose.yml         ← API + MLFlow + Jupyter
│
├── data/
│   ├── raw/                       ← market_data.csv (généré par NB01)
│   ├── processed/                 ← sentiment_market_aligned.csv
│   └── sentiment_raw/             ← fear_greed.csv, finbert_scores.csv
│
├── models/                        ← Poids sauvegardés (*_final.pt, *_tuned.pt)
├── artifacts/                     ← Métriques JSON, graphiques PNG
└── logs/                          ← Logs d'entraînement
```

---

## 3. Installation rapide

### 3.1 Pré-requis

- Python 3.10 ou 3.11
- pip ≥ 23
- (Optionnel) CUDA 12.1 pour GPU

### 3.2 Installation locale

```bash
# Cloner / décompresser le projet
cd trading_ia

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Installer les dépendances
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **GPU** : Remplacez la ligne torch par :
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 4. Exécution sur Google Colab

### Ordre recommandé des notebooks

```
NB00  (optionnel)  →  NB01  →  NB05 (EDA, optionnel)  →  NB02  →  NB03 (optionnel)  →  NB04
```

### Étape 0 — Préparer Google Drive

1. Ouvrez [drive.google.com](https://drive.google.com)
2. Créez le dossier `Mon Drive / trading_ia /`
3. Uploadez tout le contenu du projet dans ce dossier
4. Structure attendue sur Drive :
   ```
   Mon Drive/trading_ia/
   ├── config.yaml
   ├── src/
   ├── notebooks/
   ├── data/        (dossier vide, sera rempli)
   ├── models/      (dossier vide)
   └── artifacts/   (dossier vide)
   ```

### Étape 1 — NB01 : Données + Features (≈ 10 min)

**Ce notebook fait :**
- Monte Google Drive
- Installe les dépendances
- Télécharge les données OHLCV depuis Binance (API publique)
- Calcule les 21 features techniques
- Construit les labels (SELL/HOLD/BUY)
- Sauvegarde `data/raw/market_data.csv`

```
Colab → Ouvrir → NB01_Setup_Donnees_Features.ipynb → Exécuter tout
```

> ℹ️ Aucune clé API requise. L'API Binance publique suffit pour les données historiques.

### Étape 2 — NB00 : Sentiment QDM (optionnel, ≈ 15-30 min)

**Ce notebook fait :**
- Télécharge l'index Fear & Greed
- Calcule les scores FinBERT sur les titres crypto
- Combine en score QDM temporel
- Sauvegarde `data/processed/sentiment_market_aligned.csv`

> ⚠️ FinBERT nécessite ~2 Go de RAM GPU. Sur Colab gratuit, utilisez le runtime GPU.
> Si le scoring FinBERT échoue, un fallback synthétique est utilisé automatiquement.

### Étape 3 — NB05 : EDA (optionnel, ≈ 5 min)

Analyse exploratoire complète : distributions, corrélations, stationnarité, importance des features.  
**Requiert NB01 exécuté au préalable.**

### Étape 4 — NB02 : Entraînement + Backtest (≈ 20-40 min selon GPU)

**Ce notebook fait :**
- Entraîne les 4 modèles (configurable via `MODELS_TO_TRAIN`)
- Évalue sur le test set (F1, AUC)
- Exécute le backtest (Sharpe, MaxDD, WinRate)
- Log dans MLFlow (si disponible)
- Sauvegarde les poids `models/*_final.pt`
- Génère les graphiques de performance

```python
# Pour entraîner seulement certains modèles :
MODELS_TO_TRAIN = ["patchtst", "lstm_attention"]   # modifier dans la 1ère cellule
```

> 💡 **Conseil GPU** : Dans Colab, allez dans `Exécution → Modifier le type d'exécution → GPU T4`

### Étape 5 — NB03 : Optimisation Optuna (optionnel, ≈ 30-60 min)

- Optimisation multi-objectif : F1-macro + Sharpe ratio
- NSGAIISampler (algorithme évolutionnaire)
- Visualisation du front de Pareto
- Sauvegarde les meilleurs hyperparamètres `models/*_tuned.pt`

### Étape 6 — NB04 : Ensemble + Walk-Forward + Live

- Charge les modèles (`_tuned.pt` en priorité, sinon `_final.pt`)
- Crée l'ensemble pondéré (poids ∝ F1)
- Backtest de l'ensemble complet
- Walk-forward sur 4 fenêtres temporelles
- **Prédiction live** sur les dernières données Binance
- Téléchargement de tous les livrables

---

## 5. Exécution locale

### 5.1 Télécharger les données

```bash
python -c "
from src.utils import load_config
from src.data  import download_binance_ohlcv
cfg = load_config()
df  = download_binance_ohlcv(cfg['market']['symbols'],
                              cfg['market']['interval'],
                              cfg['market']['lookback_days'])
df.to_csv('data/raw/market_data.csv', index=False)
print(f'Données sauvegardées : {df.shape}')
"
```

### 5.2 Entraîner un modèle

```bash
# Exemple PatchTST sur BTC
python -c "
from src.utils        import load_config, set_seed
from src.data         import load_market_data
from src.features     import add_features
from src.targets      import build_target
from src.train_utils  import make_loaders, run_epoch, evaluate
from src.advanced_models import build_model
import torch

set_seed(42)
cfg   = load_config()
df    = load_market_data(cfg)
df    = add_features(df)
df    = build_target(df, cfg)
model = build_model('patchtst', cfg)
# ... voir NB02 pour le pipeline complet
"
```

### 5.3 Lancer l'API

```bash
cd trading_ia
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
# Dashboard : http://localhost:8000
# API docs  : http://localhost:8000/docs
```

---

## 6. Docker

### 6.1 Build et démarrage

```bash
cd trading_ia

# Build l'image
docker build -f docker/Dockerfile -t trading-ia .

# Démarrage complet (API + MLFlow)
docker compose -f docker/docker-compose.yml up -d

# Avec Jupyter Lab (mode développement)
docker compose -f docker/docker-compose.yml --profile dev up -d
```

### 6.2 Services disponibles

| Service     | URL                       | Description              |
|-------------|---------------------------|--------------------------|
| Dashboard   | http://localhost:8000      | Interface principale     |
| API docs    | http://localhost:8000/docs | Swagger UI (FastAPI)     |
| MLFlow UI   | http://localhost:5000      | Tracking expériences     |
| Jupyter Lab | http://localhost:8888      | Notebooks (profil dev)   |

### 6.3 Volumes montés

```
./data/      → /app/data/
./models/    → /app/models/
./artifacts/ → /app/artifacts/
./logs/      → /app/logs/
```

Les données et modèles sont persistants entre les redémarrages.

### 6.4 Arrêt

```bash
docker compose -f docker/docker-compose.yml down
```

---

## 7. Dashboard & API

### Dashboard HTML

Le fichier `web/dashboard.html` fonctionne de deux façons :

**Mode statique (sans API)** : Ouvrir directement dans un navigateur.  
Les données sont simulées pour démonstration.

**Mode connecté (avec API)** : Lancer l'API puis ouvrir `http://localhost:8000`.  
Les données sont chargées en temps réel depuis l'API.

### Endpoints API

```
GET /healthz               → Statut du serveur
GET /api/predict/{asset}   → Prédiction pour un actif (ex: BTCUSDT)
GET /api/predict           → Prédictions pour tous les actifs
GET /api/metrics           → Métriques des dernières expériences
GET /api/dashboard         → Payload complet pour le dashboard
GET /docs                  → Documentation Swagger
```

**Exemple de réponse `/api/predict/BTCUSDT` :**

```json
{
  "asset": "BTCUSDT",
  "signal": "BUY",
  "confidence": 0.72,
  "probabilities": {
    "SELL": 0.11,
    "HOLD": 0.17,
    "BUY": 0.72
  },
  "models_used": ["patchtst", "lstm_attention", "cnn_transformer"],
  "timestamp": "2024-03-15T14:30:00+00:00"
}
```

---

## 8. MLRun / MLFlow

### MLFlow (intégré dans NB02)

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")   # ou URI Docker
mlflow.set_experiment("trading_ia_v2")

with mlflow.start_run(run_name="patchtst_BTC"):
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.pytorch.log_model(model, "model")
```

### MLRun (optionnel)

Si MLRun est installé (`pip install mlrun`), le projet est automatiquement initialisé :

```python
from src.utils import get_mlrun_project
project = get_mlrun_project("trading_ia")
# project est None si MLRun non disponible
```

Les fonctions de NB02 détectent automatiquement si MLRun est disponible et utilisent  
MLFlow en fallback sinon.

### Visualiser les expériences

```bash
# Démarrer MLFlow UI localement
mlflow ui --port 5000 --backend-store-uri sqlite:///artifacts/mlflow.db
# Accès : http://localhost:5000
```

---

## 9. Configuration

Tous les hyperparamètres sont centralisés dans `config.yaml` :

```yaml
market:
  symbols:      [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  interval:     "1h"
  lookback_days: 180

model:
  seq_len:    96      # Fenêtre d'entrée (bougies)
  patch_len:  12      # Taille des patches (PatchTST)
  d_model:    64      # Dimension du modèle
  n_heads:    4       # Têtes d'attention
  n_layers:   3       # Couches Transformer
  dropout:    0.1

training:
  batch_size: 64
  epochs:     20
  lr:         3.0e-4
  class_weight_multipliers: [1.0, 1.8, 1.0]  # Sell, Hold, Buy

target:
  horizon:        12   # Horizon de prédiction (heures)
  buy_threshold:  0.008
  sell_threshold: -0.008

backtest:
  fee_bps:      5
  slippage_bps: 2
  threshold:    0.45   # Probabilité minimale pour déclencher un signal
  allow_short:  true
```

---

## 10. Description des modules src/

| Module              | Rôle principal                                            |
|---------------------|-----------------------------------------------------------|
| `utils.py`          | `load_config`, `set_seed`, `save_json`, `setup_logging`  |
| `data.py`           | Téléchargement Binance (retry×5), chargement CSV         |
| `features.py`       | 21 features : rendements, RSI, vol., Z-score, momentum   |
| `targets.py`        | Labels SELL/HOLD/BUY (rendement futur ±seuil)            |
| `sentiment.py`      | Score QDM : 50% Fear&Greed + 30% FinBERT + 20% Momentum |
| `advanced_models.py`| 4 architectures deep learning + `build_model()` factory  |
| `train_utils.py`    | `SeqDataset`, `WeightedRandomSampler`, `run_epoch`       |
| `metrics.py`        | Sharpe annualisé, MaxDD, WinRate (trades réels seulement)|

### Features calculées (21 colonnes)

```
ret_1h, ret_4h, ret_24h, ret_168h,   ← Rendements multi-horizon
sma_ratio_24, sma_ratio_168,          ← Prix / SMA
ema_ratio_12, ema_ratio_26,           ← Prix / EMA
rsi_14, rsi_28,                       ← RSI Wilder
vol_24h, vol_168h,                    ← Volatilité rolling
z_score_24h,                          ← Z-score 24h
body_ratio, upper_wick, lower_wick,   ← Structure bougie
momentum_12h, momentum_24h,           ← Momentum
hour_sin, hour_cos,                   ← Calendrier cyclique
sentiment_score                       ← Score QDM
```

---

## 11. Modèles de deep learning

### PatchTST (recommandé)
- Division de la séquence en patches de 12 bougies
- Encoder Transformer pré-normalisé (pre-LN)
- Global Average Pooling → classifieur 3 classes
- Inspiré de "A Time Series is Worth 64 Words" (Nie et al., 2023)

### LSTM-Attention
- BiLSTM 2 couches + mécanisme d'attention temporelle
- Softmax sur les poids d'attention → contexte pondéré
- Simple et efficace sur des séquences courtes

### CNN-Transformer
- Convolutions 1D parallèles (kernels 3, 7, 15) → extraction multi-échelle
- Concaténation → Transformer 2 couches
- Capture patterns locaux ET dépendances longues

### TFT-Lite
- Gated Residual Network (GRN) pour la sélection des variables
- LSTM + attention temporelle interprétable
- Inspiré du Temporal Fusion Transformer (Lim et al., 2021)

---

## 12. Score QDM Sentiment

Le score QDM est un composite temporel aligné sur les données OHLCV :

```
QDM(t) = 0.5 × FearGreed(t)_norm
       + 0.3 × FinBERT(asset,t)_norm
       + 0.2 × PriceMomentum(t)_norm
```

- **Fear & Greed** : Téléchargé depuis l'API alternative.me (données quotidiennes interpolées)
- **FinBERT** : Modèle `ProsusAI/finbert` — scoring de sentiment sur titres cryptonews
- **Momentum** : SMA(7j) / SMA(30j) — 1 normalisé en [-1, 1]

> Validation automatique : std > 0.01 et n_unique > 10. Alerte si score statique détecté.

---

## 13. Métriques et résultats

### Métriques de classification (test set)

| Métrique   | Description                           |
|------------|---------------------------------------|
| F1-macro   | Moyenne non pondérée des F1 par classe|
| AUC OvR    | Area Under Curve (One-vs-Rest)        |
| Accuracy   | Taux de classification correcte       |

### Métriques de trading (backtest)

| Métrique     | Description                                    |
|--------------|------------------------------------------------|
| Sharpe ratio | Rendement annualisé / volatilité (√(365×24))  |
| Max Drawdown | Perte maximale pic-à-creux                    |
| Win Rate     | % de trades positifs (trades réels seulement) |
| Avg Exposure | % du temps en position                        |
| N Trades     | Nombre total de trades                        |

> ⚠️ **CAGR non affiché** : Non interprétable sur 6 mois de données crypto.  
> ⚠️ **Buy&Hold non comparé** : Dépend fortement de la période choisie.

---

## 14. FAQ

**Q : L'API Binance est bloquée dans mon pays**  
R : Utilisez un VPN, ou modifiez `data.py` pour utiliser une autre source (Yahoo Finance via `yfinance`).

**Q : FinBERT est trop lent / out of memory**  
R : Le fallback synthétique s'active automatiquement. Le modèle tourne sans sentiment si nécessaire.

**Q : Les performances sont mauvaises**  
R : Essayez : (1) Réduire `seq_len` à 48, (2) Augmenter `epochs` à 30, (3) Exécuter NB03 pour HPO.

**Q : Comment ajouter un nouvel actif ?**  
R : Ajoutez le symbole dans `config.yaml` → `market.symbols` et relancez depuis NB01.

**Q : Comment changer l'horizon de prédiction ?**  
R : Modifiez `target.horizon` dans `config.yaml`. Valeurs raisonnables : 6, 12, 24 heures.

**Q : Le walk-forward prend trop de temps**  
R : Réduisez `n_folds` dans NB04 ou augmentez `min_train_hours`.

**Q : Comment interpréter le dashboard ?**  
R : Les signaux avec `confidence > 0.60` sont les plus fiables. En dessous de 0.45, préférer HOLD.

---

## ⚠️ Avertissement

Ce projet est fourni à des fins **éducatives et de recherche uniquement**.  
Il ne constitue **pas un conseil financier**.  
Le trading de crypto-actifs comporte des risques de perte en capital.  
Les performances passées ne garantissent pas les performances futures.

---

*Version 2.0 — Projet Trading IA Multi-Modèles*
