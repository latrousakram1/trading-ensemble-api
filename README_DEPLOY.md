# 🚀 Guide de Déploiement - Trading AI Ensemble

Guide complet pour déployer le système de trading IA multi-modèles en production.

## 📋 Prérequis

- **Docker** & **Docker Compose** (version 20.10+)
- **4GB RAM** minimum disponible
- **2GB espace disque** pour les modèles
- Accès internet pour télécharger les images Docker

## 🏗️ Structure Requise

Assurez-vous que votre répertoire de déploiement contient :

```
trading-deployment/
├── src/
│   ├── api.py                    # API FastAPI principale
│   ├── advanced_models.py        # Modèles avancés (LSTM, CNN)
│   ├── model.py                  # Implémentation PatchTST
│   ├── data.py                   # Chargement des données
│   ├── features.py               # Ingénierie des features
│   ├── targets.py                # Génération des labels
│   ├── sentiment.py              # Analyse de sentiment
│   ├── backtest.py               # Moteur de backtesting
│   └── utils.py                  # Utilitaires
├── models/
│   ├── patchtst_final.pt         # ⚠️  RÉQUIS - Modèle PatchTST
│   ├── lstm_attention_final.pt   # ⚠️  RÉQUIS - Modèle LSTM
│   ├── cnn_transformer_final.pt  # ⚠️  RÉQUIS - Modèle CNN
│   └── ensemble_final.pt         # OPTIONNEL - Configuration ensemble
├── data/
│   └── processed/
│       └── sentiment_market_aligned.csv  # Données d'entraînement
├── artifacts/
│   ├── patchtst_metrics.json
│   ├── lstm_attention_metrics.json
│   ├── cnn_transformer_metrics.json
│   ├── ensemble_metrics.json
│   └── backtest_metrics.json
├── config.yaml                   # Configuration système
├── Dockerfile                    # Image de production
├── docker-compose.yml            # Orchestration
├── requirements-deploy.txt       # Dépendances production
├── test_api.py                   # Tests de validation
└── README_DEPLOY.md             # Ce fichier
```

## ⚠️ Points Critiques

### Modèles Obligatoires

**Les fichiers de modèles suivants DOIVENT être présents dans `models/` :**
- `patchtst_final.pt` (~50MB)
- `lstm_attention_final.pt` (~30MB)
- `cnn_transformer_final.pt` (~40MB)

*Ces modèles sont générés lors de l'entraînement. Téléchargez-les depuis votre environnement Colab ou serveur d'entraînement.*

### Données Requises

- `data/processed/sentiment_market_aligned.csv` : Données fusionnées marché + sentiment
- Fichier de configuration `config.yaml` valide

## 🚀 Déploiement en 5 Étapes

### Étape 1 : Préparation

```bash
# Cloner ou copier le code de déploiement
git clone https://github.com/votre-repo/trading-ensemble-api.git
cd trading-ensemble-api

# Vérifier la structure
ls -la
# Assurer que models/ contient les 3 fichiers .pt
ls models/
```

### Étape 2 : Configuration

Vérifier et ajuster `config.yaml` :

```yaml
project:
  name: patchtst-trading-multi-model

market:
  symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  interval: 1h

model:
  seq_len: 96
  patch_len: 12
```

### Étape 3 : Build Docker

```bash
# Construire l'image (multi-étapes pour optimisation)
docker compose build

# Vérifier l'image construite
docker images | grep trading
```

### Étape 4 : Lancement

```bash
# Lancer en arrière-plan
docker compose up -d

# Vérifier les logs
docker compose logs -f api
```

### Étape 5 : Validation

```bash
# Test de santé
curl http://localhost:8000/health

# Test complet de l'API
python test_api.py

# Vérifier les modèles chargés
curl http://localhost:8000/models
```

## 🔍 Dépannage

### Problèmes Courants

#### "Model file not found"
```
ERREUR: torch.load() missing file: models/patchtst_final.pt
```
**Solution** : Vérifier que tous les fichiers `.pt` sont présents dans `models/`

#### "Port already in use"
```
ERROR: Port 8000 already in use
```
**Solution** :
```bash
# Changer le port dans docker-compose.yml
ports:
  - "8001:8000"  # Utiliser 8001 au lieu de 8000

# Redémarrer
docker compose down && docker compose up -d
```

#### "Insufficient memory"
```
ERROR: Container killed due to out of memory
```
**Solution** : Augmenter la limite mémoire Docker ou réduire `model.d_model` dans config.yaml

#### "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Solution** : Le conteneur utilise automatiquement CPU si CUDA n'est pas disponible. Vérifier avec :
```bash
curl http://localhost:8000/health | grep device
```

### Logs et Monitoring

```bash
# Logs en continu
docker compose logs -f

# Logs d'une heure
docker compose logs --since 1h

# Ressources utilisées
docker stats

# Accéder au conteneur
docker compose exec api bash
```

## 🧪 Tests de Validation

### Test Automatisé

```bash
# Exécuter tous les tests
python test_api.py
```

### Tests Manuels

```bash
# Santé générale
curl http://localhost:8000/

# Liste des modèles
curl http://localhost:8000/models

# Prédiction ensemble (avec données d'exemple)
curl -X POST "http://localhost:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "market_data": [10000, 10100, 9900, 10200, 9800],
    "sentiment_data": [0.1, 0.2, -0.1, 0.3, -0.2]
  }'
```

## 🔧 Développement Local (Sans Docker)

Pour le développement ou debugging :

```bash
# Installation des dépendances
pip install -r requirements-deploy.txt

# Variables d'environnement (optionnel)
export MODEL_DIR=models/
export CONFIG_FILE=config.yaml

# Lancement du serveur
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Avec logs détaillés
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## 📊 Métriques et Monitoring

### Endpoints de Monitoring

- `/health` : État détaillé des modèles et ressources
- `/metrics` : Métriques de performance (si Prometheus activé)
- `/` : Statut général avec poids de l'ensemble

### Logs Applicatifs

Les logs sont structurés avec niveaux :
- `INFO` : Opérations normales
- `WARNING` : Anomalies non-critiques
- `ERROR` : Erreurs nécessitant attention

## 🔄 Mise à Jour

### Mise à jour des Modèles

```bash
# Arrêter le service
docker compose down

# Remplacer les fichiers .pt dans models/
cp nouveaux_modeles/*.pt models/

# Redémarrer
docker compose up -d --build
```

### Mise à jour du Code

```bash
# Pull des changements
git pull origin main

# Rebuild et redémarrage
docker compose down
docker compose up -d --build
```

## 🛡️ Sécurité

### Recommandations

- Ne pas exposer l'API directement sur internet sans authentification
- Utiliser HTTPS en production
- Limiter les requêtes par minute
- Monitorer les logs pour détecter les abus

### Variables Sensibles

Créer un fichier `.env` pour les secrets :
```bash
# .env
API_KEY=votre_cle_secrete
DATABASE_URL=postgresql://...
```

## 📞 Support

### Diagnostics

En cas de problème, collecter ces informations :

```bash
# Informations système
docker --version
docker compose version

# État des conteneurs
docker compose ps

# Logs récents
docker compose logs --tail 100

# Ressources
docker system df
```

### Contacts

- 🐛 **Bugs** : Issues GitHub
- 💬 **Questions** : Discussions GitHub
- 📧 **Support** : [votre-email@domain.com]

---

## ✅ Checklist Déploiement

- [ ] Modèles `.pt` présents dans `models/`
- [ ] Données dans `data/processed/`
- [ ] `config.yaml` configuré
- [ ] Docker & Docker Compose installés
- [ ] Port 8000 disponible
- [ ] Tests `python test_api.py` passent
- [ ] API accessible sur `http://localhost:8000`

**Déploiement réussi ! 🎉**

Documentation interactive : **http://localhost:8000/docs**

---

## Exemples d'utilisation

### Prédiction ensemble (recommandé)

```bash
curl -s -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{"asset": "BTCUSDT", "threshold": 0.45, "allow_short": true}' \
  | python -m json.tool
```

Réponse :
```json
{
  "asset": "BTCUSDT",
  "timestamp": "2026-04-12 19:00:00+00:00",
  "close": 71084.18,
  "signal": "HOLD",
  "signal_value": 0.0,
  "confidence": 0.537,
  "prob_sell": 0.251,
  "prob_hold": 0.537,
  "prob_buy": 0.212,
  "weights": {
    "patchtst": 0.309,
    "lstm_attention": 0.349,
    "cnn_transformer": 0.343
  },
  "model_contributions": { ... },
  "latency_ms": 85.3
}
```

### Prédiction modèle individuel

```bash
curl -s -X POST http://localhost:8000/predict/lstm_attention \
  -H "Content-Type: application/json" \
  -d '{"asset": "ETHUSDT", "threshold": 0.45}' \
  | python -m json.tool
```

### Scan tous les actifs

```bash
curl "http://localhost:8000/predict/ensemble/scan?threshold=0.45&allow_short=true" \
  | python -m json.tool
```

### Python

```python
import requests

# Signal ensemble pour BTCUSDT
r = requests.post('http://localhost:8000/predict/ensemble', json={
    'asset': 'BTCUSDT',
    'threshold': 0.45,
    'allow_short': True,
})
d = r.json()
print(f"{d['asset']} → {d['signal']} (confiance={d['confidence']:.1%})")
```

---

## Dépannage

### Modèle non chargé

```bash
# Vérifier que les fichiers .pt sont présents
ls -lh models/*.pt

# Vérifier les logs du container
docker compose logs trading-api
```

### Données non chargées

```bash
# Vérifier que le CSV existe
ls -lh data/processed/sentiment_market_aligned.csv
# Ou : ls -lh data/raw/market_data.csv

# Vérifier le config.yaml
grep -A2 "sentiment:" config.yaml
```

### Réinitialiser et reconstruire

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## Logs en temps réel

```bash
docker compose logs -f trading-api
```

---

## Poids de l'ensemble

Les poids sont calculés automatiquement au démarrage depuis les fichiers
`artifacts/{model}_metrics.json` (champ `test_f1_macro`).

Si ces fichiers sont absents, les poids sont égaux (0.333 chacun).

Pour forcer des poids manuels, modifiez `config.yaml` :

```yaml
model:
  ensemble_members: [patchtst, lstm_attention, cnn_transformer]
```

---

*Rapport généré — Système PatchTST Trading Multi-Modèles — Avril 2026*
