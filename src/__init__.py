"""
src/  —  Système de Trading par Intelligence Artificielle
=========================================================
Modules :
  utils          Utilitaires (config, seed, JSON)
  data           Chargement et téléchargement des données de marché
  features       Feature engineering (20 indicateurs techniques)
  targets        Construction des labels (Sell/Hold/Buy)
  model          PatchTSTLite (Transformer sur patches)
  advanced_models LSTMAttention, TFTLite, CNNTransformer, EnsembleModel
  sentiment      Données de sentiment financier
  metrics        Métriques financières (source unique)
  train_utils    Utilitaires d'entraînement partagés (SeqDataset, evaluate...)
  backtest       Backtest multi-actifs portefeuille égal-pondéré
  realtime       Buffer Binance WebSocket temps réel
  train_advanced Entraînement des 4 modèles
  train_patchtst Entraînement PatchTST seul (compatible Colab)
  ensemble_final Ensemble pondéré final
  tune_lstm_optuna  Optuna mono-objectif F1 (LSTM)
  tune_all_optuna   Optuna multi-objectif Pareto (F1 + Sharpe)
  walk_forward   Walk-forward mensuel
  api            API FastAPI production
  download_market_data  Téléchargement initial des données
  prepare_datasets      Fusion marché + sentiment
  final_evaluation      Rapport de synthèse
"""
