from __future__ import annotations
import json
from pathlib import Path

try:
    from src.utils import load_config
except ImportError:
    from utils import load_config


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    cfg = load_config()
    artifact_dir = Path(cfg['paths']['artifact_dir'])
    patch = _load_json(artifact_dir / 'patchtst_metrics.json')
    optuna = _load_json(artifact_dir / 'optuna_best.json')
    dqn = _load_json(artifact_dir / 'dqn_metrics.json')
    backtest = _load_json(artifact_dir / 'backtest_metrics.json')

    lines = []
    lines.append('# Rapport final de synthèse')
    lines.append('')
    lines.append('## 1. Résumé du projet')
    lines.append("Ce projet combine un modèle PatchTST-Lite pour la prédiction de signaux de trading, une composante sentiment financière, et une baseline Deep Q-Learning. L'objectif n'est pas seulement de maximiser l'accuracy, mais de produire des signaux utilisables dans un backtest hors échantillon.")
    lines.append('')
    lines.append('## 2. Résultats PatchTST')
    lines.append(f"- Accuracy test: {patch.get('test_acc')}")
    lines.append(f"- Macro F1 test: {patch.get('test_f1_macro')}")
    lines.append(f"- AUC OVR test: {patch.get('test_auc_ovr')}")
    lines.append("Analyse: la Macro F1 est la métrique la plus importante ici. Si elle reste faible, cela signifie que le modèle privilégie encore trop certaines classes et qu'il faut améliorer les labels, l'équilibrage et les features.")
    lines.append('')
    lines.append('## 3. Tuning Optuna')
    lines.append(f"- Meilleure valeur: {optuna.get('best_value')}")
    lines.append(f"- Meilleurs paramètres: {optuna.get('best_params')}")
    lines.append('')
    lines.append('## 4. Deep Q-Learning')
    lines.append(f"- Episodes: {dqn.get('episodes')}")
    lines.append(f"- Epsilon final: {dqn.get('final_epsilon')}")
    lines.append(f"- Mean loss: {dqn.get('mean_loss')}")
    lines.append(f"- Mean reward: {dqn.get('mean_reward')}")
    lines.append("Analyse: cette baseline sert surtout à comparer une approche par renforcement. Une loss élevée ou un epsilon final trop grand indique un apprentissage encore instable.")
    lines.append('')
    lines.append('## 5. Backtest hors échantillon')
    lines.append(f"- Total return: {backtest.get('total_return')}")
    lines.append(f"- Sharpe: {backtest.get('sharpe')}")
    lines.append(f"- Max drawdown: {backtest.get('max_drawdown')}")
    lines.append(f"- Win rate: {backtest.get('win_rate')}")
    lines.append(f"- Exposure: {backtest.get('exposure')}")
    lines.append("Analyse: ce backtest est calculé uniquement sur le jeu de test, avec seuil de probabilité, décalage de position et coûts de transaction. Il est donc plus crédible que la première version.")
    lines.append('')
    lines.append('## 6. Conclusion')
    lines.append("Le projet est reproductible et correctement structuré. Toutefois, la qualité finale dépend surtout de trois éléments: la définition des labels, l'équilibrage des classes et le réalisme du backtest. Les prochaines améliorations doivent prioriser ces trois axes.")

    report_path = Path('reports/final_report.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(report_path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
