import json
import os

def make_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

os.makedirs("notebooks", exist_ok=True)

# 01_data_exploration
make_nb([
    {"cell_type": "markdown", "metadata": {}, "source": ["# 01 Data Exploration\nExplore raw and processed HaluEval claims dataset."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('../data/processed/claims_all.csv')\n",
        "print('Total claims:', len(df))\n",
        "print(df['label_name'].value_counts())\n",
        "df.head(3)"
    ]}
], "notebooks/01_data_exploration.ipynb")

# 02_feature_engineering
make_nb([
    {"cell_type": "markdown", "metadata": {}, "source": ["# 02 Feature Engineering\nInspect semantic and linguistic features extracted from claims."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import pandas as pd\n",
        "import json\n",
        "feat_df = pd.read_csv('../data/train/train_features.csv')\n",
        "with open('../models/feature_names.json') as f:\n",
        "    meta = json.load(f)\n",
        "print('Feature dimensions:', feat_df.shape)\n",
        "feat_df[meta['all_features'][:8]].describe()"
    ]}
], "notebooks/02_feature_engineering.ipynb")

# 03_baselines
make_nb([
    {"cell_type": "markdown", "metadata": {}, "source": ["# 03 Baselines Evaluation\nCompare Linguistic baseline vs Semantic baseline."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import json\n",
        "with open('../models/training_summary.json') as f:\n",
        "    summary = json.load(f)\n",
        "print('Best model:', summary['best_model'])\n",
        "for k, v in summary['results'].items():\n",
        "    print(f\"{k}: Val F1 = {v['metrics']['f1']:.4f}, ROC-AUC = {v['metrics']['roc_auc']:.4f}\")"
    ]}
], "notebooks/03_baselines.ipynb")

# 04_model_comparison
make_nb([
    {"cell_type": "markdown", "metadata": {}, "source": ["# 04 Model Comparison & Ablation Analysis\nExamine evaluation results on held-out test split."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import json\n",
        "import pandas as pd\n",
        "with open('../results/evaluation_report.json') as f:\n",
        "    report = json.load(f)\n",
        "eval_df = pd.DataFrame(report['model_evaluations'])\n",
        "print(eval_df[['model_name', 'feature_set', 'f1', 'accuracy', 'roc_auc', 'latency_ms_per_claim']])"
    ]}
], "notebooks/04_model_comparison.ipynb")

print("Created all 4 exploration and comparison notebooks.")
