# Empirical Evaluation Results on Held-Out Test Set (2,969 Claims)

| Model | Feature Set | F1-Score | Precision | Recall | Accuracy | ROC-AUC | Latency (ms/claim) | Throughput (claims/s) | Model Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Linguistic)** | Linguistic | 0.7370 | 0.6990 | 0.7793 | 0.7067 | 0.7812 | 0.007 | 135,284 | 11.15 |
| **Logistic Regression (Semantic)** | Semantic | 0.6564 | 0.6263 | 0.6895 | 0.6194 | 0.6694 | 0.000 | 4,305,324 | 0.00 |
| **Random Forest (Semantic)** | Semantic | 0.7120 | 0.6986 | 0.7259 | 0.6904 | 0.7766 | 0.007 | 134,955 | 6.33 |
| **Random Forest (Hybrid)** | Hybrid | 0.7838 | 0.7680 | 0.8004 | 0.7673 | 0.8550 | 0.011 | 90,752 | 21.95 |
| **XGBoost (Hybrid)** | Hybrid | 0.7798 | 0.7620 | 0.7986 | 0.7623 | 0.8549 | 0.001 | 980,833 | 0.36 |

## Feature Ablation Study

| Configuration | Features | F1-Score | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Hybrid** | 39 | 0.7831 | 0.7585 | 0.7437 | 0.8268 | 0.8509 |
| **Semantic Only** | 4 | 0.7147 | 0.6902 | 0.6945 | 0.7362 | 0.7727 |
| **Linguistic Only** | 35 | 0.7425 | 0.7065 | 0.6907 | 0.8026 | 0.7799 |
| **No Entities** | 29 | 0.7848 | 0.7625 | 0.7513 | 0.8214 | 0.8533 |
| **No Numerical** | 33 | 0.7831 | 0.7614 | 0.7519 | 0.8170 | 0.8516 |
| **No POS Ratios** | 29 | 0.7748 | 0.7453 | 0.7255 | 0.8313 | 0.8398 |
| **No Certainty Lex** | 35 | 0.7855 | 0.7611 | 0.7459 | 0.8295 | 0.8514 |
