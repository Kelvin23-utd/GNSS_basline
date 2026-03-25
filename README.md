# How to Run
Make sure all dataset files are placed in the `data/` folder.

### Run all baselines individually:

python svm_baseline.py
python xgboost_baseline.py
python cnn_baseline.py
python lstm_baseline.py
python transformer_baseline.py

# Project Structure

  baseline/
├── data/
│ ├── W_FineSat_X.pth
│ ├── W_FineSat_Y.pth
│ ├── WO_FineSat_X.pth
│ ├── WO_FineSat_Y.pth
│
├── figures/
│ ├── confusion_svm_finesat.png
│ ├── confusion_xgboost_finesat.png
│ ├── confusion_transformer_finesat.png
│ └── ...
│
├── numbers/
│ ├── baseline_results.json
│ ├── confusion_matrices.json
│
├── svm_baseline.py
├── xgboost_baseline.py
├── cnn_baseline.py
├── lstm_baseline.py
├── transformer_baseline.py
│
└── README.md

Summary table ( for now， still updating)

Accuracy (mean ± std)

| Model        | W/ FineSat (%)       | WO/ FineSat (%)      |
|--------------|----------------------|----------------------|
| SVM (RBF)    | 93.8 ± 2.4           | 73.4 ± 1.7           |
| XGBoost      | **95.6 ± 2.5**       | **84.5 ± 1.3**       |
| CNN          | 91.2 ± 2.8           | 84.0 ± 1.8           |
| Transformer  | 80.6 ± 2.5           | 60.2 ± 2.1           |
| LSTM         | 40.7 ± 0.7           | 45.9 ± 3.3           |


Macro F1 Score (mean ± std)

| Model        | W/ FineSat (%)       | WO/ FineSat (%)      |
|--------------|----------------------|----------------------|
| SVM (RBF)    | 93.9 ± 2.3           | 71.6                 |
| XGBoost      | **95.6**             | **84.5**             |
| CNN          | 91.2                 | 83.2                 |
| Transformer  | 80.4 ± 2.8           | 58.7 ± 3.1           |
| LSTM         | 39.2                 | 39.6                 |

Observation
- XGBoost achieves the best performance on both W_FineSat and WO_FineSat datasets.
- FineSat preprocessing significantly improves performance across all models.
- Transformer performs moderately but is limited by small dataset size and tabular feature structure.
- CNN shows strong performance, close to tree-based methods.
- LSTM performs poorly, indicating weak sequential structure in the data.
