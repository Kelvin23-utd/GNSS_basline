# Zhang's Task List — GNSS Gesture Recognition Baselines

**Role:** Reproduce FineSat results + build GNSS-only baselines
**Supervisor check-ins:** End of each week

---

## Required Reading (Before Anything Else)

Read the FineSat paper: `PerCom_25__FineSat_Final.pdf`

Focus on:
- Section IV (FineSat Design) — understand how the signal is processed
- Section VI-C (Case Study 2: Gesture Recognition) — this is what you will reproduce
- Figure 13 — your target results
- Note: SVM classifier, 5-fold cross-validation, 5 gestures, 1000 samples

After reading, write a **1-paragraph summary** (5-8 sentences) of how gesture recognition works in the paper. Include: what signal is used, what classifier, what accuracy, how many subjects, what gestures. Send to supervisor.

---

## Task 1 — Load and Verify Data

Load both data files and confirm:

```python
import torch

# Load FineSat-processed data
X_finesat = torch.load('data/W_FineSat_X.pth')
Y_finesat = torch.load('data/W_FineSat_Y.pth')

# Load raw data
X_raw = torch.load('data/WO_FineSat_X.pth')
Y_raw = torch.load('data/WO_FineSat_Y.pth')
```

**Output:**
- [ ] Screenshot showing: shape of X `(1000, 200)`, shape of Y `(1000,)`, class distribution (200 per class), min/max values for both versions

---

## Task 2 — Plot Gestures

Plot one sample per class for both raw and FineSat signals (10 plots total).

**Output:**
- [ ] 2 figures (raw and FineSat), each with 5 subplots. Save as PNG.

---

## Task 3 — Reproduce 96.5% SVM (THE KEY TASK)

From the FineSat paper (Section VI-C):
- Data: `W_FineSat_X.pth` (FineSat-processed)
- Classifier: SVM
- Evaluation: **5-fold cross-validation** (NOT 80/20 split)
- Target accuracy: **96.5%**

```
Steps:
1. Load W_FineSat_X.pth and W_FineSat_Y.pth
2. Use sklearn.svm.SVC (try kernel='rbf' first)
3. Use sklearn.model_selection.cross_val_score with cv=5
4. Report mean accuracy across 5 folds
5. Generate confusion matrix (use one fold as example)
```

**Output:**
- [ ] Print: `"5-fold CV accuracy: XX.X% ± X.X%"`
- [ ] Confusion matrix heatmap (save as PNG)
- [ ] Save accuracy and F1 numbers to JSON
- [ ] If accuracy is NOT close to 96.5%, document what you tried and what number you got. Do NOT fabricate results.

---

## Task 4 — Run SVM on Raw Signals

Same as Task 3 but on `WO_FineSat_X.pth`. Paper reports ~83.7%.

**Output:**
- [ ] Print: `"Raw SVM 5-fold CV accuracy: XX.X% ± X.X%"`
- [ ] Confusion matrix heatmap (save as PNG)
- [ ] Save numbers to JSON

---

## Task 5 — XGBoost Baseline

Run XGBoost on both signal versions with 5-fold CV.

**Output:**
- [ ] `"XGBoost W_FineSat: XX.X% ± X.X%"`
- [ ] `"XGBoost WO_FineSat: XX.X% ± X.X%"`
- [ ] Confusion matrices (2 PNGs)
- [ ] Save numbers to JSON

---

## Task 6 — 1D-CNN Baseline

Build a simple 1D-CNN for the (1000, 200) data. Suggested architecture:

```
Conv1d(1, 32, kernel=7) → ReLU → MaxPool
Conv1d(32, 64, kernel=5) → ReLU → MaxPool
Conv1d(64, 128, kernel=3) → ReLU → AdaptiveAvgPool
Linear(128, 5)
```

Use 5-fold CV. Train for 50 epochs per fold. Use Adam optimizer, lr=0.001.

**Output:**
- [ ] `"1D-CNN W_FineSat: XX.X% ± X.X%"`
- [ ] `"1D-CNN WO_FineSat: XX.X% ± X.X%"`
- [ ] Confusion matrices (2 PNGs)
- [ ] Document: architecture, optimizer, learning rate, epochs, batch size
- [ ] Save numbers to JSON

---

## Task 7 — LSTM Baseline

Build an LSTM for sequence classification.

```
LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
Linear(64, 5)
```

Same evaluation: 5-fold CV, 50 epochs, Adam, lr=0.001.

**Output:**
- [ ] `"LSTM W_FineSat: XX.X% ± X.X%"`
- [ ] `"LSTM WO_FineSat: XX.X% ± X.X%"`
- [ ] Confusion matrices (2 PNGs)
- [ ] Save numbers to JSON

---

## Task 8 — Baseline Summary Table

Fill in this table with all results from Tasks 3–7:

| Model | Raw (WO_FineSat) | FineSat (W_FineSat) |
|-------|-------------------|---------------------|
| SVM | ??% ± ??% | ??% (target: 96.5%) |
| XGBoost | ??% ± ??% | ??% ± ??% |
| 1D-CNN | ??% ± ??% | ??% ± ??% |
| LSTM | ??% ± ??% | ??% ± ??% |

---

## Task 9 — Few-Shot GNSS-Only Curve (MOST IMPORTANT)

This answers: **"How does GNSS-only accuracy drop when you have fewer labeled samples?"**

For each label budget K = 10, 25, 50, 100, 200, 500, 1000:
1. Randomly sample K labeled GNSS samples (balanced across 5 classes)
2. Train the best model from Tasks 3–7 on those K samples
3. Test on the remaining samples
4. Repeat with 3 different random seeds
5. Report mean ± std accuracy

**Use `W_FineSat_X.pth` only.**

**Output:**
- [ ] Table:

| K (labels) | Accuracy (mean ± std) |
|------------|----------------------|
| 10 | ??% ± ??% |
| 25 | ??% ± ??% |
| 50 | ??% ± ??% |
| 100 | ??% ± ??% |
| 200 | ??% ± ??% |
| 500 | ??% ± ??% |
| 1000 | ??% ± ??% |

- [ ] Line plot: X-axis = K, Y-axis = accuracy. Include error bars. Save as PNG.
- [ ] Save all numbers to JSON

---

## Task 10 — Few-Shot with Multiple Models

Run the same few-shot experiment (Task 9) for SVM, XGBoost, and best DL model.

**Output:**
- [ ] One plot with 3 lines (SVM, XGBoost, best DL), same X/Y axes. Save as PNG.
- [ ] Save all numbers to JSON

---

## Final Deliverable

Hand off to supervisor:

```
zhang_results/
├── code/
│   ├── 01_data_verification.py
│   ├── 02_svm_baseline.py
│   ├── 03_xgboost_baseline.py
│   ├── 04_cnn_baseline.py
│   ├── 05_lstm_baseline.py
│   └── 06_fewshot_curve.py
├── figures/
│   ├── gesture_plots_raw.png
│   ├── gesture_plots_finesat.png
│   ├── confusion_svm_finesat.png
│   ├── confusion_svm_raw.png
│   ├── confusion_xgboost_finesat.png
│   ├── confusion_xgboost_raw.png
│   ├── confusion_cnn_finesat.png
│   ├── confusion_cnn_raw.png
│   ├── confusion_lstm_finesat.png
│   ├── confusion_lstm_raw.png
│   ├── fewshot_curve.png
│   └── fewshot_curve_multimodel.png
├── numbers/
│   ├── baseline_results.json     # All accuracy/F1 numbers
│   ├── fewshot_results.json      # All few-shot curve numbers
│   └── confusion_matrices.json   # All confusion matrices as arrays
├── results_table.md              # Final summary table
└── README.md                     # How to run everything
```

### Saving Numerical Results

**Every experiment must save its raw numbers to JSON.** Figures are for visualization, but the numbers must be independently accessible. Example format:

```json
// baseline_results.json
{
  "svm_finesat": {"accuracy": 0.965, "std": 0.012, "f1_macro": 0.964, "per_class_f1": [0.97, 0.95, 0.98, 0.96, 0.96]},
  "svm_raw":     {"accuracy": 0.837, "std": 0.015, "f1_macro": 0.835, "per_class_f1": [...]},
  "xgboost_finesat": {...},
  ...
}

// fewshot_results.json
{
  "svm": {"K_10": [0.45, 0.42, 0.48], "K_25": [0.62, 0.58, 0.65], ...},
  "xgboost": {...},
  ...
}

// confusion_matrices.json
{
  "svm_finesat": [[195, 3, 0, 1, 1], [2, 190, ...], ...],
  ...
}
```

Every script must call `json.dump()` to save results at the end. The supervisor will use these JSON files directly for the paper figures.

---

## Rules

1. **Do NOT fabricate results.** If a model gets 40%, report 40%. The supervisor will run your code.
2. **All code must be runnable.** No hardcoded paths. Use relative paths to `data/` folder.
3. **Document everything.** Every hyperparameter, every random seed, every decision.
4. **Ask if stuck.** Do not spend more than 2 hours on a single bug without asking for help.
5. **Weekly check-in is mandatory.** Send deliverables on time even if incomplete — partial results are better than silence.
