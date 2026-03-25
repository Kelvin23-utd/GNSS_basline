import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay



# Paths setup

DATA_DIR = "data"
FIG_DIR = "figures"
NUM_DIR = "numbers"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NUM_DIR, exist_ok=True)



# Load FineSat data

X = torch.load(os.path.join(DATA_DIR, "W_FineSat_X.pth"), weights_only=False)
y = torch.load(os.path.join(DATA_DIR, "W_FineSat_Y.pth"), weights_only=False)

X = np.array(X, dtype=np.float64)
y = np.array(y).astype(int)



# Per-sample standardization

X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


# SVM trials (RBF first)

trials = [
    {"name": "rbf_default", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"name": "rbf_C10", "kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"name": "rbf_C100", "kernel": "rbf", "C": 100.0, "gamma": "scale"},
    {"name": "rbf_gamma_0.01", "kernel": "rbf", "C": 10.0, "gamma": 0.02},
    {"name": "rbf_gamma_0.1", "kernel": "rbf", "C": 10.0, "gamma": 0.005},
]

results = []
best_result = None
best_cfg = None


# 5-fold cross-validation

for cfg in trials:
    model = SVC(kernel=cfg["kernel"], C=cfg["C"], gamma=cfg["gamma"])

    acc_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")

    result = {
        "name": cfg["name"],
        "kernel": cfg["kernel"],
        "C": cfg["C"],
        "gamma": cfg["gamma"],
        "accuracy": float(acc_scores.mean()),
        "std": float(acc_scores.std()),
        "f1_macro": float(f1_scores.mean()),
        "f1_macro_std": float(f1_scores.std()),
    }
    results.append(result)

    if best_result is None or result["accuracy"] > best_result["accuracy"]:
        best_result = result
        best_cfg = cfg


# Print required result

print('5-fold CV accuracy: {:.1f}% ± {:.1f}%'.format(
    best_result["accuracy"] * 100,
    best_result["std"] * 100
))


# Confusion matrix using one fold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = SVC(kernel=best_cfg["kernel"], C=best_cfg["C"], gamma=best_cfg["gamma"])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
per_class_f1 = f1_score(y_test, y_pred, average=None)
example_f1_macro = f1_score(y_test, y_pred, average="macro")

# Confusion matrix 

cm = confusion_matrix(y_test, y_pred, normalize="true")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1, 2, 3, 4]
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2f",  
    colorbar=False
)

ax.set_title("SVM Confusion Matrix - FineSat")
plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "confusion_svm_finesat.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Save baseline results to JSON
baseline_path = os.path.join(NUM_DIR, "baseline_results.json")

if os.path.exists(baseline_path):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)
else:
    baseline_results = {}

baseline_results["svm_finesat"] = {
    "selected_trial": best_result["name"],
    "accuracy": best_result["accuracy"],
    "std": best_result["std"],
    "f1_macro": best_result["f1_macro"],
    "f1_macro_std": best_result["f1_macro_std"],
    "example_fold_f1_macro": float(example_f1_macro),
    "example_fold_per_class_f1": [float(x) for x in per_class_f1],
    "kernel": best_cfg["kernel"],
    "C": best_cfg["C"],
    "gamma": best_cfg["gamma"],
    "cv_folds": 5,
    "preprocessing": "per-sample standardization (zero mean, unit std)",
    "trials": results,
}

with open(baseline_path, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2)


# Save confusion matrix values to JSON

conf_path = os.path.join(NUM_DIR, "confusion_matrices.json")

if os.path.exists(conf_path):
    with open(conf_path, "r", encoding="utf-8") as f:
        confusion_results = json.load(f)
else:
    confusion_results = {}

confusion_results["svm_finesat"] = cm.tolist()

with open(conf_path, "w", encoding="utf-8") as f:
    json.dump(confusion_results, f, indent=2)



# Record tried settings if not close to target

target_acc = 0.965

if abs(best_result["accuracy"] - target_acc) > 0.02:
    print("Tried settings:")
    for r in results:
        print(" - {} -> {:.1f}% ± {:.1f}%".format(
            r["name"],
            r["accuracy"] * 100,
            r["std"] * 100
        ))
