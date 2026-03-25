# 03_svm_raw_baseline.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("numbers", exist_ok=True)

    # 1) load raw data
    X = torch.load("data/WO_FineSat_X.pth", weights_only=False)
    y = torch.load("data/WO_FineSat_Y.pth", weights_only=False)

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    print("Raw X shape:", X.shape)
    print("Raw y shape:", y.shape)

    # 2) label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]

    # 3) 5-fold CV SVM
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    print(f"Raw SVM 5-fold CV accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # 4) cross-val predictions for confusion matrix / F1
    y_pred = cross_val_predict(model, X, y_enc, cv=cv)

    acc = accuracy_score(y_enc, y_pred)
    f1_macro = f1_score(y_enc, y_pred, average="macro")
    f1_per_class = f1_score(y_enc, y_pred, average=None)

    print("Overall CV accuracy:", acc)
    print("Macro F1:", f1_macro)
    print(classification_report(y_enc, y_pred, target_names=class_names, digits=4))

    # 5) confusion matrix
    cm = confusion_matrix(y_enc, y_pred)
    cm_norm = confusion_matrix(y_enc, y_pred, normalize="true")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, interpolation="nearest", aspect="equal")
    plt.title("SVM on Raw Signals (Normalized Confusion Matrix)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig("figures/confusion_svm_raw.png", dpi=300)
    plt.close()

    # 6) save json
    baseline_path = "numbers/baseline_results.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
    else:
        baseline_results = {}

    baseline_results["svm_raw"] = {
        "accuracy": float(acc),
        "std": float(scores.std()),
        "f1_macro": float(f1_macro),
        "per_class_f1": [float(x) for x in f1_per_class]
    }

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    confusion_path = "numbers/confusion_matrices.json"
    if os.path.exists(confusion_path):
        with open(confusion_path, "r", encoding="utf-8") as f:
            confusion_results = json.load(f)
    else:
        confusion_results = {}

    confusion_results["svm_raw"] = cm.tolist()

    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion_results, f, indent=2)

    print("Saved figure to figures/confusion_svm_raw.png")
    print("Saved numbers to numbers/baseline_results.json and numbers/confusion_matrices.json")


if __name__ == "__main__":
    main()