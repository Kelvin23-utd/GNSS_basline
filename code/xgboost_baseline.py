import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_pth_data(x_path, y_path):
    X = torch.load(x_path, weights_only=False)
    y = torch.load(y_path, weights_only=False)

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    return X, y


def save_confusion_matrix_plot(cm_norm, class_names, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_xgboost_experiment(X, y, tag, figure_path):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss"
    )

    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    y_pred = cross_val_predict(model, X, y_enc, cv=cv)

    acc = accuracy_score(y_enc, y_pred)
    f1_macro = f1_score(y_enc, y_pred, average="macro")
    f1_per_class = f1_score(y_enc, y_pred, average=None)

    print(f"\n===== XGBoost {tag} =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"5-fold CV accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print("Overall CV accuracy:", acc)
    print("Macro F1:", f1_macro)
    print(classification_report(y_enc, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_enc, y_pred)
    cm_norm = confusion_matrix(y_enc, y_pred, normalize="true")

    save_confusion_matrix_plot(
        cm_norm,
        class_names,
        f"XGBoost {tag} (Normalized Confusion Matrix)",
        figure_path
    )

    return {
        "accuracy": float(acc),
        "std": float(scores.std()),
        "f1_macro": float(f1_macro),
        "per_class_f1": [float(x) for x in f1_per_class]
    }, cm.tolist()


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("numbers", exist_ok=True)

    # FineSat
    X_finesat, y_finesat = load_pth_data(
        "data/W_FineSat_X.pth",
        "data/W_FineSat_Y.pth"
    )

    finesat_result, finesat_cm = run_xgboost_experiment(
        X_finesat,
        y_finesat,
        "W_FineSat",
        "figures/confusion_xgboost_finesat.png"
    )

    # Raw
    X_raw, y_raw = load_pth_data(
        "data/WO_FineSat_X.pth",
        "data/WO_FineSat_Y.pth"
    )

    raw_result, raw_cm = run_xgboost_experiment(
        X_raw,
        y_raw,
        "WO_FineSat",
        "figures/confusion_xgboost_raw.png"
    )

    baseline_path = "numbers/baseline_results.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
    else:
        baseline_results = {}

    baseline_results["xgboost_finesat"] = finesat_result
    baseline_results["xgboost_raw"] = raw_result

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    confusion_path = "numbers/confusion_matrices.json"
    if os.path.exists(confusion_path):
        with open(confusion_path, "r", encoding="utf-8") as f:
            confusion_results = json.load(f)
    else:
        confusion_results = {}

    confusion_results["xgboost_finesat"] = finesat_cm
    confusion_results["xgboost_raw"] = raw_cm

    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion_results, f, indent=2)

    print("\nSaved figures:")
    print("- figures/confusion_xgboost_finesat.png")
    print("- figures/confusion_xgboost_raw.png")
    print("\nSaved JSON:")
    print("- numbers/baseline_results.json")
    print("- numbers/confusion_matrices.json")


if __name__ == "__main__":
    main()