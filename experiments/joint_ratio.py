import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


SAT_X_PATH = "W_FineSat_X.pth"
SAT_Y_PATH = "W_FineSat_Y.pth"

WIFI_X_PATH = "wifi_final_X.npy"
WIFI_Y_PATH = "wifi_final_Y.npy"

OUTPUT_DIR = "joint_ratio_by_modality_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_JSON = os.path.join(OUTPUT_DIR, "ratio_by_modality_results.json")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ratio_by_modality_results.csv")
SUMMARY_PLOT = os.path.join(OUTPUT_DIR, "ratio_by_modality_accuracy_plot.png")

CLASS_NAMES = ["Push", "Push&Pull", "Triangle", "M", "Square"]

#每个 class 的总样本数固定
TOTAL_SAMPLES_PER_CLASS = 200

# 搜索 WiFi 占比 
# 例如 0.8 表示 WiFi:Satellite = 8:2
WIFI_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

RANDOM_SEED = 42
N_SPLITS = 5


def load_pth(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, dict):
        for _, value in data.items():
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                return value

    if isinstance(data, list):
        return np.array(data)

    raise ValueError(f"Unsupported .pth format in file: {path}, type={type(data)}")


def minmax_per_sample(X):
    X = np.asarray(X, dtype=np.float32)
    out = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        x = X[i]
        x_min = x.min()
        x_max = x.max()

        if x_max - x_min < 1e-8:
            out[i] = np.zeros_like(x, dtype=np.float32)
        else:
            out[i] = (x - x_min) / (x_max - x_min + 1e-8)

    return out


def print_class_counts(Y, title):
    unique, counts = np.unique(Y, return_counts=True)
    print(title)
    for u, c in zip(unique, counts):
        print(f" - {u} ({CLASS_NAMES[int(u)]}): {c}")


def sample_one_class(X, Y, target_class, n, rng):
    idx = np.where(Y == target_class)[0]

    if len(idx) == 0:
        raise ValueError(f"No samples found for class {target_class} ({CLASS_NAMES[target_class]})")

    if len(idx) >= n:
        selected = rng.choice(idx, size=n, replace=False)
    else:
        selected = rng.choice(idx, size=n, replace=True)

    return X[selected], np.full(n, target_class, dtype=np.int64)


def build_joint_dataset_with_ratio(
    sat_X,
    sat_Y,
    wifi_X,
    wifi_Y,
    total_samples_per_class,
    wifi_ratio,
    seed=42
):
    rng = np.random.default_rng(seed)

    wifi_n = int(round(total_samples_per_class * wifi_ratio))
    sat_n = total_samples_per_class - wifi_n

    if wifi_n <= 0 or sat_n <= 0:
        raise ValueError(
            f"Invalid split for ratio={wifi_ratio}. "
            f"wifi_n={wifi_n}, sat_n={sat_n}. Both must be > 0."
        )

    X_parts = []
    Y_parts = []

    for cls in range(len(CLASS_NAMES)):
        wifi_x_cls, wifi_y_cls = sample_one_class(
            wifi_X, wifi_Y, cls, wifi_n, rng
        )
        sat_x_cls, sat_y_cls = sample_one_class(
            sat_X, sat_Y, cls, sat_n, rng
        )

        X_parts.extend([wifi_x_cls, sat_x_cls])
        Y_parts.extend([wifi_y_cls, sat_y_cls])

    X_joint = np.concatenate(X_parts, axis=0).astype(np.float32)
    Y_joint = np.concatenate(Y_parts, axis=0).astype(np.int64)

    perm = rng.permutation(len(X_joint))
    X_joint = X_joint[perm]
    Y_joint = Y_joint[perm]

    return X_joint, Y_joint, wifi_n, sat_n


def plot_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_one_setting(X_joint, Y_joint, wifi_ratio):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    acc_list = []
    f1_list = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_joint, Y_joint), start=1):
        X_train, X_test = X_joint[train_idx], X_joint[test_idx]
        y_train, y_test = Y_joint[train_idx], Y_joint[test_idx]

        clf = SVC(kernel="rbf", C=10, gamma="scale")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        acc_list.append(acc)
        f1_list.append(f1)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"  Fold {fold}: acc={acc*100:.2f}%, macro_f1={f1:.4f}")

    acc_mean = float(np.mean(acc_list))
    acc_std = float(np.std(acc_list))
    f1_mean = float(np.mean(f1_list))
    f1_std = float(np.std(f1_list))

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3, 4])

    ratio_str = str(wifi_ratio).replace(".", "_")
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_wifi_ratio_{ratio_str}.png")
    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        cm_path,
        title=f"Joint Confusion Matrix (WiFi ratio={wifi_ratio:.1f})"
    )

    return {
        "wifi_ratio": float(wifi_ratio),
        "satellite_ratio": float(1.0 - wifi_ratio),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "fold_accuracies": [float(x) for x in acc_list],
        "fold_macro_f1": [float(x) for x in f1_list],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path
    }


def save_summary_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wifi_ratio",
            "satellite_ratio",
            "wifi_per_class",
            "satellite_per_class",
            "total_samples",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std"
        ])

        for r in results:
            writer.writerow([
                r["wifi_ratio"],
                r["satellite_ratio"],
                r["wifi_per_class"],
                r["satellite_per_class"],
                r["total_samples"],
                r["accuracy_mean"],
                r["accuracy_std"],
                r["macro_f1_mean"],
                r["macro_f1_std"]
            ])


def plot_summary(results, save_path):
    ratios = [r["wifi_ratio"] for r in results]
    accs = [r["accuracy_mean"] * 100 for r in results]
    f1s = [r["macro_f1_mean"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, accs, marker="o", label="Accuracy (%)")
    ax.plot(ratios, f1s, marker="s", label="Macro F1")

    ax.set_xlabel("WiFi ratio")
    ax.set_ylabel("Score")
    ax.set_title("Joint Training Performance vs WiFi/Satellite Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    sat_X = load_pth(SAT_X_PATH)
    sat_Y = load_pth(SAT_Y_PATH)

    wifi_X = np.load(WIFI_X_PATH)
    wifi_Y = np.load(WIFI_Y_PATH)

    sat_X = np.asarray(sat_X, dtype=np.float32)
    sat_Y = np.asarray(sat_Y, dtype=np.int64).reshape(-1)

    wifi_X = np.asarray(wifi_X, dtype=np.float32)
    wifi_Y = np.asarray(wifi_Y, dtype=np.int64).reshape(-1)

    print("Satellite X:", sat_X.shape)
    print("Satellite Y:", sat_Y.shape)
    print("WiFi X:", wifi_X.shape)
    print("WiFi Y:", wifi_Y.shape)

    if sat_X.ndim != 2 or wifi_X.ndim != 2:
        raise ValueError("Both satellite and WiFi feature arrays must be 2D: (N, 200)")

    if sat_X.shape[1] != 200 or wifi_X.shape[1] != 200:
        raise ValueError("Both satellite and WiFi features must have shape (*, 200)")

    sat_X = minmax_per_sample(sat_X)
    wifi_X = minmax_per_sample(wifi_X)

    print_class_counts(sat_Y, "\nSatellite class counts:")
    print_class_counts(wifi_Y, "\nWiFi class counts:")

    all_results = []

    for wifi_ratio in WIFI_RATIOS:
        print(f"\n===== Testing WiFi ratio = {wifi_ratio:.1f} =====")

        X_joint, Y_joint, wifi_n, sat_n = build_joint_dataset_with_ratio(
            sat_X=sat_X,
            sat_Y=sat_Y,
            wifi_X=wifi_X,
            wifi_Y=wifi_Y,
            total_samples_per_class=TOTAL_SAMPLES_PER_CLASS,
            wifi_ratio=wifi_ratio,
            seed=RANDOM_SEED
        )

        print(
            f"Per class -> WiFi: {wifi_n}, Satellite: {sat_n}, "
            f"Total per class: {TOTAL_SAMPLES_PER_CLASS}"
        )
        print(f"Joint dataset shape: {X_joint.shape}, {Y_joint.shape}")

        result = evaluate_one_setting(X_joint, Y_joint, wifi_ratio)
        result["wifi_per_class"] = int(wifi_n)
        result["satellite_per_class"] = int(sat_n)
        result["total_samples"] = int(len(X_joint))

        all_results.append(result)

        print(
            f"wifi_ratio={wifi_ratio:.1f} -> "
            f"acc={result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%, "
            f"macro_f1={result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
        )

    best_by_acc = max(all_results, key=lambda x: x["accuracy_mean"])
    best_by_f1 = max(all_results, key=lambda x: x["macro_f1_mean"])

    summary = {
        "class_names": CLASS_NAMES,
        "total_samples_per_class": TOTAL_SAMPLES_PER_CLASS,
        "tested_wifi_ratios": WIFI_RATIOS,
        "results": all_results,
        "best_by_accuracy": best_by_acc,
        "best_by_macro_f1": best_by_f1
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_summary_csv(all_results, SUMMARY_CSV)
    plot_summary(all_results, SUMMARY_PLOT)

    print("\n===== Search Finished =====")
    print(
        f"Best by accuracy: WiFi ratio={best_by_acc['wifi_ratio']:.1f}, "
        f"Satellite ratio={best_by_acc['satellite_ratio']:.1f}, "
        f"acc={best_by_acc['accuracy_mean']*100:.2f}% ± {best_by_acc['accuracy_std']*100:.2f}%"
    )
    print(
        f"Best by macro F1: WiFi ratio={best_by_f1['wifi_ratio']:.1f}, "
        f"Satellite ratio={best_by_f1['satellite_ratio']:.1f}, "
        f"macro_f1={best_by_f1['macro_f1_mean']:.4f} ± {best_by_f1['macro_f1_std']:.4f}"
    )

    print("\nSaved:")
    print(SUMMARY_JSON)
    print(SUMMARY_CSV)
    print(SUMMARY_PLOT)
    print("Per-ratio confusion matrices are saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
