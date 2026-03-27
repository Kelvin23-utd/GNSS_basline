import json
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

CONFUSION_MATRIX_PNG = "joint_confusion_matrix.png"
NUMBERS_JSON = "joint_results.json"

CLASS_NAMES = ["Push", "Push&Pull", "Triangle", "M", "Square"]

SAMPLES_PER_CLASS_PER_MODALITY = 100

RANDOM_SEED = 42


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


def build_balanced_joint_dataset(
    sat_X,
    sat_Y,
    wifi_X,
    wifi_Y,
    samples_per_class_per_modality=100,
    seed=42
):
    rng = np.random.default_rng(seed)

    X_parts = []
    Y_parts = []

    for cls in range(len(CLASS_NAMES)):
        sat_x_cls, sat_y_cls = sample_one_class(
            sat_X, sat_Y, cls, samples_per_class_per_modality, rng
        )
        wifi_x_cls, wifi_y_cls = sample_one_class(
            wifi_X, wifi_Y, cls, samples_per_class_per_modality, rng
        )

        X_parts.extend([wifi_x_cls, sat_x_cls])
        Y_parts.extend([wifi_y_cls, sat_y_cls])

    X_joint = np.concatenate(X_parts, axis=0).astype(np.float32)
    Y_joint = np.concatenate(Y_parts, axis=0).astype(np.int64)

    perm = rng.permutation(len(X_joint))
    X_joint = X_joint[perm]
    Y_joint = Y_joint[perm]

    return X_joint, Y_joint


def plot_confusion_matrix(cm, class_names, save_path):
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
        title="Joint Training Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
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

    X_joint, Y_joint = build_balanced_joint_dataset(
        sat_X=sat_X,
        sat_Y=sat_Y,
        wifi_X=wifi_X,
        wifi_Y=wifi_Y,
        samples_per_class_per_modality=SAMPLES_PER_CLASS_PER_MODALITY,
        seed=RANDOM_SEED
    )

    print("\nBalanced joint dataset:")
    print("X_joint:", X_joint.shape)
    print("Y_joint:", Y_joint.shape)
    print_class_counts(Y_joint, "Balanced joint class counts:")

    expected_total = len(CLASS_NAMES) * SAMPLES_PER_CLASS_PER_MODALITY * 2
    print(f"\nExpected total samples: {expected_total}")

    # ===== 5-fold CV =====
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

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

        print(f"Fold {fold}: acc={acc*100:.2f}%, macro_f1={f1:.4f}")

    acc_mean = float(np.mean(acc_list))
    acc_std = float(np.std(acc_list))
    f1_mean = float(np.mean(f1_list))
    f1_std = float(np.std(f1_list))

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3, 4])

    print(f"\n5-fold CV accuracy: {acc_mean*100:.1f}% ± {acc_std*100:.1f}%")
    print(f"5-fold CV macro F1: {f1_mean:.4f} ± {f1_std:.4f}")

    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PNG)

    results = {
        "model": "SVM_RBF_joint_training_50_50_balanced",
        "class_names": CLASS_NAMES,
        "satellite_shape": list(sat_X.shape),
        "wifi_shape": list(wifi_X.shape),
        "joint_shape": list(X_joint.shape),
        "samples_per_class_per_modality": SAMPLES_PER_CLASS_PER_MODALITY,
        "samples_per_class_total": SAMPLES_PER_CLASS_PER_MODALITY * 2,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "fold_accuracies": [float(x) for x in acc_list],
        "fold_macro_f1": [float(x) for x in f1_list],
        "confusion_matrix": cm.tolist()
    }

    with open(NUMBERS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(CONFUSION_MATRIX_PNG)
    print(NUMBERS_JSON)


if __name__ == "__main__":
    main()
