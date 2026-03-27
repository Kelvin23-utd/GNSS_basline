import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


SAT_X_PATH = "W_FineSat_X.pth"
SAT_Y_PATH = "W_FineSat_Y.pth"

WIFI_X_PATH = "wifi_final_X.npy"
WIFI_Y_PATH = "wifi_final_Y.npy"

PLOT_PATH = "satellite_vs_satpluswifi_accuracy.png"
JSON_PATH = "satellite_vs_satpluswifi_results.json"

# K = 每类卫星训练样本数
K_VALUES = [10, 25, 50, 100, 160]

NUM_CLASSES = 5
FIXED_WIFI_PER_CLASS = 100

N_REPEATS = 5
RANDOM_SEED = 42

CLASS_NAMES = ["Push", "Push&Pull", "Triangle", "M", "Square"]


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

    raise ValueError(f"Unsupported format in {path}: {type(data)}")


def minmax_per_sample(X):
    X = np.asarray(X, dtype=np.float32)
    out = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        x = X[i]
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-8:
            out[i] = np.zeros_like(x, dtype=np.float32)
        else:
            out[i] = (x - mn) / (mx - mn + 1e-8)

    return out


def print_class_counts(y, title):
    unique, counts = np.unique(y, return_counts=True)
    print(title)
    for u, c in zip(unique, counts):
        print(f" - {u} ({CLASS_NAMES[int(u)]}): {c}")


def split_fixed_sat_train_test_per_class(X, y, train_per_class=160, test_per_class=40, seed=42):
    rng = np.random.default_rng(seed)

    X_train_parts = []
    y_train_parts = []
    X_test_parts = []
    y_test_parts = []

    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        need = train_per_class + test_per_class

        if len(idx) < need:
            raise ValueError(
                f"Class {c} ({CLASS_NAMES[c]}) only has {len(idx)} samples, need {need}"
            )

        shuffled = rng.permutation(idx)
        train_idx = shuffled[:train_per_class]
        test_idx = shuffled[train_per_class:train_per_class + test_per_class]

        X_train_parts.append(X[train_idx])
        y_train_parts.append(np.full(train_per_class, c, dtype=np.int64))

        X_test_parts.append(X[test_idx])
        y_test_parts.append(np.full(test_per_class, c, dtype=np.int64))

    X_train_full = np.concatenate(X_train_parts, axis=0)
    y_train_full = np.concatenate(y_train_parts, axis=0)

    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    return X_train_full, y_train_full, X_test, y_test


def sample_k_per_class(X, y, k, rng):
    X_parts = []
    y_parts = []

    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]

        if len(idx) < k:
            raise ValueError(
                f"Class {c} ({CLASS_NAMES[c]}) only has {len(idx)} samples, cannot take {k}"
            )

        chosen = rng.choice(idx, size=k, replace=False)
        X_parts.append(X[chosen])
        y_parts.append(np.full(k, c, dtype=np.int64))

    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def main():
    sat_X = load_pth(SAT_X_PATH)
    sat_Y = load_pth(SAT_Y_PATH)

    wifi_X = np.load(WIFI_X_PATH)
    wifi_Y = np.load(WIFI_Y_PATH)

    sat_X = np.asarray(sat_X, dtype=np.float32)
    sat_Y = np.asarray(sat_Y, dtype=np.int64).reshape(-1)

    wifi_X = np.asarray(wifi_X, dtype=np.float32)
    wifi_Y = np.asarray(wifi_Y, dtype=np.int64).reshape(-1)

    if sat_X.ndim != 2 or sat_X.shape[1] != 200:
        raise ValueError(f"Expected satellite X shape (N, 200), got {sat_X.shape}")
    if wifi_X.ndim != 2 or wifi_X.shape[1] != 200:
        raise ValueError(f"Expected wifi X shape (N, 200), got {wifi_X.shape}")

    sat_X = minmax_per_sample(sat_X)
    wifi_X = minmax_per_sample(wifi_X)

    print("Satellite:", sat_X.shape, sat_Y.shape)
    print("WiFi:", wifi_X.shape, wifi_Y.shape)
    print_class_counts(sat_Y, "\nSatellite class counts:")
    print_class_counts(wifi_Y, "\nWiFi class counts:")

    # 固定卫星：每类训练池160，每类测试40
    sat_train_pool_X, sat_train_pool_Y, sat_test_X, sat_test_Y = split_fixed_sat_train_test_per_class(
        sat_X, sat_Y, train_per_class=160, test_per_class=40, seed=RANDOM_SEED
    )

    print("\nFixed satellite split:")
    print("Satellite train pool:", sat_train_pool_X.shape, sat_train_pool_Y.shape)
    print("Satellite test set:", sat_test_X.shape, sat_test_Y.shape)

    sat_only_results = []
    sat_plus_wifi_results = []

    for k in K_VALUES:
        print(f"\n===== K = {k} per class =====")

        sat_only_accs = []
        sat_plus_wifi_accs = []

        for r in range(N_REPEATS):
            rng = np.random.default_rng(RANDOM_SEED + r)

            # 卫星训练集：每类 K
            sat_k_X, sat_k_Y = sample_k_per_class(sat_train_pool_X, sat_train_pool_Y, k, rng)

            # WiFi 训练集：每类固定 100
            wifi_fixed_X, wifi_fixed_Y = sample_k_per_class(wifi_X, wifi_Y, FIXED_WIFI_PER_CLASS, rng)

            # Satellite only
            clf_sat = SVC(kernel="rbf", C=10, gamma="scale")
            clf_sat.fit(sat_k_X, sat_k_Y)
            pred_sat = clf_sat.predict(sat_test_X)
            acc_sat = accuracy_score(sat_test_Y, pred_sat)
            sat_only_accs.append(acc_sat)

            # Satellite + WiFi
            train_aug_X = np.concatenate([sat_k_X, wifi_fixed_X], axis=0)
            train_aug_Y = np.concatenate([sat_k_Y, wifi_fixed_Y], axis=0)

            clf_aug = SVC(kernel="rbf", C=10, gamma="scale")
            clf_aug.fit(train_aug_X, train_aug_Y)
            pred_aug = clf_aug.predict(sat_test_X)
            acc_aug = accuracy_score(sat_test_Y, pred_aug)
            sat_plus_wifi_accs.append(acc_aug)

        sat_only_mean = float(np.mean(sat_only_accs))
        sat_only_std = float(np.std(sat_only_accs))

        sat_plus_wifi_mean = float(np.mean(sat_plus_wifi_accs))
        sat_plus_wifi_std = float(np.std(sat_plus_wifi_accs))

        print(f"Satellite only            -> {sat_only_mean*100:.2f}% ± {sat_only_std*100:.2f}%")
        print(f"Satellite + WiFi(100/class) -> {sat_plus_wifi_mean*100:.2f}% ± {sat_plus_wifi_std*100:.2f}%")

        sat_only_results.append({
            "k_per_class": int(k),
            "satellite_train_total": int(k * NUM_CLASSES),
            "test_total": int(len(sat_test_Y)),
            "accuracy_mean": sat_only_mean,
            "accuracy_std": sat_only_std,
            "accuracies": [float(x) for x in sat_only_accs]
        })

        sat_plus_wifi_results.append({
            "k_per_class": int(k),
            "satellite_train_total": int(k * NUM_CLASSES),
            "wifi_per_class": int(FIXED_WIFI_PER_CLASS),
            "wifi_total": int(FIXED_WIFI_PER_CLASS * NUM_CLASSES),
            "joint_train_total": int(k * NUM_CLASSES + FIXED_WIFI_PER_CLASS * NUM_CLASSES),
            "test_total": int(len(sat_test_Y)),
            "accuracy_mean": sat_plus_wifi_mean,
            "accuracy_std": sat_plus_wifi_std,
            "accuracies": [float(x) for x in sat_plus_wifi_accs]
        })

    # 画对比图
    ks = [r["k_per_class"] for r in sat_only_results]
    sat_only_means = [r["accuracy_mean"] * 100 for r in sat_only_results]
    sat_plus_wifi_means = [r["accuracy_mean"] * 100 for r in sat_plus_wifi_results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sat_only_means, marker="o", label="Satellite only")
    plt.plot(ks, sat_plus_wifi_means, marker="s", label="Satellite + WiFi (100/class)")
    plt.xlabel("K per class (satellite training samples)")
    plt.ylabel("Accuracy (%)")
    plt.title("Satellite Few-shot SVM vs Satellite + Fixed WiFi")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    output = {
        "dataset": "W_FineSat",
        "model": "SVM_RBF",
        "setup": {
            "fixed_satellite_test_per_class": 40,
            "fixed_satellite_train_pool_per_class": 160,
            "fixed_wifi_per_class": FIXED_WIFI_PER_CLASS,
            "k_definition": "K means number of satellite training samples per class"
        },
        "k_values": K_VALUES,
        "n_repeats": N_REPEATS,
        "satellite_only_results": sat_only_results,
        "satellite_plus_fixed_wifi_results": sat_plus_wifi_results
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSaved:")
    print(PLOT_PATH)
    print(JSON_PATH)


if __name__ == "__main__":
    main()
