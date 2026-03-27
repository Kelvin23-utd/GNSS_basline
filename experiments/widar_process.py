import os
import numpy as np
from scipy.signal import resample

# HERE NEED TO CHANGE TO YOUR WIDAR DATA DIRECTORY, THAT CONTAINS TRIANGLE AND PUSH&PULL DATA.
WIDAR_ROOT = r"C:\Users\72399\Desktop\GNSS_Project\WIDAR_DATA"

PUSHPULL_DIR = os.path.join(WIDAR_ROOT, "Push&Pull")
TRIANGLE_DIR = os.path.join(WIDAR_ROOT, "Draw-Triangle(H)")

OUTPUT_X = "widar_X.npy"
OUTPUT_Y = "widar_Y.npy"

TARGET_LENGTH = 200


def load_csv_sample(file_path):
    return np.loadtxt(file_path, delimiter=",", dtype=np.float32)


def sample_to_1d(x):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        return x

    if x.ndim == 2:
        # (22, 400) → (400,)
        if x.shape[1] >= x.shape[0]:
            return x.mean(axis=0)
        else:
            return x.mean(axis=1)

    time_axis = int(np.argmax(x.shape))
    axes = tuple(i for i in range(x.ndim) if i != time_axis)
    return x.mean(axis=axes)


def resize_to_200(x):
    x = x.reshape(-1)

    if len(x) < 2:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    if len(x) == TARGET_LENGTH:
        return x

    return resample(x, TARGET_LENGTH).astype(np.float32)


def normalize(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)


def collect_csv_files(folder):
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(".csv")
    ]


def process_pushpull(folder):
    X, Y = [], []

    files = collect_csv_files(folder)
    print(f"Push&Pull files: {len(files)}")

    for f in files:
        try:
            raw = load_csv_sample(f)
            x = sample_to_1d(raw)
            x = resize_to_200(x)
            x = normalize(x)

            X.append(x)
            Y.append(0)  # Push&Pull

        except Exception as e:
            print(f"Skip {os.path.basename(f)}: {e}")

    return X, Y


def process_triangle(folder):
    X, Y = [], []

    files = collect_csv_files(folder)
    print(f"Triangle files: {len(files)}")

    for f in files:
        try:
            raw = load_csv_sample(f)
            x = sample_to_1d(raw)
            x = resize_to_200(x)
            x = normalize(x)

            X.append(x)
            Y.append(1)  # Triangle

        except Exception as e:
            print(f"Skip {os.path.basename(f)}: {e}")

    return X, Y


def print_distribution(Y):
    unique, counts = np.unique(Y, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        name = "Push&Pull" if u == 0 else "Triangle"
        print(f"{name}: {c}")


def main():
    X_all = []
    Y_all = []

    x_pp, y_pp = process_pushpull(PUSHPULL_DIR)
    x_tri, y_tri = process_triangle(TRIANGLE_DIR)

    X_all.extend(x_pp)
    Y_all.extend(y_pp)

    X_all.extend(x_tri)
    Y_all.extend(y_tri)

    if len(X_all) == 0:
        print("No valid samples extracted.")
        return

    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.int64)

    print("Final shape:")
    print("X:", X_all.shape)
    print("Y:", Y_all.shape)
    print_distribution(Y_all)

    np.save(OUTPUT_X, X_all)
    np.save(OUTPUT_Y, Y_all)

    print("Saved:")
    print(OUTPUT_X)
    print(OUTPUT_Y)


if __name__ == "__main__":
    main()
