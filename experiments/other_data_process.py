import os
import pickle
import numpy as np
from scipy.signal import resample

#here need to change to the data directory that contains the M, Push, Square data
DATA_DIR = r"C:\Users\72399\Desktop\GNSS_Project\total_data\total_data\raw_CSI\single-user"

OUTPUT_X = "wifi_m_push_square_X.npy"
OUTPUT_Y = "wifi_m_push_square_Y.npy"

TARGET_LENGTH = 200


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def extract_csi_array_from_pkl(data):
    """
    当前这批 pkl 的结构:
    data = [meta, csi_array]

    其中:
    - data[0]: metadata/list
    - data[1]: complex ndarray, shape like (3, 30, T)
    """
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        csi = data[1]
        if isinstance(csi, np.ndarray):
            return csi

    raise ValueError("Unexpected pkl format; expected [meta, csi_array].")


def csi_to_1d(csi):
    """
    输入:
        csi: complex ndarray, shape like (3, 30, T)

    输出:
        1D float array, shape (T,)
    """
    csi = np.asarray(csi)

    # 取幅值
    csi_mag = np.abs(csi).astype(np.float32)

    if csi_mag.ndim != 3:
        raise ValueError(f"Expected 3D CSI array, got shape {csi_mag.shape}")

    # 对天线维和子载波维求均值，只保留时间维
    x = csi_mag.mean(axis=(0, 1))

    return x.astype(np.float32)


def resize_to_200(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) < 2:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    if len(x) == TARGET_LENGTH:
        return x

    return resample(x, TARGET_LENGTH).astype(np.float32)


def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return (x - x_min) / (x_max - x_min + 1e-8)


def get_label_from_filename(fname):
    fname_lower = fname.lower()

    if "splitted-m-" in fname_lower:
        return 0   # M

    if "splitted-push-" in fname_lower:
        return 1   # Push

    if "splitted-square-" in fname_lower:
        return 2   # Square

    return None


def label_name(label):
    if label == 0:
        return "M"
    if label == 1:
        return "Push"
    if label == 2:
        return "Square"
    return f"Unknown-{label}"


def main():
    X = []
    Y = []

    files = sorted(os.listdir(DATA_DIR))
    print(f"Total files: {len(files)}")

    for fname in files:
        if not fname.endswith(".pkl"):
            continue

        label = get_label_from_filename(fname)
        if label is None:
            continue

        fpath = os.path.join(DATA_DIR, fname)

        try:
            data = load_pkl(fpath)
            csi = extract_csi_array_from_pkl(data)
            x = csi_to_1d(csi)
            x = resize_to_200(x)
            x = minmax_normalize(x)

            X.append(x)
            Y.append(label)

        except Exception as e:
            print(f"Skip {fname}: {e}")

    if len(X) == 0:
        print("No valid samples extracted.")
        return

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)

    print("Final shape:")
    print("X:", X.shape)
    print("Y:", Y.shape)

    unique, counts = np.unique(Y, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"{label_name(u)}: {c}")

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, Y)

    print("Saved:")
    print(OUTPUT_X)
    print(OUTPUT_Y)


if __name__ == "__main__":
    main()
