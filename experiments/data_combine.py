import numpy as np


wifi_X_path = "wifi_m_push_square_X.npy"
wifi_Y_path = "wifi_m_push_square_Y.npy"

widar_X_path = "widar_X.npy"
widar_Y_path = "widar_Y.npy"

OUT_X = "wifi_final_X.npy"
OUT_Y = "wifi_final_Y.npy"

SAMPLES_PER_CLASS = 200


def sample_class(X, Y, target_label, n):
    idx = np.where(Y == target_label)[0]

    if len(idx) == 0:
        raise ValueError(f"No samples for label {target_label}")

    if len(idx) >= n:
        selected = np.random.choice(idx, n, replace=False)
    else:
        selected = np.random.choice(idx, n, replace=True)

    return X[selected]


def main():
    wifi_X = np.load(wifi_X_path)
    wifi_Y = np.load(wifi_Y_path)

    widar_X = np.load(widar_X_path)
    widar_Y = np.load(widar_Y_path)

    print("WiFi shape:", wifi_X.shape)
    print("Widar shape:", widar_X.shape)

    # ===== WiFi label =====
    # 0 = M
    # 1 = Push
    # 2 = Square

    X_M = sample_class(wifi_X, wifi_Y, 0, SAMPLES_PER_CLASS)
    X_Push = sample_class(wifi_X, wifi_Y, 1, SAMPLES_PER_CLASS)
    X_Square = sample_class(wifi_X, wifi_Y, 2, SAMPLES_PER_CLASS)

    # ===== Widar label =====
    # 0 = Push&Pull
    # 1 = Triangle

    X_PushPull = sample_class(widar_X, widar_Y, 0, SAMPLES_PER_CLASS)
    X_Triangle = sample_class(widar_X, widar_Y, 1, SAMPLES_PER_CLASS)

    X_all = np.concatenate([
        X_Push,        # 0
        X_PushPull,    # 1
        X_Triangle,    # 2
        X_M,           # 3
        X_Square       # 4
    ], axis=0)

    Y_all = np.array(
        [0]*SAMPLES_PER_CLASS +
        [1]*SAMPLES_PER_CLASS +
        [2]*SAMPLES_PER_CLASS +
        [3]*SAMPLES_PER_CLASS +
        [4]*SAMPLES_PER_CLASS,
        dtype=np.int64
    )

    perm = np.random.permutation(len(X_all))
    X_all = X_all[perm]
    Y_all = Y_all[perm]

    print("Final dataset:")
    print("X:", X_all.shape)
    print("Y:", Y_all.shape)


    np.save(OUT_X, X_all)
    np.save(OUT_Y, Y_all)

    print("Saved:")
    print(OUT_X)
    print(OUT_Y)


if __name__ == "__main__":
    main()
