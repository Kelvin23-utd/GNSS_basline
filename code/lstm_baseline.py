import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Dataset

class SignalSequenceDataset(Dataset):
    def __init__(self, X, y):
        # LSTM expects (N, seq_len, input_size)
        # original X: (N, 200) -> (N, 200, 1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LSTM model

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()

        # dropout in nn.LSTM only works when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (B, 200, 1)
        out, (hn, cn) = self.lstm(x)
        # hn shape: (num_layers, B, hidden_size)
        last_hidden = hn[-1]   # (B, hidden_size)
        logits = self.classifier(last_hidden)
        return logits


# Load data

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


# Standardization

def standardize_by_train(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    return X_train_std, X_val_std


# Train one fold

def train_one_fold(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(epochs):
        model.train()
        train_preds = []
        train_targets = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:02d}/{epochs} | "
                f"train acc = {train_acc:.4f} | val acc = {val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    return model


# Plot confusion matrix

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


# Run 5-fold CV

def run_lstm_experiment(X, y, tag, figure_path, device, batch_size=32, epochs=50, lr=1e-3):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    all_true = np.zeros(len(y_enc), dtype=int)
    all_pred = np.zeros(len(y_enc), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), start=1):
        print(f"\n===== {tag} | Fold {fold}/5 =====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        X_train, X_val = standardize_by_train(X_train, X_val)

        train_ds = SignalSequenceDataset(X_train, y_train)
        val_ds = SignalSequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = LSTMClassifier(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(device)

        model = train_one_fold(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            lr=lr
        )

        model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()

                preds.extend(pred)
                targets.extend(yb.numpy())

        fold_acc = accuracy_score(targets, preds)
        fold_accuracies.append(fold_acc)

        all_true[val_idx] = np.array(targets)
        all_pred[val_idx] = np.array(preds)

        print(f"Fold {fold} accuracy: {fold_acc:.4f}")

    acc_mean = float(np.mean(fold_accuracies))
    acc_std = float(np.std(fold_accuracies))
    overall_acc = float(accuracy_score(all_true, all_pred))
    f1_macro = float(f1_score(all_true, all_pred, average="macro"))
    f1_per_class = f1_score(all_true, all_pred, average=None)

    print(f"\n===== LSTM {tag} =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"5-fold CV accuracy: {acc_mean*100:.1f}% ± {acc_std*100:.1f}%")
    print("Overall CV accuracy:", overall_acc)
    print("Macro F1:", f1_macro)
    print(classification_report(all_true, all_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(all_true, all_pred)
    cm_norm = confusion_matrix(all_true, all_pred, normalize="true")

    save_confusion_matrix_plot(
        cm_norm,
        class_names,
        f"LSTM {tag} (Normalized Confusion Matrix)",
        figure_path
    )

    return {
        "accuracy": acc_mean,
        "std": acc_std,
        "f1_macro": f1_macro,
        "per_class_f1": [float(x) for x in f1_per_class],
        "architecture": "LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True) -> Dropout(0.3) -> Linear(64,5)",
        "optimizer": "Adam",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }, cm.tolist()


# JSON utils

def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Main
def main():
    set_seed(42)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("numbers", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # FineSat
    X_finesat, y_finesat = load_pth_data(
        "data/W_FineSat_X.pth",
        "data/W_FineSat_Y.pth"
    )

    finesat_result, finesat_cm = run_lstm_experiment(
        X_finesat,
        y_finesat,
        tag="W_FineSat",
        figure_path="figures/confusion_lstm_finesat.png",
        device=device,
        batch_size=32,
        epochs=50,
        lr=1e-3
    )

    # Raw
    X_raw, y_raw = load_pth_data(
        "data/WO_FineSat_X.pth",
        "data/WO_FineSat_Y.pth"
    )

    raw_result, raw_cm = run_lstm_experiment(
        X_raw,
        y_raw,
        tag="WO_FineSat",
        figure_path="figures/confusion_lstm_raw.png",
        device=device,
        batch_size=32,
        epochs=50,
        lr=1e-3
    )

    baseline_path = "numbers/baseline_results.json"
    baseline_results = load_json_if_exists(baseline_path)
    baseline_results["lstm_finesat"] = finesat_result
    baseline_results["lstm_raw"] = raw_result

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    confusion_path = "numbers/confusion_matrices.json"
    confusion_results = load_json_if_exists(confusion_path)
    confusion_results["lstm_finesat"] = finesat_cm
    confusion_results["lstm_raw"] = raw_cm

    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion_results, f, indent=2)

    print("\nSaved figures:")
    print("- figures/confusion_lstm_finesat.png")
    print("- figures/confusion_lstm_raw.png")

    print("\nSaved JSON:")
    print("- numbers/baseline_results.json")
    print("- numbers/confusion_matrices.json")


if __name__ == "__main__":
    main()