import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FineSatDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_len=200,
        num_classes=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)          # [B, 200] -> [B, 200, 1]
        x = self.input_proj(x)       # [B, 200, d_model]
        x = x + self.pos_embedding   # [B, 200, d_model]
        x = self.encoder(x)          # [B, 200, d_model]
        x = x.mean(dim=1)            # global average pooling
        x = self.norm(x)
        x = self.classifier(x)
        return x


def normalize_per_sample(X):
    x_min = X.min(axis=1, keepdims=True)
    x_max = X.max(axis=1, keepdims=True)
    return (X - x_min) / (x_max - x_min + 1e-8)


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4
):
    train_dataset = FineSatDataset(X_train, y_train)
    val_dataset = FineSatDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = TransformerClassifier(
        seq_len=X_train.shape[1],
        num_classes=len(np.unique(y_train)),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(yb.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.eval()

    final_preds = []
    final_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_true.extend(yb.numpy())

    return np.array(final_true), np.array(final_preds)


def save_confusion_matrix(cm, title, save_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    classes = np.arange(cm.shape[0])
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_title(title, fontsize=14)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_dataset(name, x_path, y_path, device):
    X = torch.load(x_path, weights_only=False)
    y = torch.load(y_path, weights_only=False)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if y.ndim > 1:
        y = y.squeeze()
    y = y.astype(int)

    X = normalize_per_sample(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_macro_f1s = []
    all_true = []
    all_pred = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        true_labels, pred_labels = train_one_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            epochs=50,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-4
        )

        acc = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro")

        fold_accuracies.append(acc)
        fold_macro_f1s.append(macro_f1)
        all_true.extend(true_labels.tolist())
        all_pred.extend(pred_labels.tolist())

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_macro_f1s)
    std_f1 = np.std(fold_macro_f1s)

    cm = confusion_matrix(all_true, all_pred)

    if name == "W_FineSat":
        fig_path = "figures/confusion_transformer_finesat.png"
    else:
        fig_path = "figures/confusion_transformer_raw.png"

    save_confusion_matrix(
        cm,
        title=f"{name} - Confusion Matrix",
        save_path=fig_path
    )

    return {
        "dataset": name,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_macro_f1": float(mean_f1),
        "std_macro_f1": float(std_f1),
        "fold_accuracies": [float(x) for x in fold_accuracies],
        "fold_macro_f1s": [float(x) for x in fold_macro_f1s],
        "confusion_matrix": cm.tolist(),
        "figure_path": fig_path
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("numbers", exist_ok=True)

    datasets = [
        ("W_FineSat", "data/W_FineSat_X.pth", "data/W_FineSat_Y.pth"),
        ("WO_FineSat", "data/WO_FineSat_X.pth", "data/WO_FineSat_Y.pth"),
    ]

    results_path = "numbers/baseline_results.json"
    cm_path = "numbers/confusion_matrices.json"

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
    else:
        baseline_results = {}

    if os.path.exists(cm_path):
        with open(cm_path, "r", encoding="utf-8") as f:
            confusion_data = json.load(f)
    else:
        confusion_data = {}

    saved_figures = []

    for name, x_path, y_path in datasets:
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            print(f"Skip {name}: missing file(s)")
            continue

        result = run_dataset(name, x_path, y_path, device)

        if name == "W_FineSat":
            result_key = "transformer_finesat"
            cm_key = "transformer_finesat"
        else:
            result_key = "transformer_raw"
            cm_key = "transformer_raw"

        baseline_results[result_key] = {
            "mean_accuracy": result["mean_accuracy"],
            "std_accuracy": result["std_accuracy"],
            "mean_macro_f1": result["mean_macro_f1"],
            "std_macro_f1": result["std_macro_f1"],
            "fold_accuracies": result["fold_accuracies"],
            "fold_macro_f1s": result["fold_macro_f1s"]
        }

        confusion_data[cm_key] = result["confusion_matrix"]

        print(f'{name}: {result["mean_accuracy"] * 100:.1f}% ± {result["std_accuracy"] * 100:.1f}%')
        saved_figures.append(result["figure_path"])

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=4)

    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(confusion_data, f, indent=4)

    print("\nSaved figures:")
    for path in saved_figures:
        print(f"- {path}")

    print("\nSaved JSON:")
    print(f"- {results_path}")
    print(f"- {cm_path}")


if __name__ == "__main__":
    main()