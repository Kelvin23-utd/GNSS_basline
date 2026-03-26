import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from train_optimized import WiFiNet


class WiFiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_map = {
            "push": 0,
            "pull": 1,
            "triangle": 2,
            "zigzag": 3,
            "n": 4
        }
        self.samples = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label = -1
            for key, val in self.class_map.items():
                if key in folder_name.lower():
                    label = val
                    break

            if label != -1:
                for f in os.listdir(folder_path):
                    if f.endswith(".csv"):
                        self.samples.append((os.path.join(folder_path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = pd.read_csv(file_path, header=None).values.astype(np.float32)

        if data.shape[0] > data.shape[1]:
            data = data.T

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)


def generate_final_report(model, sat_x, sat_y, device):
    model.eval()
    all_preds = []
    all_labels = sat_y.cpu().numpy()

    with torch.no_grad():
        for i in range(0, len(sat_x), 32):
            batch_x = sat_x[i:i + 32].to(device)
            output = model(batch_x)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    final_acc = accuracy_score(all_labels, all_preds) * 100

    class_names = ['Push', 'Push&Pull', 'Triangle', 'Draw M', 'Star']
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14, "weight": "bold"}
    )

    plt.xlabel('Predicted Label (Satellite)', fontsize=12, fontweight='bold')
    plt.ylabel('True Label (Satellite)', fontsize=12, fontweight='bold')
    plt.title(f'Satellite Recognition Confusion Matrix (Ratio 0.8)\nFinal Accuracy: {final_acc:.2f}%', fontsize=14)

    plt.tight_layout()
    save_path = 'satellite_confusion_matrix_real.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Final satellite accuracy: {final_acc:.2f}%")
    print(f"Confusion matrix saved to: {save_path}")


def train_mixed(ratio=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sat_x_raw = torch.load('W_FineSat_X.pth', weights_only=False)
    sat_y_raw = torch.load('W_FineSat_Y.pth', weights_only=False)

    sat_x = torch.from_numpy(sat_x_raw).float() if isinstance(sat_x_raw, np.ndarray) else sat_x_raw.float()
    sat_y = torch.from_numpy(sat_y_raw).long() if isinstance(sat_y_raw, np.ndarray) else sat_y_raw.long()

    sat_x = sat_x.unsqueeze(1).repeat(1, 22, 1)
    sat_x = torch.nn.functional.interpolate(sat_x, size=2000, mode='linear', align_corners=True)

    wifi_ds = WiFiDataset(root_dir=r"C:\Users\72399\Desktop\GNSS_Project\traindata")
    wifi_loader = DataLoader(wifi_ds, batch_size=64, shuffle=True)

    model = WiFiNet(num_classes=5, input_channels=22).to(device)

    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Loaded pretrained weights from best_model.pth")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()

        for w_x, w_y in wifi_loader:
            batch_size = w_x.size(0)
            num_sat = int(batch_size * ratio)

            sat_indices = torch.randperm(len(sat_x))[:num_sat]

            combined_x = w_x.clone()
            combined_y = w_y.clone()

            combined_x[:num_sat] = sat_x[sat_indices]
            combined_y[:num_sat] = sat_y[sat_indices]

            combined_x = combined_x.to(device)
            combined_y = combined_y.to(device)

            optimizer.zero_grad()
            output = model(combined_x)
            loss = criterion(output, combined_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(sat_x.to(device))
            val_acc = (val_out.argmax(1) == sat_y.to(device)).float().mean().item() * 100

        print(f"Epoch {epoch + 1:02d}: satellite accuracy = {val_acc:.2f}%")

    generate_final_report(model, sat_x, sat_y, device)


if __name__ == '__main__':
    train_mixed(ratio=0.8)
