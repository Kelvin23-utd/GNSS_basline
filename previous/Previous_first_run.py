import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class WiFiDatasetFast(Dataset):
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
                files = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith(".csv")
                ]
                for f in files:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = pd.read_csv(file_path, header=None).values.astype(np.float32)

        if data.shape[0] > data.shape[1]:
            data = data.T

        noise = np.random.randn(*data.shape).astype(np.float32) * 0.005
        data = data + noise

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)


class WiFiNet(nn.Module):
    def __init__(self, num_classes=5, input_channels=22):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.3),
            nn.MaxPool1d(4),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    data_path = r"C:\Users\72399\Desktop\GNSS_Project\traindata"
    batch_size = 128
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = WiFiDatasetFast(data_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = WiFiNet(num_classes=5, input_channels=22).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += output.argmax(1).eq(target).sum().item()

        scheduler.step()

        model.eval()
        val_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                val_correct += output.argmax(1).eq(target).sum().item()

        train_acc = 100.0 * train_correct / len(train_ds)
        val_acc = 100.0 * val_correct / len(test_ds)
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
