import torch
import numpy as np
from collections import Counter

# load
X_finesat = torch.load('data/W_FineSat_X.pth', weights_only=False)
Y_finesat = torch.load('data/W_FineSat_Y.pth', weights_only=False)

X_raw = torch.load('data/WO_FineSat_X.pth', weights_only=False)
Y_raw = torch.load('data/WO_FineSat_Y.pth', weights_only=False)

X_finesat = np.array(X_finesat)
Y_finesat = np.array(Y_finesat)

X_raw = np.array(X_raw)
Y_raw = np.array(Y_raw)

print("=== FineSat data ===")
print("X_finesat shape:", X_finesat.shape)
print("Y_finesat shape:", Y_finesat.shape)
print("Class distribution:", Counter(Y_finesat))
print("Min:", X_finesat.min())
print("Max:", X_finesat.max())

print("\n=== Raw data ===")
print("X_raw shape:", X_raw.shape)
print("Y_raw shape:", Y_raw.shape)
print("Class distribution:", Counter(Y_raw))
print("Min:", X_raw.min())
print("Max:", X_raw.max())
