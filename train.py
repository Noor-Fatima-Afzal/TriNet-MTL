import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from datasets.eeg_dataset import EEGDataset
from models.transformer import MultiBranchTransformer
from config import *
from utils.metrics import compute_accuracy

# Load Data
X_bio = np.load(data_dir + "X_biometric.npy", allow_pickle=True)
Y_bio = np.load(data_dir + "Y_biometric.npy", allow_pickle=True)
X_lang = np.load(data_dir + "X_language.npy", allow_pickle=True)
Y_lang = np.load(data_dir + "Y_language.npy", allow_pickle=True)
X_dev = np.load(data_dir + "X_device.npy", allow_pickle=True)
Y_dev = np.load(data_dir + "Y_device.npy", allow_pickle=True)

min_len = min(len(X_bio), len(X_lang), len(X_dev))
X_all = X_bio[:min_len]
Y_bio = LabelEncoder().fit_transform(Y_bio[:min_len])
Y_lang = LabelEncoder().fit_transform(Y_lang[:min_len])
Y_dev = LabelEncoder().fit_transform(Y_dev[:min_len])

dataset = EEGDataset(X_all, Y_bio, Y_lang, Y_dev, seq_len=seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MultiBranchTransformer(bio_classes=len(np.unique(Y_bio))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds_bio, all_labels_bio = [], []
    all_preds_lang, all_labels_lang = [], []
    all_preds_dev, all_labels_dev = [], []

    for x, y_bio, y_lang, y_dev in loader:
        x, y_bio, y_lang, y_dev = x.to(device), y_bio.to(device), y_lang.to(device), y_dev.to(device)

        out_bio, out_lang, out_dev = model(x)
        loss = criterion(out_bio, y_bio) + criterion(out_lang, y_lang) + criterion(out_dev, y_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds_bio.extend(torch.argmax(out_bio, 1).cpu().numpy())
        all_labels_bio.extend(y_bio.cpu().numpy())
        all_preds_lang.extend(torch.argmax(out_lang, 1).cpu().numpy())
        all_labels_lang.extend(y_lang.cpu().numpy())
        all_preds_dev.extend(torch.argmax(out_dev, 1).cpu().numpy())
        all_labels_dev.extend(y_dev.cpu().numpy())

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | "
          f"Bio Acc: {compute_accuracy(all_labels_bio, all_preds_bio):.4f} | "
          f"Lang Acc: {compute_accuracy(all_labels_lang, all_preds_lang):.4f} | "
          f"Dev Acc: {compute_accuracy(all_labels_dev, all_preds_dev):.4f}")
