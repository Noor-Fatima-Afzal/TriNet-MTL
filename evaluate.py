import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from datasets.eeg_dataset import EEGDataset
from models.transformer import MultiBranchTransformer
from config import *
from utils.metrics import compute_accuracy, plot_confusion

# Load & Encode Data (same as train.py)
X_bio = np.load(data_dir + "X_biometric.npy", allow_pickle=True)
Y_bio = np.load(data_dir + "Y_biometric.npy", allow_pickle=True)
X_lang = np.load(data_dir + "X_language.npy", allow_pickle=True)
Y_lang = np.load(data_dir + "Y_language.npy", allow_pickle=True)
X_dev = np.load(data_dir + "X_device.npy", allow_pickle=True)
Y_dev = np.load(data_dir + "Y_device.npy", allow_pickle=True)

min_len = min(len(X_bio), len(X_lang), len(X_dev))
X_all = X_bio[:min_len]
le_bio = LabelEncoder(); Y_bio_enc = le_bio.fit_transform(Y_bio[:min_len])
le_lang = LabelEncoder(); Y_lang_enc = le_lang.fit_transform(Y_lang[:min_len])
le_dev = LabelEncoder(); Y_dev_enc = le_dev.fit_transform(Y_dev[:min_len])

dataset = EEGDataset(X_all, Y_bio_enc, Y_lang_enc, Y_dev_enc, seq_len=seq_len)
loader = DataLoader(dataset, batch_size=batch_size)

model = MultiBranchTransformer(bio_classes=len(le_bio.classes_)).to(device)
model.load_state_dict(torch.load("model.pt"))  # If you saved model
model.eval()

all_preds_bio, all_labels_bio = [], []
all_preds_lang, all_labels_lang = [], []
all_preds_dev, all_labels_dev = [], []

with torch.no_grad():
    for x, y_bio, y_lang, y_dev in loader:
        x = x.to(device)
        out_bio, out_lang, out_dev = model(x)

        all_preds_bio.extend(torch.argmax(out_bio, 1).cpu().numpy())
        all_labels_bio.extend(y_bio.numpy())
        all_preds_lang.extend(torch.argmax(out_lang, 1).cpu().numpy())
        all_labels_lang.extend(y_lang.numpy())
        all_preds_dev.extend(torch.argmax(out_dev, 1).cpu().numpy())
        all_labels_dev.extend(y_dev.numpy())

# Accuracy
print("\nFinal Accuracy:")
print(f"Biometric: {compute_accuracy(all_labels_bio, all_preds_bio):.4f}")
print(f"Language:  {compute_accuracy(all_labels_lang, all_preds_lang):.4f}")
print(f"Device:    {compute_accuracy(all_labels_dev, all_preds_dev):.4f}")

# Confusion Matrices
plot_confusion(all_labels_bio, all_preds_bio, le_bio.classes_, "Biometric (20 Subjects)", "Blues")
plot_confusion(all_labels_lang, all_preds_lang, le_lang.classes_, "Language (Native/Non-native)", "Greens")
plot_confusion(all_labels_dev, all_preds_dev, le_dev.classes_, "Device (In-ear/Bone)", "Oranges")
