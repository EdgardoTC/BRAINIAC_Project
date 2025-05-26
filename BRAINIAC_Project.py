import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# === Paths ===
trs_path = '/Users/Patient_Group_1'
txr_path = '/Users/Patient_Group_2'
preprocessed_dir = '/Users/Preprocessed_Scans'
os.makedirs(preprocessed_dir, exist_ok=True)

# === Parameters ===
target_voxel_size = (1.0, 1.0, 1.0)
target_shape = (128, 128, 128)
manual_threshold = 0.45

# === Preprocessing Functions ===
def reorient_to_RAS(nii_img):
    orig_ornt = nib.io_orientation(nii_img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    return nii_img.as_reoriented(transform)

def resample_to_isotropic(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    return zoom(volume, zoom_factors, order=1)

def resize_to_shape(volume, target_shape=(128, 128, 128)):
    result = np.zeros(target_shape, dtype=np.float32)
    crop_shape = [min(volume.shape[i], target_shape[i]) for i in range(3)]
    start_src = [(volume.shape[i] - crop_shape[i]) // 2 for i in range(3)]
    start_dst = [(target_shape[i] - crop_shape[i]) // 2 for i in range(3)]
    src_slices = tuple(slice(start_src[i], start_src[i] + crop_shape[i]) for i in range(3))
    dst_slices = tuple(slice(start_dst[i], start_dst[i] + crop_shape[i]) for i in range(3))
    result[dst_slices] = volume[src_slices]
    return result

def normalize_volume(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / std if std > 0 else np.zeros_like(volume)

def preprocess_and_save(folder_path, label, save_dir):
    scans, labels = [], []
    for fname in os.listdir(folder_path):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            path = os.path.join(folder_path, fname)
            try:
                nii = nib.load(path)
                nii = reorient_to_RAS(nii)
                spacing = nii.header.get_zooms()
                volume = nii.get_fdata()
                volume = resample_to_isotropic(volume, spacing)
                volume = resize_to_shape(volume, target_shape)
                volume = normalize_volume(volume)
                save_path = os.path.join(save_dir, f"{label}_{fname.replace('.nii.gz','').replace('.nii','')}.npy")
                np.save(save_path, volume)
                scans.append(save_path)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    return scans, labels

# === Dataset Class ===
class BrainDataset(Dataset):
    def __init__(self, scan_paths, labels):
        self.scan_paths = scan_paths
        self.labels = labels

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        x = np.load(self.scan_paths[idx])
        x = torch.tensor(x).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# === CNN Model ===
class Deeper3DCNN(nn.Module):
    def __init__(self):
        super(Deeper3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(64 * 16 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(-1)

# === Train and Evaluate ===
def train_and_evaluate(scan_paths, labels, epochs=5, batch_size=2):
    X_train, X_val, y_train, y_val = train_test_split(
        scan_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = BrainDataset(X_train, y_train)
    val_dataset = BrainDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Deeper3DCNN().to(device)

    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {np.mean(losses):.4f}")

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > manual_threshold).astype(int)
            all_preds.extend(preds)
            all_targets.extend(y.numpy().astype(int))
            all_probs.extend(probs)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    cr = classification_report(all_targets, all_preds)

    print("\n📊 Validation Metrics:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("\n🔍 Sample prediction probabilities:")
    print(all_probs[:10])

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    plt.hist([p for p, t in zip(all_probs, all_targets) if t == 0], bins=20, alpha=0.5, label='TxR (0)')
    plt.hist([p for p, t in zip(all_probs, all_targets) if t == 1], bins=20, alpha=0.5, label='TRS (1)')
    plt.legend()
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.show()

# === Main Script ===
def main():
    trs_scans, trs_labels = preprocess_and_save(trs_path, 1, preprocessed_dir)
    txr_scans, txr_labels = preprocess_and_save(txr_path, 0, preprocessed_dir)
    all_scans = trs_scans + txr_scans
    all_labels = trs_labels + txr_labels
    joblib.dump((all_scans, all_labels), os.path.join(preprocessed_dir, 'scan_index.pkl'))
    train_and_evaluate(all_scans, all_labels)

if __name__ == "__main__":
    main()
