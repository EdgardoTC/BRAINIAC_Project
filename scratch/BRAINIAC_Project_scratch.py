import os
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.orientations import axcodes2ornt, ornt_transform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# === Paths ===
trs_path = '/Users/laly/Desktop/Experiments/ML_TRSDiagnosis/Advanced_NII_Model/TRS'
txr_path = '/Users/laly/Desktop/Experiments/ML_TRSDiagnosis/Advanced_NII_Model/TxR'
preprocessed_dir = '/Users/laly/Desktop/Experiments/BRAINIAC_Project/Output'
atlas_path = '/Users/laly/Desktop/Experiments/BRAINIAC/assets/Harvard-Oxford_Atlas.nii'
ref_t1_path = '/Users/laly/Desktop/Experiments/BRAINIAC/assets/Reference_T1W.nii'

os.makedirs(preprocessed_dir, exist_ok=True)

target_voxel_size = (1.0, 1.0, 1.0)
target_shape = (128, 128, 128)
manual_threshold = 0.45

# === Registration (FSL FLIRT) ===
def register_to_template(input_nii, ref_nii, out_nii, out_mat):
    flirt_cmd = [
        'flirt',
        '-in', input_nii,
        '-ref', ref_nii,
        '-out', out_nii,
        '-omat', out_mat,
        '-dof', '6',  # rigid-body registration
        '-interp', 'trilinear'
    ]
    try:
        subprocess.run(flirt_cmd, check=True)
        print(f"Registered {input_nii} to {ref_nii}")
    except Exception as e:
        print(f"Registration failed for {input_nii}: {e}")
        raise e

# === Preprocessing Functions ===
def reorient_to_RAS(nii_img):
    orig_ornt = nib.io_orientation(nii_img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    return nii_img.as_reoriented(transform)

def resample_to_isotropic(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    zoom_factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
    volume = zoom(volume, zoom_factors, order=1)
    return volume

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

def preprocess_and_save(folder_path, label, save_dir, ref_t1_path):
    scans, labels = [], []
    registered_dir = os.path.join(save_dir, "registered")
    os.makedirs(registered_dir, exist_ok=True)

    for fname in os.listdir(folder_path):
        if fname.endswith('.nii') or fname.endswith('.nii.gz'):
            path = os.path.join(folder_path, fname)
            base = os.path.splitext(os.path.splitext(fname)[0])[0]
            reg_nii = os.path.join(registered_dir, f"{base}_reg.nii.gz")
            reg_mat = os.path.join(registered_dir, f"{base}_reg.mat")

            # Registration
            if not os.path.exists(reg_nii):
                try:
                    register_to_template(path, ref_t1_path, reg_nii, reg_mat)
                except Exception as e:
                    print(f"Skipping {fname} due to registration error.")
                    continue
            try:
                nii = nib.load(reg_nii)
                nii = reorient_to_RAS(nii)
                spacing = nii.header.get_zooms()
                volume = nii.get_fdata()
                volume = resample_to_isotropic(volume, spacing, target_voxel_size)
                volume = resize_to_shape(volume, target_shape)
                volume = normalize_volume(volume)
                save_path = os.path.join(save_dir, f"{label}_{base}.npy")
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

# === Grad-CAM for 3D CNNs ===
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        handle_fw = self.target_layer.register_forward_hook(forward_hook)
        handle_bw = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles = [handle_fw, handle_bw]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[0] if class_idx is None else output[0][class_idx]
        target.backward(retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])  # [C]
        activations = self.activations[0]  # [C, D, H, W]
        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0)
        heatmap = torch.relu(heatmap)
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

def load_and_resample_atlas(atlas_path, target_shape):
    atlas = None
    if atlas_path and os.path.exists(atlas_path):
        if atlas_path.endswith('.nii') or atlas_path.endswith('.nii.gz'):
            atlas_nii = nib.load(atlas_path)
            atlas = atlas_nii.get_fdata()
            if atlas.shape != target_shape:
                zoom_factors = [target_shape[i] / atlas.shape[i] for i in range(3)]
                atlas = zoom(atlas, zoom_factors, order=0)
        else:
            atlas = np.load(atlas_path)
            if atlas.shape != target_shape:
                zoom_factors = [target_shape[i] / atlas.shape[i] for i in range(3)]
                atlas = zoom(atlas, zoom_factors, order=0)
    return atlas

def load_and_preprocess_t1(ref_t1_path, target_shape):
    t1_img = nib.load(ref_t1_path)
    t1_data = t1_img.get_fdata()
    if t1_data.shape != target_shape:
        zoom_factors = [target_shape[i] / t1_data.shape[i] for i in range(3)]
        t1_data = zoom(t1_data, zoom_factors, order=1)
    t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data) + 1e-8)
    return t1_data

def upsample_heatmap(heatmap, target_shape):
    if heatmap is None:
        return None
    if heatmap.shape == target_shape:
        return heatmap
    zoom_factors = [target_shape[i] / heatmap.shape[i] for i in range(3)]
    return zoom(heatmap, zoom_factors, order=1)

def train_and_group_gradcam(scan_paths, labels, atlas_path=None, ref_t1_path=None, epochs=5, batch_size=2, visualize_slices=None):
    X_train, X_val, y_train, y_val = train_test_split(
        scan_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_dataset = BrainDataset(X_train, y_train)
    val_dataset = BrainDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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
    val_scans, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > manual_threshold).astype(int)
            all_preds.extend(preds)
            all_targets.extend(y.numpy().astype(int))
            all_probs.extend(probs)
            val_scans.append(x.cpu().numpy()[0, 0])
            val_labels.append(int(y.item()))

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

    gradcam = GradCAM3D(model, model.conv3)
    trs_heatmaps = []
    txr_heatmaps = []
    print("\nGenerating Grad-CAM maps for validation set...")

    for i, (scan, label) in enumerate(zip(val_scans, val_labels)):
        scan_tensor = torch.tensor(scan).unsqueeze(0).unsqueeze(0).float().to(device)
        heatmap = gradcam.generate(scan_tensor)
        if label == 1:
            trs_heatmaps.append(heatmap)
        else:
            txr_heatmaps.append(heatmap)
    gradcam.close()

    trs_mean = np.mean(trs_heatmaps, axis=0) if trs_heatmaps else None
    txr_mean = np.mean(txr_heatmaps, axis=0) if txr_heatmaps else None
    diff_map = trs_mean - txr_mean if (trs_mean is not None and txr_mean is not None) else None

    atlas = load_and_resample_atlas(atlas_path, target_shape)
    t1_data = load_and_preprocess_t1(ref_t1_path, target_shape) if ref_t1_path else None

    trs_mean = upsample_heatmap(trs_mean, target_shape)
    txr_mean = upsample_heatmap(txr_mean, target_shape)
    diff_map = upsample_heatmap(diff_map, target_shape)
    if atlas is not None and atlas.shape != target_shape:
        atlas = upsample_heatmap(atlas, target_shape)

    if visualize_slices is None:
        visualize_slices = [target_shape[0] // 2]
    for slice_idx in visualize_slices:
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.title("TRS mean Grad-CAM")
        if t1_data is not None:
            plt.imshow(t1_data[slice_idx, :, :], cmap='gray', alpha=0.6)
        plt.imshow(trs_mean[slice_idx, :, :], cmap='hot', alpha=0.5)
        if atlas is not None:
            plt.contour(atlas[slice_idx, :, :], colors='w', linewidths=0.5)
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("TxR mean Grad-CAM")
        if t1_data is not None:
            plt.imshow(t1_data[slice_idx, :, :], cmap='gray', alpha=0.6)
        plt.imshow(txr_mean[slice_idx, :, :], cmap='hot', alpha=0.5)
        if atlas is not None:
            plt.contour(atlas[slice_idx, :, :], colors='w', linewidths=0.5)
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("TRS - TxR (Diff) Grad-CAM")
        if t1_data is not None:
            plt.imshow(t1_data[slice_idx, :, :], cmap='gray', alpha=0.6)
        plt.imshow(diff_map[slice_idx, :, :], cmap='bwr', alpha=0.5)
        if atlas is not None:
            plt.contour(atlas[slice_idx, :, :], colors='k', linewidths=0.5)
        plt.colorbar()
        plt.suptitle(f"Slice {slice_idx}")
        plt.tight_layout()
        plt.show()

def main():
    trs_scans, trs_labels = preprocess_and_save(trs_path, 1, preprocessed_dir, ref_t1_path)
    txr_scans, txr_labels = preprocess_and_save(txr_path, 0, preprocessed_dir, ref_t1_path)
    all_scans = trs_scans + txr_scans
    all_labels = trs_labels + txr_labels
    joblib.dump((all_scans, all_labels), os.path.join(preprocessed_dir, 'scan_index.pkl'))
    train_and_group_gradcam(
        all_scans, all_labels,
        atlas_path=atlas_path,
        ref_t1_path=ref_t1_path,
        visualize_slices=[target_shape[0] // 2]
    )

if __name__ == "__main__":
    main()
