import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.orientations import axcodes2ornt, ornt_transform

target_shape = (128, 128, 128)

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
