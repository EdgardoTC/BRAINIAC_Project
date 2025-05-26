# BRAINIAC_Project
**Brain Imaging Automated Classifier (BRAINIAC) Project**
A fully automated deep learning pipeline for classifying structural MRI scans using 3D convolutional neural networks. No manual preprocessing required — just plug in your NIfTI files.

## Features
- End-to-end automation from raw NIfTI input to classification
- RAS reorientation, isotropic resampling, resizing, normalization
- 3D CNN model (PyTorch) for patient classification
- ROC curve, AUC, precision/recall, and prediction distribution plots
- Easily extensible for other binary neuroimaging tasks

## Use Case
Initially designed to distinguish patients with **treatment-resistant schizophrenia (TRS)** from patients with **treatment-responsive (TxR)** using T1-weighted MRI.

## Getting Started
1. Place `.nii` or `.nii.gz` files into `Patient_Group_1/` and `Patient_Group_2/` folders.
2. Run `python main.py`.
3. View results in console and generated plots.

## Requirements
- Python 3.8+
- PyTorch
- nibabel
- scipy
- scikit-learn
- matplotlib
- joblib

## Citation
Coming soon...

## License
MIT
