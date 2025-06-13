<p align="center">
  <img src="assets/BRAINIAC_Logo.jpeg" alt="BRAINIAC Logo"/>
</p>

## BRAINIAC_Project
**Brain Imaging Automatic Classifier (BRAINIAC) Project**
BRAINIAC is an open-source automated deep learning pipeline for classifying structural MRI scans using 3D convolutional neural networks. No manual preprocessing required — just plug in your NIfTI files. While direct use may be limited to those with similar research aims, it provides a transparent template for neuroimaging classification workflows, and a starting point for others to adapt to their own problems. Even if others need to modify the pipeline, it accelerates development, supports reproducibility, and fosters open scientific exchange.

## Features
- End-to-end automation from raw NIfTI input to classification
- RAS reorientation, isotropic resampling, resizing, normalization
- 3D CNN model (PyTorch) for patient classification
- ROC curve, AUC, precision/recall, and prediction distribution plots
- Easily extensible for other binary neuroimaging tasks

## Use Case
The project was initially designed to distinguish patients with **treatment-resistant schizophrenia (TRS)** from patients with **treatment-responsive (TxR)** using T1-weighted MRI. 
It has now evolved to encompass the use of multi-site data for the categorization of different patient groups.

## Getting Started
1. Place `.nii` or `.nii.gz` files into `Patient_Group_1/` and `Patient_Group_2/` folders.
2. Run `BRAINIAC_Project.py`.
3. View results in console and generated plots.

## Requirements
- Python 3.8+
- PyTorch
- nibabel
- scipy
- scikit-learn
- matplotlib
- joblib

## Future Additions
- Grad-CAM for 3D-CNN to improve interpretability
- Addition of Grad-CAM activation overlays on brain scans for visualization
- Possible hybrid CNN and regional features
- Addition of multi-sequence use (T2Flair, qT1, SWI)

## Citation
Coming soon...

## Acknowledgements
- Edgardo Torres-Carmona
- Dr. Mario Graff-Guerrero
- Dr. Fumihiko Ueno
- Dr. Teruki Koizumi
- Dr. Shinichiro Nakajima
- Dr. Philip Gerretsen
- Dr. Ariel Graff-Guerrero

## License
MIT
