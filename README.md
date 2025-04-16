# Code Availability – Slim UNETR++
This repository will host the official implementation of Slim UNETR++: A Lightweight 3D Medical Image Segmentation Network for Medical Image Analysis.

We are currently in the process of organizing, cleaning, and documenting the code to ensure it meets the standards of reproducibility and clarity. Once finalized, the code and accompanying instructions will be released here for public access.

Thank you for your interest in our work. Please stay tuned for updates.
# Network Architecture
![image](https://github.com/user-attachments/assets/f96934d6-52a5-4624-9bd0-73e5ba608102)
## Data Description
Dataset Name: BraTS2021
Modality: MRI
Size: 1470 3D volumes
Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge
Register and download the official BraTS 21 dataset from the link below and place then into "TrainingData" in the dataset folder:
https://www.synapse.org/#!Synapse:syn27046444/wiki/616992
The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [BraTS 21].
