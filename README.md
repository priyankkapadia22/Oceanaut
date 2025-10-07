# FusionUNet: State-of-the-Art Underwater Image and Video Enhancement
 

## 🌊 Overview

This repository contains the official PyTorch implementation of **FusionUNet**, a novel fusion-based deep learning model designed for superior **underwater image and video enhancement**.

Our approach leverages two independently pretrained restoration **UNet** models. Their complementary outputs are then fused via a dedicated **FusionUNet**, specifically trained to enhance underwater images with significant improvements in **visual quality** and **temporal consistency** for videos.

### ✨ State-of-the-Art Performance

The FusionUNet model achieves **state-of-the-art** performance on the widely-used **EUVP dataset**, delivering:
* **PSNR: 25.38 dB**
* **SSIM: 0.8667**

This performance significantly **outperforms** many prior methods, establishing a new strong baseline.

***

## 💡 Features

* **Fusion** of complementary pretrained UNet models for **superior underwater image restoration**.
* **PyTorch-based** end-to-end pipeline including training, testing, and video enhancement.
* **Checkpoint resume** and automated evaluation scripts.
* **Video frame enhancement** with built-in **temporal context** using frame triplets.
* High-quality restoration validated on the **EUVP paired dataset**.

***

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
cd <repo-name>
```
### 2. Setup Environment and Dependencies
Create a Python environment (e.g., using conda or venv) and install the necessary dependencies:

```
pip install -r requirements.txt
Typical dependencies include: torch, torchvision, opencv-python, tqdm, scikit-image, numpy.
```

### 📂 Dataset Preparation
We use the EUVP Dataset for model training and evaluation.

Training:   
Input - Original underwater training images  
Reference - Reference ground-truth training images  
Note: Same for Test purpose

### 🏃 Training
Train the FusionUNet model end-to-end using the following script:

```bash
python fusion_cnn.py
```
The script is configured to load the two necessary pretrained UNet models from specified paths (e.g., unet_model_1.pth and unet_model_2.pth).  
It supports automatic checkpoint resume to continue training from a saved state. 
You may need to adjust hyperparameters and checkpoint paths inside the script (fusion_cnn.py) as necessary.  

### 🧪 Testing
Evaluate a trained fusion model on the test dataset:  

```bash
python test_fusion.py
```
This script computes the average PSNR and SSIM over the entire test set.  
It saves enhanced image outputs for the first 10 samples for visual inspection in the output directory.  
Remember to modify the checkpoint path inside test_fusion.py to point to the specific fusion_model_epoch_XX.pth you wish to evaluate.

### ⚠️ Model Dependencies
Crucial: The FusionUNet model requires the outputs of two separate, pretrained UNet models for inference.  
To run any enhancement (testing or video enhancement), you must have all three model checkpoints available:

unet_model_1.pth (First pretrained UNet)

unet_model_2.pth (Second pretrained UNet)

fusion_model.pth (Trained FusionUNet)

The FusionUNet's input is the stacked outputs of the first two models on a frame (or frame triplet). Without the base model checkpoints, the FusionUNet cannot run inference.  
Make sure these three checkpoint files are correctly loaded and accessible for successful enhancement.  

## 📊 Benchmarks

Our **FusionUNet** sets a new strong baseline, surpassing prior leading methods on the **EUVP paired dataset**:

| Model | PSNR (dB) | SSIM | Dataset |
| :--- | :---: | :---: | :--- |
| **FusionUNet** | **25.38** | **0.8667** | EUVP (paired) |
| WaterNet | 23.56 | 0.803 | EUVP (paired) |
| UWCNN | 22.60 | 0.787 | EUVP (paired) |
| FUnIE-GAN | 22–23 | ~0.78 | EUVP (paired) |

### Note:
Increasing GPU workload and Increasing Batch size can remove noise more and can give better score.
