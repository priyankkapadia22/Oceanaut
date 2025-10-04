# FusionUNet: State-of-the-Art Underwater Image and Video Enhancement


## 🌊 Overview

This repository contains the official PyTorch implementation of **FusionUNet**, a novel fusion-based deep learning model designed for superior **underwater image and video enhancement**.

Our approach leverages two independently pretrained restoration **UNet** models. Their complementary outputs are then fused via a dedicated **FusionUNet**, specifically trained to enhance underwater images with significant improvements in **visual quality** and **temporal consistency** for videos.

### ✨ State-of-the-Art Performance

The FusionUNet model achieves **state-of-the-art** performance on the widely-used **EUVP dataset**, delivering:
* **PSNR: 25.12 dB**
* **SSIM: 0.8637**

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
