# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space — Satellite AI Research Intern Assignment

A pixel-level binary change detection model for co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) image pairs. Built for disaster response, urban monitoring, and environmental surveillance applications.

**Architecture:** Siamese U-Net with ResNet-34 EO encoder + lightweight SAR encoder + cross-modal feature fusion  
**Loss:** DiceFocal (handles ~3.8% change-pixel class imbalance)  
**Best Val F1:** 0.681 | **Best Val IoU:** 0.517

---

## Results

| Split | IoU | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| Validation (n=334) | 0.517 | 0.580 | 0.824 | 0.681 |
| Test (n=77) | 0.171 | 0.199 | 0.556 | 0.293 |

> Test performance is lower due to cross-scene domain shift (train: scene_06, test: scene_09). See the technical report for full analysis.

---

## Model Weights

Download the best checkpoint (441 MB):  
**[best_model.pth — Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

Place it at `checkpoints/best_model.pth` before running evaluation.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA T4, 15 GB VRAM)

All dependencies with pinned versions:

```
torch==2.1.0
torchvision==0.16.0
rasterio==1.3.10
albumentations==1.3.1
segmentation-models-pytorch==0.3.3
numpy==1.24.4
opencv-python==4.8.1.78
matplotlib==3.7.3
tqdm==4.66.1
pyyaml==6.0.1
scikit-learn==1.3.2
```

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/RiteshTripathi12/galaxeye-change-detection
cd galaxeye-change-detection

# Create and activate conda environment
conda create -n galaxeye python=3.10 -y
conda activate galaxeye

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

After downloading and extracting the dataset, place it as follows:

```
data/
├── training/
│   ├── pre-event/          # EO images (.tif, 3-channel)
│   ├── post-event/         # SAR images (.tif, 1-channel)
│   └── target/             # Annotation masks (.tif, 4-class → remapped to binary)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

> **Label remapping** (applied automatically by `dataset.py`):  
> Background (0) → 0, Intact (1) → 0, Damaged (2) → 1, Destroyed (3) → 1

Update `data_root` in `config.yaml` to point to your data directory.

---

## Training

```bash
python train.py --config config.yaml
```

Key config options (`config.yaml`):

```yaml
data_root: "data/"
img_size: 256
batch_size: 8
epochs: 50
lr: 0.0001
weight_decay: 0.001
dice_weight: 0.5
focal_weight: 0.5
focal_alpha: 0.75
focal_gamma: 2.0
seed: 42
```

Checkpoints are saved to `checkpoints/` — the best validation F1 checkpoint is saved as `best_model.pth`.

---

## Evaluation

```bash
python eval.py --data_path data/test --weights checkpoints/best_model.pth
```

This reports IoU, Precision, Recall, F1, and a confusion matrix for the change class (label=1) on the specified split.

To evaluate on validation:

```bash
python eval.py --data_path data/val --weights checkpoints/best_model.pth
```

---

## Project Structure

```
galaxeye-change-detection/
├── dataset.py          # TIF dataloader + label remapping
├── model.py            # Siamese U-Net with cross-modal fusion
├── losses.py           # DiceFocal loss
├── metrics.py          # IoU, Precision, Recall, F1
├── train.py            # Training loop with early stopping
├── eval.py             # Evaluation script
├── config.yaml         # All hyperparameters
├── requirements.txt    # Pinned dependencies
├── checkpoints/        # Saved model weights
└── README.md
```

---

## Citation / References

1. Daudt et al. (2018). Fully Convolutional Siamese Networks for Change Detection. ICIP.
2. He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
4. Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
5. Milletari et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
6. Bandara & Patel (2022). A Transformer-based Siamese Network for Change Detection. IGARSS.
7. segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch
8. Buslaev et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information.

---

*Submitted by Ritesh Tripathi — 13 May 2026*
