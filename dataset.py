import os
import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

def remap_label(mask):
    unique = np.unique(mask)
    if set(unique).issubset({0, 1}):
        return mask.astype(np.uint8)
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask == 2] = 1
    binary[mask == 3] = 1
    return binary

def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read()
    return np.transpose(data.astype(np.float32), (1, 2, 0))

def normalize(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo, hi = ch.min(), ch.max()
        out[:, :, c] = (ch - lo) / (hi - lo) if hi - lo > 1e-6 else 0.0
    return out

class EOSARChangeDataset(Dataset):
    def __init__(self, split_dir, img_size=256, augment=False):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.img_size = img_size
        self.augment = augment
        self.eo_dir   = self._find_subdir(["pre-event", "pre-test", "pre-val", "pre-training", "pre"])
        self.sar_dir  = self._find_subdir(["post-event", "post-test", "post-val", "post-training", "post"])
        self.mask_dir = self._find_subdir(["target", "mask", "masks", "labels"])
        eo_files = sorted(self.eo_dir.glob("*.tif"))
        self.samples = []
        for eo_path in eo_files:
            name = eo_path.name
            sar_path  = self.sar_dir  / name
            mask_path = self.mask_dir / name
            if sar_path.exists() and mask_path.exists():
                self.samples.append((eo_path, sar_path, mask_path))
        if len(self.samples) == 0:
            raise RuntimeError(f"No matched triplets found in {split_dir}")

        # Separate transforms for EO and SAR to avoid channel confusion
        self.geo_transform = A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(p=0.4),
            A.GridDistortion(p=0.3),
        ], additional_targets={"sar": "image", "mask": "mask"})

        self.eo_transform = A.Compose([
            A.GaussNoise(p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ])

        self.resize = A.Resize(height=img_size, width=img_size)

    def _find_subdir(self, candidates):
        for name in candidates:
            p = self.split_dir / name
            if p.exists():
                return p
        subdirs = [d for d in self.split_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(f"Could not find subfolder. Available: {[d.name for d in subdirs]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eo_path, sar_path, mask_path = self.samples[idx]

        eo  = normalize(read_tif(str(eo_path)))    # (H, W, 3)
        sar = normalize(read_tif(str(sar_path)))   # (H, W, 1)
        mask = remap_label(read_tif(str(mask_path))[:, :, 0].astype(np.uint8))  # (H, W)

        # Ensure correct channels
        eo  = eo[:, :, :3] if eo.shape[2] >= 3 else np.repeat(eo, 3, axis=2)
        sar = sar[:, :, :1]  # strictly 1 channel

        if self.augment:
            # Apply geometric transforms to all three together
            result = self.geo_transform(image=eo, sar=sar, mask=mask)
            eo   = result["image"]    # (H, W, 3)
            sar  = result["sar"]      # (H, W, 1) — guaranteed by input
            mask = result["mask"]

            # Apply pixel transforms only to EO
            eo = self.eo_transform(image=eo)["image"]
        else:
            eo   = self.resize(image=eo)["image"]
            sar  = self.resize(image=sar)["image"]
            mask = self.resize(image=mask)["image"]

        # Final channel enforcement
        eo  = eo[:, :, :3]
        sar = sar[:, :, :1] if len(sar.shape) == 3 else sar[:, :, np.newaxis]
        sar = sar[:, :, :1]

        eo_t   = torch.from_numpy(eo).permute(2, 0, 1).float()    # (3, H, W)
        sar_t  = torch.from_numpy(sar).permute(2, 0, 1).float()   # (1, H, W)
        mask_t = torch.from_numpy(mask).long()                      # (H, W)

        # Absolute final check
        assert eo_t.shape[0] == 3,  f"EO must have 3 channels, got {eo_t.shape}"
        assert sar_t.shape[0] == 1, f"SAR must have 1 channel, got {sar_t.shape}"

        return eo_t, sar_t, mask_t

def get_dataloaders(root, img_size=256, batch_size=8, num_workers=0):
    root = Path(root)
    train_ds = EOSARChangeDataset(root / "train", img_size=img_size, augment=True)
    val_ds   = EOSARChangeDataset(root / "val",   img_size=img_size, augment=False)
    test_ds  = EOSARChangeDataset(root / "test",  img_size=img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
