import os, argparse, yaml, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from dataset import get_dataloaders
from model import SiameseChangeNet
from losses import DiceFocalLoss
from metrics import MetricTracker

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for eo, sar, mask in loader:
        eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(eo, sar)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()
    for eo, sar, mask in loader:
        eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
        with torch.cuda.amp.autocast():
            logits = model(eo, sar)
            loss = criterion(logits, mask)
        total_loss += loss.item()
        tracker.update(logits, mask)
    return total_loss / len(loader), tracker

def main(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _ = get_dataloaders(
        root=cfg["data_root"], img_size=cfg["img_size"],
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    model = SiameseChangeNet(pretrained=cfg["pretrained"]).to(device)
    criterion = DiceFocalLoss(dice_weight=cfg["dice_weight"], focal_weight=cfg["focal_weight"],
                               alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_tracker = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        val_metrics = val_tracker.compute()
        f1, iou = val_metrics["f1"], val_metrics["iou"]
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": f1, "val_iou": iou, "cfg": cfg},
                       save_dir / "best_model.pth")
            print(f"  ✓ Saved best model (F1={f1:.4f})")
    print(f"\n[Done] Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
