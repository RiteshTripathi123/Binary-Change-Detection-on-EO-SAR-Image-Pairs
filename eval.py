import argparse, torch, numpy as np, os
import matplotlib ; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataset import EOSARChangeDataset
from model import SiameseChangeNet
from metrics import MetricTracker
from torch.utils.data import DataLoader

def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model = SiameseChangeNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def save_qualitative(eo, sar, gt, pred, save_path, threshold=0.5):
    eo_np   = eo.permute(1,2,0).cpu().numpy()
    sar_np  = sar[0].cpu().numpy()
    gt_np   = gt.cpu().numpy()
    pred_np = (torch.sigmoid(pred).squeeze().cpu().numpy() > threshold).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(np.clip(eo_np[:,:,:3], 0, 1)) ; axes[0].set_title("EO") ; axes[0].axis("off")
    axes[1].imshow(sar_np, cmap="gray")           ; axes[1].set_title("SAR"); axes[1].axis("off")
    axes[2].imshow(gt_np, cmap="RdYlGn_r", vmin=0, vmax=1) ; axes[2].set_title("GT") ; axes[2].axis("off")
    vis = np.zeros((*gt_np.shape, 3), dtype=np.float32)
    vis[(pred_np==1)&(gt_np==1)] = [0.0,0.8,0.0]
    vis[(pred_np==1)&(gt_np==0)] = [0.9,0.1,0.1]
    vis[(pred_np==0)&(gt_np==1)] = [1.0,0.6,0.0]
    axes[3].imshow(vis) ; axes[3].set_title("Pred (TP=G,FP=R,FN=O)") ; axes[3].axis("off")
    plt.tight_layout() ; plt.savefig(save_path, dpi=100, bbox_inches="tight") ; plt.close()

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    dataset = EOSARChangeDataset(args.data_path, img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)    print(f"[Eval] {len(dataset)} samples")
    tracker = MetricTracker(threshold=args.threshold)
    vis_dir = Path(args.output_dir) / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0
    for eo, sar, mask in loader:
        eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
        with torch.cuda.amp.autocast():
            logits = model(eo, sar)
        tracker.update(logits, mask)
        if vis_count < args.n_vis:
            for i in range(eo.size(0)):
                if vis_count >= args.n_vis: break
                save_qualitative(eo[i], sar[i], mask[i], logits[i:i+1],
                                 vis_dir / f"sample_{vis_count:03d}.png", args.threshold)
                vis_count += 1
    tracker.print_results(split=args.split)
    print(f"Visualisations saved to: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--weights",    type=str, required=True)
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--n_vis",      type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    args = parser.parse_args()
    evaluate(args)
