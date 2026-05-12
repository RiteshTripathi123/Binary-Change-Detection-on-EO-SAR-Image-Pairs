import numpy as np
import torch

class MetricTracker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0
    @torch.no_grad()
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits).squeeze(1) > self.threshold).long()
        targets = targets.long()
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()
    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        iou       = tp / (tp + fp + fn + 1e-8)
        return {"iou": round(iou,4), "precision": round(precision,4),
                "recall": round(recall,4), "f1": round(f1,4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    def confusion_matrix(self):
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])
    def print_results(self, split=""):
        m = self.compute()
        print(f"\n{'='*45}")
        print(f"  Metrics — {split}")
        print(f"{'='*45}")
        print(f"  IoU       : {m['iou']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1 Score  : {m['f1']:.4f}")
        cm = self.confusion_matrix()
        print(f"\n  Confusion Matrix:")
        print(f"             Pred-0   Pred-1")
        print(f"  Actual-0   {cm[0,0]:>6}   {cm[0,1]:>6}")
        print(f"  Actual-1   {cm[1,0]:>6}   {cm[1,1]:>6}")
        print(f"{'='*45}\n")
        return m
