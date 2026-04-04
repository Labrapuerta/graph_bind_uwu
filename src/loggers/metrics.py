from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Comprehensive metrics for binding site prediction.
    All safe to push directly to W&B.
 
    Recommended thresholds:
    - 0.5 default
    - Lower (0.3) if recall is more important than precision
    - Use the PR curve in W&B to find optimal threshold per run
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)
 
    # Guard: if only one class present, some metrics are undefined
    has_both_classes = len(np.unique(y_true)) > 1
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
 
    metrics = {
        # Threshold-dependent
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred) if has_both_classes else 0.0,
 
        # Confusion matrix breakdown
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "specificity": tn / (tn + fp + 1e-8),
 
        # Threshold-independent (ranking quality)
        "auroc": roc_auc_score(y_true, probs) if has_both_classes else 0.5,
        "auprc": average_precision_score(y_true, probs) if has_both_classes else 0.0,
 
        # Class balance info (useful for debugging)
        "binding_frac": float(y_true.mean()),
        "pred_frac":    float(y_pred.mean()),
    }
    return metrics



