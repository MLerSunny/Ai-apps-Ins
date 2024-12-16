from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0

def evaluate_metrics(predictions, targets):
    pred_masks = np.array(predictions).astype(int)
    true_masks = np.array(targets).astype(int)
    iou_scores = [calculate_iou(p, t) for p, t in zip(pred_masks, true_masks)]
    avg_iou = np.mean(iou_scores)
    accuracy = accuracy_score(true_masks.flatten(), pred_masks.flatten())
    f1 = f1_score(true_masks.flatten(), pred_masks.flatten())
    return {"Accuracy": accuracy, "IoU": avg_iou, "F1-score": f1}
