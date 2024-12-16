import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import get_cnn_model, get_mask_rcnn_model
from dataset import VehicleDamageDataset
from utils import evaluate_metrics
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "data/test/"
BATCH_SIZE = 1

# Load Test Data
transforms = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])

test_dataset = VehicleDamageDataset(root_dir=TEST_DIR, transforms=transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load Models
cnn_model = get_cnn_model().to(DEVICE)
cnn_model.load_state_dict(torch.load("models/cnn_vehicle_damage.pth", weights_only=True))
cnn_model.eval()

num_classes = 2
mask_rcnn_model = get_mask_rcnn_model(num_classes).to(DEVICE)
mask_rcnn_model.load_state_dict(torch.load("models/mask_rcnn_vehicle_damage.pth", weights_only=True))
mask_rcnn_model.eval()


def evaluate_model(model, test_loader, model_type="cnn"):
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(DEVICE) for img in images)
            if model_type == "cnn":
                outputs = model(images[0].unsqueeze(0))
                preds = torch.argmax(outputs, 1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend([t["labels"].cpu().numpy() for t in targets])

            elif model_type == "mask_rcnn":
                outputs = model(images)
                masks = outputs[0]['masks'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                high_confidence_masks = masks[scores > 0.5]
                pred_mask = np.any(high_confidence_masks > 0.5, axis=0).astype(int)
                true_mask = targets[0]["masks"].cpu().numpy().sum(axis=0).astype(int)
                all_preds.append(pred_mask)
                all_targets.append(true_mask)

    metrics = evaluate_metrics(all_preds, all_targets)
    return metrics

# Evaluate Models
cnn_metrics = evaluate_model(cnn_model, test_loader, model_type="cnn")
mask_rcnn_metrics = evaluate_model(mask_rcnn_model, test_loader, model_type="mask_rcnn")

print("Evaluation Results:\n")
print("CNN Model Performance:")
for metric, value in cnn_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nMask R-CNN Model Performance:")
for metric, value in mask_rcnn_metrics.items():
    print(f"{metric}: {value:.4f}")

