import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import get_cnn_model, get_mask_rcnn_model
from dataset import VehicleDamageDataset

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/train/"
BATCH_SIZE = 1

# Data Transformations
transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Load Dataset
train_dataset = VehicleDamageDataset(root_dir=DATA_DIR, transforms=transforms)

# Initialize Models
cnn_model = get_cnn_model().to(DEVICE)
mask_rcnn_model = get_mask_rcnn_model(num_classes=2).to(DEVICE)

# Loss and Optimizers
cnn_criterion = torch.nn.CrossEntropyLoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
mask_rcnn_optimizer = torch.optim.Adam(mask_rcnn_model.parameters(), lr=0.001)

# Force Processing of Image 1
print("\n**Fixing Mask R-CNN Target Passing...**\n")
try:
    cnn_model.train()
    mask_rcnn_model.train()

    # Process Image 1 (Index 1)
    img, target = train_dataset[1]
    print(f"Processing Image 1: {train_dataset.imgs[1]}, Binary Label: {target['binary_label']}")

    # CNN Forward Pass
    outputs = cnn_model(img.unsqueeze(0).to(DEVICE))
    print(f"Model Output Shape: {outputs.shape}, Model Outputs: {outputs}")

    # CNN Loss Computation
    binary_label = torch.tensor([target["binary_label"]], dtype=torch.long).to(DEVICE)
    cnn_loss = cnn_criterion(outputs, binary_label)
    print(f"CNN Loss Computed: {cnn_loss.item():.4f}")
    cnn_optimizer.zero_grad()
    cnn_loss.backward()
    cnn_optimizer.step()
    print("CNN Training Step Completed!")

    # Inspect Mask R-CNN Targets
    print("\nInspecting Mask R-CNN Target Structure...")
    mask_rcnn_targets = {
        "boxes": target["mask_rcnn_target"]["boxes"].float(),
        "labels": target["mask_rcnn_target"]["labels"].long(),
        "masks": target["mask_rcnn_target"]["masks"].float(),
    }

    print(f"Mask R-CNN Targets After Casting: {mask_rcnn_targets}")

    # Correct Forward Pass
    print("\nStarting Corrected Mask R-CNN Forward Pass...")
    try:
        mask_rcnn_loss_dict = mask_rcnn_model([img.to(DEVICE)], [mask_rcnn_targets])
        mask_rcnn_loss = sum(loss for loss in mask_rcnn_loss_dict.values())
        print(f"Mask R-CNN Loss Computed: {mask_rcnn_loss.item():.4f}")

        mask_rcnn_optimizer.zero_grad()
        mask_rcnn_loss.backward()
        mask_rcnn_optimizer.step()
        print("Mask R-CNN Training Step Completed!")

    except Exception as e:
        print(f"Mask R-CNN Forward Pass Failed with Error: {e}")

    print("Image 1 Processed Successfully!")

    # Save Models After Processing Completes
    torch.save(cnn_model.state_dict(), "models/cnn_vehicle_damage.pth")
    torch.save(mask_rcnn_model.state_dict(), "models/mask_rcnn_vehicle_damage.pth")
    print("Models saved successfully!")

except Exception as e:
    print(f"Processing failed with error: {e}")
