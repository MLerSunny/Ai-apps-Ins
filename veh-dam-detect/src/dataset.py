import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class VehicleDamageDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Dataset loader for both CNN and Mask R-CNN models.
        root_dir: Path to the dataset containing 'images' and 'annotations'.
        transforms: Optional transformations applied to images.
        """
        self.root_dir = root_dir
        self.transforms = transforms

        # Load images and masks
        self.imgs = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.masks = sorted(os.listdir(os.path.join(root_dir, "annotations")))

        # Ensure image and mask counts match
        assert len(self.imgs) == len(self.masks), "Mismatch between images and masks!"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Returns data for CNN or Mask R-CNN:
        - CNN: image, binary label (0 = undamaged, 1 = damaged).
        - Mask R-CNN: image, full mask targets.
        """
        # Load image
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask for Mask R-CNN
        mask_path = os.path.join(self.root_dir, "annotations", self.masks[idx])
        mask = cv2.imread(mask_path, 0)  # Grayscale mask

        # Determine binary label for CNN
        obj_ids = np.unique(mask)[1:]  # Exclude background (label 0)
        binary_label = 1 if len(obj_ids) > 0 else 0  # 1 = damaged, 0 = undamaged

        # Prepare bounding boxes for Mask R-CNN
        masks = mask == obj_ids[:, None, None]
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Target structure for Mask R-CNN
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(obj_ids),), dtype=torch.int64),  # All labeled as 1 (damaged)
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
        }

        # Apply optional transforms
        if self.transforms:
            image = self.transforms(image)

        # Return for both models
        return image, {"binary_label": binary_label, "mask_rcnn_target": target}
