
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class VehicleDamageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with 'damaged' and 'undamaged' subfolders.
            transform: Transformations to apply on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, folder in enumerate(['undamaged', 'damaged']):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)  # 0: undamaged, 1: damaged

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

