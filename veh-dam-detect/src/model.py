import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


def get_cnn_model():
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()

            # Define CNN layers
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
            )

            # Calculate correct FC input size using a dummy tensor
            dummy_input = torch.zeros(1, 3, 224, 224)  # Assuming input size of 224x224
            out = self.conv_layers(dummy_input)
            fc_input_size = out.view(out.size(0), -1).size(1)

            # Define fully connected layers
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fc_input_size, 128), nn.ReLU(),
                nn.Linear(128, 2)  # 2 classes: damaged/undamaged
            )

        def forward(self, x):
            x = self.conv_layers(x)
            print(f"Shape after conv_layers: {x.shape}")  # Debugging line
            x = self.fc_layers(x)
            return x

    return CNNModel()


def get_mask_rcnn_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model