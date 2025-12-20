import torch
from torchvision.models.segmentation import deeplabv3_resnet50

def get_model():
    # Loading pretrained DeepLabV3 with ResNet-50 backbone
    model = deeplabv3_resnet50(weights="DEFAULT")

    # Modify classifier for 2 classes (background + corrosion)
    model.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=1)

    return model
