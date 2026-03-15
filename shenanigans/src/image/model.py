import torch.nn as nn
from torchvision import models
from .config import DEVICE

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(DEVICE)