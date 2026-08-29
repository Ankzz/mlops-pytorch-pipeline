import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=10):
    """
    Loads a pre-trained ResNet-18 and replaces the final layer 
    to match the desired number of classes.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model