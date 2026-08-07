from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn

def build_base_model(backbone="resnet18", pretrained=True):
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    elif backbone == "resnet50":
        print("Using ResNet50 backbone")
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    else:
        raise NotImplementedError(f"Backbone {backbone} not supported")
    
    return model