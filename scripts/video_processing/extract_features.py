import cv2
import torch
import torchvision
import numpy as np
from torchvision import models, transforms

def load_deep_learning_model():
    # Load a pre-trained ResNet50 model for feature extraction
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the final classification layer
    model.eval()  # Set the model to evaluation mode
    return model

def extract_features(frame, player_boxes, model, device):
    # Transform and normalize the player region
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            features.append(None)
            continue

        player_region = transform(player_region).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(player_region)
            features.append(feature.squeeze().cpu().numpy())  # Move data to CPU and convert to numpy array

    torch.cuda.empty_cache()
    return features

