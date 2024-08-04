import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image  # Import PIL Image to handle image transformations

def load_deep_learning_model():
    # Load a pre-trained ResNet50 model with proper weights usage
    model = resnet50(pretrained=True).eval()  # Set the model to evaluation mode
    return model

# Define a transformation to prepare the image patches
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(frame, player_boxes, model):
    """
    Extract features from players in a frame using a deep learning model.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x1, y1, x2, y2) for detected players.
    - model (torch.nn.Module): Pre-trained deep learning model for feature extraction.

    Returns:
    - list: List of feature tensors for each player.
    """
    features = []
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box)
        if x2 > x1 and y2 > y1:
            player_region = frame[y1:y2, x1:x2]
            player_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)  # Convert to RGB
            player_region = Image.fromarray(player_region)  # Convert numpy array to PIL Image
            player_region = transform(player_region)  # Apply transformations

            with torch.no_grad():
                output = model(player_region.unsqueeze(0))  # Add batch dimension
                features.append(output.squeeze(0))  # Remove batch dimension and store the feature tensor
        else:
            features.append(None)

    return features
