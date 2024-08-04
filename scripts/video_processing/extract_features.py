import cv2
import torch
import torchvision  # Import torchvision here
import torchvision.transforms as transforms

def extract_features(frame, player_boxes, model, device):
    """
    Extract features from players in a frame using a CNN model.

    Args:
    - frame (numpy.ndarray): The image frame.
    - player_boxes (list): List of bounding boxes (x1, y1, x2, y2) for detected players.
    - model (torch.nn.Module): The feature extraction model.
    - device (torch.device): The device to use for computations.

    Returns:
    - list: List of feature descriptors for each player.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Resizing to match input size of most CNNs
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
        player_region = frame[y1:y2, x1:x2]

        if player_region.size == 0:
            features.append(None)
            continue

        player_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
        player_region = transform(player_region).to(device)
        
        with torch.no_grad():
            output = model(player_region.unsqueeze(0))  # Add batch dimension
            features.append(output.cpu().numpy())

    return features

def load_deep_learning_model():
    # Example of loading a ResNet model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Remove the classification layer to use as a feature extractor
    return model

